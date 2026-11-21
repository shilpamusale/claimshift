import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from data_gen.code_sets import write_code_sets
from data_gen.generate_claims import MONTHS, build_world, generate_month
from pipeline.contracts import grade, psi, run_contract
from pipeline.session import ROOT, get_spark

ROWS_PER_MONTH = 100_000


def test_psi_zero_for_identical_distributions():
    d = {"a": 0.5, "b": 0.3, "c": 0.2}
    assert psi(d, d) == pytest.approx(0.0)


def test_psi_grows_with_shift():
    base = {"a": 0.5, "b": 0.5}
    assert psi(base, {"a": 0.6, "b": 0.4}) < psi(base, {"a": 0.9, "b": 0.1})


def test_grade_higher_is_worse():
    assert grade(0.001, 0.005, 0.02) == "pass"
    assert grade(0.01, 0.005, 0.02) == "warn"
    assert grade(0.05, 0.005, 0.02) == "fail"


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("data")
    world = build_world(seed=7)
    write_code_sets(world.codes, out / "reference")
    for i, month in enumerate(MONTHS[:4]):  # Jul-Oct
        df = generate_month(world, month, ROWS_PER_MONTH, seed=7, id_offset=i * ROWS_PER_MONTH)
        part = out / "raw" / "claims" / f"service_month={month:%Y-%m}"
        part.mkdir(parents=True)
        pq.write_table(pa.Table.from_pandas(df.drop(columns="_is_new_code"), preserve_index=False),
                       part / "part-00000.parquet")
    return out


@pytest.fixture(scope="module")
def spark():
    return get_spark(conf={"spark.master": "local[2]", "spark.sql.shuffle.partitions": "4"})


@pytest.fixture(scope="module")
def thresholds():
    return yaml.safe_load((ROOT / "configs" / "thresholds.yaml").read_text())


def checks_by_name(report):
    return {c["name"]: c for c in report["checks"]}


def test_september_proceeds(spark, data_dir, thresholds):
    report = run_contract(spark, data_dir, "2026-09", ["2026-07", "2026-08"], thresholds)
    assert report["action"] == "proceed"
    assert checks_by_name(report)["code_validity"]["value"] == 0.0


def test_october_triggers_retrain(spark, data_dir, thresholds):
    report = run_contract(spark, data_dir, "2026-10", ["2026-07", "2026-08", "2026-09"], thresholds)
    checks = checks_by_name(report)
    assert report["action"] == "retrain"
    assert checks["unseen_codes"]["status"] == "fail"
    assert checks["chapter_psi"]["status"] == "pass"  # coarse PSI misses code-level drift
    worst = [t["tenant_id"] for t in checks["unseen_codes"]["details"]["worst_tenants"][:2]]
    assert set(worst) == {"T04", "T11"}


def test_deleted_code_after_cutover_blocks(spark, data_dir, thresholds, tmp_path):
    """A claim billed with a code deleted on Oct 1 must block the pipeline."""
    bad_dir = tmp_path / "bad"
    (bad_dir / "raw").mkdir(parents=True)
    (bad_dir / "reference").symlink_to(data_dir / "reference")
    src = data_dir / "raw" / "claims"
    for part in src.iterdir():
        df = pd.read_parquet(part)
        if part.name.endswith("2026-10"):
            df.loc[df.index[:10], "icd_code"] = "I42.0"  # deleted FY2027, replaced by I42.00/.01/.09
        (bad_dir / "raw" / "claims" / part.name).mkdir(parents=True)
        df.to_parquet(bad_dir / "raw" / "claims" / part.name / "part-00000.parquet", index=False)
    report = run_contract(spark, bad_dir, "2026-10", ["2026-07", "2026-08", "2026-09"], thresholds)
    assert report["action"] == "block"
    assert checks_by_name(report)["code_validity"]["status"] == "fail"