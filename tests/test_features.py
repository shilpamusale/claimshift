import math

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pyspark.sql import functions as F

from data_gen.code_sets import write_code_sets
from data_gen.generate_claims import MONTHS, build_world, generate_month
from pipeline.contracts import load_claims
from pipeline.features import NUMERIC_FEATURES, build_month_features, month_start
from pipeline.session import get_spark
from pipeline.skew_bench import tenant_join_strategy

ROWS_PER_MONTH = 30_000


@pytest.fixture(scope="module")
def spark():
    return get_spark(conf={"spark.master": "local[2]", "spark.sql.shuffle.partitions": "4"})


@pytest.fixture(scope="module")
def claims(spark, tmp_path_factory):
    out = tmp_path_factory.mktemp("data")
    world = build_world(seed=7)
    write_code_sets(world.codes, out / "reference")
    for i, month in enumerate(MONTHS[:4]):  # Jul-Oct
        df = generate_month(world, month, ROWS_PER_MONTH, seed=7, id_offset=i * ROWS_PER_MONTH)
        part = out / "raw" / "claims" / f"service_month={month:%Y-%m}"
        part.mkdir(parents=True)
        pq.write_table(pa.Table.from_pandas(df.drop(columns="_is_new_code"), preserve_index=False),
                       part / "part-00000.parquet")
    return load_claims(spark, out).cache()


@pytest.fixture(scope="module")
def october(claims):
    return build_month_features(claims, "2026-10").cache()


def test_one_row_per_claim(claims, october):
    assert october.count() == claims.filter(F.col("service_month") == "2026-10").count()
    assert october.select("claim_id").distinct().count() == october.count()


def test_features_use_only_labels_known_before_the_month(claims, october):
    """Point-in-time: tenant volume counts only claims remitted before Oct 1."""
    start = month_start("2026-10")
    known = claims.filter((F.col("remit_date") < F.lit(start)) & (F.col("tenant_id") == "T01")).count()
    vol = october.filter(F.col("tenant_id") == "T01").select("tenant_log_volume").first()[0]
    assert vol == pytest.approx(math.log1p(known))
    later = claims.filter((F.col("remit_date") >= F.lit(start)) & (F.col("tenant_id") == "T01")).count()
    assert later > 0  # there were later labels, and they were left out


def test_unseen_codes_fall_back_up_the_hierarchy(october):
    new_codes = october.filter((F.col("code_support") == 0))
    assert new_codes.count() > 0
    assert new_codes.filter(F.col("code_level") == 0).count() == 0  # never "seen"
    # Without the hierarchy every unseen code gets the same global rate ...
    assert new_codes.select("code_rate_flat").distinct().count() == 1
    # ... with it, unseen codes keep their category's or chapter's rate.
    assert new_codes.select("code_rate").distinct().count() > 5


def test_join_strategies_give_identical_features(claims):
    cols = ["claim_id", *NUMERIC_FEATURES]
    frames = {s: build_month_features(claims, "2026-10", s).select(*cols) for s in ("plain", "broadcast", "salted")}
    base = frames["plain"]
    for name in ("broadcast", "salted"):
        assert base.exceptAll(frames[name]).count() == 0, name
        assert frames[name].exceptAll(base).count() == 0, name


def test_no_feature_is_null(october):
    nulls = october.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in NUMERIC_FEATURES]).first()
    assert all(v == 0 for v in nulls.asDict().values()), nulls


def test_tenant_join_strategy_reads_plan_text():
    plan = "(116) SortMergeJoin [codegen id : 28]\nLeft keys [1]: [tenant_id#1]\nRight keys [1]: [tenant_id#9]"
    assert tenant_join_strategy(plan) == "SortMergeJoin"
    plan = "(50) BroadcastHashJoin [codegen id : 3]\nLeft keys [2]: [tenant_id#1, _salt#2]\n"
    assert tenant_join_strategy(plan) == "BroadcastHashJoin"