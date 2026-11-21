"""Data contract for claims. Runs before feature building and training.

For one evaluation month, checked against a training window:
    schema         expected columns and types are present
    nulls          null rate per column stays under the limit
    code_validity  every code is valid in the code set in effect on its service date
    unseen_codes   share of each tenant's claims with codes the model never saw in training
    chapter_psi    population stability index (PSI) of the ICD chapter mix

The result is a drift report with one action:
    proceed  all checks pass or only warn
    block    a data-quality check failed (schema, nulls, validity): stop the pipeline
    retrain  a drift check failed (unseen codes, PSI): the retrain trigger
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

EXPECTED_SCHEMA = {
    "claim_id": "bigint",
    "tenant_id": "string",
    "payer_id": "string",
    "service_date": "date",
    "submit_date": "date",
    "remit_date": "date",
    "icd_code": "string",
    "icd_category": "string",
    "icd_chapter": "string",
    "code_set_version": "string",
    "proc_category": "string",
    "billed_amount": "double",
    "prior_auth_required": "boolean",
    "prior_auth_obtained": "boolean",
    "days_to_submit": "smallint",
    "denied": "tinyint",
}
QUALITY_CHECKS = {"schema", "nulls", "code_validity"}
DRIFT_CHECKS = {"unseen_codes", "chapter_psi"}


@dataclass
class CheckResult:
    name: str
    status: str  # pass | warn | fail
    value: float
    warn: float | None
    fail: float | None
    details: dict = field(default_factory=dict)


def grade(value: float, warn: float | None, fail: float | None) -> str:
    """Higher is worse."""
    if fail is not None and value > fail:
        return "fail"
    if warn is not None and value > warn:
        return "warn"
    return "pass"


def psi(expected: dict[str, float], actual: dict[str, float], eps: float = 1e-4) -> float:
    """Population stability index between two share distributions."""
    total = 0.0
    for key in set(expected) | set(actual):
        e = max(expected.get(key, 0.0), eps)
        a = max(actual.get(key, 0.0), eps)
        total += (a - e) * math.log(a / e)
    return total


def load_claims(spark: SparkSession, data_dir: Path) -> DataFrame:
    return spark.read.parquet(str(data_dir / "raw" / "claims"))


def load_code_sets(spark: SparkSession, data_dir: Path) -> DataFrame:
    """Official codes keyed by the code-set version they are valid in."""
    ref = data_dir / "reference"
    fy26 = spark.read.parquet(str(ref / "icd10cm_fy2026.parquet")).withColumn("code_set_version", F.lit("FY2026"))
    fy27 = spark.read.parquet(str(ref / "icd10cm_fy2027.parquet")).withColumn("code_set_version", F.lit("FY2027"))
    return fy26.unionByName(fy27).select(
        F.col("code").alias("icd_code"), "code_set_version", "description"
    )


def check_schema(df: DataFrame) -> CheckResult:
    actual = {f.name: f.dataType.simpleString() for f in df.schema.fields}
    missing = [c for c in EXPECTED_SCHEMA if c not in actual]
    wrong = {c: actual[c] for c, t in EXPECTED_SCHEMA.items() if c in actual and actual[c] != t}
    problems = len(missing) + len(wrong)
    return CheckResult("schema", "fail" if problems else "pass", float(problems), None, 0.0,
                       {"missing": missing, "wrong_type": wrong})


def check_nulls(df: DataFrame, n: int, max_rate: float) -> CheckResult:
    cols = [c for c in EXPECTED_SCHEMA if c in df.columns]
    row = df.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in cols]).first().asDict()
    rates = {c: (v or 0) / n for c, v in row.items()}
    worst = max(rates, key=rates.get)
    return CheckResult("nulls", grade(rates[worst], None, max_rate), rates[worst], None, max_rate,
                       {"worst_column": worst})


def check_code_validity(df: DataFrame, n: int, code_sets: DataFrame, max_rate: float) -> CheckResult:
    valid = F.broadcast(code_sets.select("icd_code", "code_set_version").withColumn("_valid", F.lit(1)))
    invalid = df.join(valid, ["icd_code", "code_set_version"], "left").filter(F.col("_valid").isNull())
    top = invalid.groupBy("icd_code", "code_set_version").count().orderBy(F.desc("count")).limit(5).collect()
    rate = invalid.count() / n
    return CheckResult("code_validity", grade(rate, None, max_rate), rate, None, max_rate,
                       {"top_invalid": [r.asDict() for r in top]})


def check_unseen_codes(eval_df: DataFrame, train_df: DataFrame, code_sets: DataFrame,
                       warn: float, fail: float, min_claims: int) -> CheckResult:
    """Graded per tenant; tenants with fewer than min_claims are reported, not graded,
    because a few rare codes would swing their rate."""
    vocab = F.broadcast(train_df.select("icd_code").distinct().withColumn("_seen", F.lit(1)))
    flagged = eval_df.join(vocab, "icd_code", "left").withColumn("_unseen", F.col("_seen").isNull().cast("int"))

    per_tenant = (
        flagged.groupBy("tenant_id")
        .agg(F.count("*").alias("claims"), F.sum("_unseen").alias("unseen"))
        .withColumn("rate", F.col("unseen") / F.col("claims"))
        .orderBy(F.desc("rate"))
        .collect()
    )
    total_claims = sum(r.claims for r in per_tenant)
    total_unseen = sum(r.unseen for r in per_tenant)
    graded = [r for r in per_tenant if r.claims >= min_claims]
    worst = graded[0]

    names = F.broadcast(code_sets.filter(F.col("code_set_version") == "FY2027").select("icd_code", "description"))
    top_codes = (
        flagged.filter(F.col("_unseen") == 1).groupBy("icd_code").count()
        .join(names, "icd_code", "left").orderBy(F.desc("count")).limit(5).collect()
    )
    return CheckResult(
        "unseen_codes", grade(worst.rate, warn, fail), worst.rate, warn, fail,
        {
            "overall_rate": round(total_unseen / total_claims, 5),
            "tenants_over_fail": sum(r.rate > fail for r in graded),
            "tenants_graded": len(graded),
            "tenants_total": len(per_tenant),
            "worst_tenants": [
                {"tenant_id": r.tenant_id, "rate": round(r.rate, 5), "claims": r.claims} for r in graded[:5]
            ],
            "top_unseen_codes": [
                {"icd_code": r.icd_code, "claims": r["count"], "description": r.description} for r in top_codes
            ],
        },
    )


def _shares(df: DataFrame, col: str) -> dict[str, float]:
    rows = df.groupBy(col).count().collect()
    total = sum(r["count"] for r in rows)
    return {r[col]: r["count"] / total for r in rows}


def check_chapter_psi(eval_df: DataFrame, train_df: DataFrame, warn: float, fail: float) -> CheckResult:
    value = psi(_shares(train_df, "icd_chapter"), _shares(eval_df, "icd_chapter"))
    return CheckResult("chapter_psi", grade(value, warn, fail), value, warn, fail)


def run_contract(spark: SparkSession, data_dir: Path, eval_month: str, train_months: list[str],
                 thresholds: dict) -> dict:
    t = thresholds["data_contract"]
    claims = load_claims(spark, data_dir)
    code_sets = load_code_sets(spark, data_dir).cache()
    eval_df = claims.filter(F.col("service_month") == eval_month).cache()
    train_df = claims.filter(F.col("service_month").isin(train_months))
    n = eval_df.count()
    if n == 0:
        raise ValueError(f"no claims for {eval_month}")

    checks = [
        check_schema(eval_df),
        check_nulls(eval_df, n, t["max_null_rate"]),
        check_code_validity(eval_df, n, code_sets, t["max_invalid_code_rate"]),
        check_unseen_codes(eval_df, train_df, code_sets, t["unseen_code_rate"]["warn"],
                           t["unseen_code_rate"]["fail"], t["min_tenant_claims"]),
        check_chapter_psi(eval_df, train_df, t["psi_icd_chapter"]["warn"], t["psi_icd_chapter"]["fail"]),
    ]
    failed = {c.name for c in checks if c.status == "fail"}
    action = "block" if failed & QUALITY_CHECKS else "retrain" if failed & DRIFT_CHECKS else "proceed"
    status = "fail" if failed else "warn" if any(c.status == "warn" for c in checks) else "pass"
    eval_df.unpersist()
    return {
        "eval_month": eval_month,
        "train_months": train_months,
        "rows_evaluated": n,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": status,
        "action": action,
        "checks": [asdict(c) for c in checks],
    }