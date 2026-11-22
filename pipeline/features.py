"""As-of features for one scoring month, built in Spark.

Point-in-time rule: features for claims serviced in month M use only claims whose
denial outcome (remit) arrived before M starts, so no future information leaks
into training.

Feature groups and their join keys:
    claim   columns on the claim itself
    code    denial rate by ICD code, smoothed up the hierarchy    keys: icd_code, category, chapter
    payer   payer and payer x code denial rates                   keys: payer_id, icd_code
    tenant  tenant profile: denial rate, prior-auth misses, volume key: tenant_id  <- skewed

Code rates use hierarchical smoothing (empirical Bayes):
    chapter_rate  = (denials + m * global_rate)   / (claims + m)
    category_rate = (denials + m * chapter_rate)  / (claims + m)
    code_rate     = (denials + m * category_rate) / (claims + m)
so a code never seen before falls back to its category, then its chapter
(the incident 1 fix). code_rate_flat smooths straight to the global rate, for comparison.

The tenant join is where incident 2 lives: three whale tenants hold ~57% of claims,
so a shuffle on tenant_id sends most rows to a few tasks. See join_tenant() for the
fixes compared in pipeline/skew_bench.py.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as F

SMOOTHING = 20.0  # pseudo-claims pulling small groups toward their parent rate
SALT_BUCKETS = 16

ID_COLUMNS = ["claim_id", "tenant_id", "payer_id", "service_month", "icd_code", "icd_category",
              "icd_chapter", "code_set_version", "proc_category",
              "remit_date"]  # when the label became known; training uses it for point-in-time cutoffs

NUMERIC_FEATURES = [
    "log_billed", "days_to_submit", "pa_required", "pa_missing",
    "code_rate", "code_rate_flat", "code_support", "code_level",
    "payer_rate", "payer_code_rate",
    "tenant_rate", "tenant_pa_miss_rate", "tenant_log_volume", "tenant_median_days_to_submit",
]
CATEGORICAL_FEATURES = ["payer_id", "proc_category", "icd_chapter"]
LABEL = "denied"


def month_start(month: str) -> date:
    return pd.Period(month, freq="M").start_time.date()


def _counts(df: DataFrame, keys: list[str]) -> DataFrame:
    return df.groupBy(*keys).agg(F.count("*").alias("n"), F.sum("denied").alias("d"))


def _smooth(prior: Column | float) -> Column:
    return (F.col("d") + SMOOTHING * prior) / (F.col("n") + SMOOTHING)


def join_tenant(left: DataFrame, tenant: DataFrame, strategy: str, buckets: int = SALT_BUCKETS) -> DataFrame:
    """Join the tenant profile.
        plain      no hint; the planner picks the strategy (the incident 2 baseline)
        broadcast  broadcast hint: copy the small profile to every task, no shuffle.
                   The fix, and the default for the real feature build.
        salted     split every tenant into `buckets` keys so a whale tenant is
                   processed by several tasks instead of one"""
    if strategy == "broadcast":
        return left.join(F.broadcast(tenant), "tenant_id", "left")
    if strategy == "salted":
        left = left.withColumn("_salt", F.pmod(F.xxhash64("claim_id"), F.lit(buckets)).cast("int"))
        tenant = tenant.withColumn("_salt", F.explode(F.sequence(F.lit(0), F.lit(buckets - 1))))
        return left.join(tenant, ["tenant_id", "_salt"], "left").drop("_salt")
    if strategy == "plain":
        return left.join(tenant, "tenant_id", "left")
    raise ValueError(f"unknown strategy {strategy!r}")


def build_month_features(claims: DataFrame, month: str, strategy: str = "broadcast") -> DataFrame:
    start = month_start(month)
    hist = claims.filter(F.col("remit_date") < F.lit(start))          # labels known before M
    target = claims.filter(F.col("service_month") == month)

    g = hist.agg(F.avg("denied").alias("rate"), F.count("*").alias("n")).first()
    if not g["n"]:
        raise ValueError(f"no labeled history before {start}; first scoring month must have history")
    global_rate = float(g["rate"])

    # Code hierarchy: small lookup tables, broadcast on purpose.
    chapter = _counts(hist, ["icd_chapter"]).select("icd_chapter", _smooth(global_rate).alias("chapter_rate"))
    category = (
        _counts(hist, ["icd_chapter", "icd_category"]).join(chapter, "icd_chapter")
        .select("icd_category", _smooth(F.col("chapter_rate")).alias("category_rate"))
    )
    code = (
        _counts(hist, ["icd_category", "icd_code"]).join(category, "icd_category")
        .select("icd_code", F.col("n").alias("code_support"),
                _smooth(F.col("category_rate")).alias("code_rate_hier"),
                _smooth(global_rate).alias("code_rate_flat_seen"))
    )
    payer = _counts(hist, ["payer_id"]).select("payer_id", _smooth(global_rate).alias("payer_rate"))
    payer_code = _counts(hist, ["payer_id", "icd_code"]).select(
        "payer_id", "icd_code", F.col("n").alias("pc_n"), F.col("d").alias("pc_d"))

    # Tenant profile: tiny in reality, but its size is unknown to the static planner.
    tenant = hist.groupBy("tenant_id").agg(
        F.count("*").alias("n"), F.sum("denied").alias("d"),
        F.avg(F.when(F.col("prior_auth_required"), (~F.col("prior_auth_obtained")).cast("double")))
        .alias("tenant_pa_miss_rate"),
        F.percentile_approx("days_to_submit", 0.5).alias("tenant_median_days_to_submit"),
    ).select(
        "tenant_id", _smooth(global_rate).alias("tenant_rate"), "tenant_pa_miss_rate",
        F.log1p("n").alias("tenant_log_volume"), "tenant_median_days_to_submit",
    )

    df = (
        target
        .join(F.broadcast(code), "icd_code", "left")
        .join(F.broadcast(category), "icd_category", "left")
        .join(F.broadcast(chapter), "icd_chapter", "left")
        .join(F.broadcast(payer), "payer_id", "left")
        .join(F.broadcast(payer_code), ["payer_id", "icd_code"], "left")
        .withColumn("code_support", F.coalesce("code_support", F.lit(0)))
        .withColumn("code_rate", F.coalesce("code_rate_hier", "category_rate", "chapter_rate", F.lit(global_rate)))
        .withColumn("code_rate_flat", F.coalesce("code_rate_flat_seen", F.lit(global_rate)))
        .withColumn("code_level",
                    F.when(F.col("code_rate_hier").isNotNull(), 0)
                    .when(F.col("category_rate").isNotNull(), 1)
                    .when(F.col("chapter_rate").isNotNull(), 2).otherwise(3))
        .withColumn("payer_code_rate",
                    (F.coalesce("pc_d", F.lit(0)) + SMOOTHING * F.col("code_rate"))
                    / (F.coalesce("pc_n", F.lit(0)) + SMOOTHING))
        .withColumn("log_billed", F.log1p("billed_amount"))
        .withColumn("pa_required", F.col("prior_auth_required").cast("int"))
        .withColumn("pa_missing", (F.col("prior_auth_required") & ~F.col("prior_auth_obtained")).cast("int"))
    )
    df = join_tenant(df, tenant, strategy)
    return df.select(*ID_COLUMNS, *[c for c in NUMERIC_FEATURES if c not in ID_COLUMNS], LABEL)


def build_features(spark: SparkSession, claims: DataFrame, months: list[str], out_dir: Path) -> dict[str, int]:
    """Build and write features for each month; returns rows per month."""
    rows = {}
    for month in months:
        df = build_month_features(claims, month)
        path = out_dir / f"service_month={month}"
        df.drop("service_month").write.mode("overwrite").parquet(str(path))
        rows[month] = spark.read.parquet(str(path)).count()
    return rows