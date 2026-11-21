"""One place to build the SparkSession, from configs/spark.yaml."""

from __future__ import annotations

from pathlib import Path

import yaml
from pyspark.sql import SparkSession

ROOT = Path(__file__).resolve().parents[1]
SPARK_CONFIG = ROOT / "configs" / "spark.yaml"


def get_spark(config_path: Path = SPARK_CONFIG, conf: dict[str, str] | None = None) -> SparkSession:
    """Build (or reuse) a local SparkSession. `conf` overrides any Spark setting,
    e.g. {"spark.sql.adaptive.enabled": "false"}."""
    cfg = yaml.safe_load(config_path.read_text())
    event_dir = (ROOT / cfg["event_log_dir"]).resolve()
    event_dir.mkdir(parents=True, exist_ok=True)

    builder = (
        SparkSession.builder.appName(cfg["app_name"])
        .master(cfg["master"])
        .config("spark.driver.memory", cfg["driver_memory"])
        .config("spark.sql.shuffle.partitions", str(cfg["shuffle_partitions"]))
        .config("spark.eventLog.enabled", "true")
        .config("spark.eventLog.dir", event_dir.as_uri())
    )
    for key, value in {**cfg.get("conf", {}), **(conf or {})}.items():
        builder = builder.config(key, str(value))

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark