"""Incident 2: whale tenants skew the tenant join.

Runs the same October feature build five ways and measures the result stage (where
October's claims are joined to the tenant profile) from Spark's event log, the same
data the Spark History Server shows:

    baseline       AQE off. The planner can't size the tenant profile, picks a
                   sort-merge join, and the shuffle on tenant_id sends each whale
                   tenant to a single task.
    aqe            AQE on (Spark's default). Watch whether it really fixes the skew.
    aqe_skew_join  AQE on, runtime broadcast off, skew thresholds lowered to laptop
                   scale: does AQE split the skewed partition?
    broadcast      AQE off, broadcast hint on the tenant profile: no shuffle at all.
    salted         AQE off, each tenant split into 16 salt buckets: the fix when
                   the small side is too big to broadcast.

Headline metric: the biggest task's share of rows. Merging small partitions can
make max/median ratios look better without shrinking the straggler; the share can't.
"""

from __future__ import annotations

import json
import re
import statistics
import time
from pathlib import Path

from pyspark.sql import SparkSession

from pipeline.features import build_month_features

VARIANTS: dict[str, tuple[dict[str, str], str]] = {
    "baseline": ({"spark.sql.adaptive.enabled": "false"}, "plain"),
    "aqe": ({"spark.sql.adaptive.enabled": "true"}, "plain"),
    "aqe_skew_join": (
        {
            "spark.sql.adaptive.enabled": "true",
            "spark.sql.adaptive.autoBroadcastJoinThreshold": "-1",
            "spark.sql.adaptive.skewJoin.enabled": "true",
            "spark.sql.adaptive.skewJoin.skewedPartitionFactor": "2",
            "spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes": "1MB",
            "spark.sql.adaptive.advisoryPartitionSizeInBytes": "1MB",
        },
        "plain",
    ),
    "broadcast": ({"spark.sql.adaptive.enabled": "false"}, "broadcast"),
    "salted": ({"spark.sql.adaptive.enabled": "false"}, "salted"),
}


def run_variants(spark: SparkSession, claims, month: str, repeats: int = 3) -> dict[str, list[float]]:
    """Run each variant `repeats` times; return wall-clock seconds per run."""
    touched = {k for conf, _ in VARIANTS.values() for k in conf}
    build_month_features(claims, month, "plain").write.format("noop").mode("overwrite").save()  # JVM warm-up

    walls: dict[str, list[float]] = {}
    for name, (conf, strategy) in VARIANTS.items():
        for key in touched:
            spark.conf.unset(key)
        for key, value in conf.items():
            spark.conf.set(key, value)
        walls[name] = []
        for i in range(repeats):
            spark.sparkContext.setJobDescription(f"skew_bench:{name}:{i}")
            t0 = time.perf_counter()
            build_month_features(claims, month, strategy).write.format("noop").mode("overwrite").save()
            walls[name].append(time.perf_counter() - t0)
    spark.sparkContext.setJobDescription(None)
    return walls


def event_log_files(event_dir: Path, app_id: str) -> list[Path]:
    """Spark 4 writes a folder of rolling files (eventlog_v2_<app>/events_N_<app>);
    older versions write one file named after the app."""
    rolling = event_dir / f"eventlog_v2_{app_id}"
    if rolling.is_dir():
        return sorted(rolling.glob("events_*"), key=lambda p: int(p.name.split("_")[1]))
    return [event_dir / app_id]


def read_event_log(files: list[Path]) -> dict:
    """Collect per-task metrics, stage-to-job mapping, and final SQL plans."""
    stage_desc: dict[int, str] = {}
    tasks: dict[int, list[dict]] = {}
    plans: dict[str, str] = {}
    exec_desc: dict[int, str] = {}
    for path in files:
        for line in path.open():
            e = json.loads(line)
            kind = e["Event"]
            if kind == "SparkListenerJobStart":
                desc = (e.get("Properties") or {}).get("spark.job.description", "")
                for sid in e["Stage IDs"]:
                    stage_desc[sid] = desc
            elif kind == "SparkListenerTaskEnd" and e.get("Task Metrics"):
                m = e["Task Metrics"]
                tasks.setdefault(e["Stage ID"], []).append({
                    "run_ms": m["Executor Run Time"],
                    "records": m["Shuffle Read Metrics"]["Total Records Read"] + m["Input Metrics"]["Records Read"],
                })
            elif kind.endswith("SparkListenerSQLExecutionStart"):
                exec_desc[e["executionId"]] = e.get("description", "")
                plans[exec_desc[e["executionId"]]] = e.get("physicalPlanDescription", "")
            elif kind.endswith("SparkListenerSQLAdaptiveExecutionUpdate"):
                desc = exec_desc.get(e["executionId"], "")
                plans[desc] = e.get("physicalPlanDescription", plans.get(desc, ""))
    return {"stage_desc": stage_desc, "tasks": tasks, "plans": plans}


def tenant_join_strategy(plan: str) -> str:
    """Name the physical join used for the tenant profile in the final plan."""
    joins = re.findall(r"\(\d+\) (\w+Join)[^\n]*\nLeft keys \[\d+\]: \[tenant_id", plan)
    if not joins:
        return "unknown"
    return joins[-1] + (" (skew split)" if "skew=true" in plan else "")


def summarize(log: dict, walls: dict[str, list[float]], repeats: int) -> dict:
    """For each variant's last run, the result stage: where October's claims are
    joined to the tenant profile and written. Its task spread shows the skew."""
    last = {name: f"skew_bench:{name}:{repeats - 1}" for name in walls}
    out = {}
    for name, desc in last.items():
        stages = [s for s, d in log["stage_desc"].items() if d == desc and s in log["tasks"]]
        result_stage = max(stages)  # the result stage is created last, so it has the highest id
        tasks = log["tasks"][result_stage]
        records = [t["records"] for t in tasks]
        run_ms = [t["run_ms"] for t in tasks]
        plan = log["plans"].get(desc, "")
        out[name] = {
            "wall_seconds_median": round(statistics.median(walls[name]), 2),
            "result_stage_tasks": len(tasks),
            "max_task_records": max(records),
            "median_task_records": int(statistics.median(records)),
            "max_task_share": round(max(records) / sum(records), 3),
            "max_task_ms": max(run_ms),
            "median_task_ms": int(statistics.median(run_ms)),
            "tenant_join": tenant_join_strategy(plan),
        }
    return out


def print_summary(summary: dict) -> None:
    head = (f"{'variant':14s} {'wall s':>6s} {'tasks':>5s} {'max task rows':>13s} {'share':>6s} "
            f"{'median rows':>11s} {'max ms':>6s} {'med ms':>6s}  tenant join")
    print("\n" + head + "\n" + "-" * (len(head) + 18))
    for name, s in summary.items():
        print(f"{name:14s} {s['wall_seconds_median']:6.2f} {s['result_stage_tasks']:5d} {s['max_task_records']:13,d} "
              f"{s['max_task_share']:6.1%} {s['median_task_records']:11,d} {s['max_task_ms']:6d} "
              f"{s['median_task_ms']:6d}  {s['tenant_join']}")
    print()


def run_skew_bench(spark: SparkSession, claims, month: str, event_dir: Path, repeats: int = 3) -> dict:
    """Run all variants, stop Spark so the event log is complete, and summarize."""
    app_id = spark.sparkContext.applicationId
    walls = run_variants(spark, claims, month, repeats)
    spark.stop()
    summary = summarize(read_event_log(event_log_files(event_dir, app_id)), walls, repeats)
    return {"month": month, "repeats": repeats, "app_id": app_id, "variants": summary}