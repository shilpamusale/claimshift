"""Pipeline entry point.

    python -m pipeline.run_pipeline contract --eval-month 2026-10

Exit codes: 0 proceed, 2 retrain (drift), 3 block (data quality).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

from pipeline.contracts import run_contract
from pipeline.session import ROOT, get_spark

EXIT_CODES = {"proceed": 0, "retrain": 2, "block": 3}


def previous_months(month: str, k: int) -> list[str]:
    end = pd.Period(month, freq="M")
    return [str(end - i) for i in range(k, 0, -1)]


def available_months(data_dir: Path) -> set[str]:
    claims = data_dir / "raw" / "claims"
    return {p.name.split("=", 1)[1] for p in claims.glob("service_month=*")}


def print_report(report: dict) -> None:
    print(f"\nData contract  {report['eval_month']}  vs  train {report['train_months'][0]}..{report['train_months'][-1]}")
    print(f"rows: {report['rows_evaluated']:,}\n")
    for c in report["checks"]:
        limit = c["fail"] if c["fail"] is not None else ""
        print(f"  {c['status'].upper():5s} {c['name']:14s} value={c['value']:.4f}  fail>{limit}")
        if c["name"] == "unseen_codes":
            d = c["details"]
            print(f"        overall {d['overall_rate']:.2%}, {d['tenants_over_fail']}/{d['tenants_graded']} graded tenants over the limit")
            for w in d["worst_tenants"][:3]:
                print(f"        {w['tenant_id']}: {w['rate']:.2%} of {w['claims']:,} claims")
            for u in d["top_unseen_codes"][:3]:
                print(f"        {u['icd_code']:9s} {u['claims']:>6,}  {(u['description'] or '')[:60]}")
    print(f"\nSTATUS {report['status'].upper()}  ->  ACTION {report['action'].upper()}\n")


def cmd_contract(args: argparse.Namespace) -> int:
    have = available_months(args.data)
    train = [m for m in previous_months(args.eval_month, args.train_months) if m in have]
    if args.eval_month not in have or not train:
        sys.exit(f"need claims for {args.eval_month} and at least one earlier month; have {sorted(have)}")

    thresholds = yaml.safe_load(args.thresholds.read_text())
    spark = get_spark()
    report = run_contract(spark, args.data, args.eval_month, train, thresholds)

    out = args.data / "reports" / f"drift_report_{args.eval_month}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))
    print_report(report)
    print(f"report: {out}")
    return EXIT_CODES[report["action"]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    c = sub.add_parser("contract", help="run the data contract for one month")
    c.add_argument("--eval-month", required=True, help="YYYY-MM")
    c.add_argument("--train-months", type=int, default=3, help="months before eval-month used as training window")
    c.add_argument("--data", type=Path, default=ROOT / "data")
    c.add_argument("--thresholds", type=Path, default=ROOT / "configs" / "thresholds.yaml")
    c.set_defaults(func=cmd_contract)

    args = ap.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()