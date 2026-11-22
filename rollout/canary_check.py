"""Canary analysis: does the canary serve the scores its own training produced?

Runs as a Kubernetes Job inside an Argo Rollouts AnalysisRun, or locally against
two URLs. Exit 0 = pass (the rollout continues); exit 1 = fail (Argo aborts the
rollout and scales the canary back to zero: the automatic rollback).

Checks, on the canary's own golden set:
    routing       every response came from a canary pod (pod name contains the
                  canary's pod-template hash, passed in by Argo)
    parity        max |online score - offline score| must stay under 1e-6
    errors        share of failed requests
    latency       p95 request latency
    flagged rate  canary vs stable flagged rate on the same claims

Routing matters because Argo starts the analysis as soon as the canary pod is ready,
a moment before the canary Service's endpoints switch to it. Without the routing
guard, an early check can score stable pods and pass a broken build.

Comparing canary to stable alone can't separate a better model from a broken one,
because a retrained model should score differently. Parity compares the canary to
itself: the same model, offline vs online.

    python -m rollout.canary_check --canary http://localhost:8082 --stable http://localhost:8081
    python -m rollout.canary_check --canary ... --stable ... --canary-hash 7f45f598cc   (in Kubernetes)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import requests
import yaml

DEFAULT_THRESHOLDS = Path(__file__).resolve().parents[1] / "configs" / "thresholds.yaml"


def wait_ready(url: str, timeout_s: float = 60) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            if requests.get(f"{url}/ready", timeout=2).ok:
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    raise TimeoutError(f"{url} not ready after {timeout_s}s")


def wait_for_canary_pods(url: str, canary_hash: str, timeout_s: float = 90, in_a_row: int = 5) -> None:
    """Wait until the canary Service answers only from canary pods, several times in a row."""
    deadline, streak = time.time() + timeout_s, 0
    while time.time() < deadline and streak < in_a_row:
        try:
            pod = requests.get(f"{url}/ready", timeout=2, headers={"Connection": "close"}).json().get("pod", "")
            streak = streak + 1 if canary_hash in pod else 0
        except requests.RequestException:
            streak = 0
        time.sleep(0.5)
    if streak < in_a_row:
        raise TimeoutError(f"canary Service did not route only to pods with hash {canary_hash} in {timeout_s}s")


def score_all(url: str, claims: list[dict], batch: int,
              expect_hash: str | None = None) -> tuple[np.ndarray, list[float], int, int, int]:
    """Score claims in batches. Returns scores (NaN where a batch failed or came from
    the wrong pod), latencies, errors, requests, and misrouted responses."""
    scores = np.full(len(claims), np.nan)
    latencies, errors, n, misrouted = [], 0, 0, 0
    for start in range(0, len(claims), batch):
        chunk = claims[start:start + batch]
        n += 1
        t0 = time.perf_counter()
        try:
            r = requests.post(f"{url}/predict", json={"claims": chunk}, timeout=10)
            latencies.append((time.perf_counter() - t0) * 1000)
            r.raise_for_status()
            body = r.json()
            if expect_hash and expect_hash not in body.get("pod", ""):
                misrouted += 1
                continue
            scores[start:start + len(chunk)] = body["scores"]
        except requests.RequestException:
            errors += 1
    return scores, latencies, errors, n, misrouted


def analyze(expected: np.ndarray, canary: np.ndarray, stable: np.ndarray, latencies_ms: list[float],
            errors: int, requests_made: int, threshold: float, limits: dict, misrouted: int = 0) -> list[dict]:
    """Pure decision logic, kept separate from HTTP so it can be unit tested."""
    ok = ~np.isnan(canary)
    parity = float(np.max(np.abs(canary[ok] - expected[ok]))) if ok.any() else float("inf")
    error_rate = errors / max(requests_made, 1)
    p95 = float(np.percentile(latencies_ms, 95)) if latencies_ms else float("inf")
    both = ok & ~np.isnan(stable)
    delta_pts = float(abs((canary[both] > threshold).mean() - (stable[both] > threshold).mean()) * 100)

    def check(name, value, fail, warn=None):
        status = "fail" if value > fail else "warn" if warn is not None and value > warn else "pass"
        return {"name": name, "status": status, "value": value, "fail": fail}

    return [
        check("misrouted_responses", misrouted, 0),
        check("parity_max_abs_diff", parity, limits["parity_max_abs_diff"]),
        check("error_rate", error_rate, limits["max_error_rate"]),
        check("p95_latency_ms", p95, limits["max_p95_latency_ms"]),
        check("flagged_rate_delta_pts", delta_pts, limits["flagged_rate_delta_pts"]["fail"],
              limits["flagged_rate_delta_pts"]["warn"]),
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--canary", required=True)
    ap.add_argument("--stable", required=True)
    ap.add_argument("--canary-hash", help="canary pod-template hash; every canary response must come from it")
    ap.add_argument("--wait", type=float, default=90, help="seconds to wait for the canary Service to switch")
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--thresholds", type=Path, default=DEFAULT_THRESHOLDS)
    args = ap.parse_args()
    limits = yaml.safe_load(args.thresholds.read_text())["canary"]

    wait_ready(args.canary)
    wait_ready(args.stable)
    if args.canary_hash:
        wait_for_canary_pods(args.canary, args.canary_hash, args.wait)
    info = requests.get(f"{args.canary}/model-info", timeout=5).json()
    golden = requests.get(f"{args.canary}/golden", timeout=30).json()
    if args.canary_hash and args.canary_hash not in golden["pod"]:
        sys.exit(f"golden set came from {golden['pod']}, not a canary pod ({args.canary_hash})")
    claims, expected = golden["claims"], np.array(golden["expected_score"])

    canary, latencies, errors, n, misrouted = score_all(args.canary, claims, args.batch, args.canary_hash)
    stable, _, _, _, _ = score_all(args.stable, claims, args.batch)
    checks = analyze(expected, canary, stable, latencies, errors, n, info["threshold"], limits, misrouted)
    failed = [c for c in checks if c["status"] == "fail"]

    print(f"canary {info['model_version']} (build {info['build']}, pod {golden['pod']}) on {len(claims)} golden claims")
    for c in checks:
        print(f"  {c['status'].upper():4s}  {c['name']:24s} {c['value']:.6g}  (fail > {c['fail']})")
    print("RESULT", "FAIL -> roll back" if failed else "PASS -> continue")
    print(json.dumps({"model_version": info["model_version"], "build": info["build"], "checks": checks}))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
