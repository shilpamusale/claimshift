"""Send claims to the main service and show which builds answered.

During a canary at 25%, about a quarter of responses should come from the canary.

    python -m rollout.traffic --url http://localhost:8080 --requests 200
"""

from __future__ import annotations

import argparse
from collections import Counter

import requests


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--requests", type=int, default=200)
    args = ap.parse_args()

    claim = requests.get(f"{args.url}/golden", timeout=30).json()["claims"][0]
    builds = Counter()
    session = requests.Session()
    session.headers["Connection"] = "close"  # new connection per request, so the Service spreads them
    for _ in range(args.requests):
        try:
            builds[session.post(f"{args.url}/predict", json={"claims": [claim]}, timeout=5).json()["build"]] += 1
        except requests.RequestException:
            builds["error"] += 1
    for build, n in builds.most_common():
        print(f"{build:8s} {n:5d}  {n / args.requests:6.1%}")


if __name__ == "__main__":
    main()