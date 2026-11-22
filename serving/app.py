"""Denial-scoring API: serves one model bundle.

    GET  /health       liveness: the process is up
    GET  /ready        readiness: the model is loaded
    GET  /model-info   version, build, features, threshold
    GET  /golden       the bundle's golden set: inputs and offline scores (for the canary check)
    POST /predict      {"claims": [{feature: value, ...}, ...]} -> scores and flags
    GET  /metrics      request, error, and latency counters (Prometheus text format)

Environment:
    MODEL_DIR       bundle directory (default /model)
    FEATURE_ORDER   how input rows are built: manifest (correct) or hardcoded (incident 3)
    BUILD           build label returned with every response, e.g. v2-bad or v2.1
    HOSTNAME        set by Kubernetes to the pod name, which contains the pod-template
                    hash; returned so the canary check can prove which pods answered
"""

from __future__ import annotations

import os
import socket
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
import sklearn
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from serving.features import build_matrix
from training.model_io import load_bundle

POD = os.environ.get("HOSTNAME") or socket.gethostname()
STATE: dict[str, Any] = {}
METRICS = {"requests_total": 0, "errors_total": 0, "claims_scored_total": 0, "latency_seconds_sum": 0.0}
LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)
LATENCY_COUNTS = [0] * (len(LATENCY_BUCKETS) + 1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_dir = Path(os.environ.get("MODEL_DIR", "/model"))
    model, manifest, golden = load_bundle(model_dir)
    if manifest.sklearn_version != sklearn.__version__:
        # Fail fast: a pickled model on a different scikit-learn can score differently.
        raise RuntimeError(f"bundle needs scikit-learn {manifest.sklearn_version}, image has {sklearn.__version__}")
    STATE.update(model=model, manifest=manifest, golden=golden,
                 order=os.environ.get("FEATURE_ORDER", "manifest"), build=os.environ.get("BUILD", "dev"))
    yield
    STATE.clear()


app = FastAPI(title="claimshift denial scorer", lifespan=lifespan)


class PredictRequest(BaseModel):
    claims: list[dict[str, Any]]


class PredictResponse(BaseModel):
    model_version: str
    build: str
    pod: str
    scores: list[float]
    flagged: list[bool]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict:
    if "model" not in STATE:
        raise HTTPException(503, "model not loaded")
    return {"status": "ready", "model_version": STATE["manifest"].version, "build": STATE["build"], "pod": POD}


@app.get("/model-info")
def model_info() -> dict:
    m = STATE["manifest"]
    return {"model_version": m.version, "build": STATE["build"], "pod": POD, "feature_order": STATE["order"],
            "features": m.features, "threshold": m.threshold, "train_months": m.train_months,
            "label_cutoff": m.label_cutoff, "sklearn_version": m.sklearn_version}


@app.get("/golden")
def golden() -> dict:
    g, m = STATE["golden"], STATE["manifest"]
    return {"model_version": m.version, "pod": POD, "claim_id": g.claim_id.tolist(),
            "claims": g[m.features].to_dict(orient="records"), "expected_score": g.expected_score.tolist()}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    t0 = time.perf_counter()
    METRICS["requests_total"] += 1
    try:
        m = STATE["manifest"]
        scores = STATE["model"].predict_proba(build_matrix(pd.DataFrame(req.claims), m, STATE["order"]))[:, 1]
    except (KeyError, ValueError) as err:
        METRICS["errors_total"] += 1
        raise HTTPException(422, str(err)) from err
    finally:
        elapsed = time.perf_counter() - t0
        METRICS["latency_seconds_sum"] += elapsed
        LATENCY_COUNTS[next((i for i, b in enumerate(LATENCY_BUCKETS) if elapsed <= b), len(LATENCY_BUCKETS))] += 1
    METRICS["claims_scored_total"] += len(scores)
    return PredictResponse(model_version=m.version, build=STATE["build"], pod=POD, scores=scores.tolist(),
                           flagged=(scores > m.threshold).tolist())


@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> str:
    labels = f'model_version="{STATE["manifest"].version}",build="{STATE["build"]}"'
    lines = [f"scorer_{name}{{{labels}}} {value}" for name, value in METRICS.items()]
    cumulative = 0
    for bound, count in zip((*LATENCY_BUCKETS, "+Inf"), LATENCY_COUNTS):
        cumulative += count
        lines.append(f'scorer_latency_seconds_bucket{{{labels},le="{bound}"}} {cumulative}')
    return "\n".join(lines) + "\n"
