import numpy as np
import pandas as pd
import pytest
import sklearn
import yaml
from fastapi.testclient import TestClient
from sklearn.ensemble import HistGradientBoostingClassifier

from rollout.canary_check import analyze
from serving.app import app
from serving.features import HARDCODED_FEATURES
from training.model_io import Manifest, prepare_matrix, save_bundle
from training.train import CONFIG, feature_list

CFG = yaml.safe_load(CONFIG.read_text())
FEATURES = feature_list(CFG, CFG["versions"]["v2"])
CATEGORICAL = {"payer_id": ["P01", "P02", "P03"], "proc_category": ["em_visit", "lab"], "icd_chapter": ["09", "15"]}
LIMITS = yaml.safe_load((CONFIG.parent / "thresholds.yaml").read_text())["canary"]


def make_bundle(path, sklearn_version=None):
    rng = np.random.default_rng(7)
    n = 2000
    df = pd.DataFrame({f: rng.random(n) for f in FEATURES if f not in CATEGORICAL})
    for f, values in CATEGORICAL.items():
        df[f] = rng.choice(values, n)
    df["payer_rate"] = rng.uniform(0.05, 0.10, n)          # similar scales, like the real features
    df["payer_code_rate"] = rng.uniform(0.02, 0.30, n)
    df["denied"] = (rng.random(n) < 0.05 + df.payer_code_rate + 0.5 * (df.payer_rate - 0.05)).astype(int)
    df["claim_id"] = np.arange(n)

    manifest = Manifest(model_name="denial-scorer", version="v2", features=FEATURES, categorical=CATEGORICAL,
                        threshold=0.2, train_months=["2026-08"], label_cutoff="2026-12-01",
                        sklearn_version=sklearn_version or sklearn.__version__)
    model = HistGradientBoostingClassifier(
        categorical_features=[FEATURES.index(f) for f in CATEGORICAL], max_iter=60, random_state=7)
    model.fit(prepare_matrix(df, manifest), df.denied)
    golden = df.head(300).copy()
    golden["expected_score"] = model.predict_proba(prepare_matrix(golden, manifest))[:, 1]
    save_bundle(path, model, manifest, golden)


@pytest.fixture
def client(tmp_path, monkeypatch):
    def _client(order="manifest", sklearn_version=None):
        make_bundle(tmp_path / "bundle", sklearn_version)
        monkeypatch.setenv("MODEL_DIR", str(tmp_path / "bundle"))
        monkeypatch.setenv("FEATURE_ORDER", order)
        monkeypatch.setenv("BUILD", f"test-{order}")
        return TestClient(app)
    return _client


def golden_round_trip(c):
    golden = c.get("/golden").json()
    scores = c.post("/predict", json={"claims": golden["claims"]}).json()["scores"]
    return np.array(scores), np.array(golden["expected_score"])


def test_ready_and_model_info(client):
    with client() as c:
        assert c.get("/health").json() == {"status": "ok"}
        ready = c.get("/ready").json()
        assert ready["model_version"] == "v2" and ready["pod"]
        info = c.get("/model-info").json()
        assert info["features"] == FEATURES and info["feature_order"] == "manifest"


def test_served_scores_match_offline_scores(client):
    with client() as c:
        online, offline = golden_round_trip(c)
    assert np.abs(online - offline).max() <= LIMITS["parity_max_abs_diff"]


def test_hardcoded_feature_list_mis_scores_without_errors(client):
    """Incident 3: every request succeeds, but scores no longer match training.
    (On the real v2 model the average barely moves: 12.25% vs 11.88%.)"""
    with client("hardcoded") as c:
        online, offline = golden_round_trip(c)
    assert np.isfinite(online).all()
    assert np.abs(online - offline).max() > 0.01


def test_hardcoded_list_differs_from_training_only_by_one_swap():
    assert set(HARDCODED_FEATURES) == set(FEATURES)
    moved = [f for f, g in zip(FEATURES, HARDCODED_FEATURES) if f != g]
    assert sorted(moved) == ["payer_code_rate", "payer_rate"]


def test_missing_feature_returns_422(client):
    with client() as c:
        claim = c.get("/golden").json()["claims"][0]
        del claim["tenant_rate"]
        r = c.post("/predict", json={"claims": [claim]})
    assert r.status_code == 422 and "tenant_rate" in r.text


def test_wrong_sklearn_version_refuses_to_start(client):
    with pytest.raises(RuntimeError, match="scikit-learn"):
        with client(sklearn_version="0.0.1"):
            pass


def test_metrics_count_requests(client):
    with client() as c:
        golden_round_trip(c)
        text = c.get("/metrics").text
    assert 'scorer_claims_scored_total{model_version="v2",build="test-manifest"}' in text


def test_canary_analysis_decisions():
    expected = np.linspace(0.05, 0.6, 100)
    ok = analyze(expected, expected.copy(), expected.copy(), [12.0] * 10, 0, 10, 0.2, LIMITS)
    assert all(c["status"] == "pass" for c in ok)

    off = expected.copy()
    off[3] += 0.05
    parity = {c["name"]: c["status"] for c in analyze(expected, off, expected, [12.0] * 10, 0, 10, 0.2, LIMITS)}
    assert parity["parity_max_abs_diff"] == "fail"

    errors = {c["name"]: c["status"] for c in analyze(expected, expected, expected, [12.0] * 8, 2, 10, 0.2, LIMITS)}
    assert errors["error_rate"] == "fail"

    # Answers from a stable pod during the canary check must fail it, even if they match.
    routed = {c["name"]: c["status"] for c in analyze(expected, expected, expected, [12.0] * 10, 0, 10, 0.2,
                                                       LIMITS, misrouted=1)}
    assert routed["misrouted_responses"] == "fail"
