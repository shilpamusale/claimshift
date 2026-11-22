import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier

from training.golden_set import select_golden_set
from training.model_io import Manifest, load_bundle, prepare_matrix, save_bundle


@pytest.fixture
def frame():
    rng = np.random.default_rng(7)
    n = 3000
    df = pd.DataFrame({
        "claim_id": np.arange(n),
        "tenant_id": rng.choice(["T01", "T02", "T04", "T11"], n),
        "icd_code": rng.choice(["I10", "E11.9", "O00.512", "M67.A01"], n),
        "payer_id": rng.choice(["P01", "P02", "P03"], n),
        "code_rate": rng.random(n),
        "tenant_rate": rng.random(n),
        "log_billed": rng.normal(5, 1, n),
    })
    df["denied"] = (rng.random(n) < 0.2 + 0.5 * df.code_rate * (df.payer_id == "P02")).astype(int)
    return df


@pytest.fixture
def manifest():
    return Manifest(model_name="denial-scorer", version="test",
                    features=["code_rate", "tenant_rate", "log_billed", "payer_id"],
                    categorical={"payer_id": ["P01", "P02", "P03"]}, threshold=0.2,
                    train_months=["2026-08"], label_cutoff="2026-10-01")


@pytest.fixture
def model(frame, manifest):
    m = HistGradientBoostingClassifier(categorical_features=[3], max_iter=50, random_state=7)
    return m.fit(prepare_matrix(frame, manifest), frame.denied)


def test_prepare_matrix_ignores_column_order(frame, manifest):
    shuffled = frame[frame.columns[::-1]]
    assert np.array_equal(prepare_matrix(frame, manifest), prepare_matrix(shuffled, manifest))


def test_unknown_category_becomes_missing(frame, manifest):
    frame.loc[0, "payer_id"] = "P99"  # a payer the model never saw
    assert np.isnan(prepare_matrix(frame, manifest)[0, 3])


def test_missing_feature_fails_loudly(frame, manifest):
    with pytest.raises(KeyError, match="tenant_rate"):
        prepare_matrix(frame.drop(columns="tenant_rate"), manifest)


def test_bundle_round_trip_reproduces_scores(tmp_path, frame, manifest, model):
    golden = frame.head(200).copy()
    golden["expected_score"] = model.predict_proba(prepare_matrix(golden, manifest))[:, 1]
    save_bundle(tmp_path / "test", model, manifest, golden)

    loaded, loaded_manifest, loaded_golden = load_bundle(tmp_path / "test")
    scores = loaded.predict_proba(prepare_matrix(loaded_golden, loaded_manifest))[:, 1]
    assert np.abs(scores - loaded_golden.expected_score).max() == 0.0


def test_feature_order_bug_silently_changes_scores(frame, manifest, model):
    """Incident 3 in miniature: building the row alphabetically instead of by the
    manifest raises no error, but the scores are wrong."""
    right = model.predict_proba(prepare_matrix(frame, manifest))[:, 1]
    alphabetical = Manifest(**{**manifest.__dict__, "features": sorted(manifest.features)})
    wrong = model.predict_proba(prepare_matrix(frame, alphabetical))[:, 1]
    assert np.abs(right - wrong).max() > 0.01


def test_golden_set_is_fixed_and_covers_risky_paths(frame):
    args = dict(size=500, new_codes={"O00.512", "M67.A01"}, ob_tenants=["T04", "T11"],
                new_code_claims=100, ob_tenant_claims=80, seed=7)
    a, b = select_golden_set(frame, **args), select_golden_set(frame, **args)
    assert a.claim_id.tolist() == b.claim_id.tolist()      # same claims every time
    assert len(a) == 500 and a.claim_id.is_unique
    assert a.icd_code.isin(args["new_codes"]).sum() >= 100