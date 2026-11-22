"""Train a denial-model version, evaluate it, log it to MLflow, and write its bundle.

    python -m training.train --version v1
    python -m training.train --all

Versions live in configs/training.yaml. Labels are point-in-time: a version trained
"on Oct 1" only learns from claims whose denial result arrived before Oct 1.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import pandas as pd
import sklearn
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from pipeline.features import NUMERIC_FEATURES
from training.golden_set import select_golden_set
from training.model_io import Manifest, prepare_matrix, save_bundle

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "training.yaml"

# MLflow 3 saves scikit-learn models with skops instead of pickle and only loads
# types it is told to trust. We built this model ourselves, so these are safe.
SKOPS_TRUSTED_TYPES = [
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
    "sklearn.utils.validation.check_array",
    "functools.partial",
]


def load_features(data_dir: Path, months: list[str]) -> pd.DataFrame:
    frames = []
    for month in months:
        df = pd.read_parquet(data_dir / "features" / f"service_month={month}")
        df["service_month"] = month
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def new_fy2027_codes(data_dir: Path) -> set[str]:
    changes = pd.read_parquet(data_dir / "reference" / "icd10cm_fy2027_changes.parquet")
    return set(changes.loc[changes.change_type != "remap", "new_code"])


def feature_list(cfg: dict, version: dict) -> list[str]:
    excluded = set(version.get("exclude_features", []))
    return [f for f in NUMERIC_FEATURES if f not in excluded] + cfg["categorical_features"]


def golden_inputs(cfg: dict, data_dir: Path, new_codes: set[str]) -> pd.DataFrame:
    """Selected once and cached, so every version is scored on the same claims."""
    path = data_dir / "golden" / "golden_inputs.parquet"
    if not path.exists():
        g = cfg["golden_set"]
        df = select_golden_set(load_features(data_dir, [g["month"]]), g["size"], new_codes,
                               cfg["evaluation"]["ob_tenants"], g["new_code_claims"], g["ob_tenant_claims"], g["seed"])
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
    return pd.read_parquet(path)


def evaluate(model, manifest: Manifest, df: pd.DataFrame) -> dict:
    if df.empty or df.denied.nunique() < 2:
        return {"rows": len(df)}
    score = model.predict_proba(prepare_matrix(df, manifest))[:, 1]
    y = df.denied.to_numpy()
    return {
        "rows": len(df),
        "auc": round(roc_auc_score(y, score), 4),
        "pr_auc": round(average_precision_score(y, score), 4),
        "brier": round(brier_score_loss(y, score), 4),
        "denial_rate": round(float(y.mean()), 4),
        "mean_score": round(float(score.mean()), 4),
        "flagged_rate": round(float((score > manifest.threshold).mean()), 4),
    }


def train_version(cfg: dict, name: str, data_dir: Path, models_dir: Path) -> dict:
    v = cfg["versions"][name]
    features = feature_list(cfg, v)
    train = load_features(data_dir, v["train_months"])
    train = train[pd.to_datetime(train.remit_date) < pd.Timestamp(v["label_cutoff"])]  # labels known by then

    manifest = Manifest(
        model_name=cfg["model_name"], version=name, features=features,
        categorical={f: sorted(train[f].astype(str).unique()) for f in cfg["categorical_features"]},
        threshold=cfg["flag_threshold"], train_months=v["train_months"], label_cutoff=v["label_cutoff"],
        train_rows=len(train), created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        sklearn_version=sklearn.__version__,
    )
    model = HistGradientBoostingClassifier(
        categorical_features=[i for i, f in enumerate(features) if f in manifest.categorical], **cfg["params"])
    t0 = time.perf_counter()
    model.fit(prepare_matrix(train, manifest), train.denied.to_numpy())
    fit_seconds = round(time.perf_counter() - t0, 1)

    ev = cfg["evaluation"]
    new_codes = new_fy2027_codes(data_dir)
    eval_sets = {}
    for window in ("pre", "cutover", "post"):
        if set(ev[window]) & set(v["train_months"]):
            continue  # never evaluate on training months
        df = load_features(data_dir, ev[window])
        eval_sets[window] = df
        if window != "pre":
            eval_sets[f"{window}_new_codes"] = df[df.icd_code.isin(new_codes)]
            eval_sets[f"{window}_ob_tenants"] = df[df.tenant_id.isin(ev["ob_tenants"])]
    metrics = {k: evaluate(model, manifest, df) for k, df in eval_sets.items()}

    golden = golden_inputs(cfg, data_dir, new_codes).copy()
    golden["expected_score"] = model.predict_proba(prepare_matrix(golden, manifest))[:, 1]
    metrics["golden"] = {
        "rows": len(golden),
        "mean_score": round(float(golden.expected_score.mean()), 4),
        "flagged_rate": round(float((golden.expected_score > manifest.threshold).mean()), 4),
    }
    manifest.metrics = metrics
    save_bundle(models_dir / name, model, manifest, golden)

    run_id, registered = log_to_mlflow(cfg, name, v, manifest, model, models_dir / name, fit_seconds)
    return {"version": name, "train_rows": len(train), "fit_seconds": fit_seconds, "mlflow_run_id": run_id,
            "registered_version": registered, "metrics": metrics}


def log_to_mlflow(cfg: dict, name: str, v: dict, manifest: Manifest, model, bundle_dir: Path,
                  fit_seconds: float) -> tuple[str, str | None]:
    """MLflow is the system of record; the bundle is what gets deployed."""
    mlflow.set_tracking_uri(cfg["tracking_uri"].replace("sqlite:///", f"sqlite:///{ROOT}/"))
    if mlflow.get_experiment_by_name(cfg["experiment"]) is None:
        mlflow.create_experiment(cfg["experiment"], artifact_location=(ROOT / "mlartifacts").as_uri())
    mlflow.set_experiment(cfg["experiment"])

    registered = None
    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tags({"version": name, "note": v["note"]})
        mlflow.log_params({
            "train_months": ",".join(v["train_months"]), "label_cutoff": v["label_cutoff"],
            "train_rows": manifest.train_rows, "features": ",".join(manifest.features),
            **{f"hgb_{k}": val for k, val in cfg["params"].items()},
        })
        mlflow.log_metric("fit_seconds", fit_seconds)
        for set_name, values in manifest.metrics.items():
            for metric, value in values.items():
                mlflow.log_metric(f"{set_name}_{metric}", value)
        mlflow.log_artifacts(str(bundle_dir), artifact_path="bundle")
        if v.get("register"):
            info = mlflow.sklearn.log_model(
                model, name="model", registered_model_name=cfg["model_name"],
                pip_requirements=[f"scikit-learn=={sklearn.__version__}"],
                skops_trusted_types=SKOPS_TRUSTED_TYPES)
            registered = str(info.registered_model_version)
            client = mlflow.MlflowClient()
            client.set_registered_model_alias(cfg["model_name"], v["alias"], registered)
            client.set_model_version_tag(cfg["model_name"], registered, "bundle", name)
        return run.info.run_id, registered


def print_results(results: list[dict]) -> None:
    def cell(r, key, metric="auc"):
        m = r["metrics"].get(key, {})
        if metric == "calibration":
            return f"{m['mean_score']:.1%} / {m['denial_rate']:.1%}" if "auc" in m else "—"
        return f"{m[metric]:.4f}" if metric in m else "—"
    head = (f"{'version':8s} {'train rows':>10s} {'pre AUC':>8s} {'Oct AUC':>8s} {'Oct new-code AUC':>16s} "
            f"{'Oct new-code pred/actual':>24s} {'Nov-Dec AUC':>11s} {'Nov-Dec new-code AUC':>20s}")
    print("\n" + head + "\n" + "-" * len(head))
    for r in results:
        print(f"{r['version']:8s} {r['train_rows']:>10,d} {cell(r, 'pre'):>8s} {cell(r, 'cutover'):>8s} "
              f"{cell(r, 'cutover_new_codes'):>16s} {cell(r, 'cutover_new_codes', 'calibration'):>24s} "
              f"{cell(r, 'post'):>11s} {cell(r, 'post_new_codes'):>20s}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--version", help="one version from configs/training.yaml")
    group.add_argument("--all", action="store_true", help="train every version")
    ap.add_argument("--data", type=Path, default=ROOT / "data")
    ap.add_argument("--models", type=Path, default=ROOT / "models")
    args = ap.parse_args()

    cfg = yaml.safe_load(CONFIG.read_text())
    names = list(cfg["versions"]) if args.all else [args.version]
    results = [train_version(cfg, name, args.data, args.models) for name in names]
    print_results(results)

    out = ROOT / "docs" / "results" / "training.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(out.read_text()) if out.exists() else {}
    existing.update({r["version"]: r for r in results})
    out.write_text(json.dumps(existing, indent=2))
    print(f"bundles: {args.models}   results: {out}   browse runs: make mlflow-ui")


if __name__ == "__main__":
    main()