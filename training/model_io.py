"""The model bundle: the deployable unit shared by training and serving.

A bundle directory holds:

    model.joblib        the fitted classifier
    manifest.json       ordered feature names, categorical encodings, threshold, metadata
    golden_set.parquet  fixed claims: their inputs and this model's offline scores

Serving must build each inout roe By NAME from the manifest (prepare_matrix), never by
column position. Incident 3 is what happens when that rule is broken.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

@dataclass
class Manifest:
    model_name: str
    version: str
    features: list[str]
    categorical: dict[str, list[str]]   # feature -> known values; list position = code
    threshold: float
    train_months: list[str]
    label_cutoff: str
    train_rows: int = 0
    metrics: dict = field(default_factory=dict)
    created_at: str = ""
    sklearn_version: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)
    
    @classmethod
    def from_json(cls, text: str) -> "Manifest":
        return cls(**json.loads(text))

def prepare_matrix(rows: pd.DataFrame, manifest:Manifest) -> np.ndarray:
    """Model input matrix, built by feature name in manifest order.
    
    Categorical values map to theor manifest codes; a value the model never saw
    becomes NaN, which the model treats as missing.
    """

    missing = [f for f in manifest.features if f not in rows.columns]
    if missing:
         raise KeyError(f"missing features: {missing}")
    columns = []
    for name in manifest.features:
        if name in manifest.categorical:
            codes = {value: i for i, value in enumerate(manifest.categorical[name])}
            columns.append(rows[name].map(codes).astype("float64").to_numpy())
        else:
            columns.append(rows[name].astype("float64").to_numpy())
    return np.column_stack(columns)

def save_bundle(out_dir: Path, model, manifest: Manifest, golden: pd.DataFrame) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir/"model.joblib")
    (out_dir / "manifest.json").write_text(manifest.to_json())
    golden.to_parquet(out_dir / "golden_set.parquet", index=False)

def load_bundle(bundle_dir: Path) -> tuple[object, Manifest, pd.DataFrame]:
    model = joblib.load(bundle_dir / "model.joblib")
    manifest = Manifest.from_json((bundle_dir / "manifest.json").read_text())
    golden = pd.read_parquet(bundle_dir / "golden_set.parquet")
    return model, manifest, golden


