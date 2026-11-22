"""Turning request rows into the model's input matrix.

FEATURE_ORDER=manifest      correct: columns by name, in the order stored in the 
                            model's manifest
FEATURE_ORDER=hardcoded     incident 3: a hand-maintained feature list copied from an 
                            old notebook, with two rate features in the wrong order.
                            It passes a "same feature names?" check, nothing errors,
                            and the average score barely moves, but individua; claims
                            are mis-scored by upto 0.11.

"""
from __future__ import annotations

import numpy as np
import pandas as pd

from training.model_io import Manifest, prepare_matrix

ORDERS = ("manifest","hardcoded")

HARDCODED_FEATURES = [
    "log_billed", "days_to_submit", "pa_required", "pa_missing",
    "code_rate", "code_support", "code_level",
    "payer_code_rate", "payer_rate",          # BUG: training has payer_rate first
    "tenant_rate", "tenant_pa_miss_rate", "tenant_log_volume", "tenant_median_days_to_submit",
    "payer_id", "proc_category", "icd_chapter",
]


def build_matrix(rows: pd.DataFrame, manifest: Manifest, order: str = "manifest") -> np.ndarray:
    if order == "manifest":
        return prepare_matrix(rows, manifest)
    if order == "hardcoded":
        if set(HARDCODED_FEATURES) != set(manifest.features):
            raise ValueError("hardcoded feature list does not match the model")  # this check passes
        return prepare_matrix(rows, Manifest(**{**manifest.__dict__, "features": HARDCODED_FEATURES}))
    raise ValueError(f"FEATURE_ORDER must be one of {ORDERS}, got {order!r}")