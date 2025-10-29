from __future__ import annotations
import os
from pathlib import Path
from typing import Sequence, Dict, Any
import numpy as np
import pandas as pd
import pickle

_model_bundle = None
_model_path = None

def load_model(path: str | os.PathLike) -> Dict[str, Any]:
    global _model_bundle, _model_path
    path = str(path)
    if _model_bundle is None or _model_path != path:
        # Load with pickle (current bundle format)
        try:
            with open(path, "rb") as f:
                _model_bundle = pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"Unable to load model bundle from {path}: {e}")
        _model_path = path
    return _model_bundle


def compute_derived_features(latest: pd.Series) -> Dict[str, float]:
    c = float(latest.get("close", np.nan))
    short_ma = float(latest.get("short_ma", np.nan))
    long_ma = float(latest.get("long_ma", np.nan))
    trend_ma = float(latest.get("trend_ma", np.nan))
    prev_short = float(latest.get("prev_short_ma", np.nan))
    prev_high20 = float(latest.get("prev_high20", np.nan)) if not pd.isna(latest.get("prev_high20", np.nan)) else np.nan
    ma_dist_bps = ((short_ma - long_ma) / c) * 10000.0 if c > 0 else np.nan
    above_trend = 1.0 if (c > trend_ma) else 0.0
    breakout_gap_bps = ((c - prev_high20) / c) * 10000.0 if (c > 0 and not np.isnan(prev_high20)) else 0.0
    return {
        "ma_dist_bps": ma_dist_bps,
        "above_trend": above_trend,
        "breakout_gap_bps": breakout_gap_bps,
    }


def predict_entry_prob(latest: pd.Series, model_path: str) -> float:
    bundle = load_model(model_path)
    features: Sequence[str] = bundle["features"]

    # Assemble input vector matching training features
    latest = latest.copy()
    derived = compute_derived_features(latest)
    for k, v in derived.items():
        latest[k] = v
    x = np.array([[float(latest.get(col, np.nan)) for col in features]], dtype=float)
    # If any features are nan, prediction likely invalid; return 0
    if np.isnan(x).any():
        return 0.0
    # Two bundle formats: numpy_mlp (custom) and sklearn (legacy)
    model_type = bundle.get("model_type", "sklearn")
    if model_type == "numpy_mlp":
        sc = bundle.get("scaler", {})
        mean = np.array(sc.get("mean"), dtype=float)
        std = np.array(sc.get("std"), dtype=float)
        xs = (x - mean) / (std + 1e-8)
        m = bundle["model"]
        W1, b1 = np.array(m["W1"], dtype=float), np.array(m["b1"], dtype=float)
        W2, b2 = np.array(m["W2"], dtype=float), np.array(m["b2"], dtype=float)
        W3, b3 = np.array(m["W3"], dtype=float), np.array(m["b3"], dtype=float)
        def relu(z):
            return np.maximum(0.0, z)
        def sigmoid(z):
            return 1.0 / (1.0 + np.exp(-z))
        a1 = relu(xs @ W1 + b1)
        a2 = relu(a1 @ W2 + b2)
        p = sigmoid(a2 @ W3 + b3)
        return float(np.clip(p[0, 0], 1e-6, 1 - 1e-6))
    else:
        # Legacy sklearn bundle
        try:
            model = bundle["model"]
            scaler = bundle["scaler"]
            xs = scaler.transform(x)
            proba = float(model.predict_proba(xs)[0, 1])
            return proba
        except Exception:
            return 0.0
