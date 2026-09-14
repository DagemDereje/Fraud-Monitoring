"""Shared model-loading and scoring logic.

Used by both the FastAPI service (src/app.py, for a standalone deployable API)
and the Streamlit dashboard (src/dashboard.py, which calls this in-process so
the live demo does not depend on a second always-on service).
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "fraud_detection_pipeline.joblib"

_model_bundle: dict | None = None


def load_model() -> dict:
    """Load (and cache) the trained pipeline bundle."""
    global _model_bundle
    if _model_bundle is None:
        if not MODEL_PATH.exists():
            raise RuntimeError(
                f"Model artifact not found at {MODEL_PATH}. "
                "Run `python src/generator.py` and `python src/train.py` first."
            )
        _model_bundle = joblib.load(MODEL_PATH)
    return _model_bundle


def score(
    amount: float,
    merchant_category: str,
    location_mismatch: int,
    device_velocity: int,
) -> tuple[float, bool, float, str]:
    """Run inference for one transaction.

    Returns (fraud_probability, is_fraud, decision_threshold, model_version).
    """
    bundle = load_model()
    pipeline = bundle["pipeline"]
    threshold = float(bundle["threshold"])
    model_version = str(bundle.get("model_version", "unknown"))

    input_data = pd.DataFrame([{
        "amount": amount,
        "merchant_category": merchant_category,
        "location_mismatch": location_mismatch,
        "device_velocity": device_velocity,
    }])

    probability = float(pipeline.predict_proba(input_data)[0, 1])
    is_fraud = probability >= threshold
    return probability, is_fraud, threshold, model_version
