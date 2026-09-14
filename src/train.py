"""Train, select an operating threshold, evaluate, and serialize the fraud-detection pipeline."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "historical_transactions.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "fraud_detection_pipeline.joblib"
METRICS_PATH = MODEL_DIR / "model_metrics.json"

FEATURE_COLUMNS = [
    "amount", "merchant_category", "location_mismatch", "device_velocity"
]
NUMERIC_FEATURES = ["amount", "location_mismatch", "device_velocity"]
CATEGORICAL_FEATURES = ["merchant_category"]
TARGET_COLUMN = "is_fraud"
DEFAULT_THRESHOLD = 0.50
MAX_ALERT_RATE = 0.02
RANDOM_STATE = 42


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run `python src/generator.py` first."
        )
    df = pd.read_csv(DATA_PATH)
    required = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df[TARGET_COLUMN].nunique() < 2:
        raise ValueError("Dataset must contain both legitimate and fraud labels.")
    return df


def build_pipeline(scale_pos_weight: float) -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )

    classifier = XGBClassifier(
        n_estimators=350,
        max_depth=4,
        learning_rate=0.045,
        subsample=0.85,
        colsample_bytree=0.9,
        min_child_weight=3,
        gamma=0.05,
        reg_lambda=2.0,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])


def choose_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float:
    """Choose F1-optimal validation threshold subject to a 2% alert cap."""
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if len(thresholds) == 0:
        return DEFAULT_THRESHOLD

    predicted_rate = np.array([(probabilities >= t).mean() for t in thresholds])
    f1_values = (2 * precision[:-1] * recall[:-1]) / np.maximum(
        precision[:-1] + recall[:-1], 1e-12
    )
    eligible = predicted_rate <= MAX_ALERT_RATE

    if eligible.any():
        scores = np.where(eligible, f1_values, -1.0)
        return float(thresholds[int(np.argmax(scores))])
    return float(thresholds[int(np.argmax(f1_values))])


def evaluate(
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
) -> dict:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(threshold), 6),
        "precision": round(float(precision_score(y_true, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, predictions, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, probabilities)), 4),
        "pr_auc": round(float(average_precision_score(y_true, probabilities)), 4),
        "alert_rate": round(float(predictions.mean()), 4),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }


def get_feature_importance(pipeline: Pipeline) -> list[dict[str, float | str]]:
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    names = preprocessor.get_feature_names_out()
    importances = classifier.feature_importances_
    ranked = sorted(
        zip(names, importances),
        key=lambda item: float(item[1]),
        reverse=True,
    )[:10]
    return [
        {"feature": str(name), "importance": round(float(value), 6)}
        for name, value in ranked
    ]


def train_model() -> Pipeline:
    df = load_data()
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].astype(int)

    # Train / validation / test split prevents threshold selection from
    # leaking information from the final test set into model decisions.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
    )

    fraud_count = int(y_train.sum())
    legitimate_count = len(y_train) - fraud_count
    if fraud_count == 0:
        raise ValueError("Training split contains no fraud examples.")
    scale_pos_weight = legitimate_count / fraud_count

    pipeline = build_pipeline(scale_pos_weight)
    pipeline.fit(X_train, y_train)

    validation_probabilities = pipeline.predict_proba(X_val)[:, 1]
    threshold = choose_threshold(y_val, validation_probabilities)

    validation_metrics = evaluate(y_val, validation_probabilities, threshold)
    test_probabilities = pipeline.predict_proba(X_test)[:, 1]
    test_metrics = evaluate(y_test, test_probabilities, threshold)

    metrics = {
        "model": "XGBClassifier",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(df)),
        "fraud_rate": round(float(y.mean()), 4),
        "split": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "scale_pos_weight": round(float(scale_pos_weight), 4),
        "features": FEATURE_COLUMNS,
        "threshold_selection": {
            "method": "F1 maximization on validation set",
            "max_alert_rate": MAX_ALERT_RATE,
        },
        "validation": validation_metrics,
        "test": test_metrics,
        "feature_importance": get_feature_importance(pipeline),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipeline,
            "threshold": threshold,
            "model_version": "1.1.0",
            "features": FEATURE_COLUMNS,
        },
        MODEL_PATH,
    )
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("=" * 72)
    print("FRAUD DETECTION MODEL EVALUATION")
    print("=" * 72)
    print("Test-set metrics:")
    for key in ["threshold", "precision", "recall", "f1", "roc_auc", "pr_auc", "alert_rate"]:
        print(f"{key:16}: {test_metrics[key]}")
    print("\nClassification report (test set):")
    print(classification_report(
        y_test,
        (test_probabilities >= threshold).astype(int),
        target_names=["Legitimate", "Fraud"],
        zero_division=0,
    ))
    print(f"Pipeline: {MODEL_PATH}")
    print(f"Metrics : {METRICS_PATH}")
    return pipeline


if __name__ == "__main__":
    train_model()
