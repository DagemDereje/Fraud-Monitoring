"""FastAPI inference service for real-time transaction risk scoring."""
from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from scoring import load_model, score


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_model()
    yield


app = FastAPI(
    title="Real-Time Transaction Monitoring API",
    description="Local XGBoost inference API for synthetic transaction fraud monitoring.",
    version="2.1.0",
)


class TransactionRequest(BaseModel):
    transaction_id: str = Field(..., min_length=1, max_length=100)
    timestamp: str = Field(..., min_length=1, max_length=100)
    account_id: str = Field(..., min_length=1, max_length=100)
    amount: float = Field(..., gt=0, le=1_000_000)
    merchant_category: str = Field(..., min_length=1, max_length=100)
    location_mismatch: int = Field(..., ge=0, le=1)
    device_velocity: int = Field(..., ge=0, le=1000)


class PredictionResponse(BaseModel):
    transaction_id: str
    is_fraud: bool
    fraud_probability: float
    decision_threshold: float
    risk_level: str
    risk_reasons: list[str]
    latency_ms: float
    model_version: str


def risk_reasons(transaction: TransactionRequest) -> list[str]:
    reasons: list[str] = []
    if transaction.amount > 5000:
        reasons.append("High-value transaction above $5,000")
    elif transaction.amount > 2000:
        reasons.append("Elevated transaction amount above $2,000")
    if transaction.location_mismatch:
        reasons.append("Transaction location differs from account home location")
    if transaction.device_velocity > 5:
        reasons.append("High device velocity: more than 5 transactions in the last hour")
    elif transaction.device_velocity > 3:
        reasons.append("Elevated device velocity: more than 3 transactions in the last hour")
    if transaction.merchant_category in {"jewelry", "gaming"}:
        reasons.append(f"Higher-risk merchant category: {transaction.merchant_category}")
    if transaction.location_mismatch and transaction.device_velocity > 3:
        reasons.append("Combined location mismatch and high velocity pattern")
    return reasons or ["No strong rule-based risk indicators"]


def risk_level(probability: float, is_fraud: bool) -> str:
    if is_fraud or probability >= 0.80:
        return "HIGH"
    if probability >= 0.35:
        return "MEDIUM"
    return "LOW"


@app.get("/")
async def root() -> dict[str, str]:
    bundle = load_model()
    return {
        "service": "Real-Time Transaction Monitoring",
        "status": "online",
        "model": "XGBoost",
        "model_version": str(bundle.get("model_version", "unknown")),
        "version": app.version,
    }


@app.get("/health")
async def health() -> dict[str, object]:
    try:
        bundle = load_model()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_version": str(bundle.get("model_version", "unknown")),
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(transaction: TransactionRequest) -> PredictionResponse:
    start = time.perf_counter()

    try:
        probability, is_fraud, threshold, model_version = score(
            amount=transaction.amount,
            merchant_category=transaction.merchant_category,
            location_mismatch=transaction.location_mismatch,
            device_velocity=transaction.device_velocity,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc

    latency_ms = (time.perf_counter() - start) * 1000
    return PredictionResponse(
        transaction_id=transaction.transaction_id,
        is_fraud=is_fraud,
        fraud_probability=round(probability, 6),
        decision_threshold=round(threshold, 6),
        risk_level=risk_level(probability, is_fraud),
        risk_reasons=risk_reasons(transaction),
        latency_ms=round(latency_ms, 3),
        model_version=model_version,
    )
