# API Examples

Start the service:

```bash
uvicorn src.app:app --reload
```

## Health

```bash
curl http://127.0.0.1:8000/health
```

## Prediction

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "TXN_DEMO_001",
    "timestamp": "2026-09-03T10:00:00+00:00",
    "account_id": "ACC_00001",
    "amount": 8500.00,
    "merchant_category": "jewelry",
    "location_mismatch": 1,
    "device_velocity": 6
  }'
```

The response includes the model probability, selected operating threshold, risk level, rule-based explanations, model version, and inference latency.
