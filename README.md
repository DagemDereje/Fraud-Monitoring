# Real-Time Transaction Monitoring (AI/ML)

A portfolio-ready, local-first fraud monitoring system that demonstrates an end-to-end machine-learning inference workflow:

**Synthetic transactions → XGBoost training → operating-threshold selection → FastAPI inference → Streamlit real-time monitoring**

The project intentionally avoids Kafka and other distributed infrastructure so the complete system can be run locally and understood from source code.

## What this demonstrates

- Synthetic high-volume credit-card transaction generation
- Severe class-imbalance handling with stratified train/test split and `scale_pos_weight`
- XGBoost binary classification
- Precision, Recall, F1, ROC-AUC and PR-AUC evaluation
- Operational decision-threshold selection for a fraud-alerting use case
- Serialized preprocessing + model pipeline with Joblib
- FastAPI REST inference endpoint with Pydantic validation
- Inference latency measurement
- Rule-based risk-signal explanations alongside ML probability
- Streamlit real-time dashboard using `st.fragment(run_every=1.5)`
- Controlled fraud-demo scenarios for reproducible portfolio demonstrations
- Model-performance, system-information, and live-monitor views
- Session-level monitoring metrics and live transaction feed
- Unit tests for generator and API logic

## Architecture

```text
                         OFFLINE
┌────────────────────┐       ┌──────────────────────┐
│ Synthetic Generator│──────▶│ historical CSV       │
└────────────────────┘       └──────────┬───────────┘
                                        │
                                        ▼
                              ┌─────────────────────┐
                              │ XGBoost Training    │
                              │ preprocessing       │
                              │ imbalance handling  │
                              │ threshold selection │
                              └──────────┬──────────┘
                                         │
                                         ▼
                              models/*.joblib

                         ONLINE / LOCAL
┌────────────────────┐  in-process   ┌───────────────────┐
│ Transaction        │──────────────▶│ src/scoring.py    │
│ Generator          │               │ XGBoost inference │
└─────────┬──────────┘               └─────────┬─────────┘
          │                                    │
          │                                    ▼
          │                            prediction + risk
          │                                    │
          └────────────────────────────────────┘
                       Streamlit Dashboard

The FastAPI service (src/app.py) calls the same src/scoring.py module and
can be run and deployed independently for programmatic /predict access —
the dashboard does not depend on it being started.
```

## Quick start

### 1. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Windows CMD:

```cmd
python -m venv .venv
.venv\Scripts\activate
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Generate the historical dataset

```bash
python src/generator.py
```

Expected: 10,000 rows with approximately 0.5–1.0% fraud labels.

### 4. Train the model

```bash
python src/train.py
```

This creates:

```text
models/
├── fraud_detection_pipeline.joblib
└── model_metrics.json
```

The Joblib artifact contains the fitted preprocessing pipeline, model, selected operating threshold, feature list, and model version.

### 5. Start the dashboard

```bash
streamlit run src/dashboard.py
```

The dashboard generates and scores a transaction every 1.5 seconds, calling
the model in-process — no other service needs to be running.

### 6. (Optional) Run the FastAPI service separately

The same model is also exposed as a standalone REST API, useful for
programmatic access or as its own deployment:

```bash
uvicorn src.app:app --reload
```

Open the interactive API documentation at:

```text
http://127.0.0.1:8000/docs
```

## Why the dashboard does not use `while True`

A blocking infinite loop is a poor fit for Streamlit because Streamlit reruns scripts as part of its application execution model. The original implementation could therefore produce a blank/black page.

This version uses:

```python
@st.fragment(run_every=1.5)
```

The result is still a continuous local transaction simulation, but the UI remains responsive and renderable.

## Data generation and fraud patterns

The generator intentionally creates a highly imbalanced dataset. The strongest synthetic fraud signals are:

1. **High-value transactions** — amounts above $5,000 sharply increase fraud propensity.
2. **Location mismatch** — transaction location differs from the account's normal location.
3. **Device velocity** — number of transactions from the same account/device context during the previous hour.
4. **Combined behavior** — location mismatch + velocity above 3 is a particularly strong signal.

Live generation occasionally creates burst behavior so `device_velocity` is not permanently zero and the dashboard can visibly exercise the behavioral features.

## Model methodology

The model uses:

- `StandardScaler` for numeric features
- `OneHotEncoder(handle_unknown="ignore")` for merchant category
- `XGBClassifier` for binary fraud classification
- `scale_pos_weight = legitimate / fraud` for class imbalance
- stratified 80/20 train/test split
- Precision, Recall, F1, ROC-AUC and PR-AUC
- threshold selection using validation predictions with an alert-rate constraint of at most 2%

The selected operating threshold is saved with the model so API inference does not silently fall back to `0.50`.

## API response

`POST /predict` returns:

```json
{
  "transaction_id": "TXN_ABC123",
  "is_fraud": true,
  "fraud_probability": 0.934521,
  "decision_threshold": 0.974607,
  "risk_level": "HIGH",
  "risk_reasons": [
    "High-value transaction above $5,000",
    "Combined location mismatch and high velocity pattern"
  ],
  "latency_ms": 4.812
}
```

## Testing

Run:

```bash
pytest -q
```

The tests cover the transaction schema, historical-data assumptions, controlled risk scenarios, and rule-based API risk explanations.

## Portfolio demo flow

For a quick demonstration, start the API and dashboard, keep **Live monitoring** enabled, then use the sidebar **Demo Scenario** selector.

Recommended sequence:

1. **Random realistic stream** — demonstrate continuous monitoring.
2. **High-value transaction** — demonstrate amount-based risk.
3. **Combined suspicious behavior** — demonstrate the strongest synthetic fraud pattern and show the model alert.
4. Open **Model Performance** — show precision, recall, F1, PR-AUC, ROC-AUC, threshold, confusion matrix, and feature importance.
5. Open **System Info** — explain the intentionally simple local architecture.

## Portfolio / production caveat

This is a **portfolio and local demonstration system**, not a production banking fraud platform. It uses synthetic data, in-process/local inference, no authentication, and in-memory dashboard state.

A real banking deployment would additionally require secure service-to-service authentication, TLS, secrets management, persistent storage, audit logging, model/version governance, drift monitoring, PII controls, high-availability deployment, observability, and a production streaming platform where justified.

## Suggested portfolio talking points

- Designed an end-to-end ML transaction monitoring architecture rather than a standalone notebook.
- Addressed extreme class imbalance and threshold selection as an operational fraud-detection problem.
- Separated data generation, training, model serving, and visualization into independent modules.
- Exposed the model through a validated REST API and measured inference latency.
- Built a real-time dashboard that surfaces both model probability and interpretable behavioral risk signals.
