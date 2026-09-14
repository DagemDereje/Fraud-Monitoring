# Portfolio Demo Guide

## 60-second walkthrough

### 1. Start the dashboard

```bash
streamlit run src/dashboard.py
```

This is the only service needed for the live demo — it scores transactions
in-process using the same pipeline the FastAPI service serves.

### 2. (Optional) Start the API separately

```bash
uvicorn src.app:app --reload
```

Useful if you also want to demonstrate the REST endpoint (`/docs`,
`/predict`) alongside the dashboard.

### 3. Show the live stream

Keep **Live monitoring** enabled. The dashboard processes one synthetic transaction every 1.5 seconds.

### 4. Trigger a controlled alert

In the sidebar select **Combined suspicious behavior** and click **Generate scenario now**.

The scenario intentionally combines:

- high transaction amount
- location mismatch
- high device velocity

The generated event is scored through the same pipeline as every random event. No special prediction path is used.

### 5. Explain the result

Point out:

- fraud probability
- decision threshold
- risk level
- risk reasons
- model inference latency

### 6. Show the model tab

Open **Model Performance** to explain the held-out test metrics and feature importance. Emphasize that the dataset and labels are synthetic.

## Important portfolio statement

This is a local ML engineering demonstration. It is not a production banking fraud system and should not be used to make real financial decisions.
