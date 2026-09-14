"""Portfolio-ready Streamlit dashboard for live transaction risk monitoring."""
from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path

import pandas as pd
import streamlit as st

from app import risk_level, risk_reasons, TransactionRequest
from generator import SCENARIOS, TransactionGenerator
from scoring import load_model, score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METRICS_PATH = PROJECT_ROOT / "models" / "model_metrics.json"
STREAM_INTERVAL_SECONDS = 1.5
MAX_FEED_ROWS = 30

st.set_page_config(
    page_title="Transaction Risk Monitor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Session state
# -----------------------------
def reset_state() -> None:
    st.session_state.generator = TransactionGenerator(seed=None)
    st.session_state.feed = deque(maxlen=MAX_FEED_ROWS)
    st.session_state.total = 0
    st.session_state.fraud = 0
    st.session_state.latencies = deque(maxlen=500)
    st.session_state.risk_history = deque(maxlen=60)
    st.session_state.last = None
    st.session_state.pending_scenario = None


if "generator" not in st.session_state:
    reset_state()


# -----------------------------
# Helpers
# -----------------------------
def load_metrics() -> dict:
    if not METRICS_PATH.exists():
        return {}
    try:
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def check_model() -> tuple[bool, dict]:
    """Confirm the shared model artifact is loaded (in-process, no network call)."""
    try:
        bundle = load_model()
        return True, {"model_version": str(bundle.get("model_version", "unknown"))}
    except RuntimeError:
        return False, {}


def score_transaction(transaction: dict) -> dict:
    """Run inference in-process using the same pipeline the FastAPI service uses.

    This mirrors what `POST /predict` in src/app.py returns, but calls the
    shared src/scoring.py module directly rather than over HTTP. That keeps
    the live dashboard demo working as a single self-contained service —
    important for reliably hosting it for free, since it removes any
    dependency on a second always-running API process.
    """
    start = time.perf_counter()
    request = TransactionRequest(**transaction)
    probability, is_fraud, threshold, model_version = score(
        amount=request.amount,
        merchant_category=request.merchant_category,
        location_mismatch=request.location_mismatch,
        device_velocity=request.device_velocity,
    )
    latency_ms = (time.perf_counter() - start) * 1000
    return {
        "transaction_id": request.transaction_id,
        "is_fraud": is_fraud,
        "fraud_probability": round(probability, 6),
        "decision_threshold": round(threshold, 6),
        "risk_level": risk_level(probability, is_fraud),
        "risk_reasons": risk_reasons(request),
        "latency_ms": round(latency_ms, 3),
        "model_version": model_version,
    }


def process_transaction(scenario: str | None = None) -> None:
    transaction = st.session_state.generator.generate_transaction(scenario=scenario)
    prediction = score_transaction(transaction)

    st.session_state.total += 1
    if prediction["is_fraud"]:
        st.session_state.fraud += 1

    latency = float(prediction["latency_ms"])
    probability = float(prediction["fraud_probability"])
    st.session_state.latencies.append(latency)
    st.session_state.risk_history.append(
        {
            "Transaction": st.session_state.total,
            "Fraud probability": probability,
        }
    )
    st.session_state.last = {
        "transaction": transaction,
        "prediction": prediction,
    }

    st.session_state.feed.appendleft({
        "Time": transaction["timestamp"].split(".")[0].replace("T", " "),
        "Transaction ID": transaction["transaction_id"],
        "Account": transaction["account_id"],
        "Amount": f"${transaction['amount']:,.2f}",
        "Merchant": transaction["merchant_category"],
        "Location": "MISMATCH" if transaction["location_mismatch"] else "Normal",
        "Velocity": transaction["device_velocity"],
        "Risk": prediction["risk_level"],
        "Probability": f"{probability * 100:.2f}%",
        "Decision": "🚨 FRAUD" if prediction["is_fraud"] else "✓ SAFE",
        "Latency": f"{latency:.2f} ms",
    })


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Monitor Controls")
    light_mode = st.toggle("Light mode", value=False)
    monitoring_enabled = st.toggle("Live monitoring", value=True)
    st.caption(f"Simulation interval: {STREAM_INTERVAL_SECONDS}s")

    st.divider()
    st.subheader("Demo Scenario")
    scenario_labels = {
        "random": "Random realistic stream",
        "normal": "Normal transaction",
        "high_value": "High-value transaction",
        "location_mismatch": "Location mismatch",
        "high_velocity": "High device velocity",
        "combined": "Combined suspicious behavior",
    }
    selected_scenario = st.selectbox(
        "Scenario",
        options=list(scenario_labels),
        format_func=lambda x: scenario_labels[x],
        help="Use a controlled scenario to demonstrate specific model risk signals.",
    )
    if st.button("Generate scenario now", use_container_width=True):
        st.session_state.pending_scenario = (
            None if selected_scenario == "random" else selected_scenario
        )

    st.divider()
    st.subheader("System")
    model_loaded, model_info = check_model()
    if model_loaded:
        st.markdown('<span class="status-online">● MODEL LOADED</span>', unsafe_allow_html=True)
        st.caption(f"Model: {model_info.get('model_version', 'unknown')}")
    else:
        st.markdown('<span class="status-offline">● MODEL UNAVAILABLE</span>', unsafe_allow_html=True)
    st.caption(
        "Inference runs in-process using the same pipeline as the FastAPI "
        "service (`src/app.py`), which can also be run separately for "
        "programmatic API access."
    )

    if st.button("Reset session", use_container_width=True):
        reset_state()
        st.rerun()


# -----------------------------
# Style Override Declarations
# -----------------------------
light_mode_css = """
body, .stApp {background-color: #f7f8fa; color: #172033;}
[data-testid="stSidebar"] {background-color: #eef1f5; color: #172033;}
[data-testid="stHeader"] {background-color: #f7f8fa;}
[data-testid="stWidgetLabel"] p, .stCaption, [data-testid="stMarkdownContainer"] p,
.hero p {color: #334155;}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3, [data-testid="stSidebar"] label {color: #172033;}
[data-testid="stButton"] button {background-color: #1d4ed8; color: #ffffff; border-color: #1e40af;}
[data-testid="stButton"] button:hover {background-color: #1e40af; color: #ffffff;}
[data-baseweb="input"], [data-baseweb="select"] {background-color: #ffffff; color: #172033;}
[data-baseweb="input"] input, [data-baseweb="select"] * {color: #172033;}
[data-baseweb="select"] [aria-selected="true"] {color: #172033;}
[data-testid="stTab"] {color: #334155;}
[data-testid="stTab"][aria-selected="true"] {color: #1d4ed8;}
h1, h2, h3, h4, h5, h6 {color: #172033;}
.risk-low {background:#e4eaf2; color:#172033;}

/* ============================================================
   BULLETPROOF BLACK SIDEBAR TOGGLE OVERRIDE (ALWAYS VISIBLE)
   ============================================================ */

/* Force ALL framework sidebar control button states to stay solid black */
button[aria-label="Collapse sidebar"], 
button[aria-label="Expand sidebar"],
div[data-testid="stSidebarCollapseButton"] button,
div[data-testid="collapsedControl"] button,
header button {
    background-color: #000000 !important;
    border: none !important;
    opacity: 1 !important;
    visibility: visible !important;
    display: inline-flex !important;
}

/* Maintain solid black properties when your mouse hovers over them */
button[aria-label="Collapse sidebar"]:hover, 
button[aria-label="Expand sidebar"]:hover,
div[data-testid="stSidebarCollapseButton"] button:hover,
div[data-testid="collapsedControl"] button:hover,
header button:hover {
    background-color: #000000 !important;
    border: none !important;
}

/* Force internal chevron graphic arrows to render pure crisp white inside the black block */
button[aria-label="Collapse sidebar"] svg, 
button[aria-label="Expand sidebar"] svg,
div[data-testid="stSidebarCollapseButton"] button svg,
div[data-testid="collapsedControl"] button svg,
header button svg {
    fill: #ffffff !important;
    color: #ffffff !important;
}

/* Completely strip out transparency/fade-out restrictions on the closed tray header container */
div[data-testid="collapsedControl"], 
header {
    background-color: transparent !important;
    visibility: visible !important;
    opacity: 1 !important;
}
""" if light_mode else ""

st.markdown(
    f"""
    <style>
    {light_mode_css}
    .hero {{padding: .35rem 0 .8rem 0;}}
    .hero h1 {{font-size: 2.35rem; margin-bottom: .15rem;}}
    .hero p {{color: #9aa4b2; font-size: 1rem; margin-top: 0;}}
    .risk-high {{padding: 1rem; border-radius: 12px; background:#b91c1c; color:white;
                font-weight:800; text-align:center; font-size:1.2rem;}}
    .risk-medium {{padding: 1rem; border-radius: 12px; background:#7c5a00; color:white;
                  font-weight:750; text-align:center; font-size:1.1rem;}}
    .risk-low {{padding: 1rem; border-radius: 12px; background:#172033; color:#dbe4ef;
               font-weight:650; text-align:center; font-size:1.05rem;}}
    .status-online {{color:#22c55e; font-weight:700;}}
    .status-offline {{color:#ef4444; font-weight:700;}}
    </style>
    """,
    unsafe_allow_html=True
)


st.markdown(
    f"""
    <style>
    {light_mode_css}
    .hero {{padding: .35rem 0 .8rem 0;}}
    .hero h1 {{font-size: 2.35rem; margin-bottom: .15rem;}}
    .hero p {{color: #9aa4b2; font-size: 1rem; margin-top: 0;}}
    .risk-high {{padding: 1rem; border-radius: 12px; background:#b91c1c; color:white;
                font-weight:800; text-align:center; font-size:1.2rem;}}
    .risk-medium {{padding: 1rem; border-radius: 12px; background:#7c5a00; color:white;
                  font-weight:750; text-align:center; font-size:1.1rem;}}
    .risk-low {{padding: 1rem; border-radius: 12px; background:#172033; color:#dbe4ef;
               font-weight:650; text-align:center; font-size:1.05rem;}}
    .status-online {{color:#22c55e; font-weight:700;}}
    .status-offline {{color:#ef4444; font-weight:700;}}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    f"""
    <style>
    {light_mode_css}
    .hero {{padding: .35rem 0 .8rem 0;}}
    .hero h1 {{font-size: 2.35rem; margin-bottom: .15rem;}}
    .hero p {{color: #9aa4b2; font-size: 1rem; margin-top: 0;}}
    .risk-high {{padding: 1rem; border-radius: 12px; background:#b91c1c; color:white;
                font-weight:800; text-align:center; font-size:1.2rem;}}
    .risk-medium {{padding: 1rem; border-radius: 12px; background:#7c5a00; color:white;
                  font-weight:750; text-align:center; font-size:1.1rem;}}
    .risk-low {{padding: 1rem; border-radius: 12px; background:#172033; color:#dbe4ef;
               font-weight:650; text-align:center; font-size:1.05rem;}}
    .status-online {{color:#22c55e; font-weight:700;}}
    .status-offline {{color:#ef4444; font-weight:700;}}
    </style>
    """,
    unsafe_allow_html=True
)



st.markdown(
    f"""
    <style>
    .hero {{padding: .35rem 0 .8rem 0;}}
    .hero h1 {{font-size: 2.35rem; margin-bottom: .15rem;}}
    .hero p {{color: #9aa4b2; font-size: 1rem; margin-top: 0;}}
    .risk-high {{padding: 1rem; border-radius: 12px; background:#b91c1c; color:white;
                font-weight:800; text-align:center; font-size:1.2rem;}}
    .risk-medium {{padding: 1rem; border-radius: 12px; background:#7c5a00; color:white;
                  font-weight:750; text-align:center; font-size:1.1rem;}}
    .risk-low {{padding: 1rem; border-radius: 12px; background:#172033; color:#dbe4ef;
               font-weight:650; text-align:center; font-size:1.05rem;}}
    .status-online {{color:#22c55e; font-weight:700;}}
    .status-offline {{color:#ef4444; font-weight:700;}}
    {light_mode_css}
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Header
# -----------------------------
st.markdown(
    '<div class="hero"><h1>🛡️ Real-Time Transaction Risk Monitor</h1>'
    '<p>AI/ML-powered synthetic credit-card fraud detection with live API inference.</p></div>',
    unsafe_allow_html=True,
)

metrics_data = load_metrics()

tab_monitor, tab_model, tab_system = st.tabs(
    ["Live Monitor", "Model Performance", "System Info"]
)


@st.fragment(run_every=STREAM_INTERVAL_SECONDS)
def live_monitor() -> None:
    """Process one transaction on each scheduled run without blocking Streamlit."""
    if not monitoring_enabled:
        st.info("Live monitoring is paused. Use the sidebar toggle to resume.")
        return

    scenario = st.session_state.pop("pending_scenario", None)

    try:
        process_transaction(scenario=scenario)
    except RuntimeError as exc:
        st.error(str(exc))
        return
    except Exception as exc:
        st.error(f"Unexpected monitoring error: {exc}")
        return

    avg_latency = sum(st.session_state.latencies) / len(st.session_state.latencies)
    alert_rate = st.session_state.fraud / st.session_state.total * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions", f"{st.session_state.total:,}")
    c2.metric("Fraud Alerts", f"{st.session_state.fraud:,}")
    c3.metric("Observed Alert Rate", f"{alert_rate:.2f}%")
    c4.metric("Avg. Inference", f"{avg_latency:.2f} ms")

    st.divider()

    last = st.session_state.last
    prediction = last["prediction"]
    transaction = last["transaction"]

    if prediction["risk_level"] == "HIGH":
        icon = "🚨" if prediction["is_fraud"] else "⚠️"
        label = "HIGH-RISK TRANSACTION DETECTED" if prediction["is_fraud"] else "HIGH-RISK SIGNAL — REVIEW RECOMMENDED"
        st.markdown(
            f'<div class="risk-high">{icon} {label} · '
            f'{prediction["fraud_probability"] * 100:.2f}% probability</div>',
            unsafe_allow_html=True,
        )
    elif prediction["risk_level"] == "MEDIUM":
        st.markdown(
            f'<div class="risk-medium">⚠️ MEDIUM RISK · '
            f'{prediction["fraud_probability"] * 100:.2f}% probability</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="risk-low">✓ Transaction processed · low risk</div>',
            unsafe_allow_html=True,
        )

    d1, d2 = st.columns([1, 1])
    with d1:
        st.subheader("Latest Transaction")
        details = pd.DataFrame(
            [
                ["Transaction ID", transaction["transaction_id"]],
                ["Account", transaction["account_id"]],
                ["Amount", f"${transaction['amount']:,.2f}"],
                ["Merchant", transaction["merchant_category"]],
                ["Location mismatch", "YES" if transaction["location_mismatch"] else "NO"],
                ["Device velocity", transaction["device_velocity"]],
                ["Decision", "FRAUD" if prediction["is_fraud"] else "SAFE"],
                ["Risk level", prediction["risk_level"]],
                ["Fraud probability", f"{prediction['fraud_probability'] * 100:.2f}%"],
                ["Decision threshold", f"{prediction['decision_threshold'] * 100:.2f}%"],
                ["API latency", f"{prediction['latency_ms']:.2f} ms"],
            ],
            columns=["Field", "Value"],
        )
        st.dataframe(details, hide_index=True, use_container_width=True)

    with d2:
        st.subheader("Risk Signals")
        for reason in prediction["risk_reasons"]:
            st.write(f"• {reason}")
        st.caption(
            "Rule-based signals explain the synthetic risk patterns; they are not "
            "independent model features beyond the four fields used by XGBoost."
        )

    if st.session_state.risk_history:
        st.subheader("Recent Fraud Probability")
        chart_df = pd.DataFrame(list(st.session_state.risk_history)).set_index("Transaction")
        st.line_chart(chart_df, height=220)

    st.subheader("Live Transaction Feed")
    st.dataframe(
        pd.DataFrame(list(st.session_state.feed)),
        hide_index=True,
        use_container_width=True,
        height=480,
    )


with tab_monitor:
    live_monitor()


with tab_model:
    st.subheader("Model Performance")
    st.caption("Metrics below are from the bundled synthetic-data training run.")

    test_metrics = metrics_data.get("test", metrics_data)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Precision", f"{test_metrics.get('precision', 0) * 100:.2f}%")
    m2.metric("Recall", f"{test_metrics.get('recall', 0) * 100:.2f}%")
    m3.metric("F1", f"{test_metrics.get('f1', 0) * 100:.2f}%")
    m4.metric("PR-AUC", f"{test_metrics.get('pr_auc', 0) * 100:.2f}%")

    m5, m6, m7 = st.columns(3)
    m5.metric("ROC-AUC", f"{test_metrics.get('roc_auc', 0) * 100:.2f}%")
    m6.metric("Alert Rate", f"{test_metrics.get('alert_rate', 0) * 100:.2f}%")
    m7.metric("Decision Threshold", f"{test_metrics.get('threshold', 0) * 100:.2f}%")

    st.divider()
    left, right = st.columns(2)

    with left:
        st.subheader("Confusion Matrix")
        matrix = pd.DataFrame(
            [
                [test_metrics.get("true_negatives", 0), test_metrics.get("false_positives", 0)],
                [test_metrics.get("false_negatives", 0), test_metrics.get("true_positives", 0)],
            ],
            index=["Actual Legitimate", "Actual Fraud"],
            columns=["Predicted Legitimate", "Predicted Fraud"],
        )
        st.dataframe(matrix, use_container_width=True)

    with right:
        st.subheader("Top Feature Importance")
        importance = metrics_data.get("feature_importance", [])
        if importance:
            display_importance = []
            for item in importance:
                name = str(item["feature"])
                name = name.replace("numeric__", "")
                name = name.replace("categorical__merchant_category_", "merchant: ")
                display_importance.append({"feature": name, "importance": item["importance"]})
            imp_df = pd.DataFrame(display_importance).set_index("feature")
            st.bar_chart(imp_df, height=300)
        else:
            st.info("Feature-importance metadata is unavailable.")

    st.warning(
        "These metrics are based on synthetic labels generated from known rules. "
        "They demonstrate the engineering workflow, not real-world banking performance."
    )


with tab_system:
    st.subheader("System Information")
    info = {
        "Architecture": "Transaction Generator → XGBoost → Streamlit (in-process)",
        "Streaming": "Local Python simulation",
        "Inference": "In-process (shared pipeline also served via FastAPI /predict)",
        "Model": metrics_data.get("model", "XGBClassifier"),
        "Model version": "1.1.0",
        "Historical rows": metrics_data.get("rows", 0),
        "Synthetic fraud prevalence": f"{metrics_data.get('fraud_rate', 0) * 100:.2f}%",
        "Features": ", ".join(metrics_data.get("features", [])),
        "Threshold selection": metrics_data.get("threshold_selection", {}).get(
            "method", "Validation-set selection"
        ),
    }
    st.dataframe(
        pd.DataFrame(list(info.items()), columns=["Component", "Configuration"]),
        hide_index=True,
        use_container_width=True,
    )

    st.info(
        "Demo environment: all transactions are synthetic. No real customer, card, "
        "or banking data is used."
    )

    st.subheader("Operational Scope")
    st.write(
        "This local project intentionally avoids Kafka, Kubernetes, databases, and "
        "other distributed infrastructure. The Python simulator represents the event "
        "source while FastAPI provides a clean model-serving boundary that could later "
        "be connected to production infrastructure."
    )
