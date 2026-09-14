# Project Overview

## Problem

Banks and payment platforms need to score transactions quickly enough to identify suspicious activity while keeping alert volumes manageable.

## Design

This portfolio project keeps the architecture intentionally local and understandable:

`synthetic event → HTTP inference → risk decision → monitoring dashboard`

The Python generator represents the event source. FastAPI provides the model-serving boundary. Streamlit provides the analyst-facing monitor.

## Key engineering choices

| Concern | Choice | Rationale |
|---|---|---|
| Data | Synthetic transactions | Avoids sensitive banking data |
| ML | XGBoost | Strong tabular baseline and fast local inference |
| Imbalance | `scale_pos_weight` | Native XGBoost support for rare positives |
| Encoding | ColumnTransformer | Preprocessing is serialized with the model |
| Threshold | Validation-set operating-point selection | Avoids selecting the operational threshold on the final test set |
| Serving | FastAPI | Lightweight typed REST interface |
| UI | Streamlit | Simple local monitoring interface |
| Streaming | Python + Streamlit fragment | Real-time behavior without a blocking loop or distributed infrastructure |

## Live monitoring behavior

The dashboard supports a realistic random stream plus controlled demonstration scenarios:

- Normal transaction
- High-value transaction
- Location mismatch
- High device velocity
- Combined suspicious behavior

Controlled scenarios make the fraud workflow reproducible during a portfolio demonstration without changing the model or API contract.
