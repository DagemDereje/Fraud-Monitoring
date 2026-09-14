# Model Card — Transaction Fraud Classifier

## Model

- Algorithm: XGBoost binary classifier
- Task: synthetic credit-card transaction fraud classification
- Features: amount, merchant category, location mismatch, device velocity
- Preprocessing: StandardScaler + OneHotEncoder inside a scikit-learn Pipeline
- Imbalance strategy: XGBoost `scale_pos_weight`
- Threshold selection: F1 optimization on a validation set with a 2% alert-rate cap
- Final evaluation: held-out test set

## Dataset

The dataset contains 10,000 fully synthetic transactions generated over roughly six months. Fraud prevalence is approximately 0.7%.

The labels are generated from synthetic business rules. In particular, high-value transactions and combined location/velocity anomalies are intentionally strong signals.

## Evaluation

The bundled metrics file contains both validation and final test results. The dashboard displays the final test metrics.

These metrics are for the synthetic dataset only and should not be interpreted as evidence of real-world banking performance.

## Why a separate validation set matters

The fraud decision threshold is selected on validation predictions rather than the final test set. The test set is therefore reserved for the final reported evaluation, reducing threshold-selection leakage.

## Limitations

The system does not model real card networks, merchant geolocation, device fingerprints, customer behavior histories, chargebacks, feedback delays, concept drift, or production traffic. Synthetic labels are intentionally easier to learn than real fraud labels.

## Responsible use

Do not use this model to make real financial decisions. A production fraud system requires extensive validation, governance, security controls, human review, monitoring, and regulatory/business controls.
