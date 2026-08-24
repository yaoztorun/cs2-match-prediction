# ============================================================
# MODEL 3 — XGBOOST CLASSIFIER
# Library implementation:
# xgboost.XGBClassifier
#
# Model 1 Logistic Regression was implemented from scratch
# because it was adapted from the course lab.
# XGBoost is intentionally implemented using the XGBoost library.
# ============================================================
"""
Thin wrapper around xgboost.XGBClassifier: the fixed V1 baseline
configuration and native save/load helpers. All training/evaluation
orchestration (data loading, mirroring, preprocessing, metrics, plots,
reports) lives in training/xgboost/train_xgboost_v1.py, not here - mirroring the
split used for Models 1 and 2.

Verified against the installed xgboost 3.4.0: `tree_method="hist"`,
constructor-level `eval_metric`, native NaN handling, and
`XGBClassifier.save_model`/`load_model` all work as-is. **No compatibility
adjustment to the specified configuration was required.**
"""

from xgboost import XGBClassifier

# Fixed V1 baseline configuration - NOT tuned, NOT optimized.
# Deliberately moderate: shallower trees than Random Forest V1, shrinkage via
# learning_rate, row subsampling (subsample) and feature subsampling
# (colsample_bytree), and default-style L2 regularization (reg_lambda=1.0).
# This must NOT be described as an optimized configuration - it is a single
# fixed baseline. Hyperparameter search is explicitly deferred to a later
# phase and, when it happens, must use chronological CV inside TRAIN only
# (as Random Forest V2 did), never the main validation set.
XGB_CONFIG = dict(
    objective="binary:logistic",
    eval_metric="logloss",
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.0,
    reg_alpha=0.0,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
)


def build_model():
    """Returns a fresh, untrained XGBClassifier with the fixed V1 baseline
    configuration. No other configuration is used in this phase."""
    return XGBClassifier(**XGB_CONFIG)


def save_model(model, path):
    """Native XGBoost JSON serialization through the CLASSIFIER interface, so
    future inference keeps using the same XGBClassifier/predict_proba API
    (rather than dropping down to a bare Booster)."""
    model.save_model(str(path))


def load_model(path):
    """Reload into an XGBClassifier (not a bare Booster) so `.predict_proba`
    remains available for future inference."""
    model = XGBClassifier()
    model.load_model(str(path))
    return model
