# ============================================================
# MODEL 2 — RANDOM FOREST
# Library implementation:
# sklearn.ensemble.RandomForestClassifier
#
# Unlike Model 1 Logistic Regression, this model is not
# required to be implemented from scratch.
# ============================================================
"""
Thin wrapper around sklearn.ensemble.RandomForestClassifier: the fixed V1
baseline configuration and save/load helpers. All training/evaluation
orchestration (data loading, mirroring, preprocessing, metrics, plots,
reports) lives in scripts/train_random_forest_v1.py, not here - mirrors the
split between scripts/models/logistic_regression_scratch.py (algorithm) and
scripts/train_logistic_regression_v1.py (orchestration) from Phase 4A.
"""

import joblib
from sklearn.ensemble import RandomForestClassifier

# Fixed V1 baseline configuration - NOT tuned. Structural hyperparameters are
# left at (or close to) sklearn defaults; n_estimators=300 is chosen only to
# make ensemble probabilities more stable than a very small forest;
# random_state=42 guarantees reproducibility; class_weight=None because the
# mirrored training set is exactly balanced (see preprocessing_common.py).
# Hyperparameter search is explicitly deferred to a later phase.
RF_CONFIG = dict(
    n_estimators=300,
    criterion="gini",
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    bootstrap=True,
    class_weight=None,
    random_state=42,
    n_jobs=-1,
)


def build_model():
    """Returns a fresh, untrained RandomForestClassifier with the fixed V1
    baseline configuration. No other configuration should be used in this
    phase."""
    return RandomForestClassifier(**RF_CONFIG)


def save_model(model, path):
    joblib.dump(model, path)


def load_model(path):
    return joblib.load(path)
