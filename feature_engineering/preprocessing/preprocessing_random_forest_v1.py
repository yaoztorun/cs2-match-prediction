"""
[PROJECT ADDITION - no lab code in this file]

Preprocessing for Random Forest (Phase 4B, Model 2). Turns the same 17
Phase-3 whitelist features used by Logistic Regression into a numerical
matrix for `sklearn.ensemble.RandomForestClassifier` - but, unlike
Logistic Regression, WITHOUT standardization: tree splits are invariant to
monotonic per-feature rescaling, so standardizing would add complexity
without changing what the forest can learn.

The mirroring/augmentation logic (mirror_raw_rows, build_augmented_training_raw,
assert_augmented_symmetry, transformed_feature_names, and the feature-group
constants) is shared with Logistic Regression via
feature_engineering/preprocessing/preprocessing_common.py - imported here, not re-implemented, so
"what mirroring means" has exactly one implementation for every model.

Same leakage-safety property as Logistic Regression's preprocessing: only
the training partition's own raw values are used anywhere in this file.
"""

import json

import numpy as np

from feature_engineering.preprocessing.preprocessing_common import (  # noqa: F401 - re-exported for convenience
    TARGET_COL, DIRECTIONAL_DIFF_FEATURES, SYMMETRIC_COUNT_FEATURES, CONTINUOUS_FEATURES,
    BINARY_FEATURES, BESTOF_REFERENCE, BESTOF_DUMMIES, TIER_REFERENCE, TIER_DUMMIES,
    mirror_raw_rows, build_augmented_training_raw, assert_augmented_symmetry, transformed_feature_names,
)

PREPROCESSING_VERSION = "1.0.0"


def fit_preprocessing(augmented_train_df, model_features):
    """Fit preprocessing on the AUGMENTED (original+mirrored) TRAINING data
    only. Unlike Logistic Regression's fit_preprocessing, this computes ONLY
    train medians (for imputation) - no means/stds, since Random Forest
    features are never standardized. Returns a JSON-serializable dict."""
    medians = {c: float(augmented_train_df[c].median()) for c in CONTINUOUS_FEATURES}

    return {
        "version": PREPROCESSING_VERSION,
        "original_model_feature_names": list(model_features),
        "transformed_feature_names": transformed_feature_names(),
        "train_medians": medians,
        "categorical": {
            "bestOf": {"reference": BESTOF_REFERENCE, "dummies": BESTOF_DUMMIES},
            "tier": {"reference": TIER_REFERENCE, "dummies": TIER_DUMMIES},
        },
        "binary_features": list(BINARY_FEATURES),
        "continuous_unscaled_features": list(CONTINUOUS_FEATURES),
        "scaling_applied": False,
    }


def transform(df, params):
    """Apply a fitted preprocessing artifact to ANY dataframe containing the
    17 raw model_features. Continuous features are median-imputed but kept
    at their ORIGINAL scale (no standardization). Returns (X, feature_names)."""
    out_cols = {}
    for c in CONTINUOUS_FEATURES:
        out_cols[c] = df[c].fillna(params["train_medians"][c]).to_numpy(dtype=float)
    for c in BINARY_FEATURES:
        out_cols[c] = df[c].to_numpy(dtype=float)
    for d in BESTOF_DUMMIES:
        out_cols[f"bestOf_BO{d}"] = (df["bestOf"] == d).astype(float).to_numpy()
    for t in TIER_DUMMIES:
        out_cols[f"tier_{t}"] = (df["tier"] == t).astype(float).to_numpy()

    names = transformed_feature_names()
    X = np.column_stack([out_cols[n] for n in names])
    return X, names


def save_preprocessing(params, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)


def load_preprocessing(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
