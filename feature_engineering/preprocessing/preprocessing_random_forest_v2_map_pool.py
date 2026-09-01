"""
[PROJECT ADDITION - Phase 5B.1]

RF preprocessing for the V2 map-pool series feature set. Same semantics as
feature_engineering/preprocessing/preprocessing_random_forest_v1.py - median imputation of continuous
features, NO scaling (tree splits are scale-invariant), deterministic
bestOf/tier reference-dummy encoding - generalized to V2's config-driven
directional/symmetric-continuous/symmetric-binary column lists
(feature_engineering/preprocessing/preprocessing_common_v2_map_pool.py) instead of V1's hardcoded
module-level constants.

Same leakage-safety property as every other preprocessing module in this
repo: only the training partition's own raw values are ever used here.
"""

import json

import numpy as np

from feature_engineering.preprocessing.preprocessing_common import BESTOF_REFERENCE, BESTOF_DUMMIES, TIER_REFERENCE, TIER_DUMMIES
from feature_engineering.preprocessing.preprocessing_common_v2_map_pool import transformed_feature_names

PREPROCESSING_VERSION = "1.0.0"


def fit_preprocessing(augmented_train_df, roles):
    """Fit on the AUGMENTED (original+mirrored) TRAINING data only. Computes
    ONLY train medians (for imputation) over roles["directional"] +
    roles["symmetric_continuous"] - no scaling. Returns a JSON-serializable dict."""
    continuous = list(roles["directional"]) + list(roles["symmetric_continuous"])
    medians = {c: float(augmented_train_df[c].median()) for c in continuous}

    return {
        "version": PREPROCESSING_VERSION,
        "original_model_feature_names": list(roles["model_features"]),
        "transformed_feature_names": transformed_feature_names(roles),
        "train_medians": medians,
        "categorical": {
            "bestOf": {"reference": BESTOF_REFERENCE, "dummies": BESTOF_DUMMIES},
            "tier": {"reference": TIER_REFERENCE, "dummies": TIER_DUMMIES},
        },
        "binary_features": list(roles["symmetric_binary"]),
        "continuous_unscaled_features": continuous,
        "scaling_applied": False,
    }


def transform(df, params, roles):
    """Apply a fitted preprocessing artifact to ANY dataframe containing
    roles["model_features"]. Continuous features are median-imputed but kept
    at their ORIGINAL scale. Returns (X, feature_names)."""
    continuous = list(roles["directional"]) + list(roles["symmetric_continuous"])
    out_cols = {}
    for c in continuous:
        out_cols[c] = df[c].fillna(params["train_medians"][c]).to_numpy(dtype=float)
    for c in roles["symmetric_binary"]:
        out_cols[c] = df[c].to_numpy(dtype=float)
    for d in BESTOF_DUMMIES:
        out_cols[f"bestOf_BO{d}"] = (df["bestOf"] == d).astype(float).to_numpy()
    for t in TIER_DUMMIES:
        out_cols[f"tier_{t}"] = (df["tier"] == t).astype(float).to_numpy()

    names = transformed_feature_names(roles)
    X = np.column_stack([out_cols[n] for n in names])
    return X, names


def save_preprocessing(params, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)


def load_preprocessing(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
