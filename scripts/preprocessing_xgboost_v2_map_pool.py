"""
[PROJECT ADDITION - Phase 5B.1]

XGBoost preprocessing for the V2 map-pool series feature set. Same semantics
as scripts/preprocessing_xgboost_v1.py - NaN preserved natively (no
imputation), no scaling, deterministic bestOf/tier reference-dummy encoding -
generalized to V2's config-driven column lists
(scripts/preprocessing_common_v2_map_pool.py).

Verified in Phase 5A: all 30 new map-pool features have zero missing values
across the whole artifact; only the inherited days_since_last_match_diff
carries real missingness, exactly as in V1 - so this policy affects the same
single column it always did.
"""

import json

import numpy as np

from preprocessing_common import BESTOF_REFERENCE, BESTOF_DUMMIES, TIER_REFERENCE, TIER_DUMMIES
from preprocessing_common_v2_map_pool import transformed_feature_names

PREPROCESSING_VERSION = "1.0.0"
MISSING_VALUE_POLICY = "preserve_nan_native_xgboost"


def fit_preprocessing(augmented_train_df, roles):
    """Fit on the AUGMENTED TRAINING data only. Nothing is actually applied
    beyond fixed categorical dummy encoding - no scaling, no imputation.
    Train medians are computed and stored as clearly-labeled UNUSED reference
    metadata only. Returns a JSON-serializable dict."""
    continuous = list(roles["directional"]) + list(roles["symmetric_continuous"])
    medians_unused = {c: float(augmented_train_df[c].median()) for c in continuous}

    return {
        "version": PREPROCESSING_VERSION,
        "original_model_feature_names": list(roles["model_features"]),
        "transformed_feature_names": transformed_feature_names(roles),
        "missing_value_policy": MISSING_VALUE_POLICY,
        "scaling_applied": False,
        "imputation_applied": False,
        "train_medians_unused_reference": medians_unused,
        "categorical": {
            "bestOf": {"reference": BESTOF_REFERENCE, "dummies": BESTOF_DUMMIES},
            "tier": {"reference": TIER_REFERENCE, "dummies": TIER_DUMMIES},
        },
        "binary_features": list(roles["symmetric_binary"]),
        "continuous_unscaled_features": continuous,
    }


def transform(df, params, roles):
    """Apply a fitted preprocessing artifact to ANY dataframe containing
    roles["model_features"]. Continuous features keep their ORIGINAL scale
    and their NaN values; only the deterministic bestOf/tier reference-dummy
    encoding is applied. Returns (X, feature_names)."""
    continuous = list(roles["directional"]) + list(roles["symmetric_continuous"])
    out_cols = {}
    for c in continuous:
        out_cols[c] = df[c].to_numpy(dtype=float)  # NaN preserved, no scaling, no imputation
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
