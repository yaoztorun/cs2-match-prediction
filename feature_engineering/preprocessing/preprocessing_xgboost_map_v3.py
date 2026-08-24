"""
[PROJECT ADDITION - Phase 6C]

XGBoost preprocessing for the MODERN SELECTED-MAP feature set. Same semantics
as feature_engineering/preprocessing/preprocessing_xgboost_map_v2.py - documented NaNs preserved for
XGBoost's native missing-value handling, no imputation, no scaling -
generalized to V3's config-driven column lists via the SAME generic
transform functions preprocessing_common_map_v2.py already provides.

The four NaN-capable columns (roster_map_mean_kast_diff,
roster_map_bottom_kast_diff, roster_map_mean_adr_diff,
roster_map_mean_kd_balance_diff) reach the booster as genuine missing values,
exactly as the Phase 6C cold-start contract intends. The categorical dummies
are never NaN - the unknown-category contract resolves them first.
"""

import json

import numpy as np

from feature_engineering.preprocessing.preprocessing_common_map_v2 import (
    MAP_DUMMIES, BESTOF_DUMMIES, TIER_DUMMIES,
    categorical_vocabulary, resolve_categoricals, transformed_feature_names,
)
from feature_engineering.preprocessing.preprocessing_random_forest_map_v3 import EXPECTED_TRANSFORMED_FEATURES

PREPROCESSING_VERSION = "1.0.0"
MISSING_VALUE_POLICY = "preserve_nan_native_xgboost"


def fit_preprocessing(augmented_train_df, roles):
    """Fit on AUGMENTED TRAINING data ONLY. Nothing is applied beyond the fixed
    categorical encoding - no scaling, no imputation. Train medians are stored
    as clearly-labelled UNUSED reference metadata only."""
    continuous = list(roles["directional"]) + list(roles["symmetric_continuous"])
    medians_unused = {c: (float(augmented_train_df[c].median())
                          if not augmented_train_df[c].isna().all() else None)
                      for c in continuous}
    return {
        "version": PREPROCESSING_VERSION,
        "model": "xgboost",
        "feature_config": "config/features/map_features_v3_modern_map.yaml",
        "original_model_feature_names": list(roles["model_features"]),
        "transformed_feature_names": transformed_feature_names(roles),
        "missing_value_policy": MISSING_VALUE_POLICY,
        "scaling_applied": False,
        "imputation_applied": False,
        "train_medians_unused_reference": medians_unused,
        "categorical": categorical_vocabulary(),
        "binary_features": list(roles["symmetric_binary"]),
        "continuous_unscaled_features": continuous,
    }


def transform(df, params, roles):
    """Apply a fitted artifact to ANY dataframe carrying roles["model_features"].
    Continuous features keep their original scale AND their NaN values; only the
    deterministic categorical encoding is applied. Returns (X, feature_names)."""
    continuous = list(roles["directional"]) + list(roles["symmetric_continuous"])
    map_col, bestof_col, tier_col = resolve_categoricals(df)

    out = {}
    for c in continuous:
        out[c] = df[c].to_numpy(dtype=float)   # NaN preserved, no scaling, no imputation
    for c in roles["symmetric_binary"]:
        out[c] = df[c].to_numpy(dtype=float)
    for m in MAP_DUMMIES:
        out[f"map_name_{m}"] = (map_col == m).astype(float).to_numpy()
    for d in BESTOF_DUMMIES:
        out[f"bestOf_BO{d}"] = (bestof_col == d).astype(float).to_numpy()
    for t in TIER_DUMMIES:
        out[f"tier_{t}"] = (tier_col == t).astype(float).to_numpy()

    names = params["transformed_feature_names"]
    assert names == transformed_feature_names(roles), \
        "saved transformed feature order disagrees with the current config-derived order"
    X = np.column_stack([out[n] for n in names])
    assert X.shape[1] == EXPECTED_TRANSFORMED_FEATURES, \
        f"expected {EXPECTED_TRANSFORMED_FEATURES} transformed features, got {X.shape[1]}"

    n_cat = len(MAP_DUMMIES) + len(BESTOF_DUMMIES) + len(TIER_DUMMIES)
    assert np.isfinite(X[:, -n_cat:]).all(), "categorical dummy columns must never be NaN"
    return X, names


def save_preprocessing(params, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)


def load_preprocessing(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
