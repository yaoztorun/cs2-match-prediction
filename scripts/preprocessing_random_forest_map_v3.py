"""
[PROJECT ADDITION - Phase 6C]

Random Forest preprocessing for the MODERN SELECTED-MAP feature set. Same
semantics as scripts/preprocessing_random_forest_map_v2.py - median
imputation of continuous features, NO scaling, deterministic
map_name/bestOf/tier reference-dummy encoding (unchanged vocabulary from V2 -
the categorical context itself did not change in Phase 6C) - generalized to
V3's config-driven directional/symmetric-continuous/symmetric-binary column
lists (scripts/preprocessing_common_map_v3.py) via the SAME generic
transform functions preprocessing_common_map_v2.py already provides.

Leakage safety: medians are computed from the AUGMENTED FOLD-TRAIN (or full
augmented TRAIN) rows the caller hands in and from nothing else.
"""

import json

import numpy as np

from preprocessing_common_map_v2 import (
    MAP_DUMMIES, BESTOF_DUMMIES, TIER_DUMMIES,
    categorical_vocabulary, resolve_categoricals, transformed_feature_names,
)

PREPROCESSING_VERSION = "1.0.0"
EXPECTED_TRANSFORMED_FEATURES = 131   # 80 directional + 23 continuous symmetric + 14 binary symmetric
                                       # + 9 map dummies + 2 bestOf dummies + 3 tier dummies


def fit_preprocessing(augmented_train_df, roles):
    """Fit on AUGMENTED TRAINING data ONLY. Computes only the train medians
    used for imputation over directional + continuous-symmetric columns.

    If a numeric feature is entirely NaN inside this training set, this STOPS
    with a named error rather than silently inventing a population value."""
    continuous = list(roles["directional"]) + list(roles["symmetric_continuous"])

    all_nan = [c for c in continuous if augmented_train_df[c].isna().all()]
    if all_nan:
        raise RuntimeError(
            "Cannot fit RF preprocessing: these numeric features are entirely NaN inside this training "
            f"set, so no train median exists and none will be invented: {all_nan}")

    medians = {c: float(augmented_train_df[c].median()) for c in continuous}
    return {
        "version": PREPROCESSING_VERSION,
        "model": "random_forest",
        "feature_config": "config/map_features_v3_modern_map.yaml",
        "original_model_feature_names": list(roles["model_features"]),
        "transformed_feature_names": transformed_feature_names(roles),
        "train_medians": medians,
        "categorical": categorical_vocabulary(),
        "binary_features": list(roles["symmetric_binary"]),
        "continuous_unscaled_features": continuous,
        "missing_value_policy": "median_impute_from_augmented_train_only",
        "scaling_applied": False,
        "imputation_applied": True,
    }


def transform(df, params, roles):
    """Apply a fitted artifact to ANY dataframe carrying roles["model_features"].
    Continuous features are median-imputed but kept at their ORIGINAL scale.
    Returns (X, feature_names) with the frozen column order."""
    continuous = list(roles["directional"]) + list(roles["symmetric_continuous"])
    map_col, bestof_col, tier_col = resolve_categoricals(df)

    out = {}
    for c in continuous:
        out[c] = df[c].fillna(params["train_medians"][c]).to_numpy(dtype=float)
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
    assert np.isfinite(X).all(), "RF matrix contains NaN/inf after median imputation"
    return X, names


def save_preprocessing(params, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)


def load_preprocessing(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
