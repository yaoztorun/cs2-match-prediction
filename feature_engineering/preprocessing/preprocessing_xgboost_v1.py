"""
[PROJECT ADDITION - no lab code in this file]

Preprocessing for XGBoost (Phase 4C, Model 3). Turns the same 17 Phase-3
whitelist features used by Logistic Regression and Random Forest into a
numerical matrix for `xgboost.XGBClassifier`, with two model-specific
differences from the Logistic Regression pipeline:

1. NO standardization - tree splits are invariant to monotonic per-feature
   rescaling (same reasoning as Random Forest).
2. NaN is PRESERVED, not median-imputed. XGBoost natively learns a default
   split direction for missing values, so passing NaN through lets the model
   treat "no prior match" (cold-start `days_since_last_match_diff`) as its own
   signal instead of silently collapsing it onto the median. Verified on this
   dataset before adopting: exactly one whitelist column has any missingness
   (`days_since_last_match_diff` - 463/6,619 train rows, 72/1,419 validation
   rows), so this policy affects precisely that column.

   A useful side effect, also verified: `mirror_raw_rows` negates NaN -> NaN,
   so with NaN preserved the augmented training set's directional-diff means
   are EXACTLY 0.0. Under Logistic Regression's / Random Forest's median
   imputation, a mirrored NaN-pair instead shared one imputed value, leaving a
   small residual asymmetry (the documented looser-tolerance branch in
   `preprocessing_common.assert_augmented_symmetry`).

The mirroring/augmentation logic itself is algorithm-independent and shared
via feature_engineering/preprocessing/preprocessing_common.py - imported here, never re-implemented.

Same leakage-safety property as the other models' preprocessing: only the
training partition's own raw values are ever used in this file.
"""

import json

import numpy as np

from feature_engineering.preprocessing.preprocessing_common import (  # noqa: F401 - re-exported for convenience
    TARGET_COL, DIRECTIONAL_DIFF_FEATURES, SYMMETRIC_COUNT_FEATURES, CONTINUOUS_FEATURES,
    BINARY_FEATURES, BESTOF_REFERENCE, BESTOF_DUMMIES, TIER_REFERENCE, TIER_DUMMIES,
    mirror_raw_rows, build_augmented_training_raw, assert_augmented_symmetry, transformed_feature_names,
)

PREPROCESSING_VERSION = "1.0.0"
MISSING_VALUE_POLICY = "preserve_nan_native_xgboost"


def fit_preprocessing(augmented_train_df, model_features):
    """Fit preprocessing on the AUGMENTED (original+mirrored) TRAINING data
    only. For XGBoost V1 nothing is actually *applied* at transform time
    beyond the fixed categorical encoding (no scaling, no imputation) - but
    train medians are still computed and stored as clearly-labeled UNUSED
    reference metadata, so a future model/variant that does want imputation
    has train-only values available without having to refit anything.
    Returns a JSON-serializable dict."""
    medians_unused = {c: float(augmented_train_df[c].median()) for c in CONTINUOUS_FEATURES}

    return {
        "version": PREPROCESSING_VERSION,
        "original_model_feature_names": list(model_features),
        "transformed_feature_names": transformed_feature_names(),
        "missing_value_policy": MISSING_VALUE_POLICY,
        "scaling_applied": False,
        "imputation_applied": False,
        "train_medians_unused_reference": medians_unused,
        "categorical": {
            "bestOf": {"reference": BESTOF_REFERENCE, "dummies": BESTOF_DUMMIES},
            "tier": {"reference": TIER_REFERENCE, "dummies": TIER_DUMMIES},
        },
        "binary_features": list(BINARY_FEATURES),
        "continuous_unscaled_features": list(CONTINUOUS_FEATURES),
    }


def transform(df, params):
    """Apply a fitted preprocessing artifact to ANY dataframe containing the
    17 raw model_features (augmented-train, original-train, validation, test,
    or a single future matchup row). Continuous features keep their ORIGINAL
    scale and their NaN values; only the deterministic bestOf/tier
    reference-dummy encoding is applied. Returns (X, feature_names)."""
    out_cols = {}
    for c in CONTINUOUS_FEATURES:
        out_cols[c] = df[c].to_numpy(dtype=float)  # NaN preserved, no scaling, no imputation
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
