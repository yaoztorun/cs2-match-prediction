"""
[PROJECT ADDITION - no lab code in this file]

Preprocessing for the scratch Logistic Regression model (Phase 4A). Turns the
17 Phase-3 whitelist features (config/series_features_v1.yaml) into a fully
numerical, standardized matrix.

The mirroring/augmentation logic itself (mirror_raw_rows,
build_augmented_training_raw, assert_augmented_symmetry,
transformed_feature_names, and the feature-group constants) is
algorithm-independent and lives in scripts/preprocessing_common.py, shared
with scripts/preprocessing_random_forest_v1.py - re-exported here so
existing imports (`from preprocessing_logistic_v1 import mirror_raw_rows`
etc., used by scripts/train_logistic_regression_v1.py,
scripts/validate_phase4a.py, and tests/test_preprocessing_logistic.py) keep
working unchanged. Only what's specific to Logistic Regression - fitting and
applying standardization - is implemented in this file.

CRITICAL ORDERING (corrected during Phase 4A planning): mirroring happens on
RAW feature values, BEFORE preprocessing is fit - not by negating
already-standardized values. Preprocessing statistics (median/mean/std) are
fit on the AUGMENTED (original + mirrored) raw training data. This matters
because the standardization mean is computed from an orientation-biased
sample; fitting it on the original-only data and then separately
negating+re-standardizing a mirrored raw row would NOT give exact negatives
whenever that mean is non-zero:
    (-x - mean) / std  !=  -((x - mean) / std)   whenever mean != 0
Fitting on the symmetric augmented set instead drives the mean of every
directional diff feature to (approximately) exactly zero by construction,
which is what makes standardized diff features and their raw negation
coincide - and is exactly what a genuinely reversed FUTURE matchup needs to
be treated consistently by this same fitted artifact (see
tests/test_preprocessing_logistic.py's future-inference symmetry test).

Still leakage-safe: the mirrored rows are synthesized purely from the
training partition's own raw feature values - no validation/test/future
information is used anywhere in this file.
"""

import json

import numpy as np

from preprocessing_common import (  # noqa: F401 - re-exported for backward compatibility
    TARGET_COL, DIRECTIONAL_DIFF_FEATURES, SYMMETRIC_COUNT_FEATURES, CONTINUOUS_FEATURES,
    BINARY_FEATURES, BESTOF_REFERENCE, BESTOF_DUMMIES, TIER_REFERENCE, TIER_DUMMIES,
    mirror_raw_rows, build_augmented_training_raw, assert_augmented_symmetry, transformed_feature_names,
)

PREPROCESSING_VERSION = "1.0.0"


def fit_preprocessing(augmented_train_df, model_features):
    """Fit ALL preprocessing parameters on the AUGMENTED (original+mirrored)
    TRAINING data only. Returns a JSON-serializable dict."""
    medians = {c: float(augmented_train_df[c].median()) for c in CONTINUOUS_FEATURES}

    imputed = augmented_train_df.copy()
    for c in CONTINUOUS_FEATURES:
        imputed[c] = imputed[c].fillna(medians[c])

    means = {c: float(imputed[c].mean()) for c in CONTINUOUS_FEATURES}
    stds = {c: float(imputed[c].std(ddof=1)) for c in CONTINUOUS_FEATURES}  # sample std (pandas default)

    return {
        "version": PREPROCESSING_VERSION,
        "original_model_feature_names": list(model_features),
        "transformed_feature_names": transformed_feature_names(),
        "train_medians": medians,
        "train_means": means,
        "train_stds": stds,
        "categorical": {
            "bestOf": {"reference": BESTOF_REFERENCE, "dummies": BESTOF_DUMMIES},
            "tier": {"reference": TIER_REFERENCE, "dummies": TIER_DUMMIES},
        },
        "binary_features": list(BINARY_FEATURES),
        "continuous_standardized_features": list(CONTINUOUS_FEATURES),
    }


def transform(df, params):
    """Apply a fitted preprocessing artifact to ANY dataframe containing the
    17 raw model_features (augmented-train, original-train, validation,
    test, or a single future matchup row). Returns (X, feature_names)."""
    out_cols = {}
    for c in CONTINUOUS_FEATURES:
        vals = df[c].fillna(params["train_medians"][c])
        out_cols[c] = ((vals - params["train_means"][c]) / params["train_stds"][c]).to_numpy(dtype=float)
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
