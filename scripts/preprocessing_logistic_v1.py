"""
[PROJECT ADDITION - no lab code in this file]

Preprocessing for the scratch Logistic Regression model (Phase 4A). Turns the
17 Phase-3 whitelist features (config/series_features_v1.yaml) into a fully
numerical, standardized matrix, and implements the mirrored-training
augmentation used to counter the Team1 orientation bias.

CRITICAL ORDERING (corrected from an earlier draft of this phase's plan):
mirroring happens on RAW feature values, BEFORE preprocessing is fit -
not by negating already-standardized values. Preprocessing statistics
(median/mean/std) are fit on the AUGMENTED (original + mirrored) raw
training data. This matters because the standardization mean is computed
from an orientation-biased sample; fitting it on the original-only data and
then separately negating+re-standardizing a mirrored raw row would NOT give
exact negatives whenever that mean is non-zero:
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
import pandas as pd

PREPROCESSING_VERSION = "1.0.0"
TARGET_COL = "team1_series_win"

DIRECTIONAL_DIFF_FEATURES = [
    "elo_diff", "overall_win_rate_diff", "win_rate_last_5_diff", "win_rate_last_10_diff",
    "format_win_rate_diff", "avg_series_margin_last_5_diff", "avg_series_margin_last_10_diff",
    "matches_last_30_days_diff", "days_since_last_match_diff", "total_matches_before_diff",
]
SYMMETRIC_COUNT_FEATURES = ["history_matches_min", "history_matches_sum"]
CONTINUOUS_FEATURES = DIRECTIONAL_DIFF_FEATURES + SYMMETRIC_COUNT_FEATURES  # standardized
BINARY_FEATURES = ["both_teams_have_history", "both_teams_have_5_matches", "both_teams_have_10_matches"]

BESTOF_REFERENCE = 1
BESTOF_DUMMIES = [3, 5]
TIER_REFERENCE = "tier1"
TIER_DUMMIES = ["tier2", "tier3"]


def mirror_raw_rows(df):
    """Given a RAW (pre-preprocessing) dataframe with the 17 whitelist feature
    columns + target, return the mirrored version: every directional
    Team1-Team2 diff feature negated, symmetric/binary/context columns
    unchanged, target flipped (1-y). This is the single function used both
    for building the training-time augmented set and for the future-inference
    symmetry test - so there is exactly one mirroring implementation."""
    mirrored = df.copy()
    for col in DIRECTIONAL_DIFF_FEATURES:
        mirrored[col] = -mirrored[col]
    if TARGET_COL in mirrored.columns:
        mirrored[TARGET_COL] = 1 - mirrored[TARGET_COL]
    return mirrored


def build_augmented_training_raw(train_df):
    """TRAIN ONLY. original raw train rows + their raw mirrored counterparts."""
    mirrored = mirror_raw_rows(train_df)
    augmented = pd.concat([train_df, mirrored], ignore_index=True)
    return augmented


def transformed_feature_names():
    names = list(CONTINUOUS_FEATURES)
    names += list(BINARY_FEATURES)
    names += [f"bestOf_BO{d}" for d in BESTOF_DUMMIES]
    names += [f"tier_{t}" for t in TIER_DUMMIES]
    return names


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


def assert_augmented_symmetry(augmented_train_df):
    """The augmented (original+mirrored) raw training data should be (almost)
    exactly side-symmetric for every directional diff feature, so its mean
    should be ~0. The one expected, documented exception: a feature with
    real missingness (days_since_last_match_diff, ~6.4% of original rows -
    cold-start teams) has NaN negated to NaN by mirroring, so a
    mirrored-pair's two NaNs both get imputed to the SAME median value
    rather than becoming exact negatives of one another; this introduces a
    small, bounded, well-understood non-zero mean rather than exactly zero.
    Fully-populated features get a tight tolerance; the imputed one gets a
    looser, std-relative tolerance."""
    for c in DIRECTIONAL_DIFF_FEATURES:
        col = augmented_train_df[c]
        mean_before_impute = col.mean()  # pandas .mean() skips NaN
        std = col.std(ddof=1)
        if col.isna().any():
            assert abs(mean_before_impute) < 0.1 * std, (
                f"{c}: augmented raw mean {mean_before_impute} not small relative to std {std}")
        else:
            assert abs(mean_before_impute) < 1e-8, (
                f"{c}: augmented raw mean {mean_before_impute} should be ~0 (exact symmetry, no missingness)")
