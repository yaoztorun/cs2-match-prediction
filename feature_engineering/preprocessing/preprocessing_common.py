"""
[PROJECT ADDITION - no lab code in this file]

Algorithm-independent preprocessing helpers shared by every model's
preprocessing module (Logistic Regression, Random Forest, and any future
model). Contains only the parts that do NOT depend on how a specific model
wants its features represented (standardized vs. raw-scale, etc.):

- which raw Phase-3 features are directional (Team1-Team2 diffs), symmetric
  (order-independent counts/confidence flags), or categorical context;
- the mirroring policy itself (mirror_raw_rows / build_augmented_training_raw);
- the augmented-training-set symmetry check (assert_augmented_symmetry);
- the transformed feature name/column order (transformed_feature_names) -
  identical across models because the column *layout* doesn't depend on
  whether a continuous column is later standardized or left raw.

Model-specific preprocessing (scripts/preprocessing_logistic_v1.py,
scripts/preprocessing_random_forest_v1.py) imports these rather than
re-implementing them, so "what mirroring means" has exactly one
implementation and one test suite (tests/test_preprocessing_logistic.py's
mirroring/symmetry tests apply equally to every consumer of this module).

Still leakage-safe: mirroring only ever operates on rows already isolated to
the training partition by the caller - no validation/test/future information
is used anywhere in this file.
"""

import pandas as pd

TARGET_COL = "team1_series_win"

DIRECTIONAL_DIFF_FEATURES = [
    "elo_diff", "overall_win_rate_diff", "win_rate_last_5_diff", "win_rate_last_10_diff",
    "format_win_rate_diff", "avg_series_margin_last_5_diff", "avg_series_margin_last_10_diff",
    "matches_last_30_days_diff", "days_since_last_match_diff", "total_matches_before_diff",
]
SYMMETRIC_COUNT_FEATURES = ["history_matches_min", "history_matches_sum"]
CONTINUOUS_FEATURES = DIRECTIONAL_DIFF_FEATURES + SYMMETRIC_COUNT_FEATURES
BINARY_FEATURES = ["both_teams_have_history", "both_teams_have_5_matches", "both_teams_have_10_matches"]

BESTOF_REFERENCE = 1
BESTOF_DUMMIES = [3, 5]
TIER_REFERENCE = "tier1"
TIER_DUMMIES = ["tier2", "tier3"]


def mirror_raw_rows(df):
    """Given a RAW (pre-preprocessing) dataframe with the 17 whitelist feature
    columns (+ target, if present), return the mirrored version: every
    directional Team1-Team2 diff feature negated, symmetric/binary/context
    columns unchanged, target flipped (1-y) if present. This is the single
    mirroring implementation used by every model's training augmentation and
    by every future-inference symmetry test/diagnostic."""
    mirrored = df.copy()
    for col in DIRECTIONAL_DIFF_FEATURES:
        mirrored[col] = -mirrored[col]
    if TARGET_COL in mirrored.columns:
        mirrored[TARGET_COL] = 1 - mirrored[TARGET_COL]
    return mirrored


def build_augmented_training_raw(train_df):
    """TRAIN ONLY. original raw train rows + their raw mirrored counterparts.
    The result is 2x the number of ROWS/OBSERVATIONS fed to a model, not 2x
    the number of underlying historical matches - every mirrored row is a
    synthetic re-labeling of an already-counted match, not a new match."""
    mirrored = mirror_raw_rows(train_df)
    augmented = pd.concat([train_df, mirrored], ignore_index=True)
    return augmented


def transformed_feature_names():
    names = list(CONTINUOUS_FEATURES)
    names += list(BINARY_FEATURES)
    names += [f"bestOf_BO{d}" for d in BESTOF_DUMMIES]
    names += [f"tier_{t}" for t in TIER_DUMMIES]
    return names


def assert_augmented_symmetry(augmented_train_df):
    """The augmented (original+mirrored) raw training data should be (almost)
    exactly side-symmetric for every directional diff feature, so its mean
    should be ~0. The one expected, documented exception: a feature with
    real missingness (days_since_last_match_diff - cold-start teams) has NaN
    negated to NaN by mirroring, so a mirrored-pair's two NaNs both get
    imputed to the SAME median value rather than becoming exact negatives of
    one another; this introduces a small, bounded, well-understood non-zero
    mean rather than exactly zero. Fully-populated features get a tight
    tolerance; a feature with missingness gets a looser, std-relative
    tolerance."""
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
