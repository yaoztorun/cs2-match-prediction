"""
Tests for scripts/preprocessing_logistic_v1.py, including the critical
future-inference symmetry test: a genuinely swapped future matchup must be
transformed consistently with the training-time mirrored-augmentation
scheme, using the actual fitted preprocessing artifact (not a re-derivation).
"""

import numpy as np
import pandas as pd
import pytest

from preprocessing_logistic_v1 import (
    DIRECTIONAL_DIFF_FEATURES, SYMMETRIC_COUNT_FEATURES, BINARY_FEATURES,
    mirror_raw_rows, build_augmented_training_raw, fit_preprocessing, transform,
    assert_augmented_symmetry, transformed_feature_names,
)

MODEL_FEATURES = DIRECTIONAL_DIFF_FEATURES + SYMMETRIC_COUNT_FEATURES + BINARY_FEATURES + ["bestOf", "tier"]


def _synthetic_train_df(n=200, seed=0, with_missing=True):
    rng = np.random.default_rng(seed)
    data = {c: rng.normal(scale=50, size=n) for c in DIRECTIONAL_DIFF_FEATURES}
    for c in SYMMETRIC_COUNT_FEATURES:
        data[c] = rng.integers(0, 100, size=n).astype(float)
    for c in BINARY_FEATURES:
        data[c] = rng.integers(0, 2, size=n).astype(float)
    data["bestOf"] = rng.choice([1, 3, 5], size=n)
    data["tier"] = rng.choice(["tier1", "tier2", "tier3"], size=n)
    data["team1_series_win"] = rng.integers(0, 2, size=n).astype(float)
    df = pd.DataFrame(data)
    if with_missing:
        missing_idx = rng.choice(n, size=max(1, n // 10), replace=False)
        df.loc[missing_idx, "days_since_last_match_diff"] = np.nan
    return df


# --- mirror_raw_rows basic correctness ---

def test_mirror_raw_rows_negates_directional_and_flips_target():
    df = _synthetic_train_df(n=10, with_missing=False)
    mirrored = mirror_raw_rows(df)
    for c in DIRECTIONAL_DIFF_FEATURES:
        np.testing.assert_allclose(mirrored[c].to_numpy(), -df[c].to_numpy())
    for c in SYMMETRIC_COUNT_FEATURES + BINARY_FEATURES:
        np.testing.assert_array_equal(mirrored[c].to_numpy(), df[c].to_numpy())
    pd.testing.assert_series_equal(mirrored["bestOf"], df["bestOf"])
    pd.testing.assert_series_equal(mirrored["tier"], df["tier"])
    np.testing.assert_array_equal(mirrored["team1_series_win"].to_numpy(), 1 - df["team1_series_win"].to_numpy())


# --- augmented-symmetry assertion is meaningful (not vacuous) ---

def test_assert_augmented_symmetry_passes_on_real_augmentation():
    df = _synthetic_train_df(n=300, with_missing=True)
    augmented = build_augmented_training_raw(df)
    assert len(augmented) == 2 * len(df)
    assert augmented["team1_series_win"].mean() == pytest.approx(0.5)
    assert_augmented_symmetry(augmented)  # must not raise


def test_assert_augmented_symmetry_detects_asymmetry():
    """Sanity check that the assertion actually detects a real asymmetric
    (non-mirrored) sample, rather than trivially always passing."""
    df = _synthetic_train_df(n=300, with_missing=False, seed=7)
    # deliberately shift one directional feature so the raw (unmirrored) data
    # is NOT side-symmetric
    df["elo_diff"] = df["elo_diff"] + 500
    with pytest.raises(AssertionError):
        assert_augmented_symmetry(df)  # NOT augmented - should fail the symmetry check


# --- future-inference symmetry test (the critical one) ---

def test_future_inference_symmetry_with_fitted_preprocessing():
    """Using the ACTUAL fitted preprocessing artifact: a synthetic raw
    matchup A-vs-B and its raw mirrored B-vs-A form must transform to
    exactly negated directional features and identical symmetric/context
    features. This proves training-time mirroring is compatible with the
    genuine future inference pipeline (same fit_preprocessing/transform
    functions used for both)."""
    train_df = _synthetic_train_df(n=500, seed=3, with_missing=True)
    augmented = build_augmented_training_raw(train_df)
    params = fit_preprocessing(augmented, MODEL_FEATURES)

    rng = np.random.default_rng(99)
    a_vs_b = pd.DataFrame([{
        **{c: float(rng.normal(scale=50)) for c in DIRECTIONAL_DIFF_FEATURES},
        **{c: float(rng.integers(0, 100)) for c in SYMMETRIC_COUNT_FEATURES},
        **{c: float(rng.integers(0, 2)) for c in BINARY_FEATURES},
        "bestOf": 3, "tier": "tier2",
    }])
    b_vs_a = mirror_raw_rows(a_vs_b)  # no target column present - mirror_raw_rows handles that

    X_ab, names = transform(a_vs_b, params)
    X_ba, _ = transform(b_vs_a, params)

    directional_idx = [names.index(c) for c in DIRECTIONAL_DIFF_FEATURES]
    other_idx = [i for i in range(len(names)) if i not in directional_idx]

    np.testing.assert_allclose(X_ab[0, directional_idx], -X_ba[0, directional_idx], atol=1e-10)
    np.testing.assert_allclose(X_ab[0, other_idx], X_ba[0, other_idx], atol=1e-10)


def test_future_inference_symmetry_with_missing_value():
    """Same test, but the synthetic future matchup itself has a missing
    days_since_last_match_diff (a genuine cold-start future team) - both
    orientations must still land on the SAME imputed value post-transform,
    since imputation uses a fixed train median, not a value that depends on
    which side of the matchup is "team1"."""
    train_df = _synthetic_train_df(n=500, seed=4, with_missing=True)
    augmented = build_augmented_training_raw(train_df)
    params = fit_preprocessing(augmented, MODEL_FEATURES)

    rng = np.random.default_rng(5)
    a_vs_b = pd.DataFrame([{
        **{c: float(rng.normal(scale=50)) for c in DIRECTIONAL_DIFF_FEATURES},
        **{c: float(rng.integers(0, 100)) for c in SYMMETRIC_COUNT_FEATURES},
        **{c: float(rng.integers(0, 2)) for c in BINARY_FEATURES},
        "bestOf": 1, "tier": "tier1",
    }])
    a_vs_b.loc[0, "days_since_last_match_diff"] = np.nan
    b_vs_a = mirror_raw_rows(a_vs_b)
    assert pd.isna(b_vs_a.loc[0, "days_since_last_match_diff"])  # NaN negated is still NaN

    X_ab, names = transform(a_vs_b, params)
    X_ba, _ = transform(b_vs_a, params)
    idx = names.index("days_since_last_match_diff")
    # both get imputed with the SAME train median before scaling -> identical transformed value
    assert X_ab[0, idx] == pytest.approx(X_ba[0, idx])


def test_transformed_feature_names_are_19_and_stable():
    names = transformed_feature_names()
    assert len(names) == 19
    assert names == sorted(set(names), key=names.index)  # no duplicates
