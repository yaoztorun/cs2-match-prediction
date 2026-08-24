"""
Tests for feature_engineering/preprocessing/preprocessing_random_forest_v1.py. Covers spec letters
D, E, F, G, J from Phase 4B section 17.
"""

import numpy as np
import pandas as pd
import pytest

from feature_engineering.preprocessing.preprocessing_common import (
    DIRECTIONAL_DIFF_FEATURES, SYMMETRIC_COUNT_FEATURES, BINARY_FEATURES, mirror_raw_rows,
)
from feature_engineering.preprocessing.preprocessing_random_forest_v1 import (
    build_augmented_training_raw, fit_preprocessing, transform, transformed_feature_names,
)

MODEL_FEATURES = DIRECTIONAL_DIFF_FEATURES + SYMMETRIC_COUNT_FEATURES + BINARY_FEATURES + ["bestOf", "tier"]


def _synthetic_df(n=200, seed=0, with_missing=True):
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
        idx = rng.choice(n, size=max(1, n // 10), replace=False)
        df.loc[idx, "days_since_last_match_diff"] = np.nan
    return df


# --- D: validation/test are unmirrored ---

def test_transform_does_not_double_rows():
    val_like = _synthetic_df(n=50, with_missing=False)
    train_like = _synthetic_df(n=200, seed=1)
    augmented = build_augmented_training_raw(train_like)
    params = fit_preprocessing(augmented, MODEL_FEATURES)

    X_val, _ = transform(val_like, params)
    assert X_val.shape[0] == 50  # NOT doubled - transform() never mirrors


# --- E: no validation/test statistics leak into preprocessing ---

def test_fit_preprocessing_unaffected_by_a_disjoint_validation_set():
    train_like = _synthetic_df(n=300, seed=2)
    augmented = build_augmented_training_raw(train_like)
    params_a = fit_preprocessing(augmented, MODEL_FEATURES)

    # a wildly different "validation" set must never be passed into fit_preprocessing;
    # confirm the medians only ever reflect the augmented TRAIN data by recomputing
    # them directly and comparing.
    recomputed_medians = {c: float(augmented[c].median()) for c in DIRECTIONAL_DIFF_FEATURES + SYMMETRIC_COUNT_FEATURES}
    for c, v in recomputed_medians.items():
        assert params_a["train_medians"][c] == pytest.approx(v)


# --- F: no scaling is applied ---

def test_no_scaling_continuous_features_match_raw_imputed_values():
    train_like = _synthetic_df(n=300, seed=3)
    augmented = build_augmented_training_raw(train_like)
    params = fit_preprocessing(augmented, MODEL_FEATURES)

    X, names = transform(augmented, params)
    for c in DIRECTIONAL_DIFF_FEATURES + SYMMETRIC_COUNT_FEATURES:
        idx = names.index(c)
        expected = augmented[c].fillna(params["train_medians"][c]).to_numpy(dtype=float)
        np.testing.assert_array_equal(X[:, idx], expected)  # exact equality - no mean/std transform applied
    assert params.get("scaling_applied") is False


# --- G: categorical mapping is deterministic ---

def test_categorical_mapping_is_deterministic_across_fits():
    train_a = _synthetic_df(n=200, seed=4)
    train_b = _synthetic_df(n=350, seed=5)  # different data entirely
    params_a = fit_preprocessing(build_augmented_training_raw(train_a), MODEL_FEATURES)
    params_b = fit_preprocessing(build_augmented_training_raw(train_b), MODEL_FEATURES)

    assert params_a["categorical"] == params_b["categorical"]  # reference/dummy scheme is fixed, not data-derived
    assert transformed_feature_names() == params_a["transformed_feature_names"] == params_b["transformed_feature_names"]

    rng = np.random.default_rng(9)
    row = pd.DataFrame([{
        **{c: float(rng.normal(scale=50)) for c in DIRECTIONAL_DIFF_FEATURES},
        **{c: float(rng.integers(0, 100)) for c in SYMMETRIC_COUNT_FEATURES},
        **{c: float(rng.integers(0, 2)) for c in BINARY_FEATURES},
        "bestOf": 3, "tier": "tier2",
    }])
    for params in (params_a, params_b):
        X, names = transform(row, params)
        assert X[0, names.index("bestOf_BO3")] == 1.0
        assert X[0, names.index("bestOf_BO5")] == 0.0
        assert X[0, names.index("tier_tier2")] == 1.0
        assert X[0, names.index("tier_tier3")] == 0.0


# --- J: future reversed-match preprocessing works correctly ---

def test_future_inference_symmetry_is_exact_without_scaling():
    """Unlike Logistic Regression (where exact mirror-symmetry after
    standardization depends on the augmented mean being ~0), Random Forest's
    UNSCALED transform makes x_mirror == -x_raw hold exactly for any raw
    values, regardless of any fitted statistic - there's no mean subtraction
    to break the symmetry. Verified here with the real fitted artifact."""
    train_like = _synthetic_df(n=300, seed=6)
    augmented = build_augmented_training_raw(train_like)
    params = fit_preprocessing(augmented, MODEL_FEATURES)

    rng = np.random.default_rng(77)
    a_vs_b = pd.DataFrame([{
        **{c: float(rng.normal(scale=50)) for c in DIRECTIONAL_DIFF_FEATURES},
        **{c: float(rng.integers(0, 100)) for c in SYMMETRIC_COUNT_FEATURES},
        **{c: float(rng.integers(0, 2)) for c in BINARY_FEATURES},
        "bestOf": 3, "tier": "tier2",
    }])
    b_vs_a = mirror_raw_rows(a_vs_b)

    X_ab, names = transform(a_vs_b, params)
    X_ba, _ = transform(b_vs_a, params)

    directional_idx = [names.index(c) for c in DIRECTIONAL_DIFF_FEATURES]
    other_idx = [i for i in range(len(names)) if i not in directional_idx]

    np.testing.assert_allclose(X_ab[0, directional_idx], -X_ba[0, directional_idx], atol=1e-12)
    np.testing.assert_allclose(X_ab[0, other_idx], X_ba[0, other_idx], atol=1e-12)
