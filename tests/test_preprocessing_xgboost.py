"""
Tests for scripts/preprocessing_xgboost_v1.py (Phase 4C).

Key differences from the LR/RF preprocessing that these tests pin down:
no standardization, and NaN is PRESERVED (never median-imputed) so XGBoost
can apply its native missing-value handling.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from preprocessing_common import (
    DIRECTIONAL_DIFF_FEATURES, SYMMETRIC_COUNT_FEATURES, BINARY_FEATURES,
    mirror_raw_rows, transformed_feature_names,
)
from preprocessing_xgboost_v1 import (
    build_augmented_training_raw, fit_preprocessing, transform, MISSING_VALUE_POLICY,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "series_features_v1.yaml"

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


# --- feature layout / whitelist ---

def test_transformed_feature_layout_is_19_and_matches_shared_contract():
    train = _synthetic_df(n=100, seed=1)
    params = fit_preprocessing(build_augmented_training_raw(train), MODEL_FEATURES)
    names = params["transformed_feature_names"]
    assert len(names) == 19
    assert names == transformed_feature_names()  # identical layout/order to LR and RF


def test_feature_whitelist_matches_yaml_config():
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    train = _synthetic_df(n=50, seed=2)
    params = fit_preprocessing(build_augmented_training_raw(train), cfg["model_features"])
    assert params["original_model_feature_names"] == cfg["model_features"]
    assert len(cfg["model_features"]) == 17


# --- no scaling ---

def test_no_scaling_continuous_features_match_raw_values_exactly():
    train = _synthetic_df(n=300, seed=3, with_missing=False)
    augmented = build_augmented_training_raw(train)
    params = fit_preprocessing(augmented, MODEL_FEATURES)
    X, names = transform(augmented, params)

    for c in DIRECTIONAL_DIFF_FEATURES + SYMMETRIC_COUNT_FEATURES:
        idx = names.index(c)
        np.testing.assert_array_equal(X[:, idx], augmented[c].to_numpy(dtype=float))
    assert params["scaling_applied"] is False


# --- NaN preserved, NOT imputed ---

def test_nan_is_preserved_not_imputed():
    train = _synthetic_df(n=300, seed=4, with_missing=True)
    augmented = build_augmented_training_raw(train)
    params = fit_preprocessing(augmented, MODEL_FEATURES)
    X, names = transform(augmented, params)

    idx = names.index("days_since_last_match_diff")
    n_nan_in = int(augmented["days_since_last_match_diff"].isna().sum())
    n_nan_out = int(np.isnan(X[:, idx]).sum())
    assert n_nan_in > 0, "fixture should contain NaN to make this test meaningful"
    assert n_nan_out == n_nan_in  # every NaN survived the transform
    assert params["imputation_applied"] is False
    assert params["missing_value_policy"] == MISSING_VALUE_POLICY


def test_train_medians_are_stored_but_explicitly_unused():
    train = _synthetic_df(n=200, seed=5)
    params = fit_preprocessing(build_augmented_training_raw(train), MODEL_FEATURES)
    # stored as clearly-labeled reference metadata for a future variant...
    assert "train_medians_unused_reference" in params
    assert len(params["train_medians_unused_reference"]) == len(DIRECTIONAL_DIFF_FEATURES) + len(SYMMETRIC_COUNT_FEATURES)
    # ...but there is no key implying they were applied
    assert "train_medians" not in params
    assert params["imputation_applied"] is False


# --- deterministic categorical encoding ---

def test_categorical_encoding_is_deterministic_across_different_fits():
    params_a = fit_preprocessing(build_augmented_training_raw(_synthetic_df(n=200, seed=6)), MODEL_FEATURES)
    params_b = fit_preprocessing(build_augmented_training_raw(_synthetic_df(n=350, seed=7)), MODEL_FEATURES)
    assert params_a["categorical"] == params_b["categorical"]

    rng = np.random.default_rng(11)
    row = pd.DataFrame([{
        **{c: float(rng.normal(scale=50)) for c in DIRECTIONAL_DIFF_FEATURES},
        **{c: float(rng.integers(0, 100)) for c in SYMMETRIC_COUNT_FEATURES},
        **{c: float(rng.integers(0, 2)) for c in BINARY_FEATURES},
        "bestOf": 5, "tier": "tier3",
    }])
    for params in (params_a, params_b):
        X, names = transform(row, params)
        assert X[0, names.index("bestOf_BO5")] == 1.0
        assert X[0, names.index("bestOf_BO3")] == 0.0
        assert X[0, names.index("tier_tier3")] == 1.0
        assert X[0, names.index("tier_tier2")] == 0.0


# --- reversed-match (future inference) preprocessing correctness ---

def test_future_reversed_match_transform_is_exactly_symmetric():
    train = _synthetic_df(n=400, seed=8)
    params = fit_preprocessing(build_augmented_training_raw(train), MODEL_FEATURES)

    rng = np.random.default_rng(99)
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


def test_reversed_match_with_nan_stays_nan_on_both_sides():
    """A cold-start future matchup has NaN days_since_last_match_diff; mirroring
    negates NaN -> NaN, and (unlike median imputation) the NaN-preserving
    transform keeps it NaN in BOTH orientations."""
    train = _synthetic_df(n=300, seed=9)
    params = fit_preprocessing(build_augmented_training_raw(train), MODEL_FEATURES)

    rng = np.random.default_rng(5)
    a_vs_b = pd.DataFrame([{
        **{c: float(rng.normal(scale=50)) for c in DIRECTIONAL_DIFF_FEATURES},
        **{c: float(rng.integers(0, 100)) for c in SYMMETRIC_COUNT_FEATURES},
        **{c: float(rng.integers(0, 2)) for c in BINARY_FEATURES},
        "bestOf": 1, "tier": "tier1",
    }])
    a_vs_b.loc[0, "days_since_last_match_diff"] = np.nan
    b_vs_a = mirror_raw_rows(a_vs_b)

    X_ab, names = transform(a_vs_b, params)
    X_ba, _ = transform(b_vs_a, params)
    idx = names.index("days_since_last_match_diff")
    assert np.isnan(X_ab[0, idx])
    assert np.isnan(X_ba[0, idx])
