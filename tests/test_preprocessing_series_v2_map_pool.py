"""
Tests for scripts/preprocessing_common_v2_map_pool.py,
scripts/preprocessing_random_forest_v2_map_pool.py and
scripts/preprocessing_xgboost_v2_map_pool.py - the config-driven V2 map-pool
mirroring/preprocessing layer written for Phase 5B.1. Synthetic fixtures only,
same style as tests/test_preprocessing_random_forest.py.
"""

import numpy as np
import pandas as pd
import pytest

from preprocessing_common_v2_map_pool import (
    load_v2_roles, mirror_raw_rows, build_augmented_training_raw,
    assert_augmented_symmetry, transformed_feature_names, V2_BINARY_SYMMETRIC_FEATURES,
)
from preprocessing_random_forest_v2_map_pool import (
    fit_preprocessing as rf_fit_preprocessing, transform as rf_transform,
)
from preprocessing_xgboost_v2_map_pool import (
    fit_preprocessing as xgb_fit_preprocessing, transform as xgb_transform,
)

from _common import ROOT

CONFIG_V2_PATH = ROOT / "config" / "series_features_v2_map_pool.yaml"


@pytest.fixture(scope="module")
def roles():
    return load_v2_roles(CONFIG_V2_PATH)


def _synthetic_df(roles, n=200, seed=0, with_missing=True):
    rng = np.random.default_rng(seed)
    data = {c: rng.normal(scale=50, size=n) for c in roles["directional"]}
    for c in roles["symmetric_continuous"]:
        data[c] = rng.integers(0, 100, size=n).astype(float)
    for c in roles["symmetric_binary"]:
        data[c] = rng.integers(0, 2, size=n).astype(float)
    data["bestOf"] = rng.choice([1, 3, 5], size=n)
    data["tier"] = rng.choice(["tier1", "tier2", "tier3"], size=n)
    data["team1_series_win"] = rng.integers(0, 2, size=n).astype(float)
    df = pd.DataFrame(data)
    if with_missing and "days_since_last_match_diff" in df.columns:
        idx = rng.choice(n, size=max(1, n // 10), replace=False)
        df.loc[idx, "days_since_last_match_diff"] = np.nan
    return df


# --- roles are genuinely config-driven ---

def test_load_v2_roles_matches_yaml_lists_exactly(roles):
    import yaml
    cfg = yaml.safe_load(CONFIG_V2_PATH.read_text(encoding="utf-8"))
    assert roles["directional"] == cfg["directional_features"]
    assert roles["symmetric"] == cfg["symmetric_features"]
    assert roles["categorical"] == cfg["categorical_context"]
    assert roles["target"] == cfg["target"]
    assert set(roles["symmetric_continuous"]) | set(roles["symmetric_binary"]) == set(roles["symmetric"])
    assert set(roles["symmetric_continuous"]) & set(roles["symmetric_binary"]) == set()


def test_binary_symmetric_features_are_all_declared_in_the_config(roles):
    assert set(V2_BINARY_SYMMETRIC_FEATURES) <= set(roles["symmetric"])


# --- mirror negates ONLY configured directional_features ---

def test_mirror_negates_only_directional_leaves_everything_else(roles):
    df = _synthetic_df(roles, n=50, with_missing=False)
    mirrored = mirror_raw_rows(df, roles)

    for c in roles["directional"]:
        np.testing.assert_allclose(mirrored[c].to_numpy(), -df[c].to_numpy())
    for c in roles["symmetric"]:
        pd.testing.assert_series_equal(mirrored[c], df[c])
    for c in ["bestOf", "tier"]:
        pd.testing.assert_series_equal(mirrored[c], df[c])
    assert (mirrored["team1_series_win"] == 1 - df["team1_series_win"]).all()


def test_build_augmented_training_raw_doubles_rows_only(roles):
    df = _synthetic_df(roles, n=80, seed=1)
    augmented = build_augmented_training_raw(df, roles)
    assert len(augmented) == 2 * len(df)
    assert abs(float(augmented["team1_series_win"].mean()) - 0.5) < 1e-9


def test_assert_augmented_symmetry_passes_on_fully_populated_features(roles):
    df = _synthetic_df(roles, n=300, seed=2, with_missing=False)
    augmented = build_augmented_training_raw(df, roles)
    assert_augmented_symmetry(augmented, roles)  # must not raise


# --- per-fold independence: fitting on two disjoint slices never leaks ---

def test_fit_preprocessing_reflects_only_its_own_augmented_slice_rf(roles):
    train_like = _synthetic_df(roles, n=300, seed=3)
    augmented = build_augmented_training_raw(train_like, roles)
    params = rf_fit_preprocessing(augmented, roles)

    continuous = list(roles["directional"]) + list(roles["symmetric_continuous"])
    recomputed = {c: float(augmented[c].median()) for c in continuous}
    for c, v in recomputed.items():
        assert params["train_medians"][c] == pytest.approx(v)


def test_two_different_fold_slices_each_recompute_independently_rf(roles):
    """Different chronological folds may legitimately produce identical
    medians - the correctness property is that EACH fold's fit reflects ONLY
    that fold's own data, not that the two fits must differ numerically."""
    fold_a = _synthetic_df(roles, n=300, seed=10)
    fold_b = _synthetic_df(roles, n=450, seed=11)
    params_a = rf_fit_preprocessing(build_augmented_training_raw(fold_a, roles), roles)
    params_b = rf_fit_preprocessing(build_augmented_training_raw(fold_b, roles), roles)

    aug_a = build_augmented_training_raw(fold_a, roles)
    aug_b = build_augmented_training_raw(fold_b, roles)
    continuous = list(roles["directional"]) + list(roles["symmetric_continuous"])
    for c in continuous:
        assert params_a["train_medians"][c] == pytest.approx(float(aug_a[c].median()))
        assert params_b["train_medians"][c] == pytest.approx(float(aug_b[c].median()))


# --- RF: median imputation only, no scaling ---

def test_rf_no_scaling_continuous_features_match_raw_imputed_values(roles):
    train_like = _synthetic_df(roles, n=300, seed=4)
    augmented = build_augmented_training_raw(train_like, roles)
    params = rf_fit_preprocessing(augmented, roles)

    X, names = rf_transform(augmented, params, roles)
    continuous = list(roles["directional"]) + list(roles["symmetric_continuous"])
    for c in continuous:
        idx = names.index(c)
        expected = augmented[c].fillna(params["train_medians"][c]).to_numpy(dtype=float)
        np.testing.assert_array_equal(X[:, idx], expected)
    assert params.get("scaling_applied") is False


def test_rf_transform_never_mirrors(roles):
    val_like = _synthetic_df(roles, n=40, seed=5, with_missing=False)
    train_like = _synthetic_df(roles, n=200, seed=6)
    params = rf_fit_preprocessing(build_augmented_training_raw(train_like, roles), roles)
    X_val, _ = rf_transform(val_like, params, roles)
    assert X_val.shape[0] == 40


# --- XGB: NaN preserved, no imputation, no scaling ---

def test_xgb_preserves_nan_no_imputation(roles):
    train_like = _synthetic_df(roles, n=300, seed=7)
    augmented = build_augmented_training_raw(train_like, roles)
    params = xgb_fit_preprocessing(augmented, roles)

    X, names = xgb_transform(augmented, params, roles)
    assert params.get("imputation_applied") is False
    assert params.get("scaling_applied") is False
    if "days_since_last_match_diff" in roles["directional"]:
        idx = names.index("days_since_last_match_diff")
        expected = augmented["days_since_last_match_diff"].to_numpy(dtype=float)
        np.testing.assert_array_equal(X[:, idx], expected)
        assert np.isnan(X[:, idx]).any()


# --- transformed feature count matches the config's declared totals ---

def test_transformed_feature_names_count_matches_config(roles):
    names = transformed_feature_names(roles)
    expected = len(roles["directional"]) + len(roles["symmetric"]) + 4  # +2 bestOf dummies +2 tier dummies
    assert len(names) == expected
    assert len(names) == len(set(names))  # no duplicate column names


def test_mirror_and_transform_are_config_driven_not_v1_hardcoded(roles):
    """V2 has 30 directional features (20 new + 10 inherited); V1's hardcoded
    mirror only knows about 10. Confirms the V2 mirror negates all 30, not
    just the 10 inherited ones."""
    assert len(roles["directional"]) == 30
    df = _synthetic_df(roles, n=20, with_missing=False)
    mirrored = mirror_raw_rows(df, roles)
    n_negated = sum(1 for c in roles["directional"] if np.allclose(mirrored[c].to_numpy(), -df[c].to_numpy()))
    assert n_negated == 30
