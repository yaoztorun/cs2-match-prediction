"""
Tests for scripts/preprocessing_common_v4_roster.py (Phase 5C.1). Confirms
the V4 role loader produces a `roles` dict the (unmodified) Phase 5B.1
generic mirror/fit/transform functions handle correctly, including the
NaN-capable roster-performance diffs.
"""

import numpy as np
import pandas as pd
import pytest
import yaml

from _common import ROOT
from preprocessing_common_v4_roster import load_v4_roles, V4_BINARY_SYMMETRIC_FEATURES
from preprocessing_common_v2_map_pool import mirror_raw_rows, transformed_feature_names
import preprocessing_random_forest_v2_map_pool as rf2
import preprocessing_xgboost_v2_map_pool as xgb2
from player_roster_feature_engine import ROSTER_PERFORMANCE_DIFFS

CONFIG_V4_PATH = ROOT / "config" / "series_features_v4_roster.yaml"


@pytest.fixture(scope="module")
def roles():
    return load_v4_roles(CONFIG_V4_PATH)


def test_roles_match_the_real_v4_config_exactly(roles):
    cfg = yaml.safe_load(CONFIG_V4_PATH.read_text(encoding="utf-8"))
    assert roles["directional"] == cfg["directional_features"]
    assert roles["symmetric"] == cfg["symmetric_features"]
    assert roles["categorical"] == cfg["categorical_context"]
    assert roles["target"] == cfg["target"]
    assert len(roles["directional"]) == 53
    assert len(roles["symmetric"]) == 25


def test_new_roster_features_land_in_the_correct_bucket(roles):
    for c in ["roster_size_min", "roster_min_player_history_mass",
              "roster_core_concentration_min", "roster_core_continuity_last10_min",
              "roster_form_players_min"]:
        assert c in roles["symmetric_continuous"], c
    assert "both_teams_have_5_inferred_players" in roles["symmetric_binary"]
    assert len(roles["symmetric_continuous"]) == 16
    assert len(roles["symmetric_binary"]) == 9


def test_binary_symmetric_features_all_declared_in_config(roles):
    assert set(V4_BINARY_SYMMETRIC_FEATURES) <= set(roles["symmetric"])


def _synthetic_df(roles, n=60, seed=0, with_roster_nan=True):
    rng = np.random.default_rng(seed)
    data = {c: rng.normal(scale=10, size=n) for c in roles["directional"]}
    for c in roles["symmetric_continuous"]:
        data[c] = rng.integers(0, 20, size=n).astype(float)
    for c in roles["symmetric_binary"]:
        data[c] = rng.integers(0, 2, size=n).astype(float)
    data["bestOf"] = rng.choice([1, 3, 5], size=n)
    data["tier"] = rng.choice(["tier1", "tier2", "tier3"], size=n)
    data["team1_series_win"] = rng.integers(0, 2, size=n).astype(float)
    df = pd.DataFrame(data)
    if with_roster_nan:
        idx = rng.choice(n, size=max(1, n // 5), replace=False)
        for c in ROSTER_PERFORMANCE_DIFFS:
            df.loc[idx, c] = np.nan
    return df


def test_mirror_negates_all_53_directional_and_preserves_nan(roles):
    df = _synthetic_df(roles)
    mirrored = mirror_raw_rows(df, roles)

    n_negated = 0
    for c in roles["directional"]:
        a, b = df[c].to_numpy(), mirrored[c].to_numpy()
        if np.allclose(np.nan_to_num(a, nan=0.0), -np.nan_to_num(b, nan=0.0)) and \
           np.array_equal(np.isnan(a), np.isnan(b)):
            n_negated += 1
    assert n_negated == 53

    for c in ROSTER_PERFORMANCE_DIFFS:
        assert df[c].isna().equals(mirrored[c].isna()), f"{c}: mirroring must preserve NaN positions"

    for c in roles["symmetric"]:
        pd.testing.assert_series_equal(mirrored[c], df[c])
    assert (mirrored["team1_series_win"] == 1 - df["team1_series_win"]).all()


def test_rf_median_imputation_handles_nan_roster_diffs(roles):
    df = _synthetic_df(roles)
    augmented = pd.concat([df, mirror_raw_rows(df, roles)], ignore_index=True)
    params = rf2.fit_preprocessing(augmented, roles)
    X, names = rf2.transform(augmented, params, roles)
    assert np.isfinite(X).all(), "RF-transformed matrix must have no NaN left after median imputation"
    for c in ROSTER_PERFORMANCE_DIFFS:
        assert c in params["train_medians"], f"{c} must have a fold-fitted median"


def test_xgb_preserves_native_nan_in_roster_diffs(roles):
    df = _synthetic_df(roles)
    augmented = pd.concat([df, mirror_raw_rows(df, roles)], ignore_index=True)
    params = xgb2.fit_preprocessing(augmented, roles)
    X, names = xgb2.transform(augmented, params, roles)
    idx = names.index(ROSTER_PERFORMANCE_DIFFS[0])
    assert np.isnan(X[:, idx]).any(), "XGB transform must never impute the roster performance diffs"
    assert params.get("imputation_applied") is False


def test_transformed_feature_names_count(roles):
    names = transformed_feature_names(roles)
    assert len(names) == 53 + 25 + 4   # directional + symmetric + 2 bestOf dummies + 2 tier dummies
    assert len(names) == len(set(names))
