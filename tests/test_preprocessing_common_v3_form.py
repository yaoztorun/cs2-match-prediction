"""
Tests for scripts/preprocessing_common_v3_form.py (Phase 5B.3). Confirms the
V3 role loader produces a `roles` dict that the (unmodified) Phase 5B.1
generic mirror/fit/transform functions handle correctly.
"""

import numpy as np
import pandas as pd
import pytest
import yaml

from _common import ROOT
from preprocessing_common_v3_form import load_v3_roles, V3_BINARY_SYMMETRIC_FEATURES
from preprocessing_common_v2_map_pool import mirror_raw_rows, transformed_feature_names

CONFIG_V3_PATH = ROOT / "config" / "series_features_v3_form.yaml"


@pytest.fixture(scope="module")
def roles():
    return load_v3_roles(CONFIG_V3_PATH)


def test_roles_match_the_real_v3_config_exactly(roles):
    cfg = yaml.safe_load(CONFIG_V3_PATH.read_text(encoding="utf-8"))
    assert roles["directional"] == cfg["directional_features"]
    assert roles["symmetric"] == cfg["symmetric_features"]
    assert roles["categorical"] == cfg["categorical_context"]
    assert roles["target"] == cfg["target"]
    assert len(roles["directional"]) == 38
    assert len(roles["symmetric"]) == 19


def test_new_form_features_land_in_the_correct_continuous_vs_binary_bucket(roles):
    assert "opponent_adjusted_history_min" in roles["symmetric_continuous"]
    assert "time_weighted_history_mass_min" in roles["symmetric_continuous"]
    assert "both_teams_have_5_adjusted_matches" in roles["symmetric_binary"]
    assert "both_teams_have_10_adjusted_matches" in roles["symmetric_binary"]
    assert len(roles["symmetric_continuous"]) == 11
    assert len(roles["symmetric_binary"]) == 8


def test_binary_symmetric_features_all_declared_in_config(roles):
    assert set(V3_BINARY_SYMMETRIC_FEATURES) <= set(roles["symmetric"])


def _synthetic_df(roles, n=50, seed=0):
    rng = np.random.default_rng(seed)
    data = {c: rng.normal(scale=50, size=n) for c in roles["directional"]}
    for c in roles["symmetric_continuous"]:
        data[c] = rng.integers(0, 100, size=n).astype(float)
    for c in roles["symmetric_binary"]:
        data[c] = rng.integers(0, 2, size=n).astype(float)
    data["bestOf"] = rng.choice([1, 3, 5], size=n)
    data["tier"] = rng.choice(["tier1", "tier2", "tier3"], size=n)
    data["team1_series_win"] = rng.integers(0, 2, size=n).astype(float)
    return pd.DataFrame(data)


def test_mirror_negates_all_38_directional_and_leaves_everything_else(roles):
    df = _synthetic_df(roles)
    mirrored = mirror_raw_rows(df, roles)

    n_negated = sum(1 for c in roles["directional"]
                     if np.allclose(mirrored[c].to_numpy(), -df[c].to_numpy()))
    assert n_negated == 38

    for c in roles["symmetric"]:
        pd.testing.assert_series_equal(mirrored[c], df[c])
    for c in ["bestOf", "tier"]:
        pd.testing.assert_series_equal(mirrored[c], df[c])
    assert (mirrored["team1_series_win"] == 1 - df["team1_series_win"]).all()


def test_transformed_feature_names_count(roles):
    names = transformed_feature_names(roles)
    assert len(names) == 38 + 19 + 4   # directional + symmetric + 2 bestOf dummies + 2 tier dummies
    assert len(names) == len(set(names))
