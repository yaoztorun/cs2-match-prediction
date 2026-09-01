"""
[PROJECT ADDITION - Phase 5C.1]

Role loader for the V4 player/roster series feature set
(config/series_features_v4_roster.yaml, data/features/series_features_v4_roster.parquet).

Every downstream operation (mirroring, augmentation, RF/XGB fit/transform) is
ALREADY fully generic given a `roles` dict - see
feature_engineering/preprocessing/preprocessing_common_v2_map_pool.py / preprocessing_random_forest_v2_map_pool.py /
preprocessing_xgboost_v2_map_pool.py, none of which are modified here or
anywhere in Phase 5C.1. The ONLY feature-set-specific piece in that shared
module is its `V2_BINARY_SYMMETRIC_FEATURES`-style constant (which of the
config's `symmetric_features` are 0/1 flags vs. counts/masses/ratios - not
derivable from the YAML itself). V4's config adds 6 new symmetric features,
only ONE of which is a binary flag, so `load_v3_roles` cannot be reused as-is
(its own internal assertion would fail on a name absent from V3's
`symmetric_features`). This module is the minimal V4-specific addition:
`load_v4_roles` builds the identical `roles` dict shape, just against V4's
config and an extended binary-flags list, so the same generic mirror/fit/
transform functions work unchanged for V4.

This also means the ten NaN-capable roster performance diffs
(player_roster_feature_engine.ROSTER_PERFORMANCE_DIFFS) get correct handling
"for free": they are `directional`, so RF's fit_preprocessing computes their
train median (pandas .median() skips NaN) and XGB's transform leaves them
untouched - no new imputation code is needed anywhere in Phase 5C.1.
"""

import yaml

# V3's 8 binary confidence flags + the ONE new V4 binary flag.
# The other 5 new V4 symmetric features (roster_size_min,
# roster_min_player_history_mass, roster_core_concentration_min,
# roster_core_continuity_last10_min, roster_form_players_min) are continuous
# counts/masses/ratios, not 0/1 flags.
V4_BINARY_SYMMETRIC_FEATURES = [
    "both_teams_have_map_pool_history", "both_teams_have_3_recent_maps", "both_teams_have_5_experienced_maps",
    "both_teams_have_history", "both_teams_have_5_matches", "both_teams_have_10_matches",
    "both_teams_have_5_adjusted_matches", "both_teams_have_10_adjusted_matches",
    "both_teams_have_5_inferred_players",
]


def load_v4_roles(config_path):
    """Reads config/series_features_v4_roster.yaml and returns a dict of
    feature-role lists in the exact shape
    preprocessing_common_v2_map_pool.load_v2_roles / preprocessing_common_v3_form.load_v3_roles
    return, so every function in those modules works unchanged against it."""
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    directional = list(cfg["directional_features"])
    symmetric = list(cfg["symmetric_features"])
    categorical = list(cfg["categorical_context"])
    target = cfg["target"]

    missing = set(V4_BINARY_SYMMETRIC_FEATURES) - set(symmetric)
    assert not missing, f"V4_BINARY_SYMMETRIC_FEATURES not found in config symmetric_features: {missing}"
    symmetric_binary = [c for c in symmetric if c in V4_BINARY_SYMMETRIC_FEATURES]
    symmetric_continuous = [c for c in symmetric if c not in V4_BINARY_SYMMETRIC_FEATURES]

    return {
        "directional": directional,
        "symmetric": symmetric,
        "symmetric_continuous": symmetric_continuous,
        "symmetric_binary": symmetric_binary,
        "categorical": categorical,
        "target": target,
        "model_features": directional + symmetric + categorical,
    }
