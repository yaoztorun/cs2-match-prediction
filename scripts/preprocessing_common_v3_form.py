"""
[PROJECT ADDITION - Phase 5B.3]

Role loader for the V3 team-form series feature set
(config/series_features_v3_form.yaml, data/features/series_features_v3_form.parquet).

Every downstream operation (mirroring, augmentation, RF/XGB fit/transform) is
ALREADY fully generic given a `roles` dict - see
scripts/preprocessing_common_v2_map_pool.py / preprocessing_random_forest_v2_map_pool.py /
preprocessing_xgboost_v2_map_pool.py, none of which are modified here or
anywhere in Phase 5B.3. The ONLY V2-specific piece in that Phase 5B.1 module
is its hardcoded `V2_BINARY_SYMMETRIC_FEATURES` list (which of the config's
`symmetric_features` are 0/1 flags vs. counts - not derivable from the YAML
itself). V3's config adds 4 new symmetric features, 2 of which are new binary
flags not in V2's list, so `load_v2_roles` cannot be reused as-is for V3's
config (its own internal assertion would fail on names absent from V2's
`symmetric_features`). This module is the minimal V3-specific addition:
`load_v3_roles` builds the identical `roles` dict shape, just against V3's
config and an extended binary-flags list, so the same generic mirror/fit/
transform functions work unchanged for both feature sets.
"""

import yaml

# V2's 6 binary confidence flags + the 2 new V3 binary flags
# (opponent_adjusted_history_min and time_weighted_history_mass_min are
# continuous - counts/mass, not 0/1 flags).
V3_BINARY_SYMMETRIC_FEATURES = [
    "both_teams_have_map_pool_history", "both_teams_have_3_recent_maps", "both_teams_have_5_experienced_maps",
    "both_teams_have_history", "both_teams_have_5_matches", "both_teams_have_10_matches",
    "both_teams_have_5_adjusted_matches", "both_teams_have_10_adjusted_matches",
]


def load_v3_roles(config_path):
    """Reads config/series_features_v3_form.yaml and returns a dict of
    feature-role lists in the exact shape
    preprocessing_common_v2_map_pool.load_v2_roles returns, so every function
    in that module (and in preprocessing_random_forest_v2_map_pool.py /
    preprocessing_xgboost_v2_map_pool.py) works unchanged against it."""
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    directional = list(cfg["directional_features"])
    symmetric = list(cfg["symmetric_features"])
    categorical = list(cfg["categorical_context"])
    target = cfg["target"]

    missing = set(V3_BINARY_SYMMETRIC_FEATURES) - set(symmetric)
    assert not missing, f"V3_BINARY_SYMMETRIC_FEATURES not found in config symmetric_features: {missing}"
    symmetric_binary = [c for c in symmetric if c in V3_BINARY_SYMMETRIC_FEATURES]
    symmetric_continuous = [c for c in symmetric if c not in V3_BINARY_SYMMETRIC_FEATURES]

    return {
        "directional": directional,
        "symmetric": symmetric,
        "symmetric_continuous": symmetric_continuous,
        "symmetric_binary": symmetric_binary,
        "categorical": categorical,
        "target": target,
        "model_features": directional + symmetric + categorical,
    }
