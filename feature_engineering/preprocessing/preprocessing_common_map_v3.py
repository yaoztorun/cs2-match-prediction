"""
[PROJECT ADDITION - Phase 6C]

Role loader for the MODERN SELECTED-MAP feature set
(config/map_features_v3_modern_map.yaml, data/features/map_features_v3_modern_map.parquet).

Every downstream operation (mirroring, augmentation, categorical encoding,
RF/XGB fit/transform) is ALREADY fully generic given a `roles` dict - see
feature_engineering/preprocessing/preprocessing_common_map_v2.py, none of which is modified here or
anywhere in Phase 6C. The ONLY feature-set-specific piece in that shared
module is its own binary-vs-continuous symmetric-feature split (not
derivable from the YAML itself, since it declares no dtype). This module is
the minimal V3-specific addition: `load_map_v3_roles` builds the identical
`roles` dict shape against V3's config and an extended binary-flags list, so
every generic function in preprocessing_common_map_v2.py works unchanged.
"""

import yaml

from feature_engineering.preprocessing.preprocessing_common_map_v2 import MAP_V2_BINARY_SYMMETRIC_FEATURES

# V2's 12 binary confidence flags + the 2 new Phase 6C binary flags. The other
# 5 new Phase 6C symmetric features (map_recent_history_mass_min,
# map_adjusted_history_mass_min, roster_map_players_with_history_min,
# roster_map_history_mass_min, current_core_map_continuity_min) are
# continuous counts/masses/ratios, not 0/1 flags.
MAP_V3_BINARY_SYMMETRIC_FEATURES = list(MAP_V2_BINARY_SYMMETRIC_FEATURES) + [
    "both_teams_have_recent_selected_map_history", "selected_map_in_both_recent_pools",
]

EXPECTED_RAW_PREDICTIVE_INPUTS = 120   # 80 directional + 37 symmetric + 3 categorical


def load_map_v3_roles(config_path):
    """Reads config/map_features_v3_modern_map.yaml and returns the feature-role
    dict every fit/transform/mirror function in Phase 6C consumes. Feature
    IDENTITY always comes from the YAML - never from 'every numeric column'."""
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    directional = list(cfg["directional_features"])
    symmetric = list(cfg["symmetric_features"])
    categorical = list(cfg["categorical_context"])
    target = cfg["target"]

    assert len(directional) == 80, f"expected 80 directional features, config declares {len(directional)}"
    assert len(symmetric) == 37, f"expected 37 symmetric features, config declares {len(symmetric)}"
    assert categorical == ["map_name", "bestOf", "tier"], f"unexpected categorical context: {categorical}"

    missing = set(MAP_V3_BINARY_SYMMETRIC_FEATURES) - set(symmetric)
    assert not missing, f"binary symmetric flags absent from config symmetric_features: {missing}"
    symmetric_binary = [c for c in symmetric if c in MAP_V3_BINARY_SYMMETRIC_FEATURES]
    symmetric_continuous = [c for c in symmetric if c not in MAP_V3_BINARY_SYMMETRIC_FEATURES]

    roles = {
        "directional": directional,
        "symmetric": symmetric,
        "symmetric_continuous": symmetric_continuous,
        "symmetric_binary": symmetric_binary,
        "categorical": categorical,
        "target": target,
        "model_features": directional + symmetric + categorical,
    }
    assert len(roles["model_features"]) == EXPECTED_RAW_PREDICTIVE_INPUTS
    return roles
