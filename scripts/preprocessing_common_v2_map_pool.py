"""
[PROJECT ADDITION - Phase 5B.1]

Config-driven mirroring for the V2 map-pool series feature set
(config/series_features_v2_map_pool.yaml, data/features/series_features_v2_map_pool.parquet).

scripts/preprocessing_common.py's mirror_raw_rows/build_augmented_training_raw/
assert_augmented_symmetry are hardcoded to V1's 17-feature whitelist and are left
completely UNMODIFIED here - they remain exactly the functions Phase 4's frozen
artifacts were built and validated with. This module is the generalized,
config-driven equivalent for V2's 47 features: it reads
`directional_features` / `symmetric_features` / `categorical_context` straight
from the YAML at call time instead of relying on module-level constants, so
"mirror this feature set" always matches whatever that config declares.

Mirroring negates every column in `directional_features` and flips the target;
`symmetric_features` and `categorical_context` (bestOf, tier) are never touched.
"""

import pandas as pd

from preprocessing_common import BESTOF_DUMMIES, TIER_DUMMIES

# Of series_features_v2_map_pool.yaml's 15 `symmetric_features`, these 9 are
# boolean 0/1 confidence flags; the rest are counts/ratios that get
# median-imputed by the RF preprocessing module. Not derivable from the YAML
# itself (it declares no dtype) - fixed here explicitly, the same way
# preprocessing_common.py already hardcodes V1's BINARY_FEATURES vs.
# SYMMETRIC_COUNT_FEATURES split.
V2_BINARY_SYMMETRIC_FEATURES = [
    "both_teams_have_map_pool_history", "both_teams_have_3_recent_maps", "both_teams_have_5_experienced_maps",
    "both_teams_have_history", "both_teams_have_5_matches", "both_teams_have_10_matches",
]


def load_v2_roles(config_path):
    """Reads config/series_features_v2_map_pool.yaml and returns a dict of
    feature-role lists, entirely config-driven (directional/symmetric feature
    IDENTITY comes from the YAML, not a hardcoded Python list)."""
    import yaml
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    directional = list(cfg["directional_features"])
    symmetric = list(cfg["symmetric_features"])
    categorical = list(cfg["categorical_context"])
    target = cfg["target"]

    missing = set(V2_BINARY_SYMMETRIC_FEATURES) - set(symmetric)
    assert not missing, f"V2_BINARY_SYMMETRIC_FEATURES not found in config symmetric_features: {missing}"
    symmetric_binary = [c for c in symmetric if c in V2_BINARY_SYMMETRIC_FEATURES]
    symmetric_continuous = [c for c in symmetric if c not in V2_BINARY_SYMMETRIC_FEATURES]

    return {
        "directional": directional,
        "symmetric": symmetric,
        "symmetric_continuous": symmetric_continuous,
        "symmetric_binary": symmetric_binary,
        "categorical": categorical,
        "target": target,
        "model_features": directional + symmetric + categorical,
    }


def mirror_raw_rows(df, roles):
    """Given a RAW dataframe with roles["model_features"] (+ target, if
    present), return the mirrored version: every roles["directional"] column
    negated, everything else (symmetric, categorical, metadata) unchanged,
    target flipped (1-y) if present."""
    mirrored = df.copy()
    for col in roles["directional"]:
        mirrored[col] = -mirrored[col]
    target = roles["target"]
    if target in mirrored.columns:
        mirrored[target] = 1 - mirrored[target]
    return mirrored


def build_augmented_training_raw(df, roles):
    """TRAIN ONLY. original raw rows + their raw mirrored counterparts."""
    mirrored = mirror_raw_rows(df, roles)
    return pd.concat([df, mirrored], ignore_index=True)


def transformed_feature_names(roles):
    names = list(roles["directional"]) + list(roles["symmetric_continuous"]) + list(roles["symmetric_binary"])
    names += [f"bestOf_BO{d}" for d in BESTOF_DUMMIES]
    names += [f"tier_{t}" for t in TIER_DUMMIES]
    return names


def assert_augmented_symmetry(augmented_df, roles):
    """Same tolerance policy as preprocessing_common.assert_augmented_symmetry:
    a fully-populated directional feature gets a tight absolute tolerance; a
    feature with real missingness gets a looser, std-relative tolerance
    (mirroring negates NaN -> NaN, so a mirrored pair collapses onto a single
    imputed median rather than exact negatives). Verified in Phase 5A: all 30
    new map-pool features are fully populated; only the inherited
    days_since_last_match_diff carries real missingness."""
    for c in roles["directional"]:
        col = augmented_df[c]
        mean_before_impute = col.mean()
        std = col.std(ddof=1)
        if col.isna().any():
            assert abs(mean_before_impute) < 0.1 * std, (
                f"{c}: augmented raw mean {mean_before_impute} not small relative to std {std}")
        else:
            assert abs(mean_before_impute) < 1e-8, (
                f"{c}: augmented raw mean {mean_before_impute} should be ~0 (exact symmetry, no missingness)")
