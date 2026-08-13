"""
[PROJECT ADDITION - Phase 6B]

Config-driven roles/mirroring/encoding for the KNOWN-MAP feature set
(config/map_features_v2_rich.yaml, data/features/map_features_v2_rich.parquet).

This is a NEW module, not an edit of the frozen series preprocessing modules
(preprocessing_common.py, preprocessing_common_v2_map_pool.py,
preprocessing_common_v3_form.py, preprocessing_common_v4_roster.py) - all of
which stay byte-identical, since Phase 4/5 artifacts were built and validated
with them. The map task needs one thing none of them provide: `map_name` as a
legitimate one-hot categorical, plus explicit `__UNKNOWN_MAP__` /
`__UNKNOWN_TIER__` inference categories.

CATEGORICAL CONTRACT
--------------------
Deterministic reference-category one-hot encoding, vocabulary and column order
frozen here and persisted into every preprocessing artifact:

    map_name : 9 historical identities + __UNKNOWN_MAP__ (10 categories),
               reference = Mirage           -> 9 dummy columns
    bestOf   : 1, 3, 5,        reference = 1 (BO1)       -> 2 dummy columns
    tier     : tier1/2/3 + __UNKNOWN_TIER__, reference = tier1 -> 3 dummy columns

`__UNKNOWN_MAP__` and `__UNKNOWN_TIER__` each get their OWN explicit dummy
column and are never collapsed onto the reference category - an unknown map is
NOT silently treated as Mirage and an unknown tier is NOT silently treated as
tier1. (Phase 6A's config note floated an all-zero-dummy encoding for unknown
tier as one option; that would have made `__UNKNOWN_TIER__` numerically
identical to tier1, so the explicit-dummy form specified for Phase 6B is used
instead. config/map_features_v2_rich.yaml itself is not modified.)

Maps are never ordinal-encoded as arbitrary integers.

TRANSFORMED DIMENSION
---------------------
    62 directional + 18 continuous symmetric + 12 binary symmetric
  +  9 map dummies + 2 bestOf dummies + 3 tier dummies
  = 106 transformed columns, from 95 raw predictive inputs.

Mirroring negates every `directional_features` column and flips the target;
`symmetric_features` and `categorical_context` are never touched. Mirroring is
a TRAIN-side augmentation only - a mirrored row is a synthetic re-labelling of
an already-counted map, never an additional map.
"""

import pandas as pd
import yaml

UNKNOWN_MAP_CATEGORY = "__UNKNOWN_MAP__"
UNKNOWN_TIER_CATEGORY = "__UNKNOWN_TIER__"

# The 9 map identities present in the historical data, plus the reserved
# inference category. Fixed order - persisted into every preprocessing artifact
# and asserted by scripts/validate_phase6b.py.
MAP_CATEGORIES = [
    "Ancient", "Anubis", "Dust2", "Inferno", "Mirage",
    "Nuke", "Overpass", "Train", "Vertigo", UNKNOWN_MAP_CATEGORY,
]
MAP_REFERENCE = "Mirage"
MAP_DUMMIES = [m for m in MAP_CATEGORIES if m != MAP_REFERENCE]

BESTOF_CATEGORIES = [1, 3, 5]
BESTOF_REFERENCE = 1
BESTOF_DUMMIES = [3, 5]

TIER_CATEGORIES = ["tier1", "tier2", "tier3", UNKNOWN_TIER_CATEGORY]
TIER_REFERENCE = "tier1"
TIER_DUMMIES = [t for t in TIER_CATEGORIES if t != TIER_REFERENCE]

# Which of the config's 30 `symmetric_features` are 0/1 confidence FLAGS rather
# than counts/masses/ratios. Not derivable from the YAML (it declares no
# dtype), so stated explicitly here - the same convention
# preprocessing_common_v2_map_pool.py / preprocessing_common_v4_roster.py use.
MAP_V2_BINARY_SYMMETRIC_FEATURES = [
    "both_teams_have_map_history", "both_teams_have_5_map_matches", "both_teams_have_10_map_matches",
    "both_teams_have_map_pool_history", "both_teams_have_3_recent_maps", "both_teams_have_5_experienced_maps",
    "both_teams_have_5_adjusted_matches", "both_teams_have_10_adjusted_matches",
    "both_teams_have_5_inferred_players",
    "both_teams_have_history", "both_teams_have_5_matches", "both_teams_have_10_matches",
]

EXPECTED_RAW_PREDICTIVE_INPUTS = 95      # 62 directional + 30 symmetric + 3 categorical
EXPECTED_TRANSFORMED_FEATURES = 106      # after the deterministic one-hot encoding above


def load_map_v2_roles(config_path):
    """Reads config/map_features_v2_rich.yaml and returns the feature-role dict
    every fit/transform/mirror function in Phase 6B consumes. Feature IDENTITY
    always comes from the YAML - never from 'every numeric column'."""
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    directional = list(cfg["directional_features"])
    symmetric = list(cfg["symmetric_features"])
    categorical = list(cfg["categorical_context"])
    target = cfg["target"]

    assert len(directional) == 62, f"expected 62 directional features, config declares {len(directional)}"
    assert len(symmetric) == 30, f"expected 30 symmetric features, config declares {len(symmetric)}"
    assert categorical == ["map_name", "bestOf", "tier"], f"unexpected categorical context: {categorical}"

    missing = set(MAP_V2_BINARY_SYMMETRIC_FEATURES) - set(symmetric)
    assert not missing, f"binary symmetric flags absent from config symmetric_features: {missing}"
    symmetric_binary = [c for c in symmetric if c in MAP_V2_BINARY_SYMMETRIC_FEATURES]
    symmetric_continuous = [c for c in symmetric if c not in MAP_V2_BINARY_SYMMETRIC_FEATURES]

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


def transformed_feature_names(roles):
    """The frozen transformed column order. Training and future inference MUST
    use this identical ordering - it is persisted into every preprocessing
    artifact and re-asserted on load."""
    names = list(roles["directional"]) + list(roles["symmetric_continuous"]) + list(roles["symmetric_binary"])
    names += [f"map_name_{m}" for m in MAP_DUMMIES]
    names += [f"bestOf_BO{d}" for d in BESTOF_DUMMIES]
    names += [f"tier_{t}" for t in TIER_DUMMIES]
    return names


def categorical_vocabulary():
    """The exact vocabulary/order persisted into preprocessing artifacts."""
    return {
        "map_name": {"categories": list(MAP_CATEGORIES), "reference": MAP_REFERENCE,
                      "dummies": list(MAP_DUMMIES), "unknown_category": UNKNOWN_MAP_CATEGORY},
        "bestOf": {"categories": list(BESTOF_CATEGORIES), "reference": BESTOF_REFERENCE,
                    "dummies": list(BESTOF_DUMMIES), "unknown_category": None},
        "tier": {"categories": list(TIER_CATEGORIES), "reference": TIER_REFERENCE,
                  "dummies": list(TIER_DUMMIES), "unknown_category": UNKNOWN_TIER_CATEGORY},
    }


def resolve_categoricals(df):
    """Map the three raw categorical columns onto the frozen vocabulary.

    A map identity outside the training vocabulary (or missing) becomes
    __UNKNOWN_MAP__; a missing/out-of-vocabulary tier becomes __UNKNOWN_TIER__.
    Both then get their OWN dummy column, never the reference category. bestOf
    has no unknown category - the application always knows the series format -
    so an unexpected value is an error rather than a silent fallback.

    Returns (map_series, bestof_series, tier_series), all NaN-free."""
    map_raw = df["map_name"]
    map_col = map_raw.where(map_raw.isin(MAP_CATEGORIES), UNKNOWN_MAP_CATEGORY)

    tier_raw = df["tier"]
    tier_col = tier_raw.where(tier_raw.isin(TIER_CATEGORIES), UNKNOWN_TIER_CATEGORY)

    bestof_col = df["bestOf"]
    bad = set(pd.unique(bestof_col.dropna())) - set(BESTOF_CATEGORIES)
    assert not bad and bestof_col.notna().all(), f"bestOf outside the frozen vocabulary {BESTOF_CATEGORIES}: {bad}"

    assert map_col.notna().all() and tier_col.notna().all(), \
        "categorical column still NaN after the explicit unknown-category contract"
    return map_col, bestof_col, tier_col


def mirror_raw_rows(df, roles):
    """RAW-level side mirroring: negate every directional feature, flip the
    target if present, leave symmetric/confidence, categorical context and
    metadata untouched."""
    mirrored = df.copy()
    for col in roles["directional"]:
        mirrored[col] = -mirrored[col]
    target = roles["target"]
    if target in mirrored.columns:
        mirrored[target] = 1 - mirrored[target]
    return mirrored


def build_augmented_training_raw(df, roles):
    """TRAIN ONLY. Original raw map rows + their mirrored counterparts. The
    result is 2x the number of training OBSERVATIONS, never 2x the number of
    underlying historical maps."""
    return pd.concat([df, mirror_raw_rows(df, roles)], ignore_index=True)


def assert_augmented_symmetry(augmented_df, roles):
    """Same tolerance policy as every prior phase: a fully-populated
    directional feature must have a ~0 mean over the augmented set; one with
    genuine missingness (the documented cold-start NaNs) gets a looser
    std-relative tolerance, because mirroring negates NaN to NaN and a mirrored
    pair therefore collapses onto one imputed value rather than exact
    negatives."""
    for c in roles["directional"]:
        col = augmented_df[c]
        mean_before_impute = col.mean()   # pandas .mean() skips NaN
        std = col.std(ddof=1)
        if col.isna().any():
            assert abs(mean_before_impute) < 0.1 * std, (
                f"{c}: augmented raw mean {mean_before_impute} not small relative to std {std}")
        else:
            assert abs(mean_before_impute) < 1e-8, (
                f"{c}: augmented raw mean {mean_before_impute} should be ~0 (exact symmetry, no missingness)")
