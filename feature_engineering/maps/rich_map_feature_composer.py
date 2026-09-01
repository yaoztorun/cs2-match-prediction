"""
Phase 6A future-application composer for the known-map prediction task.

Thin, PURE composition over four ALREADY-EXISTING, already-tested future
builders - one per prior phase's state type. Adds no new temporal/streaming
logic: every number returned here is produced by a function that was already
proven leakage-safe (no target, no current score, no lineup) in its own
phase's own test suite. This module only merges their output dicts into the
exact config/map_features_v2_rich.yaml schema (95 predictive inputs: 62
directional + 30 symmetric/confidence + 3 categorical context).

    series_state        : feature_engine.StateStore                    (Phase 3)
    map_state           : map_feature_engine.MapStateStore              (Phase 5A)
    form_state          : team_form_engine.TeamFormStateStore           (Phase 5B.2)
    player_roster_state : player_roster_feature_engine.PlayerRosterStateStore (Phase 5C)

TIER CONTRACT
-------------
The application asks the user for two teams, a bestOf and a map - NOT a
tournament tier - so a caller may legitimately have no tier value. `tier`
is passthrough-only context in every upstream engine (it never gates any
numeric feature computation), so passing `tier=None` here emits the reserved
`UNKNOWN_TIER_CATEGORY` ("__UNKNOWN_TIER__") instead of silently defaulting
to a real competitive tier - the same principle
`map_feature_engine.UNKNOWN_MAP_CATEGORY` already applies to an unseen map.
Downstream map-model preprocessing (not built in this phase) must treat
`__UNKNOWN_TIER__` as its own explicit category, never collapsed onto tier1.

NaN CONTRACT
------------
The returned dict is NOT guaranteed fully finite, and must not be forced to
be: cold-start features from every inherited phase carry documented NaN
(`days_since_map_played_diff`, `days_since_last_match_diff`, and the ten V4
roster-performance diffs) exactly when their paired confidence/evidence
feature says there is no history. No feature here is ever imputed with a
population mean to manufacture finiteness - see validation/validate_phase6a.py
for the exact NaN-vs-confidence invariant this is checked against.
"""

import pandas as pd

from feature_engineering.series.feature_engine import build_features
from feature_engineering.maps.map_feature_engine import (
    build_future_map_matchup_features, build_future_series_map_pool_features,
    MAP_DIRECTIONAL_FEATURES, MAP_SYMMETRIC_FEATURES,
    SERIES_MAP_POOL_DIRECTIONAL, SERIES_MAP_POOL_SYMMETRIC,
)
from feature_engineering.form.team_form_engine import (
    build_future_team_form_features, FORM_DIRECTIONAL_FEATURES, FORM_SYMMETRIC_FEATURES,
)
from feature_engineering.roster.player_roster_feature_engine import (
    build_future_player_roster_features, ROSTER_DIRECTIONAL_FEATURES, ROSTER_SYMMETRIC_FEATURES,
)

UNKNOWN_TIER_CATEGORY = "__UNKNOWN_TIER__"

# Phase 3's inherited series-level features - no exported constant list exists
# upstream (feature_engine.py predates that convention), so this is hardcoded
# here exactly as feature_engineering/maps/build_series_features_v2_map_pool.py's own
# V1_FEATURES list already does, for the same reason.
V1_SERIES_DIRECTIONAL = [
    "elo_diff", "overall_win_rate_diff", "win_rate_last_5_diff", "win_rate_last_10_diff",
    "format_win_rate_diff", "avg_series_margin_last_5_diff", "avg_series_margin_last_10_diff",
    "matches_last_30_days_diff", "days_since_last_match_diff", "total_matches_before_diff",
]
V1_SERIES_SYMMETRIC = [
    "history_matches_min", "history_matches_sum", "both_teams_have_history",
    "both_teams_have_5_matches", "both_teams_have_10_matches",
]

RICH_MAP_DIRECTIONAL_FEATURES = (
    list(MAP_DIRECTIONAL_FEATURES) + list(SERIES_MAP_POOL_DIRECTIONAL)
    + list(FORM_DIRECTIONAL_FEATURES) + list(ROSTER_DIRECTIONAL_FEATURES) + list(V1_SERIES_DIRECTIONAL)
)
RICH_MAP_SYMMETRIC_FEATURES = (
    list(MAP_SYMMETRIC_FEATURES) + list(SERIES_MAP_POOL_SYMMETRIC)
    + list(FORM_SYMMETRIC_FEATURES) + list(ROSTER_SYMMETRIC_FEATURES) + list(V1_SERIES_SYMMETRIC)
)
RICH_MAP_CATEGORICAL_CONTEXT = ["map_name", "bestOf", "tier"]

assert len(RICH_MAP_DIRECTIONAL_FEATURES) == 62
assert len(RICH_MAP_SYMMETRIC_FEATURES) == 30
assert len(RICH_MAP_CATEGORICAL_CONTEXT) == 3


def build_future_rich_map_features(team1, team2, best_of, map_name, as_of_datetime,
                                    series_state, map_state, form_state, player_roster_state,
                                    tier=None):
    """PURE / READ-ONLY. No target, no current score, no target-series lineup
    anywhere in the call chain - every delegate is independently proven pure
    in its own phase's tests. Returns exactly the 95 predictive inputs
    declared in config/map_features_v2_rich.yaml (62 directional + 30
    symmetric/confidence + 3 categorical context) - nothing else."""
    as_of = pd.Timestamp(as_of_datetime)

    v1 = build_features(series_state, team1, team2, as_of, best_of, tier=tier)
    map_specific = build_future_map_matchup_features(map_state, team1, team2, best_of, map_name, as_of, tier=tier)
    pool = build_future_series_map_pool_features(map_state, team1, team2, best_of, as_of, tier=tier)
    form = build_future_team_form_features(form_state, team1, team2, as_of)
    roster = build_future_player_roster_features(player_roster_state, team1, team2, as_of)

    out = {}
    for k in V1_SERIES_DIRECTIONAL + V1_SERIES_SYMMETRIC:
        out[k] = v1[k]
    for k in list(MAP_DIRECTIONAL_FEATURES) + list(MAP_SYMMETRIC_FEATURES):
        out[k] = map_specific[k]
    for k in list(SERIES_MAP_POOL_DIRECTIONAL) + list(SERIES_MAP_POOL_SYMMETRIC):
        out[k] = pool[k]
    for k in list(FORM_DIRECTIONAL_FEATURES) + list(FORM_SYMMETRIC_FEATURES):
        out[k] = form[k]
    for k in list(ROSTER_DIRECTIONAL_FEATURES) + list(ROSTER_SYMMETRIC_FEATURES):
        out[k] = roster[k]

    out["map_name"] = map_name
    out["bestOf"] = best_of
    out["tier"] = tier if tier is not None else UNKNOWN_TIER_CATEGORY

    expected_keys = set(RICH_MAP_DIRECTIONAL_FEATURES) | set(RICH_MAP_SYMMETRIC_FEATURES) \
        | set(RICH_MAP_CATEGORICAL_CONTEXT)
    assert set(out.keys()) == expected_keys, \
        f"composed feature dict disagrees with the declared schema: {set(out.keys()) ^ expected_keys}"
    return out
