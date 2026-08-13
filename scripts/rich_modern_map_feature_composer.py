"""
Phase 6C future-application composer for the known-map prediction task.

Thin, PURE composition over TWO already-tested future builders:
`rich_map_feature_composer.build_future_rich_map_features` (Phase 6A, 95
features, unmodified) and
`modern_map_feature_engine.build_future_modern_map_state_features` (Phase
6C, 25 features). Adds no new temporal/streaming logic of its own - every
number returned here is produced by a function already proven leakage-safe
(no target, no current score, no lineup) in its own phase's own test suite.

    series_state        : feature_engine.StateStore                    (Phase 3)
    map_state           : map_feature_engine.MapStateStore              (Phase 5A)
    form_state          : team_form_engine.TeamFormStateStore           (Phase 5B.2)
    player_roster_state : player_roster_feature_engine.PlayerRosterStateStore (Phase 5C)
    modern_map_state     : modern_map_feature_engine.ModernMapStateStore  (Phase 6C)

OVERALL-ELO SOURCE FOR SECTION 13'S MAP-SPECIALIZATION DIFFERENCES
--------------------------------------------------------------------------
modern_map_feature_engine.build_future_modern_map_state_features needs each
team's PRE-SERIES overall ELO as a plain float (never a whole StateStore -
see that module's own docstring for why it is decoupled this way). Live
application reads this from `series_state.teams[team].elo` exactly as
`feature_engine.build_features` itself does, cold-starting to `ELO_INITIAL`
for a team with no prior series history - the SAME value the offline
Phase 6C build derives from the frozen `series_team_form_states_v1.parquet`
audit artifact (exhaustively parity-checked against it before that dataset
was ever written).
"""

import pandas as pd

from feature_engine import ELO_INITIAL
from rich_map_feature_composer import (
    build_future_rich_map_features, UNKNOWN_TIER_CATEGORY,
    RICH_MAP_DIRECTIONAL_FEATURES, RICH_MAP_SYMMETRIC_FEATURES, RICH_MAP_CATEGORICAL_CONTEXT,
)
from modern_map_feature_engine import (
    build_future_modern_map_state_features,
    MODERN_MAP_DIRECTIONAL_FEATURES, MODERN_MAP_SYMMETRIC_FEATURES,
)

MODERN_RICH_MAP_DIRECTIONAL_FEATURES = list(RICH_MAP_DIRECTIONAL_FEATURES) + list(MODERN_MAP_DIRECTIONAL_FEATURES)
MODERN_RICH_MAP_SYMMETRIC_FEATURES = list(RICH_MAP_SYMMETRIC_FEATURES) + list(MODERN_MAP_SYMMETRIC_FEATURES)
MODERN_RICH_MAP_CATEGORICAL_CONTEXT = list(RICH_MAP_CATEGORICAL_CONTEXT)

assert len(MODERN_RICH_MAP_DIRECTIONAL_FEATURES) == 80
assert len(MODERN_RICH_MAP_SYMMETRIC_FEATURES) == 37
assert len(MODERN_RICH_MAP_CATEGORICAL_CONTEXT) == 3


def build_future_modern_rich_map_features(team1, team2, best_of, map_name, as_of_datetime,
                                           series_state, map_state, form_state, player_roster_state,
                                           modern_map_state, tier=None):
    """PURE / READ-ONLY. No target, no current score, no target-series lineup
    anywhere in the call chain. Returns exactly the 120 predictive inputs
    declared in config/map_features_v3_modern_map.yaml (80 directional + 37
    symmetric/confidence + 3 categorical context)."""
    as_of = pd.Timestamp(as_of_datetime)

    rich = build_future_rich_map_features(team1, team2, best_of, map_name, as_of,
                                           series_state, map_state, form_state, player_roster_state, tier=tier)

    t1 = series_state.teams.get(team1)
    t2 = series_state.teams.get(team2)
    overall_elo1 = t1.elo if t1 is not None else ELO_INITIAL
    overall_elo2 = t2.elo if t2 is not None else ELO_INITIAL

    modern = build_future_modern_map_state_features(
        map_state, modern_map_state, player_roster_state, team1, team2, map_name, as_of,
        overall_elo1, overall_elo2)

    out = dict(rich)
    out.update(modern)

    expected_keys = (set(MODERN_RICH_MAP_DIRECTIONAL_FEATURES) | set(MODERN_RICH_MAP_SYMMETRIC_FEATURES)
                     | set(MODERN_RICH_MAP_CATEGORICAL_CONTEXT))
    assert set(out.keys()) == expected_keys, \
        f"composed modern-rich feature dict disagrees with the declared schema: {set(out.keys()) ^ expected_keys}"
    return out
