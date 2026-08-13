"""
Tests for scripts/rich_map_feature_composer.py (Phase 6A). Synthetic
fixtures only - every delegate this module calls is already proven pure and
leakage-safe in its own phase's test suite; these tests cover only the NEW
composition logic (schema assembly, tier contract, NaN passthrough,
side-swap symmetry of the composed output).
"""

import inspect
import math

import pandas as pd
import pytest

from feature_engine import StateStore, HistoryEntry, TeamState, ELO_INITIAL
from map_feature_engine import MapStateStore, apply_map_result
from team_form_engine import TeamFormStateStore, apply_form_result
from player_roster_feature_engine import PlayerRosterStateStore, apply_player_observation
from rich_map_feature_composer import (
    build_future_rich_map_features, UNKNOWN_TIER_CATEGORY,
    RICH_MAP_DIRECTIONAL_FEATURES, RICH_MAP_SYMMETRIC_FEATURES, RICH_MAP_CATEGORICAL_CONTEXT,
)


def empty_stores():
    return StateStore(), MapStateStore(), TeamFormStateStore(), PlayerRosterStateStore()


# --- schema shape ---

def test_schema_is_exactly_95_predictive_inputs():
    assert len(RICH_MAP_DIRECTIONAL_FEATURES) == 62
    assert len(RICH_MAP_SYMMETRIC_FEATURES) == 30
    assert len(RICH_MAP_CATEGORICAL_CONTEXT) == 3
    total = len(RICH_MAP_DIRECTIONAL_FEATURES) + len(RICH_MAP_SYMMETRIC_FEATURES) + len(RICH_MAP_CATEGORICAL_CONTEXT)
    assert total == 95
    assert not (set(RICH_MAP_DIRECTIONAL_FEATURES) & set(RICH_MAP_SYMMETRIC_FEATURES))


def test_composed_output_has_exactly_the_declared_keys():
    series_state, map_state, form_state, player_roster_state = empty_stores()
    out = build_future_rich_map_features("A", "B", 3, "Mirage", "2025-01-01",
                                          series_state, map_state, form_state, player_roster_state)
    expected = set(RICH_MAP_DIRECTIONAL_FEATURES) | set(RICH_MAP_SYMMETRIC_FEATURES) | set(RICH_MAP_CATEGORICAL_CONTEXT)
    assert set(out.keys()) == expected


# --- tier contract (correction #2) ---

def test_missing_tier_emits_unknown_tier_category_not_a_real_tier():
    series_state, map_state, form_state, player_roster_state = empty_stores()
    out = build_future_rich_map_features("A", "B", 3, "Mirage", "2025-01-01",
                                          series_state, map_state, form_state, player_roster_state, tier=None)
    assert out["tier"] == UNKNOWN_TIER_CATEGORY
    assert out["tier"] not in {"tier1", "tier2", "tier3"}


def test_real_tier_is_passed_through_unchanged():
    series_state, map_state, form_state, player_roster_state = empty_stores()
    out = build_future_rich_map_features("A", "B", 3, "Mirage", "2025-01-01",
                                          series_state, map_state, form_state, player_roster_state, tier="tier2")
    assert out["tier"] == "tier2"


# --- NaN contract (correction #3): permitted, never forced to finite ---

def test_cold_start_output_carries_documented_nan_not_fabricated_values():
    series_state, map_state, form_state, player_roster_state = empty_stores()
    out = build_future_rich_map_features("NEW1", "NEW2", 3, "Mirage", "2025-01-01",
                                          series_state, map_state, form_state, player_roster_state)
    # documented NaN-capable features on a fully cold-start pair
    for k in ["days_since_last_match_diff", "days_since_map_played_diff",
              "roster_mean_adr_diff", "roster_mean_kast_diff", "roster_mean_kd_balance_diff"]:
        assert isinstance(out[k], float) and math.isnan(out[k]), f"{k} should be genuinely missing, not fabricated"
    # paired confidence features must explain every NaN
    assert out["both_teams_have_map_history"] == 0
    assert out["both_teams_have_history"] == 0
    assert out["roster_form_players_min"] == 0
    # non-NaN-capable features remain real numbers (neutral cold-start, not NaN)
    assert out["map_elo_diff"] == 0.0
    assert out["elo_diff"] == 0.0


def test_composer_never_fabricates_a_population_mean():
    """With real (non-cold-start) history, the roster performance diffs must
    be finite - but this must come from actual evidence, not from the
    composer silently substituting a neutral/mean value for missing players."""
    series_state, map_state, form_state, player_roster_state = empty_stores()
    t = pd.Timestamp("2025-01-01")
    for i in range(3):
        for s, pid in enumerate([1, 2, 3, 4, 5]):
            row = pd.Series({
                "match_id": f"M{i}", "game_id": f"g{i}", "series_datetime": t + pd.Timedelta(days=i),
                "team_canonical": "A", "team_eligible": True,
                "player_id": pid, "adr": 80.0, "kast": 72.0, "kd_balance": 0.1, "assists_per_round": 0.2,
                "has_usable_stats": True,
            })
            apply_player_observation(player_roster_state, row)
    out = build_future_rich_map_features("A", "B", 3, "Mirage", "2025-02-01",
                                          series_state, map_state, form_state, player_roster_state)
    assert out["roster_form_players_min"] == 0  # B has no history, so team-min is still 0
    assert math.isnan(out["roster_mean_adr_diff"])  # NaN because roster_form_players_min == 0 on B's side


# --- side-swap symmetry of the composed output ---

def _seed_asymmetric():
    series_state, map_state, form_state, player_roster_state = empty_stores()
    t0 = pd.Timestamp("2025-01-01")

    # series-level (Phase 3) state: A stronger than B
    series_state.teams["A"] = TeamState(canonical_name="A", elo=1650.0, history=[
        HistoryEntry(dt=t0, source="s", source_match_id="1", canonical_match_uid="s:1",
                     opponent_canonical="Z", opponent_identity_trusted=True, best_of=3, win=1, margin=2)])
    series_state.teams["B"] = TeamState(canonical_name="B", elo=1420.0, history=[
        HistoryEntry(dt=t0, source="s", source_match_id="2", canonical_match_uid="s:2",
                     opponent_canonical="Z", opponent_identity_trusted=True, best_of=3, win=0, margin=-2)])

    # map-level (Phase 5A) + form (5B.2) + roster (5C) state via their own real appliers
    map_row = lambda dt, gid, mid, t1, t2, s1, s2, mp: pd.Series({
        "series_datetime": pd.Timestamp(dt), "map_datetime": pd.Timestamp(dt), "match_id": mid, "game_id": gid,
        "map_name": mp, "team1_canonical": t1, "team2_canonical": t2, "score1_game": s1, "score2_game": s2,
        "team1_map_win": 1 if s1 > s2 else 0, "team1_eligible": True, "team2_eligible": True,
    })
    for i in range(3):
        apply_map_result(map_state, map_row(t0 + pd.Timedelta(days=i), f"mg{i}", f"MM{i}", "A", "Z", 13, 6, "Mirage"))
    for i in range(2):
        apply_map_result(map_state, map_row(t0 + pd.Timedelta(days=i), f"bg{i}", f"BB{i}", "B", "Z", 8, 13, "Mirage"))

    form_row = lambda dt, uid, t1, t2, s1, s2: pd.Series({
        "datetime": pd.Timestamp(dt), "source": "s", "source_match_id": uid, "canonical_match_uid": f"s:{uid}",
        "team1_canonical": t1, "team2_canonical": t2, "best_of": 3, "tier": "tier1", "score1": s1, "score2": s2,
        "team1_win": 1 if s1 > s2 else 0, "team1_eligible": True, "team2_eligible": True,
    })
    for i in range(3):
        apply_form_result(form_state, form_row(t0 + pd.Timedelta(days=i), f"f{i}", "A", "Z", 2, 0))
    for i in range(2):
        apply_form_result(form_state, form_row(t0 + pd.Timedelta(days=i), f"g{i}", "B", "Z", 0, 2))

    for i in range(3):
        for s, pid in enumerate([1, 2, 3, 4, 5]):
            apply_player_observation(player_roster_state, pd.Series({
                "match_id": f"MM{i}", "game_id": f"mg{i}", "series_datetime": t0 + pd.Timedelta(days=i),
                "team_canonical": "A", "team_eligible": True, "player_id": pid,
                "adr": 90.0, "kast": 75.0, "kd_balance": 0.2, "assists_per_round": 0.2, "has_usable_stats": True,
            }))
    for i in range(2):
        for s, pid in enumerate([11, 12, 13, 14, 15]):
            apply_player_observation(player_roster_state, pd.Series({
                "match_id": f"BB{i}", "game_id": f"bg{i}", "series_datetime": t0 + pd.Timedelta(days=i),
                "team_canonical": "B", "team_eligible": True, "player_id": pid,
                "adr": 60.0, "kast": 62.0, "kd_balance": -0.1, "assists_per_round": 0.1, "has_usable_stats": True,
            }))

    return series_state, map_state, form_state, player_roster_state


def test_directional_negate_and_symmetric_unchanged_under_side_swap():
    series_state, map_state, form_state, player_roster_state = _seed_asymmetric()
    as_of = pd.Timestamp("2025-02-01")
    ab = build_future_rich_map_features("A", "B", 3, "Mirage", as_of,
                                         series_state, map_state, form_state, player_roster_state, tier="tier1")
    ba = build_future_rich_map_features("B", "A", 3, "Mirage", as_of,
                                         series_state, map_state, form_state, player_roster_state, tier="tier1")

    nontrivial = 0
    for k in RICH_MAP_DIRECTIONAL_FEATURES:
        x, y = ab[k], ba[k]
        if isinstance(x, float) and math.isnan(x):
            assert isinstance(y, float) and math.isnan(y), k
            continue
        assert x == pytest.approx(-y, abs=1e-9), f"{k} did not negate under side swap"
        if abs(x) > 1e-9:
            nontrivial += 1
    assert nontrivial >= 10, "fixture too symmetric to prove anything"

    for k in RICH_MAP_SYMMETRIC_FEATURES:
        x, y = ab[k], ba[k]
        if isinstance(x, float) and math.isnan(x):
            assert isinstance(y, float) and math.isnan(y), k
        else:
            assert x == pytest.approx(y), f"{k} changed under side swap"

    assert ab["map_name"] == ba["map_name"] == "Mirage"
    assert ab["bestOf"] == ba["bestOf"] == 3
    assert ab["tier"] == ba["tier"] == "tier1"


# --- future-application contract: no target/score/lineup ---

def test_signature_has_no_target_score_or_lineup_parameter():
    params = set(inspect.signature(build_future_rich_map_features).parameters)
    forbidden = {"target", "winner", "score1", "score2", "team1_win", "team1_series_win", "team1_map_win",
                 "lineup", "players", "roster", "team1_players", "team2_players"}
    assert not (params & forbidden), f"composer signature leaked a forbidden parameter: {params & forbidden}"
    assert params == {"team1", "team2", "best_of", "map_name", "as_of_datetime",
                      "series_state", "map_state", "form_state", "player_roster_state", "tier"}


def test_composer_never_mutates_any_of_the_four_stores():
    series_state, map_state, form_state, player_roster_state = _seed_asymmetric()
    before = (len(series_state.teams["A"].history), len(map_state.states),
              len(form_state.teams["A"].history), len(player_roster_state.teams["A"].appearances))
    build_future_rich_map_features("A", "B", 3, "Mirage", "2025-02-01",
                                    series_state, map_state, form_state, player_roster_state)
    after = (len(series_state.teams["A"].history), len(map_state.states),
             len(form_state.teams["A"].history), len(player_roster_state.teams["A"].appearances))
    assert before == after
