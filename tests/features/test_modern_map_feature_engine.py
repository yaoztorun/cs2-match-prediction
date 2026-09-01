"""
Tests for feature_engineering/maps/modern_map_feature_engine.py (Phase 6C). Synthetic fixtures
only. Covers brief section 26's checklist: authoritative cutoff, no
current-map leakage, same-series/same-timestamp isolation, 90-day decay,
Beta(2,2) smoothing, opponent residual uses the static pre-series ELO (not a
replay), trusted-opponent gating, map-specialization DIFFERENCE arithmetic
(cold start 0.0, never a ratio), roster inferred strictly prior + follows
transfers, current-roster KAST/specialization, current-core continuity,
side-swap symmetry, cold start, future-builder signature has no
target/lineup parameter.
"""

import inspect
import math

import numpy as np
import pandas as pd
import pytest

from feature_engineering.series.feature_engine import ELO_INITIAL, elo_expected
from feature_engineering.maps.map_feature_engine import MapStateStore, apply_map_result
from feature_engineering.roster.player_roster_feature_engine import PlayerRosterStateStore, apply_player_observation
from feature_engineering.maps.modern_map_feature_engine import (
    ModernMapStateStore, apply_selected_map_team_result, apply_selected_map_player_observation,
    compute_team_selected_map_recent_features, compute_map_specialization_features,
    compute_current_roster_map_features, compute_current_core_map_continuity_features,
    build_future_modern_map_state_features, process_modern_map_stream, recency_weight,
    MAP_FORM_HALF_LIFE_DAYS, MODERN_MAP_DIRECTIONAL_FEATURES, MODERN_MAP_SYMMETRIC_FEATURES,
    COLD_START_MAP_WR, COLD_START_SPECIALIZATION, COLD_START_RANK_PERCENTILE,
)


def empty_stores():
    return MapStateStore(), ModernMapStateStore(), PlayerRosterStateStore()


def team_row(t1, t2, e1, e2, map_name, dt, mid, gid, s1, s2, elo1, elo2, elo_known=True):
    return pd.Series({
        "team1_canonical": t1, "team2_canonical": t2, "team1_eligible": e1, "team2_eligible": e2,
        "map_name": map_name, "series_datetime": pd.Timestamp(dt), "map_datetime": pd.Timestamp(dt),
        "match_id": mid, "game_id": gid, "score1_game": s1, "score2_game": s2,
        "team1_map_win": 1 if s1 > s2 else 0,
        "team1_pre_series_elo": elo1, "team2_pre_series_elo": elo2, "pre_series_elo_known": elo_known,
    })


def player_row(team, pid, map_name, dt, gid, adr, kast, kdb, eligible=True, usable=True, mid=None):
    return pd.Series({
        "team_canonical": team, "player_id": pid, "map_name": map_name, "series_datetime": pd.Timestamp(dt),
        "game_id": gid, "match_id": mid if mid is not None else f"mid_{gid}",
        "adr": adr, "kast": kast, "kd_balance": kdb, "assists_per_round": 0.2,
        "team_eligible": eligible, "has_usable_stats": usable,
    })


# --- schema shape ---

def test_schema_is_exactly_25_new_features():
    assert len(MODERN_MAP_DIRECTIONAL_FEATURES) == 18
    assert len(MODERN_MAP_SYMMETRIC_FEATURES) == 7
    assert not (set(MODERN_MAP_DIRECTIONAL_FEATURES) & set(MODERN_MAP_SYMMETRIC_FEATURES))


def test_composed_output_has_exactly_the_declared_keys():
    map_state, modern_state, roster_state = empty_stores()
    out = build_future_modern_map_state_features(map_state, modern_state, roster_state,
                                                  "A", "B", "Mirage", "2025-01-01", 1500.0, 1500.0)
    expected = set(MODERN_MAP_DIRECTIONAL_FEATURES) | set(MODERN_MAP_SYMMETRIC_FEATURES)
    assert set(out.keys()) == expected


# --- 90-day recency weighting ---

def test_recency_weight_halves_every_90_days():
    as_of = pd.Timestamp("2025-04-01")
    dt = as_of - pd.Timedelta(days=MAP_FORM_HALF_LIFE_DAYS)
    assert recency_weight(as_of, dt) == pytest.approx(0.5)
    dt2 = as_of - pd.Timedelta(days=2 * MAP_FORM_HALF_LIFE_DAYS)
    assert recency_weight(as_of, dt2) == pytest.approx(0.25)
    assert recency_weight(as_of, as_of) == pytest.approx(1.0)


# --- Beta(2,2) smoothed selected-map win rate ---

def test_weighted_map_wr_uses_beta_2_2_smoothing():
    as_of = pd.Timestamp("2025-01-10")
    dt = as_of - pd.Timedelta(days=1)
    from feature_engineering.maps.modern_map_feature_engine import SelectedMapHistoryEntry
    hist = [SelectedMapHistoryEntry(series_dt=dt, match_id="m", game_id="g", win=1, normalized_margin=0.5,
                                     opponent_identity_trusted=True, own_pre_series_elo=1500.0,
                                     opponent_pre_series_elo=1500.0, expected_win_prob=0.5,
                                     performance_residual=0.5)]
    r = compute_team_selected_map_recent_features(hist, as_of)
    w = recency_weight(as_of, dt)
    expected_wr = (w * 1 + 2) / (w + 4)
    assert r["weighted_map_wr"] == pytest.approx(expected_wr)


def test_no_history_gives_documented_cold_start():
    r = compute_team_selected_map_recent_features([], pd.Timestamp("2025-01-01"))
    assert r["weighted_map_wr"] == COLD_START_MAP_WR == 0.5
    assert r["weighted_map_margin"] == 0.0
    assert r["weighted_map_performance_residual"] == 0.0
    assert r["weighted_map_opponent_elo"] == ELO_INITIAL
    assert r["map_recent_history_mass"] == 0.0
    assert r["map_adjusted_history_mass"] == 0.0


# --- authoritative cutoff / no current-map leakage ---

def test_strictly_prior_filtering_excludes_entries_at_or_after_as_of():
    from feature_engineering.maps.modern_map_feature_engine import SelectedMapHistoryEntry
    as_of = pd.Timestamp("2025-01-10")
    hist = [
        SelectedMapHistoryEntry(series_dt=as_of, match_id="m1", game_id="g1", win=1, normalized_margin=1.0,
                                 opponent_identity_trusted=True, own_pre_series_elo=1500.0,
                                 opponent_pre_series_elo=1500.0, expected_win_prob=0.5, performance_residual=0.5),
        SelectedMapHistoryEntry(series_dt=as_of + pd.Timedelta(days=1), match_id="m2", game_id="g2", win=0,
                                 normalized_margin=-1.0, opponent_identity_trusted=True,
                                 own_pre_series_elo=1500.0, opponent_pre_series_elo=1500.0,
                                 expected_win_prob=0.5, performance_residual=-0.5),
    ]
    r = compute_team_selected_map_recent_features(hist, as_of)
    # neither entry (at or after as_of) contributes -> exactly the cold-start values
    assert r["weighted_map_wr"] == 0.5
    assert r["map_recent_history_mass"] == 0.0


def test_future_builder_signature_has_no_target_score_or_lineup_parameter():
    params = set(inspect.signature(build_future_modern_map_state_features).parameters)
    forbidden = {"target", "winner", "score1", "score2", "team1_win", "team1_map_win",
                 "lineup", "players", "roster", "team1_players", "team2_players"}
    assert not (params & forbidden), f"leaked forbidden parameter: {params & forbidden}"
    assert params == {"map_state", "modern_state", "player_roster_state", "team1", "team2", "map_name",
                       "as_of_datetime", "team1_overall_elo", "team2_overall_elo"}


# --- opponent residual uses the STATIC pre-series ELO, never a replay ---

def test_performance_residual_uses_static_pre_series_elo_not_a_running_scalar():
    modern = ModernMapStateStore()
    t = pd.Timestamp("2025-01-01")
    row = team_row("A", "B", True, True, "Mirage", t, "m1", "g1", 13, 5, elo1=1700.0, elo2=1400.0)
    apply_selected_map_team_result(modern, row)
    hist = modern.get_team_map("A", "Mirage")
    expected = elo_expected(1700.0, 1400.0)
    assert hist[0].own_pre_series_elo == 1700.0
    assert hist[0].opponent_pre_series_elo == 1400.0
    assert hist[0].expected_win_prob == pytest.approx(expected)
    assert hist[0].performance_residual == pytest.approx(1 - expected)   # team1 won


def test_missing_pre_series_elo_never_enters_the_trusted_population():
    """A row flagged pre_series_elo_known=False must never contaminate the
    trusted-only aggregates, even if both sides are otherwise eligible."""
    modern = ModernMapStateStore()
    t0 = pd.Timestamp("2025-01-01")
    bad_row = team_row("A", "B", True, True, "Mirage", t0, "m0", "g0", 13, 2,
                        elo1=1500.0, elo2=1500.0, elo_known=False)
    apply_selected_map_team_result(modern, bad_row)
    r = compute_team_selected_map_recent_features(modern.get_team_map("A", "Mirage"), t0 + pd.Timedelta(days=1))
    # own win/margin DID update (own fact, matches map_feature_engine's own identity policy)...
    assert r["map_recent_history_mass"] > 0.0
    # ...but the untrustworthy entry contributes nothing to the opponent-adjusted population
    assert r["map_adjusted_history_mass"] == 0.0
    assert r["weighted_map_performance_residual"] == 0.0
    assert r["weighted_map_opponent_elo"] == ELO_INITIAL


# --- trusted-opponent gating (own history vs opponent-adjusted history) ---

def test_own_history_uses_all_eligible_but_adjusted_uses_trusted_only():
    modern = ModernMapStateStore()
    t0 = pd.Timestamp("2025-01-01")
    # untrusted opponent (team2 not identity-eligible) - team1's own win/margin should still count
    r1 = team_row("A", "Z", True, False, "Mirage", t0, "m1", "g1", 13, 2, elo1=1600.0, elo2=1500.0)
    apply_selected_map_team_result(modern, r1)
    # trusted opponent
    r2 = team_row("A", "B", True, True, "Mirage", t0 + pd.Timedelta(days=1), "m2", "g2", 13, 10,
                   elo1=1600.0, elo2=1550.0)
    apply_selected_map_team_result(modern, r2)

    as_of = t0 + pd.Timedelta(days=5)
    feats = compute_team_selected_map_recent_features(modern.get_team_map("A", "Mirage"), as_of)
    # both own-eligible entries feed weighted_map_wr/margin/mass ...
    w1 = recency_weight(as_of, t0)
    w2 = recency_weight(as_of, t0 + pd.Timedelta(days=1))
    assert feats["map_recent_history_mass"] == pytest.approx(w1 + w2)
    # ...but only the trusted one feeds the opponent-adjusted mass
    assert feats["map_adjusted_history_mass"] == pytest.approx(w2)


# --- same-series and same-timestamp isolation (two-phase driver) ---

def test_map1_cannot_see_map2_of_the_same_series():
    map_state, modern_state, roster_state = empty_stores()
    t0 = pd.Timestamp("2025-01-01")
    rows = pd.DataFrame([
        team_row("A", "B", True, True, "Mirage", t0, "m1", "g1", 13, 5, 1500.0, 1500.0),
        team_row("A", "B", True, True, "Mirage", t0, "m1", "g2", 13, 5, 1500.0, 1500.0),
    ])
    out = process_modern_map_stream(map_state, modern_state, roster_state, rows, emit_features=True)
    out_df = pd.DataFrame(out).sort_values("game_id")
    # both maps of the SAME series/timestamp saw IDENTICAL pre-batch state
    assert out_df.iloc[0]["map_recent_history_mass_diff"] == out_df.iloc[1]["map_recent_history_mass_diff"] == 0.0
    assert out_df.iloc[0]["time_weighted_map_wr_diff"] == out_df.iloc[1]["time_weighted_map_wr_diff"] == 0.0


def test_two_simultaneous_series_cannot_see_each_other():
    map_state, modern_state, roster_state = empty_stores()
    t0 = pd.Timestamp("2025-01-01")
    rows = pd.DataFrame([
        team_row("A", "B", True, True, "Mirage", t0, "m1", "g1", 13, 1, 1500.0, 1500.0),
        team_row("C", "D", True, True, "Mirage", t0, "m2", "g2", 13, 1, 1500.0, 1500.0),
    ])
    out = process_modern_map_stream(map_state, modern_state, roster_state, rows, emit_features=True)
    out_df = pd.DataFrame(out)
    # neither series saw the other's simultaneous blowout
    assert (out_df["map_recent_history_mass_diff"] == 0.0).all()


def test_later_map_sees_earlier_maps_state():
    """Sanity check the driver is not TOO isolating: a later series_datetime
    batch DOES see the results of an earlier one. modern_state/roster_state
    are order-independent (see module docstring) and are built fully upfront
    - exactly as feature_engineering/maps/build_map_features_v3_modern_map.py does - before
    process_modern_map_stream is called; only map_state is advanced by that
    driver itself."""
    map_state, modern_state, roster_state = empty_stores()
    t0 = pd.Timestamp("2025-01-01")
    t1 = pd.Timestamp("2025-02-01")
    rows = pd.DataFrame([
        team_row("A", "B", True, True, "Mirage", t0, "m1", "g1", 13, 1, 1500.0, 1500.0),
        team_row("A", "C", True, True, "Mirage", t1, "m2", "g2", 13, 1, 1500.0, 1500.0),
    ])
    for _, r in rows.iterrows():
        apply_selected_map_team_result(modern_state, r)
    out = process_modern_map_stream(map_state, modern_state, roster_state, rows, emit_features=True)
    out_df = pd.DataFrame(out).set_index("game_id")
    assert out_df.loc["g1", "map_recent_history_mass_diff"] == 0.0     # both cold
    assert out_df.loc["g2", "map_recent_history_mass_diff"] != 0.0     # A now has Mirage history, C does not


# --- map specialization: DIFFERENCES only, cold start 0.0/0.5, never ratios ---

def test_map_specialization_is_a_difference_never_a_ratio():
    map_state = MapStateStore()
    t0 = pd.Timestamp("2025-01-01")
    apply_map_result(map_state, team_row("A", "Z", True, True, "Mirage", t0, "m1", "g1", 16, 4,
                                          1500.0, 1500.0) .drop(["team1_pre_series_elo", "team2_pre_series_elo",
                                                                  "pre_series_elo_known"]))
    as_of = t0 + pd.Timedelta(days=1)
    s = compute_map_specialization_features(map_state, "A", "Mirage", overall_elo=1500.0,
                                             weighted_map_wr=0.5, as_of=as_of)
    from feature_engineering.maps.map_feature_engine import compute_team_map_features
    map_elo = compute_team_map_features(map_state.get("A", "Mirage"), as_of)["map_elo"]
    assert s["elo_vs_overall"] == pytest.approx(map_elo - 1500.0)     # subtraction, not division
    assert s["elo_vs_pool_mean"] == pytest.approx(map_elo - map_elo)  # only map in the pool -> pool mean == map_elo


def test_map_specialization_cold_start_is_neutral_zero_and_half():
    map_state = MapStateStore()
    s = compute_map_specialization_features(map_state, "A", "Mirage", overall_elo=1650.0,
                                             weighted_map_wr=0.5, as_of=pd.Timestamp("2025-01-01"))
    assert s["elo_vs_overall"] == pytest.approx(ELO_INITIAL - 1650.0)   # map_elo cold start (1500) - overall
    assert s["elo_vs_pool_mean"] == 0.0     # empty pool -> both default to 1500.0
    assert s["wr_vs_pool_mean"] == 0.0      # 0.5 - 0.5
    assert s["rank_percentile"] == COLD_START_RANK_PERCENTILE == 0.5
    assert s["in_pool"] is False


def test_rank_percentile_best_map_is_near_one():
    map_state = MapStateStore()
    t0 = pd.Timestamp("2025-01-01")
    # A plays two maps: Mirage (dominant) and Nuke (weak) within pool lookback
    a_row = lambda name, dt, s1, s2, gid: team_row("A", "Z", True, True, name, dt, f"m{gid}", gid, s1, s2,
                                                    1500.0, 1500.0).drop(
        ["team1_pre_series_elo", "team2_pre_series_elo", "pre_series_elo_known"])
    apply_map_result(map_state, a_row("Mirage", t0, 16, 2, "g1"))
    apply_map_result(map_state, a_row("Nuke", t0 + pd.Timedelta(days=1), 2, 16, "g2"))
    as_of = t0 + pd.Timedelta(days=5)
    s = compute_map_specialization_features(map_state, "A", "Mirage", overall_elo=1500.0,
                                             weighted_map_wr=0.9, as_of=as_of)
    assert s["in_pool"] is True
    assert s["rank_percentile"] == 1.0   # Mirage's ELO rose (win), Nuke's fell (loss) -> Mirage ranks best


# --- current-roster inferred strictly prior + follows transfers ---

def test_roster_inferred_strictly_prior_and_player_history_follows_transfers():
    roster_state = PlayerRosterStateStore()
    modern = ModernMapStateStore()
    t0 = pd.Timestamp("2025-01-01")
    for i in range(3):
        dt = t0 + pd.Timedelta(days=i)
        prow = player_row("A", 42, "Mirage", dt, f"g{i}", adr=90.0, kast=75.0, kdb=0.3)
        apply_player_observation(roster_state, prow)
        apply_selected_map_player_observation(modern, prow)
    # player 42 transfers to team B AFTER their Team-A history
    t_transfer = t0 + pd.Timedelta(days=10)
    prow2 = player_row("B", 42, "Mirage", t_transfer, "g10", adr=100.0, kast=80.0, kdb=0.4)
    apply_player_observation(roster_state, prow2)
    apply_selected_map_player_observation(modern, prow2)

    as_of_before = t0 + pd.Timedelta(days=1)   # only the FIRST Team-A appearance is strictly prior
    c = compute_current_roster_map_features(roster_state, modern, "A", "Mirage", as_of_before)
    assert c["roster_map_players_with_history"] <= 1   # at most one prior map for player 42 by day 1

    # player 42's GLOBAL map-specific history (queried by player_id) follows them to Team B
    as_of_after = t_transfer + pd.Timedelta(days=1)
    b_roster = compute_current_roster_map_features(roster_state, modern, "B", "Mirage", as_of_after)
    assert b_roster["roster_map_players_with_history"] >= 1


def test_roster_never_uses_the_target_series_lineup():
    """Purely a signature check: the current-roster query never takes a
    lineup argument - it can only see what infer_expected_roster already
    proved is strictly-prior team appearance history."""
    params = set(inspect.signature(compute_current_roster_map_features).parameters)
    assert "lineup" not in params and "players" not in params


# --- current-roster KAST specialization: difference, neutral 0.0 cold start ---

def test_kast_specialization_is_a_difference_with_zero_cold_start():
    roster_state = PlayerRosterStateStore()
    modern = ModernMapStateStore()
    t0 = pd.Timestamp("2025-01-01")
    for i in range(3):
        dt = t0 + pd.Timedelta(days=i)
        # global KAST (all maps) = 70; selected-map (Mirage) KAST = 85 -> specialization should be +15
        g_row = player_row("A", 7, "Mirage" if i == 0 else "Nuke", dt, f"g{i}", adr=80, kast=70 if i else 85,
                            kdb=0.1)
        apply_player_observation(roster_state, g_row)
        if i == 0:
            apply_selected_map_player_observation(modern, g_row)
    as_of = t0 + pd.Timedelta(days=5)
    c = compute_current_roster_map_features(roster_state, modern, "A", "Mirage", as_of)
    assert c["roster_map_kast_specialization"] != 0.0   # real signal present

    # cold-start branch: brand new team with no roster at all
    c_cold = compute_current_roster_map_features(roster_state, modern, "NEWTEAM", "Mirage", as_of)
    assert c_cold["roster_map_kast_specialization"] == COLD_START_SPECIALIZATION == 0.0
    assert not math.isnan(c_cold["roster_map_kast_specialization"])   # never NaN, per the approval correction


def test_roster_performance_diffs_are_nan_only_when_no_roster_evidence():
    map_state, modern_state, roster_state = empty_stores()
    out = build_future_modern_map_state_features(map_state, modern_state, roster_state,
                                                  "NEW1", "NEW2", "Mirage", "2025-01-01", 1500.0, 1500.0)
    for k in ["roster_map_mean_kast_diff", "roster_map_bottom_kast_diff",
              "roster_map_mean_adr_diff", "roster_map_mean_kd_balance_diff"]:
        assert isinstance(out[k], float) and math.isnan(out[k])
    assert out["roster_map_players_with_history_min"] == 0
    # never NaN: the specialization diff and every mass/count feature
    assert out["roster_map_kast_specialization_diff"] == 0.0
    assert out["roster_map_mean_history_mass_diff"] == 0.0


# --- current-core continuity ---

def test_current_core_continuity_reflects_roster_overlap():
    roster_state = PlayerRosterStateStore()
    modern = ModernMapStateStore()
    t0 = pd.Timestamp("2025-01-01")
    # 5-player core plays Mirage together 3 times
    for i in range(3):
        dt = t0 + pd.Timedelta(days=i)
        for pid in [1, 2, 3, 4, 5]:
            g_row = player_row("A", pid, "Mirage", dt, f"g{i}", adr=80, kast=70, kdb=0.1)
            apply_player_observation(roster_state, g_row)
            apply_selected_map_player_observation(modern, g_row)
    as_of = t0 + pd.Timedelta(days=10)
    c = compute_current_core_map_continuity_features(roster_state, modern, "A", "Mirage", as_of)
    assert c["continuity"] == pytest.approx(1.0)   # the current roster IS exactly who played every prior map
    assert c["mass"] > 0.0


def test_current_core_continuity_cold_start_is_zero():
    roster_state = PlayerRosterStateStore()
    modern = ModernMapStateStore()
    c = compute_current_core_map_continuity_features(roster_state, modern, "NEWTEAM", "Mirage",
                                                       pd.Timestamp("2025-01-01"))
    assert c["continuity"] == 0.0
    assert c["mass"] == 0.0


# --- side-swap symmetry of the fully composed 25-feature block ---

def _seed_asymmetric():
    map_state, modern_state, roster_state = empty_stores()
    t0 = pd.Timestamp("2025-01-01")
    for i in range(3):
        dt = t0 + pd.Timedelta(days=i)
        row_a = team_row("A", "Z", True, True, "Mirage", dt, f"am{i}", f"ag{i}", 13, 4, 1650.0, 1500.0)
        apply_selected_map_team_result(modern_state, row_a)
        apply_map_result(map_state, row_a.drop(["team1_pre_series_elo", "team2_pre_series_elo",
                                                  "pre_series_elo_known"]))
        for pid in [1, 2, 3, 4, 5]:
            p = player_row("A", pid, "Mirage", dt, f"ag{i}", adr=90, kast=78, kdb=0.2)
            apply_player_observation(roster_state, p)
            apply_selected_map_player_observation(modern_state, p)
    for i in range(2):
        dt = t0 + pd.Timedelta(days=i)
        row_b = team_row("B", "Z", True, True, "Mirage", dt, f"bm{i}", f"bg{i}", 4, 13, 1420.0, 1500.0)
        apply_selected_map_team_result(modern_state, row_b)
        apply_map_result(map_state, row_b.drop(["team1_pre_series_elo", "team2_pre_series_elo",
                                                  "pre_series_elo_known"]))
        for pid in [11, 12, 13, 14, 15]:
            p = player_row("B", pid, "Mirage", dt, f"bg{i}", adr=60, kast=60, kdb=-0.1)
            apply_player_observation(roster_state, p)
            apply_selected_map_player_observation(modern_state, p)
    return map_state, modern_state, roster_state


def test_directional_negate_and_symmetric_unchanged_under_side_swap():
    map_state, modern_state, roster_state = _seed_asymmetric()
    as_of = pd.Timestamp("2025-02-01")
    ab = build_future_modern_map_state_features(map_state, modern_state, roster_state,
                                                 "A", "B", "Mirage", as_of, 1650.0, 1420.0)
    ba = build_future_modern_map_state_features(map_state, modern_state, roster_state,
                                                 "B", "A", "Mirage", as_of, 1420.0, 1650.0)
    nontrivial = 0
    for k in MODERN_MAP_DIRECTIONAL_FEATURES:
        x, y = ab[k], ba[k]
        if isinstance(x, float) and math.isnan(x):
            assert isinstance(y, float) and math.isnan(y), k
            continue
        assert x == pytest.approx(-y, abs=1e-9), f"{k} did not negate under side swap"
        if abs(x) > 1e-9:
            nontrivial += 1
    assert nontrivial >= 8, "fixture too symmetric to prove anything"

    for k in MODERN_MAP_SYMMETRIC_FEATURES:
        x, y = ab[k], ba[k]
        if isinstance(x, float) and math.isnan(x):
            assert isinstance(y, float) and math.isnan(y), k
        else:
            assert x == pytest.approx(y), f"{k} changed under side swap"


def test_composer_never_mutates_any_store():
    map_state, modern_state, roster_state = _seed_asymmetric()
    before = (len(map_state.states), len(modern_state.team_map), len(modern_state.player_map),
              len(modern_state.team_map_roster), len(roster_state.teams))
    build_future_modern_map_state_features(map_state, modern_state, roster_state,
                                            "A", "B", "Mirage", "2025-02-01", 1650.0, 1420.0)
    after = (len(map_state.states), len(modern_state.team_map), len(modern_state.player_map),
             len(modern_state.team_map_roster), len(roster_state.teams))
    assert before == after
