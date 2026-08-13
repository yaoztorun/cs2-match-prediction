"""
Tests for scripts/player_roster_feature_engine.py (Phase 5C). Synthetic
fixtures only - no dependency on real data (real-data checks live in
scripts/validate_phase5c.py), matching this repo's convention for feature
engines (tests/test_map_feature_engine.py, tests/test_team_form_engine.py).
"""

import math

import numpy as np
import pandas as pd
import pytest

from player_roster_feature_engine import (
    PlayerRosterStateStore, apply_player_observation, process_player_roster_stream,
    compute_player_form, compute_team_roster_features, infer_expected_roster,
    build_future_player_roster_features, recency_weight,
    ROSTER_DIRECTIONAL_FEATURES, ROSTER_SYMMETRIC_FEATURES, ROSTER_PERFORMANCE_DIFFS,
    ROSTER_LOOKBACK_DAYS, PLAYER_FORM_HALF_LIFE_DAYS, ROSTER_SIZE,
    COLD_START_HISTORY_MASS,
)


def mk_obs(series_dt, game_id, match_id, side, slot, team, player_id,
           adr=75.0, kast=70.0, kd_balance=0.05, apr=0.15,
           team_eligible=True, has_usable_stats=True, map_dt=None):
    return {
        "match_id": str(match_id), "game_id": str(game_id),
        "series_datetime": pd.Timestamp(series_dt),
        "map_datetime": pd.Timestamp(map_dt if map_dt is not None else series_dt),
        "side": side, "slot": slot, "team_canonical": team, "team_eligible": team_eligible,
        "player_id": int(player_id), "adr": adr, "kast": kast,
        "kd_balance": kd_balance, "assists_per_round": apr,
        "has_usable_stats": has_usable_stats,
    }


def mk_map(series_dt, game_id, match_id, team_a, players_a, team_b, players_b, **kw):
    """One full map: 5v5 (or fewer) observations."""
    rows = []
    for slot, pid in enumerate(players_a, start=1):
        rows.append(mk_obs(series_dt, game_id, match_id, 1, slot, team_a, pid, **kw))
    for slot, pid in enumerate(players_b, start=1):
        rows.append(mk_obs(series_dt, game_id, match_id, 2, slot, team_b, pid, **kw))
    return rows


def mk_request(series_dt, match_id, t1, t2, t1_elig=True, t2_elig=True):
    return {"match_id": str(match_id), "series_datetime": pd.Timestamp(series_dt),
            "team1_canonical": t1, "team2_canonical": t2,
            "team1_eligible": t1_elig, "team2_eligible": t2_elig}


def seed(store, rows):
    for r in rows:
        apply_player_observation(store, pd.Series(r))
    return store


# ===========================================================================
# A. No current-series lineup or player-stat leakage  (brief section 13)
# ===========================================================================

def test_target_series_actual_roster_does_not_change_its_own_features():
    """THE decisive leakage test: replacing the target series' actual five
    players with five completely different players must not move a single
    emitted pre-match feature."""
    hist = []
    for i in range(6):
        hist += mk_map(f"2025-0{i+1}-01", f"h{i}", f"H{i}", "A", [1, 2, 3, 4, 5], "B", [11, 12, 13, 14, 15])

    target_real = mk_map("2025-08-01", "t1", "T", "A", [1, 2, 3, 4, 5], "B", [11, 12, 13, 14, 15])
    target_swapped = mk_map("2025-08-01", "t1", "T", "A", [901, 902, 903, 904, 905],
                             "B", [911, 912, 913, 914, 915])
    reqs = pd.DataFrame([mk_request("2025-08-01", "T", "A", "B")])

    out_real = process_player_roster_stream(
        PlayerRosterStateStore(), pd.DataFrame(hist + target_real), reqs)
    out_swapped = process_player_roster_stream(
        PlayerRosterStateStore(), pd.DataFrame(hist + target_swapped), reqs)

    assert len(out_real) == len(out_swapped) == 1
    keys = ROSTER_DIRECTIONAL_FEATURES + ROSTER_SYMMETRIC_FEATURES
    for k in keys:
        a, b = out_real[0][k], out_swapped[0][k]
        if isinstance(a, float) and math.isnan(a):
            assert isinstance(b, float) and math.isnan(b), k
        else:
            assert a == pytest.approx(b), k


def test_target_series_player_stats_do_not_change_its_own_features():
    hist = [r for i in range(4)
            for r in mk_map(f"2025-0{i+1}-01", f"h{i}", f"H{i}", "A", [1, 2, 3, 4, 5],
                             "B", [11, 12, 13, 14, 15])]
    reqs = pd.DataFrame([mk_request("2025-08-01", "T", "A", "B")])

    tgt_normal = mk_map("2025-08-01", "t1", "T", "A", [1, 2, 3, 4, 5], "B", [11, 12, 13, 14, 15],
                         adr=75.0, kast=70.0)
    tgt_extreme = mk_map("2025-08-01", "t1", "T", "A", [1, 2, 3, 4, 5], "B", [11, 12, 13, 14, 15],
                          adr=180.0, kast=99.0, kd_balance=0.9, apr=1.0)

    a = process_player_roster_stream(PlayerRosterStateStore(), pd.DataFrame(hist + tgt_normal), reqs)
    b = process_player_roster_stream(PlayerRosterStateStore(), pd.DataFrame(hist + tgt_extreme), reqs)
    for k in ROSTER_DIRECTIONAL_FEATURES + ROSTER_SYMMETRIC_FEATURES:
        x, y = a[0][k], b[0][k]
        if isinstance(x, float) and math.isnan(x):
            assert math.isnan(y), k
        else:
            assert x == pytest.approx(y), k


def test_future_builder_signature_takes_no_target_score_or_lineup():
    import inspect
    params = set(inspect.signature(build_future_player_roster_features).parameters)
    forbidden = {"target", "winner", "score1", "score2", "team1_win", "team1_series_win",
                 "lineup", "players", "roster", "team1_players", "team2_players"}
    assert not (params & forbidden), f"future API leaked a forbidden parameter: {params & forbidden}"
    assert params == {"store", "team1", "team2", "as_of_datetime"}


def test_feature_computation_never_mutates_state():
    store = seed(PlayerRosterStateStore(),
                 mk_map("2025-01-01", "g0", "M0", "A", [1, 2, 3, 4, 5], "B", [11, 12, 13, 14, 15]))
    before_players = {p: len(s.history) for p, s in store.players.items()}
    before_teams = {t: len(s.appearances) for t, s in store.teams.items()}
    build_future_player_roster_features(store, "A", "ZZZ", "2026-01-01")
    assert {p: len(s.history) for p, s in store.players.items()} == before_players
    assert {t: len(s.appearances) for t, s in store.teams.items()} == before_teams


# ===========================================================================
# B. Same-series and same-timestamp isolation  (brief sections 3, 4, 5)
# ===========================================================================

def test_maps_of_one_series_do_not_leak_into_that_series_features():
    """A BO3's three maps share one authoritative cutoff; none may inform the
    series' own pre-match features."""
    rows = (mk_map("2025-06-01", "g1", "M1", "A", [1, 2, 3, 4, 5], "B", [11, 12, 13, 14, 15])
            + mk_map("2025-06-01", "g2", "M1", "A", [1, 2, 3, 4, 5], "B", [11, 12, 13, 14, 15])
            + mk_map("2025-06-01", "g3", "M1", "A", [1, 2, 3, 4, 5], "B", [11, 12, 13, 14, 15]))
    reqs = pd.DataFrame([mk_request("2025-06-01", "M1", "A", "B")])
    out = process_player_roster_stream(PlayerRosterStateStore(), pd.DataFrame(rows), reqs)
    assert len(out) == 1
    assert out[0]["roster_size_min"] == 0                 # nothing was known beforehand
    assert out[0]["roster_form_players_min"] == 0
    assert math.isnan(out[0]["roster_mean_adr_diff"])


def test_two_series_at_the_same_instant_cannot_see_each_other():
    hist = mk_map("2025-01-01", "h0", "H0", "A", [1, 2, 3, 4, 5], "B", [11, 12, 13, 14, 15])
    same_ts = (mk_map("2025-06-01 15:00", "gx", "MX", "A", [1, 2, 3, 4, 5], "C", [21, 22, 23, 24, 25])
               + mk_map("2025-06-01 15:00", "gy", "MY", "A", [1, 2, 3, 4, 5], "D", [31, 32, 33, 34, 35]))
    reqs = pd.DataFrame([mk_request("2025-06-01 15:00", "MX", "A", "C"),
                          mk_request("2025-06-01 15:00", "MY", "A", "D")])
    out = process_player_roster_stream(
        PlayerRosterStateStore(), pd.DataFrame(hist + same_ts), reqs)
    by_id = {r["match_id"]: r for r in out}
    # Team A's inferred roster/stability must be identical for both matches:
    # neither may have seen the other's map, which happened at the same instant.
    for k in ["team1_roster_size", "team1_recent_unique_players_10_maps",
              "team1_core5_continuity_last_10", "team1_roster_mean_adr"]:
        a, b = by_id["MX"][k], by_id["MY"][k]
        if isinstance(a, float) and math.isnan(a):
            assert math.isnan(b), k
        else:
            assert a == pytest.approx(b), k


def test_row_order_within_a_timestamp_is_irrelevant():
    # history sits INSIDE the 90-day roster window, so the compared feature is
    # a real number rather than a vacuous cold-start NaN
    hist = mk_map("2025-05-01", "h0", "H0", "A", [1, 2, 3, 4, 5], "B", [11, 12, 13, 14, 15])
    grp = (mk_map("2025-06-01", "gx", "MX", "A", [1, 2, 3, 4, 5], "C", [21, 22, 23, 24, 25])
           + mk_map("2025-06-01", "gy", "MY", "A", [1, 2, 3, 4, 5], "D", [31, 32, 33, 34, 35]))
    reqs = pd.DataFrame([mk_request("2025-06-01", "MX", "A", "C"),
                          mk_request("2025-06-01", "MY", "A", "D")])
    fwd = process_player_roster_stream(PlayerRosterStateStore(), pd.DataFrame(hist + grp), reqs)
    bwd = process_player_roster_stream(PlayerRosterStateStore(),
                                        pd.DataFrame(hist + grp[::-1]), reqs)
    f = {r["match_id"]: r["team1_roster_mean_adr"] for r in fwd}
    b = {r["match_id"]: r["team1_roster_mean_adr"] for r in bwd}
    assert set(f) == set(b)
    assert not any(math.isnan(v) for v in f.values()), "fixture should exercise a real value"
    for mid in f:
        assert f[mid] == pytest.approx(b[mid])


def test_differing_map_timestamps_within_a_series_use_one_authoritative_cutoff():
    """Correction #1: map timestamps are provenance only. Three maps of one
    series carrying deliberately different map_datetime values must still
    produce the identical pre-series feature row, because the engine keys
    exclusively on the authoritative series_datetime."""
    hist = [r for i in range(3)
            for r in mk_map(f"2025-0{i+1}-01", f"h{i}", f"H{i}", "A", [1, 2, 3, 4, 5],
                             "B", [11, 12, 13, 14, 15])]
    reqs = pd.DataFrame([mk_request("2025-06-01 12:00", "M1", "A", "B")])

    aligned = (mk_map("2025-06-01 12:00", "g1", "M1", "A", [1, 2, 3, 4, 5], "B", [11, 12, 13, 14, 15])
               + mk_map("2025-06-01 12:00", "g2", "M1", "A", [1, 2, 3, 4, 5], "B", [11, 12, 13, 14, 15]))
    # same authoritative series_datetime, wildly different map_datetime values
    skewed = (mk_map("2025-06-01 12:00", "g1", "M1", "A", [1, 2, 3, 4, 5], "B", [11, 12, 13, 14, 15],
                      map_dt="2025-06-01 13:20")
              + mk_map("2025-06-01 12:00", "g2", "M1", "A", [1, 2, 3, 4, 5], "B", [11, 12, 13, 14, 15],
                        map_dt="2025-06-01 14:45"))

    a = process_player_roster_stream(PlayerRosterStateStore(), pd.DataFrame(hist + aligned), reqs)
    b = process_player_roster_stream(PlayerRosterStateStore(), pd.DataFrame(hist + skewed), reqs)
    for k in ROSTER_DIRECTIONAL_FEATURES + ROSTER_SYMMETRIC_FEATURES:
        x, y = a[0][k], b[0][k]
        if isinstance(x, float) and math.isnan(x):
            assert math.isnan(y), k
        else:
            assert x == pytest.approx(y), k


# ===========================================================================
# C. Player transfer semantics  (brief section 14)
# ===========================================================================

def test_player_transfer_team_membership_does_not_move_until_observed():
    store = PlayerRosterStateStore()
    # P (id 7) plays for A three times
    for i in range(3):
        seed(store, mk_map(f"2025-01-0{i+1}", f"a{i}", f"A{i}", "A", [7, 2, 3, 4, 5],
                            "Z", [51, 52, 53, 54, 55]))
    as_of = pd.Timestamp("2025-01-10")
    assert 7 in [p for p, _m, _d in infer_expected_roster(store, "A", as_of)]
    assert 7 not in [p for p, _m, _d in infer_expected_roster(store, "B", as_of)], \
        "membership must NOT transfer before P is observed playing for B"

    # P now appears for B
    for i in range(3):
        seed(store, mk_map(f"2025-02-0{i+1}", f"b{i}", f"B{i}", "B", [7, 61, 62, 63, 64],
                            "Z", [51, 52, 53, 54, 55]))
    later = pd.Timestamp("2025-02-10")
    assert 7 in [p for p, _m, _d in infer_expected_roster(store, "B", later)], \
        "after real appearances for B, P may enter B's inferred roster"


def test_global_player_form_persists_across_a_transfer():
    store = PlayerRosterStateStore()
    for i in range(3):
        seed(store, mk_map(f"2025-01-0{i+1}", f"a{i}", f"A{i}", "A", [7, 2, 3, 4, 5],
                            "Z", [51, 52, 53, 54, 55], adr=95.0))
    form_before = compute_player_form(store.get_player(7), pd.Timestamp("2025-01-10"))
    assert form_before["prior_maps"] == 3
    assert form_before["time_weighted_adr"] == pytest.approx(95.0)

    seed(store, mk_map("2025-02-01", "b0", "B0", "B", [7, 61, 62, 63, 64],
                        "Z", [51, 52, 53, 54, 55], adr=95.0))
    form_after = compute_player_form(store.get_player(7), pd.Timestamp("2025-02-10"))
    assert form_after["prior_maps"] == 4, "individual history must survive the team change"
    assert form_after["player_history_mass"] > form_before["player_history_mass"]
    # and the history spans both teams
    assert {h.team_canonical for h in store.get_player(7).history} == {"A", "B"}


# ===========================================================================
# D. Roster inference determinism and windows  (brief sections 6, 15)
# ===========================================================================

def test_roster_inference_is_top5_by_mass_then_recency_then_id():
    store = PlayerRosterStateStore()
    # six players; 1-5 appear 3x, player 6 appears once (a stand-in)
    for i in range(3):
        seed(store, mk_map(f"2025-03-0{i+1}", f"g{i}", f"M{i}", "A", [1, 2, 3, 4, 5],
                            "Z", [51, 52, 53, 54, 55]))
    seed(store, mk_map("2025-03-05", "g9", "M9", "A", [6, 2, 3, 4, 5], "Z", [51, 52, 53, 54, 55]))

    roster = infer_expected_roster(store, "A", pd.Timestamp("2025-03-10"))
    pids = [p for p, _m, _d in roster]
    assert len(pids) == ROSTER_SIZE
    assert 6 not in pids, "a single-appearance stand-in must not displace an established core"
    assert set(pids) == {1, 2, 3, 4, 5}


def test_roster_inference_tie_break_is_deterministic_by_player_id():
    store = PlayerRosterStateStore()
    # six players, all with exactly one appearance at the SAME instant ->
    # identical mass and identical recency; only the id tie-break separates them
    seed(store, [mk_obs("2025-03-01", "g0", "M0", 1, i + 1, "A", pid)
                 for i, pid in enumerate([30, 10, 50, 20, 60, 40])])
    pids = [p for p, _m, _d in infer_expected_roster(store, "A", pd.Timestamp("2025-03-02"))]
    assert pids == [10, 20, 30, 40, 50], "ties must resolve by ascending player_id"


def test_roster_lookback_window_is_90_days_half_open():
    store = PlayerRosterStateStore()
    as_of = pd.Timestamp("2025-06-01")
    seed(store, [mk_obs(as_of - pd.Timedelta(days=ROSTER_LOOKBACK_DAYS), "gin", "MIN", 1, 1, "A", 1)])
    seed(store, [mk_obs(as_of - pd.Timedelta(days=ROSTER_LOOKBACK_DAYS + 1), "gout", "MOUT", 1, 1, "A", 2)])
    seed(store, [mk_obs(as_of, "gnow", "MNOW", 1, 1, "A", 3)])
    pids = [p for p, _m, _d in infer_expected_roster(store, "A", as_of)]
    assert 1 in pids, "an appearance exactly at the lower bound is INCLUDED"
    assert 2 not in pids, "an appearance older than the window is excluded"
    assert 3 not in pids, "an appearance exactly at as_of is the current series - excluded"


def test_recency_weight_half_life_is_60_days():
    as_of = pd.Timestamp("2025-06-01")
    assert recency_weight(as_of, as_of) == pytest.approx(1.0)
    assert recency_weight(as_of, as_of - pd.Timedelta(days=60)) == pytest.approx(0.5)
    assert recency_weight(as_of, as_of - pd.Timedelta(days=120)) == pytest.approx(0.25)
    assert PLAYER_FORM_HALF_LIFE_DAYS == 60.0


# ===========================================================================
# E. Roster stability  (brief section 15)
# ===========================================================================

def test_stable_roster_scores_higher_continuity_than_unstable_roster():
    stable = PlayerRosterStateStore()
    for i in range(10):
        seed(stable, [mk_obs(f"2025-04-{i+1:02d}", f"s{i}", f"S{i}", 1, s + 1, "STABLE", pid)
                      for s, pid in enumerate([1, 2, 3, 4, 5])])

    unstable = PlayerRosterStateStore()
    pool = [[1, 2, 3, 4, 5], [1, 2, 3, 6, 7], [1, 8, 9, 4, 5], [10, 2, 3, 11, 5],
            [1, 12, 3, 4, 13], [14, 2, 15, 4, 5], [1, 2, 16, 17, 5], [18, 19, 3, 4, 5],
            [1, 2, 3, 20, 21], [22, 23, 3, 4, 5]]
    for i, lineup in enumerate(pool):
        seed(unstable, [mk_obs(f"2025-04-{i+1:02d}", f"u{i}", f"U{i}", 1, s + 1, "UNSTABLE", pid)
                        for s, pid in enumerate(lineup)])

    as_of = pd.Timestamp("2025-04-20")
    fs = compute_team_roster_features(stable, "STABLE", as_of)
    fu = compute_team_roster_features(unstable, "UNSTABLE", as_of)

    assert fs["core5_continuity_last_10"] == pytest.approx(1.0)
    assert fu["core5_continuity_last_10"] < fs["core5_continuity_last_10"]
    assert fs["core5_appearance_concentration_90d"] == pytest.approx(1.0)
    assert fu["core5_appearance_concentration_90d"] < fs["core5_appearance_concentration_90d"]
    assert fs["recent_unique_players_10_maps"] == 5
    assert fu["recent_unique_players_10_maps"] > 5


def test_recent_unique_players_counts_distinct_maps_not_observations():
    store = PlayerRosterStateStore()
    for i in range(12):
        seed(store, [mk_obs(f"2025-05-{i+1:02d}", f"g{i}", f"M{i}", 1, s + 1, "A", pid)
                     for s, pid in enumerate([1, 2, 3, 4, 5])])
    f = compute_team_roster_features(store, "A", pd.Timestamp("2025-06-01"))
    assert f["recent_unique_players_10_maps"] == 5
    assert f["recent_unique_players_20_maps"] == 5


# ===========================================================================
# F. Malformed-source invariants  (correction #2)
# ===========================================================================

def test_duplicate_observation_cannot_inflate_history_mass_or_appearances():
    """Even if the stream somehow presents the same (game_id, player) twice,
    state must absorb it exactly once."""
    store = PlayerRosterStateStore()
    row = mk_obs("2025-01-01", "g0", "M0", 1, 1, "A", 7)
    apply_player_observation(store, pd.Series(row))
    apply_player_observation(store, pd.Series(row))
    dup_slot = dict(row); dup_slot["slot"] = 4
    apply_player_observation(store, pd.Series(dup_slot))

    assert len(store.get_player(7).history) == 1, "max ONE PlayerMapEntry per (game_id, player_id)"
    assert len(store.get_team("A").appearances) == 1, \
        "max ONE AppearanceEntry per (game_id, team, player_id)"
    form = compute_player_form(store.get_player(7), pd.Timestamp("2025-01-02"))
    assert form["prior_maps"] == 1


def test_invariant_keys_are_tracked_for_every_applied_observation():
    store = seed(PlayerRosterStateStore(),
                 mk_map("2025-01-01", "g0", "M0", "A", [1, 2, 3, 4, 5], "B", [11, 12, 13, 14, 15]))
    assert len(store.player_map_keys) == 10
    assert len(store.appearance_keys) == 10
    assert all(len(s.history) == 1 for s in store.players.values())


def test_ineligible_team_records_no_appearance_but_player_form_still_updates():
    store = PlayerRosterStateStore()
    seed(store, [mk_obs("2025-01-01", "g0", "M0", 1, 1, "INELIGIBLE", 7, team_eligible=False)])
    assert store.get_team("INELIGIBLE") is None, "team-keyed appearance requires an eligible team"
    assert len(store.get_player(7).history) == 1, "player-keyed form is a real fact regardless"


def test_observation_without_usable_stats_records_appearance_only():
    store = PlayerRosterStateStore()
    seed(store, [mk_obs("2025-01-01", "g0", "M0", 1, 1, "A", 7, has_usable_stats=False)])
    assert len(store.get_team("A").appearances) == 1
    assert store.get_player(7) is None, "no performance entry without a usable box score"


# ===========================================================================
# G. Cold start  (brief section 16)
# ===========================================================================

def test_two_completely_unknown_teams_get_documented_cold_start():
    f = build_future_player_roster_features(PlayerRosterStateStore(), "NEW1", "NEW2", "2026-01-01")
    for k in ROSTER_PERFORMANCE_DIFFS:
        assert math.isnan(f[k]), f"{k} must be genuinely missing, never a fabricated number"
    assert f["roster_size_min"] == 0
    assert f["both_teams_have_5_inferred_players"] == 0
    assert f["roster_min_player_history_mass"] == COLD_START_HISTORY_MASS
    assert f["roster_core_concentration_min"] == 0.0
    assert f["roster_core_continuity_last10_min"] == 0.0
    assert f["roster_form_players_min"] == 0
    # stability diffs remain defined (0 - 0)
    assert f["recent_unique_players_10_maps_diff"] == 0
    assert f["core5_continuity_last_10_diff"] == 0.0


def test_performance_diffs_are_nan_exactly_when_form_players_min_is_zero():
    """The invariant the builder and validator both assert on real data."""
    store = PlayerRosterStateStore()
    for i in range(3):
        seed(store, mk_map(f"2025-01-0{i+1}", f"g{i}", f"M{i}", "A", [1, 2, 3, 4, 5],
                            "B", [11, 12, 13, 14, 15]))
    as_of = pd.Timestamp("2025-02-01")

    both_known = build_future_player_roster_features(store, "A", "B", as_of)
    assert both_known["roster_form_players_min"] > 0
    assert not any(math.isnan(both_known[k]) for k in ROSTER_PERFORMANCE_DIFFS)

    one_unknown = build_future_player_roster_features(store, "A", "UNKNOWN", as_of)
    assert one_unknown["roster_form_players_min"] == 0
    assert all(math.isnan(one_unknown[k]) for k in ROSTER_PERFORMANCE_DIFFS)


def test_roster_form_players_min_differs_from_roster_size_min():
    """Correction #3: a team can have five INFERRED players while only some
    of them have usable prior box scores."""
    store = PlayerRosterStateStore()
    # all five appear (so roster_size=5), but only two ever had usable stats
    for i in range(3):
        rows = []
        for slot, pid in enumerate([1, 2, 3, 4, 5], start=1):
            rows.append(mk_obs(f"2025-01-0{i+1}", f"g{i}", f"M{i}", 1, slot, "A", pid,
                                has_usable_stats=(pid in (1, 2))))
        seed(store, rows)
    f = compute_team_roster_features(store, "A", pd.Timestamp("2025-02-01"))
    assert f["roster_size"] == 5
    assert f["roster_form_player_count"] == 2
    assert f["roster_min_player_history_mass"] == 0.0     # three players have no evidence
    assert not math.isnan(f["roster_mean_adr"])           # aggregated over the two that do


def test_fewer_than_five_players_are_never_fabricated():
    store = seed(PlayerRosterStateStore(),
                 [mk_obs("2025-01-01", "g0", "M0", 1, i + 1, "A", pid)
                  for i, pid in enumerate([1, 2, 3])])
    f = compute_team_roster_features(store, "A", pd.Timestamp("2025-02-01"))
    assert f["roster_size"] == 3
    assert len(infer_expected_roster(store, "A", pd.Timestamp("2025-02-01"))) == 3


# ===========================================================================
# H. Side-swap symmetry  (brief section 22)
# ===========================================================================

def _seed_asymmetric():
    store = PlayerRosterStateStore()
    # A: stable, strong, long history
    for i in range(8):
        seed(store, [mk_obs(f"2025-0{(i//4)+1}-{(i%4)+1:02d}", f"a{i}", f"A{i}", 1, s + 1, "A", pid,
                            adr=90.0, kast=76.0, kd_balance=0.2, apr=0.2)
                     for s, pid in enumerate([1, 2, 3, 4, 5])])
    # B: rotating, weaker, shorter history
    pool = [[11, 12, 13, 14, 15], [11, 12, 16, 17, 15], [18, 12, 13, 19, 15]]
    for i, lineup in enumerate(pool):
        seed(store, [mk_obs(f"2025-03-{i+1:02d}", f"b{i}", f"B{i}", 1, s + 1, "B", pid,
                            adr=62.0, kast=64.0, kd_balance=-0.15, apr=0.1)
                     for s, pid in enumerate(lineup)])
    return store


def test_directional_features_negate_and_symmetric_unchanged_under_side_swap():
    store = _seed_asymmetric()
    as_of = pd.Timestamp("2025-04-01")
    ab = build_future_player_roster_features(store, "A", "B", as_of)
    ba = build_future_player_roster_features(store, "B", "A", as_of)

    nontrivial = 0
    for k in ROSTER_DIRECTIONAL_FEATURES:
        if isinstance(ab[k], float) and math.isnan(ab[k]):
            assert math.isnan(ba[k]), k
            continue
        assert ab[k] == pytest.approx(-ba[k], abs=1e-12), f"{k} did not negate under side swap"
        if abs(ab[k]) > 1e-9:
            nontrivial += 1
    assert nontrivial >= 6, "fixture too symmetric to prove anything"

    for k in ROSTER_SYMMETRIC_FEATURES:
        assert ab[k] == pytest.approx(ba[k]), f"{k} changed under side swap"


# ===========================================================================
# I. Feature-set shape and JSON round-trip
# ===========================================================================

def test_feature_counts_match_the_declared_family_sizes():
    assert len(ROSTER_DIRECTIONAL_FEATURES) == 15
    assert len(ROSTER_SYMMETRIC_FEATURES) == 6
    assert len(ROSTER_PERFORMANCE_DIFFS) == 10
    assert set(ROSTER_PERFORMANCE_DIFFS) <= set(ROSTER_DIRECTIONAL_FEATURES)
    assert not set(ROSTER_DIRECTIONAL_FEATURES) & set(ROSTER_SYMMETRIC_FEATURES)


def test_state_json_round_trip_preserves_features(tmp_path):
    store = _seed_asymmetric()
    as_of = pd.Timestamp("2025-04-01")
    before = build_future_player_roster_features(store, "A", "B", as_of)

    path = tmp_path / "state.json"
    store.to_json(path, meta={"test": True})
    reloaded = PlayerRosterStateStore.from_json(path)
    after = build_future_player_roster_features(reloaded, "A", "B", as_of)

    for k in ROSTER_DIRECTIONAL_FEATURES + ROSTER_SYMMETRIC_FEATURES:
        x, y = before[k], after[k]
        if isinstance(x, float) and math.isnan(x):
            assert math.isnan(y), k
        else:
            assert x == pytest.approx(y), k
    assert len(reloaded.players) == len(store.players)
    assert len(reloaded.teams) == len(store.teams)


def test_no_player_identity_or_stat_column_leaks_into_the_feature_names():
    f = build_future_player_roster_features(_seed_asymmetric(), "A", "B", "2025-04-01")
    emitted = set(ROSTER_DIRECTIONAL_FEATURES + ROSTER_SYMMETRIC_FEATURES)
    forbidden_tokens = ("player_id", "player1", "player2", "player3", "player4", "player5",
                        "kills", "deaths", "kddiff", "score1", "score2")
    bad = [c for c in emitted if any(t in c for t in forbidden_tokens)]
    assert not bad, f"model-facing feature name exposes raw player identity/box score: {bad}"
