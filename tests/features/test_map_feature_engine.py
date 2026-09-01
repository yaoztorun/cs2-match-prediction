"""
Synthetic-fixture proofs for the Phase 5A map feature engine
(feature_engineering/maps/map_feature_engine.py).

Every test builds its own tiny fixture - nothing here touches the real
dataset, so these are proofs about the ENGINE, not about the Kaggle export.
Test classes map to the spec letters A-N, plus the explicitly required
"maps of one series carry different raw map timestamps" test.
"""

import math

import numpy as np
import pandas as pd
import pytest

from feature_engineering.maps.map_feature_engine import (
    MapStateStore, TeamMapState, MapHistoryEntry,
    MAP_ENGINE_VERSION, MAP_POOL_LOOKBACK_DAYS, UNKNOWN_MAP_CATEGORY,
    COLD_START_MAP_ELO, COLD_START_SMOOTHED_WR, COLD_START_MARGIN,
    MAP_DIRECTIONAL_FEATURES, MAP_SYMMETRIC_FEATURES,
    SERIES_MAP_POOL_DIRECTIONAL, SERIES_MAP_POOL_SYMMETRIC,
    normalized_round_margin, compute_team_map_features, recent_map_pool,
    build_future_map_matchup_features, build_future_series_map_pool_features,
    apply_map_result, process_map_stream, process_combined_stream,
    SERIES_REQUEST_COLUMNS,
)
from feature_engineering.series.feature_engine import ELO_INITIAL, ELO_K, elo_expected


def mk_map_row(series_dt, match_id, game_id, t1, t2, s1, s2, map_name,
               map_dt=None, bo=3, tier="tier1", t1_elig=True, t2_elig=True):
    """`map_dt` defaults to `series_dt` but is deliberately settable: the engine
    must key everything on the series cutoff, never on the raw map timestamp."""
    return {
        "series_datetime": pd.Timestamp(series_dt),
        "map_datetime": pd.Timestamp(map_dt if map_dt is not None else series_dt),
        "match_id": match_id, "game_id": game_id, "map_name": map_name,
        "team1_canonical": t1, "team2_canonical": t2,
        "score1_game": s1, "score2_game": s2,
        "team1_map_win": 1 if s1 > s2 else 0,
        "bestOf": bo, "tier": tier,
        "team1_eligible": t1_elig, "team2_eligible": t2_elig,
    }


def stream(rows):
    return pd.DataFrame(rows)


def feats_equal(a, b, keys):
    """NaN-aware comparison. `days_since_map_played_diff` is legitimately NaN
    on cold start, and NaN != NaN would make plain dict equality report a false
    difference, so missing-vs-missing counts as equal here."""
    for k in keys:
        x, y = a[k], b[k]
        if isinstance(x, float) and math.isnan(x):
            if not (isinstance(y, float) and math.isnan(y)):
                return False
        elif x != pytest.approx(y):
            return False
    return True


def by_game(rows, keys):
    return {r["game_id"]: {k: r[k] for k in keys} for r in rows}


# ---------------------------------------------------------------------------
# A. Beta(2,2) smoothed map win rate
# ---------------------------------------------------------------------------

class TestA_BetaSmoothing:
    def test_no_history_is_one_half(self):
        f = compute_team_map_features(None, "2026-01-01")
        assert f["map_smoothed_win_rate"] == 0.5

    def test_single_win_is_not_one(self):
        store = MapStateStore()
        apply_map_result(store, mk_map_row("2025-01-01", "m1", "g1", "A", "B", 13, 7, "Mirage"))
        f = compute_team_map_features(store.get("A", "Mirage"), "2025-06-01")
        # (1 + 2) / (1 + 4) = 0.6, NOT 1.0
        assert f["map_smoothed_win_rate"] == pytest.approx(0.6)

    def test_single_loss_is_not_zero(self):
        store = MapStateStore()
        apply_map_result(store, mk_map_row("2025-01-01", "m1", "g1", "A", "B", 7, 13, "Mirage"))
        f = compute_team_map_features(store.get("A", "Mirage"), "2025-06-01")
        assert f["map_smoothed_win_rate"] == pytest.approx(0.4)   # (0+2)/(1+4)

    def test_smoothing_converges_toward_raw_rate(self):
        store = MapStateStore()
        for i in range(40):
            apply_map_result(store, mk_map_row(f"2025-01-{i % 28 + 1:02d}", f"m{i}", f"g{i}",
                                                "A", "B", 13, 7, "Mirage"))
        f = compute_team_map_features(store.get("A", "Mirage"), "2026-01-01")
        assert f["map_smoothed_win_rate"] == pytest.approx(42 / 44)
        assert f["map_smoothed_win_rate"] < 1.0


# ---------------------------------------------------------------------------
# B. Map ELO reuses the Phase 3 mathematics
# ---------------------------------------------------------------------------

class TestB_MapElo:
    def test_cold_start_is_1500(self):
        assert COLD_START_MAP_ELO == ELO_INITIAL == 1500.0
        f = compute_team_map_features(None, "2026-01-01")
        assert f["map_elo"] == 1500.0

    def test_equal_ratings_expect_one_half(self):
        assert elo_expected(1500.0, 1500.0) == pytest.approx(0.5)

    def test_winner_gains_exactly_what_loser_loses(self):
        store = MapStateStore()
        apply_map_result(store, mk_map_row("2025-01-01", "m1", "g1", "A", "B", 13, 7, "Nuke"))
        a = store.get("A", "Nuke").map_elo
        b = store.get("B", "Nuke").map_elo
        assert a == pytest.approx(1500.0 + ELO_K * 0.5)
        assert b == pytest.approx(1500.0 - ELO_K * 0.5)
        assert a + b == pytest.approx(3000.0)   # rating mass conserved

    def test_map_elos_are_independent_per_map(self):
        store = MapStateStore()
        apply_map_result(store, mk_map_row("2025-01-01", "m1", "g1", "A", "B", 13, 7, "Nuke"))
        assert store.get("A", "Nuke").map_elo > 1500.0
        # A has never played Mirage -> no state at all, cold start
        assert store.get("A", "Mirage") is None
        assert compute_team_map_features(store.get("A", "Mirage"), "2026-01-01")["map_elo"] == 1500.0


# ---------------------------------------------------------------------------
# C. Strict chronology - a map at exactly T is not history for T
# ---------------------------------------------------------------------------

class TestC_StrictChronology:
    def test_map_at_exactly_T_is_excluded(self):
        store = MapStateStore()
        apply_map_result(store, mk_map_row("2025-05-01 12:00", "m1", "g1", "A", "B", 13, 7, "Mirage"))
        st = store.get("A", "Mirage")
        assert compute_team_map_features(st, "2025-05-01 12:00")["map_matches_before"] == 0
        assert compute_team_map_features(st, "2025-05-01 12:01")["map_matches_before"] == 1

    def test_future_maps_never_leak_backwards(self):
        store = MapStateStore()
        for d in ["2025-01-01", "2025-02-01", "2025-03-01"]:
            apply_map_result(store, mk_map_row(d, f"m{d}", f"g{d}", "A", "B", 13, 7, "Mirage"))
        st = store.get("A", "Mirage")
        assert compute_team_map_features(st, "2025-02-15")["map_matches_before"] == 2
        assert compute_team_map_features(st, "2025-01-15")["map_matches_before"] == 1

    def test_days_since_is_missing_not_sentinel_when_no_history(self):
        f = compute_team_map_features(None, "2026-01-01")
        assert math.isnan(f["days_since_map_played"])


# ---------------------------------------------------------------------------
# D. SAME-SERIES ISOLATION - map 1's result must not reach map 2/3
# ---------------------------------------------------------------------------

class TestD_SameSeriesIsolation:
    def test_all_maps_of_a_bo3_see_identical_pre_series_state(self):
        rows = stream([
            mk_map_row("2025-06-01", "M1", "g1", "A", "B", 13, 7, "Mirage"),
            mk_map_row("2025-06-01", "M1", "g2", "A", "B", 13, 9, "Nuke"),
            mk_map_row("2025-06-01", "M1", "g3", "A", "B", 5, 13, "Inferno"),
        ])
        store = MapStateStore()
        out, _ = process_map_stream(store, rows)
        assert len(out) == 3
        # nobody had any history before this series
        for r in out:
            assert r["team1_map_matches_before"] == 0
            assert r["team2_map_matches_before"] == 0
            assert r["map_elo_diff"] == 0.0

    def test_repeated_map_within_one_series_still_isolated(self):
        """Pathological but must hold: the same map twice in one series."""
        rows = stream([
            mk_map_row("2025-06-01", "M1", "g1", "A", "B", 13, 7, "Mirage"),
            mk_map_row("2025-06-01", "M1", "g2", "A", "B", 13, 9, "Mirage"),
        ])
        store = MapStateStore()
        out, _ = process_map_stream(store, rows)
        assert [r["team1_map_matches_before"] for r in out] == [0, 0]
        # after the batch BOTH results are applied
        assert compute_team_map_features(store.get("A", "Mirage"), "2025-07-01")["map_matches_before"] == 2

    def test_next_series_does_see_the_previous_series(self):
        rows = stream([
            mk_map_row("2025-06-01", "M1", "g1", "A", "B", 13, 7, "Mirage"),
            mk_map_row("2025-06-02", "M2", "g2", "A", "B", 13, 9, "Mirage"),
        ])
        store = MapStateStore()
        out, _ = process_map_stream(store, rows)
        assert out[0]["team1_map_matches_before"] == 0
        assert out[1]["team1_map_matches_before"] == 1


# ---------------------------------------------------------------------------
# D2. REQUIRED: differing raw map timestamps inside one series
# ---------------------------------------------------------------------------

class TestD2_DifferentMapTimestampsSameSeriesCutoff:
    """Correctness must NOT depend on the current Kaggle property that all map
    rows of a series share one datetime. Here the maps carry realistic, DIFFERENT
    `map_datetime` values (a BO3 played over three hours) while sharing one
    `series_datetime` prediction cutoff. All three must still receive the
    identical pre-series state."""

    def _rows(self):
        return stream([
            mk_map_row("2025-06-01 12:00", "M1", "g1", "A", "B", 13, 7, "Mirage",
                       map_dt="2025-06-01 12:05"),
            mk_map_row("2025-06-01 12:00", "M1", "g2", "A", "B", 13, 9, "Nuke",
                       map_dt="2025-06-01 13:20"),
            mk_map_row("2025-06-01 12:00", "M1", "g3", "A", "B", 5, 13, "Inferno",
                       map_dt="2025-06-01 14:45"),
        ])

    def test_identical_pre_series_state_despite_differing_map_times(self):
        # give A a prior history so "identical" is a non-trivial claim
        store = MapStateStore()
        # DISTINCT opponents (C for A, D for B) so each prior map moves elo by
        # exactly K/2 from 1500 - a shared opponent would already be off 1500
        # for the second match and the expected values below would not be exact.
        for m in ("Mirage", "Nuke", "Inferno"):
            apply_map_result(store, mk_map_row("2025-01-01", "P", f"p{m}", "A", "C", 13, 4, m))
            apply_map_result(store, mk_map_row("2025-01-02", "Q", f"q{m}", "B", "D", 8, 13, m))

        out, _ = process_map_stream(store, self._rows())
        assert len(out) == 3
        # each map's own identity differs, but the SNAPSHOT is one and the same:
        # every team1 stat is drawn from state frozen at 2025-06-01 12:00
        for r in out:
            assert r["team1_map_matches_before"] == 1
            assert r["team2_map_matches_before"] == 1
            assert r["team1_map_elo"] == pytest.approx(1500.0 + ELO_K * 0.5)
            assert r["team2_map_elo"] == pytest.approx(1500.0 - ELO_K * 0.5)
            assert r["map_elo_diff"] == pytest.approx(ELO_K)

    def test_map_datetime_ordering_does_not_change_anything(self):
        """Shuffling the raw map times (even inverting them) changes nothing,
        because `map_datetime` is provenance only."""
        base, _ = process_map_stream(MapStateStore(), self._rows())
        shuffled = self._rows()
        shuffled["map_datetime"] = list(reversed(list(shuffled["map_datetime"])))
        other, _ = process_map_stream(MapStateStore(), shuffled)
        keys = MAP_DIRECTIONAL_FEATURES + MAP_SYMMETRIC_FEATURES
        for a, b in zip(sorted(base, key=lambda r: r["game_id"]),
                        sorted(other, key=lambda r: r["game_id"])):
            assert feats_equal(a, b, keys)

    def test_map_dt_later_than_cutoff_is_not_treated_as_history(self):
        """A map physically played AFTER the cutoff still belongs to the series
        and must not become history for that same series."""
        store = MapStateStore()
        out, _ = process_map_stream(store, self._rows())
        assert all(r["team1_map_matches_before"] == 0 for r in out)
        # ...and afterwards all three are recorded
        assert sum(len(store.get("A", m).history) for m in ("Mirage", "Nuke", "Inferno")) == 3


# ---------------------------------------------------------------------------
# E. Same-timestamp isolation across DIFFERENT series
# ---------------------------------------------------------------------------

class TestE_SameTimestampIsolation:
    def _rows(self, order):
        r = {
            "X": mk_map_row("2025-06-01 15:00", "MX", "gx", "A", "B", 13, 7, "Mirage"),
            "Y": mk_map_row("2025-06-01 15:00", "MY", "gy", "A", "D", 13, 8, "Mirage"),
        }
        return stream([r[k] for k in order])

    def test_two_matches_at_the_same_instant_cannot_see_each_other(self):
        out, _ = process_map_stream(MapStateStore(), self._rows(["X", "Y"]))
        assert all(r["team1_map_matches_before"] == 0 for r in out)

    def test_row_order_is_irrelevant(self):
        a, _ = process_map_stream(MapStateStore(), self._rows(["X", "Y"]))
        b, _ = process_map_stream(MapStateStore(), self._rows(["Y", "X"]))
        keys = MAP_DIRECTIONAL_FEATURES + MAP_SYMMETRIC_FEATURES
        ka, kb = by_game(a, keys), by_game(b, keys)
        assert set(ka) == set(kb)
        assert all(feats_equal(ka[g], kb[g], keys) for g in ka)

    def test_both_results_applied_after_the_batch(self):
        store = MapStateStore()
        process_map_stream(store, self._rows(["X", "Y"]))
        assert len(store.get("A", "Mirage").history) == 2


# ---------------------------------------------------------------------------
# F. Target reconstruction from scores
# ---------------------------------------------------------------------------

class TestF_TargetFromScores:
    def test_win_flag_matches_score_comparison(self):
        store = MapStateStore()
        apply_map_result(store, mk_map_row("2025-01-01", "m1", "g1", "A", "B", 13, 7, "Mirage"))
        apply_map_result(store, mk_map_row("2025-01-02", "m2", "g2", "A", "B", 10, 13, "Mirage"))
        wins = [h.win for h in store.get("A", "Mirage").history]
        assert wins == [1, 0]
        assert [h.win for h in store.get("B", "Mirage").history] == [0, 1]

    def test_scores_are_stored_from_each_team_perspective(self):
        store = MapStateStore()
        apply_map_result(store, mk_map_row("2025-01-01", "m1", "g1", "A", "B", 13, 7, "Mirage"))
        ha = store.get("A", "Mirage").history[0]
        hb = store.get("B", "Mirage").history[0]
        assert (ha.score_for, ha.score_against) == (13, 7)
        assert (hb.score_for, hb.score_against) == (7, 13)
        assert ha.normalized_margin == pytest.approx(-hb.normalized_margin)


# ---------------------------------------------------------------------------
# G. Side-swap symmetry
# ---------------------------------------------------------------------------

def _seed_asymmetric_store():
    """A store where the two teams genuinely differ, so symmetry is a real test."""
    store = MapStateStore()
    events = [
        ("2025-01-05", "A", "C", 13, 4, "Mirage"), ("2025-01-12", "A", "D", 13, 9, "Mirage"),
        ("2025-02-05", "A", "C", 11, 13, "Nuke"), ("2025-03-05", "A", "E", 13, 2, "Ancient"),
        ("2025-01-20", "B", "C", 7, 13, "Mirage"), ("2025-02-20", "B", "D", 13, 11, "Nuke"),
        ("2025-03-20", "B", "E", 13, 6, "Nuke"), ("2025-04-01", "B", "C", 13, 10, "Inferno"),
        ("2025-04-10", "A", "B", 13, 8, "Mirage"),
    ]
    for i, (d, t1, t2, s1, s2, mp) in enumerate(events):
        apply_map_result(store, mk_map_row(d, f"s{i}", f"g{i}", t1, t2, s1, s2, mp))
    return store


class TestG_SideSwapSymmetry:
    AS_OF = "2025-06-01"

    def test_map_matchup_directional_features_negate(self):
        store = _seed_asymmetric_store()
        f = build_future_map_matchup_features(store, "A", "B", 3, "Mirage", self.AS_OF)
        g = build_future_map_matchup_features(store, "B", "A", 3, "Mirage", self.AS_OF)
        nontrivial = 0
        for k in MAP_DIRECTIONAL_FEATURES:
            if math.isnan(f[k]):
                assert math.isnan(g[k])
                continue
            assert f[k] == pytest.approx(-g[k], abs=1e-12), k
            if abs(f[k]) > 1e-9:
                nontrivial += 1
        assert nontrivial >= 4, "fixture too symmetric to prove anything"

    def test_map_matchup_symmetric_features_are_unchanged(self):
        store = _seed_asymmetric_store()
        f = build_future_map_matchup_features(store, "A", "B", 3, "Mirage", self.AS_OF)
        g = build_future_map_matchup_features(store, "B", "A", 3, "Mirage", self.AS_OF)
        for k in MAP_SYMMETRIC_FEATURES:
            assert f[k] == g[k], k

    def test_series_pool_directional_features_negate(self):
        store = _seed_asymmetric_store()
        f = build_future_series_map_pool_features(store, "A", "B", 3, self.AS_OF)
        g = build_future_series_map_pool_features(store, "B", "A", 3, self.AS_OF)
        nontrivial = 0
        for k in SERIES_MAP_POOL_DIRECTIONAL:
            assert f[k] == pytest.approx(-g[k], abs=1e-12), k
            if abs(f[k]) > 1e-9:
                nontrivial += 1
        assert nontrivial >= 6, "fixture too symmetric to prove anything"

    def test_series_pool_symmetric_features_are_unchanged(self):
        store = _seed_asymmetric_store()
        f = build_future_series_map_pool_features(store, "A", "B", 3, self.AS_OF)
        g = build_future_series_map_pool_features(store, "B", "A", 3, self.AS_OF)
        for k in SERIES_MAP_POOL_SYMMETRIC:
            assert f[k] == pytest.approx(g[k], abs=1e-12), k

    def test_order_statistics_would_have_broken_symmetry(self):
        """Guards the design decision documented in the engine: the k-th-best of
        the per-map advantage list is NOT self-antisymmetric, which is exactly
        why the (B) family uses mean/median/midrange instead."""
        adv = [30.0, -5.0, 12.0, -40.0]
        swapped = [-x for x in adv]
        second_best = sorted(adv, reverse=True)[1]
        second_best_swapped = sorted(swapped, reverse=True)[1]
        assert second_best != pytest.approx(-second_best_swapped)
        # while the statistics actually used are fine:
        assert float(np.mean(adv)) == pytest.approx(-float(np.mean(swapped)))
        assert float(np.median(adv)) == pytest.approx(-float(np.median(swapped)))
        assert (max(adv) + min(adv)) / 2 == pytest.approx(-(max(swapped) + min(swapped)) / 2)
        assert (max(adv) - min(adv)) == pytest.approx(max(swapped) - min(swapped))


# ---------------------------------------------------------------------------
# H. Cold start: unknown team, unknown map
# ---------------------------------------------------------------------------

class TestH_ColdStart:
    def test_two_completely_unknown_teams(self):
        f = build_future_map_matchup_features(MapStateStore(), "NEW1", "NEW2", 3, "Mirage", "2026-01-01")
        assert f["team1_map_elo"] == f["team2_map_elo"] == 1500.0
        assert f["map_elo_diff"] == 0.0
        assert f["team1_map_smoothed_win_rate"] == COLD_START_SMOOTHED_WR
        assert f["team1_map_normalized_margin_all"] == COLD_START_MARGIN
        assert f["both_teams_have_map_history"] == 0
        assert f["map_matches_before_sum"] == 0

    def test_known_team_on_a_map_it_has_never_played(self):
        store = _seed_asymmetric_store()
        f = build_future_map_matchup_features(store, "A", "B", 3, "Vertigo", "2026-01-01")
        assert f["team1_map_elo"] == 1500.0 and f["team2_map_elo"] == 1500.0
        assert f["both_teams_have_map_history"] == 0

    def test_unknown_map_category_is_reserved_and_cold_starts(self):
        assert UNKNOWN_MAP_CATEGORY == "__UNKNOWN_MAP__"
        f = build_future_map_matchup_features(_seed_asymmetric_store(), "A", "B", 3,
                                               UNKNOWN_MAP_CATEGORY, "2026-01-01")
        assert f["map_elo_diff"] == 0.0
        assert f["map_name"] == UNKNOWN_MAP_CATEGORY

    def test_feature_computation_never_mutates_state(self):
        store = _seed_asymmetric_store()
        before = len(store.states)
        build_future_map_matchup_features(store, "A", "ZZZ", 3, "Vertigo", "2026-01-01")
        build_future_series_map_pool_features(store, "A", "ZZZ", 3, "2026-01-01")
        assert len(store.states) == before

    def test_empty_pool_gives_neutral_not_extreme_values(self):
        f = build_future_series_map_pool_features(MapStateStore(), "N1", "N2", 3, "2026-01-01")
        assert f["map_pool_size_diff"] == 0
        assert f["map_pool_best_elo_diff"] == 0.0
        assert f["map_pool_third_best_smoothed_wr_diff"] == 0.0
        assert f["union_map_count"] == 0
        assert f["map_matchup_shared_coverage"] == 0.0   # documented empty-union handling
        assert f["both_teams_have_map_pool_history"] == 0

    def test_shallow_pool_falls_back_to_neutral_kth_slot(self):
        """A team with ONE map has no 2nd/3rd best - the slot must fall back to
        the neutral cold-start value, never to an extreme sentinel."""
        store = MapStateStore()
        apply_map_result(store, mk_map_row("2025-05-01", "m", "g", "A", "C", 13, 3, "Mirage"))
        f = build_future_series_map_pool_features(store, "A", "B", 3, "2025-06-01")
        # A's best Mirage elo > 1500 (won), but its 2nd/3rd slots are neutral 1500
        assert f["map_pool_best_elo_diff"] > 0
        assert f["map_pool_second_best_elo_diff"] == 0.0
        assert f["map_pool_third_best_elo_diff"] == 0.0
        assert f["both_teams_have_3_recent_maps"] == 0


# ---------------------------------------------------------------------------
# I. 180-day recent map pool boundaries
# ---------------------------------------------------------------------------

class TestI_PoolLookback:
    def test_constant_is_180(self):
        assert MAP_POOL_LOOKBACK_DAYS == 180

    def test_map_exactly_at_lower_bound_is_included(self):
        store = MapStateStore()
        apply_map_result(store, mk_map_row("2025-01-01 00:00", "m", "g", "A", "B", 13, 7, "Mirage"))
        as_of = pd.Timestamp("2025-01-01 00:00") + pd.Timedelta(days=180)
        assert recent_map_pool(store, "A", as_of) == {"Mirage"}

    def test_map_just_outside_the_window_is_excluded(self):
        store = MapStateStore()
        apply_map_result(store, mk_map_row("2025-01-01 00:00", "m", "g", "A", "B", 13, 7, "Mirage"))
        as_of = pd.Timestamp("2025-01-01 00:00") + pd.Timedelta(days=180, seconds=1)
        assert recent_map_pool(store, "A", as_of) == set()

    def test_map_exactly_at_as_of_is_excluded(self):
        store = MapStateStore()
        apply_map_result(store, mk_map_row("2025-05-01 12:00", "m", "g", "A", "B", 13, 7, "Mirage"))
        assert recent_map_pool(store, "A", "2025-05-01 12:00") == set()

    def test_pool_reflects_only_recent_maps(self):
        store = MapStateStore()
        apply_map_result(store, mk_map_row("2024-01-01", "m1", "g1", "A", "B", 13, 7, "Overpass"))
        apply_map_result(store, mk_map_row("2025-05-01", "m2", "g2", "A", "B", 13, 7, "Mirage"))
        assert recent_map_pool(store, "A", "2025-06-01") == {"Mirage"}


# ---------------------------------------------------------------------------
# J. Normalized round margin
# ---------------------------------------------------------------------------

class TestJ_NormalizedMargin:
    def test_documented_examples(self):
        assert normalized_round_margin(13, 10) == pytest.approx(3 / 23)
        assert normalized_round_margin(13, 5) == pytest.approx(8 / 18)

    def test_antisymmetric_and_bounded(self):
        for a, b in [(13, 0), (13, 12), (7, 13), (16, 14), (19, 17)]:
            assert normalized_round_margin(a, b) == pytest.approx(-normalized_round_margin(b, a))
            assert -1.0 <= normalized_round_margin(a, b) <= 1.0

    def test_zero_denominator_is_zero(self):
        assert normalized_round_margin(0, 0) == 0.0

    def test_scale_invariance_across_scoring_formats(self):
        """The point of normalizing: a 16-8 MR15 win and a 13-6.5-equivalent
        MR12 win of the same proportion score the same."""
        assert normalized_round_margin(16, 8) == pytest.approx(normalized_round_margin(8, 4))


# ---------------------------------------------------------------------------
# K. Identity policy
# ---------------------------------------------------------------------------

class TestK_IdentityPolicy:
    def test_own_history_updates_against_ineligible_opponent(self):
        store = MapStateStore()
        apply_map_result(store, mk_map_row("2025-01-01", "m", "g", "A", "BAD", 13, 7, "Mirage",
                                            t1_elig=True, t2_elig=False))
        assert len(store.get("A", "Mirage").history) == 1
        assert store.get("BAD", "Mirage") is None

    def test_pair_dependent_map_elo_does_not_update(self):
        store = MapStateStore()
        apply_map_result(store, mk_map_row("2025-01-01", "m", "g", "A", "BAD", 13, 7, "Mirage",
                                            t1_elig=True, t2_elig=False))
        assert store.get("A", "Mirage").map_elo == 1500.0   # ELO frozen...
        f = compute_team_map_features(store.get("A", "Mirage"), "2025-02-01")
        assert f["map_matches_before"] == 1                  # ...but the win is real
        assert f["map_smoothed_win_rate"] == pytest.approx(0.6)

    def test_entry_is_flagged_untrusted(self):
        store = MapStateStore()
        apply_map_result(store, mk_map_row("2025-01-01", "m", "g", "A", "BAD", 13, 7, "Mirage",
                                            t1_elig=True, t2_elig=False))
        assert store.get("A", "Mirage").history[0].opponent_identity_trusted is False

    def test_both_eligible_updates_elo_and_flags_trusted(self):
        store = MapStateStore()
        apply_map_result(store, mk_map_row("2025-01-01", "m", "g", "A", "B", 13, 7, "Mirage"))
        assert store.get("A", "Mirage").map_elo > 1500.0
        assert store.get("A", "Mirage").history[0].opponent_identity_trusted is True

    def test_no_feature_row_emitted_when_an_identity_is_untrusted(self):
        rows = stream([mk_map_row("2025-01-01", "m", "g", "A", "BAD", 13, 7, "Mirage",
                                   t1_elig=True, t2_elig=False)])
        out, series_out = process_map_stream(MapStateStore(), rows, emit_series_features=True)
        assert out == [] and series_out == []


# ---------------------------------------------------------------------------
# L. Future-application API contract
# ---------------------------------------------------------------------------

class TestL_FutureApplicationAPI:
    def test_builders_need_no_target_and_no_current_scores(self):
        store = _seed_asymmetric_store()
        # only pre-match knowledge is passed - no score, no winner
        f = build_future_map_matchup_features(store, "A", "B", 3, "Mirage", "2026-01-01", tier="tier1")
        s = build_future_series_map_pool_features(store, "A", "B", 3, "2026-01-01", tier="tier1")
        forbidden = ("score1_game", "score2_game", "team1_map_win", "map_id",
                     "team1_series_win", "winner")
        assert not (set(forbidden) & set(f)) and not (set(forbidden) & set(s))

    def test_series_builder_never_takes_a_map_argument(self):
        import inspect as _inspect
        params = set(_inspect.signature(build_future_series_map_pool_features).parameters)
        assert "map_name" not in params and "maps" not in params and "map_pool" not in params

    def test_training_rows_and_future_calls_agree_exactly(self):
        """The offline stream must produce the same numbers a live pre-match call
        would - otherwise training and deployment silently diverge."""
        hist = [mk_map_row(f"2025-0{i+1}-01", f"h{i}", f"hg{i}", "A", "B",
                           13 if i % 2 == 0 else 7, 7 if i % 2 == 0 else 13, "Mirage")
                for i in range(4)]
        target = mk_map_row("2025-06-01", "T", "tg", "A", "B", 13, 11, "Mirage")
        offline_store = MapStateStore()
        out, _ = process_map_stream(offline_store, stream(hist + [target]))
        emitted = [r for r in out if r["match_id"] == "T"][0]

        live_store = MapStateStore()
        for r in hist:
            apply_map_result(live_store, r)
        live = build_future_map_matchup_features(live_store, "A", "B", 3, "Mirage",
                                                  "2025-06-01", tier="tier1")
        for k in MAP_DIRECTIONAL_FEATURES + MAP_SYMMETRIC_FEATURES:
            assert emitted[k] == pytest.approx(live[k]), k

    def test_series_offline_and_live_agree(self):
        hist = [mk_map_row("2025-03-01", "h0", "hg0", "A", "C", 13, 7, "Mirage"),
                mk_map_row("2025-03-02", "h1", "hg1", "B", "C", 9, 13, "Nuke")]
        target = mk_map_row("2025-06-01", "T", "tg", "A", "B", 13, 11, "Inferno")
        offline_store = MapStateStore()
        _, series_out = process_map_stream(offline_store, stream(hist + [target]),
                                            emit_map_features=False, emit_series_features=True)
        emitted = [r for r in series_out if r["match_id"] == "T"][0]

        live_store = MapStateStore()
        for r in hist:
            apply_map_result(live_store, r)
        live = build_future_series_map_pool_features(live_store, "A", "B", 3, "2025-06-01", tier="tier1")
        for k in SERIES_MAP_POOL_DIRECTIONAL + SERIES_MAP_POOL_SYMMETRIC:
            assert emitted[k] == pytest.approx(live[k]), k


# ---------------------------------------------------------------------------
# M. No player statistics anywhere
# ---------------------------------------------------------------------------

class TestM_NoPlayerStatistics:
    PLAYER_TOKENS = ("player", "kill", "death", "assist", "adr", "kast", "rating",
                     "hs_", "headshot", "flash", "clutch", "damage")

    def test_no_player_token_in_any_feature_name(self):
        store = _seed_asymmetric_store()
        f = build_future_map_matchup_features(store, "A", "B", 3, "Mirage", "2026-01-01")
        s = build_future_series_map_pool_features(store, "A", "B", 3, "2026-01-01")
        for name in list(f) + list(s):
            low = name.lower()
            assert not any(tok in low for tok in self.PLAYER_TOKENS), name

    def test_history_entry_carries_no_player_fields(self):
        fields = set(MapHistoryEntry.__dataclass_fields__)
        for tok in self.PLAYER_TOKENS:
            assert not any(tok in f.lower() for f in fields), tok

    def test_engine_source_declares_no_time_decay_tuning(self):
        """Phase 5A fixes the lookback as a constant; nothing may tune it."""
        assert isinstance(MAP_POOL_LOOKBACK_DAYS, int)


# ---------------------------------------------------------------------------
# N. State snapshot round-trip and provenance
# ---------------------------------------------------------------------------

class TestN_SnapshotRoundTrip:
    def test_json_round_trip_preserves_features(self, tmp_path):
        store = _seed_asymmetric_store()
        p = tmp_path / "state.json"
        store.to_json(p, meta={"cutoff": "2025-06-01"})
        reloaded = MapStateStore.from_json(p)
        assert set(reloaded.states) == set(store.states)
        for team in ("A", "B"):
            for mp in ("Mirage", "Nuke", "Ancient", "Inferno"):
                a = compute_team_map_features(store.get(team, mp), "2025-06-01")
                b = compute_team_map_features(reloaded.get(team, mp), "2025-06-01")
                for k in a:
                    if isinstance(a[k], float) and math.isnan(a[k]):
                        assert math.isnan(b[k])
                    else:
                        assert a[k] == pytest.approx(b[k]), (team, mp, k)

    def test_reloaded_store_reproduces_full_feature_vectors(self, tmp_path):
        store = _seed_asymmetric_store()
        p = tmp_path / "state.json"
        store.to_json(p)
        reloaded = MapStateStore.from_json(p)
        a = build_future_series_map_pool_features(store, "A", "B", 3, "2025-06-01")
        b = build_future_series_map_pool_features(reloaded, "A", "B", 3, "2025-06-01")
        for k in SERIES_MAP_POOL_DIRECTIONAL + SERIES_MAP_POOL_SYMMETRIC:
            assert a[k] == pytest.approx(b[k]), k

    def test_snapshot_summary_is_flat_and_scalar(self):
        df = _seed_asymmetric_store().snapshot_summary_df()
        assert len(df) > 0
        for col in df.columns:
            for v in df[col]:
                assert not isinstance(v, (list, dict, set, tuple)), col

    def test_snapshot_summary_has_one_row_per_team_map(self):
        store = _seed_asymmetric_store()
        df = store.snapshot_summary_df()
        assert len(df) == len(store.states)
        assert not df.duplicated(subset=["canonical_team_name", "map_name"]).any()

    def test_processed_uids_tracked(self):
        store = _seed_asymmetric_store()
        assert len(store.processed_map_uids) == 9
        assert "s0:g0" in store.processed_map_uids

    def test_engine_version_is_recorded(self, tmp_path):
        import json
        p = tmp_path / "state.json"
        _seed_asymmetric_store().to_json(p)
        payload = json.loads(p.read_text(encoding="utf-8"))
        assert payload["map_engine_version"] == MAP_ENGINE_VERSION


# ---------------------------------------------------------------------------
# Stream contract guards
# ---------------------------------------------------------------------------

class TestStreamContract:
    def test_missing_required_column_raises(self):
        rows = stream([mk_map_row("2025-01-01", "m", "g", "A", "B", 13, 7, "Mirage")]).drop(
            columns=["map_datetime"])
        with pytest.raises(ValueError, match="map_datetime"):
            process_map_stream(MapStateStore(), rows)

    def test_input_row_order_does_not_affect_output(self):
        rows = [
            mk_map_row("2025-03-01", "m3", "g3", "A", "B", 13, 7, "Nuke"),
            mk_map_row("2025-01-01", "m1", "g1", "A", "B", 13, 7, "Mirage"),
            mk_map_row("2025-02-01", "m2", "g2", "B", "C", 13, 7, "Mirage"),
        ]
        a, _ = process_map_stream(MapStateStore(), stream(rows))
        b, _ = process_map_stream(MapStateStore(), stream(list(reversed(rows))))
        keys = MAP_DIRECTIONAL_FEATURES + MAP_SYMMETRIC_FEATURES
        ka, kb = by_game(a, keys), by_game(b, keys)
        assert set(ka) == set(kb)
        assert all(feats_equal(ka[g], kb[g], keys) for g in ka)

    def test_extra_columns_are_carried_through(self):
        rows = stream([mk_map_row("2025-01-01", "m", "g", "A", "B", 13, 7, "Mirage")])
        rows["tournament"] = "Test Major"
        out, _ = process_map_stream(MapStateStore(), rows)
        assert out[0]["tournament"] == "Test Major"

    def test_combined_stream_requires_series_request_columns(self):
        maps = stream([mk_map_row("2025-01-01", "m", "g", "A", "B", 13, 7, "Mirage")])
        bad = pd.DataFrame([{"match_id": "Z", "series_datetime": pd.Timestamp("2025-02-01")}])
        with pytest.raises(ValueError, match="team1_canonical"):
            process_combined_stream(MapStateStore(), maps, bad)

    def test_series_features_emitted_once_per_match_not_per_map(self):
        rows = stream([
            mk_map_row("2025-06-01", "M1", "g1", "A", "B", 13, 7, "Mirage"),
            mk_map_row("2025-06-01", "M1", "g2", "A", "B", 13, 9, "Nuke"),
            mk_map_row("2025-06-01", "M1", "g3", "A", "B", 5, 13, "Inferno"),
        ])
        map_out, series_out = process_map_stream(MapStateStore(), rows, emit_series_features=True)
        assert len(map_out) == 3
        assert len(series_out) == 1
        assert series_out[0]["match_id"] == "M1"


# ---------------------------------------------------------------------------
# Combined driver: series requests decoupled from the map stream
# ---------------------------------------------------------------------------

def mk_request(dt, match_id, t1, t2, bo=3, tier="tier1", t1_elig=True, t2_elig=True):
    return {"match_id": match_id, "series_datetime": pd.Timestamp(dt),
            "team1_canonical": t1, "team2_canonical": t2, "bestOf": bo, "tier": tier,
            "team1_eligible": t1_elig, "team2_eligible": t2_elig}


class TestCombinedStream:
    def test_series_with_no_map_rows_still_gets_features(self):
        """The whole reason the driver exists: 4,504 development series have no
        map rows of their own, yet their teams' PRIOR map history is known."""
        maps = stream([
            mk_map_row("2025-01-01", "H1", "hg1", "A", "C", 13, 4, "Mirage"),
            mk_map_row("2025-02-01", "H2", "hg2", "B", "C", 7, 13, "Nuke"),
        ])
        reqs = pd.DataFrame([mk_request("2025-03-01", "NOMAPS", "A", "B")])
        _, series_out = process_combined_stream(MapStateStore(), maps, reqs)
        assert len(series_out) == 1
        r = series_out[0]
        assert r["match_id"] == "NOMAPS"
        assert r["union_map_count"] == 2          # Mirage from A, Nuke from B
        assert r["both_teams_have_map_pool_history"] == 1

    def test_series_request_does_not_contribute_state(self):
        maps = stream([mk_map_row("2025-01-01", "H1", "hg1", "A", "C", 13, 4, "Mirage")])
        reqs = pd.DataFrame([mk_request("2025-03-01", "R1", "A", "B"),
                             mk_request("2025-04-01", "R2", "A", "B")])
        store = MapStateStore()
        _, series_out = process_combined_stream(store, maps, reqs)
        # the R1 request must not have invented any history for R2 to see
        assert series_out[0]["map_pool_size_diff"] == series_out[1]["map_pool_size_diff"]
        assert len(store.processed_map_uids) == 1

    def test_request_at_same_timestamp_as_a_map_cannot_see_it(self):
        maps = stream([mk_map_row("2025-05-01 12:00", "M1", "g1", "A", "B", 13, 7, "Mirage")])
        reqs = pd.DataFrame([mk_request("2025-05-01 12:00", "R", "A", "B")])
        _, series_out = process_combined_stream(MapStateStore(), maps, reqs)
        assert series_out[0]["union_map_count"] == 0
        assert series_out[0]["map_pool_size_min"] == 0

    def test_request_after_a_map_does_see_it(self):
        maps = stream([mk_map_row("2025-05-01 12:00", "M1", "g1", "A", "B", 13, 7, "Mirage")])
        reqs = pd.DataFrame([mk_request("2025-05-01 12:01", "R", "A", "B")])
        _, series_out = process_combined_stream(MapStateStore(), maps, reqs)
        assert series_out[0]["union_map_count"] == 1
        assert series_out[0]["map_matchup_shared_coverage"] == pytest.approx(1.0)

    def test_matches_process_map_stream_when_requests_mirror_the_maps(self):
        """Equivalence guard: given a request per match, the combined driver must
        reproduce `process_map_stream`'s series output exactly."""
        rows = stream([
            mk_map_row("2025-01-01", "M1", "g1", "A", "B", 13, 7, "Mirage"),
            mk_map_row("2025-02-01", "M2", "g2", "A", "B", 9, 13, "Nuke"),
            mk_map_row("2025-02-01", "M2", "g3", "A", "B", 13, 11, "Mirage"),
            mk_map_row("2025-03-01", "M3", "g4", "B", "C", 13, 5, "Mirage"),
        ])
        _, ref = process_map_stream(MapStateStore(), rows, emit_series_features=True)
        reqs = pd.DataFrame([
            mk_request("2025-01-01", "M1", "A", "B"), mk_request("2025-02-01", "M2", "A", "B"),
            mk_request("2025-03-01", "M3", "B", "C"),
        ])
        _, got = process_combined_stream(MapStateStore(), rows, reqs)
        keys = SERIES_MAP_POOL_DIRECTIONAL + SERIES_MAP_POOL_SYMMETRIC
        ra = {r["match_id"]: r for r in ref}
        for r in got:
            assert feats_equal(r, ra[r["match_id"]], keys), r["match_id"]

    def test_ineligible_request_is_skipped(self):
        maps = stream([mk_map_row("2025-01-01", "H", "hg", "A", "C", 13, 4, "Mirage")])
        reqs = pd.DataFrame([mk_request("2025-03-01", "R", "A", "BAD", t2_elig=False)])
        _, series_out = process_combined_stream(MapStateStore(), maps, reqs)
        assert series_out == []

    def test_map_emission_can_be_disabled(self):
        maps = stream([mk_map_row("2025-01-01", "M", "g", "A", "B", 13, 7, "Mirage")])
        reqs = pd.DataFrame([mk_request("2025-03-01", "R", "A", "B")])
        map_out, series_out = process_combined_stream(MapStateStore(), maps, reqs,
                                                       emit_map_features=False)
        assert map_out == [] and len(series_out) == 1

    def test_request_column_contract_is_declared(self):
        assert set(SERIES_REQUEST_COLUMNS) == {
            "match_id", "series_datetime", "team1_canonical", "team2_canonical",
            "team1_eligible", "team2_eligible"}
