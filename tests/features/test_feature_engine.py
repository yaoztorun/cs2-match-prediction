"""
Synthetic-fixture proofs for the Phase 3 reusable feature engine
(feature_engineering/series/feature_engine.py). No dependency on the real dataset - every test
constructs its own tiny fixture. Maps to the leakage-test letters A-J from
the Phase 3 spec, plus the offline-batch vs single-call equivalence test
required for future Mode C (sequential prediction) support.
"""

import math

import pandas as pd
import pytest

from feature_engineering.series.feature_engine import (
    StateStore, HistoryEntry, TeamState,
    build_features, update_state, process_chronological_stream,
    elo_expected, elo_update, ELO_INITIAL, ELO_K,
)


def mk_row(dt, uid, t1, t2, s1, s2, bo=3, tier="tier1", t1_elig=True, t2_elig=True, source="test"):
    return {
        "datetime": pd.Timestamp(dt), "source": source, "source_match_id": uid.split(":")[-1],
        "canonical_match_uid": uid, "team1_canonical": t1, "team2_canonical": t2,
        "best_of": bo, "tier": tier, "score1": s1, "score2": s2,
        "team1_win": 1 if s1 > s2 else 0, "team1_eligible": t1_elig, "team2_eligible": t2_elig,
        "match_id": uid,
    }


# --- A. Strict chronology ---

def test_strict_chronology_excludes_future_matches():
    store = StateStore()
    t0, t1 = pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-10")
    process_chronological_stream(store, pd.DataFrame([mk_row(t1, "x:1", "A", "B", 2, 0)]))

    f_before = build_features(store, "A", "B", t0, 3)
    assert f_before["team1_total_matches_before"] == 0
    assert f_before["team2_total_matches_before"] == 0

    f_after = build_features(store, "A", "B", t1 + pd.Timedelta(days=1), 3)
    assert f_after["team1_total_matches_before"] == 1


def test_history_time_strictly_less_than_current_match_time():
    """A match AT exactly t must not count as its own history when querying at t."""
    t = pd.Timestamp("2024-05-01 12:00:00")
    state = TeamState(canonical_name="A")
    state.history.append(HistoryEntry(dt=t, source="s", source_match_id="1", canonical_match_uid="s:1",
                                       opponent_canonical="B", opponent_identity_trusted=True,
                                       best_of=3, win=1, margin=2))
    store = StateStore()
    store.teams["A"] = state
    f = build_features(store, "A", "B", t, 3)
    assert f["team1_total_matches_before"] == 0


# --- B. Same-timestamp safety ---

def test_same_timestamp_matches_do_not_see_each_other():
    store = StateStore()
    t0 = pd.Timestamp("2024-02-01 10:00:00")
    rows = pd.DataFrame([mk_row(t0, "x:1", "A", "B", 2, 0), mk_row(t0, "x:2", "A", "C", 2, 1)])
    processed, _ = process_chronological_stream(store, rows)

    m1 = next(p for p in processed if p["canonical_match_uid"] == "x:1")
    m2 = next(p for p in processed if p["canonical_match_uid"] == "x:2")
    assert m1["team1_elo_before"] == m2["team1_elo_before"] == ELO_INITIAL
    assert m1["team1_total_matches_before"] == m2["team1_total_matches_before"] == 0
    assert len(store.get("A").history) == 2  # both results applied AFTER the batch


# --- C. Target-update ordering ---

def test_target_update_ordering():
    store = StateStore()
    t0 = pd.Timestamp("2024-03-01")
    processed, _ = process_chronological_stream(store, pd.DataFrame([mk_row(t0, "x:1", "A", "B", 2, 1)]))
    assert processed[0]["team1_total_matches_before"] == 0
    assert processed[0]["team1_elo_before"] == ELO_INITIAL
    assert store.get("A").elo != ELO_INITIAL
    assert len(store.get("A").history) == 1


# --- F. ELO sanity ---

def test_elo_expected_5050_for_equal_ratings():
    assert elo_expected(1500, 1500) == pytest.approx(0.5)


def test_elo_update_conserves_total_and_moves_correctly():
    new_a, new_b = elo_update(1500, 1500, 1.0)
    assert new_a > 1500
    assert new_b < 1500
    assert (new_a + new_b) == pytest.approx(3000.0)
    assert new_a == pytest.approx(1500 + ELO_K * 0.5)


# --- G. Rolling window correctness ---

def test_rolling_window_uses_at_most_n_preceding_matches():
    store = StateStore()
    t = pd.Timestamp("2024-01-01")
    rows = [mk_row(t + pd.Timedelta(days=i), f"x:{i}", "A", f"Opp{i}",
                    2 if i % 2 == 0 else 0, 0 if i % 2 == 0 else 2) for i in range(12)]
    process_chronological_stream(store, pd.DataFrame(rows))

    query_t = t + pd.Timedelta(days=100)
    f = build_features(store, "A", "NeverSeen", query_t, 3)
    hist = sorted(store.get("A").history, key=lambda h: h.dt)
    assert len(hist) == 12

    last5, last10 = hist[-5:], hist[-10:]
    assert f["team1_win_rate_last_5"] == pytest.approx(sum(h.win for h in last5) / 5)
    assert f["team1_win_rate_last_10"] == pytest.approx(sum(h.win for h in last10) / 10)
    assert f["team1_total_matches_before"] == 12


def test_rolling_window_with_fewer_than_n_matches_uses_all_available():
    store = StateStore()
    t = pd.Timestamp("2024-01-01")
    rows = [mk_row(t + pd.Timedelta(days=i), f"x:{i}", "A", f"Opp{i}", 2, 0) for i in range(3)]
    process_chronological_stream(store, pd.DataFrame(rows))
    f = build_features(store, "A", "NeverSeen", t + pd.Timedelta(days=100), 3)
    assert f["team1_win_rate_last_5"] == 1.0
    assert f["team1_win_rate_last_10"] == 1.0


def test_zero_history_cold_start_defaults():
    store = StateStore()
    f = build_features(store, "Fresh1", "Fresh2", pd.Timestamp("2024-01-01"), 3)
    assert f["team1_overall_win_rate_smoothed"] == pytest.approx(0.5)
    assert f["team1_win_rate_last_5"] == pytest.approx(0.5)
    assert f["team1_win_rate_last_10"] == pytest.approx(0.5)
    assert f["team1_format_win_rate_smoothed"] == pytest.approx(0.5)
    assert f["team1_avg_series_margin_last_5"] == 0
    assert f["team1_avg_series_margin_last_10"] == 0
    assert f["team1_elo_before"] == ELO_INITIAL
    assert math.isnan(f["team1_days_since_last_match"])
    assert f["team1_total_matches_before"] == 0
    assert f["both_teams_have_history"] == 0


# --- H. Format isolation ---

def test_format_isolation_bo1_does_not_leak_into_bo3():
    store = StateStore()
    t = pd.Timestamp("2024-01-01")
    rows = [
        mk_row(t, "x:1", "A", "B", 1, 0, bo=1),
        mk_row(t + pd.Timedelta(days=1), "x:2", "A", "C", 0, 2, bo=1),
        mk_row(t + pd.Timedelta(days=2), "x:3", "A", "D", 2, 0, bo=3),
    ]
    process_chronological_stream(store, pd.DataFrame(rows))
    query_t = t + pd.Timedelta(days=100)

    f_bo3 = build_features(store, "A", "NeverSeen", query_t, 3)
    f_bo1 = build_features(store, "A", "NeverSeen", query_t, 1)

    assert f_bo3["team1_format_win_rate_smoothed"] == pytest.approx((1 + 2) / (1 + 4))
    assert f_bo1["team1_format_win_rate_smoothed"] == pytest.approx((1 + 2) / (2 + 4))
    assert f_bo3["team1_overall_win_rate_smoothed"] == pytest.approx((2 + 2) / (3 + 4))


# --- I. Side symmetry ---

def test_side_symmetry_of_diff_and_confidence_features():
    store = StateStore()
    t = pd.Timestamp("2024-01-01")
    rows = [
        mk_row(t, "x:1", "A", "C", 2, 0),
        mk_row(t + pd.Timedelta(days=1), "x:2", "A", "D", 2, 1),
        mk_row(t + pd.Timedelta(days=2), "x:3", "B", "E", 0, 2),
    ]
    process_chronological_stream(store, pd.DataFrame(rows))
    query_t = t + pd.Timedelta(days=50)

    f_ab = build_features(store, "A", "B", query_t, 3)
    f_ba = build_features(store, "B", "A", query_t, 3)

    diff_keys = ["elo_diff", "overall_win_rate_diff", "win_rate_last_5_diff", "win_rate_last_10_diff",
                 "format_win_rate_diff", "avg_series_margin_last_5_diff", "avg_series_margin_last_10_diff",
                 "matches_last_30_days_diff", "total_matches_before_diff"]
    for k in diff_keys:
        assert f_ab[k] == pytest.approx(-f_ba[k]), f"{k} did not negate under side swap"

    conf_keys = ["history_matches_min", "history_matches_sum", "both_teams_have_history",
                 "both_teams_have_5_matches", "both_teams_have_10_matches"]
    for k in conf_keys:
        assert f_ab[k] == f_ba[k], f"{k} changed under side swap (should stay symmetric)"


def test_side_symmetry_target_flip():
    s1, s2 = 2, 1
    team1_win_original = 1 if s1 > s2 else 0
    team1_win_swapped = 1 if s2 > s1 else 0
    assert team1_win_swapped == 1 - team1_win_original


# --- J. Identity exclusion ---

def test_identity_exclusion_blocks_elo_but_not_independent_stats():
    store = StateStore()
    t = pd.Timestamp("2024-01-01")
    rows = pd.DataFrame([mk_row(t, "x:1", "Trusted", "Untrusted", 1, 0, bo=1, t1_elig=True, t2_elig=False)])
    processed, excluded = process_chronological_stream(store, rows)

    assert len(processed) == 0
    assert len(excluded) == 1
    assert store.get("Untrusted") is None

    trusted = store.get("Trusted")
    assert trusted is not None
    assert trusted.elo == ELO_INITIAL          # pair-dependent ELO must NOT update
    assert len(trusted.history) == 1            # independent stats DO update
    assert trusted.history[0].opponent_identity_trusted is False
    assert trusted.history[0].win == 1


def test_update_elo_requires_both_sides_eligible():
    store = StateStore()
    with pytest.raises(ValueError):
        update_state(store, "A", "B", winner="team1", source="s", source_match_id="1",
                      canonical_match_uid="s:1", prediction_time=pd.Timestamp("2024-01-01"),
                      best_of=3, score1=2, score2=0, update_team1=True, update_team2=False, update_elo=True)


# --- Offline batch vs single "future prediction" call equivalence ---

def test_batch_vs_single_call_equivalence():
    """Feature generation during offline dataset construction (process_chronological_stream)
    and through the future 'predict one match at time t' interface (direct
    build_features/update_state calls) must produce identical features given
    identical prior history - this is what makes Mode C (future sequential
    prediction) safe to build later without rewriting feature logic."""
    t = pd.Timestamp("2024-01-01")
    rows = [mk_row(t + pd.Timedelta(days=i), f"x:{i}", "A", f"Opp{i}", 2, 1) for i in range(7)]

    store_batch = StateStore()
    process_chronological_stream(store_batch, pd.DataFrame(rows))

    store_seq = StateStore()
    for r in rows:
        build_features(store_seq, r["team1_canonical"], r["team2_canonical"], r["datetime"], r["best_of"], r["tier"])
        update_state(store_seq, r["team1_canonical"], r["team2_canonical"],
                      winner=("team1" if r["team1_win"] == 1 else "team2"),
                      source=r["source"], source_match_id=r["source_match_id"],
                      canonical_match_uid=r["canonical_match_uid"], prediction_time=r["datetime"],
                      best_of=r["best_of"], score1=r["score1"], score2=r["score2"])

    query_t = t + pd.Timedelta(days=100)
    f_batch = build_features(store_batch, "A", "SomeoneNew", query_t, 3)
    f_seq = build_features(store_seq, "A", "SomeoneNew", query_t, 3)

    assert f_batch.keys() == f_seq.keys()
    for k in f_batch:
        va, vb = f_batch[k], f_seq[k]
        if isinstance(va, float) and math.isnan(va):
            assert isinstance(vb, float) and math.isnan(vb), f"NaN mismatch on {k}"
        elif isinstance(va, float):
            assert va == pytest.approx(vb), f"mismatch on {k}: {va} vs {vb}"
        else:
            assert va == vb, f"mismatch on {k}: {va} vs {vb}"
