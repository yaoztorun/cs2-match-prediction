"""
Tests for feature_engineering/form/team_form_engine.py (Phase 5B.2). Synthetic-fixture only -
no dependency on real data (real-data checks live in
validation/validate_phase5b2.py), matching this repo's established convention
for feature-engine test suites (tests/features/test_map_feature_engine.py).
"""

import math

import pandas as pd
import pytest

from feature_engineering.series.feature_engine import (
    ELO_INITIAL, ELO_K, elo_expected, elo_update,
    StateStore as Phase3StateStore, process_chronological_stream,
)
from feature_engineering.form.team_form_engine import (
    FormHistoryEntry, TeamFormState, TeamFormStateStore,
    apply_form_result, process_form_stream,
    compute_team_form_features, build_future_team_form_features,
    normalized_series_margin, FORM_HALF_LIFE_DAYS,
    FORM_DIRECTIONAL_FEATURES, FORM_SYMMETRIC_FEATURES,
    COLD_START_OPPONENT_ELO, COLD_START_RESIDUAL, COLD_START_WIN_RATE,
    COLD_START_MARGIN, COLD_START_HISTORY_MASS,
)


def mk_row(dt, uid, t1, t2, s1, s2, bo=3, tier="tier1", t1_elig=True, t2_elig=True, source="test"):
    return {
        "datetime": pd.Timestamp(dt), "source": source, "source_match_id": uid.split(":")[-1],
        "canonical_match_uid": uid, "team1_canonical": t1, "team2_canonical": t2,
        "best_of": bo, "tier": tier, "score1": s1, "score2": s2,
        "team1_win": 1 if s1 > s2 else 0, "team1_eligible": t1_elig, "team2_eligible": t2_elig,
        "match_id": uid,
    }


def stream(rows):
    return pd.DataFrame(rows)


def _seed_asymmetric_store():
    """A store where the two teams genuinely differ, so symmetry is a real
    test (mirrors test_map_feature_engine.py's _seed_asymmetric_store)."""
    store = TeamFormStateStore()
    events = [
        ("2025-01-05", "A", "C", 2, 0), ("2025-01-12", "A", "D", 2, 1),
        ("2025-02-05", "A", "C", 0, 2), ("2025-03-05", "A", "E", 2, 0),
        ("2025-01-20", "B", "C", 0, 2), ("2025-02-20", "B", "D", 2, 1),
        ("2025-03-20", "B", "E", 2, 0), ("2025-04-01", "B", "C", 2, 1),
    ]
    for i, (d, t1, t2, s1, s2) in enumerate(events):
        row = pd.Series(mk_row(d, f"s:{i}", t1, t2, s1, s2))
        apply_form_result(store, row)
    return store


# --- A. Pre-match ELO expectation and residual arithmetic (brief section 2) ---

def test_expected_prob_and_residual_use_pre_match_elo_not_post_update():
    store = TeamFormStateStore()
    store.teams["A"] = TeamFormState(canonical_name="A", elo=1600.0)
    store.teams["B"] = TeamFormState(canonical_name="B", elo=1400.0)
    row = pd.Series(mk_row("2024-01-01", "x:1", "A", "B", 2, 0))  # A wins
    apply_form_result(store, row)

    expected_a = elo_expected(1600.0, 1400.0)   # from PRE-match ratings
    entry_a = store.teams["A"].history[-1]
    assert entry_a.expected_win_prob == pytest.approx(expected_a)
    assert entry_a.performance_residual == pytest.approx(1 - expected_a)

    # post-update elo differs from the ratings used to compute the expectation
    assert store.teams["A"].elo != pytest.approx(1600.0)
    post_expected = elo_expected(store.teams["A"].elo, store.teams["B"].elo)
    assert entry_a.expected_win_prob != pytest.approx(post_expected)


def test_residual_sign_scenarios():
    """The brief's four scenarios, read from the relevant side each time:
    upset win (large +), expected win (small +), favorite upset loss (large -),
    underdog expected loss (small -, since their own expectation was already low)."""
    # Favorite (1700) vs underdog (1300); favorite wins as expected.
    store_win = TeamFormStateStore()
    store_win.teams["Fav"] = TeamFormState(canonical_name="Fav", elo=1700.0)
    store_win.teams["Dog"] = TeamFormState(canonical_name="Dog", elo=1300.0)
    apply_form_result(store_win, pd.Series(mk_row("2024-01-01", "w:1", "Fav", "Dog", 2, 0)))
    fav_expected_win = store_win.teams["Fav"].history[-1].performance_residual
    dog_expected_loss = store_win.teams["Dog"].history[-1].performance_residual

    # Same matchup, but the underdog upsets the favorite instead.
    store_upset = TeamFormStateStore()
    store_upset.teams["Fav"] = TeamFormState(canonical_name="Fav", elo=1700.0)
    store_upset.teams["Dog"] = TeamFormState(canonical_name="Dog", elo=1300.0)
    apply_form_result(store_upset, pd.Series(mk_row("2024-01-01", "u:1", "Fav", "Dog", 0, 2)))
    fav_upset_loss = store_upset.teams["Fav"].history[-1].performance_residual
    dog_upset_win = store_upset.teams["Dog"].history[-1].performance_residual

    assert dog_upset_win > 0.5, "underdog win against a strong favorite must give a large positive residual"
    assert 0 < fav_expected_win < 0.5, "favorite winning as expected must give a small positive residual"
    assert fav_upset_loss < -0.5, "favorite loss (upset) must give a large negative residual"
    assert -0.5 < dog_expected_loss < 0, "underdog losing as expected must give a smaller negative residual"
    assert abs(dog_expected_loss) < abs(fav_upset_loss), \
        "expected loss against a much stronger opponent must be smaller in magnitude than an upset loss as favorite"


# --- B. Trusted-opponent gating (correction #1) ---

def test_opponent_adjusted_stats_use_trusted_population_only():
    store = TeamFormStateStore()
    t = pd.Timestamp("2024-01-01")
    # A plays 3 matches: two vs trusted opponents, one vs an untrusted opponent.
    apply_form_result(store, pd.Series(mk_row(t, "x:1", "A", "TrustedC", 2, 0)))
    apply_form_result(store, pd.Series(mk_row(t + pd.Timedelta(days=1), "x:2", "A", "TrustedD", 2, 1)))
    apply_form_result(store, pd.Series(mk_row(t + pd.Timedelta(days=2), "x:3", "A", "Untrusted", 2, 0,
                                                t2_elig=False)))
    query_t = t + pd.Timedelta(days=10)
    f = compute_team_form_features(store.get("A"), query_t)
    assert f["adjusted_matches_before"] == 2, "confidence must count only TRUSTED opponent-adjusted observations"

    trusted_hist = [h for h in store.get("A").history if h.opponent_identity_trusted]
    assert len(trusted_hist) == 2
    expected_avg_opp_elo = sum(h.opponent_elo_before for h in trusted_hist) / 2
    assert f["avg_opponent_elo_last_5"] == pytest.approx(expected_avg_opp_elo)
    expected_resid = sum(h.performance_residual for h in trusted_hist) / 2
    assert f["performance_residual_last_5"] == pytest.approx(expected_resid)
    assert f["performance_residual_all"] == pytest.approx(expected_resid)


def test_time_weighted_win_rate_and_margin_use_all_eligible_history_including_untrusted():
    store = TeamFormStateStore()
    t = pd.Timestamp("2024-01-01")
    apply_form_result(store, pd.Series(mk_row(t, "x:1", "A", "Trusted", 2, 0)))
    apply_form_result(store, pd.Series(mk_row(t + pd.Timedelta(days=1), "x:2", "A", "Untrusted", 0, 2,
                                                t2_elig=False)))
    query_t = t + pd.Timedelta(days=10)
    f = compute_team_form_features(store.get("A"), query_t)

    hist_all = sorted(store.get("A").history, key=lambda h: h.dt)
    assert len(hist_all) == 2, "own history updates even when the opponent is untrusted"
    # both entries (including the untrusted-opponent one) must contribute
    weights = [0.5 ** (((query_t - h.dt).total_seconds() / 86400.0) / FORM_HALF_LIFE_DAYS) for h in hist_all]
    expected_tw_win_rate = sum(w * h.win for w, h in zip(weights, hist_all)) / sum(weights)
    assert f["time_weighted_win_rate"] == pytest.approx(expected_tw_win_rate)
    expected_tw_margin = sum(w * h.normalized_margin for w, h in zip(weights, hist_all)) / sum(weights)
    assert f["time_weighted_normalized_series_margin"] == pytest.approx(expected_tw_margin)
    assert f["time_weighted_history_mass"] == pytest.approx(sum(weights))


# --- C. Last-5 / last-10 chronology (strictly prior, count-windowed) ---

def test_last5_and_last10_are_strictly_prior_and_windowed_by_count():
    store = TeamFormStateStore()
    t0 = pd.Timestamp("2024-01-01")
    for i in range(12):
        apply_form_result(store, pd.Series(
            mk_row(t0 + pd.Timedelta(days=i), f"x:{i}", "A", f"Opp{i}", 2, 0)))
    # a match strictly AFTER the query must not count
    future_row = pd.Series(mk_row(t0 + pd.Timedelta(days=100), "x:future", "A", "OppFuture", 2, 0))
    apply_form_result(store, future_row)

    query_t = t0 + pd.Timedelta(days=50)
    f = compute_team_form_features(store.get("A"), query_t)
    assert f["adjusted_matches_before"] == 12, "future match must not be counted"

    trusted = sorted(store.get("A").history, key=lambda h: h.dt)
    trusted = [h for h in trusted if h.dt < query_t]
    last5_expected = sum(h.opponent_elo_before for h in trusted[-5:]) / 5
    last10_expected = sum(h.opponent_elo_before for h in trusted[-10:]) / 10
    assert f["avg_opponent_elo_last_5"] == pytest.approx(last5_expected)
    assert f["avg_opponent_elo_last_10"] == pytest.approx(last10_expected)


def test_history_at_exactly_query_time_does_not_count():
    t = pd.Timestamp("2024-05-01 12:00:00")
    state = TeamFormState(canonical_name="A")
    state.history.append(FormHistoryEntry(
        dt=t, source="s", source_match_id="1", canonical_match_uid="s:1",
        opponent_canonical="B", opponent_identity_trusted=True, win=1,
        own_elo_before=1500.0, opponent_elo_before=1500.0,
        expected_win_prob=0.5, performance_residual=0.5, normalized_margin=1.0))
    f = compute_team_form_features(state, t)
    assert f["adjusted_matches_before"] == 0


# --- D. 60-day exponential weighting ---

def test_recency_weight_formula_exact_values():
    from feature_engineering.form.team_form_engine import _recency_weight
    as_of = pd.Timestamp("2024-04-01")
    assert _recency_weight(as_of, as_of) == pytest.approx(1.0)
    assert _recency_weight(as_of, as_of - pd.Timedelta(days=60)) == pytest.approx(0.5)
    assert _recency_weight(as_of, as_of - pd.Timedelta(days=120)) == pytest.approx(0.25)
    assert _recency_weight(as_of, as_of - pd.Timedelta(days=180)) == pytest.approx(0.125)


# --- E. No leakage: read-only feature computation, same-timestamp isolation ---

def test_feature_computation_never_mutates_state():
    store = _seed_asymmetric_store()
    before = {name: len(ts.history) for name, ts in store.teams.items()}
    build_future_team_form_features(store, "A", "ZZZ", "2026-01-01")
    after = {name: len(ts.history) for name, ts in store.teams.items()}
    assert before == after


def test_two_series_at_the_same_instant_cannot_see_each_other():
    t0 = pd.Timestamp("2024-02-01 10:00:00")
    rows = stream([mk_row(t0, "x:1", "A", "B", 2, 0), mk_row(t0, "x:2", "A", "C", 2, 1)])
    store = TeamFormStateStore()
    processed, _ = process_form_stream(store, rows)

    m1 = next(p for p in processed if p["canonical_match_uid"] == "x:1")
    m2 = next(p for p in processed if p["canonical_match_uid"] == "x:2")
    assert m1["team1_elo_before"] == m2["team1_elo_before"] == ELO_INITIAL
    assert m1["team1_adjusted_matches_before"] == m2["team1_adjusted_matches_before"] == 0
    assert len(store.get("A").history) == 2   # both results applied only AFTER the batch was read


def test_row_order_within_a_timestamp_group_is_irrelevant_to_emitted_features():
    t0 = pd.Timestamp("2024-02-01 10:00:00")
    fwd = stream([mk_row(t0, "x:1", "A", "B", 2, 0), mk_row(t0, "x:2", "A", "C", 2, 1)])
    bwd = stream([mk_row(t0, "x:2", "A", "C", 2, 1), mk_row(t0, "x:1", "A", "B", 2, 0)])
    p1, _ = process_form_stream(TeamFormStateStore(), fwd)
    p2, _ = process_form_stream(TeamFormStateStore(), bwd)
    by_uid_1 = {r["canonical_match_uid"]: r["team1_elo_before"] for r in p1}
    by_uid_2 = {r["canonical_match_uid"]: r["team1_elo_before"] for r in p2}
    assert by_uid_1 == by_uid_2


# --- F. Cold start ---

def test_two_completely_unknown_teams_get_documented_neutral_values():
    f = build_future_team_form_features(TeamFormStateStore(), "NEW1", "NEW2", "2026-01-01")
    assert f["team1_avg_opponent_elo_last_5"] == COLD_START_OPPONENT_ELO
    assert f["team1_performance_residual_all"] == COLD_START_RESIDUAL
    assert f["team1_time_weighted_win_rate"] == COLD_START_WIN_RATE
    assert f["team1_time_weighted_normalized_series_margin"] == COLD_START_MARGIN
    assert f["team1_time_weighted_history_mass"] == COLD_START_HISTORY_MASS
    assert f["opponent_adjusted_history_min"] == 0
    assert f["both_teams_have_5_adjusted_matches"] == 0
    assert f["both_teams_have_10_adjusted_matches"] == 0
    assert f["time_weighted_history_mass_min"] == 0.0


# --- G. Side-swap symmetry ---

def test_directional_features_negate_and_confidence_features_unchanged_under_swap():
    store = _seed_asymmetric_store()
    query_t = pd.Timestamp("2025-06-01")
    f_ab = build_future_team_form_features(store, "A", "B", query_t)
    f_ba = build_future_team_form_features(store, "B", "A", query_t)

    nontrivial = 0
    for k in FORM_DIRECTIONAL_FEATURES:
        assert f_ab[k] == pytest.approx(-f_ba[k], abs=1e-12), f"{k} did not negate under side swap"
        if abs(f_ab[k]) > 1e-9:
            nontrivial += 1
    assert nontrivial >= 3, "fixture too symmetric to prove anything"

    for k in FORM_SYMMETRIC_FEATURES:
        assert f_ab[k] == pytest.approx(f_ba[k]), f"{k} changed under side swap (should stay symmetric)"


# --- H. Future-application contract: no target/score required ---

def test_build_future_team_form_features_signature_has_no_target_or_score():
    import inspect
    sig = inspect.signature(build_future_team_form_features)
    params = set(sig.parameters)
    forbidden = {"target", "winner", "score1", "score2", "team1_win", "team1_series_win"}
    assert not (params & forbidden), f"future-application API leaked a target/score parameter: {params & forbidden}"


# --- I. Phase-3 ELO parity, at the synthetic level (complements the real-data
#        exhaustive check in feature_engineering/form/build_series_features_v3_form.py) ---

def test_independent_replay_matches_phase3_elo_exactly_on_a_synthetic_stream():
    rows = stream([
        mk_row("2024-01-01", "x:1", "A", "B", 2, 0),
        mk_row("2024-01-05", "x:2", "A", "C", 1, 2),
        mk_row("2024-01-05", "x:3", "D", "B", 2, 1),   # same timestamp as x:2, different pair
        mk_row("2024-01-10", "x:4", "C", "D", 0, 2),
        mk_row("2024-01-15", "x:5", "A", "D", 2, 1),
    ])

    phase3_store = Phase3StateStore()
    phase3_processed, _ = process_chronological_stream(phase3_store, rows)

    form_store = TeamFormStateStore()
    form_processed, _ = process_form_stream(form_store, rows)

    phase3_by_uid = {r["canonical_match_uid"]: r["elo_diff"] for r in phase3_processed}
    form_by_uid = {r["canonical_match_uid"]: r["team1_elo_before"] - r["team2_elo_before"] for r in form_processed}

    assert set(phase3_by_uid) == set(form_by_uid)
    for uid in phase3_by_uid:
        assert form_by_uid[uid] == pytest.approx(phase3_by_uid[uid], abs=1e-12), uid


# --- J. time_weighted_history_mass_min (correction #3) ---

def test_history_mass_min_is_the_min_of_each_teams_own_time_weighted_mass():
    store = TeamFormStateStore()
    t0 = pd.Timestamp("2024-01-01")
    # A: 3 matches. B: 1 match. Both strictly before query.
    for i in range(3):
        apply_form_result(store, pd.Series(mk_row(t0 + pd.Timedelta(days=i), f"a:{i}", "A", f"OppA{i}", 2, 0)))
    apply_form_result(store, pd.Series(mk_row(t0, "b:0", "B", "OppB0", 2, 0)))

    query_t = t0 + pd.Timedelta(days=30)
    f = build_future_team_form_features(store, "A", "B", query_t)

    fa = compute_team_form_features(store.get("A"), query_t)
    fb = compute_team_form_features(store.get("B"), query_t)
    assert fa["time_weighted_history_mass"] > fb["time_weighted_history_mass"]
    assert f["time_weighted_history_mass_min"] == pytest.approx(
        min(fa["time_weighted_history_mass"], fb["time_weighted_history_mass"]))


def test_history_mass_is_zero_not_neutral_for_a_team_with_no_history():
    f = compute_team_form_features(None, pd.Timestamp("2026-01-01"))
    assert f["time_weighted_history_mass"] == 0.0


# --- normalized_series_margin sanity ---

def test_normalized_series_margin_bounds_and_zero_denominator():
    assert normalized_series_margin(2, 0) == pytest.approx(1.0)
    assert normalized_series_margin(0, 2) == pytest.approx(-1.0)
    assert normalized_series_margin(1, 1) == pytest.approx(0.0)
    assert normalized_series_margin(0, 0) == 0.0
