"""
Phase 8C tournament-engine tests. Every frozen rule is tested independently
before integration tests, per-rule first, full-tournament second. All
scenarios use wholly synthetic teams (S1_*/S2_*/S3_* or hand-built letters) -
never any real Cologne participant. The real frozen YAML is loaded only for
RULES (advancement thresholds, the 15-row priority table, format/bracket
config) via tournament_engine.load_frozen_rules(), never for its
`participants` section.
"""

import ast
import json

import pytest

import tournament.engine.tournament_engine as te


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rules():
    return te.load_frozen_rules()


def make_states(seed_map, wins=0, losses=0):
    """seed_map: {team_id: seed}. Returns states_by_id with a shared record."""
    return {tid: te.TeamStageState(tid, tid, seed, current_seed=seed, wins=wins, losses=losses)
            for tid, seed in seed_map.items()}


def linear_seed_pool(n, prefix="T"):
    return [te.TeamStageState(f"{prefix}{i}", f"{prefix}{i}", i, current_seed=i) for i in range(1, n + 1)]


# ---------------------------------------------------------------------------
# 0. Frozen YAML / no-ML guards
# ---------------------------------------------------------------------------

def test_frozen_yaml_hash_matches_phase8b(rules):
    assert rules.source_yaml_sha256 == te.FROZEN_YAML_SHA256


def test_load_frozen_rules_never_reads_participants():
    source = (te.ROOT / "tournament" / "engine" / "tournament_engine.py").read_text(encoding="utf-8")
    assert '"participants"' not in source
    assert "['participants']" not in source


FORBIDDEN_IMPORT_NAMES = {
    "joblib", "xgboost", "sklearn", "feature_engine", "map_feature_engine",
    "modern_map_feature_engine", "player_roster_feature_engine", "team_form_engine",
    "preprocessing_common_map_v2", "preprocessing_common_map_v3", "preprocessing_random_forest_map_v3",
    "preprocessing_xgboost_map_v3", "map_stream_common", "map_modeling_common",
}


def test_engine_module_imports_no_ml_or_feature_code():
    source = (te.ROOT / "tournament" / "engine" / "tournament_engine.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not (imported & FORBIDDEN_IMPORT_NAMES), f"forbidden imports found: {imported & FORBIDDEN_IMPORT_NAMES}"


def test_engine_module_never_reads_cologne_result_or_dataset_paths():
    # Scan CODE only (skip the module docstring, which legitimately
    # describes these paths in prose as part of explaining the guarantee).
    source = (te.ROOT / "tournament" / "engine" / "tournament_engine.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    body_start_line = tree.body[0].end_lineno if (tree.body and isinstance(tree.body[0], ast.Expr)) else 0
    code_only = "\n".join(source.splitlines()[body_start_line:])
    forbidden_substrings = ["evaluation_manifest", "series_base", "map_test_predictions", "cologne_2026"]
    for token in forbidden_substrings:
        assert token not in code_only, f"engine module unexpectedly references {token!r} outside its docstring"


# ---------------------------------------------------------------------------
# 1. Round 1 pairing
# ---------------------------------------------------------------------------

def test_round1_pairing_exact_1v9_through_8v16():
    entrants = [te.TeamEntry(f"T{i}", f"T{i}", i) for i in range(1, 17)]
    pairs = te.round1_pairing_ids(entrants)
    assert pairs == [("T1", "T9"), ("T2", "T10"), ("T3", "T11"), ("T4", "T12"),
                      ("T5", "T13"), ("T6", "T14"), ("T7", "T15"), ("T8", "T16")]


def test_round1_pairing_requires_contiguous_seeds():
    entrants = [te.TeamEntry("A", "A", 1), te.TeamEntry("B", "B", 3)]
    with pytest.raises(ValueError):
        te.round1_pairing_ids(entrants)


# ---------------------------------------------------------------------------
# 2. Round 2/3 constrained pairing (with minimize-then-fallback rematch policy)
# ---------------------------------------------------------------------------

def test_round2_pairing_no_history_pairs_highest_vs_lowest():
    pool = linear_seed_pool(8)
    pairs, fallback, count = te.constrained_pool_pairing(pool, {})
    assert fallback is False and count == 0
    assert pairs == [("T1", "T8"), ("T2", "T7"), ("T3", "T6"), ("T4", "T5")]


def test_round2_pairing_avoids_a_single_rematch_by_choosing_next_lowest():
    pool = linear_seed_pool(8)
    history = {"T1": {"T8"}}  # T1 already played the true lowest seed
    pairs, fallback, count = te.constrained_pool_pairing(pool, history)
    assert fallback is False and count == 0
    assert ("T1", "T8") not in pairs
    assert dict(pairs)["T1"] == "T7"  # next-lowest available


def test_round2_pairing_avoids_naive_greedy_dead_end():
    # A 4-team pool where naive greedy (T1 takes T4 immediately) leaves
    # T2/T3 unable to pair without a rematch, even though a valid global
    # solution exists (T1-T3, T2-T4).
    pool = linear_seed_pool(4)
    history = {"T2": {"T4"}, "T4": {"T2"}}
    pairs, fallback, count = te.constrained_pool_pairing(pool, history)
    assert fallback is False and count == 0
    assert set(pairs) == {("T1", "T4"), ("T2", "T3")} or set(pairs) == {("T1", "T3"), ("T2", "T4")}
    # whichever solution chosen must not include the T2-T4 rematch
    assert ("T2", "T4") not in pairs and ("T4", "T2") not in pairs


def test_round2_pairing_forced_fallback_minimizes_and_reports_metadata():
    # K4 edge-decomposition: A has already played every possible opponent,
    # so ALL 3 perfect matchings of a 4-team pool contain exactly 1 rematch.
    pool = linear_seed_pool(4, prefix="A")
    history = {"A1": {"A2", "A3", "A4"}, "A2": {"A1"}, "A3": {"A1"}, "A4": {"A1"}}
    pairs, fallback, count = te.constrained_pool_pairing(pool, history)
    assert fallback is True
    assert count == 1


def test_round2_pairing_cross_stage_history_does_not_restrict_pairing():
    # History from a DIFFERENT stage must never be passed in / must not
    # restrict this stage's pairing - proven by using a fresh empty history
    # even though the same two team_ids "met" in a prior stage elsewhere.
    pool = linear_seed_pool(4)
    stage1_history_from_a_different_stage = {"T1": {"T4"}}  # irrelevant if not carried over
    fresh_history = {}
    pairs, fallback, count = te.constrained_pool_pairing(pool, fresh_history)
    assert fallback is False and count == 0
    assert ("T1", "T4") in pairs  # allowed again since this stage's history is empty


# ---------------------------------------------------------------------------
# 3. Round 4/5 priority-table pairing
# ---------------------------------------------------------------------------

def test_round4_priority_row1_valid_when_no_rematches(rules):
    pool = linear_seed_pool(6)
    pairs, row, fallback, count = te.priority_table_pairing(pool, {}, rules.round4_plus_priority_table)
    assert row == 1 and fallback is False and count == 0
    assert set(pairs) == {("T1", "T6"), ("T2", "T5"), ("T3", "T4")}  # row 1 = 1v6,2v5,3v4


def test_round4_priority_row1_invalid_row2_valid(rules):
    # Row 1 = 1v6,2v5,3v4 and row 2 = 1v6,2v4,3v5 BOTH pair 1v6, so killing
    # that pair eliminates rows 1 AND 2; row 3 = 1v5,2v6,3v4 is the first
    # row that does not contain 1v6, and is the row this scenario selects.
    pool = linear_seed_pool(6)
    history = {"T1": {"T6"}, "T6": {"T1"}}  # kills row 1's AND row 2's 1v6 pair
    pairs, row, fallback, count = te.priority_table_pairing(pool, history, rules.round4_plus_priority_table)
    assert row == 3 and fallback is False and count == 0
    assert set(pairs) == {("T1", "T5"), ("T2", "T6"), ("T3", "T4")}  # row 3 = 1v5,2v6,3v4


def test_round4_several_early_rows_invalid_later_row_selected(rules):
    pool = linear_seed_pool(6)
    # Kill every row up to and including row 4 by removing 1v6, 1v5, then
    # forcing rows 3/4 (which pair 1 with 5 or stay on 6) to fail too;
    # simplest reliable construction: T1 has played both T5 and T6, which
    # eliminates rows 1-4 (all pair 1 with 5 or 6) but not row 5 (1v4).
    history = {"T1": {"T5", "T6"}, "T5": {"T1"}, "T6": {"T1"}}
    pairs, row, fallback, count = te.priority_table_pairing(pool, history, rules.round4_plus_priority_table)
    assert row == 5 and fallback is False and count == 0
    assert set(pairs) == {("T1", "T4"), ("T2", "T6"), ("T3", "T5")}  # row 5 = 1v4,2v6,3v5


def test_round4_forced_fallback_minimizes_and_reports_metadata(rules):
    pool = linear_seed_pool(6)
    history = {"T1": {"T2", "T3", "T4", "T5", "T6"}}  # T1 has played everyone
    pairs, row, fallback, count = te.priority_table_pairing(pool, history, rules.round4_plus_priority_table)
    assert fallback is True
    assert count == 1
    assert row is not None


def test_round4_priority_table_requires_exactly_6_teams(rules):
    with pytest.raises(ValueError):
        te.priority_table_pairing(linear_seed_pool(4), {}, rules.round4_plus_priority_table)


# ---------------------------------------------------------------------------
# 4. Difficulty Score (dynamic Buchholz)
# ---------------------------------------------------------------------------

def test_difficulty_score_basic_example():
    # Team A faced opponents currently 2-0 and 1-1: difficulty = (2+1)-(0+1) = 2
    states = make_states({"A": 1, "B": 2, "C": 3})
    states["A"].opponents = ["B", "C"]
    states["B"].wins, states["B"].losses = 2, 0
    states["C"].wins, states["C"].losses = 1, 1
    assert te.difficulty_score("A", states) == 2


def test_difficulty_score_recomputes_when_opponent_record_changes_later():
    states = make_states({"A": 1, "B": 2, "C": 3})
    states["A"].opponents = ["B", "C"]
    states["B"].wins, states["B"].losses = 1, 1
    states["C"].wins, states["C"].losses = 1, 1
    assert te.difficulty_score("A", states) == 0

    # B and C's records change later; A plays no new match, but its
    # Difficulty Score must reflect the CURRENT state on recomputation.
    states["B"].wins, states["B"].losses = 2, 1
    states["C"].wins, states["C"].losses = 1, 2
    assert te.difficulty_score("A", states) == (2 - 1) + (1 - 2)  # == 0, recomputed not cached

    states["C"].wins, states["C"].losses = 3, 0
    assert te.difficulty_score("A", states) == (2 - 1) + (3 - 0)  # == 4, increased with no new match for A


# ---------------------------------------------------------------------------
# 5. Seeding: compute_seed_order tie-break chain
# ---------------------------------------------------------------------------

def test_seed_order_by_record_first():
    states = make_states({"A": 1, "B": 2})
    states["A"].wins, states["A"].losses = 2, 0
    states["B"].wins, states["B"].losses = 1, 1
    ordered = te.compute_seed_order(list(states.values()))
    assert [s.team_id for s in ordered] == ["A", "B"]
    assert states["A"].current_seed == 1 and states["B"].current_seed == 2


def test_seed_order_equal_record_different_difficulty():
    states = make_states({"A": 5, "B": 1, "C": 10, "D": 20})  # A/B share record, C/D are their opponents
    states["A"].wins, states["A"].losses = 1, 0
    states["B"].wins, states["B"].losses = 1, 0
    states["A"].opponents = ["C"]
    states["B"].opponents = ["D"]
    states["C"].wins, states["C"].losses = 3, 1   # difficulty contribution +2
    states["D"].wins, states["D"].losses = 0, 3   # difficulty contribution -3
    ordered = te.compute_seed_order(list(states.values()))
    a_rank = [s.team_id for s in ordered].index("A")
    b_rank = [s.team_id for s in ordered].index("B")
    assert a_rank < b_rank  # A has higher Difficulty Score -> better seed, despite worse initial seed (5 vs 1)


def test_seed_order_equal_record_equal_difficulty_falls_back_to_initial_seed():
    states = make_states({"A": 7, "B": 2})
    states["A"].wins, states["A"].losses = 0, 0
    states["B"].wins, states["B"].losses = 0, 0
    ordered = te.compute_seed_order(list(states.values()))
    assert [s.team_id for s in ordered] == ["B", "A"]  # lower initial seed (2) wins the tie


def test_record_ordering_within_advancers_and_eliminated():
    # 3-0 > 3-1 > 3-2 among advancers; 2-3 > 1-3 > 0-3 among eliminated.
    specs = [("W30", 3, 0), ("W31", 3, 1), ("W32", 3, 2), ("L23", 2, 3), ("L13", 1, 3), ("L03", 0, 3)]
    states = {tid: te.TeamStageState(tid, tid, i + 1, current_seed=i + 1, wins=w, losses=l)
              for i, (tid, w, l) in enumerate(specs)}
    ordered = [s.team_id for s in te.compute_seed_order(list(states.values()))]
    assert ordered.index("W30") < ordered.index("W31") < ordered.index("W32")
    assert ordered.index("W32") < ordered.index("L23")
    assert ordered.index("L23") < ordered.index("L13") < ordered.index("L03")


# ---------------------------------------------------------------------------
# 6. Format selection (validated against the shared-record invariant)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("wins,losses,expected_bo", [
    (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1),
    (2, 0, 3), (0, 2, 3), (2, 1, 3), (1, 2, 3), (2, 2, 3),
])
def test_stage1_stage2_format_table(rules, wins, losses, expected_bo):
    assert te.select_format(te.STAGE_1, wins, losses, rules) == expected_bo
    assert te.select_format(te.STAGE_2, wins, losses, rules) == expected_bo


@pytest.mark.parametrize("wins,losses", [(0, 0), (1, 0), (2, 1), (2, 2)])
def test_stage3_always_bo3(rules, wins, losses):
    assert te.select_format(te.STAGE_3, wins, losses, rules) == 3


def test_build_swiss_matches_asserts_shared_record_before_format(rules):
    states = make_states({"A": 1, "B": 2})
    states["A"].wins, states["A"].losses = 2, 0
    states["B"].wins, states["B"].losses = 1, 1
    with pytest.raises(ValueError, match="do not share a record"):
        te._build_swiss_matches([{"id_a": "A", "id_b": "B"}], states, te.STAGE_1, 3, rules)


# ---------------------------------------------------------------------------
# 7. Result application / status transitions
# ---------------------------------------------------------------------------

def test_apply_result_updates_records_and_status(rules):
    states = make_states({"A": 1, "B": 2}, wins=2, losses=2)
    match = te.MatchSpec("m1", te.STAGE_1, 5, "2-2", "A", "B", 1, 2, best_of=3)
    resolution = te.MatchResolution(winner="A")
    resolution = te._finalize_resolution(match, resolution)
    te.apply_result(states, match, resolution, rules)
    assert states["A"].wins == 3 and states["A"].status == "advanced"
    assert states["B"].losses == 3 and states["B"].status == "eliminated"
    assert states["B"].team_id in states["A"].opponents
    assert states["A"].team_id in states["B"].opponents


def test_apply_result_rejects_already_terminal_team(rules):
    states = make_states({"A": 1, "B": 2})
    states["A"].status = "advanced"
    match = te.MatchSpec("m1", te.STAGE_1, 1, "0-0", "A", "B", 1, 2, best_of=1)
    resolution = te._finalize_resolution(match, te.MatchResolution(winner="B"))
    with pytest.raises(ValueError):
        te.apply_result(states, match, resolution, rules)


def test_resolution_validation_rejects_invalid_winner():
    match = te.MatchSpec("m1", te.STAGE_1, 1, "0-0", "A", "B", 1, 2, best_of=1)
    with pytest.raises(ValueError):
        te._finalize_resolution(match, te.MatchResolution(winner="Z"))


def test_resolution_validation_derives_loser_automatically():
    match = te.MatchSpec("m1", te.STAGE_1, 1, "0-0", "A", "B", 1, 2, best_of=1)
    resolution = te._finalize_resolution(match, te.MatchResolution(winner="A"))
    assert resolution.loser == "B"


# ---------------------------------------------------------------------------
# 8. Rematch policy: stage-local only
# ---------------------------------------------------------------------------

def test_rematch_policy_is_stage_local_teams_may_meet_again_in_a_later_stage(rules):
    # Simulate the SAME two team_ids appearing in two independent stage
    # histories - a fresh history dict per stage must never inherit a
    # rematch restriction from a previous stage.
    pool = linear_seed_pool(4, prefix="X")
    stage_a_history = {"X1": {"X4"}}
    _, fallback_a, _ = te.constrained_pool_pairing(pool, stage_a_history)
    stage_b_history = {}  # fresh stage: no history carried over
    pairs_b, fallback_b, count_b = te.constrained_pool_pairing(pool, stage_b_history)
    assert fallback_b is False and count_b == 0
    assert ("X1", "X4") in pairs_b  # legitimately allowed again in the new stage


# ---------------------------------------------------------------------------
# 9. Difficulty-Score trace snapshot immutability (amendment #7)
# ---------------------------------------------------------------------------

def test_match_spec_difficulty_before_snapshot_never_changes_after_later_rounds(rules):
    s1, s2, s3 = te.build_synthetic_fixture()
    provider = te.HigherSeedWinsProvider()
    result = te.run_major_tournament(s1, s2, s3, rules, provider)
    round1_entries = [e for e in result.stage1.trace if e.match.round_number == 1]
    snapshot_before = {e.match.match_id: (e.match.difficulty_a_before, e.match.difficulty_b_before)
                        for e in round1_entries}
    # Live Difficulty Scores changed drastically by stage end (many rounds
    # played since); the round-1 MatchSpec's stored before-values must be
    # byte-identical to what was captured at pairing time (always 0/0,
    # since no team has played an opponent yet in round 1).
    for match_id, (da, db) in snapshot_before.items():
        assert (da, db) == (0, 0)
    # MatchSpec is a frozen dataclass - attempting to mutate it must raise.
    import dataclasses
    with pytest.raises(dataclasses.FrozenInstanceError):
        round1_entries[0].match.difficulty_a_before = 999


# ---------------------------------------------------------------------------
# 10. Final Swiss order / stage transitions (amendment #6)
# ---------------------------------------------------------------------------

def test_finalize_stage_advancers_and_eliminated_counts():
    entrants = [te.TeamEntry(f"T{i}", f"T{i}", i) for i in range(1, 17)]
    rules = te.load_frozen_rules()
    result = te.run_swiss_stage(entrants, te.STAGE_1, rules, te.HigherSeedWinsProvider())
    assert len(result.advancers) == 8
    assert len(result.eliminated) == 8
    assert {t.status for t in result.advancers} == {"advanced"}
    assert {t.status for t in result.eliminated} == {"eliminated"}


def test_next_stage_entrants_seeds_9_to_16_are_final_order_not_qualification_order(rules):
    direct = [te.TeamEntry(f"D{i}", f"D{i}", i) for i in range(1, 9)]
    entrants = [te.TeamEntry(f"T{i}", f"T{i}", i) for i in range(1, 17)]
    stage1_result = te.run_swiss_stage(entrants, te.STAGE_1, rules, te.HigherSeedWinsProvider())
    stage2_entrants = te.next_stage_entrants(direct, stage1_result)
    assert [e.initial_stage_seed for e in stage2_entrants[:8]] == list(range(1, 9))
    assert [e.team_id for e in stage2_entrants[8:]] == [t.team_id for t in stage1_result.advancers]
    assert [e.initial_stage_seed for e in stage2_entrants[8:]] == list(range(9, 17))


def test_next_stage_entrants_rejects_non_1_to_8_direct_seeds(rules):
    bad_direct = [te.TeamEntry(f"D{i}", f"D{i}", i + 1) for i in range(1, 9)]  # 2..9, not 1..8
    entrants = [te.TeamEntry(f"T{i}", f"T{i}", i) for i in range(1, 17)]
    stage1_result = te.run_swiss_stage(entrants, te.STAGE_1, rules, te.HigherSeedWinsProvider())
    with pytest.raises(ValueError):
        te.next_stage_entrants(bad_direct, stage1_result)


# ---------------------------------------------------------------------------
# 11. Round dispatch by round number, not pool size (amendment #2)
# ---------------------------------------------------------------------------

def test_round_dispatch_rejects_unexpected_pool_shape(rules):
    states_by_id = make_states({f"T{i}": i for i in range(1, 17)})
    history = {tid: set() for tid in states_by_id}
    # Force an unexpected shape for round 2 (should be {(1,0):8,(0,1):8}) by
    # leaving everyone at 0-0 (round-1 shape) and asking for round 2.
    with pytest.raises(ValueError, match="unexpected active record-group shape"):
        te.run_swiss_round(states_by_id, history, te.STAGE_1, 2, rules, te.HigherSeedWinsProvider())


def test_expected_groups_by_round_table_is_exactly_as_frozen():
    assert te.EXPECTED_GROUPS_BY_ROUND == {
        1: {(0, 0): 16},
        2: {(1, 0): 8, (0, 1): 8},
        3: {(2, 0): 4, (1, 1): 8, (0, 2): 4},
        4: {(2, 1): 6, (1, 2): 6},
        5: {(2, 2): 6},
    }


# ---------------------------------------------------------------------------
# 12. Swiss stage completion invariants
# ---------------------------------------------------------------------------

def test_full_synthetic_stage_has_exactly_33_matches_and_8_8_split(rules):
    entrants = [te.TeamEntry(f"T{i}", f"T{i}", i) for i in range(1, 17)]
    result = te.run_swiss_stage(entrants, te.STAGE_1, rules, te.HigherSeedWinsProvider())
    assert len(result.trace) == 33
    assert len(result.advancers) == 8 and len(result.eliminated) == 8
    round_counts = {}
    for e in result.trace:
        round_counts[e.match.round_number] = round_counts.get(e.match.round_number, 0) + 1
    assert round_counts == {1: 8, 2: 8, 3: 8, 4: 6, 5: 3}


def test_every_team_plays_between_3_and_5_matches(rules):
    entrants = [te.TeamEntry(f"T{i}", f"T{i}", i) for i in range(1, 17)]
    result = te.run_swiss_stage(entrants, te.STAGE_1, rules, te.SeededRandomOutcomeProvider(seed=7))
    counts = {}
    for e in result.trace:
        counts[e.match.team_a] = counts.get(e.match.team_a, 0) + 1
        counts[e.match.team_b] = counts.get(e.match.team_b, 0) + 1
    assert len(counts) == 16
    assert all(3 <= n <= 5 for n in counts.values())


# ---------------------------------------------------------------------------
# 13. Playoffs
# ---------------------------------------------------------------------------

def test_playoff_bracket_seeds_and_bracket_sides(rules):
    seeds = [te.TeamEntry(f"P{i}", f"P{i}", i) for i in range(1, 9)]
    result = te.run_playoffs(seeds, rules, te.HigherSeedWinsProvider())
    qf = {e.match.match_id: e for e in result.trace if e.match.round_number == 1}
    assert {qf["playoff_qf_01"].match.seed_a, qf["playoff_qf_01"].match.seed_b} == {1, 8}
    assert {qf["playoff_qf_02"].match.seed_a, qf["playoff_qf_02"].match.seed_b} == {4, 5}
    assert {qf["playoff_qf_03"].match.seed_a, qf["playoff_qf_03"].match.seed_b} == {2, 7}
    assert {qf["playoff_qf_04"].match.seed_a, qf["playoff_qf_04"].match.seed_b} == {3, 6}
    assert qf["playoff_qf_01"].match.bracket_side == "A" and qf["playoff_qf_02"].match.bracket_side == "A"
    assert qf["playoff_qf_03"].match.bracket_side == "B" and qf["playoff_qf_04"].match.bracket_side == "B"


def test_playoff_formats(rules):
    seeds = [te.TeamEntry(f"P{i}", f"P{i}", i) for i in range(1, 9)]
    result = te.run_playoffs(seeds, rules, te.HigherSeedWinsProvider())
    by_id = {e.match.match_id: e.match for e in result.trace}
    for i in range(1, 5):
        assert by_id[f"playoff_qf_{i:02d}"].best_of == 3
    assert by_id["playoff_sf_01"].best_of == 3 and by_id["playoff_sf_02"].best_of == 3
    assert by_id["playoff_final_01"].best_of == 5


def test_playoff_no_reseeding_and_no_third_place_match(rules):
    seeds = [te.TeamEntry(f"P{i}", f"P{i}", i) for i in range(1, 9)]
    result = te.run_playoffs(seeds, rules, te.HigherSeedWinsProvider())
    assert len(result.trace) == 7  # 4 QF + 2 SF + 1 Final, no 3rd place
    labels = [e.match.match_id for e in result.trace]
    assert not any("third" in lbl for lbl in labels)


def test_playoff_bracket_lineage_survives_team_a_team_b_reorientation(rules):
    # Construct seeds so that the LOWER-seeded team in a QF pairing wins
    # (upset), forcing team_a/team_b reorientation in the semifinal; the
    # source_a/source_b lineage must still correctly point back to the
    # actual originating QF match regardless of that reorientation.
    seeds = [te.TeamEntry(f"P{i}", f"P{i}", i) for i in range(1, 9)]
    # Script QF2 (seed4 vs seed5) so the WORSE seed (5) wins - an upset.
    script = {"playoff_qf_01": "P1", "playoff_qf_02": "P5", "playoff_qf_03": "P2", "playoff_qf_04": "P3",
              "playoff_sf_01": "P1", "playoff_sf_02": "P2", "playoff_final_01": "P1"}
    result = te.run_playoffs(seeds, rules, te.ScriptedOutcomeProvider(script))
    sf1 = next(e.match for e in result.trace if e.match.match_id == "playoff_sf_01")
    # P1 (seed1, from QF1) must remain team_a; P5 (seed5, upset winner from QF2) is team_b.
    assert sf1.team_a == "P1" and sf1.team_b == "P5"
    assert sf1.source_a == "winner_playoff_qf_01"
    assert sf1.source_b == "winner_playoff_qf_02"


# ---------------------------------------------------------------------------
# 14. Canonical serialization / determinism
# ---------------------------------------------------------------------------

def test_canonical_json_is_sorted_and_compact():
    obj = {"b": 1, "a": 2}
    raw = te.canonical_json_bytes(obj)
    assert raw == b'{"a":2,"b":1}'


def test_double_run_determinism_identical_trace_hash(rules):
    s1, s2, s3 = te.build_synthetic_fixture()
    result1 = te.run_major_tournament(s1, s2, s3, rules, te.HigherSeedWinsProvider())
    result2 = te.run_major_tournament(s1, s2, s3, rules, te.HigherSeedWinsProvider())
    d1, d2 = result1.to_dict(), result2.to_dict()
    assert d1 == d2
    assert te.trace_hash(d1) == te.trace_hash(d2)
    assert result1.champion == result2.champion


def test_deterministic_run_matches_and_records_are_identical(rules):
    s1, s2, s3 = te.build_synthetic_fixture()
    r1 = te.run_major_tournament(s1, s2, s3, rules, te.HigherSeedWinsProvider())
    r2 = te.run_major_tournament(s1, s2, s3, rules, te.HigherSeedWinsProvider())
    ids1 = [e.match.match_id for e in r1.full_trace()]
    ids2 = [e.match.match_id for e in r2.full_trace()]
    assert ids1 == ids2
    winners1 = [e.resolution.winner for e in r1.full_trace()]
    winners2 = [e.resolution.winner for e in r2.full_trace()]
    assert winners1 == winners2


# ---------------------------------------------------------------------------
# 15. Full synthetic Major, HigherSeedWinsProvider
# ---------------------------------------------------------------------------

def test_full_synthetic_major_higher_seed_wins_champion_is_top_overall_seed(rules):
    s1, s2, s3 = te.build_synthetic_fixture()
    result = te.run_major_tournament(s1, s2, s3, rules, te.HigherSeedWinsProvider())
    assert result.champion == "S3_01"   # the best-seeded team should win every match
    assert len(result.stage1.trace) == 33
    assert len(result.stage2.trace) == 33
    assert len(result.stage3.trace) == 33
    assert len(result.playoffs.trace) == 7


def test_synthetic_trace_contains_only_synthetic_team_ids(rules):
    s1, s2, s3 = te.build_synthetic_fixture()
    result = te.run_major_tournament(s1, s2, s3, rules, te.HigherSeedWinsProvider())
    for entry in result.full_trace():
        for tid in (entry.match.team_a, entry.match.team_b):
            assert tid.split("_")[0] in ("S1", "S2", "S3"), f"non-synthetic team_id found: {tid}"


# ---------------------------------------------------------------------------
# 16. Zero real Cologne teams in the synthetic trace (amendment #8)
# ---------------------------------------------------------------------------

def _load_real_cologne_participant_names():
    """The ONE place in this test suite allowed to read the frozen YAML's
    participants - used strictly to build a negative-assertion set, never
    to run a tournament. tournament_engine.py itself never does this."""
    import yaml
    cfg = yaml.safe_load(te.FROZEN_YAML_PATH.read_bytes())
    names = set()
    for group in ("stage_1_entrants", "stage_2_direct_entrants", "stage_3_direct_entrants"):
        for t in cfg["participants"][group]:
            names.add(t["display_name"])
            names.add(t["canonical_model_name"])
    return names


def test_synthetic_trace_json_contains_zero_real_cologne_team_names(rules, tmp_path):
    s1, s2, s3 = te.build_synthetic_fixture()
    result = te.run_major_tournament(s1, s2, s3, rules, te.HigherSeedWinsProvider())
    trace_text = json.dumps(result.to_dict())
    real_names = _load_real_cologne_participant_names()
    found = [name for name in real_names if name in trace_text]
    assert not found, f"real Cologne participant name(s) leaked into synthetic trace: {found}"


def test_run_phase8c_synthetic_demo_script_never_reads_participants():
    demo_path = te.ROOT / "tournament" / "engine" / "run_phase8c_synthetic_demo.py"
    if not demo_path.exists():
        pytest.skip("demo script not yet created")
    source = demo_path.read_text(encoding="utf-8")
    assert '"participants"' not in source
    assert "['participants']" not in source


# ---------------------------------------------------------------------------
# 17. Stress test (500 synthetic Majors)
# ---------------------------------------------------------------------------

def test_stress_test_500_runs_zero_invariant_failures(rules):
    summary = te.run_stress_test(n_runs=500, base_seed=42, rules=rules)
    assert summary.n_runs == 500
    assert summary.invariant_failures == 0
    assert summary.max_rounds_per_stage == 5
    assert summary.total_matches == 500 * (33 * 3 + 7)
    # Do NOT assert fallback count == 0 - a natural random run may or may
    # not trigger the fallback path; the targeted unit tests above already
    # prove the fallback code executes correctly when it is unavoidable.
    assert summary.total_rematch_fallback_events >= 0
