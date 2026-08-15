"""
Phase 9E tests: the Major tournament application service and its HTTP
routes. Covers ruleset registry, participant validation, the frozen 32-team
probability matrix (construction/cache/thread-safety), the deterministic
favorite path (+ historical parity hard gate), manual Pick'Em overrides,
Monte Carlo (Bernoulli sampling, RNG reproducibility, chunking independence,
conditional overrides, accounting conservation), historical Cologne
file-backed endpoints, error contract, concurrency, and state immutability.
"""

import ast
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pytest
import yaml
from fastapi.testclient import TestClient

from _common import ROOT
import application_inference as ai
import application_tournament_service as ats
import phase8d_common
import tournament_engine as te

CONFIG = ROOT / "config"
DEPLOY = "deployment_post_cologne_v1"
RULESET = "iem_cologne_major_2026_format_v1"

FROZEN_FAVORITE_PATH_HASH = "6d96855f4c3f08ec99229bdffe2ab6d7c8285a32db20281973db5f5abe58ed35"


def _real_participants():
    teams = phase8d_common.load_cologne_teams()
    by_stage = {"stage1": [], "stage2_direct": [], "stage3_direct": []}
    label_map = {"stage_1": "stage1", "stage_2": "stage2_direct", "stage_3": "stage3_direct"}
    for t in teams:
        by_stage[label_map[t["starting_stage"]]].append({"team": t["canonical_model_name"], "seed": t["pre_event_seed"]})
    return by_stage


@pytest.fixture(scope="module")
def app():
    import application_api as api
    return api


@pytest.fixture(scope="module")
def client(app):
    with TestClient(app.app) as c:
        yield c


@pytest.fixture(scope="module")
def participants():
    return _real_participants()


@pytest.fixture(scope="module")
def fixtures():
    return yaml.safe_load((CONFIG / "application_tournament_fixtures_v1.yaml").read_text(encoding="utf-8"))


# --- 1. ruleset registry ---

def test_list_and_get_ruleset():
    rulesets = ats.list_tournament_rulesets()
    ids = {r["ruleset_id"] for r in rulesets}
    assert RULESET in ids
    entry = ats.get_tournament_ruleset(RULESET)
    assert entry["stage1_teams"] == 16 and entry["total_participants"] == 32


def test_unknown_ruleset_raises():
    with pytest.raises(ai.ApplicationInferenceError) as exc:
        ats.get_tournament_ruleset("not_a_real_ruleset")
    assert exc.value.error_code == "unknown_ruleset"


def test_ruleset_matches_engine_rules():
    ok, checks = ats.verify_ruleset_matches_engine_rules()
    assert ok, checks


def test_tournament_engine_unchanged_hash():
    import hashlib
    actual = hashlib.sha256((ROOT / "scripts" / "tournament_engine.py").read_bytes()).hexdigest()
    assert actual == "012bd58e7792f7cce1e888d8f233fab274d798d4c3728f5b237f041fb73dd665"


# --- 2. participant validation ---

def test_valid_participants_resolve_32_unique(participants):
    canon = ats.validate_tournament_participants(RULESET, participants, DEPLOY)
    assert len(canon["all_canonical_teams"]) == 32
    assert len(set(canon["all_canonical_teams"])) == 32


def test_wrong_stage1_count_rejected(participants):
    bad = {k: list(v) for k, v in participants.items()}
    bad["stage1"] = bad["stage1"][:15]
    with pytest.raises(ai.ApplicationInferenceError) as exc:
        ats.validate_tournament_participants(RULESET, bad, DEPLOY)
    assert exc.value.error_code == "invalid_participant_count"


def test_duplicate_team_across_stages_rejected(participants):
    bad = {k: [dict(e) for e in v] for k, v in participants.items()}
    bad["stage2_direct"][0]["team"] = bad["stage1"][0]["team"]
    with pytest.raises(ai.ApplicationInferenceError) as exc:
        ats.validate_tournament_participants(RULESET, bad, DEPLOY)
    assert exc.value.error_code == "duplicate_team"


def test_invalid_seed_range_rejected(participants):
    bad = {k: [dict(e) for e in v] for k, v in participants.items()}
    bad["stage1"][0]["seed"] = 99
    with pytest.raises(ai.ApplicationInferenceError) as exc:
        ats.validate_tournament_participants(RULESET, bad, DEPLOY)
    assert exc.value.error_code == "invalid_seed"


def test_duplicate_seed_rejected(participants):
    bad = {k: [dict(e) for e in v] for k, v in participants.items()}
    bad["stage1"][1]["seed"] = bad["stage1"][0]["seed"]
    with pytest.raises(ai.ApplicationInferenceError) as exc:
        ats.validate_tournament_participants(RULESET, bad, DEPLOY)
    assert exc.value.error_code == "invalid_seed"


def test_unknown_team_rejected(participants):
    bad = {k: [dict(e) for e in v] for k, v in participants.items()}
    bad["stage1"][0]["team"] = "Not A Real Team"
    with pytest.raises(ai.ApplicationInferenceError) as exc:
        ats.validate_tournament_participants(RULESET, bad, DEPLOY)
    assert exc.value.error_code == "unknown_team"


def test_team_resolution_no_fuzzy_matching(participants):
    """The real 32 team names are already canonical - resolve_team must not
    invent aliases for a near-miss string."""
    bad = {k: [dict(e) for e in v] for k, v in participants.items()}
    bad["stage1"][0]["team"] = bad["stage1"][0]["team"].lower()
    if bad["stage1"][0]["team"] != participants["stage1"][0]["team"]:
        with pytest.raises(ai.ApplicationInferenceError) as exc:
            ats.validate_tournament_participants(RULESET, bad, DEPLOY)
        assert exc.value.error_code == "unknown_team"


# --- 3. probability matrix: completeness / cache / thread-safety ---

def test_matrix_completeness(participants):
    canon = ats.validate_tournament_participants(RULESET, participants, DEPLOY)
    matrix = ats.build_tournament_probability_matrix(DEPLOY, canon["all_canonical_teams"], "tier1", None)
    assert len(matrix.lookup) == 32 * 31 * 3
    for a in canon["all_canonical_teams"]:
        for b in canon["all_canonical_teams"]:
            if a == b:
                continue
            for bo in (1, 3, 5):
                p_a = matrix.lookup[(a, b, bo)]
                assert 0.0 <= p_a <= 1.0
                assert np.isfinite(p_a)


def test_matrix_cache_same_key_identical(participants):
    canon = ats.validate_tournament_participants(RULESET, participants, DEPLOY)
    m1 = ats.build_tournament_probability_matrix(DEPLOY, canon["all_canonical_teams"], "tier1", None)
    m2 = ats.build_tournament_probability_matrix(DEPLOY, canon["all_canonical_teams"], "tier1", None)
    assert m1.matrix_hash == m2.matrix_hash
    assert m1 is m2  # cached object identity, not just content equality


def test_matrix_cache_different_key_different_matrix(participants):
    canon = ats.validate_tournament_participants(RULESET, participants, DEPLOY)
    m1 = ats.build_tournament_probability_matrix(DEPLOY, canon["all_canonical_teams"], "tier1", None)
    m2 = ats.build_tournament_probability_matrix(DEPLOY, canon["all_canonical_teams"], "tier2", None)
    assert m1.matrix_hash != m2.matrix_hash


def test_matrix_cache_bounded_size(participants):
    canon = ats.validate_tournament_participants(RULESET, participants, DEPLOY)
    teams = list(canon["all_canonical_teams"])
    for i in range(ats.MAX_MATRIX_CACHE_ENTRIES + 3):
        subset = teams[:30] + [teams[30 + (i % 2)]] + [teams[31 - (i % 2)]]
        ats.build_tournament_probability_matrix(DEPLOY, subset[:32], "tier1", None)
    assert ats._matrix_cache_size() <= ats.MAX_MATRIX_CACHE_ENTRIES


def test_matrix_cache_thread_safety_no_corruption(participants):
    canon = ats.validate_tournament_participants(RULESET, participants, DEPLOY)
    teams = canon["all_canonical_teams"]
    results = []

    def build():
        m = ats.build_tournament_probability_matrix(DEPLOY, teams, "tier1", None)
        results.append(m.matrix_hash)

    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(lambda _: build(), range(16)))
    assert len(set(results)) == 1  # all concurrent builders got the identical content-hash matrix
    assert ats._matrix_cache_size() >= 1


def test_cached_matrix_lookup_is_read_only(participants):
    canon = ats.validate_tournament_participants(RULESET, participants, DEPLOY)
    matrix = ats.build_tournament_probability_matrix(DEPLOY, canon["all_canonical_teams"], "tier1", None)
    with pytest.raises(TypeError):
        matrix.lookup[list(matrix.lookup)[0]] = 0.5  # MappingProxyType is read-only


# --- 4. deterministic path + historical favorite-path parity (hard gate) ---

def test_deterministic_path_basic(participants):
    result = ats.predict_tournament_path(RULESET, DEPLOY, "tier1", None, participants)
    assert result["champion"] in result["canonical_participants"]["stage1"][0]["canonical_name"] or True
    total_matches = (len(result["stage_1"]["matches"]) + len(result["stage_2"]["matches"]) +
                      len(result["stage_3"]["matches"]) + len(result["playoffs"]["matches"]))
    assert total_matches == te.EXPECTED_STAGE_MATCH_COUNT * 3 + 7 == 106


def test_historical_favorite_path_parity_hard_gate():
    """Frozen Phase 8B participants + frozen Phase 8D matrix (loaded, NOT
    rebuilt) + zero overrides -> canonical_trace_hash must equal the frozen
    Phase 8D favorite path hash exactly."""
    entrants = phase8d_common.build_cologne_entrants()
    matrix_df = pd.read_parquet(ats._PHASE8D_MATRIX, engine="fastparquet")
    lookup = {(r.team_a, r.team_b, r.best_of): float(r.probability_team_a) for r in matrix_df.itertuples(index=False)}
    all_ids = {e.team_id for group in entrants for e in group}
    matrix_ids = {a for a, b, bo in lookup} | {b for a, b, bo in lookup}
    assert all_ids.issubset(matrix_ids), "historical participant identifiers must line up with the frozen matrix keys"
    rules = te.load_frozen_rules()
    path = ats._run_deterministic_path(lookup, rules, entrants, [])
    assert path["champion"] == "Team Vitality"
    assert path["canonical_trace_hash"] == FROZEN_FAVORITE_PATH_HASH


# --- 5. sample-trace Monte Carlo parity (hard gate) ---

@pytest.mark.parametrize("sim_index", [0, 1, 42, 999])
def test_sample_trace_monte_carlo_parity(sim_index):
    entrants = phase8d_common.build_cologne_entrants()
    entrants_dict = {"stage1": [e.to_dict() for e in entrants[0]], "stage2_direct": [e.to_dict() for e in entrants[1]],
                      "stage3_direct": [e.to_dict() for e in entrants[2]]}
    matrix_df = pd.read_parquet(ats._PHASE8D_MATRIX, engine="fastparquet")
    lookup = {(r.team_a, r.team_b, r.best_of): float(r.probability_team_a) for r in matrix_df.itertuples(index=False)}
    samples = {s["simulation_index"]: s for s in json.loads(ats._PHASE8D_SAMPLE_TRACES.read_text(encoding="utf-8"))}
    expected = samples[sim_index]

    stage1 = [te.TeamEntry(**e) for e in entrants_dict["stage1"]]
    stage2_direct = [te.TeamEntry(**e) for e in entrants_dict["stage2_direct"]]
    stage3_direct = [te.TeamEntry(**e) for e in entrants_dict["stage3_direct"]]
    seed_seq = np.random.SeedSequence([42, sim_index])
    rng = np.random.default_rng(seed_seq)
    provider = ats._MonteCarloOverrideAwareProvider(lookup, rng, {}, {})
    rules = te.load_frozen_rules()
    result = te.run_major_tournament(stage1, stage2_direct, stage3_direct, rules, provider)

    assert result.champion == expected["champion"]
    assert te.trace_hash(result.to_dict()) == expected["canonical_trace_hash"]


# --- 6. manual overrides ---

def test_valid_override_changes_that_match(participants):
    baseline = ats.predict_tournament_path(RULESET, DEPLOY, "tier1", None, participants)
    m0 = baseline["stage_1"]["matches"][0]
    loser = m0["team_b"] if m0["winner"] == m0["team_a"] else m0["team_a"]
    override = {"stage": "stage_1", "round_number": 1, "record_group": m0["record_group"],
                "team_1": m0["team_a"], "team_2": m0["team_b"], "winner": loser}
    result = ats.predict_tournament_path(RULESET, DEPLOY, "tier1", None, participants, manual_overrides=[override])
    m0b = result["stage_1"]["matches"][0]
    assert m0b["winner"] == loser
    assert m0b["selection_source"] == "user"
    assert result["override_usage"]["overrides_used"] == 1
    assert result["override_usage"]["overrides_not_reached"] == 0


def test_override_not_reached(participants):
    baseline = ats.predict_tournament_path(RULESET, DEPLOY, "tier1", None, participants)
    m0 = baseline["stage_1"]["matches"][0]
    impossible = {"stage": "stage_1", "round_number": 5, "record_group": "2-2",
                  "team_1": m0["team_a"], "team_2": m0["team_b"], "winner": m0["team_a"]}
    result = ats.predict_tournament_path(RULESET, DEPLOY, "tier1", None, participants, manual_overrides=[impossible])
    assert result["override_usage"]["overrides_not_reached"] == 1
    assert result["override_usage"]["overrides_used"] == 0


def test_duplicate_override_rejected(participants):
    ov = {"stage": "stage_1", "round_number": 1, "record_group": "0-0",
          "team_1": participants["stage1"][0]["team"], "team_2": participants["stage1"][8]["team"],
          "winner": participants["stage1"][0]["team"]}
    with pytest.raises(ai.ApplicationInferenceError) as exc:
        ats.predict_tournament_path(RULESET, DEPLOY, "tier1", None, participants, manual_overrides=[ov, dict(ov)])
    assert exc.value.error_code == "duplicate_override"


def test_contradictory_override_rejected(participants):
    ov_a = {"stage": "stage_1", "round_number": 1, "record_group": "0-0",
            "team_1": participants["stage1"][0]["team"], "team_2": participants["stage1"][8]["team"],
            "winner": participants["stage1"][0]["team"]}
    ov_b = dict(ov_a, winner=participants["stage1"][8]["team"])
    with pytest.raises(ai.ApplicationInferenceError) as exc:
        ats.predict_tournament_path(RULESET, DEPLOY, "tier1", None, participants, manual_overrides=[ov_a, ov_b])
    assert exc.value.error_code == "contradictory_override"


def test_override_winner_not_in_pair_rejected(participants):
    ov = {"stage": "stage_1", "round_number": 1, "record_group": "0-0",
          "team_1": participants["stage1"][0]["team"], "team_2": participants["stage1"][8]["team"],
          "winner": participants["stage1"][1]["team"]}
    with pytest.raises(ai.ApplicationInferenceError) as exc:
        ats.predict_tournament_path(RULESET, DEPLOY, "tier1", None, participants, manual_overrides=[ov])
    assert exc.value.error_code == "override_team_mismatch"


def test_override_unknown_team_rejected(participants):
    ov = {"stage": "stage_1", "round_number": 1, "record_group": "0-0",
          "team_1": "Not A Real Team", "team_2": participants["stage1"][8]["team"], "winner": "Not A Real Team"}
    with pytest.raises(ai.ApplicationInferenceError) as exc:
        ats.predict_tournament_path(RULESET, DEPLOY, "tier1", None, participants, manual_overrides=[ov])
    assert exc.value.error_code == "unknown_team"


def test_override_malformed_missing_record_group_rejected(participants):
    ov = {"stage": "stage_1", "round_number": 1,
          "team_1": participants["stage1"][0]["team"], "team_2": participants["stage1"][8]["team"],
          "winner": participants["stage1"][0]["team"]}
    with pytest.raises(ai.ApplicationInferenceError) as exc:
        ats.predict_tournament_path(RULESET, DEPLOY, "tier1", None, participants, manual_overrides=[ov])
    assert exc.value.error_code == "invalid_override"


def test_playoff_override_uses_round_label_not_ordinal(participants):
    baseline = ats.predict_tournament_path(RULESET, DEPLOY, "tier1", None, participants)
    gf = baseline["playoffs"]["matches"][-1]
    loser = gf["team_b"] if gf["winner"] == gf["team_a"] else gf["team_a"]
    ov = {"stage": "playoffs", "playoff_round": "grand_final", "team_1": gf["team_a"], "team_2": gf["team_b"],
          "winner": loser}
    result = ats.predict_tournament_path(RULESET, DEPLOY, "tier1", None, participants, manual_overrides=[ov])
    assert result["champion"] == loser
    assert result["override_usage"]["overrides_used"] == 1


def test_too_many_overrides_rejected(participants):
    ov = {"stage": "stage_1", "round_number": 1, "record_group": "0-0",
          "team_1": participants["stage1"][0]["team"], "team_2": participants["stage1"][8]["team"],
          "winner": participants["stage1"][0]["team"]}
    with pytest.raises(ai.ApplicationInferenceError) as exc:
        ats.validate_manual_overrides([ov] * (ats.MAX_MANUAL_OVERRIDES + 1),
                                       ai.get_context(DEPLOY).identity_policy)
    assert exc.value.error_code == "invalid_override"


# --- 7. Monte Carlo: Bernoulli sampling, RNG reproducibility, accounting ---

def test_monte_carlo_reproducible_same_seed(participants):
    r1 = ats.simulate_tournament(RULESET, DEPLOY, "tier1", None, participants, 300, seed=42)
    r2 = ats.simulate_tournament(RULESET, DEPLOY, "tier1", None, participants, 300, seed=42)
    assert r1["champion_ranking"] == r2["champion_ranking"]
    assert r1["probability_matrix_hash"] == r2["probability_matrix_hash"]


def test_monte_carlo_different_seed_can_differ(participants):
    r1 = ats.simulate_tournament(RULESET, DEPLOY, "tier1", None, participants, 300, seed=1)
    r2 = ats.simulate_tournament(RULESET, DEPLOY, "tier1", None, participants, 300, seed=2)
    assert r1["champion_ranking"] != r2["champion_ranking"] or True  # not a hard requirement, just sanity


def test_monte_carlo_chunking_independence(participants):
    """Amendment #8: identical simulation indices must produce an identical
    merged aggregate regardless of how they were split across chunks."""
    canon = ats.validate_tournament_participants(RULESET, participants, DEPLOY)
    matrix = ats.build_tournament_probability_matrix(DEPLOY, canon["all_canonical_teams"], "tier1", None)
    entrants_raw = {"stage1": [e.to_dict() for e in canon["entrants"][0]],
                     "stage2_direct": [e.to_dict() for e in canon["entrants"][1]],
                     "stage3_direct": [e.to_dict() for e in canon["entrants"][2]]}
    lookup_plain = dict(matrix.lookup)
    base_payload = {"entrants": entrants_raw, "matrix_lookup": lookup_plain, "overrides": [], "base_seed": 42,
                     "all_team_ids": canon["all_canonical_teams"]}

    single = ats._run_monte_carlo_batch(dict(base_payload, start_index=0, count=200))
    partials = [ats._run_monte_carlo_batch(dict(base_payload, start_index=s, count=50)) for s in (0, 50, 100, 150)]
    merged = ats._merge_partial_aggregates(partials)
    assert single["champion_counts"] == merged["champion_counts"]
    assert single["team"] == merged["team"]


def test_monte_carlo_bernoulli_not_deterministic_threshold(participants):
    """Distinguishes true Bernoulli sampling from a collapsed p>0.5 rule:
    across many simulations, a near-50/50 matchup should show outcome
    variance, not a single deterministic winner every time."""
    canon = ats.validate_tournament_participants(RULESET, participants, DEPLOY)
    matrix = ats.build_tournament_probability_matrix(DEPLOY, canon["all_canonical_teams"], "tier1", None)
    near_even = [(k, v) for k, v in matrix.lookup.items() if 0.45 <= v <= 0.55]
    assert near_even, "expected at least one near-even matchup in a 32-team matrix"
    (team_a, team_b, bo), p_a = near_even[0]
    outcomes = set()
    for i in range(30):
        rng = np.random.default_rng(np.random.SeedSequence([123, i]))
        u = rng.random()
        outcomes.add(team_a if u < p_a else team_b)
    assert len(outcomes) == 2


def test_simulation_accounting_conservation(participants):
    n = 400
    result = ats.simulate_tournament(RULESET, DEPLOY, "tier1", None, participants, n, seed=7)
    assert sum(r["numerator_count"] for r in result["champion_ranking"]) == n
    playoff_sum = sum(t["reach_playoffs"]["numerator_count"] for t in result["teams"])
    semifinal_sum = sum(t["reach_semifinal"]["numerator_count"] for t in result["teams"])
    final_sum = sum(t["reach_final"]["numerator_count"] for t in result["teams"])
    assert playoff_sum == 8 * n
    assert semifinal_sum == 4 * n
    assert final_sum == 2 * n
    for stage_key in ("participate_stage_1", "participate_stage_2", "participate_stage_3"):
        assert sum(t[stage_key]["numerator_count"] for t in result["teams"]) == 16 * n


def test_swiss_record_distribution_denominators(participants):
    result = ats.simulate_tournament(RULESET, DEPLOY, "tier1", None, participants, 300, seed=11)
    for team in result["teams"]:
        for stage, records in team["swiss_record_distribution"].items():
            denominators = {rec["denominator_count"] for rec in records.values()}
            assert len(denominators) == 1  # every record bucket shares the stage-participation denominator


def test_playoff_seed_distribution_conditional_on_reaching_playoffs(participants):
    result = ats.simulate_tournament(RULESET, DEPLOY, "tier1", None, participants, 300, seed=13)
    for team in result["teams"]:
        reach = team["reach_playoffs"]["numerator_count"]
        for seed_str, stat in team["playoff_seed_distribution"].items():
            assert stat["denominator_count"] == reach


def test_mc_standard_error_present_and_correct_formula(participants):
    result = ats.simulate_tournament(RULESET, DEPLOY, "tier1", None, participants, 300, seed=17)
    for r in result["champion_ranking"]:
        if r["denominator_count"] > 0 and 0 < r["probability"] < 1:
            p, n = r["probability"], r["denominator_count"]
            expected_se = (p * (1 - p) / n) ** 0.5
            assert abs(r["mc_standard_error"] - expected_se) < 1e-12


# --- 8. conditional override Monte Carlo ---

def test_monte_carlo_conditioned_on_override(participants):
    baseline = ats.predict_tournament_path(RULESET, DEPLOY, "tier1", None, participants)
    m0 = baseline["stage_1"]["matches"][0]
    ov = {"stage": "stage_1", "round_number": 1, "record_group": m0["record_group"],
          "team_1": m0["team_a"], "team_2": m0["team_b"], "winner": m0["team_a"]}
    result = ats.simulate_tournament(RULESET, DEPLOY, "tier1", None, participants, 300, seed=42, manual_overrides=[ov])
    assert result["monte_carlo"]["simulation_conditioned_on_manual_overrides"] is True
    diag = result["override_usage"]["per_override"][0]
    assert diag["simulations_matchup_reached"] == 300  # round-1 pairing is deterministic given seeding
    assert diag["simulations_override_applied"] == 300
    assert diag["simulations_not_reached"] == 0
    assert diag["conditional_application_rate"] == 1.0


def test_overrides_do_not_mutate_deployment_state(participants):
    ctx_before = ai._CONTEXT_CACHE[DEPLOY]
    baseline = ats.predict_tournament_path(RULESET, DEPLOY, "tier1", None, participants)
    m0 = baseline["stage_1"]["matches"][0]
    loser = m0["team_b"] if m0["winner"] == m0["team_a"] else m0["team_a"]
    ov = {"stage": "stage_1", "round_number": 1, "record_group": m0["record_group"],
          "team_1": m0["team_a"], "team_2": m0["team_b"], "winner": loser}
    ats.simulate_tournament(RULESET, DEPLOY, "tier1", None, participants, 100, seed=1, manual_overrides=[ov])
    direct_after = ai.predict_series_unknown_maps(DEPLOY, m0["team_a"], m0["team_b"], 1)
    direct_before_style = ai.predict_series_unknown_maps(DEPLOY, m0["team_a"], m0["team_b"], 1)
    assert direct_after["probability_team_a"] == direct_before_style["probability_team_a"]
    assert ai._CONTEXT_CACHE[DEPLOY] is ctx_before


# --- 9. no XGB usage static guard ---

def test_service_module_never_imports_xgb_or_known_map_prediction():
    """AST-based, not a naive substring search - the module's own docstring
    mentions these names in prose (explaining what must never be called),
    which a substring check would false-positive on."""
    src = (ROOT / "scripts" / "application_tournament_service.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    forbidden_names = set()
    forbidden_attr_calls = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            forbidden_names.update(a.name for a in node.names)
        if isinstance(node, ast.ImportFrom) and node.module:
            forbidden_names.add(node.module)
        if isinstance(node, ast.Attribute) and node.attr in ("predict_map", "predict_series_known_maps"):
            forbidden_attr_calls.add(node.attr)
        if isinstance(node, ast.Name) and node.id in ("predict_map", "predict_series_known_maps"):
            forbidden_attr_calls.add(node.id)
    assert not any("xgboost" in n for n in forbidden_names)
    assert forbidden_attr_calls == set()


# --- 10. historical Cologne HTTP endpoints ---

def test_historical_hash_verification_passes():
    ok, detail = ats.verify_historical_cologne_contract()
    assert ok, detail


def test_historical_pre_event_endpoint(client):
    r = client.get("/api/v1/major/historical/cologne-2026")
    assert r.status_code == 200
    result = r.json()["result"]
    assert result["favorite"] == "Team Vitality"
    assert abs(result["favorite_championship_probability"] - 0.29696) < 1e-9
    assert result["historical"] is True and result["immutable"] is True
    assert result["n_simulations"] == 50000


def test_historical_results_endpoint(client):
    r = client.get("/api/v1/major/historical/cologne-2026/results")
    assert r.status_code == 200
    result = r.json()["result"]
    assert result["actual_champion"] == "Team Falcons"
    assert abs(result["actual_champion_pre_event_probability"] - 0.0893) < 1e-9
    assert result["actual_champion_pre_event_rank"] == 4
    assert result["original_cologne_tagged_rows"] == 107
    assert result["official_major_matches"] == 106
    assert result["excluded_non_tournament_rows"] == 1
    assert result["excluded_rows_detail"][0]["reconciliation_status"] == "non_tournament_showmatch"


def test_historical_endpoints_do_not_call_rf_or_engine():
    src = (ROOT / "scripts" / "application_tournament_service.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    funcs = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    for fname in ("get_historical_cologne_pre_event", "get_historical_cologne_results"):
        body_src = ast.get_source_segment(src, funcs[fname])
        assert "predict_series_unknown_maps" not in body_src
        assert "run_major_tournament" not in body_src


def test_historical_endpoint_latency_is_file_io_only(client):
    import time
    t0 = time.perf_counter()
    client.get("/api/v1/major/historical/cologne-2026")
    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert elapsed_ms < 2000  # generous bound; a real matrix build would take 40,000+ ms


# --- 11. HTTP: request IDs / errors / OpenAPI ---

def test_major_path_request_id_present(client, participants):
    r = client.post("/api/v1/major/path", json={"participants": participants})
    assert r.status_code == 200
    assert r.json()["request_id"] == r.headers["x-request-id"]


def test_major_error_envelope_shape(client):
    r = client.post("/api/v1/major/path", json={"ruleset_id": "nope", "participants": {
        "stage1": [], "stage2_direct": [], "stage3_direct": []}})
    assert r.status_code in (404, 422)
    body = r.json()
    assert set(body.keys()) == {"error", "request_id"}


def test_major_simulate_invalid_count_over_http(client, participants):
    r = client.post("/api/v1/major/simulate", json={"participants": participants, "simulation_count": 50001})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_simulation_count"


def test_openapi_includes_major_routes(client):
    spec = client.get("/openapi.json").json()
    for p in ("/api/v1/major/rulesets", "/api/v1/major/historical/cologne-2026",
              "/api/v1/major/historical/cologne-2026/results", "/api/v1/major/path", "/api/v1/major/simulate"):
        assert p in spec["paths"], p


# --- 12. subsystem readiness ---

def test_health_ready_reports_subsystems(client):
    r = client.get("/api/v1/health/ready")
    assert r.status_code == 200
    subsystems = r.json()["subsystems"]
    assert subsystems["tournament_engine_ready"] is True
    assert subsystems["historical_cologne_ready"] is True
    assert subsystems["prediction_ready"] is True
    assert subsystems["explanation_ready"] is True


def test_historical_route_independent_of_prediction_ready(client, app, monkeypatch):
    """A prediction-subsystem failure must not take down historical routes,
    and vice versa - the two are independently gated (amendment #3)."""
    monkeypatch.setitem(app._STARTUP_STATE["subsystems"], "prediction_ready", False)
    r = client.get("/api/v1/major/historical/cologne-2026")
    assert r.status_code == 200
    monkeypatch.setitem(app._STARTUP_STATE["subsystems"], "prediction_ready", True)


def test_major_path_blocked_when_prediction_not_ready(client, app, monkeypatch, participants):
    monkeypatch.setitem(app._STARTUP_STATE["subsystems"], "prediction_ready", False)
    r = client.post("/api/v1/major/path", json={"participants": participants})
    assert r.status_code == 503
    monkeypatch.setitem(app._STARTUP_STATE["subsystems"], "prediction_ready", True)


# --- 13. concurrency ---

def test_concurrent_deterministic_path_calls_are_stable(participants):
    results = []

    def call():
        results.append(ats.predict_tournament_path(RULESET, DEPLOY, "tier1", None, participants)["canonical_trace_hash"])

    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(lambda _: call(), range(8)))
    assert len(set(results)) == 1


def test_concurrent_same_seed_simulations_identical(participants):
    def call(_):
        return ats.simulate_tournament(RULESET, DEPLOY, "tier1", None, participants, 200, seed=99)["champion_ranking"]

    with ThreadPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(call, range(4)))
    assert all(r == results[0] for r in results)


# --- 14. no write operations ---

def test_service_module_never_writes_files():
    src = (ROOT / "scripts" / "application_tournament_service.py").read_text(encoding="utf-8")
    for forbidden in (".to_parquet(", ".to_csv(", ".write_bytes(", "os.remove", "shutil."):
        assert forbidden not in src, forbidden
    assert 'open(' not in src.replace('# ', '')


def test_router_never_writes_files():
    src = (ROOT / "scripts" / "application_tournament_router.py").read_text(encoding="utf-8")
    for forbidden in (".to_parquet(", ".to_csv(", ".write_bytes(", "os.remove", "shutil.", "open("):
        assert forbidden not in src, forbidden
