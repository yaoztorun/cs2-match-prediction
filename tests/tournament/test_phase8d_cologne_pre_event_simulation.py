"""
Phase 8D tests. Correctness tests use small subsets / direct function calls
so the suite stays fast - the expensive full 2,976-row matrix and 50,000-run
simulation are generated once by tournament/cologne_2026/run_phase8d_pipeline.py, and a
separate small set of "artifact" tests here check the canonical outputs IF
they exist (skipped otherwise, matching the Phase 8C precedent), so this
file works whether or not the real historical run has completed yet.
"""

import ast
import json

import numpy as np
import pandas as pd
import pytest

import tournament.cologne_2026.build_phase8d_protocol as protocol_mod
import tournament.simulation.cologne_pre_event_simulation as sim
import feature_engineering.series.feature_engine as fe
import tournament.simulation.generate_cologne_pre_event_probability_matrix as gen_matrix
import tournament.cologne_2026.phase8d_common as pc
import tournament.simulation.pre_veto_series_predictor as pvp
import tournament.cologne_2026.run_phase8d_pipeline as pipe
import tournament.engine.tournament_engine as te

FORBIDDEN_RESULT_PATHS = [
    "data/interim/series_base.parquet", "data/interim/map_base.parquet",
    "data/evaluation/map_test_predictions_v1.parquet", "series_base.parquet", "map_base.parquet",
]
FORBIDDEN_NETWORK_MODULES = {"requests", "urllib", "http", "httpx", "socket", "aiohttp"}

PHASE8D_MODULE_FILES = [
    "tournament/simulation/pre_veto_series_predictor.py", "tournament/cologne_2026/build_phase8d_protocol.py",
    "tournament/simulation/generate_cologne_pre_event_probability_matrix.py", "tournament/simulation/cologne_pre_event_simulation.py",
    "tournament/cologne_2026/run_phase8d_pipeline.py", "tournament/cologne_2026/phase8d_figures.py", "tournament/cologne_2026/phase8d_common.py",
]


@pytest.fixture(scope="module")
def context():
    return pvp.load_predictor_context()


@pytest.fixture(scope="module")
def rules():
    return te.load_frozen_rules()


@pytest.fixture(scope="module")
def teams():
    return pc.load_cologne_teams()


@pytest.fixture(scope="module")
def entrants():
    return pc.build_cologne_entrants()


CUTOFF = "2026-06-02T13:30:00"


# ---------------------------------------------------------------------------
# 1. Frozen artifact loading / model contract
# ---------------------------------------------------------------------------

def test_model_contract_verification_passes(context):
    assert context.model_contract["all_checks_pass"] is True
    assert context.model_contract["classes_are_0_1_with_class1_meaning_team_a_wins"] is True
    assert context.model_contract["raw_feature_count"] == 17
    assert context.model_contract["transformed_feature_count"] == 19
    assert context.model_contract["n_features_in_matches_transformed_count"] is True


def test_verify_model_contract_raises_on_bad_classes():
    class FakeModel:
        classes_ = [0.0, 2.0]
        n_features_in_ = 19
    preprocessing = {"original_model_feature_names": list(range(17)),
                      "transformed_feature_names": pvp.EXPECTED_TRANSFORMED_FEATURE_NAMES}
    with pytest.raises(ValueError):
        pvp.verify_model_contract(FakeModel(), preprocessing)


def test_pipeline_hashes_cover_required_inputs(context):
    hashes = pvp.hash_pipeline_inputs()
    required = {"model", "preprocessing", "selected_config", "series_features_v1_yaml", "feature_engine",
                "preprocessing_random_forest_v1", "strict_pre_cologne_state", "predictor_adapter",
                "tournament_yaml", "tournament_engine"}
    assert required <= set(hashes.keys())
    assert all(len(h) == 64 for h in hashes.values())


def test_environment_versions_recorded():
    versions = pvp.environment_versions()
    assert {"python", "numpy", "pandas", "sklearn", "joblib"} <= set(versions.keys())


# ---------------------------------------------------------------------------
# 2. Adapter determinism / no state mutation
# ---------------------------------------------------------------------------

def test_predict_series_unknown_maps_deterministic(context):
    r1 = pvp.predict_series_unknown_maps(context, "Team Vitality", "FURIA Esports", 3, CUTOFF)
    r2 = pvp.predict_series_unknown_maps(context, "Team Vitality", "FURIA Esports", 3, CUTOFF)
    assert r1["probability_team_a"] == r2["probability_team_a"]


def test_predict_series_unknown_maps_does_not_mutate_state_store(context):
    n_teams_before = len(context.store.teams)
    history_len_before = len(context.store.get("Team Vitality").history)
    pvp.predict_series_unknown_maps(context, "Team Vitality", "FURIA Esports", 3, CUTOFF)
    assert len(context.store.teams) == n_teams_before
    assert len(context.store.get("Team Vitality").history) == history_len_before


def test_tier_override_mismatch_raises(context):
    with pytest.raises(ValueError):
        pvp.predict_series_unknown_maps(context, "Team Vitality", "FURIA Esports", 3, CUTOFF, tier="tier2")


def test_probability_a_plus_b_equals_one(context):
    r = pvp.predict_series_unknown_maps(context, "Team Vitality", "FURIA Esports", 3, CUTOFF)
    assert abs((r["probability_team_a"] + r["probability_team_b"]) - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# 3. BO1/BO3/BO5 + tier1 inference-contract validation
# ---------------------------------------------------------------------------

def test_bo_support_all_three_formats(context):
    results = pvp.validate_bo_support(context, "Team Vitality", "FURIA Esports", CUTOFF)
    assert set(results.keys()) == {1, 3, 5}
    assert results[1]["transformed_one_hot_pattern"] == {"bestOf_BO3": 0.0, "bestOf_BO5": 0.0}
    assert results[3]["transformed_one_hot_pattern"] == {"bestOf_BO3": 1.0, "bestOf_BO5": 0.0}
    assert results[5]["transformed_one_hot_pattern"] == {"bestOf_BO3": 0.0, "bestOf_BO5": 1.0}
    for r in results.values():
        assert r["finite"] and r["in_unit_interval"]


def test_tier1_produces_reference_category(context):
    r = pvp.verify_tier_representation(context, "Team Vitality", "FURIA Esports", CUTOFF)
    assert r["is_reference_category"] is True
    assert r["tier_tier2"] == 0.0 and r["tier_tier3"] == 0.0


# ---------------------------------------------------------------------------
# 4. 32-team coverage, incl. THUNDERdOWNUNDER cold start
# ---------------------------------------------------------------------------

def test_team_coverage_32_rows(context, teams):
    df = pvp.audit_team_coverage(context, teams)
    assert len(df) == 32


def test_thunderdownunder_is_cold_start_and_scores_without_error(context, teams):
    df = pvp.audit_team_coverage(context, teams)
    row = df[df["canonical_model_name"] == "THUNDERdOWNUNDER"].iloc[0]
    assert row["present_in_pre_cologne_state"] == False
    assert row["cold_start"] == True
    assert row["total_matches_before"] == 0
    r = pvp.predict_series_unknown_maps(context, "THUNDERdOWNUNDER", "Team Vitality", 3, CUTOFF)
    assert np.isfinite(r["probability_team_a"]) and 0.0 <= r["probability_team_a"] <= 1.0


def test_coverage_31_of_32_present(context, teams):
    df = pvp.audit_team_coverage(context, teams)
    assert df["present_in_pre_cologne_state"].sum() == 31


# ---------------------------------------------------------------------------
# 5. Probability matrix (small subset - full 2,976-row matrix is generated
# once by the real pipeline, checked separately in the artifact section)
# ---------------------------------------------------------------------------

def test_matrix_small_subset_shape_and_validation(context, teams):
    subset = teams[:5]
    df = gen_matrix.build_probability_matrix(context, teams=subset, prediction_datetime=CUTOFF)
    assert len(df) == 5 * 4 * 3
    assert len(df[["team_a", "team_b", "best_of"]].drop_duplicates()) == 5 * 4 * 3


def test_matrix_validation_does_not_clip_or_alter_probabilities(context, teams, monkeypatch):
    subset = teams[:5]
    df = gen_matrix.build_probability_matrix(context, teams=subset, prediction_datetime=CUTOFF)
    monkeypatch.setattr(gen_matrix, "EXPECTED_ROW_COUNT", len(df))
    original = df["probability_team_a"].copy()
    gen_matrix.validate_probability_matrix(df)
    pd.testing.assert_series_equal(df["probability_team_a"], original)


def test_matrix_validation_rejects_wrong_row_count():
    df = pd.DataFrame([{"team_a": "A", "team_b": "B", "best_of": 1,
                         "probability_team_a": 0.5, "probability_team_b": 0.5}])
    with pytest.raises(ValueError):
        gen_matrix.validate_probability_matrix(df)


def test_matrix_validation_rejects_forbidden_result_column():
    rows = []
    for i in range(32 * 31 * 3):
        rows.append({"team_a": f"T{i}", "team_b": f"U{i}", "best_of": 1,
                      "probability_team_a": 0.5, "probability_team_b": 0.5, "winner": "T"})
    df = pd.DataFrame(rows)
    with pytest.raises(ValueError):
        gen_matrix.validate_probability_matrix(df)


def test_orientation_diagnostic_finite(context, teams):
    subset = teams[:6]
    df = gen_matrix.build_probability_matrix(context, teams=subset, prediction_datetime=CUTOFF)
    diag = gen_matrix.compute_orientation_diagnostic(df)
    for k in ("mean", "median", "p95", "max"):
        assert np.isfinite(diag[k])


# ---------------------------------------------------------------------------
# 6. Cached provider / favorite-wins provider
# ---------------------------------------------------------------------------

def test_cached_provider_deterministic_given_same_rng_state():
    matrix = {("A", "B", 3): 0.7}
    match = te.MatchSpec("m1", "stage_1", 1, "0-0", "A", "B", 1, 2, best_of=3)
    rng1 = np.random.default_rng(np.random.SeedSequence([42, 0]))
    rng2 = np.random.default_rng(np.random.SeedSequence([42, 0]))
    p1 = sim.CachedProbabilityOutcomeProvider(matrix, rng1)
    p2 = sim.CachedProbabilityOutcomeProvider(matrix, rng2)
    r1, r2 = p1.resolve_match(match), p2.resolve_match(match)
    assert r1.winner == r2.winner
    assert r1.probability_team_a == 0.7


def test_cached_provider_missing_key_raises():
    provider = sim.CachedProbabilityOutcomeProvider({}, np.random.default_rng(0))
    match = te.MatchSpec("m1", "stage_1", 1, "0-0", "A", "B", 1, 2, best_of=3)
    with pytest.raises(ValueError):
        provider.resolve_match(match)


def test_favorite_wins_provider_tie_policy_p_equals_half_favors_team_a():
    matrix = {("A", "B", 3): 0.5}
    match = te.MatchSpec("m1", "stage_1", 1, "0-0", "A", "B", 1, 2, best_of=3)
    provider = sim.FavoriteWinsProvider(matrix)
    resolution = provider.resolve_match(match)
    assert resolution.winner == "A"


def test_favorite_wins_provider_below_half_favors_team_b():
    matrix = {("A", "B", 3): 0.49}
    match = te.MatchSpec("m1", "stage_1", 1, "0-0", "A", "B", 1, 2, best_of=3)
    provider = sim.FavoriteWinsProvider(matrix)
    assert provider.resolve_match(match).winner == "B"


# ---------------------------------------------------------------------------
# 7. Per-index RNG reproducibility
# ---------------------------------------------------------------------------

def test_simulation_rng_reproducible_by_index():
    rng_a, seq_a = sim.simulation_rng(42, 17)
    rng_b, seq_b = sim.simulation_rng(42, 17)
    draws_a = [rng_a.random() for _ in range(5)]
    draws_b = [rng_b.random() for _ in range(5)]
    assert draws_a == draws_b
    assert list(seq_a.entropy) == list(seq_b.entropy) == [42, 17]


def test_simulation_rng_differs_by_index():
    rng_a, _ = sim.simulation_rng(42, 0)
    rng_b, _ = sim.simulation_rng(42, 1)
    assert rng_a.random() != rng_b.random()


def test_single_simulation_index_reproducible(rules, entrants, teams):
    flat_matrix = {(a["canonical_model_name"], b["canonical_model_name"], bo): 0.5 + 0.001 * i
                   for i, (a, b) in enumerate((x, y) for x in teams for y in teams if x != y)
                   for bo in (1, 3, 5)}
    r1 = sim.run_one_simulation(7, 42, entrants, rules, flat_matrix)
    r2 = sim.run_one_simulation(7, 42, entrants, rules, flat_matrix)
    assert te.trace_hash(r1.to_dict()) == te.trace_hash(r2.to_dict())
    assert r1.champion == r2.champion


# ---------------------------------------------------------------------------
# 8. Aggregate accounting (small run)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_flat_matrix(teams):
    rows = []
    for a in teams:
        for b in teams:
            if a["canonical_model_name"] == b["canonical_model_name"]:
                continue
            for bo in (1, 3, 5):
                rows.append({"team_a": a["canonical_model_name"], "team_b": b["canonical_model_name"],
                             "best_of": bo, "probability_team_a": 0.55})
    return sim.build_matrix_lookup(pd.DataFrame(rows))


@pytest.fixture(scope="module")
def small_run(rules, entrants, teams, small_flat_matrix):
    all_ids = [t["canonical_model_name"] for t in teams]
    agg, samples = sim.run_monte_carlo(120, 42, entrants, rules, small_flat_matrix, all_ids,
                                        sample_indices=[0, 1])
    return agg, samples


def test_accounting_identities_small_run(small_run):
    agg, _ = small_run
    accounting = pipe.validate_accounting(agg, agg.n_simulations)
    assert accounting["all_pass"] is True
    assert accounting["champion_sum"] == agg.n_simulations
    assert accounting["playoff_sum"] == 8 * agg.n_simulations
    assert accounting["semifinal_sum"] == 4 * agg.n_simulations
    assert accounting["final_sum"] == 2 * agg.n_simulations


def test_stage1_entrants_participate_stage1_always(small_run, entrants):
    agg, _ = small_run
    stage1_ids = {e.team_id for e in entrants[0]}
    for tid in stage1_ids:
        assert agg.team[tid]["participate_stage_1"] == agg.n_simulations


def test_stage3_direct_entrants_never_participate_stage1(small_run, entrants):
    agg, _ = small_run
    stage3_direct_ids = {e.team_id for e in entrants[2]}
    for tid in stage3_direct_ids:
        assert agg.team[tid]["participate_stage_1"] == 0
        assert agg.team[tid]["participate_stage_3"] == agg.n_simulations


def test_sample_traces_captured(small_run):
    _, samples = small_run
    assert len(samples) == 2
    for s in samples:
        assert "canonical_trace_hash" in s and "trace" in s and "seed_sequence_entropy" in s


# ---------------------------------------------------------------------------
# 9. MC standard error
# ---------------------------------------------------------------------------

def test_mc_stats_formula():
    stats = sim.mc_stats(25000, 50000)
    assert stats["probability"] == 0.5
    expected_se = (0.5 * 0.5 / 50000) ** 0.5
    assert abs(stats["mc_standard_error"] - expected_se) < 1e-12
    assert abs(expected_se - 0.00223606797749979) < 1e-9


def test_mc_stats_null_when_denominator_zero():
    stats = sim.mc_stats(0, 0)
    assert stats["probability"] is None
    assert stats["mc_standard_error"] is None


def test_mc_stats_ci_bounded_to_unit_interval():
    stats = sim.mc_stats(49999, 50000)
    assert 0.0 <= stats["mc_ci95_low"] <= stats["mc_ci95_high"] <= 1.0


# ---------------------------------------------------------------------------
# 10. Output-table builders use counts as source of truth (amendment #5)
# ---------------------------------------------------------------------------

def test_team_probabilities_recomputable_from_counts(small_run, teams):
    agg, _ = small_run
    df = sim.build_team_probabilities_df(agg, teams)
    for _, row in df.iterrows():
        if row["denominator_count"] == 0:
            assert pd.isna(row["probability"])
        else:
            assert abs(row["probability"] - row["numerator_count"] / row["denominator_count"]) < 1e-12


def test_swiss_record_distributions_sum_to_participation(small_run, teams):
    agg, _ = small_run
    df = sim.build_swiss_record_distributions_df(agg, teams)
    for (tid, stage), sub in df.groupby(["canonical_model_name", "stage"]):
        expected = agg.team[tid][f"{stage}_participations"]
        assert sub["count"].sum() == expected


def test_favorite_path_uses_named_convention(rules, entrants, small_flat_matrix):
    fav = sim.build_favorite_path_trace(entrants, rules, small_flat_matrix)
    assert fav["path_name"] == "deterministic favorite-wins path"
    assert len(fav["matches"]) == 106


def test_rematch_fallback_report_structure(small_run):
    agg, _ = small_run
    report = sim.build_rematch_fallback_report(agg)
    assert report["total_swiss_pairing_events"] == agg.n_simulations * 99
    assert set(report["breakdown_by_stage"].keys()) == set(sim.SWISS_STAGES)


# ---------------------------------------------------------------------------
# 11. Reproducibility subset (amendment #29 / #11): rerun N sims twice
# ---------------------------------------------------------------------------

def test_reproducibility_subset_identical_across_two_runs(rules, entrants, teams, small_flat_matrix):
    all_ids = [t["canonical_model_name"] for t in teams]
    agg1, _ = sim.run_monte_carlo(50, 42, entrants, rules, small_flat_matrix, all_ids)
    agg2, _ = sim.run_monte_carlo(50, 42, entrants, rules, small_flat_matrix, all_ids)
    assert dict(agg1.champion_counts) == dict(agg2.champion_counts)
    for tid in all_ids:
        assert agg1.team[tid]["win_tournament"] == agg2.team[tid]["win_tournament"]
        assert agg1.team[tid]["reach_playoffs"] == agg2.team[tid]["reach_playoffs"]


# ---------------------------------------------------------------------------
# 12. Result-access / network-access static guards
# ---------------------------------------------------------------------------

def _reads_forbidden_path(source, forbidden_tokens):
    """AST-based (not blunt substring): only flags forbidden paths that
    appear as an argument to an actual read call (open/read_csv/read_parquet/
    Path(...).read_*), not paths that appear as policy-declaration string
    literals (e.g. build_phase8d_protocol.py's own forbidden_paths list)."""
    tree = ast.parse(source)
    read_call_names = {"read_csv", "read_parquet", "read_json", "open", "read_text", "read_bytes"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            fname = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else "")
            if fname in read_call_names:
                for arg in list(node.args) + [kw.value for kw in node.keywords]:
                    try:
                        arg_src = ast.unparse(arg)
                    except Exception:
                        continue
                    if any(tok in arg_src for tok in forbidden_tokens):
                        return True
    return False


@pytest.mark.parametrize("relpath", PHASE8D_MODULE_FILES)
def test_no_forbidden_result_paths_in_phase8d_source(relpath):
    source = (te.ROOT / relpath).read_text(encoding="utf-8")
    assert not _reads_forbidden_path(source, FORBIDDEN_RESULT_PATHS), \
        f"{relpath} actually READS a forbidden result path"


@pytest.mark.parametrize("relpath", PHASE8D_MODULE_FILES)
def test_no_network_imports_in_phase8d_source(relpath):
    source = (te.ROOT / relpath).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not (imported & FORBIDDEN_NETWORK_MODULES), \
        f"{relpath} imports forbidden network module(s): {imported & FORBIDDEN_NETWORK_MODULES}"


def test_tournament_engine_still_never_reads_participants():
    source = (te.ROOT / "tournament" / "engine" / "tournament_engine.py").read_text(encoding="utf-8")
    assert '"participants"' not in source and "['participants']" not in source


# ---------------------------------------------------------------------------
# 13. Transactional pipeline: preflight abort behavior
# ---------------------------------------------------------------------------

def test_preflight_passes_on_clean_slate(tmp_path, monkeypatch):
    eval_dir = tmp_path / "evaluation"
    eval_dir.mkdir()
    monkeypatch.setattr(pipe, "EVAL_DIR", eval_dir)
    monkeypatch.setattr(pipe, "FIGURES_CANONICAL_DIR", tmp_path / "figures")
    monkeypatch.setattr(pipe, "RECEIPT_CANONICAL_PATH", eval_dir / "receipt.json")
    monkeypatch.setattr(pipe, "PROTOCOL_CANONICAL_PATH", tmp_path / "config" / "protocol.yaml")
    monkeypatch.setattr(pipe, "CANONICAL_FILES", [
        ("protocol.yaml", pipe.PROTOCOL_CANONICAL_PATH), ("receipt.json", pipe.RECEIPT_CANONICAL_PATH)])
    pipe.preflight_check()  # must not raise


def test_preflight_aborts_on_completed_receipt(tmp_path, monkeypatch):
    eval_dir = tmp_path / "evaluation"
    eval_dir.mkdir()
    receipt_path = eval_dir / "receipt.json"
    receipt_path.write_text(json.dumps({"created_before_results_opened": True, "hashes": {"x": "y"}}))
    monkeypatch.setattr(pipe, "EVAL_DIR", eval_dir)
    monkeypatch.setattr(pipe, "FIGURES_CANONICAL_DIR", tmp_path / "figures")
    monkeypatch.setattr(pipe, "RECEIPT_CANONICAL_PATH", receipt_path)
    monkeypatch.setattr(pipe, "PROTOCOL_CANONICAL_PATH", tmp_path / "config" / "protocol.yaml")
    monkeypatch.setattr(pipe, "CANONICAL_FILES", [
        ("protocol.yaml", pipe.PROTOCOL_CANONICAL_PATH), ("receipt.json", receipt_path)])
    with pytest.raises(pipe.Phase8DPipelineError, match="already exists"):
        pipe.preflight_check()


def test_preflight_aborts_on_partial_state_without_valid_receipt(tmp_path, monkeypatch):
    eval_dir = tmp_path / "evaluation"
    eval_dir.mkdir()
    (eval_dir / "team_probabilities.csv").write_text("a,b\n1,2\n")
    monkeypatch.setattr(pipe, "EVAL_DIR", eval_dir)
    monkeypatch.setattr(pipe, "FIGURES_CANONICAL_DIR", tmp_path / "figures")
    monkeypatch.setattr(pipe, "RECEIPT_CANONICAL_PATH", eval_dir / "receipt.json")
    monkeypatch.setattr(pipe, "PROTOCOL_CANONICAL_PATH", tmp_path / "config" / "protocol.yaml")
    monkeypatch.setattr(pipe, "CANONICAL_FILES", [
        ("protocol.yaml", pipe.PROTOCOL_CANONICAL_PATH),
        ("team_probabilities.csv", eval_dir / "team_probabilities.csv"),
        ("receipt.json", pipe.RECEIPT_CANONICAL_PATH)])
    with pytest.raises(pipe.Phase8DPipelineError, match="Partial Phase 8D"):
        pipe.preflight_check()


# ---------------------------------------------------------------------------
# 14. Real canonical artifacts (skipped until the real pipeline has run)
# ---------------------------------------------------------------------------

MATRIX_PATH = te.ROOT / "data" / "evaluation" / "cologne_2026_pre_event_matchup_probabilities_v1.parquet"
RECEIPT_PATH = te.ROOT / "data" / "evaluation" / "cologne_2026_pre_event_simulation_receipt_v1.json"


@pytest.mark.skipif(not MATRIX_PATH.exists(), reason="real probability matrix not generated yet")
def test_real_probability_matrix_has_2976_rows():
    df = pd.read_parquet(MATRIX_PATH, engine="fastparquet")
    assert len(df) == 2976
    assert len(df[["team_a", "team_b", "best_of"]].drop_duplicates()) == 2976


@pytest.mark.skipif(not RECEIPT_PATH.exists(), reason="real simulation receipt not generated yet")
def test_real_simulation_receipt_complete():
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    assert receipt["created_before_results_opened"] is True
    assert receipt["n_simulations"] == 50000
    assert receipt["accounting"]["all_pass"] is True
    assert receipt["cologne_results_status"] == "UNOPENED"


@pytest.mark.skipif(not RECEIPT_PATH.exists(), reason="real simulation receipt not generated yet")
def test_real_accounting_totals():
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    acc = receipt["accounting"]
    n = receipt["n_simulations"]
    assert acc["champion_sum"] == n
    assert acc["playoff_sum"] == 8 * n
    assert acc["semifinal_sum"] == 4 * n
    assert acc["final_sum"] == 2 * n
    assert acc["total_matches"] == n * 106
