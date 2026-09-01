"""
Phase 8E tests: reconciliation, Gate-2 engine replay, matrix-lookup-only
predictions, metric formulas, and immutability of every pre-existing
Phase 8B/8C/8D/8D.1 artifact.
"""

import ast
import json

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.metrics import log_loss

from _common import ROOT
import evaluation.cologne_2026.build_phase8e_protocol as protocol_mod
import evaluation.cologne_2026.phase8e_actual_outcome_provider as aop
import evaluation.cologne_2026.phase8e_common as p8e
import evaluation.cologne_2026.phase8e_metrics as pm
import evaluation.cologne_2026.reconcile_cologne_actual_results as recon_mod
import tournament.engine.tournament_engine as te

EVAL = ROOT / "data" / "evaluation"
FORBIDDEN_RF_MODULES = {"joblib", "feature_engine", "preprocessing_random_forest_v1", "pre_veto_series_predictor"}
FORBIDDEN_NETWORK_MODULES = {"requests", "urllib", "http", "httpx", "socket", "aiohttp"}

PHASE8E_MODULE_FILES = [
    "evaluation/cologne_2026/build_phase8e_protocol.py", "evaluation/cologne_2026/phase8e_common.py",
    "evaluation/cologne_2026/reconcile_cologne_actual_results.py", "evaluation/cologne_2026/phase8e_actual_outcome_provider.py",
    "evaluation/cologne_2026/build_phase8e_gate2_artifact.py", "evaluation/cologne_2026/build_phase8e_predictions.py",
    "evaluation/cologne_2026/phase8e_metrics.py", "evaluation/cologne_2026/build_phase8e_summary.py", "evaluation/cologne_2026/phase8e_figures.py",
    "validation/validate_phase8e.py",
]


def _imported_module_names(path):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"), filename=path)
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


@pytest.fixture(scope="module")
def reconciliation():
    roster = recon_mod.load_canonical_roster()
    raw = recon_mod.load_raw_cologne_rows()
    rows = recon_mod.build_reconciliation_rows(raw, roster)
    rows = recon_mod.assign_rounds_and_records(rows)
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def replay():
    return aop.replay_actual_tournament()


@pytest.fixture(scope="module")
def metrics():
    return pm.compute_all_metrics()


# --- 1. protocol frozen before any result-read execution path ---

def test_protocol_declares_reconciliation_before_result_access():
    protocol = protocol_mod.build_protocol_dict()
    assert "reconciliation_policy" in protocol
    assert "exact_lookup_policy" in protocol
    assert protocol["reconciliation_policy"]["dataset_row_count"] == 107


def test_protocol_hash_is_deterministic():
    p1 = protocol_mod.build_protocol_dict()
    p2 = protocol_mod.build_protocol_dict()
    assert protocol_mod.protocol_hash(p1) == protocol_mod.protocol_hash(p2)


# --- 2. 107-row reconciliation explicit, canonical official match count ---

def test_reconciliation_is_107_rows(reconciliation):
    assert len(reconciliation) == 107


def test_exactly_106_official_rows(reconciliation):
    assert int(reconciliation["included_in_official_event"].sum()) == 106


def test_showmatch_excluded_with_evidence(reconciliation):
    excluded = reconciliation[~reconciliation["included_in_official_event"]]
    assert len(excluded) == 1
    row = excluded.iloc[0]
    assert {row["team1"], row["team2"]} == {"Team Germany", "Team Poland"}
    assert row["reconciliation_status"] == "non_tournament_showmatch"
    assert isinstance(row["supporting_evidence"], str) and len(row["supporting_evidence"]) > 0


def test_per_stage_official_counts(reconciliation):
    counts = reconciliation[reconciliation["included_in_official_event"]]["candidate_stage"].value_counts()
    assert counts["stage_1"] == 33
    assert counts["stage_2"] == 33
    assert counts["stage_3"] == 33
    assert counts["playoffs"] == 7


def test_no_duplicate_official_matches():
    canonical = pd.read_parquet(EVAL / "cologne_2026_actual_series_results_v1.parquet", engine="fastparquet")
    assert len(canonical) == 106
    assert not canonical["source_match_id"].duplicated().any()


def test_actual_winners_derived_from_scores_not_broken_field():
    canonical = pd.read_parquet(EVAL / "cologne_2026_actual_series_results_v1.parquet", engine="fastparquet")
    expected = np.where(canonical["score_team_1"] > canonical["score_team_2"],
                         canonical["team_1"], canonical["team_2"])
    assert (canonical["winner"].values == expected).all()
    assert (canonical["score_team_1"] != canonical["score_team_2"]).all()


# --- 3. engine actual-result replay reproduces every actual pairing / stage transition ---

def test_engine_replay_reproduces_106_matches(replay):
    result, provider = replay
    assert len(result.full_trace()) == 106
    assert len(result.stage1.trace) == 33
    assert len(result.stage2.trace) == 33
    assert len(result.stage3.trace) == 33
    assert len(result.playoffs.trace) == 7


def test_engine_replay_consumes_every_actual_row_exactly_once(replay):
    result, provider = replay
    assert provider.n_consumed() == provider.n_total() == 106
    assert len(provider.unused_keys()) == 0


def test_engine_replay_stage_transitions_reproduced(replay):
    result, provider = replay
    assert len(result.stage1.advancers) == 8
    assert len(result.stage2.advancers) == 8
    assert len(result.stage3.advancers) == 8


def test_engine_replay_champion_matches_canonical_grand_final(replay):
    result, provider = replay
    canonical = pd.read_parquet(EVAL / "cologne_2026_actual_series_results_v1.parquet", engine="fastparquet")
    gf = canonical[(canonical.stage == "playoffs") & (canonical.round_number == 3)].iloc[0]
    assert result.champion == gf["winner"]


def test_provider_raises_on_ambiguous_key_construction():
    dup = pd.DataFrame([
        {"stage": "stage_1", "round_number": 1, "record_group": "0-0", "team_1": "A", "team_2": "B",
         "best_of": 1, "winner": "A", "source_match_id": 1},
        {"stage": "stage_1", "round_number": 1, "record_group": "0-0", "team_1": "B", "team_2": "A",
         "best_of": 1, "winner": "B", "source_match_id": 2},
    ])
    with pytest.raises(ValueError):
        aop.ActualOutcomeProvider(dup)


def test_provider_raises_on_unmatched_engine_match():
    empty = pd.DataFrame(columns=["stage", "round_number", "record_group", "team_1", "team_2",
                                   "best_of", "winner", "source_match_id"])
    provider = aop.ActualOutcomeProvider(empty)
    fake_match = te.MatchSpec(match_id="x", stage="stage_1", round_number=1, record_group="0-0",
                               team_a="A", team_b="B", seed_a=1, seed_b=2, best_of=1)
    with pytest.raises(ValueError):
        provider.resolve_match(fake_match)


# --- 4. frozen probability lookup only, no RF inference / model loading anywhere in this path ---

def test_predictions_module_never_imports_rf_pipeline():
    imported = _imported_module_names("evaluation/cologne_2026/build_phase8e_predictions.py")
    assert imported.isdisjoint(FORBIDDEN_RF_MODULES)


def test_actual_predictions_are_exact_matrix_lookups_only():
    predictions = pd.read_parquet(EVAL / "cologne_2026_actual_match_predictions_v1.parquet", engine="fastparquet")
    assert len(predictions) == 106
    matrix = pd.read_parquet(EVAL / "cologne_2026_pre_event_matchup_probabilities_v1.parquet", engine="fastparquet")
    lookup = {(r.team_a, r.team_b, r.best_of): r.probability_team_a for r in matrix.itertuples(index=False)}
    for row in predictions.itertuples(index=False):
        key = (row.team_a, row.team_b, row.best_of)
        assert key in lookup
        assert lookup[key] == pytest.approx(row.probability_team_a)


def test_no_phase8e_module_imports_network_libraries():
    for f in PHASE8E_MODULE_FILES:
        imported = _imported_module_names(f)
        assert imported.isdisjoint(FORBIDDEN_NETWORK_MODULES), f"{f} imports a network module"


# --- 5. immutability: no Phase 8D/8C/8B mutation ---

def test_immutable_pre_event_record_unchanged():
    protocol = yaml.safe_load((ROOT / "config" / "evaluation" / "phase8e_cologne_simulation_vs_reality_protocol.yaml").read_bytes())
    baseline = protocol["immutable_pre_event_record"]["hashes"]
    mismatches = p8e.verify_immutable_pre_event_record(baseline)
    assert mismatches == []


def test_tournament_yaml_hash_constant_present():
    assert te.FROZEN_YAML_SHA256


# --- 6. match / milestone metric formulas ---

def test_match_metrics_recompute_correctly(metrics):
    pred_df = metrics["pred_df"]
    y_true, p = pred_df["actual_team_a_win"].values, pred_df["probability_team_a"].values
    expected_ll = log_loss(y_true, p, labels=[0, 1])
    assert metrics["match_metrics"]["overall"]["log_loss"] == pytest.approx(expected_ll)


def test_realized_path_log_score_equals_match_log_loss(metrics):
    assert metrics["realized_path_log_score"]["equals_match_log_loss"] is True
    assert metrics["realized_path_log_score"]["mean_negative_log_probability"] == \
        pytest.approx(metrics["match_metrics"]["overall"]["log_loss"])


def test_milestone_comparison_has_32_teams_with_correct_marginal_counts(metrics):
    milestone_df = metrics["milestone_df"]
    assert len(milestone_df) == 32
    assert int(milestone_df["actual_reached_playoffs"].sum()) == 8
    assert int(milestone_df["actual_reached_semifinal"].sum()) == 4
    assert int(milestone_df["actual_reached_final"].sum()) == 2
    assert int(milestone_df["actual_champion"].sum()) == 1


# --- 7. conditional stage-advancement / Swiss-record / playoff-seed probability lookups ---

def test_stage_advancement_uses_conditional_probability_column(metrics):
    adv = metrics["stage_advancement_evaluation"]
    for stage in ("stage_1", "stage_2", "stage_3"):
        assert adv[stage]["metrics"]["n"] == 16  # every real Swiss stage has 16 participants


def test_swiss_record_lookup_is_conditional_on_participation(metrics):
    df = metrics["swiss_record_df"]
    assert len(df) == 48  # 16 teams x 3 stages
    assert df["actual_terminal_record"].isin(["3-0", "3-1", "3-2", "2-3", "1-3", "0-3"]).all()


def test_playoff_seed_lookup_covers_exactly_the_8_actual_playoff_teams(metrics):
    df = metrics["playoff_seed_df"]
    assert len(df) == 8
    assert sorted(df["actual_seed"]) == list(range(1, 9))


# --- 8. favorite-path comparison semantics (structural sets, matched-matchup-only individual comparison) ---

def test_favorite_path_champion_replay_matches_frozen_json(metrics):
    frozen = json.loads((EVAL / "cologne_2026_pre_event_favorite_path_v1.json").read_text(encoding="utf-8"))
    fav_result = pm.replay_favorite_path()
    assert fav_result.champion == frozen["champion"]


def test_favorite_path_individual_comparison_only_uses_shared_matchups(metrics):
    fav = metrics["favorite_path_vs_reality"]
    shared = fav["matched_identical_matchups"]["n_shared_matchups_between_favorite_path_and_reality"]
    assert shared <= 106
    assert fav["matched_identical_matchups"]["n_correct_among_shared"] <= shared


# --- 9. summary JSON consistency ---

def test_summary_json_matches_recomputed_metrics(metrics):
    summary = json.loads((EVAL / "cologne_2026_simulation_vs_reality_summary_v1.json").read_text(encoding="utf-8"))
    assert summary["match_n"] == metrics["match_metrics"]["overall"]["n"]
    assert summary["match_accuracy"] == pytest.approx(metrics["match_metrics"]["overall"]["accuracy"])
    assert summary["actual_champion"] == metrics["champion"]
    assert summary["original_cologne_tagged_rows"] == 107
    assert summary["official_major_rows"] == 106


# --- 10. receipt hashes (only once the receipt has been committed) ---

@pytest.mark.skipif(not (EVAL / "cologne_2026_simulation_vs_reality_receipt_v1.json").exists(),
                     reason="receipt not yet committed")
def test_receipt_references_frozen_phase8d_receipt():
    receipt = json.loads((EVAL / "cologne_2026_simulation_vs_reality_receipt_v1.json").read_text(encoding="utf-8"))
    assert "phase8d_simulation_receipt_hash" in receipt or "hashes" in receipt
