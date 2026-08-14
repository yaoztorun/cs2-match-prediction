"""
Phase 8D transactional pipeline orchestrator. Implements the 12-step
staging -> validate -> promote lifecycle (amendment #7): nothing is written
to a canonical path until every validation has passed and the final
simulation receipt has been computed. The receipt is written to staging
LAST and promoted LAST - it is the commit marker. A partially-promoted run
without a valid final receipt is never silently treated as complete.

No network access anywhere in this module or anything it imports (amendment
#8) - only local frozen artifacts are read, via an explicit allowlist
(amendment #9), never by scanning data/raw or data/interim.
"""

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import yaml

from _common import ROOT
import build_phase8d_protocol as protocol_mod
import cologne_pre_event_simulation as sim
import generate_cologne_pre_event_probability_matrix as gen_matrix
import phase8d_common as pc
import phase8d_figures as figs
import pre_veto_series_predictor as pvp
import tournament_engine as te

EVAL_DIR = ROOT / "data" / "evaluation"
STAGING_DIR = EVAL_DIR / ".phase8d_staging"
FIGURES_CANONICAL_DIR = ROOT / "reports" / "figures" / "phase8d"
PROTOCOL_CANONICAL_PATH = ROOT / "config" / "phase8d_cologne_pre_event_simulation_protocol.yaml"
RECEIPT_CANONICAL_PATH = EVAL_DIR / "cologne_2026_pre_event_simulation_receipt_v1.json"

# (staging-relative-name, canonical Path). Receipt is LAST - the commit marker.
CANONICAL_FILES = [
    (PROTOCOL_CANONICAL_PATH.name, PROTOCOL_CANONICAL_PATH),
    ("cologne_2026_pre_event_matchup_probabilities_v1.parquet",
     EVAL_DIR / "cologne_2026_pre_event_matchup_probabilities_v1.parquet"),
    ("cologne_2026_pre_event_probability_receipt_v1.json",
     EVAL_DIR / "cologne_2026_pre_event_probability_receipt_v1.json"),
    ("cologne_2026_pre_event_team_probabilities_v1.csv", EVAL_DIR / "cologne_2026_pre_event_team_probabilities_v1.csv"),
    ("cologne_2026_pre_event_matchup_frequencies_v1.csv", EVAL_DIR / "cologne_2026_pre_event_matchup_frequencies_v1.csv"),
    ("cologne_2026_pre_event_swiss_record_distributions_v1.csv",
     EVAL_DIR / "cologne_2026_pre_event_swiss_record_distributions_v1.csv"),
    ("cologne_2026_pre_event_playoff_seed_distributions_v1.csv",
     EVAL_DIR / "cologne_2026_pre_event_playoff_seed_distributions_v1.csv"),
    ("cologne_2026_pre_event_simulation_summary_v1.json", EVAL_DIR / "cologne_2026_pre_event_simulation_summary_v1.json"),
    ("cologne_2026_pre_event_favorite_path_v1.json", EVAL_DIR / "cologne_2026_pre_event_favorite_path_v1.json"),
    ("cologne_2026_pre_event_sample_traces_v1.json", EVAL_DIR / "cologne_2026_pre_event_sample_traces_v1.json"),
    # figures directory is promoted separately (a directory move), before the receipt
    (RECEIPT_CANONICAL_PATH.name, RECEIPT_CANONICAL_PATH),
]


class Phase8DPipelineError(Exception):
    pass


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    return sha256_bytes(Path(path).read_bytes())


# ---------------------------------------------------------------------------
# Step 1: preflight
# ---------------------------------------------------------------------------

def preflight_check():
    canonical_paths = [p for _, p in CANONICAL_FILES]
    existing = [p for p in canonical_paths if p.exists()]
    figures_exist = FIGURES_CANONICAL_DIR.exists() and any(FIGURES_CANONICAL_DIR.iterdir())
    if figures_exist:
        existing.append(FIGURES_CANONICAL_DIR)

    if not existing:
        return  # clean slate - normal case

    receipt_valid = False
    if RECEIPT_CANONICAL_PATH.exists():
        try:
            receipt = json.loads(RECEIPT_CANONICAL_PATH.read_text(encoding="utf-8"))
            receipt_valid = receipt.get("created_before_results_opened") is True and "hashes" in receipt
        except Exception:
            receipt_valid = False

    if receipt_valid:
        raise Phase8DPipelineError(
            "A completed, frozen Phase 8D simulation receipt already exists at "
            f"{RECEIPT_CANONICAL_PATH}. Refusing to overwrite a completed historical prediction run. "
            "If regeneration is genuinely intended, remove the existing canonical Phase 8D artifacts "
            "manually first.")

    raise Phase8DPipelineError(
        "Partial Phase 8D canonical artifacts detected WITHOUT a valid final simulation receipt - "
        "a previous run likely crashed mid-promotion. Refusing to silently continue from a mixture of "
        f"old and new files. Present: {[str(p) for p in existing]}. "
        "Manual recovery required: inspect and remove partial artifacts before rerunning.")


def setup_staging():
    if STAGING_DIR.exists():
        shutil.rmtree(STAGING_DIR)
    STAGING_DIR.mkdir(parents=True)
    return STAGING_DIR


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(n_simulations=None, base_seed=None, progress=print):
    preflight_check()
    staging = setup_staging()

    protocol = protocol_mod.build_protocol_dict()
    n_simulations = n_simulations or protocol["monte_carlo"]["n_simulations"]
    base_seed = base_seed if base_seed is not None else protocol["monte_carlo"]["base_seed"]
    sample_indices = protocol["reproducibility"]["fixed_sample_trace_indices"]

    protocol_yaml_text = yaml.safe_dump(protocol, sort_keys=True, default_flow_style=False)
    (staging / PROTOCOL_CANONICAL_PATH.name).write_text(protocol_yaml_text, encoding="utf-8")
    protocol_hash = sha256_bytes(protocol_yaml_text.encode("utf-8"))
    progress(f"[1/9] protocol written, hash={protocol_hash[:16]}...")

    context = pvp.load_predictor_context(tier=protocol["frozen_inputs"]["tier"])
    teams = pc.load_cologne_teams()
    bo_check = pvp.validate_bo_support(context, teams[0]["canonical_model_name"], teams[1]["canonical_model_name"],
                                        protocol["frozen_inputs"]["prediction_cutoff"])
    tier_check = pvp.verify_tier_representation(context, teams[0]["canonical_model_name"],
                                                 teams[1]["canonical_model_name"],
                                                 protocol["frozen_inputs"]["prediction_cutoff"])
    coverage_df = pvp.audit_team_coverage(context, teams)
    progress(f"[2/9] predictor context loaded, model contract verified, "
             f"BO1/3/5 + tier1 inference-contract validated, {len(teams)}-team coverage audited")

    matrix_df = gen_matrix.build_probability_matrix(context, teams=teams,
                                                      prediction_datetime=protocol["frozen_inputs"]["prediction_cutoff"])
    matrix_report = gen_matrix.validate_probability_matrix(matrix_df)
    orientation_diag = gen_matrix.compute_orientation_diagnostic(matrix_df)
    matrix_path = staging / "cologne_2026_pre_event_matchup_probabilities_v1.parquet"
    matrix_df.to_parquet(matrix_path, engine="fastparquet", index=False)
    matrix_hash = sha256_file(matrix_path)
    progress(f"[3/9] probability matrix: {matrix_report['row_count']} rows, "
             f"orientation mean symmetry error={orientation_diag['mean']:.5f}")

    probability_receipt = {
        "event_id": "iem_cologne_major_2026",
        "artifact": "cologne_2026_pre_event_matchup_probabilities_v1.parquet",
        "row_count": matrix_report["row_count"],
        "model_id": pvp.MODEL_ID,
        "prediction_cutoff": protocol["frozen_inputs"]["prediction_cutoff"],
        "tier": protocol["frozen_inputs"]["tier"],
        "model_contract": context.model_contract,
        "bo_support_validation": bo_check,
        "tier_representation_validation": tier_check,
        "matrix_validation_report": matrix_report,
        "orientation_diagnostic": orientation_diag,
        "protocol_hash": protocol_hash,
        "pipeline_hashes": protocol["frozen_inputs"]["pipeline_hashes"],
        "matrix_sha256": matrix_hash,
        "environment_versions": protocol["frozen_inputs"]["environment_versions"],
    }
    (staging / "cologne_2026_pre_event_probability_receipt_v1.json").write_text(
        json.dumps(probability_receipt, indent=2), encoding="utf-8")
    progress("[4/9] probability receipt written")

    rules = te.load_frozen_rules()
    entrants = pc.build_cologne_entrants()
    all_team_ids = [t["canonical_model_name"] for t in teams]
    matrix_lookup = sim.build_matrix_lookup(matrix_df)

    agg, sample_traces = sim.run_monte_carlo(n_simulations, base_seed, entrants, rules, matrix_lookup,
                                              all_team_ids, sample_indices=sample_indices)
    progress(f"[5/9] Monte Carlo complete: {agg.n_simulations} simulations")

    team_probabilities_df = sim.build_team_probabilities_df(agg, teams)
    swiss_records_df = sim.build_swiss_record_distributions_df(agg, teams)
    playoff_seeds_df = sim.build_playoff_seed_distributions_df(agg, teams)
    matchup_frequencies_df = sim.build_matchup_frequencies_df(agg, matrix_lookup, agg.n_simulations)
    rematch_report = sim.build_rematch_fallback_report(agg)
    favorite_path = sim.build_favorite_path_trace(entrants, rules, matrix_lookup)

    accounting = validate_accounting(agg, n_simulations)
    progress(f"[6/9] accounting identities: {accounting['all_pass']}")

    team_probabilities_df.to_csv(staging / "cologne_2026_pre_event_team_probabilities_v1.csv", index=False)
    matchup_frequencies_df.to_csv(staging / "cologne_2026_pre_event_matchup_frequencies_v1.csv", index=False)
    swiss_records_df.to_csv(staging / "cologne_2026_pre_event_swiss_record_distributions_v1.csv", index=False)
    playoff_seeds_df.to_csv(staging / "cologne_2026_pre_event_playoff_seed_distributions_v1.csv", index=False)
    (staging / "cologne_2026_pre_event_favorite_path_v1.json").write_text(
        json.dumps(favorite_path, indent=2), encoding="utf-8")
    (staging / "cologne_2026_pre_event_sample_traces_v1.json").write_text(
        json.dumps(sample_traces, indent=2), encoding="utf-8")

    simulation_summary = {
        "event_id": "iem_cologne_major_2026", "n_simulations": agg.n_simulations, "base_seed": base_seed,
        "expected_total_matches": n_simulations * 106,
        "actual_total_matches": accounting["total_matches"],
        "champion_counts": dict(agg.champion_counts),
        "accounting": accounting,
        "rematch_fallback_report": rematch_report,
        "orientation_diagnostic": orientation_diag,
        "bo5_data_sparsity_caveat": protocol["bo5_data_sparsity_caveat"]["note"],
    }
    (staging / "cologne_2026_pre_event_simulation_summary_v1.json").write_text(
        json.dumps(simulation_summary, indent=2), encoding="utf-8")
    progress("[7/9] aggregate artifacts written to staging")

    staging_figures_dir = staging / "figures"
    written_figures = figs.generate_all_figures(staging, staging_figures_dir)
    progress(f"[8/9] {len(written_figures)} figure files generated from staged aggregates")

    if not accounting["all_pass"]:
        raise Phase8DPipelineError(f"accounting identities failed: {accounting}")

    staged_hashes = {}
    for name, _ in CANONICAL_FILES:
        if name == RECEIPT_CANONICAL_PATH.name:
            continue
        p = staging / name
        if p.exists():
            staged_hashes[name] = sha256_file(p)
    figure_hashes = {str(p.relative_to(staging)): sha256_file(p) for p in staging_figures_dir.rglob("*") if p.is_file()}

    receipt = {
        "event_id": "iem_cologne_major_2026",
        "simulation_version": "phase8d_v1",
        "n_simulations": n_simulations,
        "base_seed": base_seed,
        "prediction_cutoff": protocol["frozen_inputs"]["prediction_cutoff"],
        "tier": protocol["frozen_inputs"]["tier"],
        "model_id": pvp.MODEL_ID,
        "protocol_hash": protocol_hash,
        "hashes": {**protocol["frozen_inputs"]["pipeline_hashes"], "protocol_yaml": protocol_hash,
                   "probability_matrix": matrix_hash, **staged_hashes, **figure_hashes},
        "environment_versions": protocol["frozen_inputs"]["environment_versions"],
        "accounting": accounting,
        "created_before_results_opened": True,
        "cologne_results_status": "UNOPENED",
    }
    (staging / RECEIPT_CANONICAL_PATH.name).write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    progress("[9/9] final simulation receipt written to staging (commit marker)")

    promote(staging)
    progress("promotion complete - Phase 8D artifacts are now canonical")
    return receipt


def validate_accounting(agg, n_simulations):
    total_matches = sum(b["total_matches"] for b in agg.rematch_by_stage_round.values()) + 7 * n_simulations
    champ_sum = sum(t["win_tournament"] for t in agg.team.values())
    playoff_sum = sum(t["reach_playoffs"] for t in agg.team.values())
    sf_sum = sum(t["reach_semifinal"] for t in agg.team.values())
    final_sum = sum(t["reach_final"] for t in agg.team.values())
    stage_advance_sums = {s: sum(t[f"{s}_advances"] for t in agg.team.values()) for s in sim.SWISS_STAGES}
    checks = {
        "total_matches_expected": n_simulations * 106,
        "total_matches_actual": total_matches,
        "champion_sum_equals_n": champ_sum == n_simulations,
        "playoff_sum_equals_8n": playoff_sum == 8 * n_simulations,
        "semifinal_sum_equals_4n": sf_sum == 4 * n_simulations,
        "final_sum_equals_2n": final_sum == 2 * n_simulations,
        "stage_advance_sums": stage_advance_sums,
        "stage_advance_sums_equal_8n": all(v == 8 * n_simulations for v in stage_advance_sums.values()),
        "champion_sum": champ_sum, "playoff_sum": playoff_sum, "semifinal_sum": sf_sum, "final_sum": final_sum,
        "total_matches": total_matches,
    }
    checks["all_pass"] = (
        total_matches == n_simulations * 106 and checks["champion_sum_equals_n"]
        and checks["playoff_sum_equals_8n"] and checks["semifinal_sum_equals_4n"]
        and checks["final_sum_equals_2n"] and checks["stage_advance_sums_equal_8n"]
    )
    return checks


def promote(staging):
    staging_figures_dir = staging / "figures"
    if FIGURES_CANONICAL_DIR.exists():
        raise Phase8DPipelineError(f"canonical figures directory already exists: {FIGURES_CANONICAL_DIR}")
    FIGURES_CANONICAL_DIR.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging_figures_dir, FIGURES_CANONICAL_DIR)

    for name, canonical_path in CANONICAL_FILES:
        staged_path = staging / name
        if not staged_path.exists():
            raise Phase8DPipelineError(f"expected staged file missing before promotion: {name}")
        if canonical_path.exists():
            raise Phase8DPipelineError(f"canonical path already exists, aborting promotion: {canonical_path}")
        canonical_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staged_path, canonical_path)

    if STAGING_DIR.exists():
        shutil.rmtree(STAGING_DIR)


if __name__ == "__main__":
    receipt = run_pipeline()
    print(json.dumps({"champion_counts_top5": sorted(receipt["hashes"].items())[:1]}, indent=2))
    sys.exit(0)
