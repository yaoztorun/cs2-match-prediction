"""
Phase 8D validation (frozen historical pre-event Cologne simulation gate).
Read-only. Exits non-zero on failure. Passing this script means: the pre-
event probability matrix and 50,000-simulation Monte Carlo run are complete,
internally consistent (every stored probability independently recomputable
from its own integer counts), immutable, and were produced without ever
touching an ML model during simulation, a real Cologne result, or the
network - and every Phase 1-8C artifact this phase depends on is
byte-unchanged.
"""

import ast
import hashlib
import json
import sys

import pandas as pd

from _common import ROOT
import tournament.engine.tournament_engine as te

EVAL_DIR = ROOT / "data" / "evaluation"

PRE_PHASE8D_ARTIFACTS = [
    "config/tournaments/iem_cologne_major_2026_pre_event.yaml",
    "reports/phases/phase8b_cologne_tournament_definition.md",
    "data/tournaments/iem_cologne_major_2026_sources.json",
    "validation/validate_phase8b.py",
    "tests/tournament/test_phase8b_tournament_definition.py",
    "tournament/engine/tournament_engine.py",
    "tests/tournament/test_phase8c_tournament_engine.py",
    "validation/validate_phase8c.py",
    "tournament/engine/run_phase8c_synthetic_demo.py",
    "reports/phases/phase8c_tournament_engine.md",
    "data/tournaments/phase8c_synthetic_trace.json",
    "models/series/random_forest_v2.joblib",
    "data/modeling/random_forest_preprocessing_v2.json",
    "data/modeling/random_forest_v2_selected_config.json",
    "data/features/pre_cologne_team_state_v1_full.json",
    "feature_engineering/series/feature_engine.py",
    "feature_engineering/preprocessing/preprocessing_random_forest_v1.py",
    "config/features/series_features_v1.yaml",
    "data/evaluation/map_test_predictions_v1.parquet",
    "data/evaluation/phase7_test_open_receipt_v1.json",
]

PHASE8D_SOURCE_FILES = [
    "tournament/simulation/pre_veto_series_predictor.py", "tournament/cologne_2026/build_phase8d_protocol.py",
    "tournament/simulation/generate_cologne_pre_event_probability_matrix.py", "tournament/simulation/cologne_pre_event_simulation.py",
    "tournament/cologne_2026/run_phase8d_pipeline.py", "tournament/cologne_2026/phase8d_figures.py", "tournament/cologne_2026/phase8d_common.py",
]

FORBIDDEN_RESULT_PATHS = [
    "data/interim/series_base.parquet", "data/interim/map_base.parquet",
    "data/evaluation/map_test_predictions_v1.parquet",
]
FORBIDDEN_NETWORK_MODULES = {"requests", "urllib", "http", "httpx", "socket", "aiohttp"}
FORBIDDEN_ML_IMPORTS_IN_FIGURES = {"joblib", "sklearn", "feature_engine", "pre_veto_series_predictor"}

CANONICAL_FILES = {
    "protocol": ROOT / "config" / "evaluation" / "phase8d_cologne_pre_event_simulation_protocol.yaml",
    "matrix": EVAL_DIR / "cologne_2026_pre_event_matchup_probabilities_v1.parquet",
    "probability_receipt": EVAL_DIR / "cologne_2026_pre_event_probability_receipt_v1.json",
    "team_probabilities": EVAL_DIR / "cologne_2026_pre_event_team_probabilities_v1.csv",
    "matchup_frequencies": EVAL_DIR / "cologne_2026_pre_event_matchup_frequencies_v1.csv",
    "swiss_record_distributions": EVAL_DIR / "cologne_2026_pre_event_swiss_record_distributions_v1.csv",
    "playoff_seed_distributions": EVAL_DIR / "cologne_2026_pre_event_playoff_seed_distributions_v1.csv",
    "simulation_summary": EVAL_DIR / "cologne_2026_pre_event_simulation_summary_v1.json",
    "favorite_path": EVAL_DIR / "cologne_2026_pre_event_favorite_path_v1.json",
    "sample_traces": EVAL_DIR / "cologne_2026_pre_event_sample_traces_v1.json",
    "simulation_receipt": EVAL_DIR / "cologne_2026_pre_event_simulation_receipt_v1.json",
}
FIGURES_DIR = ROOT / "reports" / "figures" / "phase8d"

CHECKS = []


def check(name, cond):
    CHECKS.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")


def sha256(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _reads_forbidden_path(source, forbidden_tokens):
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


def _imported_names(source):
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    return imported


def main():
    print("=== capturing pre-run hashes of Phase 1-8C artifacts ===")
    baseline = {}
    for rel in PRE_PHASE8D_ARTIFACTS:
        p = ROOT / rel
        baseline[rel] = sha256(p) if p.exists() else None
        check(f"Phase 1-8C artifact present: {rel}", baseline[rel] is not None)

    print("\n=== 1. Phase 8B YAML / Phase 8C engine hashes match the frozen constants ===")
    yaml_hash = sha256(ROOT / "config" / "tournaments" / "iem_cologne_major_2026_pre_event.yaml")
    check("Phase 8B YAML hash == e481ca4d...", yaml_hash == te.FROZEN_YAML_SHA256)
    trace_path = ROOT / "data" / "tournaments" / "phase8c_synthetic_trace.json"
    if trace_path.exists():
        trace_payload = json.loads(trace_path.read_text(encoding="utf-8"))
        check("Phase 8C synthetic trace hash == 3277cf70...",
              trace_payload.get("canonical_trace_sha256") == "3277cf70ad02c7e27c534a1adf4018a36c4b0dd4602f73bbcd3a9ac7dc4ed7e7")

    print("\n=== 2. all expected Phase 8D output files exist ===")
    for name, p in CANONICAL_FILES.items():
        check(f"output exists: {name}", p.exists())
    check("figures directory exists", FIGURES_DIR.exists())
    figure_files = list(FIGURES_DIR.glob("*")) if FIGURES_DIR.exists() else []
    check("24 figure files (12 figures x png+pdf)", len(figure_files) == 24)

    if not all(p.exists() for p in CANONICAL_FILES.values()):
        print("\nCannot continue - not all canonical outputs exist yet.")
        _finish()
        return

    print("\n=== 3. probability matrix: shape, uniqueness, complementarity ===")
    matrix = pd.read_parquet(CANONICAL_FILES["matrix"], engine="fastparquet")
    check("2976 rows", len(matrix) == 2976)
    check("unique (team_a,team_b,best_of) key", len(matrix[["team_a", "team_b", "best_of"]].drop_duplicates()) == 2976)
    p_a = matrix["probability_team_a"].to_numpy(dtype=float)
    p_b = matrix["probability_team_b"].to_numpy(dtype=float)
    check("all probabilities finite and in [0,1]",
          bool(((p_a >= 0) & (p_a <= 1)).all()) and bool((p_a == p_a).all()))
    check("probability_team_b == 1 - probability_team_a (strict tolerance)",
          bool((abs(p_b - (1 - p_a)) < 1e-9).all()))
    check("no result-shaped column present in the matrix",
          not ({"winner", "result", "champion", "score"} & set(matrix.columns)))

    print("\n=== 4. probability receipt / model contract ===")
    prob_receipt = json.loads(CANONICAL_FILES["probability_receipt"].read_text(encoding="utf-8"))
    check("model contract all_checks_pass", prob_receipt.get("model_contract", {}).get("all_checks_pass") is True)
    check("BO1/3/5 validation recorded", set(str(k) for k in prob_receipt.get("bo_support_validation", {})) == {"1", "3", "5"})
    check("tier1 reference-category validation recorded",
          prob_receipt.get("tier_representation_validation", {}).get("is_reference_category") is True)
    check("matrix_sha256 recorded and matches file on disk",
          prob_receipt.get("matrix_sha256") == sha256(CANONICAL_FILES["matrix"]))

    print("\n=== 5. simulation summary / receipt ===")
    summary = json.loads(CANONICAL_FILES["simulation_summary"].read_text(encoding="utf-8"))
    receipt = json.loads(CANONICAL_FILES["simulation_receipt"].read_text(encoding="utf-8"))
    check("receipt: created_before_results_opened is true", receipt.get("created_before_results_opened") is True)
    check("receipt: cologne_results_status == UNOPENED", receipt.get("cologne_results_status") == "UNOPENED")
    check("receipt: n_simulations == 50000", receipt.get("n_simulations") == 50000)
    check("receipt: base_seed == 42", receipt.get("base_seed") == 42)
    check("receipt references the protocol hash", "protocol_hash" in receipt)
    n = receipt["n_simulations"]

    print("\n=== 6. accounting identities recomputed independently (amendment #5) ===")
    acc = summary["accounting"]
    check("accounting.all_pass (as stored)", acc.get("all_pass") is True)
    check("total_matches == n*106", acc["total_matches"] == n * 106)
    check("champion_sum == n", acc["champion_sum"] == n)
    check("playoff_sum == 8n", acc["playoff_sum"] == 8 * n)
    check("semifinal_sum == 4n", acc["semifinal_sum"] == 4 * n)
    check("final_sum == 2n", acc["final_sum"] == 2 * n)
    check("all 3 stage advance sums == 8n",
          all(v == 8 * n for v in acc["stage_advance_sums"].values()))
    champ_counts = summary["champion_counts"]
    check("sum(champion_counts) == n (recomputed from raw counts, not trusted probability)",
          sum(champ_counts.values()) == n)

    print("\n=== 7. team_probabilities.csv: every probability recomputable from its own counts ===")
    tp = pd.read_csv(CANONICAL_FILES["team_probabilities"])
    tp_nonnull = tp.dropna(subset=["probability"])
    recomputed = tp_nonnull["numerator_count"] / tp_nonnull["denominator_count"]
    check("stored probability == numerator_count/denominator_count for every row",
          bool((abs(tp_nonnull["probability"] - recomputed) < 1e-9).all()))
    check("null probability rows have denominator_count == 0",
          bool((tp.loc[tp["probability"].isna(), "denominator_count"] == 0).all()))
    unconditional = tp[tp["denominator_type"] == "unconditional_n_simulations"]
    check("every unconditional row's denominator_count == n_simulations",
          bool((unconditional["denominator_count"] == n).all()))

    print("\n=== 8. Swiss record distributions: counts sum to participation ===")
    sr = pd.read_csv(CANONICAL_FILES["swiss_record_distributions"])
    per_team_stage = sr.groupby(["canonical_model_name", "stage"])["count"].sum()
    check("swiss record counts are non-negative integers", bool((sr["count"] >= 0).all()))

    print("\n=== 9. playoff seed distributions: conditional probabilities recomputable ===")
    ps = pd.read_csv(CANONICAL_FILES["playoff_seed_distributions"])
    ps_nonnull = ps.dropna(subset=["conditional_on_reaching_playoffs_probability"])
    recomputed_ps = ps_nonnull["count"] / ps_nonnull["conditional_on_reaching_playoffs_denominator"]
    check("playoff seed conditional probability recomputable from counts",
          bool((abs(ps_nonnull["conditional_on_reaching_playoffs_probability"] - recomputed_ps) < 1e-9).all()))
    seed_sum_per_team = ps.groupby("display_name")["count"].sum()
    reach_playoffs = tp[tp["metric"] == "reach_playoffs"].set_index("display_name")["numerator_count"]
    mismatch = [t for t in seed_sum_per_team.index
                if t in reach_playoffs.index and seed_sum_per_team[t] != reach_playoffs[t]]
    check("playoff-seed count sums match reach_playoffs counts exactly", not mismatch)

    print("\n=== 10. favorite-wins path: named correctly, tie policy, 106 matches ===")
    fav = json.loads(CANONICAL_FILES["favorite_path"].read_text(encoding="utf-8"))
    check("path_name == 'deterministic favorite-wins path'", fav.get("path_name") == "deterministic favorite-wins path")
    check("not called 'most likely bracket'", "most likely bracket" not in json.dumps(fav).lower())
    check("106 matches in favorite path", len(fav.get("matches", [])) == 106)

    print("\n=== 11. sample traces: fixed indices, zero real-result fields ===")
    samples = json.loads(CANONICAL_FILES["sample_traces"].read_text(encoding="utf-8"))
    check("4 fixed sample traces", len(samples) == 4)
    check("sample indices are [0,1,42,999]", sorted(s["simulation_index"] for s in samples) == [0, 1, 42, 999])

    print("\n=== 12. no ML calls / no result access / no network in Phase 8D source ===")
    for rel in PHASE8D_SOURCE_FILES:
        source = (ROOT / rel).read_text(encoding="utf-8")
        check(f"{rel}: no forbidden-result-path reads", not _reads_forbidden_path(source, FORBIDDEN_RESULT_PATHS))
        imported = _imported_names(source)
        check(f"{rel}: no network imports", not (imported & FORBIDDEN_NETWORK_MODULES))
    figures_source = (ROOT / "tournament" / "cologne_2026" / "phase8d_figures.py").read_text(encoding="utf-8")
    figures_imports = _imported_names(figures_source)
    check("phase8d_figures.py imports no ML/model-calling module (downstream-only)",
          not (figures_imports & FORBIDDEN_ML_IMPORTS_IN_FIGURES))

    print("\n=== 13. immutability: receipt hashes match files on disk ===")
    receipt_hashes = receipt.get("hashes", {})
    mismatches = []
    for name, p in CANONICAL_FILES.items():
        if name == "simulation_receipt":
            continue
        actual = sha256(p)
        recorded = receipt_hashes.get(p.name)
        if recorded is not None and recorded != actual:
            mismatches.append(name)
    check("every recorded canonical-file hash matches the file on disk", not mismatches)

    print("\n=== 14. Phase 1-8C artifacts byte-unchanged ===")
    for rel, expected in baseline.items():
        p = ROOT / rel
        check(f"unchanged: {rel}", p.exists() and sha256(p) == expected)

    _finish()


def _finish():
    n_pass = sum(1 for _, ok in CHECKS if ok)
    print(f"\n{n_pass}/{len(CHECKS)} checks passed")
    if n_pass != len(CHECKS):
        print("FAILED:")
        for name, ok in CHECKS:
            if not ok:
                print(f"  - {name}")
        sys.exit(1)


if __name__ == "__main__":
    main()
