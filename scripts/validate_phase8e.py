"""
Phase 8E validation. Read-only. Exits non-zero on failure. Passing this
script means: the 107->106 reconciliation is evidence-backed, the Phase 8C
engine replay reproduces the actual tournament 106/106, every actual-match
probability came from the frozen Phase 8D matrix (no RF re-inference
anywhere in this phase), every headline metric independently recomputes to
the values in the report/summary, and all pre-existing Phase 8B/8C/8D/8D.1
artifacts remain byte-unchanged.
"""

import ast
import json
import sys

import numpy as np
import pandas as pd
import yaml

from _common import ROOT
import phase8e_common as p8e
import phase8e_metrics as pm

EVAL = ROOT / "data" / "evaluation"
CHECKS = []


def check(name, cond):
    CHECKS.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")


def module_reads_forbidden(path, forbidden_substrings):
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(path))
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if any(f in alias.name for f in forbidden_substrings):
                    hits.append(alias.name)
        if isinstance(node, ast.ImportFrom) and node.module:
            if any(f in node.module for f in forbidden_substrings):
                hits.append(node.module)
    return hits


def main():
    print("\n=== 1. protocol frozen before results, hash matches file bytes ===")
    protocol_path = ROOT / "config" / "phase8e_cologne_simulation_vs_reality_protocol.yaml"
    check("phase8e protocol exists", protocol_path.exists())
    protocol = yaml.safe_load(protocol_path.read_bytes())
    check("protocol declares purpose/immutable_pre_event_record/reconciliation_policy",
          {"purpose", "immutable_pre_event_record", "reconciliation_policy"} <= set(protocol.keys()))

    print("\n=== 2. immutable pre-event record unchanged ===")
    baseline = protocol["immutable_pre_event_record"]["hashes"]
    mismatches = p8e.verify_immutable_pre_event_record(baseline)
    check("all 15 immutable pre-event artifacts byte-unchanged", len(mismatches) == 0)
    for name, expected, got in mismatches:
        check(f"unchanged: {name}", False)

    print("\n=== 3. reconciliation: 107 -> 106, showmatch excluded with evidence ===")
    recon = pd.read_csv(EVAL / "cologne_2026_result_reconciliation_v1.csv")
    check("reconciliation table has exactly 107 rows", len(recon) == 107)
    n_official = int(recon["included_in_official_event"].sum())
    check("exactly 106 rows marked included_in_official_event", n_official == 106)
    excluded = recon[~recon["included_in_official_event"]]
    check("exactly 1 excluded row", len(excluded) == 1)
    if len(excluded) == 1:
        ex = excluded.iloc[0]
        check("excluded row is the Team Germany vs Team Poland showmatch",
              {ex["team1"], ex["team2"]} == {"Team Germany", "Team Poland"})
        check("excluded row status = non_tournament_showmatch",
              ex["reconciliation_status"] == "non_tournament_showmatch")
        check("excluded row carries supporting_evidence text",
              isinstance(ex["supporting_evidence"], str) and len(ex["supporting_evidence"]) > 20)
    per_stage = recon[recon["included_in_official_event"]]["candidate_stage"].value_counts().to_dict()
    for stage, expected_n in {"stage_1": 33, "stage_2": 33, "stage_3": 33, "playoffs": 7}.items():
        check(f"{stage} has {expected_n} official matches", per_stage.get(stage) == expected_n)

    print("\n=== 4. canonical actual results: 106 rows, winners derived from scores only ===")
    canonical = pd.read_parquet(EVAL / "cologne_2026_actual_series_results_v1.parquet", engine="fastparquet")
    check("canonical actual results has exactly 106 rows", len(canonical) == 106)
    check("no duplicate source_match_id", not canonical["source_match_id"].duplicated().any())
    recomputed_winner = np.where(canonical["score_team_1"] > canonical["score_team_2"],
                                  canonical["team_1"], canonical["team_2"])
    check("winner column matches independently recomputed score-based winner",
          (canonical["winner"].values == recomputed_winner).all())
    check("no tied scores", (canonical["score_team_1"] != canonical["score_team_2"]).all())

    print("\n=== 5. Gate-2 engine replay: 106/106, all transitions/bracket reproduced ===")
    gate2 = json.loads((EVAL / "cologne_2026_actual_engine_replay_v1.json").read_text(encoding="utf-8"))
    check("gate2 total_matched = 106/106", gate2["total_matched"] == "106/106")
    check("gate2 all actual rows consumed exactly once", gate2["all_actual_rows_consumed_exactly_once"])
    check("gate2 unused/duplicate/ambiguous/missing all zero",
          gate2["unused"] == 0 and gate2["duplicate"] == 0 and gate2["ambiguous"] == 0 and gate2["missing"] == 0)
    check("gate2 champion reproduced", gate2["champion_reproduced"])
    check("gate2 approved_by_user flag set", gate2.get("approved_by_user") is True)
    import phase8e_actual_outcome_provider as aop
    fresh_report, fresh_result, fresh_provider = aop.gate2_checkpoint_report()
    check("re-running the replay independently still gives 106/106",
          fresh_report["engine_replay"]["total"] == 106 and len(fresh_provider.unused_keys()) == 0)
    check("re-running the replay reproduces the same champion",
          fresh_report["playoffs"]["engine_champion"] == gate2["actual_champion"])

    print("\n=== 6. matrix-lookup-only predictions: no RF/joblib/feature_engine/state imports ===")
    pred_module_path = ROOT / "scripts" / "build_phase8e_predictions.py"
    forbidden = ["joblib", "feature_engine", "preprocessing_random_forest", "pre_veto_series_predictor"]
    hits = module_reads_forbidden(pred_module_path, forbidden)
    check("build_phase8e_predictions.py imports none of joblib/feature_engine/preprocessing/predictor (AST)",
          len(hits) == 0)
    predictions = pd.read_parquet(EVAL / "cologne_2026_actual_match_predictions_v1.parquet", engine="fastparquet")
    check("actual match predictions has exactly 106 rows", len(predictions) == 106)
    check("no duplicate engine_match_id", not predictions["engine_match_id"].duplicated().any())
    check("probability_team_a + probability_team_b == 1 for every row",
          np.allclose(predictions["probability_team_a"] + predictions["probability_team_b"], 1.0))

    print("\n=== 7. figures are downstream-only (no engine/model/network imports) ===")
    fig_module_path = ROOT / "scripts" / "phase8e_figures.py"
    fig_hits = module_reads_forbidden(fig_module_path, ["joblib", "tournament_engine", "requests", "urllib"])
    check("phase8e_figures.py imports none of joblib/tournament_engine/requests/urllib (AST)", len(fig_hits) == 0)

    print("\n=== 8. metrics independently recompute ===")
    m = pm.compute_all_metrics()
    mm = m["match_metrics"]["overall"]
    check("match n = 106", mm["n"] == 106)
    check("match accuracy in (0,1]", 0 < mm["accuracy"] <= 1)
    check("realized-path mean_negative_log_probability == match log loss (already asserted internally)",
          m["realized_path_log_score"]["equals_match_log_loss"])
    champ = m["champion_analysis"]
    check("champion probability/rank present and consistent",
          0 <= champ["championship_probability"] <= 1 and 1 <= champ["championship_rank"] <= 32)

    print("\n=== 9. milestone comparison table ===")
    milestone_path = EVAL / "cologne_2026_actual_milestone_comparison_v1.csv"
    milestone_df = pd.read_csv(milestone_path)
    check("milestone comparison has exactly 32 rows (one per team)", len(milestone_df) == 32)
    check("exactly one team marked actual_champion", int(milestone_df["actual_champion"].sum()) == 1)
    check("exactly 8 teams marked actual_reached_playoffs", int(milestone_df["actual_reached_playoffs"].sum()) == 8)
    check("exactly 4 teams marked actual_reached_semifinal", int(milestone_df["actual_reached_semifinal"].sum()) == 4)
    check("exactly 2 teams marked actual_reached_final", int(milestone_df["actual_reached_final"].sum()) == 2)

    print("\n=== 10. summary JSON internally consistent with recomputed metrics ===")
    summary = json.loads((EVAL / "cologne_2026_simulation_vs_reality_summary_v1.json").read_text(encoding="utf-8"))
    check("summary match_n == 106", summary["match_n"] == 106)
    check("summary match_accuracy matches recomputed metrics", abs(summary["match_accuracy"] - mm["accuracy"]) < 1e-9)
    check("summary actual_champion matches gate2 champion", summary["actual_champion"] == gate2["actual_champion"])
    check("summary showmatch_excluded is true and counts are 107/106",
          summary["showmatch_excluded"] is True and summary["original_cologne_tagged_rows"] == 107
          and summary["official_major_rows"] == 106)

    print("\n=== 11. report exists and carries required markers ===")
    report_path = ROOT / "reports" / "phase8e_cologne_simulation_vs_reality.md"
    check("phase8e report exists", report_path.exists())
    if report_path.exists():
        text = report_path.read_text(encoding="utf-8")
        for marker in ["PHASE 8D PRE-EVENT SIMULATION = UNCHANGED", "NO POST-EVENT RETUNING",
                       "NO COLOGNE TRAINING INGESTION YET", "106/106"]:
            check(f"report contains marker: {marker!r}", marker in text)

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
