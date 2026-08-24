"""
Builds the frozen Phase 8E "simulation vs reality" evaluation protocol dict,
written to config/phase8e_cologne_simulation_vs_reality_protocol.yaml BEFORE
any actual Cologne result is read. Every policy the user mandated (sections
A-I of the original task spec, plus the 22 mandatory amendments to the
approved plan) is declared here first, exactly once, and every later script
must conform to it rather than re-deciding policy at read time.
"""

import hashlib

import yaml

from _common import ROOT
import evaluation.cologne_2026.phase8e_common as p8e

ARTIFACT_PATHS = {
    "protocol": "config/evaluation/phase8e_cologne_simulation_vs_reality_protocol.yaml",
    "result_source_manifest": "data/tournaments/iem_cologne_major_2026_actual_results_sources.json",
    "result_source_snapshot": "data/tournaments/iem_cologne_major_2026_actual_result_source_snapshot_v1.json",
    "reconciliation_table": "data/evaluation/cologne_2026_result_reconciliation_v1.csv",
    "actual_series_results": "data/evaluation/cologne_2026_actual_series_results_v1.parquet",
    "actual_match_predictions": "data/evaluation/cologne_2026_actual_match_predictions_v1.parquet",
    "milestone_comparison": "data/evaluation/cologne_2026_actual_milestone_comparison_v1.csv",
    "simulation_vs_reality_summary": "data/evaluation/cologne_2026_simulation_vs_reality_summary_v1.json",
    "simulation_vs_reality_receipt": "data/evaluation/cologne_2026_simulation_vs_reality_receipt_v1.json",
    "figures_dir": "reports/figures/phase8e",
    "report": "reports/phases/phase8e_cologne_simulation_vs_reality.md",
}

EXPECTED_OFFICIAL_MATCH_COUNTS = {
    "stage_1": 33, "stage_2": 33, "stage_3": 33, "playoffs": 7, "total": 106,
}

DEV_VALIDATION_CONTEXT = {
    "phase": "phase4_logistic_regression_and_rf_v2_selection",
    "log_loss": 0.6514, "brier": 0.2298, "roc_auc": 0.6566, "accuracy": 0.6068,
    "note": "Frozen BEFORE Cologne, on chronologically later development validation data. "
            "Shown as context only - never reinterpreted or used to re-select a model.",
}


def build_protocol_dict():
    return {
        "event_id": "iem_cologne_major_2026",
        "phase": "phase8e_simulation_vs_reality",
        "purpose": "What actually happened at IEM Cologne Major 2026, and how well did the frozen "
                   "Phase 8D pre-event prediction system (RF V2 + Phase 8C tournament engine) "
                   "represent that realized tournament? Evaluation-only: no retraining, "
                   "recalibration, feature change, or state update happens in this phase.",

        # ---- immutable pre-event record: hashed here, re-verified by every downstream script ----
        "immutable_pre_event_record": {
            "note": "These 14 items (15 files - Phase 8D.1's provenance bullet is 2 files) are the "
                    "complete frozen pre-event record. None may change during or after Phase 8E. "
                    "Hashes below are the baseline; evaluation/cologne_2026/phase8e_common.py re-verifies them.",
            "hashes": p8e.hash_immutable_pre_event_record(),
        },

        # ---- A. actual-result reconciliation rules ----
        "reconciliation_policy": {
            "expected_official_match_counts": EXPECTED_OFFICIAL_MATCH_COUNTS,
            "dataset_row_count": 107,
            "rule": "Inspect all 107 cologne_2026-tagged series_base.parquet rows. Do not assume 106, "
                    "delete one arbitrarily, take the first 106, or force the count to fit. Every one "
                    "of the 107 rows gets an explicit reconciliation_status/reconciliation_reason.",
            "winner_derivation": "winner = team with score_team_1 > score_team_2 (or independently "
                                  "confirmed official outcome). NEVER the historically-broken "
                                  "team1_series_win/team1_win field. Assert no ties; assert winner is "
                                  "one of the two teams.",
            "visibility_policy_amendment_4": (
                "The reconciliation CSV preserves all 107 original dataset rows (never deletes/hides "
                "any) and adds an explicit included_in_official_event: true/false column. Every "
                "excluded row records source_match_id, datetime, teams, best_of, score, tournament "
                "label, reconciliation_status, reconciliation_reason, supporting_evidence, and a "
                "confidence/status field for that explanation. The canonical 106-row parquet is a "
                "derived, clean view of this table - the original 107-row dataset is never modified."
            ),
            "bo1_arithmetic_diagnostic_amendment_5": (
                "Pre-planning arithmetic (BO5=1 matches expectation; BO3=65 implies Stage1+2 BO3=26 "
                "implies expected Stage1+2 BO1=40 vs observed 41) is recorded as a DIAGNOSTIC LEAD "
                "only. It is NOT sufficient evidence to select which row is extra. The excluded row "
                "must be identified through matchup/stage/source reconciliation evidence, never "
                "chosen merely because arithmetic says one extra BO1 row must exist somewhere."
            ),
            "stop_condition": "If the 107-row set cannot be confidently reconciled to the official "
                               "106-match event with documented evidence for the excluded row(s), "
                               "STOP. Do not compute any evaluation metric from an uncertain event table.",
        },

        # amendment 6: team identity
        "team_identity_policy": {
            "rule": "Reuse the frozen Phase 8B identity mapping (config/tournaments/"
                    "iem_cologne_major_2026_pre_event.yaml participants[*].canonical_model_name, "
                    "cross-checked against data/interim/team_identity_policy.csv). No fuzzy/"
                    "edit-distance matching is introduced in Phase 8E. Every actual-result source "
                    "display name must resolve to exactly one frozen canonical_model_name.",
            "requirement": "All 32/32 Cologne team identities must resolve consistently with Phase 8B.",
            "stop_condition": "If a result source appears to represent a different roster/org identity "
                               "than the frozen Phase 8B resolution, STOP and investigate. Never "
                               "silently create a new alias during evaluation.",
        },

        # amendment 1: result source provenance
        "result_source_provenance_policy": {
            "preferred_hierarchy": [
                "official ESL / event source where practical", "Valve / official tournament documentation",
                "HLTV event/match records", "Liquipedia", "dataset (kaggle_ektarr) as the last resort",
            ],
            "manifest_artifact": ARTIFACT_PATHS["result_source_manifest"],
            "snapshot_artifact": ARTIFACT_PATHS["result_source_snapshot"],
            "required_fields_per_official_match": [
                "official/source match identifier where available", "stage", "round", "team names as "
                "published", "best_of", "score", "winner", "source publisher", "source URL/reference",
                "retrieval date", "notes",
            ],
            "note": "Provenance data, not model input. Hashed and included in the Phase 8E receipt. If "
                    "raw source pages/files are locally archived for reproducibility, those are hashed "
                    "too. Phase 8B's PRE-EVENT source manifest (data/tournaments/"
                    "iem_cologne_major_2026_sources.json) is never modified.",
        },

        # amendment 3: stage/round labels from evidence, not engine placement (avoids circularity)
        "stage_round_labeling_policy": {
            "rule": "stage/round/playoff_round/best_of on the canonical actual-results table are "
                    "derived from actual tournament/source evidence, never by asking the Phase 8C "
                    "engine where a match 'would fit'. The Phase 8C engine, run independently with "
                    "actual winners, must then reproduce that structure on its own. Only after both "
                    "are built independently are they compared.",
            "pipeline": "actual tournament record -> canonical 106-match table  [INDEPENDENTLY]  "
                        "Phase 8C engine + actual winners -> generated tournament path  =>  compare "
                        "generated path == canonical actual path.",
        },

        # B. actual-tournament replay validation / amendment 2
        "replay_validation_policy": {
            "actual_outcome_provider": "New module outside tournament_engine.py (never modified). "
                                        "Built from the evidence-labeled canonical 106-row table.",
            "match_key_contract": {
                "swiss_stages": ["stage", "round_number", "record_group (where applicable)",
                                  "unordered canonical team pair", "best_of"],
                "playoffs": ["playoff round", "unordered canonical team pair", "best_of"],
            },
            "rule": "resolve_match() requires exactly one unconsumed actual result satisfying the full "
                    "key. Zero matches -> STOP. More than one match -> STOP. The provider never picks "
                    "'the first' ambiguous candidate.",
            "requirement": "106/106 actual matches must match the engine-generated tournament path: "
                            "every matchup, every stage transition, every Stage-1/2/3 qualifier, every "
                            "playoff seed, every quarterfinal/semifinal/grand final.",
            "stop_condition": "If any engine-generated pairing disagrees with the actual tournament, "
                               "STOP. Investigate before evaluating predictions.",
        },

        # amendment 7 + 18: matrix-lookup purity / score isolation
        "matrix_lookup_purity_policy": {
            "rule": "The code path producing cologne_2026_actual_match_predictions_v1.parquet must not "
                    "import joblib, the RF V2 model, feature_engine, RF preprocessing, or pre-Cologne "
                    "state. It needs only the validated actual replay trace and the frozen Phase 8D "
                    "probability matrix. Statically testable (AST import guard, mirroring Phase 8D's "
                    "forbidden-path guard).",
            "score_isolation": "Actual scores are used only for winner derivation, result verification, "
                                "and reporting. They are never passed into feature computation, model "
                                "input, probability generation, or any state update - mechanically "
                                "enforced by this matrix-lookup-only design, not just by convention.",
        },

        # amendment 8: exact lookup requirement
        "exact_lookup_policy": {
            "key": "(team_a, team_b, best_of) against data/evaluation/"
                   "cologne_2026_pre_event_matchup_probabilities_v1.parquet",
            "requirement": "106/106 exact successful lookups required.",
            "forbidden": ["orientation reversal fallback", "approximate team matching", "symmetrization",
                          "probability averaging", "recomputation/re-inference"],
            "stop_condition": "If an exact lookup fails for any actual match, STOP.",
        },

        # C. match-level metrics / amendment 9, 10, 17
        "match_level_metrics_policy": {
            "class_semantics": "y_true = 1 iff actual winner == engine-oriented team_a; p = "
                                "probability_team_a. All binary metrics computed from this contract. "
                                "predicted_team_a_win = (p >= 0.5), using the already-frozen Phase 8D "
                                "tie policy (p==0.5 resolves to team_a). ROC-AUC/Precision/Recall/F1's "
                                "positive class is 'engine-oriented team_a wins' - an evaluation "
                                "representation only, not a CS side.",
            "primary_metrics": ["log_loss", "brier_score", "roc_auc"],
            "secondary_metrics": ["accuracy", "precision", "recall", "f1"],
            "threshold": 0.5,
            "no_threshold_optimization": True,
            "no_calibration": True,
            "constant_baseline_amendment_10": "p=0.5 for all 106 actual matches, reported as Log Loss/"
                                               "Brier/AUC(=0.5)/accuracy under the same >=0.5 -> team_a "
                                               "convention, explicitly labeled a constant-probability "
                                               "baseline (not a newly-developed model baseline).",
            "n_reporting": "n=106 unless authoritative evidence proves the mechanical 106 expectation "
                           "itself wrong, in which case the reconciled official count is used and "
                           "stated explicitly.",
            "small_sample_policy_amendment_17": "Every subgroup (stage x BO) displays n prominently. "
                                                 "BO5 (n=1 expected) is always labeled 'INSUFFICIENT "
                                                 "SAMPLE FOR GENERAL INFERENCE'. ROC-AUC is never "
                                                 "computed/interpreted where only one class is present "
                                                 "(reported N/A). Tiny-n subgroups get point estimates "
                                                 "only, never ranking claims.",
        },

        # D. tournament-level metrics / amendments 12, 13, 14, 15
        "tournament_level_metrics_policy": {
            "realized_path_log_score_amendment_12": {
                "definition": "log_probability_realized_path = sum(log(p_actual_winner_i)) over the "
                               "106 actual matches; mean_negative_log_probability = "
                               "-1/106 * sum(log(p_actual_winner_i)), which must equal the binary "
                               "match-level Log Loss within numerical tolerance.",
                "naming": "Call the sum 'conditional realized-path log probability' / 'realized "
                          "tournament path log score' - never an unconditional probability of the "
                          "entire Major independent of tournament mechanics (the path is conditional "
                          "on deterministic pairing rules induced by preceding outcomes). If "
                          "exp(sum_log_probability) is computed for completeness, it stays in "
                          "log-space in any foregrounded report text (raw product is astronomically "
                          "small and not foregrounded).",
            },
            "milestone_denominator_policy_amendment_13": "Numerator, denominator, and which "
                "conditional probability was used are always preserved together. Stage-2 advancement "
                "uses P(advance Stage 2 | participates Stage 2), not an unconditional probability over "
                "all 50,000 simulations (direct Stage-2 entrants have a structurally-50,000 "
                "participation denominator). Playoff seed uses P(seed=k | reaches playoffs), with "
                "playoff-qualification probability reported separately - never silently multiplied "
                "together unless explicitly computing and labeling a joint probability.",
            "champion_context_amendment_14": ["pre-event championship probability", "rank among 32",
                "cumulative championship probability of teams ranked above the actual champion",
                "whether the champion was inside top 1 / top 3 / top 5 / top 8 by pre-event "
                "championship probability"],
            "champion_context_note": "Descriptive framing only - no significance is invented from "
                                      "these ranks.",
            "top_k_tiebreak_policy_amendment_15": {
                "rule_frozen_before_results": ["1. higher frozen probability", "2. better frozen "
                    "pre-event/VRS seed", "3. canonical team name"],
                "applies_to": ["top-8 playoff", "top-4 semifinal", "top-2 final", "top-1 champion"],
                "note": "Probability ties are never resolved using actual outcome information.",
            },
            "multiclass_champion_score": "-log(P(actual champion)) may additionally be reported. The "
                "32-team champion distribution is never presented as 32 independent binary events.",
        },

        # E. stage-level comparison rules
        "stage_level_comparison_policy": {
            "report_separately": ["stage_1", "stage_2", "stage_3", "playoffs"],
            "per_stage_metrics": ["n", "accuracy", "auc (N/A if one class absent)", "log_loss", "brier"],
            "advancement_conditional_probabilities": {
                "stage_1": "P(advance from Stage 1)",
                "stage_2": "P(advance from Stage 2 | participates Stage 2)",
                "stage_3": "P(advance from Stage 3 | participates Stage 3)",
            },
            "swiss_record_probability": "P(actual terminal record | participates in that stage) - "
                                         "conditional on participation, never unconditional record "
                                         "probability where participation itself was conditional.",
            "bo_breakdown": {"groups": ["BO1", "BO3", "BO5"], "metrics": ["n", "accuracy",
                              "auc where defined", "log_loss", "brier"]},
        },

        # F. small-sample warnings (see also match_level_metrics_policy.small_sample_policy_amendment_17)
        "small_sample_warning_policy": {
            "bo5_n": 1,
            "bo5_label": "INSUFFICIENT SAMPLE FOR GENERAL INFERENCE",
            "rule": "One BO5 result is never interpreted as evidence of BO5 model quality. Any "
                    "subgroup with very small effective n gets point estimates only, descriptively, "
                    "with no ranking claims.",
        },

        # amendment 25 (uncertainty discipline) folded in here since it governs presentation/language
        "uncertainty_discipline_policy": {
            "rule": "This is one real tournament. Never claim 'the model is proven to generalize' or "
                    "'the simulation is statistically validated' from one event. If bootstrap intervals "
                    "are produced for the 106 actual matches they are labeled DESCRIPTIVE (same-team "
                    "repeated matches create dependence, so ordinary IID bootstrap intervals are not "
                    "fully valid inferential confidence intervals). Point estimates alone are "
                    "acceptable.",
        },

        # amendment 11: dev-vs-Cologne language policy
        "development_vs_external_event_policy": {
            "dev_validation_context": DEV_VALIDATION_CONTEXT,
            "rule": "Never call the difference between Phase 4 development validation and Cologne "
                    "'improvement' or 'degradation' without qualification (sample size, opponent "
                    "distribution, BO distribution, and tournament composition all differ, and Cologne "
                    "is one event). Use 'external-event metric difference' wording, e.g. 'Cologne "
                    "external-event AUC was X, compared with 0.6566 on the earlier chronological "
                    "development validation.' The prior model-selection decision is never "
                    "reinterpreted.",
        },

        # amendment 16: favorite-path comparison semantics
        "favorite_path_comparison_policy": {
            "rule": "Keep ACTUAL-MATCH MODEL ACCURACY strictly distinct from DETERMINISTIC FAVORITE-"
                    "PATH SIMILARITY. Never report 'X/106 path matches correct' past the point the "
                    "deterministic path diverges from reality. Compare at the structural level "
                    "instead: Stage-1 advancer set, Stage-2 advancer set, Stage-3/playoff set, "
                    "semifinalist set, finalist set, champion. Individual match predictions are only "
                    "compared where the exact same matchup occurs in both the deterministic path and "
                    "reality - divergent downstream matchups are never penalized as if they were "
                    "wrong predictions.",
        },

        # amendment 19: network access scoping
        "network_access_policy": {
            "network_access_permitted": True,
            "scope": "Isolated to reconciliation/result-source acquisition only (building the result "
                     "source manifest/snapshot and the reconciliation table). Unlike Phase 8D, this is "
                     "intentional: actual-result verification is the point of Phase 8E.",
            "downstream_offline_requirement": "Engine replay, prediction lookup, metrics, summary, "
                                               "figures, and the validator operate only on frozen local "
                                               "Phase 8E actual-result artifacts and frozen Phase 8D "
                                               "artifacts. Once the canonical actual-results artifact is "
                                               "frozen, evaluation is reproducible fully offline.",
        },

        # G. figure list / amendment 21
        "figures": {
            "tooling": "matplotlib only, no seaborn. High-resolution PNG + PDF (300 DPI, mirroring "
                       "phase8d_figures.py's _save() helper and palette: BAR_COLOR=#2b6cb0, "
                       "ACCENT_COLOR=#c05621, GRID_COLOR=#d9d9d9, cmap=Blues for heatmaps).",
            "list": [
                "01 pre-event championship probabilities, actual champion highlighted",
                "02 pre-event playoff probabilities, actual playoff teams marked",
                "03 predicted vs actual tournament progression matrix",
                "04 top-8 predicted playoff teams vs actual playoff teams",
                "05 match-level confusion matrix",
                "06 match-level ROC curve",
                "07 actual-winner probability distribution",
                "08 match log loss / accuracy by stage",
                "09 stage advancement prediction vs reality",
                "10 Swiss actual-record probability matrix",
                "11 model-favorite path vs actual progression",
                "12 compact simulation-vs-reality presentation summary",
            ],
            "provenance_policy_amendment_21": "Every figure is generated downstream only from the "
                "canonical actual-results artifact, the actual-match predictions artifact, milestone/"
                "metric tables, and frozen Phase 8D aggregates. No figure script opens the RF model, "
                "runs the tournament engine, reads raw match data, or accesses the network - statically "
                "checked (AST import guard) where practical.",
        },

        # H. artifact paths
        "artifact_paths": ARTIFACT_PATHS,

        # I. no-retuning / no-model-change policy / amendment 22
        "no_model_development_policy": {
            "forbidden": ["retraining", "feature engineering", "calibration", "probability correction",
                          "threshold adjustment", "model replacement", "new ensemble", "changed state",
                          "changed cutoff", "new baseline model not already frozen before Cologne",
                          "RF re-inference of any kind"],
            "cologne_training_ingestion": "NOT in Phase 8E. Cologne results are evaluation data only "
                                           "in this phase. Only a later, explicit phase may append "
                                           "Cologne to training/history, rebuild states, or build a "
                                           "future deployment model.",
            "post_commit": "After the Phase 8E receipt is committed: STOP. Do not automatically start "
                            "post-Cologne model/data updates, an API, or a PWA.",
        },

        # amendment 20: transactional freeze (mirrors Phase 8D exactly)
        "transactional_lifecycle": {
            "staging_directory": "data/evaluation/.phase8e_staging/",
            "commit_marker": "simulation_vs_reality_receipt",
            "note": "All Phase 8E outputs are generated into a staging directory first, validated in "
                    "full, then promoted atomically (os.replace) in a fixed order with the "
                    "simulation-vs-reality receipt written LAST. If reconciliation or replay validation "
                    "fails, no final receipt is produced. If a valid final receipt already exists, the "
                    "pipeline aborts rather than overwriting it.",
        },
    }


def protocol_hash(protocol_dict):
    canonical = yaml.safe_dump(protocol_dict, sort_keys=True, default_flow_style=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def main():
    protocol = build_protocol_dict()
    out_path = ROOT / "config" / "evaluation" / "phase8e_cologne_simulation_vs_reality_protocol.yaml"
    canonical = yaml.safe_dump(protocol, sort_keys=True, default_flow_style=False)
    out_path.write_text(canonical, encoding="utf-8")
    h = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    print(f"wrote {out_path}")
    print(f"protocol_hash = {h}")
    return protocol, h


if __name__ == "__main__":
    main()
