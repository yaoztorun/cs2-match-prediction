"""
Phase 7, Stage C (tables + main report).

READ CONTRACT (brief correction #3, enforced by validation/validate_phase7.py):
  * ALL TEST-derived quantities come EXCLUSIVELY from
    data/evaluation/map_test_predictions_v1.parquet - this script never
    reopens data/features/map_features_v3_modern_map.parquet and never reads
    data/modeling/map_split_v1.csv at all.
  * The ONE exception: already-frozen PRE-Phase-7 machine-readable
    development-metric artifacts (reports/tables/map_xgboost_v3_final_oof_metrics.csv,
    reports/tables/map_model_validation_metrics_v1.csv), read read-only,
    solely to populate the contextual development-vs-TEST table (brief
    section 25) - never recomputed, never used for any TEST-derived number.

Writes:
    reports/tables/map_test_metrics_v1.csv
    reports/tables/map_test_baseline_comparison_v1.csv
    reports/tables/map_test_series_macro_v1.csv
    reports/tables/map_test_per_map_v1.csv
    reports/tables/map_test_bestof_v1.csv
    reports/tables/map_test_tier_v1.csv
    reports/tables/map_test_coverage_v1.csv
    reports/tables/map_test_calibration_bins_v1.csv
    reports/tables/map_development_vs_test_v1.csv
    reports/phase7_internal_test_results.md
"""

import json

import numpy as np
import pandas as pd

from _common import ROOT, REPORTS
from training.map_models.map_modeling_common import compute_metrics, series_macro_metrics

EVAL_DIR = ROOT / "data" / "evaluation"
PRED_PATH = EVAL_DIR / "map_test_predictions_v1.parquet"
TABLES_DIR = REPORTS / "tables"

# Frozen PRE-Phase-7 machine-readable development-metric artifacts (read-only,
# development context ONLY - never a TEST-derived quantity).
PHASE6D_OOF_PATH = TABLES_DIR / "map_xgboost_v3_final_oof_metrics.csv"
PHASE6B_VALIDATION_PATH = TABLES_DIR / "map_model_validation_metrics_v1.csv"

SMALL_SAMPLE_N = 30
CAL_EDGES = [round(i / 10, 1) for i in range(11)]

MODEL_COL = {"final_xgb": "p_xgb_v3_final", "constant_05": "p_constant_05",
             "overall_elo": "p_overall_elo", "map_elo": "p_map_elo"}

COVERAGE_GROUPS = {
    "A_recent_map_evidence": lambda d: d["both_teams_have_recent_selected_map_history"] == 1,
    "B_trusted_adjusted_evidence": lambda d: d["map_adjusted_history_mass_min"] > 0,
    "C_roster_map_evidence_ge3": lambda d: d["roster_map_players_with_history_min"] >= 3,
    "D_strong_roster_map_evidence_ge5": lambda d: d["roster_map_players_with_history_min"] >= 5,
    "E_roster_map_cold_start": lambda d: d["roster_map_players_with_history_min"] == 0,
    "F_team_map_cold_start": lambda d: d["both_teams_have_recent_selected_map_history"] == 0,
    "G_current_core_evidence": lambda d: d["current_core_map_continuity_min"] > 0,
    "H_high_evidence": lambda d: ((d["both_teams_have_recent_selected_map_history"] == 1)
                                   & (d["map_adjusted_history_mass_min"] > 0)
                                   & (d["roster_map_players_with_history_min"] >= 3)),
}


def calibration_bin_index(p):
    """Fixed 10-bin contract (brief correction #4): bin i = [i/10,(i+1)/10),
    EXCEPT the last, which is closed at 1.0 - [0.9, 1.0]. p==1.0 lands in
    bin 9, never dropped, never an 11th bin."""
    return np.minimum((p * 10).astype(int), 9)


def metrics_row(label, y, p, with_confusion=True):
    m = compute_metrics(y, p, with_confusion=with_confusion)
    row = {"model": label, "n": m["n"], "accuracy": m["accuracy"], "roc_auc": m["roc_auc"],
           "log_loss": m["log_loss"], "brier": m["brier"]}
    if with_confusion:
        row.update({"precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
                     "confusion_matrix": json.dumps(m["confusion_matrix"])})
    return row


def main():
    if not PRED_PATH.exists():
        raise RuntimeError(f"{PRED_PATH} does not exist - run evaluation/internal_test/evaluate_phase7_test_once.py first.")
    pred = pd.read_parquet(PRED_PATH, engine="fastparquet")
    print(f"Loaded canonical TEST prediction artifact: {len(pred)} rows (the ONLY TEST-derived source this "
          "script reads).")

    y = pred["y_true"].to_numpy(dtype=float)

    # =====================================================================
    # final model + baseline metrics
    # =====================================================================
    rows = [metrics_row("final_xgb_v3", y, pred["p_xgb_v3_final"].to_numpy(), with_confusion=True)]
    for name, col in [("baseline_constant_05", "p_constant_05"), ("baseline_overall_elo", "p_overall_elo"),
                       ("baseline_map_elo", "p_map_elo")]:
        rows.append(metrics_row(name, y, pred[col].to_numpy(), with_confusion=False))
    metrics_df = pd.DataFrame(rows)
    TABLES_DIR.mkdir(exist_ok=True, parents=True)
    metrics_df.to_csv(TABLES_DIR / "map_test_metrics_v1.csv", index=False, encoding="utf-8")

    baseline_df = metrics_df[metrics_df["model"] != "final_xgb_v3"][["model", "n", "accuracy", "roc_auc",
                                                                       "log_loss", "brier"]]
    baseline_df = pd.concat([baseline_df, metrics_df[metrics_df["model"] == "final_xgb_v3"][
        ["model", "n", "accuracy", "roc_auc", "log_loss", "brier"]]], ignore_index=True)
    baseline_df.to_csv(TABLES_DIR / "map_test_baseline_comparison_v1.csv", index=False, encoding="utf-8")

    # =====================================================================
    # series-macro
    # =====================================================================
    sm_rows = []
    for label, col in [("final_xgb_v3", "p_xgb_v3_final"), ("baseline_constant_05", "p_constant_05"),
                        ("baseline_overall_elo", "p_overall_elo"), ("baseline_map_elo", "p_map_elo")]:
        sm = series_macro_metrics(pred["match_id"].to_numpy(), y, pred[col].to_numpy())
        sm_rows.append({"model": label, **sm})
    sm_df = pd.DataFrame(sm_rows)
    sm_df.to_csv(TABLES_DIR / "map_test_series_macro_v1.csv", index=False, encoding="utf-8")

    # =====================================================================
    # per-map (sorted alphabetically, never by performance)
    # =====================================================================
    per_map_rows = []
    for map_name in sorted(pred["map_name"].unique()):
        mask = (pred["map_name"] == map_name).to_numpy()
        yy, pp = y[mask], pred["p_xgb_v3_final"].to_numpy()[mask]
        both_classes = len(set(yy.tolist())) > 1
        m = compute_metrics(yy, pp)
        per_map_rows.append({
            "map_name": map_name, "n": int(mask.sum()), "small_sample": bool(mask.sum() < SMALL_SAMPLE_N),
            "accuracy": m["accuracy"], "log_loss": m["log_loss"], "brier": m["brier"],
            "roc_auc": m["roc_auc"] if both_classes else float("nan"), "both_classes_present": both_classes,
            "team1_target_rate": float(yy.mean()), "mean_predicted_probability": float(pp.mean()),
        })
    per_map_df = pd.DataFrame(per_map_rows)
    per_map_df.to_csv(TABLES_DIR / "map_test_per_map_v1.csv", index=False, encoding="utf-8")

    # =====================================================================
    # best-of
    # =====================================================================
    bo_rows = []
    for bo in sorted(pred["bestOf"].unique()):
        mask = (pred["bestOf"] == bo).to_numpy()
        yy, pp = y[mask], pred["p_xgb_v3_final"].to_numpy()[mask]
        both_classes = len(set(yy.tolist())) > 1
        m = compute_metrics(yy, pp)
        bo_rows.append({"bestOf": int(bo), "n": int(mask.sum()), "small_sample": bool(mask.sum() < SMALL_SAMPLE_N),
                         "accuracy": m["accuracy"], "roc_auc": m["roc_auc"] if both_classes else float("nan"),
                         "log_loss": m["log_loss"], "brier": m["brier"]})
    bo_df = pd.DataFrame(bo_rows)
    bo_df.to_csv(TABLES_DIR / "map_test_bestof_v1.csv", index=False, encoding="utf-8")

    # =====================================================================
    # tier
    # =====================================================================
    tier_rows = []
    for tier in sorted(pred["tier"].unique()):
        mask = (pred["tier"] == tier).to_numpy()
        yy, pp = y[mask], pred["p_xgb_v3_final"].to_numpy()[mask]
        both_classes = len(set(yy.tolist())) > 1
        m = compute_metrics(yy, pp)
        tier_rows.append({"tier": tier, "n": int(mask.sum()), "small_sample": bool(mask.sum() < SMALL_SAMPLE_N),
                           "accuracy": m["accuracy"], "roc_auc": m["roc_auc"] if both_classes else float("nan"),
                           "log_loss": m["log_loss"], "brier": m["brier"]})
    tier_df = pd.DataFrame(tier_rows)
    tier_df.to_csv(TABLES_DIR / "map_test_tier_v1.csv", index=False, encoding="utf-8")

    # =====================================================================
    # coverage / evidence groups
    # =====================================================================
    cov_rows = []
    p_all = pred["p_xgb_v3_final"].to_numpy()
    for name, fn in COVERAGE_GROUPS.items():
        mask = fn(pred).to_numpy()
        if mask.sum() == 0:
            continue
        yy, pp = y[mask], p_all[mask]
        both_classes = len(set(yy.tolist())) > 1
        m = compute_metrics(yy, pp)
        cov_rows.append({"subgroup": name, "n": int(mask.sum()),
                          "pct_of_test": 100.0 * mask.sum() / len(pred),
                          "accuracy": m["accuracy"], "roc_auc": m["roc_auc"] if both_classes else float("nan"),
                          "log_loss": m["log_loss"], "brier": m["brier"]})
    cov_df = pd.DataFrame(cov_rows)
    cov_df.to_csv(TABLES_DIR / "map_test_coverage_v1.csv", index=False, encoding="utf-8")

    # =====================================================================
    # calibration bins (fixed 10-bin contract)
    # =====================================================================
    bin_idx = calibration_bin_index(p_all)
    cal_rows = []
    for i in range(10):
        mask = bin_idx == i
        if mask.sum() == 0:
            continue
        lo, hi = CAL_EDGES[i], CAL_EDGES[i + 1]
        label = f"[{lo:.1f},{hi:.1f}]" if i == 9 else f"[{lo:.1f},{hi:.1f})"
        cal_rows.append({"bin": label, "bin_index": i, "n": int(mask.sum()),
                          "mean_predicted_probability": float(p_all[mask].mean()),
                          "empirical_team1_win_rate": float(y[mask].mean())})
    cal_df = pd.DataFrame(cal_rows)
    cal_df.to_csv(TABLES_DIR / "map_test_calibration_bins_v1.csv", index=False, encoding="utf-8")

    # =====================================================================
    # side-symmetry (computed here for the report text; figure made downstream)
    # =====================================================================
    sym_err = np.abs(pred["p_xgb_v3_final"].to_numpy() - (1 - pred["p_xgb_v3_mirrored"].to_numpy()))
    symmetry_stats = {"mean": float(sym_err.mean()), "median": float(np.median(sym_err)),
                       "p95": float(np.percentile(sym_err, 95)), "max": float(sym_err.max())}

    # =====================================================================
    # development-vs-TEST (reads ONLY frozen pre-Phase-7 machine-readable artifacts)
    # =====================================================================
    dev_rows = []
    if PHASE6D_OOF_PATH.exists():
        d6d = pd.read_csv(PHASE6D_OOF_PATH)
        r = d6d[d6d["population"] == "pooled_train_oof_development"].iloc[0]
        dev_rows.append({"source": "Phase 6D TRAIN-only selected-model OOF (development)",
                          "model_identity": "map_xgboost_v3_final (the SAME frozen model evaluated on TEST here)",
                          "n": int(r["n"]), "accuracy": r["accuracy"], "roc_auc": r["roc_auc"],
                          "log_loss": r["log_loss"], "brier": r["brier"]})
    if PHASE6B_VALIDATION_PATH.exists():
        d6b = pd.read_csv(PHASE6B_VALIDATION_PATH)
        r = d6b[(d6b["model"] == "xgboost") & (d6b["split"] == "validation")].iloc[0]
        dev_rows.append({"source": "Phase 6B consumed main VALIDATION (EARLIER model/version, V2 features, "
                                    "random_013 @ 124 trees - NOT the Phase 6D final model)",
                          "model_identity": "map_xgboost_v2_random_013 (an earlier, different model)",
                          "n": int(r["n"]), "accuracy": r["accuracy"], "roc_auc": r["roc_auc"],
                          "log_loss": r["log_loss"], "brier": r["brier"]})
    final_m = metrics_df[metrics_df["model"] == "final_xgb_v3"].iloc[0]
    dev_rows.append({"source": "Phase 7 sealed internal TEST (this evaluation)",
                      "model_identity": "map_xgboost_v3_final (the frozen model)",
                      "n": int(final_m["n"]), "accuracy": final_m["accuracy"], "roc_auc": final_m["roc_auc"],
                      "log_loss": final_m["log_loss"], "brier": final_m["brier"]})
    dev_df = pd.DataFrame(dev_rows)
    dev_df.to_csv(TABLES_DIR / "map_development_vs_test_v1.csv", index=False, encoding="utf-8")

    write_report(metrics_df, baseline_df, sm_df, per_map_df, bo_df, tier_df, cov_df, cal_df, dev_df,
                 symmetry_stats, pred)

    print("Wrote reports/tables/map_test_{metrics,baseline_comparison,series_macro,per_map,bestof,tier,"
          "coverage,calibration_bins}_v1.csv")
    print("Wrote reports/tables/map_development_vs_test_v1.csv")
    print("Wrote reports/phase7_internal_test_results.md")


def write_report(metrics_df, baseline_df, sm_df, per_map_df, bo_df, tier_df, cov_df, cal_df, dev_df,
                  symmetry_stats, pred):
    final_m = metrics_df[metrics_df["model"] == "final_xgb_v3"].iloc[0]
    base_overall = metrics_df[metrics_df["model"] == "baseline_overall_elo"].iloc[0]
    base_map = metrics_df[metrics_df["model"] == "baseline_map_elo"].iloc[0]
    base_const = metrics_df[metrics_df["model"] == "baseline_constant_05"].iloc[0]
    sm_final = sm_df[sm_df["model"] == "final_xgb_v3"].iloc[0]

    md = []
    md.append("# Phase 7 - Sealed Internal TEST Results\n")
    md.append("**This is the first and only unbiased internal evaluation of the fully frozen known-map "
              "system.** The model, features, preprocessing, threshold (0.5) and selection procedure were all "
              "frozen before this partition was opened (Phase 6D). No fitting, tuning, calibration, "
              "symmetrization, or ensemble construction occurs in this phase or after it based on these "
              "results.\n")

    md.append("## 1. Final model TEST performance\n")
    md.append(f"n={int(final_m['n'])}, Accuracy={final_m['accuracy']:.4f}, Precision={final_m['precision']:.4f}, "
              f"Recall={final_m['recall']:.4f}, F1={final_m['f1']:.4f}, ROC-AUC={final_m['roc_auc']:.4f}, "
              f"Log Loss={final_m['log_loss']:.4f}, Brier={final_m['brier']:.4f}\n")
    md.append(f"Series-macro: Log Loss={sm_final['series_macro_log_loss']:.4f}, "
              f"Brier={sm_final['series_macro_brier']:.4f}, "
              f"Accuracy={sm_final['series_macro_accuracy']:.4f}\n")
    md.append(f"Side-symmetry (diagnostic only, never corrected): mean={symmetry_stats['mean']:.4f}, "
              f"median={symmetry_stats['median']:.4f}, p95={symmetry_stats['p95']:.4f}, "
              f"max={symmetry_stats['max']:.4f}\n")

    md.append("## 2. Baselines on the same TEST rows\n")
    md.append("| model | n | accuracy | ROC-AUC | log loss | Brier |")
    md.append("|---|---|---|---|---|---|")
    for _, r in baseline_df.iterrows():
        md.append(f"| {r['model']} | {int(r['n'])} | {r['accuracy']:.4f} | {r['roc_auc']:.4f} | "
                  f"{r['log_loss']:.4f} | {r['brier']:.4f} |")
    md.append(f"\nFinal XGB vs overall-ELO: Δlog loss {final_m['log_loss']-base_overall['log_loss']:+.4f}, "
              f"ΔROC-AUC {final_m['roc_auc']-base_overall['roc_auc']:+.4f}. Final XGB vs map-ELO: Δlog loss "
              f"{final_m['log_loss']-base_map['log_loss']:+.4f}, ΔROC-AUC "
              f"{final_m['roc_auc']-base_map['roc_auc']:+.4f}. (Point estimates only - see "
              "`reports/phase7_internal_test_uncertainty.md` for cluster-bootstrap intervals on these deltas.)\n")

    md.append("## 3. Series-macro (all models)\n")
    md.append("| model | n series | series-macro log loss | series-macro Brier | series-macro accuracy |")
    md.append("|---|---|---|---|---|")
    for _, r in sm_df.iterrows():
        md.append(f"| {r['model']} | {int(r['n_series'])} | {r['series_macro_log_loss']:.4f} | "
                  f"{r['series_macro_brier']:.4f} | {r['series_macro_accuracy']:.4f} |")
    md.append("")

    md.append("## 4. Performance by CS2 Map\n")
    md.append("Sorted alphabetically (never by performance) to avoid cherry-picking. `n < 30` is flagged "
              "SMALL SAMPLE - INTERPRET CAUTIOUSLY.\n")
    md.append("| map | n | accuracy | ROC-AUC | log loss | Brier | team1 rate | mean p |")
    md.append("|---|---|---|---|---|---|---|---|")
    for _, r in per_map_df.iterrows():
        flag = " **[SMALL SAMPLE]**" if r["small_sample"] else ""
        auc = "n/a (single class)" if not r["both_classes_present"] else f"{r['roc_auc']:.4f}"
        md.append(f"| {r['map_name']}{flag} | {int(r['n'])} | {r['accuracy']:.4f} | {auc} | "
                  f"{r['log_loss']:.4f} | {r['brier']:.4f} | {r['team1_target_rate']:.3f} | "
                  f"{r['mean_predicted_probability']:.3f} |")
    md.append("")
    best_by_ll = per_map_df.sort_values("log_loss").iloc[0]
    worst_by_ll = per_map_df.sort_values("log_loss").iloc[-1]
    md.append(f"Descriptively (not a ranking claim): `{best_by_ll['map_name']}` has the lowest observed TEST "
              f"log loss ({best_by_ll['log_loss']:.4f}, n={int(best_by_ll['n'])}"
              f"{', SMALL SAMPLE - interpret cautiously' if best_by_ll['small_sample'] else ''}); "
              f"`{worst_by_ll['map_name']}` has the highest ({worst_by_ll['log_loss']:.4f}, "
              f"n={int(worst_by_ll['n'])}{', SMALL SAMPLE - interpret cautiously' if worst_by_ll['small_sample'] else ''}). "
              "Every per-map estimate here should be read together with its own `n` - a striking result on a "
              "small map subsample is far less certain than a similar result on a well-populated one (see the "
              "map-level figures and `reports/phase7_internal_test_uncertainty.md`). No map is removed and no "
              "map-specific model is created based on this table.\n")

    md.append("## 5. Performance by BO1/BO3/BO5\n")
    md.append("| bestOf | n | accuracy | ROC-AUC | log loss | Brier |")
    md.append("|---|---|---|---|---|---|")
    for _, r in bo_df.iterrows():
        flag = " [SMALL SAMPLE]" if r["small_sample"] else ""
        auc = "n/a" if pd.isna(r["roc_auc"]) else f"{r['roc_auc']:.4f}"
        md.append(f"| BO{int(r['bestOf'])}{flag} | {int(r['n'])} | {r['accuracy']:.4f} | {auc} | "
                  f"{r['log_loss']:.4f} | {r['brier']:.4f} |")
    md.append("")

    md.append("## 6. Performance by tier\n")
    md.append("| tier | n | accuracy | ROC-AUC | log loss | Brier |")
    md.append("|---|---|---|---|---|---|")
    for _, r in tier_df.iterrows():
        flag = " [SMALL SAMPLE]" if r["small_sample"] else ""
        auc = "n/a" if pd.isna(r["roc_auc"]) else f"{r['roc_auc']:.4f}"
        md.append(f"| {r['tier']}{flag} | {int(r['n'])} | {r['accuracy']:.4f} | {auc} | "
                  f"{r['log_loss']:.4f} | {r['brier']:.4f} |")
    md.append("")

    md.append("## 7. Coverage / evidence diagnostics (descriptive only)\n")
    md.append("| subgroup | n | % of TEST | accuracy | ROC-AUC | log loss | Brier |")
    md.append("|---|---|---|---|---|---|---|")
    for _, r in cov_df.iterrows():
        auc = "n/a" if pd.isna(r["roc_auc"]) else f"{r['roc_auc']:.4f}"
        md.append(f"| {r['subgroup']} | {int(r['n'])} | {r['pct_of_test']:.1f}% | {r['accuracy']:.4f} | {auc} | "
                  f"{r['log_loss']:.4f} | {r['brier']:.4f} |")
    md.append("\nNo subgroup-specific model is trained; these are descriptive slices of the single frozen "
              "model's predictions.\n")

    md.append("## 8. Calibration (fixed 10-bin reliability, raw probabilities, no calibration fitted)\n")
    md.append("| bin | n | mean predicted | empirical Team1 win rate |")
    md.append("|---|---|---|---|")
    for _, r in cal_df.iterrows():
        md.append(f"| {r['bin']} | {int(r['n'])} | {r['mean_predicted_probability']:.3f} | "
                  f"{r['empirical_team1_win_rate']:.3f} |")
    md.append("")

    md.append("## 9. Development vs TEST (context only)\n")
    md.append("| source | model identity | n | accuracy | ROC-AUC | log loss | Brier |")
    md.append("|---|---|---|---|---|---|---|")
    for _, r in dev_df.iterrows():
        md.append(f"| {r['source']} | {r['model_identity']} | {int(r['n'])} | {r['accuracy']:.4f} | "
                  f"{r['roc_auc']:.4f} | {r['log_loss']:.4f} | {r['brier']:.4f} |")
    dev_row = dev_df.iloc[0]
    md.append(f"\n**Direct generalization comparison** (same model, same features): Phase 6D TRAIN-only OOF "
              f"development -> Phase 7 sealed TEST: Δaccuracy {final_m['accuracy']-dev_row['accuracy']:+.4f}, "
              f"ΔROC-AUC {final_m['roc_auc']-dev_row['roc_auc']:+.4f}, "
              f"Δlog loss {final_m['log_loss']-dev_row['log_loss']:+.4f}, "
              f"ΔBrier {final_m['brier']-dev_row['brier']:+.4f}. The Phase 6B validation row above used an "
              "EARLIER model/version (V2 features, a different XGB configuration) and is shown only as broader "
              "historical context, never as an evaluation of the exact Phase 6D final model.\n")

    md.append("## Interpretation\n")
    md.append("1. **Generalization**: see the direct comparison above and the cluster-bootstrap intervals in "
              "`reports/phase7_internal_test_uncertainty.md` for whether the TEST result is inside, above, or "
              "below the development-era range once sampling uncertainty is accounted for.\n")
    md.append(f"2. **Vs. baselines**: final XGB log loss {final_m['log_loss']:.4f} against constant-0.5 "
              f"{base_const['log_loss']:.4f}, overall-ELO {base_overall['log_loss']:.4f}, map-ELO "
              f"{base_map['log_loss']:.4f}; ROC-AUC {final_m['roc_auc']:.4f} against "
              f"{base_overall['roc_auc']:.4f} / {base_map['roc_auc']:.4f}.\n")
    md.append("3. **Uncertainty**: see `reports/phase7_internal_test_uncertainty.md` for 2,000-replicate "
              "series-cluster bootstrap intervals.\n")
    md.append(f"4. **Probability quality**: Brier {final_m['brier']:.4f} vs constant-0.5's "
              f"{base_const['brier']:.4f}; see the calibration table above and "
              "`reports/figures/map_xgb_v3_test_calibration.png`.\n")
    md.append("5. **Consistency across maps/formats/coverage**: see sections 4-7 above and the map-level "
              "figures - read every subgroup number together with its own `n`.\n")
    md.append("6. **Cold-start behavior**: compare coverage subgroups E/F (cold start) against A/B/C/D/H "
              "(evidenced) in section 7.\n")
    md.append("7. **Side-orientation stability**: see the side-symmetry statistics in section 1 - diagnostic "
              "only, no prediction is symmetrized.\n")
    md.append("\nNo model change is proposed in this section. Any improvement discussion belongs in "
              "limitations/future-work only, and does not alter the frozen system evaluated here.\n")

    md.append("## Status\n")
    md.append("- **FINAL MODEL = FROZEN BEFORE TEST**\n- **TEST = OPENED FOR FINAL INTERNAL EVALUATION**\n"
              "- **TEST = NOT USED FOR MODEL DEVELOPMENT**\n- **NO POST-TEST RETUNING**\n- **THRESHOLD = 0.5**\n"
              "- **NO CALIBRATION**\n- **NO NEW ENSEMBLE**\n- **COLOGNE = UNTOUCHED**\n- **SRC = UNCHANGED**\n")

    (REPORTS / "phases" / "phase7_internal_test_results.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
