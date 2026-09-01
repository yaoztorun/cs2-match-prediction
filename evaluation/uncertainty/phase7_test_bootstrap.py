"""
Phase 7, Stage C (cluster-bootstrap uncertainty).

READ CONTRACT: this script reads ONLY
data/evaluation/map_test_predictions_v1.parquet - never
data/features/map_features_v3_modern_map.parquet, never
data/modeling/map_split_v1.csv.

Predefined contract (frozen in config/phase7_test_evaluation_protocol.yaml
BEFORE TEST was opened): match_id-cluster bootstrap, N_BOOTSTRAP=2000,
RANDOM_STATE=42, 95% PERCENTILE intervals. Each replicate resamples the
UNIQUE match_ids with replacement (never individual map rows independently);
a resampled match_id contributes every one of its map rows. The SAME
per-replicate match_id draw is reused to score the final model AND both
baselines, so the paired deltas (final XGB - overall ELO, final XGB - map
ELO) are computed on identical resampled data each time - true pairing, not
independent resampling. AUC is skipped (and separately counted) on any
replicate whose y_true slice is single-class.

Writes:
    reports/tables/map_test_bootstrap_ci_v1.csv
    reports/tables/map_test_paired_bootstrap_deltas_v1.csv
    reports/phase7_internal_test_uncertainty.md
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, brier_score_loss

from _common import ROOT, REPORTS

EVAL_DIR = ROOT / "data" / "evaluation"
PRED_PATH = EVAL_DIR / "map_test_predictions_v1.parquet"
TABLES_DIR = REPORTS / "tables"

N_BOOTSTRAP = 2000
RANDOM_STATE = 42
CI = 0.95

METRIC_DIRECTION = {"accuracy": "higher_is_better", "roc_auc": "higher_is_better",
                     "log_loss": "lower_is_better", "brier": "lower_is_better"}


def per_replicate_metrics(y, p):
    """y/p already expanded to the resampled row set for one replicate."""
    pred_label = (p >= 0.5).astype(int)
    out = {"accuracy": accuracy_score(y, pred_label),
           "log_loss": log_loss(y, p, labels=[0, 1]),
           "brier": brier_score_loss(y, p)}
    if len(set(y.tolist())) > 1:
        out["roc_auc"] = roc_auc_score(y, p)
    else:
        out["roc_auc"] = None
    return out


def build_match_row_index(match_ids):
    idx = {}
    for i, m in enumerate(match_ids):
        idx.setdefault(m, []).append(i)
    return idx


def run_cluster_bootstrap(pred_df, prob_cols, n_bootstrap=N_BOOTSTRAP, random_state=RANDOM_STATE):
    """Returns {col: DataFrame(replicate metrics)} plus the shared per-replicate
    row-index arrays (so callers can verify pairing)."""
    y_all = pred_df["y_true"].to_numpy(dtype=float)
    match_ids = pred_df["match_id"].to_numpy()
    unique_matches = np.array(sorted(set(match_ids.tolist())))
    match_to_rows = build_match_row_index(match_ids)

    rng = np.random.RandomState(random_state)
    replicate_row_indices = []
    for _ in range(n_bootstrap):
        sampled_matches = rng.choice(unique_matches, size=len(unique_matches), replace=True)
        rows = []
        for m in sampled_matches:
            rows.extend(match_to_rows[m])
        replicate_row_indices.append(np.array(rows, dtype=int))

    results = {col: [] for col in prob_cols}
    for rep_idx in replicate_row_indices:
        y_rep = y_all[rep_idx]
        for col in prob_cols:
            p_rep = pred_df[col].to_numpy()[rep_idx]
            results[col].append(per_replicate_metrics(y_rep, p_rep))

    return {col: pd.DataFrame(v) for col, v in results.items()}, replicate_row_indices


def percentile_ci(values, ci=CI):
    values = [v for v in values if v is not None and np.isfinite(v)]
    if not values:
        return None, None
    alpha = (1 - ci) / 2
    lo = float(np.percentile(values, 100 * alpha))
    hi = float(np.percentile(values, 100 * (1 - alpha)))
    return lo, hi


def main():
    if not PRED_PATH.exists():
        raise RuntimeError(f"{PRED_PATH} does not exist - run evaluation/internal_test/evaluate_phase7_test_once.py first.")
    pred = pd.read_parquet(PRED_PATH, engine="fastparquet")
    print(f"Loaded canonical TEST prediction artifact: {len(pred)} rows, "
          f"{pred['match_id'].nunique()} unique match_ids (the ONLY source this script reads).")

    prob_cols = ["p_xgb_v3_final", "p_overall_elo", "p_map_elo"]
    replicate_metrics, replicate_row_indices = run_cluster_bootstrap(pred, prob_cols)

    # ---------------- 95% CI table (final model only, per brief section 14) ----------------
    ci_rows = []
    final_df = replicate_metrics["p_xgb_v3_final"]
    point = per_replicate_metrics(pred["y_true"].to_numpy(dtype=float), pred["p_xgb_v3_final"].to_numpy())
    n_valid_auc = int(final_df["roc_auc"].notna().sum())
    for metric in ["accuracy", "roc_auc", "log_loss", "brier"]:
        lo, hi = percentile_ci(final_df[metric].tolist())
        ci_rows.append({"model": "final_xgb_v3", "metric": metric, "point_estimate": point[metric],
                         "ci_lower_95": lo, "ci_upper_95": hi,
                         "n_bootstrap": N_BOOTSTRAP,
                         "n_valid_replicates": n_valid_auc if metric == "roc_auc" else N_BOOTSTRAP})
    ci_df = pd.DataFrame(ci_rows)
    TABLES_DIR.mkdir(exist_ok=True, parents=True)
    ci_df.to_csv(TABLES_DIR / "map_test_bootstrap_ci_v1.csv", index=False, encoding="utf-8")
    print(f"AUC valid replicates: {n_valid_auc}/{N_BOOTSTRAP} (single-class replicates skipped and counted)")

    # ---------------- paired bootstrap deltas (SAME draws, both arms, brief section 15) ----------------
    y_all = pred["y_true"].to_numpy(dtype=float)
    paired_rows = []
    for baseline_name, baseline_col in [("overall_elo", "p_overall_elo"), ("map_elo", "p_map_elo")]:
        deltas = {"accuracy": [], "roc_auc": [], "log_loss": [], "brier": []}
        for rep_idx in replicate_row_indices:
            y_rep = y_all[rep_idx]
            m_final = per_replicate_metrics(y_rep, pred["p_xgb_v3_final"].to_numpy()[rep_idx])
            m_base = per_replicate_metrics(y_rep, pred[baseline_col].to_numpy()[rep_idx])
            for metric in ["accuracy", "log_loss", "brier"]:
                deltas[metric].append(m_final[metric] - m_base[metric])
            if m_final["roc_auc"] is not None and m_base["roc_auc"] is not None:
                deltas["roc_auc"].append(m_final["roc_auc"] - m_base["roc_auc"])

        obs_final = per_replicate_metrics(y_all, pred["p_xgb_v3_final"].to_numpy())
        obs_base = per_replicate_metrics(y_all, pred[baseline_col].to_numpy())
        for metric in ["accuracy", "roc_auc", "log_loss", "brier"]:
            lo, hi = percentile_ci(deltas[metric])
            paired_rows.append({
                "comparison": f"final_xgb_v3_minus_{baseline_name}", "metric": metric,
                "observed_delta": obs_final[metric] - obs_base[metric],
                "ci_lower_95": lo, "ci_upper_95": hi,
                "favors_final_xgb": (
                    (obs_final[metric] - obs_base[metric] > 0) if METRIC_DIRECTION[metric] == "higher_is_better"
                    else (obs_final[metric] - obs_base[metric] < 0)),
                "n_bootstrap": N_BOOTSTRAP,
                "n_valid_replicates": len(deltas[metric]),
            })
    paired_df = pd.DataFrame(paired_rows)
    paired_df.to_csv(TABLES_DIR / "map_test_paired_bootstrap_deltas_v1.csv", index=False, encoding="utf-8")

    write_report(ci_df, paired_df, n_valid_auc)
    print("Wrote reports/tables/map_test_bootstrap_ci_v1.csv")
    print("Wrote reports/tables/map_test_paired_bootstrap_deltas_v1.csv")
    print("Wrote reports/phase7_internal_test_uncertainty.md")


def write_report(ci_df, paired_df, n_valid_auc):
    md = []
    md.append("# Phase 7 - Sealed Internal TEST Uncertainty (Cluster Bootstrap)\n")
    md.append(f"Predefined contract, frozen before TEST was opened: match_id-cluster bootstrap, "
              f"N_BOOTSTRAP={N_BOOTSTRAP}, RANDOM_STATE={RANDOM_STATE}, 95% percentile intervals. Reads ONLY "
              "`data/evaluation/map_test_predictions_v1.parquet`.\n")

    md.append("## Final model: 95% cluster-bootstrap intervals\n")
    md.append("| metric | point estimate | 95% CI lower | 95% CI upper | valid replicates |")
    md.append("|---|---|---|---|---|")
    for _, r in ci_df.iterrows():
        md.append(f"| {r['metric']} | {r['point_estimate']:.4f} | {r['ci_lower_95']:.4f} | "
                  f"{r['ci_upper_95']:.4f} | {int(r['n_valid_replicates'])}/{N_BOOTSTRAP} |")
    md.append(f"\nAUC: {n_valid_auc}/{N_BOOTSTRAP} replicates were valid (single-class replicates skipped and "
              "counted separately, per the predefined contract).\n")

    md.append("## Paired bootstrap deltas vs. baselines (identical per-replicate draws)\n")
    md.append("Positive favors the final XGB for Accuracy/ROC-AUC; negative favors it for Log Loss/Brier.\n")
    md.append("| comparison | metric | observed Δ | 95% CI lower | 95% CI upper | favors final XGB |")
    md.append("|---|---|---|---|---|---|")
    for _, r in paired_df.iterrows():
        md.append(f"| {r['comparison']} | {r['metric']} | {r['observed_delta']:+.4f} | "
                  f"{r['ci_lower_95']:+.4f} | {r['ci_upper_95']:+.4f} | {r['favors_final_xgb']} |")
    md.append("\nThis is inferential/descriptive evaluation of the already-frozen model - it is not a "
              "model-selection mechanism, and no configuration changes follow from it.\n")

    (REPORTS / "phases" / "phase7_internal_test_uncertainty.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
