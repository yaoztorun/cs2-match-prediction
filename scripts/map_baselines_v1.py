"""
Phase 6B, brief section 11: TRAIN-only chronological-CV reference baselines for
the KNOWN-MAP task, computed BEFORE any hyperparameter search runs.

    A. 0.5 constant probability
    B. overall-ELO probability, from the pre-series `elo_diff`
    C. map-ELO probability, from the pre-series `map_elo_diff`

B and C invert the project's UNCHANGED ELO expected-score formula
(feature_engine.elo_expected) on features that already exist in
map_features_v2_rich.parquet. Nothing here is fitted or tuned - these are fixed
reference points a tuned map model should ideally beat, and the map model
should ideally add value beyond raw map ELO specifically.

Evaluated on the same four TRAIN-only outer-fold validation blocks the models
are tuned against, so the numbers are directly comparable. The main VALIDATION
partition is never opened by this script.

Writes:
    reports/tables/map_baselines_cv_v1.csv
"""

import numpy as np
import pandas as pd

from _common import REPORTS
from map_modeling_common import (
    N_FOLDS, baseline_probabilities, compute_metrics, fold_frames,
    load_cv_manifest, load_features, load_roles, series_macro_metrics,
    assert_target_and_no_forbidden_columns,
)

TABLES_DIR = REPORTS / "tables"
BASELINES = [("half", "0.5 constant"), ("overall_elo", "overall ELO"), ("map_elo", "map ELO")]


def main():
    roles = load_roles()
    features = load_features()
    assert_target_and_no_forbidden_columns(features, roles)
    cv = load_cv_manifest(verify_against_split=False)

    rows = []
    pooled = {k: {"y": [], "p": [], "mid": []} for k, _ in BASELINES}

    for fold in range(1, N_FOLDS + 1):
        _, raw_val = fold_frames(cv, features, fold)
        y = raw_val[roles["target"]].to_numpy(dtype=float)
        for kind, label in BASELINES:
            p = baseline_probabilities(raw_val, kind)
            m = compute_metrics(y, p)
            rows.append({"baseline": label, "kind": kind, "fold": fold, "row_type": "fold", **m})
            pooled[kind]["y"].append(y)
            pooled[kind]["p"].append(p)
            pooled[kind]["mid"].append(raw_val["match_id"].to_numpy())
        print(f"  fold {fold}: n_val={len(raw_val)}")

    for kind, label in BASELINES:
        y = np.concatenate(pooled[kind]["y"])
        p = np.concatenate(pooled[kind]["p"])
        mid = np.concatenate(pooled[kind]["mid"])
        m = compute_metrics(y, p)
        sm = series_macro_metrics(mid, y, p)
        rows.append({"baseline": label, "kind": kind, "fold": np.nan, "row_type": "pooled_oof", **m, **sm})

        fold_rows = [r for r in rows if r["kind"] == kind and r["row_type"] == "fold"]
        rows.append({
            "baseline": label, "kind": kind, "fold": np.nan, "row_type": "fold_mean",
            "n": int(np.sum([r["n"] for r in fold_rows])),
            "accuracy": float(np.mean([r["accuracy"] for r in fold_rows])),
            "roc_auc": float(np.mean([r["roc_auc"] for r in fold_rows])),
            "log_loss": float(np.mean([r["log_loss"] for r in fold_rows])),
            "brier": float(np.mean([r["brier"] for r in fold_rows])),
        })

    out = pd.DataFrame(rows)
    TABLES_DIR.mkdir(exist_ok=True, parents=True)
    out.to_csv(TABLES_DIR / "map_baselines_cv_v1.csv", index=False, encoding="utf-8")

    print("\nTRAIN-only CV baselines (pooled out-of-fold):")
    pooled_rows = out[out["row_type"] == "pooled_oof"]
    for _, r in pooled_rows.iterrows():
        auc = "n/a (constant)" if not np.isfinite(r["roc_auc"]) else f"{r['roc_auc']:.4f}"
        print(f"  {r['baseline']:<14} n={int(r['n'])} acc={r['accuracy']:.4f} auc={auc} "
              f"logloss={r['log_loss']:.4f} brier={r['brier']:.4f}")
    print(f"\nWrote {TABLES_DIR / 'map_baselines_cv_v1.csv'}")


if __name__ == "__main__":
    main()
