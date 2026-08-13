"""
Phase 6B: Random Forest hyperparameter search for the KNOWN-MAP task, via the
frozen four TRAIN-only expanding-window chronological folds
(data/modeling/map_cv_folds_v1.csv, reused byte-identically from Phase 6A).

This script NEVER opens data/modeling/map_split_v1.csv, so the 1,129-map main
VALIDATION partition and the 1,427-map TEST partition are structurally absent
from selection - not merely avoided by convention. Cologne is absent from the
feature artifact entirely.

SELECTION RULE - fixed here BEFORE the search runs, never altered afterwards:
    PRIMARY   lowest mean chronological-CV log loss
    EQUIVALENCE  within LOG_LOSS_EQUIVALENCE_EPSILON (0.002) counts as tied
    then      highest mean ROC-AUC
    then      lowest mean Brier
    then      highest mean accuracy
    then      lower log-loss standard deviation
    then      simpler model: lower max_depth -> higher min_samples_leaf ->
              higher min_samples_split -> lower effective max_features
              fraction -> candidate_id (deterministic final tie)
Accuracy is never optimized alone - the application consumes probabilities.
Fit time is recorded for operational reporting only and never participates in
selection.

n_estimators is FIXED at 400 for every candidate: the search budget belongs to
the structural parameters, and 400 trees is enough for stable candidate
comparison.

Writes:
    reports/tables/map_random_forest_tuning_v1.csv
    reports/phase6b_map_random_forest_tuning.md
    data/modeling/map_random_forest_v1_selected_config.json
    data/modeling/map_rf_search_plan_v1.json          (deterministic, no timestamps)
    data/modeling/map_rf_tuning_progress_v1.jsonl     (checkpoint/resume)
"""

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import REPORTS                                          # noqa: E402
from map_modeling_common import (                                    # noqa: E402
    FoldCache, LOG_LOSS_EQUIVALENCE_EPSILON, MODELING_DIR, N_FOLDS, RANDOM_STATE,
    append_checkpoint, assert_target_and_no_forbidden_columns, build_search_plan,
    compute_metrics, load_checkpoint, load_cv_manifest, load_features, load_roles,
    reset_checkpoint_if_stale,
)
from preprocessing_common_map_v2 import EXPECTED_TRANSFORMED_FEATURES  # noqa: E402

TABLES_DIR = REPORTS / "tables"
PLAN_PATH = MODELING_DIR / "map_rf_search_plan_v1.json"
PROGRESS_PATH = MODELING_DIR / "map_rf_tuning_progress_v1.jsonl"
SELECTED_PATH = MODELING_DIR / "map_random_forest_v1_selected_config.json"

N_ESTIMATORS = 400
N_RANDOM_CANDIDATES = 28
FIXED = {"n_estimators": N_ESTIMATORS, "bootstrap": True}

SEARCH_SPACE = {
    "max_depth": [5, 7, 8, 10, 12, None],
    "min_samples_leaf": [5, 10, 20, 40, 80],
    "min_samples_split": [10, 20, 40],
    "max_features": ["sqrt", 0.25, 0.40, 0.60],
    "criterion": ["gini", "log_loss"],
}
SEARCH_KEYS = list(SEARCH_SPACE.keys())

# Deterministic anchors, defined before running anything and never adjusted
# after seeing a result. The previous SERIES RF V2 structure is included as ONE
# reference/anchor candidate - it is eligible for selection, but is not adopted
# automatically: the known-map task has a different target and a richer feature
# space, so its best structure is an open question.
ANCHOR_CONFIGS = [
    {"candidate_id": "anchor_series_rf_v2_structure",
     "max_depth": 8, "min_samples_leaf": 20, "min_samples_split": 10,
     "max_features": "sqrt", "criterion": "gini"},
    {"candidate_id": "anchor_shallow_high_regularization",
     "max_depth": 5, "min_samples_leaf": 80, "min_samples_split": 40,
     "max_features": "sqrt", "criterion": "gini"},
    {"candidate_id": "anchor_moderate_depth8_leaf10",
     "max_depth": 8, "min_samples_leaf": 10, "min_samples_split": 20,
     "max_features": "sqrt", "criterion": "gini"},
    {"candidate_id": "anchor_moderate_depth10_leaf20",
     "max_depth": 10, "min_samples_leaf": 20, "min_samples_split": 20,
     "max_features": "sqrt", "criterion": "gini"},
    {"candidate_id": "anchor_leaf40_conservative",
     "max_depth": None, "min_samples_leaf": 40, "min_samples_split": 40,
     "max_features": "sqrt", "criterion": "gini"},
    {"candidate_id": "anchor_deeper_but_regularized",
     "max_depth": 12, "min_samples_leaf": 40, "min_samples_split": 40,
     "max_features": "sqrt", "criterion": "gini"},
    {"candidate_id": "anchor_fractional_max_features",
     "max_depth": 8, "min_samples_leaf": 20, "min_samples_split": 10,
     "max_features": 0.40, "criterion": "gini"},
    {"candidate_id": "anchor_log_loss_criterion",
     "max_depth": 8, "min_samples_leaf": 20, "min_samples_split": 10,
     "max_features": "sqrt", "criterion": "log_loss"},
]

SELECTION_RULE_TEXT = (
    "1) PRIMARY: lowest mean chronological-CV log loss. "
    f"2) EQUIVALENCE: candidates within {LOG_LOSS_EQUIVALENCE_EPSILON} of the best mean log loss are treated "
    "as essentially equivalent. 3) highest mean ROC-AUC. 4) lowest mean Brier. 5) highest mean accuracy. "
    "6) lower log-loss standard deviation. 7) COMPLEXITY (deterministic): lower max_depth -> higher "
    "min_samples_leaf -> higher min_samples_split -> lower effective max_features fraction -> candidate_id. "
    "Accuracy is never optimized alone; fit time never participates in selection."
)


def build_candidates():
    """Deterministic: 8 fixed anchors + 28 RandomState(42) draws from the grid,
    no duplicates. Built and persisted BEFORE any metric is computed."""
    grid = list(itertools.product(*[SEARCH_SPACE[k] for k in SEARCH_KEYS]))
    anchor_tuples = {tuple(a[k] for k in SEARCH_KEYS) for a in ANCHOR_CONFIGS}
    eligible = [t for t in grid if t not in anchor_tuples]

    rng = np.random.RandomState(RANDOM_STATE)
    idx = sorted(rng.choice(len(eligible), size=N_RANDOM_CANDIDATES, replace=False).tolist())

    candidates = [dict(a, **FIXED) for a in ANCHOR_CONFIGS]
    for n, i in enumerate(idx, start=1):
        params = dict(zip(SEARCH_KEYS, eligible[i]))
        params["candidate_id"] = f"random_{n:03d}"
        candidates.append(dict(params, **FIXED))

    seen = set()
    for c in candidates:
        key = tuple(c[k] for k in SEARCH_KEYS)
        assert key not in seen, f"duplicate candidate hyperparameters: {c['candidate_id']}"
        seen.add(key)
    ids = [c["candidate_id"] for c in candidates]
    assert len(ids) == len(set(ids)), "duplicate candidate_id"
    return candidates


def max_features_fraction(mf):
    if mf == "sqrt":
        return float(EXPECTED_TRANSFORMED_FEATURES ** 0.5 / EXPECTED_TRANSFORMED_FEATURES)
    return float(mf)


def complexity_key(params):
    """LOWER is simpler / more regularized."""
    depth = 999 if params["max_depth"] is None else params["max_depth"]
    return (depth, -params["min_samples_leaf"], -params["min_samples_split"],
            max_features_fraction(params["max_features"]), params["candidate_id"])


def evaluate_on_fold(params, cache, fold):
    """Fit on the MIRRORED/AUGMENTED fold-train; report TRAIN metrics on the
    ORIGINAL unmirrored fold-train and validation metrics on the ORIGINAL,
    never-mirrored fold-validation block."""
    entry = cache[fold]
    kwargs = {k: v for k, v in params.items() if k != "candidate_id"}
    model = RandomForestClassifier(**kwargs, random_state=RANDOM_STATE, n_jobs=-1)

    t0 = time.perf_counter()
    model.fit(entry["rf_X_aug"], entry["y_aug"])
    fit_seconds = time.perf_counter() - t0

    p_tr = model.predict_proba(entry["rf_X_train_orig"])[:, 1]
    p_va = model.predict_proba(entry["rf_X_val"])[:, 1]
    train_m = compute_metrics(entry["y_train_orig"], p_tr)
    val_m = compute_metrics(entry["y_val"], p_va)
    return train_m, val_m, fit_seconds


def aggregate(fold_df, params_by_id):
    records = []
    for cid, g in fold_df.groupby("candidate_id"):
        p = params_by_id[cid]
        records.append({
            "candidate_id": cid, "fold": np.nan, "row_type": "aggregate",
            **{k: p[k] for k in SEARCH_KEYS}, "n_estimators": p["n_estimators"], "bootstrap": p["bootstrap"],
            "val_log_loss_mean": g["val_log_loss"].mean(), "val_log_loss_std": g["val_log_loss"].std(ddof=0),
            "val_roc_auc_mean": g["val_roc_auc"].mean(), "val_roc_auc_std": g["val_roc_auc"].std(ddof=0),
            "val_brier_mean": g["val_brier"].mean(), "val_brier_std": g["val_brier"].std(ddof=0),
            "val_accuracy_mean": g["val_accuracy"].mean(), "val_accuracy_std": g["val_accuracy"].std(ddof=0),
            "val_precision_mean": g["val_precision"].mean(), "val_recall_mean": g["val_recall"].mean(),
            "val_f1_mean": g["val_f1"].mean(),
            "train_log_loss_mean": g["train_log_loss"].mean(), "train_roc_auc_mean": g["train_roc_auc"].mean(),
            "train_accuracy_mean": g["train_accuracy"].mean(),
            "train_val_auc_gap_mean": g["train_val_auc_gap"].mean(),
            "train_val_log_loss_gap_mean": g["train_val_log_loss_gap"].mean(),
            "mean_fit_seconds": g["fit_seconds"].mean(),
        })
    return pd.DataFrame(records).sort_values("candidate_id").reset_index(drop=True)


def select_winner(agg, params_by_id):
    best = agg["val_log_loss_mean"].min()
    tied = agg[agg["val_log_loss_mean"] <= best + LOG_LOSS_EQUIVALENCE_EPSILON].copy()
    stage = "primary (lowest mean CV log loss, unique)"
    if len(tied) > 1:
        stage = "secondary (log-loss tie within epsilon, resolved by highest mean CV ROC-AUC)"
        tied = tied[tied["val_roc_auc_mean"] == tied["val_roc_auc_mean"].max()]
    if len(tied) > 1:
        stage = "tertiary (resolved by lowest mean CV Brier)"
        tied = tied[tied["val_brier_mean"] == tied["val_brier_mean"].min()]
    if len(tied) > 1:
        stage = "quaternary (resolved by highest mean CV accuracy)"
        tied = tied[tied["val_accuracy_mean"] == tied["val_accuracy_mean"].max()]
    if len(tied) > 1:
        stage = "quinary (resolved by lower CV log-loss standard deviation)"
        tied = tied[tied["val_log_loss_std"] == tied["val_log_loss_std"].min()]
    if len(tied) > 1:
        stage = "complexity (all metrics tied, resolved by deterministic complexity ordering)"
        tied = tied.assign(_ck=tied["candidate_id"].map(lambda c: complexity_key(params_by_id[c]))) \
                    .sort_values("_ck")
    return tied.iloc[0]["candidate_id"], stage


def main():
    roles = load_roles()
    features = load_features()
    assert_target_and_no_forbidden_columns(features, roles)
    # verify_against_split=False: this script must never open map_split_v1.csv.
    cv = load_cv_manifest(verify_against_split=False)

    candidates = build_candidates()
    params_by_id = {c["candidate_id"]: c for c in candidates}
    plan = build_search_plan("map_random_forest_v1", candidates, {
        "n_estimators_fixed": N_ESTIMATORS,
        "search_space": {k: [str(v) for v in vs] for k, vs in SEARCH_SPACE.items()},
        "selection_rule": SELECTION_RULE_TEXT,
        "cv_manifest": "data/modeling/map_cv_folds_v1.csv",
        "main_validation_used_in_selection": False,
    })
    MODELING_DIR.mkdir(exist_ok=True, parents=True)
    PLAN_PATH.write_text(json.dumps(plan, indent=2, default=str), encoding="utf-8")

    if reset_checkpoint_if_stale(PROGRESS_PATH, plan["plan_hash"]):
        print("Search plan changed since the last run - previous progress discarded (never mixed).")
    done = load_checkpoint(PROGRESS_PATH, plan["plan_hash"])
    print(f"Plan hash {plan['plan_hash'][:12]}... | {len(candidates)} candidates x {N_FOLDS} folds = "
          f"{len(candidates) * N_FOLDS} fits | {len(done)} already completed, resuming the rest")

    print("Building per-fold preprocessing cache once (reused by every candidate)...")
    cache = FoldCache(cv, features, roles, build_rf=True, build_xgb=False)
    for f in range(1, N_FOLDS + 1):
        e = cache[f]
        print(f"  fold {f}: {e['n_train_unique']} unique train maps -> {e['n_train_augmented']} augmented "
              f"observations | {e['n_val']} validation maps | X {e['rf_X_aug'].shape}")
    assert cache[1]["rf_X_aug"].shape[1] == EXPECTED_TRANSFORMED_FEATURES

    rows = []
    for ci, params in enumerate(candidates, start=1):
        for fold in range(1, N_FOLDS + 1):
            key = (params["candidate_id"], fold)
            if key in done:
                rows.append(done[key]["row"])
                continue
            print(f"RF candidate {ci}/{len(candidates)} ({params['candidate_id']}) - fold {fold}/{N_FOLDS}")
            train_m, val_m, fit_seconds = evaluate_on_fold(params, cache, fold)
            row = {
                "candidate_id": params["candidate_id"], "fold": fold, "row_type": "fold",
                **{k: params[k] for k in SEARCH_KEYS},
                "n_estimators": params["n_estimators"], "bootstrap": params["bootstrap"],
                "n_train_unique": cache[fold]["n_train_unique"],
                "n_train_augmented": cache[fold]["n_train_augmented"], "n_val": cache[fold]["n_val"],
                "train_accuracy": train_m["accuracy"], "train_roc_auc": train_m["roc_auc"],
                "train_log_loss": train_m["log_loss"], "train_brier": train_m["brier"],
                "val_accuracy": val_m["accuracy"], "val_precision": val_m["precision"],
                "val_recall": val_m["recall"], "val_f1": val_m["f1"], "val_roc_auc": val_m["roc_auc"],
                "val_log_loss": val_m["log_loss"], "val_brier": val_m["brier"],
                "train_val_auc_gap": train_m["roc_auc"] - val_m["roc_auc"],
                "train_val_log_loss_gap": val_m["log_loss"] - train_m["log_loss"],
                "fit_seconds": fit_seconds,
            }
            rows.append(row)
            append_checkpoint(PROGRESS_PATH, {"plan_hash": plan["plan_hash"],
                                               "candidate_id": params["candidate_id"], "fold": fold, "row": row})

    fold_df = pd.DataFrame(rows)
    agg = aggregate(fold_df, params_by_id)

    TABLES_DIR.mkdir(exist_ok=True, parents=True)
    pd.concat([fold_df, agg], ignore_index=True, sort=False).to_csv(
        TABLES_DIR / "map_random_forest_tuning_v1.csv", index=False, encoding="utf-8")

    winner_id, stage = select_winner(agg, params_by_id)
    w = params_by_id[winner_id]
    wagg = agg[agg["candidate_id"] == winner_id].iloc[0]
    print(f"\nSELECTED: {winner_id} via {stage}")
    print("  params: " + str({k: v for k, v in w.items() if k != "candidate_id"}))
    print(f"  mean CV log loss {wagg['val_log_loss_mean']:.4f} +/- {wagg['val_log_loss_std']:.4f} | "
          f"mean CV ROC-AUC {wagg['val_roc_auc_mean']:.4f}")

    selected = {
        "selected_candidate_id": winner_id,
        "params": {k: w[k] for k in SEARCH_KEYS + ["n_estimators", "bootstrap"]},
        "random_state": RANDOM_STATE,
        "selection_stage": stage,
        "selection_rule": SELECTION_RULE_TEXT,
        "log_loss_epsilon": LOG_LOSS_EQUIVALENCE_EPSILON,
        "cv_mean_log_loss": float(wagg["val_log_loss_mean"]),
        "cv_std_log_loss": float(wagg["val_log_loss_std"]),
        "cv_mean_roc_auc": float(wagg["val_roc_auc_mean"]),
        "cv_std_roc_auc": float(wagg["val_roc_auc_std"]),
        "cv_mean_brier": float(wagg["val_brier_mean"]),
        "cv_mean_accuracy": float(wagg["val_accuracy_mean"]),
        "cv_mean_train_val_auc_gap": float(wagg["train_val_auc_gap_mean"]),
        "n_candidates": len(candidates),
        "n_anchor_candidates": len(ANCHOR_CONFIGS),
        "n_random_candidates": len(candidates) - len(ANCHOR_CONFIGS),
        "transformed_feature_count": EXPECTED_TRANSFORMED_FEATURES,
        "cv_folds_artifact": "data/modeling/map_cv_folds_v1.csv",
        "tuning_results_artifact": "reports/tables/map_random_forest_tuning_v1.csv",
        "search_plan_artifact": "data/modeling/map_rf_search_plan_v1.json",
        "search_plan_hash": plan["plan_hash"],
        "main_validation_used_in_selection": False,
    }
    SELECTED_PATH.write_text(json.dumps(selected, indent=2, default=str), encoding="utf-8")

    write_report(agg, winner_id, stage, params_by_id, wagg)
    print("\nWrote reports/tables/map_random_forest_tuning_v1.csv")
    print("Wrote data/modeling/map_random_forest_v1_selected_config.json")
    print("Wrote reports/phase6b_map_random_forest_tuning.md")


def write_report(agg, winner_id, stage, params_by_id, wagg):
    md = []
    md.append("# Phase 6B - Known-Map Random Forest Tuning (TRAIN-only chronological CV)\n")
    md.append(f"Generated by `scripts/models/map_random_forest_tuning_v1.py`. **{len(agg)} candidates** "
              f"({len(ANCHOR_CONFIGS)} deterministic anchors + {len(agg) - len(ANCHOR_CONFIGS)} "
              f"`RandomState({RANDOM_STATE})` draws) x {N_FOLDS} expanding-window folds from "
              "`data/modeling/map_cv_folds_v1.csv`.\n")
    md.append("**Target: `team1_map_win` - the winner of one specific, user-selected map.** This is not the "
              "pre-veto series task, and these numbers are not comparable with the series models' accuracies.\n")

    md.append("## Why the main validation partition cannot have influenced this\n")
    md.append("This script never opens `data/modeling/map_split_v1.csv`. It reads only the Phase 6A CV manifest, "
              "which by construction contains **TRAIN match_ids only** (independently re-verified in "
              "`scripts/validate_phase6b.py`). The 1,129-map main validation partition and the 1,427-map TEST "
              "partition are therefore structurally absent from candidate selection.\n")

    md.append("## Fixed protocol\n")
    md.append(f"- `n_estimators = {N_ESTIMATORS}` for every candidate - the search budget goes to the structural "
              "parameters, not to tree count.")
    md.append(f"- `random_state = {RANDOM_STATE}`, `n_jobs = -1`, `bootstrap = True`.")
    md.append("- Each fold's TRAIN block is mirrored (original + side-swapped) and the preprocessing (median "
              "imputation) is fit on that augmented fold-train **only**. Fold validation is original and "
              "unmirrored.")
    md.append("- TRAIN metrics are reported on the **original unmirrored** fold-train rows, so train-validation "
              "gaps stay comparable across phases.")
    md.append(f"- Transformed dimension: **{EXPECTED_TRANSFORMED_FEATURES}** columns from 95 raw predictive "
              "inputs (62 directional + 18 continuous symmetric + 12 binary symmetric + 9 map dummies + 2 "
              "bestOf dummies + 3 tier dummies).\n")

    md.append("## Selection rule (fixed before the search ran)\n")
    md.append(SELECTION_RULE_TEXT + "\n")

    ranked = agg.sort_values("val_log_loss_mean").reset_index(drop=True)
    md.append("## Top 12 candidates by mean CV log loss\n")
    md.append("| rank | candidate_id | log loss (mean±std) | ROC-AUC | Brier | acc | depth | leaf | split | "
              "max_features | criterion | train-val AUC gap |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for i, r in ranked.head(12).iterrows():
        md.append(f"| {i+1} | {r['candidate_id']} | {r['val_log_loss_mean']:.4f}±{r['val_log_loss_std']:.4f} | "
                  f"{r['val_roc_auc_mean']:.4f} | {r['val_brier_mean']:.4f} | {r['val_accuracy_mean']:.4f} | "
                  f"{r['max_depth']} | {int(r['min_samples_leaf'])} | {int(r['min_samples_split'])} | "
                  f"{r['max_features']} | {r['criterion']} | {r['train_val_auc_gap_mean']:+.4f} |")
    md.append("")

    anchor_ref = agg[agg["candidate_id"] == "anchor_series_rf_v2_structure"].iloc[0]
    anchor_rank = int(ranked.index[ranked["candidate_id"] == "anchor_series_rf_v2_structure"][0]) + 1
    md.append("## The previous SERIES RF V2 structure as a reference\n")
    md.append(f"`anchor_series_rf_v2_structure` (the Phase 4B.1 series configuration, re-run here at "
              f"`n_estimators={N_ESTIMATORS}`) ranked **#{anchor_rank}** of {len(agg)}: log loss "
              f"{anchor_ref['val_log_loss_mean']:.4f} ± {anchor_ref['val_log_loss_std']:.4f}, ROC-AUC "
              f"{anchor_ref['val_roc_auc_mean']:.4f}, Brier {anchor_ref['val_brier_mean']:.4f}. It was included as "
              "an eligible anchor, never adopted automatically.\n")

    md.append("## Observations\n")
    for key, label in [("max_depth", "max_depth"), ("min_samples_leaf", "min_samples_leaf"),
                        ("criterion", "criterion")]:
        by = agg.groupby(key, dropna=False)[["val_log_loss_mean", "train_val_auc_gap_mean"]].mean().sort_index()
        md.append(f"Mean CV log loss by `{label}`:\n")
        md.append(f"| {label} | mean CV log loss | mean train-val AUC gap |")
        md.append("|---|---|---|")
        for idx, r in by.iterrows():
            md.append(f"| {idx} | {r['val_log_loss_mean']:.4f} | {r['train_val_auc_gap_mean']:+.4f} |")
        md.append("")
    md.append(f"Differences of a few thousandths across {len(agg)} candidates should not be over-interpreted - "
              f"that is exactly why the {LOG_LOSS_EQUIVALENCE_EPSILON} equivalence epsilon and the deterministic "
              "tie-break ladder exist.\n")

    w = params_by_id[winner_id]
    md.append("## Selected configuration (FROZEN)\n")
    md.append(f"**`{winner_id}`**, selected via: {stage}.\n")
    md.append("```\n" + ", ".join(f"{k}={w[k]}" for k in SEARCH_KEYS + ["n_estimators", "bootstrap"]) + "\n```\n")
    md.append(f"- mean CV log loss: {wagg['val_log_loss_mean']:.4f} ± {wagg['val_log_loss_std']:.4f}")
    md.append(f"- mean CV ROC-AUC: {wagg['val_roc_auc_mean']:.4f} ± {wagg['val_roc_auc_std']:.4f}")
    md.append(f"- mean CV Brier: {wagg['val_brier_mean']:.4f} | mean CV accuracy: {wagg['val_accuracy_mean']:.4f}")
    md.append(f"- mean train-validation ROC-AUC gap: {wagg['train_val_auc_gap_mean']:+.4f}")
    md.append(f"- mean fit time: {wagg['mean_fit_seconds']:.2f}s (operational reporting only - never part of "
              "selection)\n")
    md.append("Frozen in `data/modeling/map_random_forest_v1_selected_config.json`. Only after this freeze (and "
              "the XGBoost freeze, and the ensemble-weight freeze) may the main map validation partition be "
              "evaluated, exactly once.\n")

    (REPORTS / "phase6b_map_random_forest_tuning.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
