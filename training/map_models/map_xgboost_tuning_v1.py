"""
Phase 6B: XGBoost hyperparameter search for the KNOWN-MAP task, over the same
four TRAIN-only expanding-window folds as the map Random Forest
(data/modeling/map_cv_folds_v1.csv, reused byte-identically).

This script NEVER opens data/modeling/map_split_v1.csv - the main VALIDATION
and TEST partitions are structurally absent from selection.

THE TWO-STAGE FIT PER CANDIDATE x FOLD (brief sections 17-18)
--------------------------------------------------------------
The outer fold's validation block must never be consulted to decide when to
stop boosting, or its score becomes optimistic. So, inside each outer fold:

  1. Split the fold's OWN train history chronologically into
        INNER FIT       (earliest ~85% of series-timestamp groups, mirrored)
        INNER EARLY STOP(latest ~15%, original/unmirrored)
     Every map of a series - and every exact-timestamp group - stays entirely
     on one side.
  2. Fit preprocessing on the augmented INNER FIT only; fit with
     early_stopping_rounds against INNER EARLY STOP to learn `best_iteration`.
  3. Record best_iteration + 1, then DISCARD that model.
  4. Re-fit preprocessing on the FULL outer-fold TRAIN, mirror it, and refit a
     NEW model with n_estimators = best_iteration + 1, no eval_set, no early
     stopping - so all available fold-train history is used.
  5. Only then score the untouched OUTER FOLD VALIDATION block.

Asserted every time:
    max(inner_fit.dt) < min(inner_early_stop.dt) < ... < min(fold_val.dt)

Writes:
    reports/tables/map_xgboost_tuning_v1.csv
    reports/phase6b_map_xgboost_tuning.md
    data/modeling/map_xgboost_v1_selected_config.json
    data/modeling/map_xgb_search_plan_v1.json         (deterministic, no timestamps)
    data/modeling/map_xgb_tuning_progress_v1.jsonl    (checkpoint/resume)
"""

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier


from _common import REPORTS                                          # noqa: E402
from training.map_models.map_modeling_common import (                                    # noqa: E402
    INNER_EARLY_STOP_FRACTION, LOG_LOSS_EQUIVALENCE_EPSILON, MODELING_DIR, N_FOLDS, RANDOM_STATE,
    append_checkpoint, assert_target_and_no_forbidden_columns, build_search_plan, compute_metrics,
    fold_frames, load_checkpoint, load_cv_manifest, load_features, load_roles,
    reset_checkpoint_if_stale, split_inner_early_stop,
)
from feature_engineering.preprocessing.preprocessing_common_map_v2 import (                            # noqa: E402
    EXPECTED_TRANSFORMED_FEATURES, build_augmented_training_raw, assert_augmented_symmetry,
)
import feature_engineering.preprocessing.preprocessing_xgboost_map_v2 as prep_xgb                      # noqa: E402

TABLES_DIR = REPORTS / "tables"
PLAN_PATH = MODELING_DIR / "map_xgb_search_plan_v1.json"
PROGRESS_PATH = MODELING_DIR / "map_xgb_tuning_progress_v1.jsonl"
SELECTED_PATH = MODELING_DIR / "map_xgboost_v1_selected_config.json"

N_RANDOM_CANDIDATES = 30
N_ESTIMATORS_CAP = 2500          # UPPER CAP during fold tuning only
EARLY_STOPPING_ROUNDS = 120

FIXED_PARAMS = dict(
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

SEARCH_SPACE = {
    "learning_rate": [0.015, 0.02, 0.03, 0.05, 0.08],
    "max_depth": [2, 3, 4, 5, 6],
    "min_child_weight": [5, 10, 20, 40, 80],
    "subsample": [0.60, 0.75, 0.90, 1.00],
    "colsample_bytree": [0.50, 0.70, 0.85, 1.00],
    "gamma": [0.0, 0.5, 1.0, 2.0, 5.0],
    "reg_alpha": [0.0, 0.01, 0.1, 0.5, 1.0],
    "reg_lambda": [1.0, 2.0, 5.0, 10.0, 20.0],
}
SEARCH_KEYS = list(SEARCH_SPACE.keys())

# Deterministic anchors, fixed before running anything. The previous SERIES XGB
# V2 structure is ONE eligible reference anchor - included for comparison, never
# adopted automatically.
ANCHOR_CONFIGS = [
    {"candidate_id": "anchor_series_xgb_v2_structure",
     "learning_rate": 0.02, "max_depth": 4, "min_child_weight": 20, "subsample": 0.60,
     "colsample_bytree": 0.85, "gamma": 2.0, "reg_alpha": 0.01, "reg_lambda": 1.0},
    {"candidate_id": "anchor_shallow_conservative_depth2",
     "learning_rate": 0.03, "max_depth": 2, "min_child_weight": 20, "subsample": 0.90,
     "colsample_bytree": 0.85, "gamma": 0.0, "reg_alpha": 0.0, "reg_lambda": 2.0},
    {"candidate_id": "anchor_shallow_conservative_depth3",
     "learning_rate": 0.03, "max_depth": 3, "min_child_weight": 20, "subsample": 0.90,
     "colsample_bytree": 0.85, "gamma": 0.0, "reg_alpha": 0.0, "reg_lambda": 2.0},
    {"candidate_id": "anchor_strong_min_child_weight",
     "learning_rate": 0.03, "max_depth": 4, "min_child_weight": 80, "subsample": 0.90,
     "colsample_bytree": 0.85, "gamma": 0.0, "reg_alpha": 0.0, "reg_lambda": 1.0},
    {"candidate_id": "anchor_strong_gamma",
     "learning_rate": 0.03, "max_depth": 4, "min_child_weight": 20, "subsample": 0.90,
     "colsample_bytree": 0.85, "gamma": 5.0, "reg_alpha": 0.0, "reg_lambda": 1.0},
    {"candidate_id": "anchor_strong_l1_l2",
     "learning_rate": 0.03, "max_depth": 4, "min_child_weight": 20, "subsample": 0.90,
     "colsample_bytree": 0.85, "gamma": 0.0, "reg_alpha": 1.0, "reg_lambda": 20.0},
    {"candidate_id": "anchor_high_subsampling",
     "learning_rate": 0.03, "max_depth": 4, "min_child_weight": 10, "subsample": 0.60,
     "colsample_bytree": 0.50, "gamma": 0.0, "reg_alpha": 0.0, "reg_lambda": 1.0},
    {"candidate_id": "anchor_low_lr_0p015",
     "learning_rate": 0.015, "max_depth": 3, "min_child_weight": 20, "subsample": 0.90,
     "colsample_bytree": 0.85, "gamma": 0.0, "reg_alpha": 0.0, "reg_lambda": 1.0},
    {"candidate_id": "anchor_moderate_lr_0p05",
     "learning_rate": 0.05, "max_depth": 3, "min_child_weight": 20, "subsample": 0.90,
     "colsample_bytree": 0.85, "gamma": 0.0, "reg_alpha": 0.0, "reg_lambda": 1.0},
    {"candidate_id": "anchor_depth2_strong_regularization",
     "learning_rate": 0.05, "max_depth": 2, "min_child_weight": 40, "subsample": 0.75,
     "colsample_bytree": 0.70, "gamma": 1.0, "reg_alpha": 0.1, "reg_lambda": 5.0},
]

SELECTION_RULE_TEXT = (
    "1) PRIMARY: lowest mean outer-fold log loss. "
    f"2) EQUIVALENCE: within {LOG_LOSS_EQUIVALENCE_EPSILON} of the best mean log loss counts as tied. "
    "3) highest mean ROC-AUC. 4) lowest mean Brier. 5) highest mean accuracy. 6) lower log-loss standard "
    "deviation. 7) COMPLEXITY (deterministic): lower max_depth -> higher min_child_weight -> higher gamma -> "
    "higher reg_lambda -> higher reg_alpha -> lower median effective tree count -> candidate_id. "
    "subsample/colsample_bytree are tunable predictive hyperparameters and are deliberately NOT used as a "
    "complexity ranking. Fit time never participates in selection."
)


def build_candidates():
    grid = list(itertools.product(*[SEARCH_SPACE[k] for k in SEARCH_KEYS]))
    anchor_tuples = {tuple(a[k] for k in SEARCH_KEYS) for a in ANCHOR_CONFIGS}
    eligible = [t for t in grid if t not in anchor_tuples]

    rng = np.random.RandomState(RANDOM_STATE)
    idx = sorted(rng.choice(len(eligible), size=N_RANDOM_CANDIDATES, replace=False).tolist())

    candidates = [dict(a) for a in ANCHOR_CONFIGS]
    for n, i in enumerate(idx, start=1):
        params = dict(zip(SEARCH_KEYS, eligible[i]))
        params["candidate_id"] = f"random_{n:03d}"
        candidates.append(params)

    seen = set()
    for c in candidates:
        key = tuple(c[k] for k in SEARCH_KEYS)
        assert key not in seen, f"duplicate candidate hyperparameters: {c['candidate_id']}"
        seen.add(key)
    ids = [c["candidate_id"] for c in candidates]
    assert len(ids) == len(set(ids)), "duplicate candidate_id"
    return candidates


class XgbFoldCache:
    """Per-fold structures reused by every candidate. Two independent
    preprocessing fits per fold, each on exactly the data its stage is allowed
    to see:
        inner:  fit on augmented INNER FIT      -> inner-fit / inner-early-stop matrices
        outer:  fit on augmented FULL fold TRAIN -> outer-refit / fold-validation matrices
    """

    def __init__(self, cv, features_df, roles):
        self.folds = {}
        target = roles["target"]
        for fold in range(1, N_FOLDS + 1):
            raw_tr, raw_va = fold_frames(cv, features_df, fold)
            inner_fit, inner_es = split_inner_early_stop(raw_tr)

            # three-stage temporal separation, asserted every fold
            assert inner_fit["series_datetime"].max() < inner_es["series_datetime"].min(), \
                f"fold {fold}: inner-fit / inner-early-stop chronology violated"
            assert inner_es["series_datetime"].max() < raw_va["series_datetime"].min(), \
                f"fold {fold}: inner-early-stop / fold-validation chronology violated"
            assert set(inner_fit["series_datetime"]).isdisjoint(set(inner_es["series_datetime"]))
            assert set(inner_es["series_datetime"]).isdisjoint(set(raw_va["series_datetime"]))
            assert set(inner_es["match_id"]).isdisjoint(set(raw_va["match_id"]))

            aug_inner = build_augmented_training_raw(inner_fit, roles)      # mirrored: INNER FIT only
            assert_augmented_symmetry(aug_inner, roles)
            p_inner = prep_xgb.fit_preprocessing(aug_inner, roles)          # fit on augmented INNER FIT only
            X_inner_fit, _ = prep_xgb.transform(aug_inner, p_inner, roles)
            X_inner_es, _ = prep_xgb.transform(inner_es, p_inner, roles)    # never mirrored

            aug_outer = build_augmented_training_raw(raw_tr, roles)          # mirrored: FULL fold train
            assert_augmented_symmetry(aug_outer, roles)
            p_outer = prep_xgb.fit_preprocessing(aug_outer, roles)           # refit from scratch
            X_outer_aug, self.feature_names = prep_xgb.transform(aug_outer, p_outer, roles)
            X_outer_train_orig, _ = prep_xgb.transform(raw_tr, p_outer, roles)
            X_val, _ = prep_xgb.transform(raw_va, p_outer, roles)            # never mirrored

            self.folds[fold] = {
                "X_inner_fit": X_inner_fit, "y_inner_fit": aug_inner[target].to_numpy(dtype=float),
                "X_inner_es": X_inner_es, "y_inner_es": inner_es[target].to_numpy(dtype=float),
                "X_outer_aug": X_outer_aug, "y_outer_aug": aug_outer[target].to_numpy(dtype=float),
                "X_outer_train_orig": X_outer_train_orig,
                "y_outer_train_orig": raw_tr[target].to_numpy(dtype=float),
                "X_val": X_val, "y_val": raw_va[target].to_numpy(dtype=float),
                "n_train_unique": len(raw_tr), "n_train_augmented": len(aug_outer),
                "n_inner_fit_unique": len(inner_fit), "n_inner_es": len(inner_es), "n_val": len(raw_va),
            }

    def __getitem__(self, fold):
        return self.folds[fold]


def evaluate_on_fold(params, cache, fold):
    e = cache[fold]
    hp = {k: v for k, v in params.items() if k != "candidate_id"}

    # --- stage 1: learn best_iteration from the INNER EARLY STOP block only ---
    es_model = XGBClassifier(n_estimators=N_ESTIMATORS_CAP,
                              early_stopping_rounds=EARLY_STOPPING_ROUNDS, **hp, **FIXED_PARAMS)
    es_model.fit(e["X_inner_fit"], e["y_inner_fit"],
                  eval_set=[(e["X_inner_es"], e["y_inner_es"])], verbose=False)
    best_iteration = int(es_model.best_iteration)
    effective_n_estimators = best_iteration + 1
    best_score = float(es_model.best_score) if es_model.best_score is not None else float("nan")
    del es_model                                   # discarded: it never scores anything

    # --- stage 2: refit on the FULL augmented outer-fold train, fixed tree count ---
    t0 = time.perf_counter()
    model = XGBClassifier(n_estimators=effective_n_estimators, **hp, **FIXED_PARAMS)
    model.fit(e["X_outer_aug"], e["y_outer_aug"], verbose=False)   # NO eval_set, NO early stopping
    fit_seconds = time.perf_counter() - t0

    train_m = compute_metrics(e["y_outer_train_orig"], model.predict_proba(e["X_outer_train_orig"])[:, 1])
    val_m = compute_metrics(e["y_val"], model.predict_proba(e["X_val"])[:, 1])
    return train_m, val_m, best_iteration, effective_n_estimators, best_score, fit_seconds


def aggregate(fold_df, params_by_id):
    records = []
    for cid, g in fold_df.groupby("candidate_id"):
        p = params_by_id[cid]
        records.append({
            "candidate_id": cid, "fold": np.nan, "row_type": "aggregate",
            **{k: p[k] for k in SEARCH_KEYS},
            "val_log_loss_mean": g["val_log_loss"].mean(), "val_log_loss_std": g["val_log_loss"].std(ddof=0),
            "val_roc_auc_mean": g["val_roc_auc"].mean(), "val_roc_auc_std": g["val_roc_auc"].std(ddof=0),
            "val_brier_mean": g["val_brier"].mean(), "val_brier_std": g["val_brier"].std(ddof=0),
            "val_accuracy_mean": g["val_accuracy"].mean(), "val_accuracy_std": g["val_accuracy"].std(ddof=0),
            "val_precision_mean": g["val_precision"].mean(), "val_recall_mean": g["val_recall"].mean(),
            "val_f1_mean": g["val_f1"].mean(),
            "train_roc_auc_mean": g["train_roc_auc"].mean(), "train_log_loss_mean": g["train_log_loss"].mean(),
            "train_accuracy_mean": g["train_accuracy"].mean(),
            "train_val_auc_gap_mean": g["train_val_auc_gap"].mean(),
            "train_val_log_loss_gap_mean": g["train_val_log_loss_gap"].mean(),
            "best_iterations": json.dumps(g.sort_values("fold")["best_iteration"].astype(int).tolist()),
            "median_effective_n_estimators": float(np.median(g["effective_n_estimators"])),
            "mean_fit_seconds": g["fit_seconds"].mean(),
        })
    return pd.DataFrame(records).sort_values("candidate_id").reset_index(drop=True)


def complexity_key(params, median_trees):
    return (params["max_depth"], -params["min_child_weight"], -params["gamma"],
            -params["reg_lambda"], -params["reg_alpha"], median_trees, params["candidate_id"])


def select_winner(agg, params_by_id):
    best = agg["val_log_loss_mean"].min()
    tied = agg[agg["val_log_loss_mean"] <= best + LOG_LOSS_EQUIVALENCE_EPSILON].copy()
    stage = "primary (lowest mean outer-fold log loss, unique)"
    if len(tied) > 1:
        stage = "secondary (log-loss tie within epsilon, resolved by highest mean ROC-AUC)"
        tied = tied[tied["val_roc_auc_mean"] == tied["val_roc_auc_mean"].max()]
    if len(tied) > 1:
        stage = "tertiary (resolved by lowest mean Brier)"
        tied = tied[tied["val_brier_mean"] == tied["val_brier_mean"].min()]
    if len(tied) > 1:
        stage = "quaternary (resolved by highest mean accuracy)"
        tied = tied[tied["val_accuracy_mean"] == tied["val_accuracy_mean"].max()]
    if len(tied) > 1:
        stage = "quinary (resolved by lower log-loss standard deviation)"
        tied = tied[tied["val_log_loss_std"] == tied["val_log_loss_std"].min()]
    if len(tied) > 1:
        stage = "complexity (all metrics tied, resolved by deterministic complexity ordering)"
        tied = tied.assign(_ck=tied.apply(
            lambda r: complexity_key(params_by_id[r["candidate_id"]],
                                      r["median_effective_n_estimators"]), axis=1)).sort_values("_ck")
    return tied.iloc[0]["candidate_id"], stage


def derive_final_n_estimators(best_iterations):
    """FIXED BEFORE THE SEARCH: round(median(best_iteration + 1 across the 4 outer folds))."""
    return int(round(float(np.median([b + 1 for b in best_iterations]))))


def main():
    roles = load_roles()
    features = load_features()
    assert_target_and_no_forbidden_columns(features, roles)
    cv = load_cv_manifest(verify_against_split=False)

    candidates = build_candidates()
    params_by_id = {c["candidate_id"]: c for c in candidates}
    plan = build_search_plan("map_xgboost_v1", candidates, {
        "n_estimators_cap": N_ESTIMATORS_CAP,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "inner_early_stop_fraction": INNER_EARLY_STOP_FRACTION,
        "fixed_params": FIXED_PARAMS,
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
    print(f"Plan hash {plan['plan_hash'][:12]}... | {len(candidates)} candidates x {N_FOLDS} folds | "
          f"{len(done)} already completed, resuming the rest")

    print("Building per-fold XGB preprocessing cache once (inner + outer stages)...")
    cache = XgbFoldCache(cv, features, roles)
    for f in range(1, N_FOLDS + 1):
        e = cache[f]
        print(f"  fold {f}: inner-fit {e['n_inner_fit_unique']} maps | inner-early-stop {e['n_inner_es']} maps | "
              f"outer-train {e['n_train_unique']} -> {e['n_train_augmented']} augmented | val {e['n_val']}")
    assert cache[1]["X_outer_aug"].shape[1] == EXPECTED_TRANSFORMED_FEATURES

    rows = []
    for ci, params in enumerate(candidates, start=1):
        for fold in range(1, N_FOLDS + 1):
            key = (params["candidate_id"], fold)
            if key in done:
                rows.append(done[key]["row"])
                continue
            train_m, val_m, best_iteration, eff_n, best_score, fit_seconds = \
                evaluate_on_fold(params, cache, fold)
            print(f"XGB candidate {ci}/{len(candidates)} ({params['candidate_id']}) - fold {fold}/{N_FOLDS} - "
                  f"best_iter={best_iteration}")
            row = {
                "candidate_id": params["candidate_id"], "fold": fold, "row_type": "fold",
                **{k: params[k] for k in SEARCH_KEYS},
                "n_train_unique": cache[fold]["n_train_unique"],
                "n_train_augmented": cache[fold]["n_train_augmented"],
                "n_inner_fit_unique": cache[fold]["n_inner_fit_unique"],
                "n_inner_early_stop": cache[fold]["n_inner_es"], "n_val": cache[fold]["n_val"],
                "best_iteration": best_iteration, "effective_n_estimators": eff_n,
                "best_score_inner_early_stop": best_score,
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
        TABLES_DIR / "map_xgboost_tuning_v1.csv", index=False, encoding="utf-8")

    winner_id, stage = select_winner(agg, params_by_id)
    w = params_by_id[winner_id]
    wagg = agg[agg["candidate_id"] == winner_id].iloc[0]
    best_iterations = fold_df[fold_df["candidate_id"] == winner_id].sort_values("fold")["best_iteration"] \
        .astype(int).tolist()
    final_n_estimators = derive_final_n_estimators(best_iterations)

    print(f"\nSELECTED: {winner_id} via {stage}")
    print("  params: " + str({k: w[k] for k in SEARCH_KEYS}))
    print(f"  mean CV log loss {wagg['val_log_loss_mean']:.4f} +/- {wagg['val_log_loss_std']:.4f} | "
          f"mean CV ROC-AUC {wagg['val_roc_auc_mean']:.4f}")
    print(f"  best_iterations by fold: {best_iterations} -> final_n_estimators = {final_n_estimators}")

    selected = {
        "selected_candidate_id": winner_id,
        "params": {k: w[k] for k in SEARCH_KEYS},
        "fixed_params": dict(FIXED_PARAMS),
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
        "best_iterations_by_fold": best_iterations,
        "median_best_iteration": float(np.median(best_iterations)),
        "final_n_estimators": final_n_estimators,
        "final_n_estimators_rule": "round(median(best_iteration + 1 across the 4 outer folds))",
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "n_estimators_cap_during_cv": N_ESTIMATORS_CAP,
        "inner_early_stop_fraction": INNER_EARLY_STOP_FRACTION,
        "n_candidates": len(candidates),
        "n_anchor_candidates": len(ANCHOR_CONFIGS),
        "n_random_candidates": len(candidates) - len(ANCHOR_CONFIGS),
        "transformed_feature_count": EXPECTED_TRANSFORMED_FEATURES,
        "cv_folds_artifact": "data/modeling/map_cv_folds_v1.csv",
        "tuning_results_artifact": "reports/tables/map_xgboost_tuning_v1.csv",
        "search_plan_artifact": "data/modeling/map_xgb_search_plan_v1.json",
        "search_plan_hash": plan["plan_hash"],
        "main_validation_used_in_selection": False,
    }
    SELECTED_PATH.write_text(json.dumps(selected, indent=2, default=str), encoding="utf-8")

    write_report(agg, winner_id, stage, params_by_id, wagg, best_iterations, final_n_estimators, cache)
    print("\nWrote reports/tables/map_xgboost_tuning_v1.csv")
    print("Wrote data/modeling/map_xgboost_v1_selected_config.json")
    print("Wrote reports/phase6b_map_xgboost_tuning.md")


def write_report(agg, winner_id, stage, params_by_id, wagg, best_iterations, final_n_estimators, cache):
    md = []
    md.append("# Phase 6B - Known-Map XGBoost Tuning (TRAIN-only chronological CV)\n")
    md.append(f"Generated by `training/map_models/map_xgboost_tuning_v1.py`. **{len(agg)} candidates** "
              f"({len(ANCHOR_CONFIGS)} deterministic anchors + {len(agg) - len(ANCHOR_CONFIGS)} "
              f"`RandomState({RANDOM_STATE})` draws) x {N_FOLDS} expanding-window folds.\n")
    md.append("**Target: `team1_map_win`** - the winner of one specific, user-selected map. Not the pre-veto "
              "series task; these numbers are not comparable with the series models' accuracies.\n")

    md.append("## The two-stage fit per candidate x fold\n")
    md.append("```\nINNER FIT             earliest ~85% of the fold's own train history (mirrored)\n"
              "    v strictly earlier than\n"
              "INNER EARLY STOP      latest ~15% of the fold's own train history (unmirrored)\n"
              "    v strictly earlier than\n"
              "OUTER FOLD VALIDATION scored ONLY after the refit - never consulted for tree count\n```\n")
    md.append("1. Fit preprocessing on the augmented INNER FIT only; fit with "
              f"`early_stopping_rounds={EARLY_STOPPING_ROUNDS}` and `n_estimators={N_ESTIMATORS_CAP}` (cap) "
              "against INNER EARLY STOP to learn `best_iteration`.\n"
              "2. Record `best_iteration + 1` and **discard** that model.\n"
              "3. Re-fit preprocessing on the FULL outer-fold TRAIN, mirror it, and refit a **new** model with "
              "`n_estimators = best_iteration + 1`, **no `eval_set`, no early stopping**.\n"
              "4. Only then score the untouched outer-fold validation block.\n")
    md.append("Every map of a series - and every exact-timestamp group - stays entirely on one side of the inner "
              "boundary, asserted per fold. Block sizes:\n")
    md.append("| fold | inner-fit maps | inner-early-stop maps | outer-train maps (augmented) | validation maps |")
    md.append("|---|---|---|---|---|")
    for f in range(1, N_FOLDS + 1):
        e = cache[f]
        md.append(f"| {f} | {e['n_inner_fit_unique']} | {e['n_inner_es']} | {e['n_train_unique']} "
                  f"({e['n_train_augmented']}) | {e['n_val']} |")
    md.append("")

    md.append("## Why the main validation partition cannot have influenced this\n")
    md.append("This script never opens `data/modeling/map_split_v1.csv`; it reads only the Phase 6A CV manifest, "
              "which contains TRAIN match_ids only. Early stopping consumes a block that lives entirely inside "
              "the fold's own training history, never the outer validation block and never the main validation "
              "partition.\n")

    md.append("## Selection rule (fixed before the search ran)\n")
    md.append(SELECTION_RULE_TEXT + "\n")
    md.append(f"`final_n_estimators = round(median(best_iteration + 1 across the {N_FOLDS} folds))`, also fixed "
              "in advance and derived from CV only.\n")

    ranked = agg.sort_values("val_log_loss_mean").reset_index(drop=True)
    md.append("## Top 12 candidates by mean outer-fold log loss\n")
    md.append("| rank | candidate_id | log loss (mean±std) | ROC-AUC | Brier | acc | lr | depth | mcw | sub | "
              "col | gamma | alpha | lambda | med trees | AUC gap |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for i, r in ranked.head(12).iterrows():
        md.append(f"| {i+1} | {r['candidate_id']} | {r['val_log_loss_mean']:.4f}±{r['val_log_loss_std']:.4f} | "
                  f"{r['val_roc_auc_mean']:.4f} | {r['val_brier_mean']:.4f} | {r['val_accuracy_mean']:.4f} | "
                  f"{r['learning_rate']} | {int(r['max_depth'])} | {int(r['min_child_weight'])} | "
                  f"{r['subsample']} | {r['colsample_bytree']} | {r['gamma']} | {r['reg_alpha']} | "
                  f"{r['reg_lambda']} | {r['median_effective_n_estimators']:.0f} | "
                  f"{r['train_val_auc_gap_mean']:+.4f} |")
    md.append("")

    ref = agg[agg["candidate_id"] == "anchor_series_xgb_v2_structure"].iloc[0]
    ref_rank = int(ranked.index[ranked["candidate_id"] == "anchor_series_xgb_v2_structure"][0]) + 1
    md.append("## The previous SERIES XGB V2 structure as a reference\n")
    md.append(f"`anchor_series_xgb_v2_structure` ranked **#{ref_rank}** of {len(agg)}: log loss "
              f"{ref['val_log_loss_mean']:.4f} ± {ref['val_log_loss_std']:.4f}, ROC-AUC "
              f"{ref['val_roc_auc_mean']:.4f}, Brier {ref['val_brier_mean']:.4f}, median effective trees "
              f"{ref['median_effective_n_estimators']:.0f}. Included as an eligible anchor, never adopted "
              "automatically.\n")

    md.append("## Observations\n")
    by_depth = agg.groupby("max_depth")[["val_log_loss_mean", "train_val_auc_gap_mean"]].mean().sort_index()
    md.append("Mean outer-fold log loss by `max_depth`:\n")
    md.append("| max_depth | mean log loss | mean train-val AUC gap |")
    md.append("|---|---|---|")
    for d, r in by_depth.iterrows():
        md.append(f"| {int(d)} | {r['val_log_loss_mean']:.4f} | {r['train_val_auc_gap_mean']:+.4f} |")
    md.append("")
    by_lr = agg.groupby("learning_rate")["median_effective_n_estimators"].mean().sort_index()
    md.append("Mean median effective boosting rounds by `learning_rate`:\n")
    md.append("| learning_rate | mean median effective trees |")
    md.append("|---|---|")
    for lr, v in by_lr.items():
        md.append(f"| {lr} | {v:.0f} |")
    md.append("")

    md.append("## Selected configuration (FROZEN)\n")
    md.append(f"**`{winner_id}`**, selected via: {stage}.\n")
    md.append("```\n" + ", ".join(f"{k}={params_by_id[winner_id][k]}" for k in SEARCH_KEYS) + "\n```\n")
    md.append(f"- mean outer-fold log loss: {wagg['val_log_loss_mean']:.4f} ± {wagg['val_log_loss_std']:.4f}")
    md.append(f"- mean outer-fold ROC-AUC: {wagg['val_roc_auc_mean']:.4f} ± {wagg['val_roc_auc_std']:.4f}")
    md.append(f"- mean Brier: {wagg['val_brier_mean']:.4f} | mean accuracy: {wagg['val_accuracy_mean']:.4f}")
    md.append(f"- mean train-validation ROC-AUC gap: {wagg['train_val_auc_gap_mean']:+.4f}")
    md.append(f"- `best_iteration` by fold: {best_iterations}")
    md.append(f"- **final_n_estimators = round(median({[b + 1 for b in best_iterations]})) = "
              f"{final_n_estimators}**, frozen before any validation data is opened.\n")
    md.append("Frozen in `data/modeling/map_xgboost_v1_selected_config.json`. The full-TRAIN refit uses this "
              "fixed tree count with no `eval_set` and no early stopping.\n")

    (REPORTS / "phases" / "phase6b_map_xgboost_tuning.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
