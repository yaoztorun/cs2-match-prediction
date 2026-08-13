"""
Phase 6D: FINAL, LOCAL TRAIN-only XGBoost hyperparameter search for the frozen
V3 known-map feature representation (config/map_features_v3_modern_map.yaml,
data/features/map_features_v3_modern_map.parquet - NEITHER is modified
anywhere in this phase).

This is deliberately NOT another broad search: Phase 6B already searched
broadly, and Phase 6C showed V3 helps only slightly under that frozen
configuration. Phase 6D asks only whether a MODEST local change - depth,
regularization, sampling, learning rate, tree count - does better, using the
SAME four TRAIN-only expanding-window folds (data/modeling/map_cv_folds_v1.csv,
reused byte-identically). The 1,129-map main VALIDATION partition was already
consumed once in Phase 6B and is NEVER opened here - this script contains no
read of data/modeling/map_split_v1.csv at all.

A0 vs "the old V3 reference" - do not conflate these
------------------------------------------------------------------------
`A0` in the ANCHOR_CONFIGS below carries the Phase 6B structural
hyperparameters (random_013), but its `n_estimators` for each outer fold is
whatever THIS script's own inner-early-stop procedure derives - it is
evaluated by the exact same two-stage fit every other candidate goes through,
not given a fixed tree count. This is a DIFFERENT quantity from "V3 + frozen
Phase 6B config (fixed n_estimators=124)", which is the Phase 6C Stage B
arm-B reference used only in the final three-way comparison in
scripts/finalize_map_xgboost_v3.py. A0 here answers "does this structure
still look good under V3, given V3-appropriate tree counts"; the frozen
124-tree arm answers "how much did feature engineering alone gain". Never
described interchangeably.

Writes:
    data/modeling/map_xgboost_v3_final_search_plan.json
    data/modeling/map_xgboost_v3_final_tuning_progress.jsonl   (checkpoint/resume)
    reports/tables/map_xgboost_v3_final_tuning.csv
    reports/phase6d_final_xgboost_v3_tuning.md
    data/modeling/map_xgboost_v3_final_selected_config.json
"""

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import ROOT, REPORTS                                    # noqa: E402
from map_modeling_common import (                                    # noqa: E402
    INNER_EARLY_STOP_FRACTION, LOG_LOSS_EQUIVALENCE_EPSILON, MODELING_DIR, N_FOLDS, RANDOM_STATE,
    append_checkpoint, compute_metrics, fold_frames, load_checkpoint, load_cv_manifest,
    package_versions, reset_checkpoint_if_stale, sha256_file, split_inner_early_stop,
)
from preprocessing_common_map_v3 import load_map_v3_roles, EXPECTED_RAW_PREDICTIVE_INPUTS
from preprocessing_common_map_v2 import build_augmented_training_raw, assert_augmented_symmetry
import preprocessing_xgboost_map_v3 as prep_xgb
from preprocessing_random_forest_map_v3 import EXPECTED_TRANSFORMED_FEATURES

CONFIG_PATH = ROOT / "config" / "map_features_v3_modern_map.yaml"
FEATURES_PATH = ROOT / "data" / "features" / "map_features_v3_modern_map.parquet"
CV_FOLDS_PATH = ROOT / "data" / "modeling" / "map_cv_folds_v1.csv"

TABLES_DIR = REPORTS / "tables"
PLAN_PATH = MODELING_DIR / "map_xgboost_v3_final_search_plan.json"
PROGRESS_PATH = MODELING_DIR / "map_xgboost_v3_final_tuning_progress.jsonl"
SELECTED_PATH = MODELING_DIR / "map_xgboost_v3_final_selected_config.json"

N_RANDOM_CANDIDATES = 16
N_ESTIMATORS_CAP = 2000
EARLY_STOPPING_ROUNDS = 120
EXPECTED_TOTAL_MAP_ROWS = 10318

FIXED_PARAMS = dict(
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

# Local search space, concentrated around the Phase 6B winner (brief section 10).
# max_depth is restricted to {2, 3} for the RANDOM draws - the single A7 anchor
# supplies the one permitted, controlled depth-4 check.
SEARCH_SPACE = {
    "learning_rate": [0.02, 0.03, 0.04, 0.05],
    "max_depth": [2, 3],
    "min_child_weight": [5, 10, 20, 40],
    "subsample": [0.70, 0.75, 0.85, 0.90, 1.00],
    "colsample_bytree": [0.70, 0.85, 1.00],
    "gamma": [2.0, 5.0, 8.0],
    "reg_alpha": [0.0, 0.01, 0.1, 0.5],
    "reg_lambda": [5.0, 10.0, 20.0, 30.0],
}
SEARCH_KEYS = list(SEARCH_SPACE.keys())

# The 8 required anchors (brief section 11), verbatim. A0 is the exact Phase 6B
# winner's structural hyperparameters - eligible for selection, evaluated
# through the identical two-stage procedure as every other candidate (see
# module docstring for why this is NOT the same as the frozen 124-tree arm).
ANCHOR_CONFIGS = [
    {"candidate_id": "A0_phase6b_reference_structure",
     "learning_rate": 0.03, "max_depth": 2, "min_child_weight": 10, "subsample": 0.75,
     "colsample_bytree": 0.85, "gamma": 5.0, "reg_alpha": 0.01, "reg_lambda": 10.0},
    {"candidate_id": "A1_less_l2",
     "learning_rate": 0.03, "max_depth": 2, "min_child_weight": 10, "subsample": 0.75,
     "colsample_bytree": 0.85, "gamma": 5.0, "reg_alpha": 0.01, "reg_lambda": 5.0},
    {"candidate_id": "A2_more_l2",
     "learning_rate": 0.03, "max_depth": 2, "min_child_weight": 10, "subsample": 0.75,
     "colsample_bytree": 0.85, "gamma": 5.0, "reg_alpha": 0.01, "reg_lambda": 20.0},
    {"candidate_id": "A3_stronger_child_constraint",
     "learning_rate": 0.03, "max_depth": 2, "min_child_weight": 20, "subsample": 0.75,
     "colsample_bytree": 0.85, "gamma": 5.0, "reg_alpha": 0.01, "reg_lambda": 10.0},
    {"candidate_id": "A4_less_gamma",
     "learning_rate": 0.03, "max_depth": 2, "min_child_weight": 10, "subsample": 0.75,
     "colsample_bytree": 0.85, "gamma": 2.0, "reg_alpha": 0.01, "reg_lambda": 10.0},
    {"candidate_id": "A5_more_sampling",
     "learning_rate": 0.03, "max_depth": 2, "min_child_weight": 10, "subsample": 0.90,
     "colsample_bytree": 1.00, "gamma": 5.0, "reg_alpha": 0.01, "reg_lambda": 10.0},
    {"candidate_id": "A6_depth3_strongly_regularized",
     "learning_rate": 0.03, "max_depth": 3, "min_child_weight": 20, "subsample": 0.75,
     "colsample_bytree": 0.85, "gamma": 5.0, "reg_alpha": 0.01, "reg_lambda": 10.0},
    {"candidate_id": "A7_depth4_controlled_check",
     "learning_rate": 0.03, "max_depth": 4, "min_child_weight": 40, "subsample": 0.75,
     "colsample_bytree": 0.85, "gamma": 8.0, "reg_alpha": 0.01, "reg_lambda": 20.0},
]

SELECTION_RULE_TEXT = (
    "1) PRIMARY: lowest mean outer-fold log loss. "
    f"2) EQUIVALENCE: within {LOG_LOSS_EQUIVALENCE_EPSILON} of the best mean log loss counts as tied. "
    "3) highest mean ROC-AUC. 4) lowest mean Brier. 5) lowest mean ABSOLUTE train-validation AUC gap "
    "(mean over folds of |train_auc - val_auc|). 6) highest mean accuracy. 7) lower log-loss standard "
    "deviation. 8) COMPLEXITY (deterministic): lower max_depth -> higher min_child_weight -> higher gamma -> "
    "higher reg_lambda -> higher reg_alpha -> lower median effective tree count -> candidate_id. "
    "subsample/colsample_bytree are deliberately NOT used as a complexity ranking. Fit time never "
    "participates in selection."
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
        params["candidate_id"] = f"local_{n:03d}"
        candidates.append(params)

    seen = set()
    for c in candidates:
        key = tuple(c[k] for k in SEARCH_KEYS)
        assert key not in seen, f"duplicate candidate hyperparameters: {c['candidate_id']}"
        seen.add(key)
    ids = [c["candidate_id"] for c in candidates]
    assert len(ids) == len(set(ids)), "duplicate candidate_id"
    return candidates


def assert_feature_freeze(roles):
    """Brief section 3: verify the V3 schema is exactly what Phase 6C froze,
    before any tuning happens. Aborts (raises) if anything differs."""
    assert len(roles["directional"]) == 80, roles["directional"]
    assert len(roles["symmetric"]) == 37, roles["symmetric"]
    assert len(roles["categorical"]) == 3
    assert len(roles["model_features"]) == EXPECTED_RAW_PREDICTIVE_INPUTS == 120
    assert roles["target"] == "team1_map_win"
    from map_modeling_common import FORBIDDEN_PREDICTORS
    bad = set(roles["model_features"]) & set(FORBIDDEN_PREDICTORS)
    assert not bad, f"forbidden column in the V3 model feature list: {bad}"
    print("Feature-freeze assertion PASSED: 120 raw inputs (80 directional + 37 symmetric + 3 categorical), "
          "target=team1_map_win, no forbidden columns.")


class XgbFoldCache:
    """Mirrors scripts/models/map_xgboost_tuning_v1.py's own XgbFoldCache
    exactly, pointed at V3's roles/preprocessing instead of V2's."""

    def __init__(self, cv, features_df, roles):
        self.folds = {}
        target = roles["target"]
        for fold in range(1, N_FOLDS + 1):
            raw_tr, raw_va = fold_frames(cv, features_df, fold)
            inner_fit, inner_es = split_inner_early_stop(raw_tr)

            assert inner_fit["series_datetime"].max() < inner_es["series_datetime"].min(), \
                f"fold {fold}: inner-fit / inner-early-stop chronology violated"
            assert inner_es["series_datetime"].max() < raw_va["series_datetime"].min(), \
                f"fold {fold}: inner-early-stop / fold-validation chronology violated"
            assert set(inner_fit["series_datetime"]).isdisjoint(set(inner_es["series_datetime"]))
            assert set(inner_es["series_datetime"]).isdisjoint(set(raw_va["series_datetime"]))
            assert set(inner_es["match_id"]).isdisjoint(set(raw_va["match_id"]))

            aug_inner = build_augmented_training_raw(inner_fit, roles)
            assert_augmented_symmetry(aug_inner, roles)
            p_inner = prep_xgb.fit_preprocessing(aug_inner, roles)
            X_inner_fit, _ = prep_xgb.transform(aug_inner, p_inner, roles)
            X_inner_es, _ = prep_xgb.transform(inner_es, p_inner, roles)

            aug_outer = build_augmented_training_raw(raw_tr, roles)
            assert_augmented_symmetry(aug_outer, roles)
            p_outer = prep_xgb.fit_preprocessing(aug_outer, roles)
            X_outer_aug, self.feature_names = prep_xgb.transform(aug_outer, p_outer, roles)
            X_outer_train_orig, _ = prep_xgb.transform(raw_tr, p_outer, roles)
            X_val, _ = prep_xgb.transform(raw_va, p_outer, roles)

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

    es_model = XGBClassifier(n_estimators=N_ESTIMATORS_CAP,
                              early_stopping_rounds=EARLY_STOPPING_ROUNDS, **hp, **FIXED_PARAMS)
    es_model.fit(e["X_inner_fit"], e["y_inner_fit"],
                  eval_set=[(e["X_inner_es"], e["y_inner_es"])], verbose=False)
    best_iteration = int(es_model.best_iteration)
    effective_n_estimators = best_iteration + 1
    del es_model

    t0 = time.perf_counter()
    model = XGBClassifier(n_estimators=effective_n_estimators, **hp, **FIXED_PARAMS)
    model.fit(e["X_outer_aug"], e["y_outer_aug"], verbose=False)   # NO eval_set, NO early stopping
    fit_seconds = time.perf_counter() - t0

    train_m = compute_metrics(e["y_outer_train_orig"], model.predict_proba(e["X_outer_train_orig"])[:, 1])
    val_m = compute_metrics(e["y_val"], model.predict_proba(e["X_val"])[:, 1])
    return train_m, val_m, best_iteration, effective_n_estimators, fit_seconds


def aggregate(fold_df, params_by_id):
    records = []
    for cid, g in fold_df.groupby("candidate_id"):
        p = params_by_id[cid]
        records.append({
            "candidate_id": cid, "fold": np.nan, "row_type": "aggregate",
            **{k: p[k] for k in SEARCH_KEYS},
            "val_log_loss_mean": g["val_log_loss"].mean(), "val_log_loss_std": g["val_log_loss"].std(ddof=0),
            "val_roc_auc_mean": g["val_roc_auc"].mean(), "val_roc_auc_std": g["val_roc_auc"].std(ddof=0),
            "val_brier_mean": g["val_brier"].mean(), "val_accuracy_mean": g["val_accuracy"].mean(),
            "val_precision_mean": g["val_precision"].mean(), "val_recall_mean": g["val_recall"].mean(),
            "val_f1_mean": g["val_f1"].mean(),
            "train_roc_auc_mean": g["train_roc_auc"].mean(), "train_log_loss_mean": g["train_log_loss"].mean(),
            "mean_abs_train_val_auc_gap": g["abs_train_val_auc_gap"].mean(),
            "train_val_auc_gap_mean": g["train_val_auc_gap"].mean(),
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
        stage = "quaternary (resolved by lowest mean ABSOLUTE train-validation AUC gap)"
        tied = tied[tied["mean_abs_train_val_auc_gap"] == tied["mean_abs_train_val_auc_gap"].min()]
    if len(tied) > 1:
        stage = "quinary (resolved by highest mean accuracy)"
        tied = tied[tied["val_accuracy_mean"] == tied["val_accuracy_mean"].max()]
    if len(tied) > 1:
        stage = "senary (resolved by lower log-loss standard deviation)"
        tied = tied[tied["val_log_loss_std"] == tied["val_log_loss_std"].min()]
    if len(tied) > 1:
        stage = "complexity (all metrics tied, resolved by deterministic complexity ordering)"
        tied = tied.assign(_ck=tied.apply(
            lambda r: complexity_key(params_by_id[r["candidate_id"]],
                                      r["median_effective_n_estimators"]), axis=1)).sort_values("_ck")
    return tied.iloc[0]["candidate_id"], stage, agg[agg["val_log_loss_mean"] <= best + LOG_LOSS_EQUIVALENCE_EPSILON]


def derive_final_n_estimators(best_iterations):
    return int(round(float(np.median([b + 1 for b in best_iterations]))))


def build_search_plan_v3(candidates):
    plan = {
        "model": "map_xgboost_v3_final",
        "seed": RANDOM_STATE,
        "n_folds": N_FOLDS,
        "log_loss_equivalence_epsilon": LOG_LOSS_EQUIVALENCE_EPSILON,
        "n_estimators_cap": N_ESTIMATORS_CAP,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "inner_early_stop_fraction": INNER_EARLY_STOP_FRACTION,
        "fixed_params": FIXED_PARAMS,
        "search_space": {k: [str(v) for v in vs] for k, vs in SEARCH_SPACE.items()},
        "candidates": candidates,
        "selection_rule": SELECTION_RULE_TEXT,
        "artifact_hashes": {
            "config/map_features_v3_modern_map.yaml": sha256_file(CONFIG_PATH),
            "data/features/map_features_v3_modern_map.parquet": sha256_file(FEATURES_PATH),
            "data/modeling/map_cv_folds_v1.csv": sha256_file(CV_FOLDS_PATH),
        },
        "package_versions": package_versions(),
        "cv_manifest": "data/modeling/map_cv_folds_v1.csv",
        "main_validation_used_in_selection": False,
        "main_validation_status": "consumed_in_phase_6b_not_reopened",
    }
    plan["plan_hash"] = __import__("hashlib").sha256(
        json.dumps({k: v for k, v in plan.items() if k != "plan_hash"},
                    sort_keys=True, default=str).encode("utf-8")).hexdigest()
    return plan


def main():
    roles = load_map_v3_roles(CONFIG_PATH)
    assert_feature_freeze(roles)

    features = pd.read_parquet(FEATURES_PATH, engine="fastparquet")
    assert len(features) == EXPECTED_TOTAL_MAP_ROWS

    # STRUCTURAL GUARANTEE: this script never reads map_split_v1.csv - only the
    # pre-built, TRAIN-only CV fold manifest.
    cv = load_cv_manifest(verify_against_split=False)

    candidates = build_candidates()
    assert len(candidates) == 24, len(candidates)
    params_by_id = {c["candidate_id"]: c for c in candidates}
    for req in ["A0_phase6b_reference_structure", "A1_less_l2", "A2_more_l2", "A3_stronger_child_constraint",
                "A4_less_gamma", "A5_more_sampling", "A6_depth3_strongly_regularized",
                "A7_depth4_controlled_check"]:
        assert req in params_by_id, f"required anchor missing: {req}"

    plan = build_search_plan_v3(candidates)
    MODELING_DIR.mkdir(exist_ok=True, parents=True)
    PLAN_PATH.write_text(json.dumps(plan, indent=2, default=str), encoding="utf-8")

    if reset_checkpoint_if_stale(PROGRESS_PATH, plan["plan_hash"]):
        print("Search plan changed since the last run - previous progress discarded (never mixed).")
    done = load_checkpoint(PROGRESS_PATH, plan["plan_hash"])
    print(f"Plan hash {plan['plan_hash'][:12]}... | {len(candidates)} candidates x {N_FOLDS} folds | "
          f"{len(done)} already completed, resuming the rest")

    print("Building per-fold V3 preprocessing cache once (inner + outer stages)...")
    cache = XgbFoldCache(cv, features, roles)
    for f in range(1, N_FOLDS + 1):
        e = cache[f]
        print(f"  fold {f}: inner-fit {e['n_inner_fit_unique']} maps | inner-early-stop {e['n_inner_es']} maps | "
              f"outer-train {e['n_train_unique']} -> {e['n_train_augmented']} augmented | val {e['n_val']}")
    assert cache[1]["X_outer_aug"].shape[1] == EXPECTED_TRANSFORMED_FEATURES == 131

    rows = []
    for ci, params in enumerate(candidates, start=1):
        for fold in range(1, N_FOLDS + 1):
            key = (params["candidate_id"], fold)
            if key in done:
                rows.append(done[key]["row"])
                continue
            train_m, val_m, best_iteration, eff_n, fit_seconds = evaluate_on_fold(params, cache, fold)
            print(f"XGB-V3-final candidate {ci}/{len(candidates)} ({params['candidate_id']}) - "
                  f"fold {fold}/{N_FOLDS} - best_iter={best_iteration}")
            gap = train_m["roc_auc"] - val_m["roc_auc"]
            row = {
                "candidate_id": params["candidate_id"], "fold": fold, "row_type": "fold",
                **{k: params[k] for k in SEARCH_KEYS},
                "n_train_unique": cache[fold]["n_train_unique"],
                "n_train_augmented": cache[fold]["n_train_augmented"],
                "best_iteration": best_iteration, "effective_n_estimators": eff_n,
                "train_accuracy": train_m["accuracy"], "train_roc_auc": train_m["roc_auc"],
                "train_log_loss": train_m["log_loss"],
                "val_accuracy": val_m["accuracy"], "val_precision": val_m["precision"],
                "val_recall": val_m["recall"], "val_f1": val_m["f1"], "val_roc_auc": val_m["roc_auc"],
                "val_log_loss": val_m["log_loss"], "val_brier": val_m["brier"],
                "train_val_auc_gap": gap, "abs_train_val_auc_gap": abs(gap),
                "fit_seconds": fit_seconds,
            }
            rows.append(row)
            append_checkpoint(PROGRESS_PATH, {"plan_hash": plan["plan_hash"],
                                               "candidate_id": params["candidate_id"], "fold": fold, "row": row})

    fold_df = pd.DataFrame(rows)
    agg = aggregate(fold_df, params_by_id)

    TABLES_DIR.mkdir(exist_ok=True, parents=True)
    pd.concat([fold_df, agg], ignore_index=True, sort=False).to_csv(
        TABLES_DIR / "map_xgboost_v3_final_tuning.csv", index=False, encoding="utf-8")

    winner_id, stage, equivalent = select_winner(agg, params_by_id)
    w = params_by_id[winner_id]
    wagg = agg[agg["candidate_id"] == winner_id].iloc[0]
    best_iterations = fold_df[fold_df["candidate_id"] == winner_id].sort_values("fold")["best_iteration"] \
        .astype(int).tolist()
    final_n_estimators = derive_final_n_estimators(best_iterations)

    # ---------------- stability guard: fold-wise deltas vs A0 (brief section 18) ----------------
    a0_folds = fold_df[fold_df["candidate_id"] == "A0_phase6b_reference_structure"].set_index("fold")
    stability_rows = []
    for cid in equivalent["candidate_id"]:
        cfolds = fold_df[fold_df["candidate_id"] == cid].set_index("fold")
        for fold in range(1, N_FOLDS + 1):
            stability_rows.append({
                "candidate_id": cid, "fold": fold,
                "delta_log_loss_vs_A0": cfolds.loc[fold, "val_log_loss"] - a0_folds.loc[fold, "val_log_loss"],
                "delta_roc_auc_vs_A0": cfolds.loc[fold, "val_roc_auc"] - a0_folds.loc[fold, "val_roc_auc"],
                "delta_brier_vs_A0": cfolds.loc[fold, "val_brier"] - a0_folds.loc[fold, "val_brier"],
                "delta_accuracy_vs_A0": cfolds.loc[fold, "val_accuracy"] - a0_folds.loc[fold, "val_accuracy"],
            })
    stability_df = pd.DataFrame(stability_rows)
    winner_stability = stability_df[stability_df["candidate_id"] == winner_id]
    n_folds_improving = {
        "log_loss": int((winner_stability["delta_log_loss_vs_A0"] < 0).sum()),
        "roc_auc": int((winner_stability["delta_roc_auc_vs_A0"] > 0).sum()),
        "brier": int((winner_stability["delta_brier_vs_A0"] < 0).sum()),
        "accuracy": int((winner_stability["delta_accuracy_vs_A0"] > 0).sum()),
    }

    print(f"\nSELECTED: {winner_id} via {stage}")
    print("  params: " + str({k: w[k] for k in SEARCH_KEYS}))
    print(f"  mean CV log loss {wagg['val_log_loss_mean']:.4f} | mean CV ROC-AUC {wagg['val_roc_auc_mean']:.4f}")
    print(f"  best_iterations by fold: {best_iterations} -> final_n_estimators = {final_n_estimators}")
    print(f"  folds improving vs A0 (diagnostic only): {n_folds_improving}")

    selected = {
        "selected_candidate_id": winner_id,
        "is_A0_phase6b_reference_structure": winner_id == "A0_phase6b_reference_structure",
        "params": {k: w[k] for k in SEARCH_KEYS},
        "fixed_params": dict(FIXED_PARAMS),
        "selection_stage": stage,
        "selection_rule": SELECTION_RULE_TEXT,
        "log_loss_epsilon": LOG_LOSS_EQUIVALENCE_EPSILON,
        "n_equivalent_candidates": int(len(equivalent)),
        "equivalent_candidate_ids": equivalent["candidate_id"].tolist(),
        "cv_mean_log_loss": float(wagg["val_log_loss_mean"]),
        "cv_std_log_loss": float(wagg["val_log_loss_std"]),
        "cv_mean_roc_auc": float(wagg["val_roc_auc_mean"]),
        "cv_mean_brier": float(wagg["val_brier_mean"]),
        "cv_mean_accuracy": float(wagg["val_accuracy_mean"]),
        "cv_mean_abs_train_val_auc_gap": float(wagg["mean_abs_train_val_auc_gap"]),
        "best_iterations_by_fold": best_iterations,
        "median_best_iteration": float(np.median(best_iterations)),
        "final_n_estimators": final_n_estimators,
        "final_n_estimators_rule": "round(median(best_iteration + 1 across the 4 outer folds)) of the SELECTED "
                                    "candidate's own fold best-iterations",
        "n_folds_improving_vs_A0_diagnostic_only": n_folds_improving,
        "n_candidates": len(candidates), "n_anchor_candidates": len(ANCHOR_CONFIGS),
        "n_random_candidates": len(candidates) - len(ANCHOR_CONFIGS),
        "transformed_feature_count": EXPECTED_TRANSFORMED_FEATURES,
        "cv_folds_artifact": "data/modeling/map_cv_folds_v1.csv",
        "tuning_results_artifact": "reports/tables/map_xgboost_v3_final_tuning.csv",
        "search_plan_artifact": "data/modeling/map_xgboost_v3_final_search_plan.json",
        "search_plan_hash": plan["plan_hash"],
        "main_validation_used_in_selection": False,
        "note_on_A0": ("A0 carries the Phase 6B structural hyperparameters but its tree count is derived by "
                        "THIS script's own inner-early-stop procedure per fold - not the frozen 124-tree count "
                        "used for the Phase 6C Stage B 'old V3 reference' comparison arm."),
    }
    SELECTED_PATH.write_text(json.dumps(selected, indent=2, default=str), encoding="utf-8")

    write_report(agg, winner_id, stage, params_by_id, wagg, best_iterations, final_n_estimators,
                 equivalent, stability_df, n_folds_improving, cache)
    print("\nWrote reports/tables/map_xgboost_v3_final_tuning.csv")
    print("Wrote data/modeling/map_xgboost_v3_final_selected_config.json")
    print("Wrote reports/phase6d_final_xgboost_v3_tuning.md")


def write_report(agg, winner_id, stage, params_by_id, wagg, best_iterations, final_n_estimators,
                  equivalent, stability_df, n_folds_improving, cache):
    md = []
    md.append("# Phase 6D - Final Local XGBoost V3 Tuning (TRAIN-only chronological CV)\n")
    md.append(f"Generated by `scripts/models/map_xgboost_v3_final_tuning.py`. **{len(agg)} candidates** "
              f"({len(ANCHOR_CONFIGS)} required anchors + {len(agg) - len(ANCHOR_CONFIGS)} "
              f"`RandomState({RANDOM_STATE})` local draws) x {N_FOLDS} folds. This is a LOCAL search around the "
              "Phase 6B winner, not a repeat of Phase 6B's broad search - features are frozen (Phase 6C); only "
              "hyperparameters are searched here.\n")
    md.append("**The 1,129-map main VALIDATION partition was already consumed in Phase 6B and is NOT reopened "
              "here** - this script contains no read of `data/modeling/map_split_v1.csv` at all.\n")

    md.append("## A0 vs the frozen 124-tree V3 reference - not the same quantity\n")
    md.append("`A0_phase6b_reference_structure` carries the Phase 6B structural hyperparameters, but its "
              "per-fold tree count is derived by THIS script's own inner-early-stop procedure - it is evaluated "
              "exactly like every other candidate here. This is different from the frozen "
              "`n_estimators=124` reference used in Phase 6C Stage B's V2-vs-V3 comparison; the two are never "
              "conflated in this report or in `scripts/finalize_map_xgboost_v3.py`'s three-way comparison.\n")

    md.append("## Selection rule (fixed before the search ran)\n")
    md.append(SELECTION_RULE_TEXT + "\n")

    ranked = agg.sort_values("val_log_loss_mean").reset_index(drop=True)
    md.append("## All candidates by mean CV log loss\n")
    md.append("| rank | candidate_id | log loss (mean±std) | ROC-AUC | Brier | acc | |gap| | lr | depth | mcw | "
              "sub | col | gamma | alpha | lambda | med trees |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for i, r in ranked.iterrows():
        md.append(f"| {i+1} | {r['candidate_id']} | {r['val_log_loss_mean']:.4f}±{r['val_log_loss_std']:.4f} | "
                  f"{r['val_roc_auc_mean']:.4f} | {r['val_brier_mean']:.4f} | {r['val_accuracy_mean']:.4f} | "
                  f"{r['mean_abs_train_val_auc_gap']:.4f} | {r['learning_rate']} | {int(r['max_depth'])} | "
                  f"{int(r['min_child_weight'])} | {r['subsample']} | {r['colsample_bytree']} | {r['gamma']} | "
                  f"{r['reg_alpha']} | {r['reg_lambda']} | {r['median_effective_n_estimators']:.0f} |")
    md.append("")

    md.append(f"## Equivalence set ({len(equivalent)} candidate(s) within {LOG_LOSS_EQUIVALENCE_EPSILON} of the "
              "best mean log loss)\n")
    md.append(", ".join(f"`{c}`" for c in equivalent["candidate_id"]) + "\n")

    md.append("## Stability guard: fold-wise deltas vs A0 (diagnostic only, brief section 18)\n")
    md.append("| candidate_id | fold | Δ log loss | Δ ROC-AUC | Δ Brier | Δ accuracy |")
    md.append("|---|---|---|---|---|---|")
    for _, r in stability_df.iterrows():
        md.append(f"| {r['candidate_id']} | {int(r['fold'])} | {r['delta_log_loss_vs_A0']:+.4f} | "
                  f"{r['delta_roc_auc_vs_A0']:+.4f} | {r['delta_brier_vs_A0']:+.4f} | "
                  f"{r['delta_accuracy_vs_A0']:+.4f} |")
    md.append("")

    w = params_by_id[winner_id]
    md.append("## Selected configuration (FROZEN)\n")
    md.append(f"**`{winner_id}`**, selected via: {stage}.\n")
    md.append("```\n" + ", ".join(f"{k}={w[k]}" for k in SEARCH_KEYS) + "\n```\n")
    md.append(f"- mean CV log loss: {wagg['val_log_loss_mean']:.4f} ± {wagg['val_log_loss_std']:.4f}")
    md.append(f"- mean CV ROC-AUC: {wagg['val_roc_auc_mean']:.4f}")
    md.append(f"- mean CV Brier: {wagg['val_brier_mean']:.4f} | mean CV accuracy: {wagg['val_accuracy_mean']:.4f}")
    md.append(f"- mean absolute train-validation AUC gap: {wagg['mean_abs_train_val_auc_gap']:.4f}")
    md.append(f"- `best_iteration` by fold: {best_iterations}")
    md.append(f"- **final_n_estimators = round(median({[b + 1 for b in best_iterations]})) = "
              f"{final_n_estimators}**, frozen before the full-TRAIN refit.")
    md.append(f"- folds improving vs A0 (diagnostic only, never overrides the selection rule): "
              f"{n_folds_improving}\n")
    if winner_id == "A0_phase6b_reference_structure":
        md.append("**The Phase 6B structural configuration remains best under this predefined rule - Phase 6D "
                  "does not require selecting a different one, and does not.**\n")

    (REPORTS / "phase6d_final_xgboost_v3_tuning.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
