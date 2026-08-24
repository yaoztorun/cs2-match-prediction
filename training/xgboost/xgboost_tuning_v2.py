"""
Phase 4C.1: XGBoost V2 hyperparameter search via expanding-window
chronological CV, using ONLY the 6,619-match TRAIN partition (through the
SAME folds as Random Forest V2: data/modeling/random_forest_cv_folds_v2.csv).
This script NEVER loads the 1,419-match main VALIDATION partition or the
TEST partition.

THREE-STAGE TEMPORAL SEPARATION (the key methodological property)
-----------------------------------------------------------------
Using one block for BOTH early stopping AND candidate scoring would make the
score optimistic - the model would already have used that block to decide
when to stop. So within each outer fold, the fold's training history is split
chronologically again:

    INNER FIT            (earliest ~80-85% of fold-train)   -> model fitting
        v  (strictly earlier than)
    INNER EARLY STOP     (latest ~15-20% of fold-train)     -> eval_set / best_iteration ONLY
        v  (strictly earlier than)
    OUTER FOLD VALIDATION (the fold's own validation block) -> candidate scoring ONLY

Enforced by assertion for every fold:
    max(inner_fit.dt)        < min(inner_early_stop.dt)
    max(inner_early_stop.dt) < min(fold_validation.dt)
and no exact-timestamp group may cross either boundary.

Mirroring is applied to INNER FIT only. Preprocessing is fit on the mirrored
inner-fit only. Inner-early-stop and outer-fold-validation are never mirrored.
TRAIN metrics are reported on the ORIGINAL UNMIRRORED inner-fit rows.

Early stopping is legitimate here because the inner-early-stop block lives
entirely inside the global TRAIN period and is never the block a candidate is
scored on.

Writes:
    reports/tables/xgboost_tuning_v2.csv
    reports/phase4c1_xgboost_tuning.md
    data/modeling/xgboost_v2_selected_config.json
"""

import itertools
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss, brier_score_loss
from xgboost import XGBClassifier

from _common import ROOT, REPORTS
from feature_engineering.series.build_series_split_v1 import pick_boundary_index
from feature_engineering.preprocessing.preprocessing_xgboost_v1 import build_augmented_training_raw, fit_preprocessing, transform

CONFIG_PATH = ROOT / "config" / "features" / "series_features_v1.yaml"
FEATURES_PATH = ROOT / "data" / "features" / "series_features_v1.parquet"
CV_FOLDS_PATH = ROOT / "data" / "modeling" / "random_forest_cv_folds_v2.csv"
MODELING_DIR = ROOT / "data" / "modeling"
TABLES_DIR = REPORTS / "tables"

RANDOM_STATE = 42
N_RANDOM_CANDIDATES = 30
N_FOLDS = 4
LOG_LOSS_EQUIVALENCE_EPSILON = 0.002   # fixed BEFORE the search - never altered afterwards
N_ESTIMATORS_CAP = 2000
EARLY_STOPPING_ROUNDS = 100
INNER_EARLY_STOP_FRACTION = 0.175      # latest ~17.5% of fold-train -> inner early-stop block
TUNING_VERSION = "1.0.0"

# xgboost 3.4.0: `early_stopping_rounds` is not in the explicit __init__ or fit()
# signature, but IS accepted as a constructor keyword (absorbed via **kwargs) and
# correctly populates best_iteration/best_score. Verified by running it before
# writing this script. Documented in the report as the API actually used.
FIXED_PARAMS = dict(
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

SEARCH_SPACE = {
    "learning_rate": [0.02, 0.03, 0.05, 0.08, 0.10],
    "max_depth": [2, 3, 4, 5, 6],
    "min_child_weight": [1, 3, 5, 10, 20],
    "subsample": [0.60, 0.75, 0.90, 1.00],
    "colsample_bytree": [0.60, 0.75, 0.90, 1.00],
    "gamma": [0.0, 0.1, 0.5, 1.0, 2.0],
    "reg_alpha": [0.0, 0.01, 0.1, 0.5, 1.0],
    "reg_lambda": [0.5, 1.0, 2.0, 5.0, 10.0],
}
SEARCH_KEYS = list(SEARCH_SPACE.keys())

# Deterministic anchors, defined BEFORE running anything - never chosen or
# adjusted after seeing results. `xgb_v1_structure_reference` reproduces
# XGBoost V1's exact structural hyperparameters and IS ELIGIBLE FOR SELECTION:
# V1's structure is already moderately regularized, and chronological CV may
# legitimately conclude the structure was fine but 300 fixed rounds was not.
ANCHOR_CONFIGS = [
    {"candidate_id": "xgb_v1_structure_reference",
     "learning_rate": 0.05, "max_depth": 4, "min_child_weight": 1, "subsample": 0.90,
     "colsample_bytree": 0.90, "gamma": 0.0, "reg_alpha": 0.0, "reg_lambda": 1.0},
    {"candidate_id": "anchor_shallow_depth2_conservative",
     "learning_rate": 0.05, "max_depth": 2, "min_child_weight": 20, "subsample": 0.90,
     "colsample_bytree": 0.90, "gamma": 0.0, "reg_alpha": 0.0, "reg_lambda": 1.0},
    {"candidate_id": "anchor_shallow_depth3_conservative",
     "learning_rate": 0.05, "max_depth": 3, "min_child_weight": 10, "subsample": 0.90,
     "colsample_bytree": 0.90, "gamma": 0.0, "reg_alpha": 0.0, "reg_lambda": 1.0},
    {"candidate_id": "anchor_moderate_depth3_mcw5",
     "learning_rate": 0.05, "max_depth": 3, "min_child_weight": 5, "subsample": 0.75,
     "colsample_bytree": 0.75, "gamma": 0.0, "reg_alpha": 0.0, "reg_lambda": 1.0},
    {"candidate_id": "anchor_moderate_depth4_mcw3",
     "learning_rate": 0.05, "max_depth": 4, "min_child_weight": 3, "subsample": 0.75,
     "colsample_bytree": 0.75, "gamma": 0.0, "reg_alpha": 0.0, "reg_lambda": 2.0},
    {"candidate_id": "anchor_strong_reg_gamma",
     "learning_rate": 0.05, "max_depth": 4, "min_child_weight": 5, "subsample": 0.90,
     "colsample_bytree": 0.90, "gamma": 2.0, "reg_alpha": 0.0, "reg_lambda": 2.0},
    {"candidate_id": "anchor_strong_reg_lambda_alpha",
     "learning_rate": 0.05, "max_depth": 4, "min_child_weight": 5, "subsample": 0.90,
     "colsample_bytree": 0.90, "gamma": 0.0, "reg_alpha": 1.0, "reg_lambda": 10.0},
    {"candidate_id": "anchor_strong_subsampling",
     "learning_rate": 0.05, "max_depth": 4, "min_child_weight": 3, "subsample": 0.60,
     "colsample_bytree": 0.60, "gamma": 0.0, "reg_alpha": 0.0, "reg_lambda": 1.0},
    {"candidate_id": "anchor_low_lr_0p02",
     "learning_rate": 0.02, "max_depth": 3, "min_child_weight": 5, "subsample": 0.90,
     "colsample_bytree": 0.90, "gamma": 0.0, "reg_alpha": 0.0, "reg_lambda": 1.0},
    {"candidate_id": "anchor_faster_lr_0p10",
     "learning_rate": 0.10, "max_depth": 3, "min_child_weight": 5, "subsample": 0.90,
     "colsample_bytree": 0.90, "gamma": 0.0, "reg_alpha": 0.0, "reg_lambda": 1.0},
]

SELECTION_RULE_TEXT = (
    "1) PRIMARY: lowest mean CV log loss. "
    f"2) EQUIVALENCE: candidates within {LOG_LOSS_EQUIVALENCE_EPSILON} of the best mean log loss are "
    "treated as essentially equivalent. 3) SECONDARY: highest mean CV ROC-AUC. "
    "4) TERTIARY: lower CV log-loss standard deviation. "
    "5) COMPLEXITY (deterministic, in order): lower max_depth -> higher min_child_weight -> higher gamma "
    "-> higher reg_lambda -> higher reg_alpha -> lower median effective tree count -> candidate_id "
    "lexicographic order. subsample/colsample_bytree are tunable predictive hyperparameters and are "
    "deliberately NOT used as a model-complexity ranking."
)


def build_random_candidates():
    grid = list(itertools.product(*[SEARCH_SPACE[k] for k in SEARCH_KEYS]))  # deterministic order
    anchor_tuples = {tuple(a[k] for k in SEARCH_KEYS) for a in ANCHOR_CONFIGS}
    eligible = [t for t in grid if t not in anchor_tuples]

    rng = np.random.RandomState(RANDOM_STATE)
    idx = sorted(rng.choice(len(eligible), size=N_RANDOM_CANDIDATES, replace=False).tolist())

    out = []
    for n, i in enumerate(idx, start=1):
        params = dict(zip(SEARCH_KEYS, eligible[i]))
        params["candidate_id"] = f"random_{n:03d}"
        out.append(params)
    return out


def all_candidates():
    cands = list(ANCHOR_CONFIGS) + build_random_candidates()
    seen = set()
    for c in cands:
        key = tuple(c[k] for k in SEARCH_KEYS)
        assert key not in seen, f"duplicate candidate hyperparameters: {c['candidate_id']}"
        seen.add(key)
    ids = [c["candidate_id"] for c in cands]
    assert len(ids) == len(set(ids)), "duplicate candidate_id"
    return cands


def split_inner_early_stop(fold_train_df, fraction=INNER_EARLY_STOP_FRACTION):
    """Chronologically split a fold's training history into INNER FIT (earlier)
    and INNER EARLY STOP (later), never splitting an exact-timestamp group.
    Returns (inner_fit_df, inner_es_df)."""
    d = fold_train_df.sort_values(["datetime", "match_id"]).reset_index(drop=True)
    total = len(d)
    group_sizes = d.groupby("datetime").size().sort_index()
    datetimes = group_sizes.index.tolist()
    cum, running = [], 0
    for c in group_sizes.values.tolist():
        running += c
        cum.append(running)

    target = (1.0 - fraction) * total
    b = pick_boundary_index(cum, target)
    # guarantee both sides are non-empty
    b = max(0, min(b, len(datetimes) - 2))
    cutoff_dt = datetimes[b]

    inner_fit = d[d["datetime"] <= cutoff_dt]
    inner_es = d[d["datetime"] > cutoff_dt]
    assert len(inner_fit) > 0 and len(inner_es) > 0, "inner split produced an empty side"
    return inner_fit.reset_index(drop=True), inner_es.reset_index(drop=True)


def compute_metrics(y_true, y_proba, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "log_loss": float(log_loss(y_true, y_proba, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def evaluate_candidate_on_fold(params, fold, cv_df, features_df, model_features, target_col):
    fold_train_ids = set(cv_df.loc[(cv_df["fold"] == fold) & (cv_df["role"] == "train"), "match_id"])
    fold_val_ids = set(cv_df.loc[(cv_df["fold"] == fold) & (cv_df["role"] == "validation"), "match_id"])

    fold_train_raw = features_df[features_df["match_id"].isin(fold_train_ids)].reset_index(drop=True)
    fold_val_raw = features_df[features_df["match_id"].isin(fold_val_ids)].reset_index(drop=True)

    inner_fit_raw, inner_es_raw = split_inner_early_stop(fold_train_raw)

    # --- THREE-STAGE TEMPORAL SEPARATION, asserted every time ---
    assert inner_fit_raw["datetime"].max() < inner_es_raw["datetime"].min(), \
        f"fold {fold}: inner-fit/inner-early-stop chronology violated"
    assert inner_es_raw["datetime"].max() < fold_val_raw["datetime"].min(), \
        f"fold {fold}: inner-early-stop/fold-validation chronology violated"
    # no exact-timestamp group crosses either boundary
    assert set(inner_fit_raw["datetime"]).isdisjoint(set(inner_es_raw["datetime"]))
    assert set(inner_es_raw["datetime"]).isdisjoint(set(fold_val_raw["datetime"]))

    # mirror INNER FIT only; fit preprocessing on THAT mirrored set only
    augmented_inner_fit = build_augmented_training_raw(inner_fit_raw)
    fold_params = fit_preprocessing(augmented_inner_fit, model_features)

    X_fit, _ = transform(augmented_inner_fit, fold_params)
    y_fit = augmented_inner_fit[target_col].to_numpy(dtype=float)

    X_inner_fit_orig, _ = transform(inner_fit_raw, fold_params)          # UNMIRRORED -> train metrics
    y_inner_fit_orig = inner_fit_raw[target_col].to_numpy(dtype=float)

    X_es, _ = transform(inner_es_raw, fold_params)                        # never mirrored -> early stopping only
    y_es = inner_es_raw[target_col].to_numpy(dtype=float)

    X_val, _ = transform(fold_val_raw, fold_params)                       # never mirrored -> scoring only
    y_val = fold_val_raw[target_col].to_numpy(dtype=float)

    hp = {k: v for k, v in params.items() if k != "candidate_id"}
    model = XGBClassifier(n_estimators=N_ESTIMATORS_CAP,
                           early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                           **hp, **FIXED_PARAMS)
    model.fit(X_fit, y_fit, eval_set=[(X_es, y_es)], verbose=False)

    best_iteration = int(model.best_iteration)
    effective_n_estimators = best_iteration + 1
    best_score = float(model.best_score) if model.best_score is not None else float("nan")

    proba_tr = model.predict_proba(X_inner_fit_orig)[:, 1]
    pred_tr = model.predict(X_inner_fit_orig)
    train_m = compute_metrics(y_inner_fit_orig, proba_tr, pred_tr)

    proba_val = model.predict_proba(X_val)[:, 1]
    pred_val = model.predict(X_val)
    val_m = compute_metrics(y_val, proba_val, pred_val)

    return {
        "train_metrics": train_m, "val_metrics": val_m,
        "best_iteration": best_iteration, "effective_n_estimators": effective_n_estimators,
        "best_score": best_score,
        "unique_fold_train_matches": len(fold_train_raw),
        "unique_inner_fit_matches": len(inner_fit_raw),
        "augmented_inner_fit_observations": len(augmented_inner_fit),
        "inner_early_stop_matches": len(inner_es_raw),
        "fold_validation_matches": len(fold_val_raw),
    }


def complexity_key(params, median_trees):
    """Deterministic complexity ordering - LOWER is 'simpler/more regularized'.
    subsample/colsample_bytree deliberately excluded: they are tunable
    predictive hyperparameters, not a monotone complexity ranking."""
    return (
        params["max_depth"],                 # lower is simpler
        -params["min_child_weight"],         # higher is simpler
        -params["gamma"],                    # higher is simpler
        -params["reg_lambda"],               # higher is simpler
        -params["reg_alpha"],                # higher is simpler
        median_trees,                        # fewer trees is simpler
        params["candidate_id"],              # final deterministic tie-break
    )


def select_winner(agg_df, params_by_id):
    """Applies the fixed selection rule. agg_df needs columns:
    candidate_id, val_log_loss_mean, val_roc_auc_mean, val_log_loss_std,
    median_effective_n_estimators."""
    best_ll = agg_df["val_log_loss_mean"].min()
    tied = agg_df[agg_df["val_log_loss_mean"] <= best_ll + LOG_LOSS_EQUIVALENCE_EPSILON].copy()
    stage = "primary (lowest mean CV log loss, unique)"

    if len(tied) > 1:
        best_auc = tied["val_roc_auc_mean"].max()
        tied2 = tied[tied["val_roc_auc_mean"] == best_auc]
        stage = "secondary (log-loss tie within epsilon, resolved by highest mean CV ROC-AUC)"
        if len(tied2) > 1:
            best_std = tied2["val_log_loss_std"].min()
            tied3 = tied2[tied2["val_log_loss_std"] == best_std]
            stage = "tertiary (log-loss and ROC-AUC tied, resolved by lower CV log-loss std)"
            if len(tied3) > 1:
                tied3 = tied3.copy()
                tied3["_ck"] = tied3.apply(
                    lambda r: complexity_key(params_by_id[r["candidate_id"]],
                                              r["median_effective_n_estimators"]), axis=1)
                tied3 = tied3.sort_values("_ck")
                stage = "complexity (all metrics tied, resolved by deterministic complexity ordering)"
            tied2 = tied3
        tied = tied2

    return tied.iloc[0]["candidate_id"], stage


def derive_final_n_estimators(best_iterations):
    """FIXED BEFORE THE SEARCH: round(median(best_iteration + 1 across folds))."""
    return int(round(float(np.median([b + 1 for b in best_iterations]))))


def main():
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    model_features = cfg["model_features"]
    target_col = cfg["target"]

    features_df = pd.read_parquet(FEATURES_PATH, engine="fastparquet")
    cv_df = pd.read_csv(CV_FOLDS_PATH, parse_dates=["datetime"])
    # STRUCTURAL GUARANTEE: series_split_v1.csv's validation/test rows are never
    # read here - only the pre-built RF V2 fold manifest, which by construction
    # contains only TRAIN match_ids.

    # re-verify outer fold chronology rather than trusting the manifest
    for fold in range(1, N_FOLDS + 1):
        ft = cv_df[(cv_df["fold"] == fold) & (cv_df["role"] == "train")]
        fv = cv_df[(cv_df["fold"] == fold) & (cv_df["role"] == "validation")]
        assert ft["datetime"].max() < fv["datetime"].min(), f"fold {fold}: outer chronology violated"

    candidates = all_candidates()
    print(f"Evaluating {len(candidates)} candidates ({len(ANCHOR_CONFIGS)} anchors + "
          f"{len(candidates) - len(ANCHOR_CONFIGS)} random) x {N_FOLDS} folds = "
          f"{len(candidates) * N_FOLDS} fits, each with an inner early-stop split")

    params_by_id = {c["candidate_id"]: c for c in candidates}
    rows = []
    for ci, params in enumerate(candidates, start=1):
        for fold in range(1, N_FOLDS + 1):
            r = evaluate_candidate_on_fold(params, fold, cv_df, features_df, model_features, target_col)
            tm, vm = r["train_metrics"], r["val_metrics"]
            rows.append({
                "candidate_id": params["candidate_id"], "fold": fold, "row_type": "fold",
                **{k: params[k] for k in SEARCH_KEYS},
                "unique_fold_train_matches": r["unique_fold_train_matches"],
                "unique_inner_fit_matches": r["unique_inner_fit_matches"],
                "augmented_inner_fit_observations": r["augmented_inner_fit_observations"],
                "inner_early_stop_matches": r["inner_early_stop_matches"],
                "fold_validation_matches": r["fold_validation_matches"],
                "best_iteration": r["best_iteration"],
                "effective_n_estimators": r["effective_n_estimators"],
                "best_score_inner_early_stop": r["best_score"],
                "train_accuracy": tm["accuracy"], "train_roc_auc": tm["roc_auc"],
                "train_log_loss": tm["log_loss"], "train_brier": tm["brier"], "train_f1": tm["f1"],
                "val_accuracy": vm["accuracy"], "val_roc_auc": vm["roc_auc"],
                "val_log_loss": vm["log_loss"], "val_brier": vm["brier"], "val_f1": vm["f1"],
                "train_val_acc_gap": tm["accuracy"] - vm["accuracy"],
                "train_val_auc_gap": tm["roc_auc"] - vm["roc_auc"],
                "train_val_log_loss_gap": vm["log_loss"] - tm["log_loss"],
            })
        print(f"  [{ci}/{len(candidates)}] {params['candidate_id']} done")

    fold_df = pd.DataFrame(rows)

    agg_records = []
    for cid, g in fold_df.groupby("candidate_id"):
        p = params_by_id[cid]
        agg_records.append({
            "candidate_id": cid, "fold": np.nan, "row_type": "aggregate",
            **{k: p[k] for k in SEARCH_KEYS},
            "val_log_loss_mean": g["val_log_loss"].mean(), "val_log_loss_std": g["val_log_loss"].std(ddof=0),
            "val_roc_auc_mean": g["val_roc_auc"].mean(), "val_roc_auc_std": g["val_roc_auc"].std(ddof=0),
            "val_brier_mean": g["val_brier"].mean(), "val_brier_std": g["val_brier"].std(ddof=0),
            "val_accuracy_mean": g["val_accuracy"].mean(), "val_accuracy_std": g["val_accuracy"].std(ddof=0),
            "val_f1_mean": g["val_f1"].mean(), "val_f1_std": g["val_f1"].std(ddof=0),
            "train_accuracy_mean": g["train_accuracy"].mean(), "train_roc_auc_mean": g["train_roc_auc"].mean(),
            "train_val_acc_gap_mean": g["train_val_acc_gap"].mean(),
            "train_val_auc_gap_mean": g["train_val_auc_gap"].mean(),
            "train_val_log_loss_gap_mean": g["train_val_log_loss_gap"].mean(),
            "median_effective_n_estimators": float(np.median(g["effective_n_estimators"])),
        })
    agg_df = pd.DataFrame(agg_records)

    TABLES_DIR.mkdir(exist_ok=True, parents=True)
    pd.concat([fold_df, agg_df], ignore_index=True, sort=False).to_csv(
        TABLES_DIR / "xgboost_tuning_v2.csv", index=False, encoding="utf-8")

    # ---------------- selection (main validation never involved) ----------------
    winner_id, stage = select_winner(agg_df, params_by_id)
    winner_params = params_by_id[winner_id]
    winner_agg = agg_df[agg_df["candidate_id"] == winner_id].iloc[0]

    winner_folds = fold_df[fold_df["candidate_id"] == winner_id].sort_values("fold")
    best_iterations = winner_folds["best_iteration"].astype(int).tolist()
    final_n_estimators = derive_final_n_estimators(best_iterations)

    print(f"\nSELECTED: {winner_id} via {stage}")
    print(f"  params: {winner_params}")
    print(f"  mean CV log loss: {winner_agg['val_log_loss_mean']:.4f} +/- {winner_agg['val_log_loss_std']:.4f}")
    print(f"  mean CV ROC-AUC:  {winner_agg['val_roc_auc_mean']:.4f} +/- {winner_agg['val_roc_auc_std']:.4f}")
    print(f"  best_iterations by fold: {best_iterations} -> final_n_estimators = {final_n_estimators}")

    selected = {
        "selected_candidate_id": winner_id,
        "params": {k: winner_params[k] for k in SEARCH_KEYS},
        "fixed_params": dict(FIXED_PARAMS),
        "cv_mean_log_loss": float(winner_agg["val_log_loss_mean"]),
        "cv_std_log_loss": float(winner_agg["val_log_loss_std"]),
        "cv_mean_roc_auc": float(winner_agg["val_roc_auc_mean"]),
        "cv_std_roc_auc": float(winner_agg["val_roc_auc_std"]),
        "cv_mean_brier": float(winner_agg["val_brier_mean"]),
        "cv_mean_accuracy": float(winner_agg["val_accuracy_mean"]),
        "cv_mean_train_val_auc_gap": float(winner_agg["train_val_auc_gap_mean"]),
        "best_iterations_by_fold": best_iterations,
        "median_best_iteration": float(np.median(best_iterations)),
        "final_n_estimators": final_n_estimators,
        "final_n_estimators_rule": "round(median(best_iteration + 1 across the 4 folds))",
        "selection_stage": stage,
        "selection_rule": SELECTION_RULE_TEXT,
        "log_loss_epsilon": LOG_LOSS_EQUIVALENCE_EPSILON,
        "early_stopping_api": ("xgboost 3.4.0: early_stopping_rounds passed as an XGBClassifier "
                                "constructor keyword + fit(eval_set=[(inner_early_stop)])"),
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "n_estimators_cap_during_cv": N_ESTIMATORS_CAP,
        "inner_early_stop_fraction": INNER_EARLY_STOP_FRACTION,
        "n_candidates": len(candidates),
        "n_anchor_candidates": len(ANCHOR_CONFIGS),
        "n_random_candidates": len(candidates) - len(ANCHOR_CONFIGS),
        "cv_folds_artifact": "data/modeling/random_forest_cv_folds_v2.csv",
        "tuning_results_artifact": "reports/tables/xgboost_tuning_v2.csv",
        "tuning_version": TUNING_VERSION,
        "selection_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "main_validation_used_in_selection": False,
    }
    (MODELING_DIR / "xgboost_v2_selected_config.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")

    write_report(fold_df, agg_df, winner_id, stage, params_by_id, best_iterations, final_n_estimators)

    print("\nWrote reports/tables/xgboost_tuning_v2.csv")
    print("Wrote data/modeling/xgboost_v2_selected_config.json")
    print("Wrote reports/phase4c1_xgboost_tuning.md")


def write_report(fold_df, agg_df, winner_id, stage, params_by_id, best_iterations, final_n_estimators):
    md = []
    md.append("# XGBoost V2 - Hyperparameter Tuning (Chronological, TRAIN-only)\n")
    md.append(f"Generated by `training/xgboost/xgboost_tuning_v2.py`. **{len(agg_df)} candidates** "
              f"({len(ANCHOR_CONFIGS)} deterministic anchors + {len(agg_df) - len(ANCHOR_CONFIGS)} "
              f"`RandomState({RANDOM_STATE})` random-search draws) x {N_FOLDS} expanding-window folds.\n")

    md.append("## CV methodology: three-stage temporal separation\n")
    md.append("The outer folds are the **exact same** expanding-window folds used for Random Forest V2 "
              "(`data/modeling/random_forest_cv_folds_v2.csv`), reused unchanged so RF and XGBoost tuning are "
              "directly comparable. Within each outer fold, the fold's own training history is split "
              "chronologically **again**, so that the block used to decide *when to stop boosting* is never the "
              "block a candidate is *scored* on:\n")
    md.append("```\nINNER FIT             (earliest ~82.5% of fold-train)  -> model fitting (mirrored)\n"
              "    v  strictly earlier than\n"
              "INNER EARLY STOP      (latest ~17.5% of fold-train)    -> eval_set / best_iteration ONLY\n"
              "    v  strictly earlier than\n"
              "OUTER FOLD VALIDATION (the fold's validation block)     -> candidate scoring ONLY\n```\n")
    md.append("Asserted for every candidate x fold: `max(inner_fit.dt) < min(inner_early_stop.dt)` and "
              "`max(inner_early_stop.dt) < min(fold_validation.dt)`, with no exact-timestamp group crossing "
              "either boundary. **Without this, a fold-validation score would be optimistic** - the model would "
              "already have consulted that block to choose its stopping point.\n")
    md.append("Mirroring is applied to the **inner-fit block only**; preprocessing is fit on that mirrored "
              "inner-fit only; inner-early-stop and outer-fold-validation are never mirrored. Fold TRAIN metrics "
              "are computed on the **original unmirrored inner-fit rows**, keeping train-validation gaps "
              "comparable with LR/RF.\n")

    md.append("## Proof the main validation partition was absent\n")
    md.append("This script never opens `data/modeling/series_split_v1.csv`; it reads only the RF V2 fold "
              "manifest, which by construction contains **only TRAIN match_ids** (independently re-verified in "
              "`validation/validate_phase4c1.py`). The 1,419-match main validation partition therefore cannot have "
              "influenced candidate selection, early stopping, or `final_n_estimators`.\n")

    md.append("## Early-stopping API actually used\n")
    md.append(f"xgboost 3.4.0 does not expose `early_stopping_rounds` in `XGBClassifier.__init__`'s explicit "
              "signature or in `fit()`, but accepts it as a **constructor keyword argument** (absorbed via "
              "`**kwargs`), correctly populating `best_iteration`/`best_score`. Verified before use. Configuration: "
              f"`n_estimators={N_ESTIMATORS_CAP}` (cap), `early_stopping_rounds={EARLY_STOPPING_ROUNDS}`, "
              "`fit(..., eval_set=[(inner_early_stop_X, inner_early_stop_y)])`. `n_estimators` is therefore **not** "
              "an ordinary search dimension.\n")

    md.append("## Selection rule (fixed before the search ran)\n")
    md.append(SELECTION_RULE_TEXT + "\n")
    md.append("Accuracy is explicitly **not** an objective at any stage: the project consumes probabilities in a "
              "tournament simulator, so log loss leads and ROC-AUC follows.\n")
    md.append(f"`final_n_estimators = round(median(best_iteration + 1 across the {N_FOLDS} folds))`, also fixed "
              "in advance and derived from CV only.\n")
    md.append("**`xgb_v1_structure_reference` is included in the selection pool**, not excluded: XGBoost V1's "
              "structure is already moderately regularized, and chronological CV may legitimately conclude that "
              "the structure was fine and only the fixed 300-round count was suboptimal.\n")

    md.append("## Top 10 candidates by mean CV log loss\n")
    ranked = agg_df.sort_values("val_log_loss_mean").reset_index(drop=True)
    md.append("| rank | candidate_id | log loss (mean±std) | ROC-AUC (mean±std) | Brier | acc | lr | depth | "
              "mcw | sub | col | gamma | alpha | lambda | med trees | acc gap | AUC gap |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for i, r in ranked.head(10).iterrows():
        md.append(f"| {i+1} | {r['candidate_id']} | {r['val_log_loss_mean']:.4f}±{r['val_log_loss_std']:.4f} | "
                  f"{r['val_roc_auc_mean']:.4f}±{r['val_roc_auc_std']:.4f} | {r['val_brier_mean']:.4f} | "
                  f"{r['val_accuracy_mean']:.4f} | {r['learning_rate']} | {int(r['max_depth'])} | "
                  f"{int(r['min_child_weight'])} | {r['subsample']} | {r['colsample_bytree']} | {r['gamma']} | "
                  f"{r['reg_alpha']} | {r['reg_lambda']} | {r['median_effective_n_estimators']:.0f} | "
                  f"{r['train_val_acc_gap_mean']:+.4f} | {r['train_val_auc_gap_mean']:+.4f} |")
    md.append("")

    v1ref = agg_df[agg_df["candidate_id"] == "xgb_v1_structure_reference"].iloc[0]
    v1_rank = int(ranked.index[ranked["candidate_id"] == "xgb_v1_structure_reference"][0]) + 1
    md.append("## XGBoost V1's structure through the same CV harness\n")
    md.append(f"`xgb_v1_structure_reference` ranked **#{v1_rank}** of {len(agg_df)} by mean CV log loss: "
              f"log loss {v1ref['val_log_loss_mean']:.4f} ± {v1ref['val_log_loss_std']:.4f}, ROC-AUC "
              f"{v1ref['val_roc_auc_mean']:.4f} ± {v1ref['val_roc_auc_std']:.4f}, Brier "
              f"{v1ref['val_brier_mean']:.4f}, median effective trees "
              f"{v1ref['median_effective_n_estimators']:.0f}, mean train-val AUC gap "
              f"{v1ref['train_val_auc_gap_mean']:+.4f}.\n")

    # concept-level observations, computed rather than asserted
    md.append("## Observations\n")
    by_depth = agg_df.groupby("max_depth")["val_log_loss_mean"].mean().sort_index()
    md.append("Mean CV log loss by `max_depth` (lower is better):\n")
    md.append("| max_depth | mean CV log loss | mean train-val AUC gap |")
    md.append("|---|---|---|")
    gap_by_depth = agg_df.groupby("max_depth")["train_val_auc_gap_mean"].mean().sort_index()
    for d in by_depth.index:
        md.append(f"| {int(d)} | {by_depth[d]:.4f} | {gap_by_depth[d]:+.4f} |")
    md.append("")
    by_lr = agg_df.groupby("learning_rate")["median_effective_n_estimators"].mean().sort_index()
    md.append("Mean median effective boosting rounds by `learning_rate`:\n")
    md.append("| learning_rate | mean median effective trees |")
    md.append("|---|---|")
    for lr in by_lr.index:
        md.append(f"| {lr} | {by_lr[lr]:.0f} |")
    md.append("\nDifferences of a few thousandths in mean CV log loss across {} candidates should not be "
              "over-interpreted - that is precisely why the {} equivalence epsilon and the deterministic "
              "tie-break ladder exist.\n".format(len(agg_df), LOG_LOSS_EQUIVALENCE_EPSILON))

    w = params_by_id[winner_id]
    wagg = agg_df[agg_df["candidate_id"] == winner_id].iloc[0]
    md.append("## Selected configuration (FROZEN)\n")
    md.append(f"**`{winner_id}`**, selected via: {stage}.\n")
    md.append("```\n" + ", ".join(f"{k}={w[k]}" for k in SEARCH_KEYS) + "\n```\n")
    md.append(f"- mean CV log loss: {wagg['val_log_loss_mean']:.4f} ± {wagg['val_log_loss_std']:.4f}")
    md.append(f"- mean CV ROC-AUC: {wagg['val_roc_auc_mean']:.4f} ± {wagg['val_roc_auc_std']:.4f}")
    md.append(f"- mean CV Brier: {wagg['val_brier_mean']:.4f} | mean CV accuracy: {wagg['val_accuracy_mean']:.4f}")
    md.append(f"- mean train-validation AUC gap: {wagg['train_val_auc_gap_mean']:+.4f}")
    md.append(f"- `best_iteration` by fold: {best_iterations}")
    md.append(f"- **final_n_estimators = round(median({[b+1 for b in best_iterations]})) = {final_n_estimators}**\n")
    md.append("This configuration is now frozen in `data/modeling/xgboost_v2_selected_config.json`. Only after "
              "this freeze may the main validation partition be evaluated, exactly once, in "
              "`training/xgboost/train_xgboost_v2.py`.\n")

    (REPORTS / "phases" / "phase4c1_xgboost_tuning.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
