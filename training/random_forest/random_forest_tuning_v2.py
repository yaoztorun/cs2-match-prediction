"""
Phase 4B.1: Random Forest V2 hyperparameter search via expanding-window
chronological CV, using ONLY the 6,619-match TRAIN partition (through
data/modeling/random_forest_cv_folds_v2.csv). This script NEVER loads the
1,419-match main VALIDATION partition or the TEST partition - selection is
structurally blind to them.

Selection rule (documented here, BEFORE the search runs, and never altered
after seeing results):
    1. PRIMARY: lowest mean CV log loss.
    2. "Essentially equivalent" log loss = within LOGLOSS_EPSILON of the best
       candidate's mean log loss.
    3. Among essentially-equivalent candidates, SECONDARY: highest mean CV ROC-AUC.
    4. Remaining ties broken by LOWER complexity, ranked by
       (max_depth [None->999], -min_samples_leaf, -min_samples_split,
        max_features-as-fraction, n_estimators), ascending.

Writes:
    reports/tables/random_forest_tuning_v2.csv
    reports/phase4b1_random_forest_tuning.md
    data/modeling/random_forest_v2_selected_config.json
"""

import itertools
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss, brier_score_loss

import yaml

from _common import ROOT, REPORTS, INTERIM
from feature_engineering.preprocessing.preprocessing_random_forest_v1 import build_augmented_training_raw, fit_preprocessing, transform

CONFIG_PATH = ROOT / "config" / "features" / "series_features_v1.yaml"
FEATURES_PATH = ROOT / "data" / "features" / "series_features_v1.parquet"
CV_FOLDS_PATH = ROOT / "data" / "modeling" / "random_forest_cv_folds_v2.csv"
MODELING_DIR = ROOT / "data" / "modeling"
TABLES_DIR = REPORTS / "tables"

RANDOM_STATE = 42
N_RANDOM_CANDIDATES = 40
LOGLOSS_EPSILON = 0.002  # fixed BEFORE the search runs - never altered after seeing results
N_FOLDS = 4
N_FEATURES_TOTAL = 19  # for expressing max_features="sqrt" as a fraction when ranking complexity

SEARCH_SPACE = {
    "n_estimators": [300, 500, 800],
    "max_depth": [4, 6, 8, 10, 12, 16, None],
    "min_samples_leaf": [2, 5, 10, 20, 40],
    "min_samples_split": [5, 10, 20, 40],
    "max_features": ["sqrt", 0.5, 0.75, 1.0],
}
FIXED = {"bootstrap": True, "criterion": "gini"}

# Deterministic anchor configurations, chosen BEFORE running the search to
# guarantee representative complexity levels are evaluated regardless of what
# the random draw happens to sample - NOT chosen based on any observed CV
# result. Covers: shallow depth (4-6), moderate depth (8-12), min_samples_leaf
# around 10/20/40 (isolated from depth by leaving max_depth unrestricted for
# the leaf-focused anchors), sqrt max_features, and the exact unrestricted
# V1 configuration as an explicit reference point (intentionally OFF the
# tunable grid - min_samples_leaf=1/min_samples_split=2 are not in the
# search space - included only for comparison, per instruction).
ANCHOR_CONFIGS = [
    {"candidate_id": "anchor_shallow_depth4", "n_estimators": 300, "max_depth": 4,
     "min_samples_leaf": 20, "min_samples_split": 20, "max_features": "sqrt", **FIXED},
    {"candidate_id": "anchor_shallow_depth6", "n_estimators": 300, "max_depth": 6,
     "min_samples_leaf": 10, "min_samples_split": 10, "max_features": "sqrt", **FIXED},
    {"candidate_id": "anchor_moderate_depth8", "n_estimators": 500, "max_depth": 8,
     "min_samples_leaf": 10, "min_samples_split": 10, "max_features": "sqrt", **FIXED},
    {"candidate_id": "anchor_moderate_depth10", "n_estimators": 500, "max_depth": 10,
     "min_samples_leaf": 20, "min_samples_split": 20, "max_features": "sqrt", **FIXED},
    {"candidate_id": "anchor_moderate_depth12", "n_estimators": 300, "max_depth": 12,
     "min_samples_leaf": 10, "min_samples_split": 10, "max_features": "sqrt", **FIXED},
    {"candidate_id": "anchor_leaf10_unrestricted_depth", "n_estimators": 300, "max_depth": None,
     "min_samples_leaf": 10, "min_samples_split": 10, "max_features": "sqrt", **FIXED},
    {"candidate_id": "anchor_leaf20_unrestricted_depth", "n_estimators": 500, "max_depth": None,
     "min_samples_leaf": 20, "min_samples_split": 20, "max_features": "sqrt", **FIXED},
    {"candidate_id": "anchor_leaf40_strong_regularization", "n_estimators": 300, "max_depth": None,
     "min_samples_leaf": 40, "min_samples_split": 40, "max_features": "sqrt", **FIXED},
    {"candidate_id": "anchor_moderate_depth8_altfeatures", "n_estimators": 800, "max_depth": 8,
     "min_samples_leaf": 20, "min_samples_split": 20, "max_features": 0.5, **FIXED},
    {"candidate_id": "v1_reference_unrestricted", "n_estimators": 300, "max_depth": None,
     "min_samples_leaf": 1, "min_samples_split": 2, "max_features": "sqrt", **FIXED},
]


def build_random_candidates():
    keys = list(SEARCH_SPACE.keys())
    grid = list(itertools.product(*[SEARCH_SPACE[k] for k in keys]))  # deterministic order
    anchor_tuples = {tuple(a[k] for k in keys) for a in ANCHOR_CONFIGS if a["candidate_id"] != "v1_reference_unrestricted"}
    eligible = [t for t in grid if t not in anchor_tuples]

    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(len(eligible), size=N_RANDOM_CANDIDATES, replace=False)
    idx = sorted(idx.tolist())  # stable, deterministic order

    candidates = []
    for n, i in enumerate(idx, start=1):
        params = dict(zip(keys, eligible[i]))
        params.update(FIXED)
        params["candidate_id"] = f"random_{n:03d}"
        candidates.append(params)
    return candidates


def all_candidates():
    return list(ANCHOR_CONFIGS) + build_random_candidates()


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

    # mirror FOLD-TRAIN ONLY, fit preprocessing on THIS FOLD's augmented train ONLY
    augmented = build_augmented_training_raw(fold_train_raw)
    fold_params = fit_preprocessing(augmented, model_features)

    X_aug, _ = transform(augmented, fold_params)
    y_aug = augmented[target_col].to_numpy(dtype=float)

    # train metrics use the ORIGINAL UNMIRRORED fold-train rows (per correction #1),
    # not the augmented matrix used to fit the model
    X_train_orig, _ = transform(fold_train_raw, fold_params)
    y_train_orig = fold_train_raw[target_col].to_numpy(dtype=float)

    X_val, _ = transform(fold_val_raw, fold_params)
    y_val = fold_val_raw[target_col].to_numpy(dtype=float)

    rf_kwargs = {k: v for k, v in params.items() if k != "candidate_id"}
    model = RandomForestClassifier(**rf_kwargs, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_aug, y_aug)

    proba_train = model.predict_proba(X_train_orig)[:, 1]
    pred_train = model.predict(X_train_orig)
    proba_val = model.predict_proba(X_val)[:, 1]
    pred_val = model.predict(X_val)

    train_m = compute_metrics(y_train_orig, proba_train, pred_train)
    val_m = compute_metrics(y_val, proba_val, pred_val)
    return train_m, val_m


def complexity_key(params):
    depth = 999 if params["max_depth"] is None else params["max_depth"]
    leaf = -params["min_samples_leaf"]
    split = -params["min_samples_split"]
    mf = params["max_features"]
    mf_val = (N_FEATURES_TOTAL ** 0.5) / N_FEATURES_TOTAL if mf == "sqrt" else float(mf)
    return (depth, leaf, split, mf_val, params["n_estimators"])


def select_winner(agg_df, params_by_id):
    best_ll = agg_df["val_log_loss_mean"].min()
    tied = agg_df[agg_df["val_log_loss_mean"] <= best_ll + LOGLOSS_EPSILON].copy()
    stage = "primary (lowest mean CV log loss, unique)"
    if len(tied) > 1:
        best_auc = tied["val_roc_auc_mean"].max()
        tied2 = tied[tied["val_roc_auc_mean"] == best_auc]
        stage = "secondary (log-loss tie within epsilon, resolved by highest mean CV ROC-AUC)"
        if len(tied2) > 1:
            tied2 = tied2.copy()
            tied2["complexity_key"] = tied2["candidate_id"].map(lambda c: complexity_key(params_by_id[c]))
            tied2 = tied2.sort_values("complexity_key")
            stage = "tertiary (log-loss and ROC-AUC tied, resolved by lowest complexity)"
        tied = tied2
    winner_row = tied.iloc[0]
    return winner_row["candidate_id"], stage


def main():
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    model_features = cfg["model_features"]
    target_col = cfg["target"]

    features_df = pd.read_parquet(FEATURES_PATH, engine="fastparquet")
    cv_df = pd.read_csv(CV_FOLDS_PATH)
    # STRUCTURAL GUARANTEE: this script never reads series_split_v1.csv's
    # validation/test rows - only the pre-built CV fold file, which by
    # construction (training/random_forest/random_forest_cv_folds_v2.py) contains only
    # TRAIN match_ids.

    candidates = all_candidates()
    print(f"Evaluating {len(candidates)} candidates ({len(ANCHOR_CONFIGS)} anchors + "
          f"{len(candidates) - len(ANCHOR_CONFIGS)} random) x {N_FOLDS} folds = "
          f"{len(candidates) * N_FOLDS} total fits")

    params_by_id = {c["candidate_id"]: c for c in candidates}
    fold_rows = []
    for ci, params in enumerate(candidates, start=1):
        for fold in range(1, N_FOLDS + 1):
            train_m, val_m = evaluate_candidate_on_fold(params, fold, cv_df, features_df, model_features, target_col)
            fold_rows.append({
                "candidate_id": params["candidate_id"], "fold": fold, "row_type": "fold",
                "n_estimators": params["n_estimators"], "max_depth": params["max_depth"],
                "min_samples_leaf": params["min_samples_leaf"], "min_samples_split": params["min_samples_split"],
                "max_features": params["max_features"], "bootstrap": params["bootstrap"], "criterion": params["criterion"],
                "train_accuracy": train_m["accuracy"], "train_roc_auc": train_m["roc_auc"],
                "train_log_loss": train_m["log_loss"], "train_brier": train_m["brier"],
                "val_accuracy": val_m["accuracy"], "val_roc_auc": val_m["roc_auc"],
                "val_log_loss": val_m["log_loss"], "val_brier": val_m["brier"], "val_f1": val_m["f1"],
                "train_val_acc_gap": train_m["accuracy"] - val_m["accuracy"],
                "train_val_auc_gap": train_m["roc_auc"] - val_m["roc_auc"],
            })
        print(f"  [{ci}/{len(candidates)}] {params['candidate_id']} done")

    fold_df = pd.DataFrame(fold_rows)

    # ---------------- aggregate per candidate ----------------
    agg_records = []
    for cid, g in fold_df.groupby("candidate_id"):
        params = params_by_id[cid]
        agg_records.append({
            "candidate_id": cid, "fold": np.nan, "row_type": "aggregate",
            "n_estimators": params["n_estimators"], "max_depth": params["max_depth"],
            "min_samples_leaf": params["min_samples_leaf"], "min_samples_split": params["min_samples_split"],
            "max_features": params["max_features"], "bootstrap": params["bootstrap"], "criterion": params["criterion"],
            "val_accuracy_mean": g["val_accuracy"].mean(), "val_accuracy_std": g["val_accuracy"].std(ddof=0),
            "val_roc_auc_mean": g["val_roc_auc"].mean(), "val_roc_auc_std": g["val_roc_auc"].std(ddof=0),
            "val_log_loss_mean": g["val_log_loss"].mean(), "val_log_loss_std": g["val_log_loss"].std(ddof=0),
            "val_brier_mean": g["val_brier"].mean(), "val_brier_std": g["val_brier"].std(ddof=0),
            "val_f1_mean": g["val_f1"].mean(), "val_f1_std": g["val_f1"].std(ddof=0),
            "train_accuracy_mean": g["train_accuracy"].mean(), "train_roc_auc_mean": g["train_roc_auc"].mean(),
            "train_val_acc_gap_mean": g["train_val_acc_gap"].mean(), "train_val_auc_gap_mean": g["train_val_auc_gap"].mean(),
        })
    agg_df = pd.DataFrame(agg_records)

    full_df = pd.concat([fold_df, agg_df], ignore_index=True, sort=False)
    TABLES_DIR.mkdir(exist_ok=True, parents=True)
    full_df.to_csv(TABLES_DIR / "random_forest_tuning_v2.csv", index=False, encoding="utf-8")

    # ---------------- selection (never touches main validation) ----------------
    is_v1_ref = agg_df["candidate_id"] == "v1_reference_unrestricted"
    tunable_agg = agg_df[~is_v1_ref].reset_index(drop=True)
    winner_id, selection_stage = select_winner(tunable_agg, params_by_id)
    winner_params = params_by_id[winner_id]
    winner_agg = agg_df[agg_df["candidate_id"] == winner_id].iloc[0]

    print(f"\nSELECTED: {winner_id} via {selection_stage}")
    print(f"  params: {winner_params}")
    print(f"  mean CV log loss: {winner_agg['val_log_loss_mean']:.4f}, mean CV ROC-AUC: {winner_agg['val_roc_auc_mean']:.4f}")

    selected_config = {k: v for k, v in winner_params.items() if k != "candidate_id"}
    selected_payload = {
        "candidate_id": winner_id,
        "params": selected_config,
        "selection_stage": selection_stage,
        "log_loss_epsilon": LOGLOSS_EPSILON,
        "mean_cv_log_loss": float(winner_agg["val_log_loss_mean"]),
        "std_cv_log_loss": float(winner_agg["val_log_loss_std"]),
        "mean_cv_roc_auc": float(winner_agg["val_roc_auc_mean"]),
        "std_cv_roc_auc": float(winner_agg["val_roc_auc_std"]),
        "mean_cv_train_val_acc_gap": float(winner_agg["train_val_acc_gap_mean"]),
        "mean_cv_train_val_auc_gap": float(winner_agg["train_val_auc_gap_mean"]),
        "n_candidates_evaluated": len(candidates),
        "n_anchor_candidates": len(ANCHOR_CONFIGS),
        "n_random_candidates": len(candidates) - len(ANCHOR_CONFIGS),
        "cv_folds_artifact": "data/modeling/random_forest_cv_folds_v2.csv",
        "tuning_results_artifact": "reports/tables/random_forest_tuning_v2.csv",
    }
    (MODELING_DIR / "random_forest_v2_selected_config.json").write_text(
        json.dumps(selected_payload, indent=2), encoding="utf-8")

    write_report(agg_df, is_v1_ref, winner_id, selection_stage, params_by_id)

    print("\nWrote reports/tables/random_forest_tuning_v2.csv")
    print("Wrote data/modeling/random_forest_v2_selected_config.json")
    print("Wrote reports/phase4b1_random_forest_tuning.md")


def write_report(agg_df, is_v1_ref, winner_id, selection_stage, params_by_id):
    md = []
    md.append("# Random Forest V2 - Hyperparameter Tuning (Chronological CV)\n")
    md.append(f"Generated by `training/random_forest/random_forest_tuning_v2.py`. **{len(agg_df)} candidates** evaluated "
              f"({len(ANCHOR_CONFIGS)} deterministic anchors + {len(agg_df) - len(ANCHOR_CONFIGS)} "
              f"`RandomState({RANDOM_STATE})` random-search candidates over the grid) x {N_FOLDS} expanding-window "
              "CV folds built from the 6,619-match TRAIN partition only. **The 1,419-match main VALIDATION "
              "partition was never loaded by this script** - selection below is structurally blind to it.\n")

    md.append("## Selection rule (fixed before the search ran)\n")
    md.append(f"1. **Primary**: lowest mean CV log loss.")
    md.append(f"2. \"Essentially equivalent\" log loss = within **{LOGLOSS_EPSILON}** of the best candidate's mean "
              "log loss (fixed constant, not adjusted after seeing results).")
    md.append("3. Among essentially-equivalent candidates, **secondary**: highest mean CV ROC-AUC.")
    md.append("4. Remaining ties broken by **lower complexity**: ascending "
              "`(max_depth [None treated as 999], -min_samples_leaf, -min_samples_split, "
              "max_features-as-fraction, n_estimators)`.\n")
    md.append("The exact unrestricted V1 configuration (`min_samples_leaf=1, min_samples_split=2` - off the "
              "tunable grid) is included as a labeled reference candidate, evaluated through the identical CV "
              "harness, but is **excluded from the selection pool itself** (V2's purpose is to find a "
              "complexity-controlled alternative; V1 is the baseline being compared against, not a tuning "
              "option).\n")

    tunable = agg_df[~is_v1_ref].sort_values("val_log_loss_mean").reset_index(drop=True)
    md.append("## Top 10 configurations by mean CV log loss\n")
    md.append("| rank | candidate_id | log_loss (mean±std) | ROC-AUC (mean±std) | Brier (mean) | accuracy (mean) | "
              "n_estimators | max_depth | min_leaf | min_split | max_features | train-val acc gap | train-val AUC gap |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for i, r in tunable.head(10).iterrows():
        md.append(f"| {i+1} | {r['candidate_id']} | {r['val_log_loss_mean']:.4f}±{r['val_log_loss_std']:.4f} | "
                  f"{r['val_roc_auc_mean']:.4f}±{r['val_roc_auc_std']:.4f} | {r['val_brier_mean']:.4f} | "
                  f"{r['val_accuracy_mean']:.4f} | {int(r['n_estimators'])} | {r['max_depth']} | "
                  f"{int(r['min_samples_leaf'])} | {int(r['min_samples_split'])} | {r['max_features']} | "
                  f"{r['train_val_acc_gap_mean']:+.4f} | {r['train_val_auc_gap_mean']:+.4f} |")
    md.append("")

    v1ref = agg_df[is_v1_ref].iloc[0]
    md.append("## Reference: unrestricted V1 configuration through the same CV harness\n")
    md.append(f"- mean CV log loss: {v1ref['val_log_loss_mean']:.4f} ± {v1ref['val_log_loss_std']:.4f}")
    md.append(f"- mean CV ROC-AUC: {v1ref['val_roc_auc_mean']:.4f} ± {v1ref['val_roc_auc_std']:.4f}")
    md.append(f"- mean CV accuracy: {v1ref['val_accuracy_mean']:.4f}")
    md.append(f"- mean train-validation accuracy gap: {v1ref['train_val_acc_gap_mean']:+.4f}")
    md.append(f"- mean train-validation ROC-AUC gap: {v1ref['train_val_auc_gap_mean']:+.4f}\n")
    md.append("(These CV numbers are a fresh, independent measurement of V1's configuration under this phase's "
              "temporal CV harness - not the same as V1's original single-split validation numbers reported in "
              "Phase 4B, though both are expected to show the same qualitative severe-overfitting pattern.)\n")

    winner = params_by_id[winner_id]
    winner_agg = agg_df[agg_df["candidate_id"] == winner_id].iloc[0]
    md.append("## Selected configuration\n")
    md.append(f"**`{winner_id}`**, selected via: {selection_stage}.\n")
    md.append("```\n" + ", ".join(f"{k}={v}" for k, v in winner.items() if k != "candidate_id") + "\n```\n")
    md.append(f"- mean CV log loss: {winner_agg['val_log_loss_mean']:.4f} ± {winner_agg['val_log_loss_std']:.4f}")
    md.append(f"- mean CV ROC-AUC: {winner_agg['val_roc_auc_mean']:.4f} ± {winner_agg['val_roc_auc_std']:.4f}")
    md.append(f"- mean train-validation accuracy gap: {winner_agg['train_val_acc_gap_mean']:+.4f} "
              f"(vs. V1 reference's {v1ref['train_val_acc_gap_mean']:+.4f})")
    md.append(f"- mean train-validation ROC-AUC gap: {winner_agg['train_val_auc_gap_mean']:+.4f} "
              f"(vs. V1 reference's {v1ref['train_val_auc_gap_mean']:+.4f})\n")
    md.append("This configuration is now **frozen** (saved to "
              "`data/modeling/random_forest_v2_selected_config.json`) and will be refit on the full augmented "
              "TRAIN partition and evaluated exactly once on the main validation partition in "
              "`training/random_forest/train_random_forest_v2.py` - no further tuning iteration follows.\n")

    (REPORTS / "phases" / "phase4b1_random_forest_tuning.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
