"""
Phase 6B, brief sections 28-34 and 37: refit the frozen known-map
configurations on the FULL augmented TRAIN partition and evaluate them EXACTLY
ONCE on the 1,129-map main VALIDATION partition.

Preconditions, all checked before any data is loaded - this script refuses to
run otherwise:
    * data/modeling/map_random_forest_v1_selected_config.json exists
    * data/modeling/map_xgboost_v1_selected_config.json exists (with a frozen
      final_n_estimators)
    * data/modeling/map_ensemble_v1_config.json exists (weight frozen from
      TRAIN-only OOF)

This is the FIRST and ONLY time the main map validation partition is opened in
Phase 6B. Nothing is retuned afterwards: no hyperparameter, no feature, no
threshold, no ensemble weight, no preprocessing, no map category, no
calibration. No probability calibration is fitted and no prediction is
symmetrized - Phase 6B measures raw behaviour first.

TEST is never loaded. Cologne is structurally absent from the feature artifact.

Writes:
    models/map_random_forest_v1.joblib + map_random_forest_v1_metadata.json
    models/map_xgboost_v1.json         + map_xgboost_v1_metadata.json
    data/modeling/map_random_forest_preprocessing_v1.json
    data/modeling/map_xgboost_preprocessing_v1.json
    reports/tables/map_model_validation_metrics_v1.csv
    reports/tables/map_model_series_macro_metrics_v1.csv
    reports/tables/map_model_per_map_validation_v1.csv
    reports/tables/map_model_coverage_validation_v1.csv
    reports/figures/map_{rf,xgb,ensemble}_v1_{roc,calibration,probability_distribution}.png
    reports/phase6b_known_map_model_results.md
"""

import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
from xgboost import XGBClassifier

from _common import REPORTS
from map_modeling_common import (
    EXPECTED_TEST_N, EXPECTED_TRAIN_N, EXPECTED_VAL_N, MODELING_DIR, MODELS_DIR, RANDOM_STATE,
    SPLIT_PATH, assert_target_and_no_forbidden_columns, baseline_probabilities, compute_metrics,
    load_features, load_roles, series_macro_metrics,
)
from preprocessing_common_map_v2 import (
    EXPECTED_TRANSFORMED_FEATURES, assert_augmented_symmetry, build_augmented_training_raw, mirror_raw_rows,
)
import preprocessing_random_forest_map_v2 as prep_rf
import preprocessing_xgboost_map_v2 as prep_xgb

TABLES_DIR = REPORTS / "tables"
FIGURES_DIR = REPORTS / "figures"
RF_SELECTED_PATH = MODELING_DIR / "map_random_forest_v1_selected_config.json"
XGB_SELECTED_PATH = MODELING_DIR / "map_xgboost_v1_selected_config.json"
ENSEMBLE_PATH = MODELING_DIR / "map_ensemble_v1_config.json"
OOF_PATH = MODELING_DIR / "map_selected_models_oof_v1.parquet"

MODEL_LABELS = {"random_forest": "Random Forest", "xgboost": "XGBoost", "ensemble": "RF/XGB ensemble"}
FIG_STEMS = {"random_forest": "map_rf_v1", "xgboost": "map_xgb_v1", "ensemble": "map_ensemble_v1"}


def require(path, what):
    if not path.exists():
        raise RuntimeError(f"{path} does not exist - {what} must be frozen before validation is opened.")
    return json.loads(path.read_text(encoding="utf-8"))


def calibration_table(y, p, n_bins=10):
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    rows = []
    for i in range(n_bins):
        m = idx == i
        if not m.any():
            continue
        rows.append({"bin": f"[{edges[i]:.1f},{edges[i+1]:.1f})", "n": int(m.sum()),
                      "mean_predicted": float(p[m].mean()), "empirical_win_rate": float(y[m].mean())})
    return pd.DataFrame(rows)


def make_figures(y, p, model_key, auc):
    FIGURES_DIR.mkdir(exist_ok=True, parents=True)
    stem, label = FIG_STEMS[model_key], MODEL_LABELS[model_key]

    fpr, tpr, _ = roc_curve(y, p)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="chance")
    ax.set_xlabel("False Positive Rate"), ax.set_ylabel("True Positive Rate")
    ax.set_title(f"Map validation ROC - {label}")
    ax.legend(), fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{stem}_roc.png", dpi=150)
    plt.close(fig)

    cal = calibration_table(y, p)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
    ax.scatter(cal["mean_predicted"], cal["empirical_win_rate"], s=np.clip(cal["n"], 10, None), alpha=0.8)
    ax.set_xlabel("mean predicted P(team1 wins the map)")
    ax.set_ylabel("empirical team1 map win rate")
    ax.set_title(f"Map validation calibration (10 bins) - {label}")
    ax.set_xlim(0, 1), ax.set_ylim(0, 1), ax.legend(), fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{stem}_calibration.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(p, bins=30, range=(0, 1))
    ax.set_xlabel("P(team1 wins the map)"), ax.set_ylabel("count")
    ax.set_title(f"Map validation predicted-probability distribution - {label}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{stem}_probability_distribution.png", dpi=150)
    plt.close(fig)
    return cal


def main():
    rf_sel = require(RF_SELECTED_PATH, "the Random Forest configuration")
    xgb_sel = require(XGB_SELECTED_PATH, "the XGBoost configuration")
    ens_cfg = require(ENSEMBLE_PATH, "the ensemble weight")
    weight_rf = float(ens_cfg["weight_rf"])
    final_n_estimators = int(xgb_sel["final_n_estimators"])
    rf_params = dict(rf_sel["params"])
    if rf_params.get("max_depth") in ("None", None):
        rf_params["max_depth"] = None
    xgb_params = dict(xgb_sel["params"])
    print(f"Frozen RF:  {rf_sel['selected_candidate_id']} {rf_params}")
    print(f"Frozen XGB: {xgb_sel['selected_candidate_id']} {xgb_params} n_estimators={final_n_estimators}")
    print(f"Frozen ensemble weight: w_rf={weight_rf}")

    roles = load_roles()
    target = roles["target"]
    features = load_features()
    assert_target_and_no_forbidden_columns(features, roles)

    split = pd.read_csv(SPLIT_PATH)
    df = features.merge(split[["match_id", "game_id", "split"]], on=["match_id", "game_id"],
                         how="left", validate="one_to_one")
    assert df["split"].notna().all()
    train_raw = df[df["split"] == "train"].reset_index(drop=True)
    val_raw = df[df["split"] == "validation"].reset_index(drop=True)
    test_count = int((df["split"] == "test").sum())
    # TEST IS DELIBERATELY NEVER MATERIALIZED AS A DATAFRAME HERE.
    assert len(train_raw) == EXPECTED_TRAIN_N, len(train_raw)
    assert len(val_raw) == EXPECTED_VAL_N, len(val_raw)
    assert test_count == EXPECTED_TEST_N, test_count
    assert set(train_raw["match_id"]).isdisjoint(set(val_raw["match_id"])), "a match_id crosses the partition"
    print(f"Split: train={len(train_raw)} validation={len(val_raw)} test={test_count} (sealed, never loaded)")

    # ---------------- mirror FULL TRAIN only ----------------
    aug_train = build_augmented_training_raw(train_raw, roles)
    assert len(aug_train) == 2 * len(train_raw) == 15524
    assert abs(float(aug_train[target].mean()) - 0.5) < 1e-9
    assert_augmented_symmetry(aug_train, roles)
    print(f"TRAIN: {len(train_raw):,} unique historical maps -> {len(aug_train):,} augmented training "
          "OBSERVATIONS after mirroring (never described as that many maps)")

    y_train = train_raw[target].to_numpy(dtype=float)
    y_val = val_raw[target].to_numpy(dtype=float)
    y_aug = aug_train[target].to_numpy(dtype=float)

    # ---------------- Random Forest ----------------
    rf_prep = prep_rf.fit_preprocessing(aug_train, roles)
    prep_rf.save_preprocessing(rf_prep, MODELING_DIR / "map_random_forest_preprocessing_v1.json")
    X_rf_aug, feature_names = prep_rf.transform(aug_train, rf_prep, roles)
    X_rf_train, _ = prep_rf.transform(train_raw, rf_prep, roles)
    X_rf_val, _ = prep_rf.transform(val_raw, rf_prep, roles)
    assert X_rf_aug.shape[1] == EXPECTED_TRANSFORMED_FEATURES

    rf = RandomForestClassifier(**rf_params, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_rf_aug, y_aug)
    MODELS_DIR.mkdir(exist_ok=True, parents=True)
    joblib.dump(rf, MODELS_DIR / "map_random_forest_v1.joblib")

    # ---------------- XGBoost ----------------
    xgb_prep = prep_xgb.fit_preprocessing(aug_train, roles)
    prep_xgb.save_preprocessing(xgb_prep, MODELING_DIR / "map_xgboost_preprocessing_v1.json")
    X_xgb_aug, _ = prep_xgb.transform(aug_train, xgb_prep, roles)
    X_xgb_train, _ = prep_xgb.transform(train_raw, xgb_prep, roles)
    X_xgb_val, _ = prep_xgb.transform(val_raw, xgb_prep, roles)

    xgb = XGBClassifier(n_estimators=final_n_estimators, **xgb_params, **xgb_sel["fixed_params"])
    xgb.fit(X_xgb_aug, y_aug, verbose=False)   # NO eval_set, NO early stopping
    xgb.save_model(str(MODELS_DIR / "map_xgboost_v1.json"))

    # ---------------- save/load prediction parity ----------------
    rf_reloaded = joblib.load(MODELS_DIR / "map_random_forest_v1.joblib")
    xgb_reloaded = XGBClassifier()
    xgb_reloaded.load_model(str(MODELS_DIR / "map_xgboost_v1.json"))
    p_rf_val = rf.predict_proba(X_rf_val)[:, 1]
    p_xgb_val = xgb.predict_proba(X_xgb_val)[:, 1]
    rf_parity = float(np.abs(p_rf_val - rf_reloaded.predict_proba(X_rf_val)[:, 1]).max())
    xgb_parity = float(np.abs(p_xgb_val - xgb_reloaded.predict_proba(X_xgb_val)[:, 1]).max())
    # Tolerance-safe, not bit-exact: a joblib/native round-trip can reorder the
    # final float summation, so differences at machine epsilon are expected and
    # meaningless. Anything above 1e-9 would indicate a real serialization bug.
    assert rf_parity < 1e-9, f"RF reload changed predictions (max abs diff {rf_parity})"
    assert xgb_parity < 1e-9, f"XGB reload changed predictions (max abs diff {xgb_parity})"
    print(f"Save/load parity: RF max|dP|={rf_parity:.2e}, XGB max|dP|={xgb_parity:.2e}")

    # ================= MAIN VALIDATION - OPENED EXACTLY ONCE =================
    p_ens_val = weight_rf * p_rf_val + (1 - weight_rf) * p_xgb_val
    p_rf_train = rf.predict_proba(X_rf_train)[:, 1]
    p_xgb_train = xgb.predict_proba(X_xgb_train)[:, 1]
    p_ens_train = weight_rf * p_rf_train + (1 - weight_rf) * p_xgb_train

    val_probs = {"random_forest": p_rf_val, "xgboost": p_xgb_val, "ensemble": p_ens_val}
    train_probs = {"random_forest": p_rf_train, "xgboost": p_xgb_train, "ensemble": p_ens_train}
    for k, p in val_probs.items():
        assert np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all(), f"{k}: bad probabilities"

    # ---------------- headline metrics + baselines ----------------
    metric_rows, calib_tables = [], {}
    for key, p in val_probs.items():
        m = compute_metrics(y_val, p, with_confusion=True)
        tm = compute_metrics(y_train, train_probs[key])
        metric_rows.append({
            "model": key, "split": "validation", **{k: v for k, v in m.items() if k != "confusion_matrix"},
            "confusion_matrix": json.dumps(m["confusion_matrix"]),
            "train_accuracy": tm["accuracy"], "train_roc_auc": tm["roc_auc"], "train_log_loss": tm["log_loss"],
            "train_val_auc_gap": tm["roc_auc"] - m["roc_auc"],
        })
        calib_tables[key] = make_figures(y_val, p, key, m["roc_auc"])

    baseline_rows = []
    for kind, label in [("half", "0.5 constant"), ("overall_elo", "overall ELO"), ("map_elo", "map ELO")]:
        p = baseline_probabilities(val_raw, kind)
        baseline_rows.append({"model": f"baseline_{kind}", "split": "validation", **{
            k: v for k, v in compute_metrics(y_val, p, with_confusion=True).items() if k != "confusion_matrix"}})
    val_metrics = pd.DataFrame(metric_rows + baseline_rows)
    TABLES_DIR.mkdir(exist_ok=True, parents=True)
    val_metrics.to_csv(TABLES_DIR / "map_model_validation_metrics_v1.csv", index=False, encoding="utf-8")

    # ---------------- series-macro diagnostics (OOF + validation) ----------------
    oof = pd.read_parquet(OOF_PATH, engine="fastparquet")
    oof_p = {"random_forest": oof["p_rf"].to_numpy(), "xgboost": oof["p_xgb"].to_numpy()}
    oof_p["ensemble"] = weight_rf * oof_p["random_forest"] + (1 - weight_rf) * oof_p["xgboost"]
    macro_rows = []
    for key in MODEL_LABELS:
        macro_rows.append({"model": key, "population": "pooled_train_oof",
                            **series_macro_metrics(oof["match_id"].to_numpy(),
                                                    oof["y_true"].to_numpy(dtype=float), oof_p[key])})
        macro_rows.append({"model": key, "population": "validation",
                            **series_macro_metrics(val_raw["match_id"].to_numpy(), y_val, val_probs[key])})
    for kind, _ in [("half", 0), ("overall_elo", 0), ("map_elo", 0)]:
        macro_rows.append({"model": f"baseline_{kind}", "population": "validation",
                            **series_macro_metrics(val_raw["match_id"].to_numpy(), y_val,
                                                    baseline_probabilities(val_raw, kind))})
    series_macro = pd.DataFrame(macro_rows)
    series_macro.to_csv(TABLES_DIR / "map_model_series_macro_metrics_v1.csv", index=False, encoding="utf-8")

    # ---------------- per-map validation diagnostics ----------------
    per_map_rows = []
    for map_name, g in val_raw.groupby("map_name"):
        mask = (val_raw["map_name"] == map_name).to_numpy()
        yy = y_val[mask]
        both_classes = len(set(yy.tolist())) > 1
        for key, p in val_probs.items():
            m = compute_metrics(yy, p[mask])
            per_map_rows.append({
                "map_name": map_name, "model": key, "n": int(mask.sum()),
                "small_sample": bool(mask.sum() < 50),
                "both_classes_present": both_classes,
                "accuracy": m["accuracy"], "log_loss": m["log_loss"], "brier": m["brier"],
                "roc_auc": m["roc_auc"] if both_classes else float("nan"),
            })
    per_map = pd.DataFrame(per_map_rows).sort_values(["model", "n"], ascending=[True, False])
    per_map.to_csv(TABLES_DIR / "map_model_per_map_validation_v1.csv", index=False, encoding="utf-8")

    # ---------------- coverage subgroups (predefined, descriptive only) ----------------
    subgroups = {
        "A_both_teams_have_map_history": (val_raw["both_teams_have_map_history"] == 1).to_numpy(),
        "B_both_teams_have_5_map_matches": (val_raw["both_teams_have_5_map_matches"] == 1).to_numpy(),
        "C_roster_form_players_min_ge_5": (val_raw["roster_form_players_min"] >= 5).to_numpy(),
        "D_map_cold_start_at_least_one_side": (val_raw["both_teams_have_map_history"] == 0).to_numpy(),
    }
    cov_rows = []
    for name, mask in subgroups.items():
        if mask.sum() == 0:
            continue
        for key, p in val_probs.items():
            m = compute_metrics(y_val[mask], p[mask])
            cov_rows.append({"subgroup": name, "model": key, "n": int(mask.sum()),
                              "pct_of_validation": 100.0 * mask.sum() / len(val_raw),
                              "accuracy": m["accuracy"], "roc_auc": m["roc_auc"],
                              "log_loss": m["log_loss"], "brier": m["brier"]})
    coverage = pd.DataFrame(cov_rows)
    coverage.to_csv(TABLES_DIR / "map_model_coverage_validation_v1.csv", index=False, encoding="utf-8")

    # ---------------- side-symmetry diagnostic (measured, never corrected) ----------------
    val_mirrored = mirror_raw_rows(val_raw, roles)
    X_rf_val_m, _ = prep_rf.transform(val_mirrored, rf_prep, roles)
    X_xgb_val_m, _ = prep_xgb.transform(val_mirrored, xgb_prep, roles)
    mirror_probs = {"random_forest": rf.predict_proba(X_rf_val_m)[:, 1],
                     "xgboost": xgb.predict_proba(X_xgb_val_m)[:, 1]}
    mirror_probs["ensemble"] = weight_rf * mirror_probs["random_forest"] + \
        (1 - weight_rf) * mirror_probs["xgboost"]
    symmetry = {}
    for key in MODEL_LABELS:
        err = np.abs(val_probs[key] - (1 - mirror_probs[key]))
        symmetry[key] = {"mean": float(err.mean()), "median": float(np.median(err)),
                          "p95": float(np.percentile(err, 95)), "max": float(err.max())}
    print("Side-symmetry (validation):", json.dumps(symmetry, indent=2))

    # ---------------- model metadata ----------------
    common_meta = {
        "task": "known_map", "target": target,
        "feature_config": "config/map_features_v2_rich.yaml",
        "features_artifact": "data/features/map_features_v2_rich.parquet",
        "split_manifest": "data/modeling/map_split_v1.csv",
        "cv_folds_artifact": "data/modeling/map_cv_folds_v1.csv",
        "raw_predictive_inputs": 95, "transformed_feature_count": EXPECTED_TRANSFORMED_FEATURES,
        "original_train_maps": int(len(train_raw)), "augmented_train_observations": int(len(aug_train)),
        "training_cutoff": str(train_raw["series_datetime"].max()),
        "random_state": RANDOM_STATE,
        "test_status": "SEALED - never loaded in Phase 6B",
        "cologne_status": "UNTOUCHED - structurally absent from map_features_v2_rich.parquet",
        "calibration_applied": False, "predictions_symmetrized": False,
    }
    rf_val = val_metrics[(val_metrics.model == "random_forest") & (val_metrics.split == "validation")].iloc[0]
    xgb_val = val_metrics[(val_metrics.model == "xgboost") & (val_metrics.split == "validation")].iloc[0]
    (MODELS_DIR / "map_random_forest_v1_metadata.json").write_text(json.dumps({
        "model_type": "random_forest_classifier", "implementation": "sklearn.ensemble.RandomForestClassifier",
        "selected_candidate_id": rf_sel["selected_candidate_id"], "params": rf_params,
        "selection_stage": rf_sel["selection_stage"], "cv_mean_log_loss": rf_sel["cv_mean_log_loss"],
        "cv_mean_roc_auc": rf_sel["cv_mean_roc_auc"],
        "preprocessing_artifact": "data/modeling/map_random_forest_preprocessing_v1.json",
        "validation_metrics": {k: (float(rf_val[k]) if k != "confusion_matrix" else rf_val[k])
                                for k in ["n", "accuracy", "precision", "recall", "f1", "roc_auc",
                                          "log_loss", "brier", "confusion_matrix"]},
        "validation_symmetry_error": symmetry["random_forest"],
        "reload_prediction_max_abs_diff": rf_parity, **common_meta,
    }, indent=2, default=str), encoding="utf-8")
    (MODELS_DIR / "map_xgboost_v1_metadata.json").write_text(json.dumps({
        "model_type": "xgboost_classifier", "implementation": "xgboost.XGBClassifier",
        "selected_candidate_id": xgb_sel["selected_candidate_id"], "params": xgb_params,
        "fixed_params": xgb_sel["fixed_params"], "final_n_estimators": final_n_estimators,
        "final_n_estimators_rule": xgb_sel["final_n_estimators_rule"],
        "best_iterations_by_fold": xgb_sel["best_iterations_by_fold"],
        "early_stopping_used_in_final_fit": False, "eval_set_used_in_final_fit": False,
        "selection_stage": xgb_sel["selection_stage"], "cv_mean_log_loss": xgb_sel["cv_mean_log_loss"],
        "cv_mean_roc_auc": xgb_sel["cv_mean_roc_auc"],
        "preprocessing_artifact": "data/modeling/map_xgboost_preprocessing_v1.json",
        "validation_metrics": {k: (float(xgb_val[k]) if k != "confusion_matrix" else xgb_val[k])
                                for k in ["n", "accuracy", "precision", "recall", "f1", "roc_auc",
                                          "log_loss", "brier", "confusion_matrix"]},
        "validation_symmetry_error": symmetry["xgboost"],
        "reload_prediction_max_abs_diff": xgb_parity, **common_meta,
    }, indent=2, default=str), encoding="utf-8")

    write_report(rf_sel, xgb_sel, ens_cfg, val_metrics, series_macro, per_map, coverage, calib_tables,
                  symmetry, val_probs, y_val, train_raw, val_raw, aug_train, test_count)

    print("\nWrote models/map_random_forest_v1.joblib + metadata")
    print("Wrote models/map_xgboost_v1.json + metadata")
    print("Wrote data/modeling/map_{random_forest,xgboost}_preprocessing_v1.json")
    print("Wrote reports/tables/map_model_{validation_metrics,series_macro_metrics,per_map_validation,"
          "coverage_validation}_v1.csv")
    print("Wrote reports/figures/map_{rf,xgb,ensemble}_v1_{roc,calibration,probability_distribution}.png")
    print("Wrote reports/phase6b_known_map_model_results.md")


def write_report(rf_sel, xgb_sel, ens_cfg, val_metrics, series_macro, per_map, coverage, calib_tables,
                  symmetry, val_probs, y_val, train_raw, val_raw, aug_train, test_count):
    oof_metrics = pd.read_csv(TABLES_DIR / "map_model_oof_metrics_v1.csv")
    pooled_oof = oof_metrics[oof_metrics["population"] == "pooled_train_oof"].set_index("model")
    vm = val_metrics[val_metrics["split"] == "validation"].set_index("model")

    md = []
    md.append("# Phase 6B - Known-Map Model Results\n")
    md.append("## What this model predicts\n")
    md.append("Given Team A, Team B, the series format (BO1/BO3/BO5) and a **user-selected map**, estimate "
              "P(Team A wins that map), using only information available before the series starts. The selected "
              "map is legitimate input here because the user supplies it.\n")
    md.append("**This is a different prediction task from the pre-veto series model.** The earlier ~61% figures "
              "were SERIES-winner accuracies; everything below is MAP-outcome accuracy. A map accuracy and a "
              "series accuracy cannot be differenced or described as one being N percentage points better than "
              "the other - they are different targets. Converting map probabilities into BO3/BO5 series "
              "probabilities is an application-level simulation step that is not part of this phase.\n")

    md.append("## Data\n")
    md.append(f"- TRAIN: **{len(train_raw):,}** unique historical maps -> {len(aug_train):,} augmented training "
              "observations after side mirroring (mirrored rows are re-labellings of already-counted maps, never "
              "additional maps)")
    md.append(f"- VALIDATION: **{len(val_raw):,}** maps - opened **exactly once**, in this script, after every "
              "configuration was frozen")
    md.append(f"- TEST: **{test_count:,}** maps - **SEALED**, never loaded")
    md.append("- Maps of one series never cross a partition (enforced by `data/modeling/map_split_v1.csv`)\n")

    md.append("## Frozen configurations\n")
    md.append(f"**Random Forest** `{rf_sel['selected_candidate_id']}`: "
              + ", ".join(f"`{k}={v}`" for k, v in rf_sel["params"].items()) + "\n")
    md.append(f"**XGBoost** `{xgb_sel['selected_candidate_id']}`: "
              + ", ".join(f"`{k}={v}`" for k, v in xgb_sel["params"].items())
              + f", `n_estimators={xgb_sel['final_n_estimators']}` "
              f"(= {xgb_sel['final_n_estimators_rule']}, from fold best_iterations "
              f"{xgb_sel['best_iterations_by_fold']}), fitted with **no eval_set and no early stopping**\n")
    md.append(f"**Ensemble** `p = {ens_cfg['weight_rf']} * p_rf + {ens_cfg['weight_xgb']} * p_xgb`, weight "
              f"selected from TRAIN-only OOF via {ens_cfg['selection_stage']}\n")

    md.append("## Main map validation - map-level metrics (opened once)\n")
    md.append("| model | n | accuracy | precision | recall | F1 | ROC-AUC | log loss | Brier | confusion "
              "[[TN,FP],[FN,TP]] |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    for key in list(MODEL_LABELS) + ["baseline_half", "baseline_overall_elo", "baseline_map_elo"]:
        r = vm.loc[key]
        cm = r["confusion_matrix"] if isinstance(r.get("confusion_matrix"), str) else "-"
        name = MODEL_LABELS.get(key, key.replace("baseline_", "baseline: "))
        md.append(f"| {name} | {int(r['n'])} | {r['accuracy']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | "
                  f"{r['f1']:.4f} | {r['roc_auc']:.4f} | {r['log_loss']:.4f} | {r['brier']:.4f} | {cm} |")
    md.append("")
    md.append("Train-validation gaps (overfitting control):\n")
    md.append("| model | train accuracy | validation accuracy | train ROC-AUC | validation ROC-AUC | AUC gap |")
    md.append("|---|---|---|---|---|---|")
    for key in MODEL_LABELS:
        r = vm.loc[key]
        md.append(f"| {MODEL_LABELS[key]} | {r['train_accuracy']:.4f} | {r['accuracy']:.4f} | "
                  f"{r['train_roc_auc']:.4f} | {r['roc_auc']:.4f} | {r['train_val_auc_gap']:+.4f} |")
    md.append("")

    md.append("## TRAIN-only out-of-fold vs. main validation\n")
    md.append("| model | OOF log loss | validation log loss | OOF ROC-AUC | validation ROC-AUC | OOF accuracy | "
              "validation accuracy |")
    md.append("|---|---|---|---|---|---|---|")
    for key in MODEL_LABELS:
        o, v = pooled_oof.loc[key], vm.loc[key]
        md.append(f"| {MODEL_LABELS[key]} | {o['log_loss']:.4f} | {v['log_loss']:.4f} | {o['roc_auc']:.4f} | "
                  f"{v['roc_auc']:.4f} | {o['accuracy']:.4f} | {v['accuracy']:.4f} |")
    md.append("\nThe OOF column is development evidence over the four TRAIN folds; the validation column is a "
              "single held-out period. They measure different things and are shown side by side only to expose "
              "how much the picture moved.\n")

    md.append("## Series-macro diagnostics\n")
    md.append("Multiple maps of one series are dependent observations. Map-level metrics above remain PRIMARY "
              "(the task is map prediction); these average each series' own per-map mean and then average those "
              "equally across match_ids, so BO3/BO5 series do not receive disproportionate weight. No per-series "
              "ROC-AUC is computed - most series have too few maps, often of a single class, for it to be "
              "meaningful.\n")
    md.append("| model | population | n series | series-macro log loss | series-macro Brier | "
              "series-macro accuracy |")
    md.append("|---|---|---|---|---|---|")
    for _, r in series_macro.iterrows():
        name = MODEL_LABELS.get(r["model"], r["model"].replace("baseline_", "baseline: "))
        md.append(f"| {name} | {r['population']} | {int(r['n_series'])} | {r['series_macro_log_loss']:.4f} | "
                  f"{r['series_macro_brier']:.4f} | {r['series_macro_accuracy']:.4f} |")
    md.append("")

    md.append("## Per-map validation diagnostics\n")
    md.append("Predefined before validation was opened. Small samples are marked; ROC-AUC is reported only where "
              "both target classes are present. Low-history maps such as Train and Overpass are shown as-is and "
              "are **not** removed, and nothing here is used to retune anything.\n")
    for key in MODEL_LABELS:
        sub = per_map[per_map["model"] == key].sort_values("n", ascending=False)
        md.append(f"**{MODEL_LABELS[key]}**\n")
        md.append("| map | n | small sample | accuracy | log loss | Brier | ROC-AUC |")
        md.append("|---|---|---|---|---|---|---|")
        for _, r in sub.iterrows():
            auc = "n/a (single class)" if not np.isfinite(r["roc_auc"]) else f"{r['roc_auc']:.4f}"
            md.append(f"| {r['map_name']} | {int(r['n'])} | {'yes' if r['small_sample'] else 'no'} | "
                      f"{r['accuracy']:.4f} | {r['log_loss']:.4f} | {r['brier']:.4f} | {auc} |")
        md.append("")

    md.append("## Coverage diagnostics (descriptive only)\n")
    md.append("Subgroups predefined before validation was opened. No subgroup-specific model is built in "
              "Phase 6B.\n")
    md.append("| subgroup | model | n | % of validation | accuracy | ROC-AUC | log loss | Brier |")
    md.append("|---|---|---|---|---|---|---|---|")
    for _, r in coverage.iterrows():
        md.append(f"| {r['subgroup']} | {MODEL_LABELS[r['model']]} | {int(r['n'])} | "
                  f"{r['pct_of_validation']:.1f}% | {r['accuracy']:.4f} | {r['roc_auc']:.4f} | "
                  f"{r['log_loss']:.4f} | {r['brier']:.4f} |")
    md.append("")

    md.append("## Calibration (diagnostic only - no calibration is fitted in Phase 6B)\n")
    md.append("Figures: `reports/figures/map_{rf,xgb,ensemble}_v1_calibration.png`, plus ROC curves and "
              "probability-distribution histograms under the same stems. No isotonic or Platt calibration is "
              "applied.\n")
    for key in MODEL_LABELS:
        md.append(f"**{MODEL_LABELS[key]}** reliability (10 bins):\n")
        md.append("| bin | n | mean predicted | empirical map win rate |")
        md.append("|---|---|---|---|")
        for _, r in calib_tables[key].iterrows():
            md.append(f"| {r['bin']} | {int(r['n'])} | {r['mean_predicted']:.3f} | "
                      f"{r['empirical_win_rate']:.3f} |")
        md.append("")
    md.append("Predicted-probability spread on validation:\n")
    md.append("| model | min | p5 | p25 | median | p75 | p95 | max |")
    md.append("|---|---|---|---|---|---|---|---|")
    for key in MODEL_LABELS:
        q = np.percentile(val_probs[key], [0, 5, 25, 50, 75, 95, 100])
        md.append(f"| {MODEL_LABELS[key]} | " + " | ".join(f"{v:.3f}" for v in q) + " |")
    md.append("")

    md.append("## Side-symmetry diagnostic\n")
    md.append("Each validation matchup is scored as A vs B on map X and again as B vs A on the same map X; the "
              "error is `|P(A wins) - (1 - P(B wins))|`. **Measured, not corrected** - Phase 6B deliberately "
              "records raw behaviour rather than symmetrizing predictions.\n")
    md.append("| model | mean | median | p95 | max |")
    md.append("|---|---|---|---|---|")
    for key in MODEL_LABELS:
        s = symmetry[key]
        md.append(f"| {MODEL_LABELS[key]} | {s['mean']:.4f} | {s['median']:.4f} | {s['p95']:.4f} | "
                  f"{s['max']:.4f} |")
    md.append("")

    md.append("## Verdict\n")
    best_ll = min(MODEL_LABELS, key=lambda k: vm.loc[k, "log_loss"])
    base_map_elo = vm.loc["baseline_map_elo"]
    base_overall = vm.loc["baseline_overall_elo"]
    b = vm.loc[best_ll]
    md.append(f"On this single held-out validation period, **{MODEL_LABELS[best_ll]}** has the lowest map-level "
              f"log loss ({b['log_loss']:.4f} vs {base_map_elo['log_loss']:.4f} for the map-ELO baseline and "
              f"{base_overall['log_loss']:.4f} for the overall-ELO baseline), with ROC-AUC {b['roc_auc']:.4f} "
              f"against {base_map_elo['roc_auc']:.4f} / {base_overall['roc_auc']:.4f} respectively.\n")
    md.append("No final project model is declared. The internal TEST partition is the final unbiased internal "
              "evaluation and remains sealed; the external Cologne protocol follows after that. No "
              "hyperparameter, feature, threshold, ensemble weight, preprocessing rule, map category or "
              "calibration was changed after these numbers were seen.\n")

    md.append("## Status\n")
    md.append("- **MAIN MAP VALIDATION = USED ONCE AFTER FREEZE**\n- **TEST = SEALED**\n"
              "- **COLOGNE = UNTOUCHED**\n- **NO POST-VALIDATION RETUNING**\n- **SRC = UNCHANGED**\n")

    (REPORTS / "phase6b_known_map_model_results.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
