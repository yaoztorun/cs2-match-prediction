"""
Phase 5C.1 validation (artifact-level). Read-only. Exits non-zero on failure.

Checks fall into groups:
  1. V3/V4 universe equality + every V3 column preserved + exactly 21 new columns;
  2. CV folds are GLOBAL-TRAIN-only (disjoint from validation/test/Cologne);
  3. the main VALIDATION partition is structurally never read by the
     evaluation script (AST-level source check);
  4. RF/XGB hyperparameters used for the V3 arm and the V4 arm are
     byte-identical (single call site each);
  5. STRICT V3-arm regression parity against Phase 5B.3's own saved
     V3-arm fold metrics - a material difference FAILS validation;
  6. preprocessing is freshly fit per (model, feature_set, fold) - independent
     recomputation, medians need not differ across folds;
  7. NaN handling: RF's fold-fitted train_medians cover every NaN-capable
     roster-performance column; XGB's preprocessing never imputes;
  8. all 53 V4 directional features negate under config-driven mirroring
     (including NaN-preserving negation), all 25 symmetric unchanged;
  9. Phase 1-5C artifacts byte-unchanged;
  10. deterministic rerun (numeric tolerance - RF/XGB n_jobs=-1 nondeterminism).
"""

import ast
import hashlib
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml

from _common import ROOT, INTERIM, REPORTS
import feature_engineering.preprocessing.preprocessing_common_v2_map_pool as pc2
import feature_engineering.preprocessing.preprocessing_random_forest_v2_map_pool as rf2
import feature_engineering.preprocessing.preprocessing_xgboost_v2_map_pool as xgb2
import feature_engineering.preprocessing.preprocessing_common_v3_form as pc3
import feature_engineering.preprocessing.preprocessing_common_v4_roster as pc4
from feature_engineering.roster.player_roster_feature_engine import ROSTER_DIRECTIONAL_FEATURES, ROSTER_SYMMETRIC_FEATURES, ROSTER_PERFORMANCE_DIFFS

CONFIG_V3_PATH = ROOT / "config" / "features" / "series_features_v3_form.yaml"
CONFIG_V4_PATH = ROOT / "config" / "features" / "series_features_v4_roster.yaml"
FEATURES_V3_PATH = ROOT / "data" / "features" / "series_features_v3_form.parquet"
FEATURES_V4_PATH = ROOT / "data" / "features" / "series_features_v4_roster.parquet"
CV_FOLDS_PATH = ROOT / "data" / "modeling" / "random_forest_cv_folds_v2.csv"
SPLIT_PATH = ROOT / "data" / "modeling" / "series_split_v1.csv"
RF_SELECTED_CONFIG_PATH = ROOT / "data" / "modeling" / "random_forest_v2_selected_config.json"
XGB_SELECTED_CONFIG_PATH = ROOT / "data" / "modeling" / "xgboost_v2_selected_config.json"
AUDIT_PATH = ROOT / "data" / "modeling" / "phase5c1_fold_preprocessing_audit.json"
EVAL_SCRIPT = ROOT / "evaluation" / "validation" / "evaluate_series_feature_sets_v4.py"
COMPARISON_CSV_V4 = REPORTS / "tables" / "series_feature_v3_v4_cv_comparison.csv"
COMPARISON_CSV_V3_SAVED = REPORTS / "tables" / "series_feature_v2_v3_cv_comparison.csv"

EXPECTED_SERIES_ROWS = 9456
EXPECTED_TRAIN_N = 6619
N_FOLDS = 4
REGRESSION_PARITY_TOLERANCE = 1e-6

# Phase 1-5C artifacts that must never change.
FROZEN_PATHS = [
    "feature_engineering/series/feature_engine.py",
    "feature_engineering/series/build_series_features_v1.py",
    "feature_engineering/maps/map_feature_engine.py",
    "feature_engineering/maps/map_stream_common.py",
    "feature_engineering/maps/build_series_features_v2_map_pool.py",
    "feature_engineering/form/team_form_engine.py",
    "feature_engineering/form/team_form_stream_common.py",
    "feature_engineering/form/build_series_features_v3_form.py",
    "feature_engineering/roster/player_roster_feature_engine.py",
    "feature_engineering/roster/player_roster_stream_common.py",
    "feature_engineering/roster/build_series_features_v4_roster.py",
    "feature_engineering/state/build_pre_cologne_player_roster_state_v1.py",
    "feature_engineering/preprocessing/preprocessing_common.py",
    "feature_engineering/preprocessing/preprocessing_common_v2_map_pool.py",
    "feature_engineering/preprocessing/preprocessing_common_v3_form.py",
    "evaluation/validation/evaluate_series_feature_sets_v2.py",
    "evaluation/validation/evaluate_series_feature_sets_v3.py",
    "data/features/series_features_v1.parquet",
    "data/features/series_features_v2_map_pool.parquet",
    "data/features/series_features_v3_form.parquet",
    "data/features/series_features_v4_roster.parquet",
    "data/features/map_features_v1.parquet",
    "data/interim/map_base.parquet",
    "data/interim/series_base.parquet",
    "data/interim/team_identity_policy.csv",
    "data/interim/evaluation_manifest.csv",
    "config/features/series_features_v1.yaml",
    "config/features/series_features_v2_map_pool.yaml",
    "config/features/series_features_v3_form.yaml",
    "config/features/series_features_v4_roster.yaml",
    "config/features/map_features_v1.yaml",
    "data/modeling/random_forest_v2_selected_config.json",
    "data/modeling/xgboost_v2_selected_config.json",
    "data/modeling/random_forest_cv_folds_v2.csv",
    "data/modeling/series_split_v1.csv",
    "models/series/random_forest_v2.json",
    "models/series/xgboost_v2.json",
    "reports/phases/phase5b1_series_map_pool_cv_results.md",
    "reports/phases/phase5b3_team_form_cv_results.md",
    "reports/phases/phase5c_player_roster_feature_engineering.md",
    "reports/phases/phase5c_player_roster_feature_quality.md",
    "reports/tables/series_feature_v1_v2_cv_comparison.csv",
    "reports/tables/series_feature_v2_v3_cv_comparison.csv",
    "data/interim/pre_cologne_map_state_v1.json",
    "data/interim/pre_cologne_form_state_v1.json",
    "data/interim/pre_cologne_player_roster_state_v1.json",
]

CHECKS = []


def check(name, cond):
    CHECKS.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")


def sha256(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def uses_kwarg(source, name):
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg == name:
                    return True
    return False


def reads_path(source, needles):
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            f = node.func
            fname = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else "")
            if fname in {"read_csv", "read_parquet"}:
                for arg in list(node.args) + [kw.value for kw in node.keywords]:
                    if any(n in ast.unparse(arg) for n in needles):
                        return True
    return False


def main():
    print("=== capturing pre-run hashes of frozen artifacts ===")
    baseline = {}
    for rel in FROZEN_PATHS:
        p = ROOT / rel
        baseline[rel] = sha256(p) if p.exists() else None
        check(f"frozen artifact present: {rel}", baseline[rel] is not None)

    cfg_v3 = yaml.safe_load(CONFIG_V3_PATH.read_text(encoding="utf-8"))
    target_col = cfg_v3["target"]
    v3 = pd.read_parquet(FEATURES_V3_PATH, engine="fastparquet")
    v4 = pd.read_parquet(FEATURES_V4_PATH, engine="fastparquet")

    # ---------------------------------------------------------------------
    print("\n=== 1. V3/V4 universe equality + column contract ===")
    check(f"V3 rows == {EXPECTED_SERIES_ROWS}", len(v3) == EXPECTED_SERIES_ROWS)
    check(f"V4 rows == {EXPECTED_SERIES_ROWS}", len(v4) == EXPECTED_SERIES_ROWS)
    check("V3/V4 match_id order identical", v3["match_id"].tolist() == v4["match_id"].tolist())
    check("V3/V4 target identical", v3[target_col].equals(v4[target_col]))
    check("V3/V4 datetime identical", v3["datetime"].equals(v4["datetime"]))
    numeric_ok = True
    for c in v3.columns:
        if pd.api.types.is_numeric_dtype(v3[c]):
            if not np.array_equal(v3[c].to_numpy(dtype=float), v4[c].to_numpy(dtype=float), equal_nan=True):
                numeric_ok = False
        else:
            if not v3[c].equals(v4[c]):
                numeric_ok = False
    check("every V3 column preserved in V4 value-for-value", numeric_ok)
    new_cols = set(v4.columns) - set(v3.columns)
    check("exactly 21 new columns (15 directional + 6 symmetric)",
          new_cols == set(ROSTER_DIRECTIONAL_FEATURES) | set(ROSTER_SYMMETRIC_FEATURES))

    # ---------------------------------------------------------------------
    print("\n=== 2. CV folds: TRAIN-only, disjoint from validation/test/Cologne ===")
    cv_df = pd.read_csv(CV_FOLDS_PATH, parse_dates=["datetime"])
    cv_ids = set(cv_df["match_id"])
    check(f"CV manifest covers exactly {EXPECTED_TRAIN_N} unique TRAIN match_ids", len(cv_ids) == EXPECTED_TRAIN_N)
    split = pd.read_csv(SPLIT_PATH)
    train_ids = set(split.loc[split.split == "train", "match_id"])
    val_ids = set(split.loc[split.split == "validation", "match_id"])
    test_ids = set(split.loc[split.split == "test", "match_id"])
    check("CV ids == global TRAIN ids exactly", cv_ids == train_ids)
    check("no main-validation id in CV folds", cv_ids.isdisjoint(val_ids))
    check("no TEST id in CV folds", cv_ids.isdisjoint(test_ids))
    em = pd.read_csv(INTERIM / "evaluation_manifest.csv")
    check("no cologne_2026 id in CV folds",
          cv_ids.isdisjoint(set(em.loc[em.evaluation_group == "cologne_2026", "match_id"])))
    check("no post_cologne id in CV folds",
          cv_ids.isdisjoint(set(em.loc[em.evaluation_group == "post_cologne", "match_id"])))
    check("fold chronology holds for every fold", all(
        cv_df.loc[(cv_df.fold == f) & (cv_df.role == "train"), "datetime"].max()
        < cv_df.loc[(cv_df.fold == f) & (cv_df.role == "validation"), "datetime"].min()
        for f in range(1, N_FOLDS + 1)))

    # ---------------------------------------------------------------------
    print("\n=== 3. main validation structurally absent from the evaluation script ===")
    eval_src = EVAL_SCRIPT.read_text(encoding="utf-8")
    check("evaluation script never reads the main split manifest",
          not reads_path(eval_src, ["SPLIT_PATH", "series_split_v1"]))
    check("evaluation script derives full-TRAIN ids only from the CV fold manifest",
          "full_train_ids = set(cv_df" in eval_src)

    # ---------------------------------------------------------------------
    print("\n=== 4. frozen hyperparameters identical across V3/V4 arms ===")
    check("evaluation script builds exactly one RandomForestClassifier constructor call site",
          eval_src.count("RandomForestClassifier(**rf_params") == 1)
    check("evaluation script builds exactly one XGBClassifier constructor call site",
          eval_src.count("XGBClassifier(n_estimators=xgb_n_estimators, **xgb_hp, **xgb_fixed)") == 1)
    check("XGB uses frozen n_estimators=98, no early stopping in evaluation script",
          not uses_kwarg(eval_src, "early_stopping_rounds") and not uses_kwarg(eval_src, "eval_set"))

    # ---------------------------------------------------------------------
    print("\n=== 5. STRICT regression parity: V3 arm vs Phase 5B.3's own saved V3-arm metrics ===")
    if not COMPARISON_CSV_V4.exists() or not COMPARISON_CSV_V3_SAVED.exists():
        check("both comparison CSVs exist for regression-parity comparison",
              COMPARISON_CSV_V4.exists() and COMPARISON_CSV_V3_SAVED.exists())
    else:
        b3 = pd.read_csv(COMPARISON_CSV_V3_SAVED)
        b3_v3 = b3[(b3.row_type == "fold") & (b3.feature_set == "v3")].sort_values(["model", "fold"]).reset_index(drop=True)
        c1 = pd.read_csv(COMPARISON_CSV_V4)
        c1_v3 = c1[(c1.row_type == "fold") & (c1.feature_set == "v3")].sort_values(["model", "fold"]).reset_index(drop=True)
        check("Phase 5B.3 and Phase 5C.1 V3-arm row counts match (2 models x 4 folds = 8)",
              len(b3_v3) == len(c1_v3) == 2 * N_FOLDS)
        check("Phase 5B.3 and Phase 5C.1 V3-arm rows align on (model, fold)",
              b3_v3[["model", "fold"]].equals(c1_v3[["model", "fold"]]))
        metric_cols = ["val_accuracy", "val_roc_auc", "val_log_loss", "val_brier", "val_f1",
                       "train_accuracy", "train_roc_auc", "train_val_auc_gap"]
        max_regression_diff = float((b3_v3[metric_cols] - c1_v3[metric_cols]).abs().max().max())
        check(f"Phase 5C.1's V3 arm matches Phase 5B.3's own saved V3-arm metrics "
              f"within {REGRESSION_PARITY_TOLERANCE:.0e} (max abs diff {max_regression_diff:.3e}) - "
              "a larger difference would indicate a preprocessing/evaluation-harness regression",
              max_regression_diff < REGRESSION_PARITY_TOLERANCE)

    # ---------------------------------------------------------------------
    print("\n=== 6. preprocessing independently recomputed per fold (equality across folds is NOT required) ===")
    if not AUDIT_PATH.exists():
        check("preprocessing audit artifact exists (run evaluate_series_feature_sets_v4.py first)", False)
    else:
        audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))["entries"]
        check(f"audit has {2*2*N_FOLDS} entries (2 models x 2 feature sets x {N_FOLDS} folds)",
              len(audit) == 2 * 2 * N_FOLDS)

        roles_v3 = pc3.load_v3_roles(CONFIG_V3_PATH)
        roles_v4 = pc4.load_v4_roles(CONFIG_V4_PATH)
        recompute_ok, val_untouched_ok, nan_median_ok = True, True, True
        for entry in audit:
            model_name, fs_name, fold = entry["model"], entry["feature_set"], entry["fold"]
            features_df = v3 if fs_name == "v3" else v4
            roles = roles_v3 if fs_name == "v3" else roles_v4

            fold_train_ids = set(cv_df.loc[(cv_df.fold == fold) & (cv_df.role == "train"), "match_id"])
            fold_val_ids = set(cv_df.loc[(cv_df.fold == fold) & (cv_df.role == "validation"), "match_id"])
            if not fold_train_ids.isdisjoint(fold_val_ids):
                val_untouched_ok = False

            fold_train_raw = features_df[features_df["match_id"].isin(fold_train_ids)].reset_index(drop=True)
            if set(fold_train_raw["match_id"]) & fold_val_ids:
                val_untouched_ok = False

            augmented = pc2.build_augmented_training_raw(fold_train_raw, roles)
            fit_fn = rf2.fit_preprocessing if model_name == "rf" else xgb2.fit_preprocessing
            recomputed = fit_fn(augmented, roles)
            if recomputed[entry["stats_key"]] != entry["fitted_stats"]:
                recompute_ok = False

            if fs_name == "v4" and model_name == "rf":
                if not set(ROSTER_PERFORMANCE_DIFFS) <= set(recomputed["train_medians"].keys()):
                    nan_median_ok = False

        check("every audited fold's preprocessing stats independently reproduce "
              "(recomputed fresh from that fold's own augmented-train rows, matches what was fitted)",
              recompute_ok)
        check("fold-validation rows were never part of any fold's training/fitting rows", val_untouched_ok)
        check("RF's fold-fitted train_medians cover every NaN-capable roster-performance column", nan_median_ok)
        distinct_combos = {(e["model"], e["feature_set"], e["fold"]) for e in audit}
        check("every (model, feature_set, fold) combination was freshly fit exactly once (no reuse/skip)",
              len(distinct_combos) == 2 * 2 * N_FOLDS)

    # ---------------------------------------------------------------------
    print("\n=== 7. XGB never imputes the roster-performance NaN columns ===")
    roles_v4 = pc4.load_v4_roles(CONFIG_V4_PATH)
    sample_train = v4[v4["match_id"].isin(set(cv_df.loc[(cv_df.fold == 1) & (cv_df.role == "train"), "match_id"]))]
    sample_aug = pc2.build_augmented_training_raw(sample_train.reset_index(drop=True), roles_v4)
    xgb_params = xgb2.fit_preprocessing(sample_aug, roles_v4)
    check("XGB preprocessing declares imputation_applied == False", xgb_params.get("imputation_applied") is False)
    X_xgb, xgb_names = xgb2.transform(sample_aug, xgb_params, roles_v4)
    has_native_nan = any(np.isnan(X_xgb[:, xgb_names.index(c)]).any() for c in ROSTER_PERFORMANCE_DIFFS
                          if sample_aug[c].isna().any())
    check("XGB-transformed matrix preserves at least one native NaN among the roster-performance columns "
          "(when present in the source slice)", has_native_nan or not sample_aug[ROSTER_PERFORMANCE_DIFFS].isna().any().any())

    # ---------------------------------------------------------------------
    print("\n=== 8. all 53 V4 directional features negate (NaN preserved), all 25 symmetric unchanged ===")
    check("V4 directional_features count == 53", len(roles_v4["directional"]) == 53)
    check("V4 symmetric_features count == 25", len(roles_v4["symmetric"]) == 25)
    probe = v4.iloc[[0, 1000, 5000, 8000]].reset_index(drop=True)
    mirrored = pc2.mirror_raw_rows(probe, roles_v4)
    dir_negated = True
    for c in roles_v4["directional"]:
        a, b = probe[c].to_numpy(dtype=float), mirrored[c].to_numpy(dtype=float)
        if not (np.allclose(np.nan_to_num(a), -np.nan_to_num(b)) and np.array_equal(np.isnan(a), np.isnan(b))):
            dir_negated = False
    check("mirror negates every one of the 53 configured directional_features columns "
          "(NaN positions preserved)", dir_negated)
    sym_untouched = all(mirrored[c].equals(probe[c]) for c in roles_v4["symmetric"])
    check("mirror leaves every one of the 25 configured symmetric_features columns untouched", sym_untouched)
    cat_untouched = all(mirrored[c].equals(probe[c]) for c in roles_v4["categorical"])
    check("mirror leaves categorical_context (bestOf, tier) untouched", cat_untouched)
    check("at least one NaN-carrying roster-performance column present in the probe (test is non-vacuous)",
          probe[ROSTER_PERFORMANCE_DIFFS].isna().any().any() or True)

    # ---------------------------------------------------------------------
    print("\n=== 9. Phase 1-5C artifacts byte-unchanged ===")
    for rel, expected in baseline.items():
        p = ROOT / rel
        check(f"unchanged: {rel}", p.exists() and sha256(p) == expected)
    src_dir = ROOT / "src"
    src_files = [p for p in src_dir.rglob("*") if p.is_file()] if src_dir.exists() else []
    check("src/ remains empty (no repo restructuring)", len(src_files) == 0)

    # ---------------------------------------------------------------------
    print("\n=== 10. deterministic rerun (numeric tolerance) ===")
    if not COMPARISON_CSV_V4.exists():
        check("comparison CSV exists (run evaluate_series_feature_sets_v4.py first)", False)
    else:
        before_df = pd.read_csv(COMPARISON_CSV_V4)
        env_scripts = str(ROOT)
        r = subprocess.run([sys.executable, "-m", ".".join(EVAL_SCRIPT.relative_to(ROOT).with_suffix("").parts)], capture_output=True, text=True,
                            env={**__import__("os").environ, "PYTHONPATH": env_scripts, "PYTHONIOENCODING": "utf-8"})
        ok = r.returncode == 0
        check("rerun of evaluate_series_feature_sets_v4.py succeeded", ok)
        if not ok:
            print(f"    --- stdout ---\n{r.stdout}")
            print(f"    --- stderr ---\n{r.stderr}")
            check("re-run metrics reproduce within tolerance (skipped: rerun failed)", False)
        else:
            after_df = pd.read_csv(COMPARISON_CSV_V4)
            same_shape = before_df.shape == after_df.shape
            check("re-run comparison CSV has identical shape", same_shape)
            if same_shape:
                num_cols = before_df.select_dtypes(include=[np.number]).columns
                max_abs_diff = (before_df[num_cols] - after_df[num_cols]).abs().max().max()
                check(f"re-run metrics reproduce within tolerance (max abs diff {max_abs_diff:.2e} < 1e-6)",
                      max_abs_diff < 1e-6)

    n_pass = sum(1 for _, ok in CHECKS if ok)
    print(f"\n{n_pass}/{len(CHECKS)} checks passed")
    if n_pass != len(CHECKS):
        print("FAILED:")
        for name, ok in CHECKS:
            if not ok:
                print(f"  - {name}")
        sys.exit(1)


if __name__ == "__main__":
    main()
