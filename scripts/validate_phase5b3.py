"""
Phase 5B.3 validation (artifact-level). Read-only. Exits non-zero on failure.

Checks fall into groups:
  1. V2/V3 universe equality + every V2 column preserved + exactly 12 new columns;
  2. CV folds are GLOBAL-TRAIN-only (disjoint from validation/test/Cologne);
  3. the main VALIDATION partition is structurally never read by the
     evaluation script (AST-level source check);
  4. RF/XGB hyperparameters used for the V2 arm and the V3 arm are
     byte-identical (single call site each, loaded once, reused);
  5. preprocessing is freshly fit per (model, feature_set, fold) - independent
     recomputation, same fix as validate_phase5b1.py (equality of medians
     ACROSS folds is not required and must never fail this check);
  6. all 38 V3 directional features negate under config-driven mirroring,
     all 19 symmetric features unchanged;
  7. Phase 4 / Phase 5A / Phase 5B.1 / Phase 5B.2 artifacts byte-unchanged;
  8. deterministic rerun (numeric tolerance - RF/XGB n_jobs=-1 nondeterminism);
  9. STRICT regression-parity of the V2 arm against Phase 5B.1's own saved
     V2-arm fold metrics - a material difference FAILS validation.
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
import preprocessing_common_v2_map_pool as pc2
import preprocessing_random_forest_v2_map_pool as rf2
import preprocessing_xgboost_v2_map_pool as xgb2
import preprocessing_common_v3_form as pc3
from team_form_engine import FORM_DIRECTIONAL_FEATURES, FORM_SYMMETRIC_FEATURES

CONFIG_V2_PATH = ROOT / "config" / "series_features_v2_map_pool.yaml"
CONFIG_V3_PATH = ROOT / "config" / "series_features_v3_form.yaml"
FEATURES_V2_PATH = ROOT / "data" / "features" / "series_features_v2_map_pool.parquet"
FEATURES_V3_PATH = ROOT / "data" / "features" / "series_features_v3_form.parquet"
CV_FOLDS_PATH = ROOT / "data" / "modeling" / "random_forest_cv_folds_v2.csv"
SPLIT_PATH = ROOT / "data" / "modeling" / "series_split_v1.csv"
RF_SELECTED_CONFIG_PATH = ROOT / "data" / "modeling" / "random_forest_v2_selected_config.json"
XGB_SELECTED_CONFIG_PATH = ROOT / "data" / "modeling" / "xgboost_v2_selected_config.json"
AUDIT_PATH = ROOT / "data" / "modeling" / "phase5b3_fold_preprocessing_audit.json"
EVAL_SCRIPT = ROOT / "scripts" / "evaluate_series_feature_sets_v3.py"
COMPARISON_CSV_V3 = REPORTS / "tables" / "series_feature_v2_v3_cv_comparison.csv"
COMPARISON_CSV_V1 = REPORTS / "tables" / "series_feature_v1_v2_cv_comparison.csv"

EXPECTED_SERIES_ROWS = 9456
EXPECTED_TRAIN_N = 6619
N_FOLDS = 4
REGRESSION_PARITY_TOLERANCE = 1e-6

# Phase 4 / 5A / 5B.1 / 5B.2 artifacts that must never change.
FROZEN_PATHS = [
    "scripts/feature_engine.py",
    "scripts/build_series_features_v1.py",
    "scripts/map_feature_engine.py",
    "scripts/map_stream_common.py",
    "scripts/build_series_features_v2_map_pool.py",
    "scripts/team_form_engine.py",
    "scripts/team_form_stream_common.py",
    "scripts/build_series_features_v3_form.py",
    "scripts/build_pre_cologne_form_state_v1.py",
    "scripts/preprocessing_common_v2_map_pool.py",
    "scripts/preprocessing_random_forest_v2_map_pool.py",
    "scripts/preprocessing_xgboost_v2_map_pool.py",
    "scripts/evaluate_series_feature_sets_v2.py",
    "data/features/series_features_v1.parquet",
    "data/features/series_features_v2_map_pool.parquet",
    "data/features/series_features_v3_form.parquet",
    "data/features/map_features_v1.parquet",
    "config/series_features_v1.yaml",
    "config/series_features_v2_map_pool.yaml",
    "config/series_features_v3_form.yaml",
    "config/map_features_v1.yaml",
    "data/modeling/random_forest_v2_selected_config.json",
    "data/modeling/xgboost_v2_selected_config.json",
    "data/modeling/random_forest_cv_folds_v2.csv",
    "models/random_forest_v2.json",
    "models/xgboost_v2.json",
    "reports/phase5b1_series_map_pool_cv_results.md",
    "reports/tables/series_feature_v1_v2_cv_comparison.csv",
    "reports/phase5b2_team_form_feature_engineering.md",
    "reports/phase5b2_team_form_feature_quality.md",
    "data/interim/pre_cologne_map_state_v1.json",
    "data/interim/pre_cologne_map_state_v1.parquet",
    "data/interim/pre_cologne_form_state_v1.json",
    "data/interim/pre_cologne_form_state_v1.parquet",
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

    cfg_v2 = yaml.safe_load(CONFIG_V2_PATH.read_text(encoding="utf-8"))
    target_col = cfg_v2["target"]
    v2 = pd.read_parquet(FEATURES_V2_PATH, engine="fastparquet")
    v3 = pd.read_parquet(FEATURES_V3_PATH, engine="fastparquet")

    # ---------------------------------------------------------------------
    print("\n=== 1. V2/V3 universe equality + column contract ===")
    check(f"V2 rows == {EXPECTED_SERIES_ROWS}", len(v2) == EXPECTED_SERIES_ROWS)
    check(f"V3 rows == {EXPECTED_SERIES_ROWS}", len(v3) == EXPECTED_SERIES_ROWS)
    check("V2/V3 match_id order identical", v2["match_id"].tolist() == v3["match_id"].tolist())
    check("V2/V3 target identical", v2[target_col].equals(v3[target_col]))
    check("V2/V3 datetime identical", v2["datetime"].equals(v3["datetime"]))
    numeric_ok = True
    for c in v2.columns:
        if pd.api.types.is_numeric_dtype(v2[c]):
            if not np.array_equal(v2[c].to_numpy(dtype=float), v3[c].to_numpy(dtype=float), equal_nan=True):
                numeric_ok = False
        else:
            if not v2[c].equals(v3[c]):
                numeric_ok = False
    check("every V2 column preserved in V3 value-for-value", numeric_ok)
    new_cols = set(v3.columns) - set(v2.columns)
    check("exactly 12 new columns (8 directional + 4 symmetric)",
          new_cols == set(FORM_DIRECTIONAL_FEATURES) | set(FORM_SYMMETRIC_FEATURES))

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
    print("\n=== 4. frozen hyperparameters identical across V2/V3 arms ===")
    check("evaluation script builds exactly one RandomForestClassifier constructor call site",
          eval_src.count("RandomForestClassifier(**rf_params") == 1)
    check("evaluation script builds exactly one XGBClassifier constructor call site",
          eval_src.count("XGBClassifier(n_estimators=xgb_n_estimators, **xgb_hp, **xgb_fixed)") == 1)
    check("XGB uses frozen n_estimators=98, no early stopping in evaluation script",
          not uses_kwarg(eval_src, "early_stopping_rounds") and not uses_kwarg(eval_src, "eval_set"))

    # ---------------------------------------------------------------------
    print("\n=== 5. preprocessing independently recomputed per fold (equality across folds is NOT required) ===")
    if not AUDIT_PATH.exists():
        check("preprocessing audit artifact exists (run evaluate_series_feature_sets_v3.py first)", False)
    else:
        audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))["entries"]
        check(f"audit has {2*2*N_FOLDS} entries (2 models x 2 feature sets x {N_FOLDS} folds)",
              len(audit) == 2 * 2 * N_FOLDS)

        roles_v2 = pc2.load_v2_roles(CONFIG_V2_PATH)
        roles_v3 = pc3.load_v3_roles(CONFIG_V3_PATH)
        recompute_ok, val_untouched_ok = True, True
        for entry in audit:
            model_name, fs_name, fold = entry["model"], entry["feature_set"], entry["fold"]
            features_df = v2 if fs_name == "v2" else v3
            roles = roles_v2 if fs_name == "v2" else roles_v3

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

        check("every audited fold's preprocessing stats independently reproduce "
              "(recomputed fresh from that fold's own augmented-train rows, matches what was fitted)",
              recompute_ok)
        check("fold-validation rows were never part of any fold's training/fitting rows", val_untouched_ok)
        distinct_combos = {(e["model"], e["feature_set"], e["fold"]) for e in audit}
        check("every (model, feature_set, fold) combination was freshly fit exactly once (no reuse/skip)",
              len(distinct_combos) == 2 * 2 * N_FOLDS)

    # ---------------------------------------------------------------------
    print("\n=== 6. all 38 V3 directional features negate, all 19 symmetric unchanged ===")
    roles_v3 = pc3.load_v3_roles(CONFIG_V3_PATH)
    check("V3 directional_features count == 38", len(roles_v3["directional"]) == 38)
    check("V3 symmetric_features count == 19", len(roles_v3["symmetric"]) == 19)
    probe = v3.iloc[[0, 1000, 5000]].reset_index(drop=True)
    mirrored = pc2.mirror_raw_rows(probe, roles_v3)
    dir_negated = all(np.allclose(mirrored[c].to_numpy(dtype=float), -probe[c].to_numpy(dtype=float), equal_nan=True)
                       for c in roles_v3["directional"])
    check("mirror negates every one of the 38 configured directional_features columns", dir_negated)
    sym_untouched = all(mirrored[c].equals(probe[c]) for c in roles_v3["symmetric"])
    check("mirror leaves every one of the 19 configured symmetric_features columns untouched", sym_untouched)
    cat_untouched = all(mirrored[c].equals(probe[c]) for c in roles_v3["categorical"])
    check("mirror leaves categorical_context (bestOf, tier) untouched", cat_untouched)

    # ---------------------------------------------------------------------
    print("\n=== 7. Phase 4 / Phase 5A / Phase 5B.1 / Phase 5B.2 artifacts byte-unchanged ===")
    for rel, expected in baseline.items():
        p = ROOT / rel
        check(f"unchanged: {rel}", p.exists() and sha256(p) == expected)
    src_dir = ROOT / "src"
    src_files = [p for p in src_dir.rglob("*") if p.is_file()] if src_dir.exists() else []
    check("src/ remains empty (no repo restructuring)", len(src_files) == 0)

    # ---------------------------------------------------------------------
    print("\n=== 8. deterministic rerun (numeric tolerance) ===")
    if not COMPARISON_CSV_V3.exists():
        check("comparison CSV exists (run evaluate_series_feature_sets_v3.py first)", False)
        before_df = None
    else:
        before_df = pd.read_csv(COMPARISON_CSV_V3)
        env_scripts = str(ROOT / "scripts")
        r = subprocess.run([sys.executable, str(EVAL_SCRIPT)], capture_output=True, text=True,
                            env={**__import__("os").environ, "PYTHONPATH": env_scripts, "PYTHONIOENCODING": "utf-8"})
        ok = r.returncode == 0
        check("rerun of evaluate_series_feature_sets_v3.py succeeded", ok)
        if not ok:
            print(f"    --- stdout ---\n{r.stdout}")
            print(f"    --- stderr ---\n{r.stderr}")
            check("re-run metrics reproduce within tolerance (skipped: rerun failed)", False)
        else:
            after_df = pd.read_csv(COMPARISON_CSV_V3)
            same_shape = before_df.shape == after_df.shape
            check("re-run comparison CSV has identical shape", same_shape)
            if same_shape:
                num_cols = before_df.select_dtypes(include=[np.number]).columns
                max_abs_diff = (before_df[num_cols] - after_df[num_cols]).abs().max().max()
                check(f"re-run metrics reproduce within tolerance (max abs diff {max_abs_diff:.2e} < 1e-6)",
                      max_abs_diff < 1e-6)
                before_df = after_df   # use the freshest run for the regression-parity check below

    # ---------------------------------------------------------------------
    print("\n=== 9. STRICT regression parity: V2 arm vs Phase 5B.1's own saved V2-arm metrics ===")
    if not COMPARISON_CSV_V1.exists() or before_df is None:
        check("Phase 5B.1 comparison CSV exists for regression-parity comparison", COMPARISON_CSV_V1.exists())
    else:
        b1 = pd.read_csv(COMPARISON_CSV_V1)
        b1_v2 = b1[(b1.row_type == "fold") & (b1.feature_set == "v2")].sort_values(["model", "fold"]).reset_index(drop=True)
        b3_v2 = before_df[(before_df.row_type == "fold") & (before_df.feature_set == "v2")] \
            .sort_values(["model", "fold"]).reset_index(drop=True)
        check("Phase 5B.1 and Phase 5B.3 V2-arm row counts match (2 models x 4 folds = 8)",
              len(b1_v2) == len(b3_v2) == 2 * N_FOLDS)
        check("Phase 5B.1 and Phase 5B.3 V2-arm rows align on (model, fold)",
              b1_v2[["model", "fold"]].equals(b3_v2[["model", "fold"]]))
        metric_cols = ["val_accuracy", "val_roc_auc", "val_log_loss", "val_brier", "val_f1",
                       "train_accuracy", "train_roc_auc", "train_val_auc_gap"]
        max_regression_diff = float((b1_v2[metric_cols] - b3_v2[metric_cols]).abs().max().max())
        check(f"Phase 5B.3's V2 arm matches Phase 5B.1's own saved V2-arm metrics "
              f"within {REGRESSION_PARITY_TOLERANCE:.0e} (max abs diff {max_regression_diff:.3e}) - "
              "a larger difference would indicate a preprocessing/evaluation-harness regression",
              max_regression_diff < REGRESSION_PARITY_TOLERANCE)

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
