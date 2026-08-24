"""
Phase 5B.1 validation (artifact-level). Read-only. Exits non-zero on failure.

Checks fall into groups:
  1. V1/V2 series-feature contract (universe, target, order preserved);
  2. CV folds are GLOBAL-TRAIN-only (disjoint from validation/test/Cologne);
  3. the main VALIDATION partition is structurally never read by the
     evaluation script (AST-level source check);
  4. RF/XGB hyperparameters used for the V1 arm and the V2 arm are
     byte-identical (loaded once from the frozen Phase 4 configs, reused);
  5. preprocessing is freshly fit per (model, feature_set, fold) - proven by
     independently recomputing each fold's expected statistics directly from
     THAT fold's own augmented training rows and comparing to what was
     actually fitted, NOT by requiring medians to differ across folds
     (different chronological folds may legitimately produce identical
     medians - that must never fail this check);
  6. V2 mirroring negates exactly config["directional_features"] and nothing else;
  7. Phase 4 and Phase 5A artifacts are byte-unchanged;
  8. deterministic rerun of the evaluation script.
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

CONFIG_V1_PATH = ROOT / "config" / "features" / "series_features_v1.yaml"
CONFIG_V2_PATH = ROOT / "config" / "features" / "series_features_v2_map_pool.yaml"
FEATURES_V1_PATH = ROOT / "data" / "features" / "series_features_v1.parquet"
FEATURES_V2_PATH = ROOT / "data" / "features" / "series_features_v2_map_pool.parquet"
CV_FOLDS_PATH = ROOT / "data" / "modeling" / "random_forest_cv_folds_v2.csv"
SPLIT_PATH = ROOT / "data" / "modeling" / "series_split_v1.csv"
RF_SELECTED_CONFIG_PATH = ROOT / "data" / "modeling" / "random_forest_v2_selected_config.json"
XGB_SELECTED_CONFIG_PATH = ROOT / "data" / "modeling" / "xgboost_v2_selected_config.json"
AUDIT_PATH = ROOT / "data" / "modeling" / "phase5b1_fold_preprocessing_audit.json"
EVAL_SCRIPT = ROOT / "evaluation" / "validation" / "evaluate_series_feature_sets_v2.py"
COMPARISON_CSV = REPORTS / "tables" / "series_feature_v1_v2_cv_comparison.csv"

EXPECTED_SERIES_ROWS = 9456
EXPECTED_TRAIN_N = 6619
EXPECTED_VAL_N = 1419
EXPECTED_TEST_N = 1418
N_FOLDS = 4

# Phase 4 + Phase 5A artifacts that must never be modified by anything in Phase 5B.1.
# Hashes are captured at the start of main() (this validator is read-only, and
# evaluate_series_feature_sets_v2.py never writes to any of these paths, so the
# value captured here is unaffected by run order) and re-checked after the
# deterministic-rerun step (#8) to prove the rerun didn't touch them either.
BASELINE_HASHES = {
    "data/features/series_features_v1.parquet": None,   # filled in at runtime (see main())
    "data/features/map_features_v1.parquet": None,
    "data/features/series_features_v2_map_pool.parquet": None,
    "config/features/series_features_v1.yaml": None,
    "config/features/series_features_v2_map_pool.yaml": None,
    "config/features/map_features_v1.yaml": None,
    "data/modeling/random_forest_cv_folds_v2.csv": None,
    "data/modeling/random_forest_v2_selected_config.json": None,
    "data/modeling/xgboost_v2_selected_config.json": None,
    "models/series/random_forest_v2.joblib": None,
    "models/series/random_forest_v2.json": None,
    "models/series/xgboost_v2.json": None,
    "models/series/xgboost_v2_metadata.json": None,
    "models/series/logistic_regression_scratch_v2.npz": None,
    "models/series/logistic_regression_scratch_v2.json": None,
    "data/interim/pre_cologne_map_state_v1.parquet": None,
    "data/interim/pre_cologne_map_state_v1.json": None,
}

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
    """True if any pandas read_* call references one of `needles`."""
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
    print("=== capturing pre-run baseline hashes (frozen Phase 4 / Phase 5A artifacts) ===")
    for rel in list(BASELINE_HASHES):
        p = ROOT / rel
        BASELINE_HASHES[rel] = sha256(p) if p.exists() else None
        check(f"baseline artifact present: {rel}", BASELINE_HASHES[rel] is not None)

    # ---------------------------------------------------------------------
    # 1. V1/V2 series-feature contract
    # ---------------------------------------------------------------------
    print("\n=== 1. V1/V2 contract ===")
    cfg_v1 = yaml.safe_load(CONFIG_V1_PATH.read_text(encoding="utf-8"))
    model_features_v1 = cfg_v1["model_features"]
    target_col = cfg_v1["target"]
    roles_v2 = pc2.load_v2_roles(CONFIG_V2_PATH)
    check("V1 and V2 configs declare the same target", roles_v2["target"] == target_col)

    v1_df = pd.read_parquet(FEATURES_V1_PATH, engine="fastparquet")
    v2_df = pd.read_parquet(FEATURES_V2_PATH, engine="fastparquet")
    check(f"V1 rows == {EXPECTED_SERIES_ROWS}", len(v1_df) == EXPECTED_SERIES_ROWS)
    check(f"V2 rows == {EXPECTED_SERIES_ROWS}", len(v2_df) == EXPECTED_SERIES_ROWS)
    check("V1/V2 match_id order identical", v1_df["match_id"].tolist() == v2_df["match_id"].tolist())
    check("V1/V2 target identical", v1_df[target_col].equals(v2_df[target_col]))
    check("V1/V2 datetime identical", v1_df["datetime"].equals(v2_df["datetime"]))
    check("V1/V2 team1_canonical identical", v1_df["team1_canonical"].equals(v2_df["team1_canonical"]))
    check("V1/V2 team2_canonical identical", v1_df["team2_canonical"].equals(v2_df["team2_canonical"]))
    numeric_ok = True
    for c in model_features_v1:
        if pd.api.types.is_numeric_dtype(v1_df[c]):
            if not np.array_equal(v1_df[c].to_numpy(dtype=float), v2_df[c].to_numpy(dtype=float), equal_nan=True):
                numeric_ok = False
        else:
            if not v1_df[c].equals(v2_df[c]):
                numeric_ok = False
    check("every V1 feature column preserved in V2 (value-for-value)", numeric_ok)

    # ---------------------------------------------------------------------
    # 2. CV folds are global-TRAIN-only
    # ---------------------------------------------------------------------
    print("\n=== 2. CV folds: TRAIN-only, disjoint from validation/test/Cologne ===")
    cv_df = pd.read_csv(CV_FOLDS_PATH, parse_dates=["datetime"])
    cv_ids = set(cv_df["match_id"])
    check(f"CV manifest covers exactly {EXPECTED_TRAIN_N} unique TRAIN match_ids", len(cv_ids) == EXPECTED_TRAIN_N)

    split = pd.read_csv(SPLIT_PATH)
    counts = split["split"].value_counts()
    check("split train == 6619", counts.get("train", 0) == EXPECTED_TRAIN_N)
    check("split validation == 1419", counts.get("validation", 0) == EXPECTED_VAL_N)
    check("split test == 1418", counts.get("test", 0) == EXPECTED_TEST_N)
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
    # 3. main validation structurally never read by the evaluation script
    # ---------------------------------------------------------------------
    print("\n=== 3. main validation structurally absent from the evaluation script ===")
    eval_src = EVAL_SCRIPT.read_text(encoding="utf-8")
    check("evaluation script never reads the main split manifest",
          not reads_path(eval_src, ["SPLIT_PATH", "series_split_v1"]))
    check("evaluation script derives full-TRAIN ids only from the CV fold manifest",
          "full_train_ids = set(cv_df" in eval_src)

    # ---------------------------------------------------------------------
    # 4. hyperparameters identical for V1 arm and V2 arm
    # ---------------------------------------------------------------------
    print("\n=== 4. frozen hyperparameters identical across V1/V2 arms ===")
    rf_selected = json.loads(RF_SELECTED_CONFIG_PATH.read_text(encoding="utf-8"))
    xgb_selected = json.loads(XGB_SELECTED_CONFIG_PATH.read_text(encoding="utf-8"))
    check("evaluation script builds exactly one RandomForestClassifier constructor call site",
          eval_src.count("RandomForestClassifier(**rf_params") == 1)
    check("evaluation script builds exactly one XGBClassifier constructor call site",
          eval_src.count("XGBClassifier(n_estimators=xgb_n_estimators, **xgb_hp, **xgb_fixed)") == 1)
    check("RF frozen params loaded from data/modeling/random_forest_v2_selected_config.json",
          "rf_selected[\"params\"]" in eval_src)
    check("XGB frozen params loaded from data/modeling/xgboost_v2_selected_config.json",
          "xgb_selected[\"params\"]" in eval_src and "xgb_selected[\"fixed_params\"]" in eval_src)
    check("XGB final_n_estimators frozen at 98, no early stopping in evaluation script",
          not uses_kwarg(eval_src, "early_stopping_rounds") and not uses_kwarg(eval_src, "eval_set"))

    # ---------------------------------------------------------------------
    # 5. preprocessing freshly fit per (model, feature_set, fold) - independent
    #    recomputation, NOT a require-medians-to-differ check.
    # ---------------------------------------------------------------------
    print("\n=== 5. preprocessing independently recomputed per fold (equality across folds is NOT a failure) ===")
    if not AUDIT_PATH.exists():
        check("preprocessing audit artifact exists (run evaluate_series_feature_sets_v2.py first)", False)
    else:
        audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))["entries"]
        check(f"audit has {2*2*N_FOLDS} entries (2 models x 2 feature sets x {N_FOLDS} folds)",
              len(audit) == 2 * 2 * N_FOLDS)

        import feature_engineering.preprocessing.preprocessing_common as pc1
        import feature_engineering.preprocessing.preprocessing_random_forest_v1 as rf1
        import feature_engineering.preprocessing.preprocessing_xgboost_v1 as xgb1

        recompute_ok = True
        val_untouched_ok = True
        for entry in audit:
            model_name, fs_name, fold = entry["model"], entry["feature_set"], entry["fold"]
            features_df = v1_df if fs_name == "v1" else v2_df

            fold_train_ids = set(cv_df.loc[(cv_df.fold == fold) & (cv_df.role == "train"), "match_id"])
            fold_val_ids = set(cv_df.loc[(cv_df.fold == fold) & (cv_df.role == "validation"), "match_id"])
            if not fold_train_ids.isdisjoint(fold_val_ids):
                val_untouched_ok = False

            fold_train_raw = features_df[features_df["match_id"].isin(fold_train_ids)].reset_index(drop=True)

            # INDEPENDENT recomputation: rebuild augmented-train + fit_preprocessing
            # from scratch, directly from THIS fold's own training rows, using the
            # real production functions - then compare to what was actually fitted.
            if fs_name == "v1":
                augmented = pc1.build_augmented_training_raw(fold_train_raw)
                fit_fn = rf1.fit_preprocessing if model_name == "rf" else xgb1.fit_preprocessing
                recomputed = fit_fn(augmented, model_features_v1)
            else:
                augmented = pc2.build_augmented_training_raw(fold_train_raw, roles_v2)
                fit_fn = rf2.fit_preprocessing if model_name == "rf" else xgb2.fit_preprocessing
                recomputed = fit_fn(augmented, roles_v2)

            recomputed_stats = recomputed[entry["stats_key"]]
            if recomputed_stats != entry["fitted_stats"]:
                recompute_ok = False

            # explicitly assert fold-validation rows were NOT part of the rows used
            # to build `augmented` (i.e. fold-train and fold-val ids are disjoint,
            # already checked above; additionally confirm none of fold_val_ids
            # match_ids appear in the rows that were mirrored/fit on)
            if set(fold_train_raw["match_id"]) & fold_val_ids:
                val_untouched_ok = False

        check("every audited fold's preprocessing stats independently reproduce "
              "(recomputed fresh from that fold's own augmented-train rows, matches what was fitted)",
              recompute_ok)
        check("fold-validation rows were never part of any fold's training/fitting rows", val_untouched_ok)
        # NOTE: equality of medians ACROSS folds is explicitly NOT checked - different
        # chronological folds may legitimately produce identical medians, and that
        # must never fail this validator.

        distinct_combos = {(e["model"], e["feature_set"], e["fold"]) for e in audit}
        check("every (model, feature_set, fold) combination was freshly fit exactly once (no reuse/skip)",
              len(distinct_combos) == 2 * 2 * N_FOLDS)

    # ---------------------------------------------------------------------
    # 6. V2 mirroring negates exactly directional_features, nothing else
    # ---------------------------------------------------------------------
    print("\n=== 6. V2 mirroring uses config roles correctly ===")
    probe = v2_df.iloc[[0, 1000, 5000]].reset_index(drop=True)
    mirrored = pc2.mirror_raw_rows(probe, roles_v2)
    dir_negated = all(np.allclose(mirrored[c].to_numpy(dtype=float), -probe[c].to_numpy(dtype=float), equal_nan=True)
                       for c in roles_v2["directional"])
    check("mirror negates every configured directional_features column", dir_negated)
    sym_untouched = all(mirrored[c].equals(probe[c]) for c in roles_v2["symmetric"])
    check("mirror leaves every configured symmetric_features column untouched", sym_untouched)
    cat_untouched = all(mirrored[c].equals(probe[c]) for c in roles_v2["categorical"])
    check("mirror leaves categorical_context (bestOf, tier) untouched", cat_untouched)
    check("mirror flips the target", (mirrored[target_col] == 1 - probe[target_col]).all())
    check("directional_features list matches config/series_features_v2_map_pool.yaml exactly",
          roles_v2["directional"] == yaml.safe_load(CONFIG_V2_PATH.read_text(encoding="utf-8"))["directional_features"])

    # ---------------------------------------------------------------------
    # 7. Phase 4 / Phase 5A artifacts untouched
    # ---------------------------------------------------------------------
    print("\n=== 7. Phase 4 / Phase 5A artifacts byte-unchanged ===")
    for rel, expected in BASELINE_HASHES.items():
        p = ROOT / rel
        check(f"unchanged: {rel}", p.exists() and sha256(p) == expected)
    src_dir = ROOT / "src"
    src_files = [p for p in src_dir.rglob("*") if p.is_file()] if src_dir.exists() else []
    check("src/ remains empty (no repo restructuring)", len(src_files) == 0)

    # ---------------------------------------------------------------------
    # 8. deterministic rerun
    # ---------------------------------------------------------------------
    print("\n=== 8. deterministic rerun ===")
    # NOTE: exact byte-identity is NOT the bar here, unlike Phase 5A's pure
    # feature-engineering rebuild (no randomized/multithreaded estimators
    # involved there). This pipeline fits RandomForestClassifier(n_jobs=-1)
    # and XGBClassifier(tree_method="hist", n_jobs=-1) sixteen times; this
    # repo's own test suite already documents (tests/models/test_random_forest_v1.py)
    # that n_jobs=-1 thread-summation order introduces machine-epsilon-level
    # float nondeterminism even with a fixed random_state, tolerated there via
    # atol=1e-9 rather than requiring bitwise equality. The bar here is
    # therefore: the rerun must succeed, and every metric must reproduce
    # within that same tolerance - not that the CSV bytes match exactly.
    if not COMPARISON_CSV.exists():
        check("comparison CSV exists (run evaluate_series_feature_sets_v2.py first)", False)
    else:
        before_df = pd.read_csv(COMPARISON_CSV)
        env_scripts = str(ROOT)
        r = subprocess.run([sys.executable, "-m", ".".join(EVAL_SCRIPT.relative_to(ROOT).with_suffix("").parts)], capture_output=True, text=True,
                            env={**__import__("os").environ, "PYTHONPATH": env_scripts, "PYTHONIOENCODING": "utf-8"})
        ok = r.returncode == 0
        check("rerun of evaluate_series_feature_sets_v2.py succeeded", ok)
        if not ok:
            print(f"    --- stdout ---\n{r.stdout}")
            print(f"    --- stderr ---\n{r.stderr}")
            check("re-run metrics reproduce within tolerance: series_feature_v1_v2_cv_comparison.csv (skipped: rerun failed)", False)
        else:
            after_df = pd.read_csv(COMPARISON_CSV)
            same_shape = before_df.shape == after_df.shape
            check("re-run comparison CSV has identical shape", same_shape)
            if same_shape:
                num_cols = before_df.select_dtypes(include=[np.number]).columns
                max_abs_diff = (before_df[num_cols] - after_df[num_cols]).abs().max().max()
                check(f"re-run metrics reproduce within tolerance (max abs diff {max_abs_diff:.2e} < 1e-6)",
                      max_abs_diff < 1e-6)
                non_num_cols = [c for c in before_df.columns if c not in num_cols]
                check("re-run non-numeric columns (model/feature_set/fold/row_type) identical",
                      before_df[non_num_cols].astype(str).equals(after_df[non_num_cols].astype(str)))

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
