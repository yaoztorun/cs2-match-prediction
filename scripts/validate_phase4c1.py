"""
Phase 4C.1 validation (artifact-level). Read-only. Exits non-zero on failure.
"""

import ast
import hashlib
import json
import sys

import numpy as np
import pandas as pd
import yaml
from xgboost import XGBClassifier

from _common import ROOT, INTERIM, REPORTS
from preprocessing_xgboost_v1 import transform
from xgboost_tuning_v2 import (
    select_winner, derive_final_n_estimators, build_random_candidates,
    ANCHOR_CONFIGS, SEARCH_KEYS, N_RANDOM_CANDIDATES, LOG_LOSS_EQUIVALENCE_EPSILON,
)

CONFIG_PATH = ROOT / "config" / "series_features_v1.yaml"
FEATURES_PATH = ROOT / "data" / "features" / "series_features_v1.parquet"
SPLIT_PATH = ROOT / "data" / "modeling" / "series_split_v1.csv"
CV_FOLDS_PATH = ROOT / "data" / "modeling" / "random_forest_cv_folds_v2.csv"
TUNING_CSV = REPORTS / "tables" / "xgboost_tuning_v2.csv"
SELECTED_CONFIG = ROOT / "data" / "modeling" / "xgboost_v2_selected_config.json"
V2_PREPROC = ROOT / "data" / "modeling" / "xgboost_preprocessing_v2.json"
V2_MODEL = ROOT / "models" / "xgboost_v2.json"
V2_META = ROOT / "models" / "xgboost_v2_metadata.json"
TRAIN_SOURCE = ROOT / "scripts" / "train_xgboost_v2.py"
TUNING_SOURCE = ROOT / "scripts" / "xgboost_tuning_v2.py"
SRC_DIR = ROOT / "src"

EXPECTED_TRAIN_N, EXPECTED_VAL_N, EXPECTED_TEST_N = 6619, 1419, 1418
N_FOLDS = 4

# sha256 captured read-only before any Phase 4C.1 work began.
BASELINE_HASHES = {
    "models/logistic_regression_scratch_v1.npz": "504c7f83d9e3162daa0680aeaaa2bf9e7051882e3c5cf17dc05cf9ab494402a3",
    "models/logistic_regression_scratch_v1.json": "584bb916f6260276d09245c8804dca386d32eecbba49691b193f819a6a0c0046",
    "data/modeling/logistic_preprocessing_v1.json": "d8dda783e3f029c31e9d03112c0d676a1947665d6409edcd272030211c09f972",
    "data/modeling/series_split_v1.csv": "fe1b947a3dd9829f1fd9b3e8ac8cc8ae796b8426ef728f609523ae8c48c0c253",
    "models/random_forest_v1.joblib": "05b4cdd377694ad10a5ee8c163cfbaa3daa542c50802538031cf12ac85d051f7",
    "models/random_forest_v1.json": "8e1e11137fe7972d9a1f55de020d32e2f655b1733736dbfd1d11a99824412ffb",
    "data/modeling/random_forest_preprocessing_v1.json": "8415859db8d238f48bc04426872c8f97d8487678a4b8f0ab7706f8502e3c0111",
    "models/random_forest_v2.joblib": "e26e97fd8f1ea7676659605af2d9abd4d4e4cb0c5b767d1df506fb0a9cfac4a9",
    "models/random_forest_v2.json": "c5e527161925758718e5597b8ff730e67cdbb4626c3dee0065968be78499456d",
    "data/modeling/random_forest_preprocessing_v2.json": "8415859db8d238f48bc04426872c8f97d8487678a4b8f0ab7706f8502e3c0111",
    "data/modeling/random_forest_v2_selected_config.json": "3666622740fe27a8cb51647133d2707e421010e865ecc56ce04916ed2b422934",
    "data/modeling/random_forest_cv_folds_v2.csv": "152864c64ef558139af8b588d80e94102a13f52786275dc386357b52ac524247",
    "models/xgboost_v1.json": "9e9719a62b10b07b422683057a6f59ae5cd6a7ef367883e07f828ac41ec38794",
    "models/xgboost_v1_metadata.json": "42d47c175116eb0c3baa4e737b4b930db04fdd9808377bae3fca6ea1044f6a73",
    "data/modeling/xgboost_preprocessing_v1.json": "1bfa2b227b92200e2c205ed997679d54649b3a284126872079f68afef87f0f8e",
    "reports/tables/xgboost_feature_importance_v1.csv": "2ff0bc6e497106d9518f7363ae5b601d700dc0d123d17ae3bff6b1ba91255e86",
    "reports/tables/xgboost_permutation_importance_v1.csv": "10c7251fd321f86a53e5d47492bce95dcd9f9c63792019727a7d7eb600959f36",
    "reports/tables/xgboost_metrics_v1.csv": "3768d347f60d4ae782281ac5311e2fb816b19d7c31b641a2b99798a02b11a6f2",
    "reports/phase4c_xgboost_v1.md": "0200f07e174523cabe86d8a3ba66561a666f0d01cf7a8104b91d6cb3cd4f2642",
    "reports/xgboost_symmetry_v1.md": "cecaeb8144b345b6ff7f51273077b2514d99a50ebced66a3ff46e8c0d2454fcf",
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


def count_calls(source, func_name):
    n = 0
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            f = node.func
            if (isinstance(f, ast.Name) and f.id == func_name) or \
               (isinstance(f, ast.Attribute) and f.attr == func_name):
                n += 1
    return n


def main():
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    model_features = cfg["model_features"]

    # ---- every frozen artifact unchanged ----
    for rel, expected in BASELINE_HASHES.items():
        p = ROOT / rel
        check(f"unchanged: {rel}", (sha256(p) if p.exists() else None) == expected)

    src_files = [p for p in SRC_DIR.rglob("*") if p.is_file()] if SRC_DIR.exists() else []
    check("src/ remains empty", len(src_files) == 0)
    raw_dir = ROOT / "data" / "raw"
    check("data/raw/ present and readable", raw_dir.exists() and any(raw_dir.iterdir()))
    ref_dir = ROOT / "reference"
    check("reference/ present and readable", ref_dir.exists() and any(ref_dir.iterdir()))

    # ---- split unchanged / counts ----
    split = pd.read_csv(SPLIT_PATH)
    counts = split["split"].value_counts()
    check("train == 6619", counts.get("train", 0) == EXPECTED_TRAIN_N)
    check("validation == 1419", counts.get("validation", 0) == EXPECTED_VAL_N)
    check("test == 1418", counts.get("test", 0) == EXPECTED_TEST_N)

    train_ids = set(split.loc[split.split == "train", "match_id"])
    val_ids = set(split.loc[split.split == "validation", "match_id"])
    test_ids = set(split.loc[split.split == "test", "match_id"])

    # ---- CV folds: TRAIN-only, chronology, three-stage separation feasible ----
    cv = pd.read_csv(CV_FOLDS_PATH, parse_dates=["datetime"])
    cv_ids = set(cv["match_id"])
    check("candidate search used only original TRAIN ids", cv_ids <= train_ids)
    check("no main-validation id in the CV folds", cv_ids.isdisjoint(val_ids))
    check("no TEST id in the CV folds", cv_ids.isdisjoint(test_ids))

    em = pd.read_csv(INTERIM / "evaluation_manifest.csv")
    cologne = set(em.loc[em.evaluation_group == "cologne_2026", "match_id"])
    post = set(em.loc[em.evaluation_group == "post_cologne", "match_id"])
    check("no cologne_2026 id in the CV folds", cv_ids.isdisjoint(cologne))
    check("no post_cologne id in the CV folds", cv_ids.isdisjoint(post))

    chrono_ok = all(
        cv.loc[(cv.fold == f) & (cv.role == "train"), "datetime"].max()
        < cv.loc[(cv.fold == f) & (cv.role == "validation"), "datetime"].min()
        for f in range(1, N_FOLDS + 1)
    )
    check("outer fold chronology holds for every fold", chrono_ok)

    # three-stage separation, recomputed from the real data
    from xgboost_tuning_v2 import split_inner_early_stop
    feats = pd.read_parquet(FEATURES_PATH, engine="fastparquet")
    three_stage_ok = True
    for f in range(1, N_FOLDS + 1):
        tid = set(cv.loc[(cv.fold == f) & (cv.role == "train"), "match_id"])
        vid = set(cv.loc[(cv.fold == f) & (cv.role == "validation"), "match_id"])
        ft = feats[feats.match_id.isin(tid)].reset_index(drop=True)
        fv = feats[feats.match_id.isin(vid)].reset_index(drop=True)
        inner_fit, inner_es = split_inner_early_stop(ft)
        if not (inner_fit["datetime"].max() < inner_es["datetime"].min()):
            three_stage_ok = False
        if not (inner_es["datetime"].max() < fv["datetime"].min()):
            three_stage_ok = False
        if not set(inner_es.match_id).isdisjoint(vid):
            three_stage_ok = False
    check("three-stage separation holds: inner_fit < inner_early_stop < outer_fold_validation", three_stage_ok)

    # ---- tuning table shape + deterministic candidates ----
    tuning = pd.read_csv(TUNING_CSV)
    fold_rows = tuning[tuning.row_type == "fold"]
    agg_rows = tuning[tuning.row_type == "aggregate"]
    n_cand = len(ANCHOR_CONFIGS) + N_RANDOM_CANDIDATES
    check(f"tuning CSV has {n_cand * N_FOLDS} fold rows", len(fold_rows) == n_cand * N_FOLDS)
    check(f"tuning CSV has {n_cand} aggregate rows", len(agg_rows) == n_cand)

    recomputed_ids = sorted(c["candidate_id"] for c in build_random_candidates())
    saved_random_ids = sorted(set(agg_rows.candidate_id) - {a["candidate_id"] for a in ANCHOR_CONFIGS})
    check("random-candidate ids reproduce deterministically", recomputed_ids == saved_random_ids)
    check("V1 structural reference was in the candidate pool",
          "xgb_v1_structure_reference" in set(agg_rows.candidate_id))

    # ---- selection independently recomputed from the saved tuning table ----
    params_by_id = {}
    for _, r in agg_rows.iterrows():
        params_by_id[r["candidate_id"]] = {k: r[k] for k in SEARCH_KEYS} | {"candidate_id": r["candidate_id"]}
    recomputed_winner, recomputed_stage = select_winner(agg_rows.reset_index(drop=True), params_by_id)

    selected = json.loads(SELECTED_CONFIG.read_text(encoding="utf-8"))
    check("recomputed selection matches the frozen candidate_id",
          recomputed_winner == selected["selected_candidate_id"])
    check("recomputed selection stage matches", recomputed_stage == selected["selection_stage"])
    check("log-loss epsilon recorded as 0.002", selected["log_loss_epsilon"] == LOG_LOSS_EQUIVALENCE_EPSILON)
    check("selected config records main_validation_used_in_selection == False",
          selected.get("main_validation_used_in_selection") is False)

    # ---- final_n_estimators independently recomputed ----
    winner_folds = fold_rows[fold_rows.candidate_id == selected["selected_candidate_id"]].sort_values("fold")
    best_iters = winner_folds["best_iteration"].astype(int).tolist()
    check("best_iterations_by_fold matches the tuning table", best_iters == selected["best_iterations_by_fold"])
    check("final_n_estimators == round(median(best_iteration+1)) recomputed independently",
          derive_final_n_estimators(best_iters) == selected["final_n_estimators"])

    # ---- exactly one frozen config, used by the final model ----
    meta = json.loads(V2_META.read_text(encoding="utf-8"))
    check("V2 metadata references the frozen candidate", meta.get("cv_candidate_id") == selected["selected_candidate_id"])
    check("V2 model n_estimators == frozen final_n_estimators",
          meta.get("n_estimators") == selected["final_n_estimators"])
    check("V2 trees actually built == frozen final_n_estimators",
          meta.get("trees_actually_built") == selected["final_n_estimators"])
    for k, v in selected["params"].items():
        check(f"V2 metadata hyperparameter {k} matches frozen config", meta.get(k) == v)

    # ---- no early stopping / eval_set in the FINAL refit ----
    train_src = TRAIN_SOURCE.read_text(encoding="utf-8")
    check("final refit passes no early_stopping_rounds", not uses_kwarg(train_src, "early_stopping_rounds"))
    check("final refit passes no eval_set", not uses_kwarg(train_src, "eval_set"))
    check("final refit constructs exactly one XGBClassifier", count_calls(train_src, "XGBClassifier") == 1)
    check("V2 metadata declares early_stopping_used_in_final_refit == False",
          meta.get("early_stopping_used_in_final_refit") is False)
    check("V2 metadata declares eval_set_used_in_final_refit == False",
          meta.get("eval_set_used_in_final_refit") is False)

    # the tuning script SHOULD use early stopping (inside TRAIN folds) - confirm it does
    tuning_src = TUNING_SOURCE.read_text(encoding="utf-8")
    check("tuning script does use eval_set (inner early-stop block, inside TRAIN)",
          uses_kwarg(tuning_src, "eval_set"))
    # Precise check: the tuning script must never READ the main split file. It may
    # legitimately import a helper from build_series_split_v1 and mention the
    # filename in comments/report prose - neither loads validation/test rows. So
    # scan the AST for a pandas read of that path rather than matching the string.
    tuning_reads_split = False
    for node in ast.walk(ast.parse(tuning_src)):
        if isinstance(node, ast.Call):
            f = node.func
            fname = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else "")
            if fname in {"read_csv", "read_parquet"}:
                for arg in list(node.args) + [kw.value for kw in node.keywords]:
                    src_seg = ast.unparse(arg)
                    if "SPLIT_PATH" in src_seg or "series_split_v1" in src_seg:
                        tuning_reads_split = True
    check("tuning script never READS series_split_v1.csv (main validation structurally absent)",
          not tuning_reads_split)

    # ---- preprocessing / features ----
    preproc = json.loads(V2_PREPROC.read_text(encoding="utf-8"))
    check("V2 preprocessing whitelist matches the YAML config",
          preproc["original_model_feature_names"] == model_features)
    check("V2 feature_count == 19", meta.get("feature_count") == 19)
    check("V2 NaN policy unchanged", preproc.get("missing_value_policy") == "preserve_nan_native_xgboost")
    check("V2 applies no scaling", preproc.get("scaling_applied") is False)
    check("V2 applies no imputation", preproc.get("imputation_applied") is False)

    # ---- mirroring accounting ----
    check("unique_training_matches == 6619", meta.get("unique_training_matches") == EXPECTED_TRAIN_N)
    check("augmented_training_observations == 13238", meta.get("augmented_training_observations") == 13238)
    check("mirrored target mean exactly 0.5", abs(meta.get("mirrored_train_target_mean", -1) - 0.5) < 1e-12)

    # ---- evaluation scope ----
    check("validation_metrics n == 1419", meta.get("validation_metrics", {}).get("n") == EXPECTED_VAL_N)
    check("train_metrics n == 6619", meta.get("train_metrics", {}).get("n") == EXPECTED_TRAIN_N)
    check("no test_metrics key", "test_metrics" not in meta)
    check("no cologne_metrics key", "cologne_metrics" not in meta)
    check("test_status == SEALED", meta.get("test_status", "").startswith("SEALED"))
    check("cologne_status == UNTOUCHED", meta.get("cologne_status", "").startswith("UNTOUCHED"))

    stray = []
    for base in [REPORTS, ROOT / "data" / "modeling", ROOT / "models"]:
        if base.exists():
            for pat in ["*test_metric*", "*internal_test*", "*test_prediction*",
                         "*cologne_metric*", "*cologne_evaluation*", "*cologne_prediction*"]:
                stray.extend(base.rglob(pat))
    check("no stray test/Cologne artifacts", len(stray) == 0)

    # ---- probabilities + reload ----
    df = feats.merge(split[["match_id", "split"]], on="match_id", how="inner")
    val_raw = df[df["split"] == "validation"].reset_index(drop=True)
    X_val, names = transform(val_raw, preproc)
    check("transform yields 19 columns on real validation data", X_val.shape[1] == 19)
    check("validation row count == 1419", X_val.shape[0] == EXPECTED_VAL_N)

    m = XGBClassifier()
    m.load_model(str(V2_MODEL))
    proba = m.predict_proba(X_val)[:, 1]
    check("validation probabilities finite", np.isfinite(proba).all())
    check("validation probabilities in [0,1]", (proba >= 0).all() and (proba <= 1).all())

    m2 = XGBClassifier()
    m2.load_model(str(V2_MODEL))
    proba2 = m2.predict_proba(X_val)[:, 1]
    check("reload reproduces validation probabilities (tol=1e-9)", np.allclose(proba, proba2, atol=1e-9))
    check("reloaded artifact exposes predict_proba", hasattr(m, "predict_proba"))

    n_pass = sum(1 for _, ok in CHECKS if ok)
    print(f"\n{n_pass}/{len(CHECKS)} checks passed")
    if n_pass != len(CHECKS):
        sys.exit(1)


if __name__ == "__main__":
    main()
