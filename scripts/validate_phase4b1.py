"""
Phase 4B.1 validation (artifact-level, like validate_phase4a/4b.py). Read-only.
Exits non-zero if any check fails.
"""

import hashlib
import json
import sys

import joblib
import numpy as np
import pandas as pd
import yaml

from _common import ROOT, INTERIM, REPORTS
from preprocessing_random_forest_v1 import transform, fit_preprocessing, build_augmented_training_raw
from random_forest_tuning_v2 import build_random_candidates, select_winner, ANCHOR_CONFIGS, N_RANDOM_CANDIDATES

CONFIG_PATH = ROOT / "config" / "series_features_v1.yaml"
FEATURES_PATH = ROOT / "data" / "features" / "series_features_v1.parquet"
SPLIT_PATH = ROOT / "data" / "modeling" / "series_split_v1.csv"
CV_FOLDS_PATH = ROOT / "data" / "modeling" / "random_forest_cv_folds_v2.csv"
TUNING_CSV_PATH = ROOT / "reports" / "tables" / "random_forest_tuning_v2.csv"
SELECTED_CONFIG_PATH = ROOT / "data" / "modeling" / "random_forest_v2_selected_config.json"
RF_V2_PREPROC_PATH = ROOT / "data" / "modeling" / "random_forest_preprocessing_v2.json"
RF_V2_MODEL_JSON_PATH = ROOT / "models" / "random_forest_v2.json"
RF_V2_MODEL_JOBLIB_PATH = ROOT / "models" / "random_forest_v2.joblib"
SRC_DIR = ROOT / "src"

EXPECTED_TRAIN_N = 6619
EXPECTED_VAL_N = 1419
EXPECTED_TEST_N = 1418
N_FOLDS = 4

# sha256 hashes of Logistic Regression + Random Forest V1 artifacts, captured
# read-only immediately before any Phase 4B.1 code was written.
BASELINE_HASHES = {
    "models/logistic_regression_scratch_v1.npz": "504c7f83d9e3162daa0680aeaaa2bf9e7051882e3c5cf17dc05cf9ab494402a3",
    "models/logistic_regression_scratch_v1.json": "584bb916f6260276d09245c8804dca386d32eecbba49691b193f819a6a0c0046",
    "data/modeling/logistic_preprocessing_v1.json": "d8dda783e3f029c31e9d03112c0d676a1947665d6409edcd272030211c09f972",
    "data/modeling/series_split_v1.csv": "fe1b947a3dd9829f1fd9b3e8ac8cc8ae796b8426ef728f609523ae8c48c0c253",
    "models/random_forest_v1.joblib": "05b4cdd377694ad10a5ee8c163cfbaa3daa542c50802538031cf12ac85d051f7",
    "models/random_forest_v1.json": "8e1e11137fe7972d9a1f55de020d32e2f655b1733736dbfd1d11a99824412ffb",
    "data/modeling/random_forest_preprocessing_v1.json": "8415859db8d238f48bc04426872c8f97d8487678a4b8f0ab7706f8502e3c0111",
    "reports/tables/random_forest_feature_importance_v1.csv": "9ad245fa6789d940731a0620db5e13f8943717361d57faa6ae75f162026113b8",
    "reports/tables/random_forest_permutation_importance_v1.csv": "aaa02a1fb088c8bd326a65a9569d2672ab945491a78d3b5ba517a25a1e2567a7",
    "reports/tables/random_forest_metrics_v1.csv": "9d91d6eeff8b68c9d86072230af0966178289a20b3f1ae95548a29b93d3bd3c2",
    "reports/phase4b_random_forest_v1.md": "5b4e17fa0db4bbe3a50e1d5908a57c69e21879cba8fb53098fdf07d414370a61",
}

CHECKS = []


def check(name, condition):
    CHECKS.append((name, bool(condition)))
    print(f"[{'PASS' if condition else 'FAIL'}] {name}")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    model_features = cfg["model_features"]

    # ---- V1 / LR artifacts byte-identical to their prior state ----
    for rel_path, expected_hash in BASELINE_HASHES.items():
        p = ROOT / rel_path
        actual = sha256(p) if p.exists() else None
        check(f"unchanged: {rel_path}", actual == expected_hash)

    src_files = [p for p in SRC_DIR.rglob("*") if p.is_file()] if SRC_DIR.exists() else []
    check("src/ subdirectories remain empty", len(src_files) == 0)

    # ---- split reused, exact counts ----
    split = pd.read_csv(SPLIT_PATH)
    counts = split["split"].value_counts()
    check("train count == 6619", counts.get("train", 0) == EXPECTED_TRAIN_N)
    check("validation count == 1419", counts.get("validation", 0) == EXPECTED_VAL_N)
    check("test count == 1418", counts.get("test", 0) == EXPECTED_TEST_N)

    train_ids = set(split.loc[split["split"] == "train", "match_id"])
    val_ids = set(split.loc[split["split"] == "validation", "match_id"])
    test_ids = set(split.loc[split["split"] == "test", "match_id"])

    # ---- CV fold file: only TRAIN match_ids, no main validation/test/Cologne info ----
    cv_df = pd.read_csv(CV_FOLDS_PATH, parse_dates=["datetime"])
    cv_ids = set(cv_df["match_id"])
    check("CV fold file uses ONLY original TRAIN match_ids", cv_ids <= train_ids)
    check("CV fold file contains ZERO main-validation match_ids", cv_ids.isdisjoint(val_ids))
    check("CV fold file contains ZERO test match_ids", cv_ids.isdisjoint(test_ids))

    em = pd.read_csv(INTERIM / "evaluation_manifest.csv")
    cologne_ids = set(em.loc[em["evaluation_group"] == "cologne_2026", "match_id"])
    post_cologne_ids = set(em.loc[em["evaluation_group"] == "post_cologne", "match_id"])
    check("CV fold file contains ZERO cologne_2026 match_ids", cv_ids.isdisjoint(cologne_ids))
    check("CV fold file contains ZERO post_cologne match_ids", cv_ids.isdisjoint(post_cologne_ids))

    # ---- fold chronology + no group crossing, recomputed directly from the saved CSV ----
    chrono_ok = True
    for fold in range(1, N_FOLDS + 1):
        ft_max = cv_df.loc[(cv_df["fold"] == fold) & (cv_df["role"] == "train"), "datetime"].max()
        fv_min = cv_df.loc[(cv_df["fold"] == fold) & (cv_df["role"] == "validation"), "datetime"].min()
        if not (ft_max < fv_min):
            chrono_ok = False
    check("every fold: fold-train max datetime < fold-validation min datetime", chrono_ok)

    # a match_id must never appear as both train and validation WITHIN the same fold
    overlap_ok = True
    for fold in range(1, N_FOLDS + 1):
        ft_ids = set(cv_df.loc[(cv_df["fold"] == fold) & (cv_df["role"] == "train"), "match_id"])
        fv_ids = set(cv_df.loc[(cv_df["fold"] == fold) & (cv_df["role"] == "validation"), "match_id"])
        if not ft_ids.isdisjoint(fv_ids):
            overlap_ok = False
    check("no match_id is both fold-train and fold-validation within the same fold", overlap_ok)

    # no datetime group split across roles within any single fold's train/val boundary
    features_df = pd.read_parquet(FEATURES_PATH, engine="fastparquet")
    merged_dt = features_df[["match_id", "datetime"]].merge(cv_df[["match_id", "fold", "role"]], on="match_id")
    group_ok = True
    for fold in range(1, N_FOLDS + 1):
        sub = merged_dt[merged_dt["fold"] == fold]
        per_dt_role = sub.groupby("datetime")["role"].nunique()
        if not (per_dt_role == 1).all():
            group_ok = False
    check("no datetime group split across fold-train/fold-validation", group_ok)

    # ---- tuning results shape ----
    tuning_df = pd.read_csv(TUNING_CSV_PATH)
    fold_rows = tuning_df[tuning_df["row_type"] == "fold"]
    agg_rows = tuning_df[tuning_df["row_type"] == "aggregate"]
    n_candidates = len(ANCHOR_CONFIGS) + N_RANDOM_CANDIDATES
    check(f"tuning CSV has {n_candidates * N_FOLDS} fold rows", len(fold_rows) == n_candidates * N_FOLDS)
    check(f"tuning CSV has {n_candidates} aggregate rows", len(agg_rows) == n_candidates)

    # ---- candidate search determinism: independent re-derivation matches the saved CSV ----
    recomputed_random = build_random_candidates()
    saved_random_ids = sorted(set(agg_rows["candidate_id"]) - {a["candidate_id"] for a in ANCHOR_CONFIGS})
    recomputed_ids = sorted(c["candidate_id"] for c in recomputed_random)
    check("recomputed random-candidate id list matches the saved tuning results exactly", saved_random_ids == recomputed_ids)

    first_random_saved = agg_rows[agg_rows["candidate_id"] == "random_001"].iloc[0]
    first_random_recomputed = recomputed_random[0]
    # max_features is a mixed str/float column ("sqrt", 0.5, 0.75, 1.0) - CSV
    # round-trips numeric entries as strings (e.g. '1.0'), so compare via str()
    # rather than raw equality to avoid a false mismatch on dtype alone.
    check("recomputed random_001 hyperparameters match the saved tuning results",
          (int(first_random_saved["n_estimators"]) == first_random_recomputed["n_estimators"]) and
          (str(first_random_saved["max_features"]) == str(first_random_recomputed["max_features"])) and
          (int(first_random_saved["min_samples_leaf"]) == first_random_recomputed["min_samples_leaf"]))

    # ---- selection independently recomputed from the tuning CSV matches the saved config ----
    params_by_id = {}
    for _, r in agg_rows.iterrows():
        max_depth = None if pd.isna(r["max_depth"]) else int(r["max_depth"])
        params_by_id[r["candidate_id"]] = {
            "n_estimators": int(r["n_estimators"]), "max_depth": max_depth,
            "min_samples_leaf": int(r["min_samples_leaf"]), "min_samples_split": int(r["min_samples_split"]),
            "max_features": r["max_features"],
        }
    tunable_agg = agg_rows[agg_rows["candidate_id"] != "v1_reference_unrestricted"].reset_index(drop=True)
    recomputed_winner_id, recomputed_stage = select_winner(tunable_agg, params_by_id)

    selected = json.loads(SELECTED_CONFIG_PATH.read_text(encoding="utf-8"))
    check("independently recomputed selection matches the saved selected config (candidate_id)",
          recomputed_winner_id == selected["candidate_id"])
    check("independently recomputed selection matches the saved selection stage",
          recomputed_stage == selected["selection_stage"])

    # ---- RF V2 artifacts ----
    for p in [RF_V2_PREPROC_PATH, RF_V2_MODEL_JSON_PATH, RF_V2_MODEL_JOBLIB_PATH]:
        check(f"artifact exists: {p.relative_to(ROOT)}", p.exists())

    rf2_meta = json.loads(RF_V2_MODEL_JSON_PATH.read_text(encoding="utf-8"))
    check("RF V2 random_state == 42", rf2_meta.get("random_state") == 42)
    check("RF V2 feature_count == 19", rf2_meta.get("feature_count") == 19)
    check("RF V2 config matches the frozen selected config",
          all(rf2_meta.get(k) == v for k, v in selected["params"].items()))
    check("RF V2 augmented_train_rows == 13238", rf2_meta.get("augmented_train_rows") == 13238)
    check("RF V2 original_train_rows == 6619 (not described as 13238 matches)", rf2_meta.get("original_train_rows") == EXPECTED_TRAIN_N)
    check("RF V2 mirrored training target mean is exactly 0.5",
          abs(rf2_meta.get("mirrored_train_target_mean", -1) - 0.5) < 1e-9)
    check("RF V2 validation_metrics n == 1419", rf2_meta.get("validation_metrics", {}).get("n") == EXPECTED_VAL_N)
    check("no test_metrics key in RF V2 metadata", "test_metrics" not in rf2_meta)
    check("no cologne_metrics key in RF V2 metadata", "cologne_metrics" not in rf2_meta)
    check("RF V2 metadata declares test_status == SEALED", rf2_meta.get("test_status", "").startswith("SEALED"))
    check("RF V2 metadata declares cologne_status == UNTOUCHED", rf2_meta.get("cologne_status", "").startswith("UNTOUCHED"))

    forbidden_patterns = ["*test_metric*", "*internal_test*", "*cologne_metric*", "*cologne_evaluation*"]
    stray = []
    for base in [REPORTS, ROOT / "data" / "modeling", ROOT / "models"]:
        if base.exists():
            for pat in forbidden_patterns:
                stray.extend(base.rglob(pat))
    check("no stray test-metrics/cologne-metrics artifact files exist", len(stray) == 0)

    # ---- probabilities finite/[0,1]; reload reproduces them ----
    rf2_preproc = json.loads(RF_V2_PREPROC_PATH.read_text(encoding="utf-8"))
    df = features_df.merge(split[["match_id", "split"]], on="match_id", how="inner")
    val_raw = df[df["split"] == "validation"].reset_index(drop=True)
    X_val, _ = transform(val_raw, rf2_preproc)

    model = joblib.load(RF_V2_MODEL_JOBLIB_PATH)
    proba = model.predict_proba(X_val)[:, 1]
    check("RF V2 validation probabilities finite", np.isfinite(proba).all())
    check("RF V2 validation probabilities in [0,1]", (proba >= 0).all() and (proba <= 1).all())

    model_reloaded = joblib.load(RF_V2_MODEL_JOBLIB_PATH)
    proba_reloaded = model_reloaded.predict_proba(X_val)[:, 1]
    check("reloading RF V2 reproduces the same validation probabilities (tol=1e-9)",
          np.allclose(proba, proba_reloaded, atol=1e-9))

    # ---- fold preprocessing is genuinely fit per-fold (real-data check) ----
    fold1_train_ids = set(cv_df.loc[(cv_df["fold"] == 1) & (cv_df["role"] == "train"), "match_id"])
    fold4_train_ids = set(cv_df.loc[(cv_df["fold"] == 4) & (cv_df["role"] == "train"), "match_id"])
    fold1_raw = features_df[features_df["match_id"].isin(fold1_train_ids)]
    fold4_raw = features_df[features_df["match_id"].isin(fold4_train_ids)]
    fold1_params = fit_preprocessing(build_augmented_training_raw(fold1_raw), model_features)
    fold4_params = fit_preprocessing(build_augmented_training_raw(fold4_raw), model_features)
    recomputed_fold1_median = float(build_augmented_training_raw(fold1_raw)["elo_diff"].median())
    check("fold 1 preprocessing median matches independent recomputation from fold 1's own data only",
          abs(fold1_params["train_medians"]["elo_diff"] - recomputed_fold1_median) < 1e-9)
    check("fold 1 and fold 4 preprocessing were fit independently (different underlying data)",
          fold1_train_ids != fold4_train_ids)

    # ---- data/raw/ untouched ----
    raw_dir = ROOT / "data" / "raw"
    check("data/raw/ still present and readable", raw_dir.exists() and any(raw_dir.iterdir()))

    n_pass = sum(1 for _, ok in CHECKS if ok)
    n_total = len(CHECKS)
    print(f"\n{n_pass}/{n_total} checks passed")
    if n_pass != n_total:
        sys.exit(1)


if __name__ == "__main__":
    main()
