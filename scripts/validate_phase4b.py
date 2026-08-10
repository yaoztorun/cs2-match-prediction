"""
Phase 4B validation (artifact-level, like validate_phase4a.py). Read-only.
Exits non-zero if any check fails.
"""

import hashlib
import json
import sys

import numpy as np
import pandas as pd
import yaml

from _common import ROOT, INTERIM, REPORTS
from models.random_forest_v1 import load_model
from preprocessing_random_forest_v1 import transform

CONFIG_PATH = ROOT / "config" / "series_features_v1.yaml"
FEATURES_PATH = ROOT / "data" / "features" / "series_features_v1.parquet"
SPLIT_PATH = ROOT / "data" / "modeling" / "series_split_v1.csv"
RF_PREPROC_PATH = ROOT / "data" / "modeling" / "random_forest_preprocessing_v1.json"
RF_MODEL_JSON_PATH = ROOT / "models" / "random_forest_v1.json"
RF_MODEL_JOBLIB_PATH = ROOT / "models" / "random_forest_v1.joblib"
SRC_DIR = ROOT / "src"

EXPECTED_TRAIN_N = 6619
EXPECTED_VAL_N = 1419
EXPECTED_TEST_N = 1418
EXPECTED_AUGMENTED_N = 13238

# sha256 hashes of the Logistic Regression (Phase 4A) artifacts, captured
# read-only immediately before any Phase 4B code was written - Phase 4B must
# never modify these files.
LR_BASELINE_HASHES = {
    "models/logistic_regression_scratch_v1.npz": "504c7f83d9e3162daa0680aeaaa2bf9e7051882e3c5cf17dc05cf9ab494402a3",
    "models/logistic_regression_scratch_v1.json": "584bb916f6260276d09245c8804dca386d32eecbba49691b193f819a6a0c0046",
    "data/modeling/logistic_preprocessing_v1.json": "d8dda783e3f029c31e9d03112c0d676a1947665d6409edcd272030211c09f972",
    "data/modeling/series_split_v1.csv": "fe1b947a3dd9829f1fd9b3e8ac8cc8ae796b8426ef728f609523ae8c48c0c253",
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

    # ---- Logistic Regression artifacts byte-identical to their Phase 4A state ----
    for rel_path, expected_hash in LR_BASELINE_HASHES.items():
        p = ROOT / rel_path
        actual = sha256(p) if p.exists() else None
        check(f"Logistic Regression artifact untouched: {rel_path}", actual == expected_hash)

    # ---- src/ remains untouched (still empty) ----
    src_files = list(SRC_DIR.rglob("*")) if SRC_DIR.exists() else []
    src_real_files = [p for p in src_files if p.is_file()]
    check("src/ subdirectories remain empty (no files added)", len(src_real_files) == 0)

    # ---- split manifest reused exactly, exact counts ----
    split = pd.read_csv(SPLIT_PATH, parse_dates=["datetime"])
    counts = split["split"].value_counts()
    check("train count == 6619 (Phase 4A split reused)", counts.get("train", 0) == EXPECTED_TRAIN_N)
    check("validation count == 1419", counts.get("validation", 0) == EXPECTED_VAL_N)
    check("test count == 1418", counts.get("test", 0) == EXPECTED_TEST_N)

    # ---- RF artifacts exist ----
    for p in [RF_PREPROC_PATH, RF_MODEL_JSON_PATH, RF_MODEL_JOBLIB_PATH]:
        check(f"artifact exists: {p.relative_to(ROOT)}", p.exists())

    rf_meta = json.loads(RF_MODEL_JSON_PATH.read_text(encoding="utf-8"))

    # ---- augmentation ----
    check("augmented_train_rows == 13238", rf_meta.get("augmented_train_rows") == EXPECTED_AUGMENTED_N)
    check("original_train_rows == 6619 (NOT described as 13238 matches)", rf_meta.get("original_train_rows") == EXPECTED_TRAIN_N)
    check("mirrored training target mean is exactly 0.5",
          abs(rf_meta.get("mirrored_train_target_mean", -1) - 0.5) < 1e-9)

    # ---- feature whitelist sourced from config, not hand-picked ----
    rf_preproc = json.loads(RF_PREPROC_PATH.read_text(encoding="utf-8"))
    check("RF preprocessing's original feature list matches config whitelist exactly",
          rf_preproc["original_model_feature_names"] == model_features)
    check("RF preprocessing applies no scaling", rf_preproc.get("scaling_applied") is False)

    # ---- model config ----
    check("model random_state == 42", rf_meta.get("random_state") == 42)
    check("feature_count == 19", rf_meta.get("feature_count") == 19)

    # ---- no test/Cologne metrics anywhere ----
    check("no test_metrics key in RF model metadata", "test_metrics" not in rf_meta)
    check("no cologne_metrics key in RF model metadata", "cologne_metrics" not in rf_meta)
    check("RF model metadata declares test_status == SEALED", rf_meta.get("test_status", "").startswith("SEALED"))
    check("RF model metadata declares cologne_status == UNTOUCHED", rf_meta.get("cologne_status", "").startswith("UNTOUCHED"))

    forbidden_patterns = ["*test_metric*", "*internal_test*", "*cologne_metric*", "*cologne_evaluation*"]
    stray_files = []
    for base in [REPORTS, ROOT / "data" / "modeling", ROOT / "models"]:
        if base.exists():
            for pat in forbidden_patterns:
                stray_files.extend(base.rglob(pat))
    check("no stray test-metrics/cologne-metrics artifact files exist", len(stray_files) == 0)

    # ---- no Cologne / post-Cologne match_ids anywhere in the split ----
    em = pd.read_csv(INTERIM / "evaluation_manifest.csv")
    cologne_ids = set(em.loc[em["evaluation_group"] == "cologne_2026", "match_id"])
    post_cologne_ids = set(em.loc[em["evaluation_group"] == "post_cologne", "match_id"])
    check("no cologne_2026 match_id in the split file", set(split["match_id"]).isdisjoint(cologne_ids))
    check("no post_cologne match_id in the split file", set(split["match_id"]).isdisjoint(post_cologne_ids))

    # ---- validation prediction count ----
    check("RF validation_metrics n == 1419", rf_meta.get("validation_metrics", {}).get("n") == EXPECTED_VAL_N)
    check("RF train_metrics n == 6619 (unmirrored)", rf_meta.get("train_metrics", {}).get("n") == EXPECTED_TRAIN_N)

    # ---- probabilities finite and in [0,1]; model reload reproduces them ----
    features = pd.read_parquet(FEATURES_PATH, engine="fastparquet")
    df = features.merge(split[["match_id", "split"]], on="match_id", how="inner")
    val_raw = df[df["split"] == "validation"].reset_index(drop=True)
    X_val, _ = transform(val_raw, rf_preproc)

    model = load_model(RF_MODEL_JOBLIB_PATH)
    proba_val = model.predict_proba(X_val)[:, 1]
    check("validation probabilities finite", np.isfinite(proba_val).all())
    check("validation probabilities in [0,1]", (proba_val >= 0).all() and (proba_val <= 1).all())

    model_reloaded = load_model(RF_MODEL_JOBLIB_PATH)
    proba_val_reloaded = model_reloaded.predict_proba(X_val)[:, 1]
    # Tolerance, not exact equality: with n_jobs=-1, RandomForestClassifier
    # averages 300 trees' predictions across threads, and floating-point
    # addition is not associative under thread-scheduling non-determinism -
    # verified this produces up to ~1 ULP (1e-16) of difference even calling
    # predict_proba twice on the SAME loaded model object, not just across a
    # reload. 1e-9 is tight enough to catch any real reload/serialization bug
    # while tolerating this expected, harmless floating-point noise.
    check("reloading the saved model reproduces the same validation probabilities (tol=1e-9)",
          np.allclose(proba_val, proba_val_reloaded, atol=1e-9))

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
