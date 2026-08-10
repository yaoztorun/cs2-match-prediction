"""
Phase 4C validation (artifact-level, like validate_phase4a/4b/4b1.py).
Read-only. Exits non-zero if any check fails.
"""

import ast
import hashlib
import json
import sys

import numpy as np
import pandas as pd
import yaml

from _common import ROOT, INTERIM, REPORTS
from models.xgboost_v1 import load_model, XGB_CONFIG
from preprocessing_xgboost_v1 import transform

CONFIG_PATH = ROOT / "config" / "series_features_v1.yaml"
FEATURES_PATH = ROOT / "data" / "features" / "series_features_v1.parquet"
SPLIT_PATH = ROOT / "data" / "modeling" / "series_split_v1.csv"
XGB_PREPROC_PATH = ROOT / "data" / "modeling" / "xgboost_preprocessing_v1.json"
XGB_MODEL_PATH = ROOT / "models" / "xgboost_v1.json"
XGB_META_PATH = ROOT / "models" / "xgboost_v1_metadata.json"
TRAIN_SOURCE = ROOT / "scripts" / "train_xgboost_v1.py"
MODEL_SOURCE = ROOT / "scripts" / "models" / "xgboost_v1.py"
SRC_DIR = ROOT / "src"

EXPECTED_TRAIN_N = 6619
EXPECTED_VAL_N = 1419
EXPECTED_TEST_N = 1418
EXPECTED_AUGMENTED_N = 13238

# sha256 of every Logistic Regression / Random Forest artifact, captured
# read-only immediately before any Phase 4C code was written.
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
    "reports/phase4b_random_forest_v1.md": "5b4e17fa0db4bbe3a50e1d5908a57c69e21879cba8fb53098fdf07d414370a61",
    "reports/phase4b1_random_forest_v2_results.md": "834e91360297e7092d140e27a813b86155871247d39f0f08bbd0b42795b124c6",
}

CHECKS = []


def check(name, condition):
    CHECKS.append((name, bool(condition)))
    print(f"[{'PASS' if condition else 'FAIL'}] {name}")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def uses_keyword_argument(source, kwarg_name):
    """True if `kwarg_name` is passed as a keyword argument to ANY call in the
    module. Precise by construction: a metadata dict key like
    `"eval_set_used": False`, or the words appearing in report prose, are NOT
    keyword arguments and correctly do not trigger this."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg == kwarg_name:
                    return True
    return False


def count_call_sites(source, func_name):
    """AST-accurate count of actual invocations of `func_name` (imports and
    mere mentions do not count)."""
    tree = ast.parse(source)
    n = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if (isinstance(f, ast.Name) and f.id == func_name) or \
               (isinstance(f, ast.Attribute) and f.attr == func_name):
                n += 1
    return n


def main():
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    model_features = cfg["model_features"]

    # ---- LR / RF artifacts byte-identical ----
    for rel_path, expected in BASELINE_HASHES.items():
        p = ROOT / rel_path
        check(f"unchanged: {rel_path}", (sha256(p) if p.exists() else None) == expected)

    src_files = [p for p in SRC_DIR.rglob("*") if p.is_file()] if SRC_DIR.exists() else []
    check("src/ remains empty (no files added)", len(src_files) == 0)

    # ---- split reuse ----
    split = pd.read_csv(SPLIT_PATH)
    counts = split["split"].value_counts()
    check("train count == 6619", counts.get("train", 0) == EXPECTED_TRAIN_N)
    check("validation count == 1419", counts.get("validation", 0) == EXPECTED_VAL_N)
    check("test count == 1418", counts.get("test", 0) == EXPECTED_TEST_N)

    # ---- XGB artifacts exist ----
    for p in [XGB_PREPROC_PATH, XGB_MODEL_PATH, XGB_META_PATH]:
        check(f"artifact exists: {p.relative_to(ROOT)}", p.exists())

    meta = json.loads(XGB_META_PATH.read_text(encoding="utf-8"))
    preproc = json.loads(XGB_PREPROC_PATH.read_text(encoding="utf-8"))

    # ---- feature whitelist from YAML ----
    check("preprocessing feature whitelist matches config/series_features_v1.yaml exactly",
          preproc["original_model_feature_names"] == model_features)
    check("feature_count == 19", meta.get("feature_count") == 19)
    check("no standardization applied", preproc.get("scaling_applied") is False)
    check("no imputation applied (NaN preserved for native XGBoost handling)",
          preproc.get("imputation_applied") is False)
    check("missing-value policy recorded", preproc.get("missing_value_policy") == "preserve_nan_native_xgboost")

    # ---- fixed configuration, no tuning, no early stopping ----
    check("model random_state == 42", meta.get("random_state") == 42)
    check("saved config matches the fixed XGB_CONFIG exactly",
          all(meta.get(k) == v for k, v in XGB_CONFIG.items()))
    check("early_stopping_used == False", meta.get("early_stopping_used") is False)
    check("eval_set_used == False", meta.get("eval_set_used") is False)

    train_src = TRAIN_SOURCE.read_text(encoding="utf-8")
    model_src = MODEL_SOURCE.read_text(encoding="utf-8")
    check("no early_stopping_rounds keyword argument is passed anywhere",
          not uses_keyword_argument(train_src, "early_stopping_rounds")
          and not uses_keyword_argument(model_src, "early_stopping_rounds"))
    check("no eval_set keyword argument is passed anywhere",
          not uses_keyword_argument(train_src, "eval_set")
          and not uses_keyword_argument(model_src, "eval_set"))
    check("no early_stopping_rounds/eval_set set via XGB_CONFIG",
          "early_stopping_rounds" not in XGB_CONFIG and "eval_set" not in XGB_CONFIG)
    check("exactly one XGBoost configuration was trained (single build_model() invocation)",
          count_call_sites(train_src, "build_model") == 1)
    check("training source constructs no XGBClassifier directly (only via build_model)",
          count_call_sites(train_src, "XGBClassifier") == 0)

    # ---- mirroring accounting: observations vs unique matches never conflated ----
    check("unique_training_matches == 6619", meta.get("unique_training_matches") == EXPECTED_TRAIN_N)
    check("augmented_training_observations == 13238", meta.get("augmented_training_observations") == EXPECTED_AUGMENTED_N)
    check("mirrored training target mean is exactly 0.5",
          abs(meta.get("mirrored_train_target_mean", -1) - 0.5) < 1e-12)

    # ---- evaluation scope ----
    check("validation_metrics n == 1419", meta.get("validation_metrics", {}).get("n") == EXPECTED_VAL_N)
    check("train_metrics n == 6619 (unmirrored)", meta.get("train_metrics", {}).get("n") == EXPECTED_TRAIN_N)
    check("no test_metrics key in metadata", "test_metrics" not in meta)
    check("no cologne_metrics key in metadata", "cologne_metrics" not in meta)
    check("metadata declares test_status == SEALED", meta.get("test_status", "").startswith("SEALED"))
    check("metadata declares cologne_status == UNTOUCHED", meta.get("cologne_status", "").startswith("UNTOUCHED"))

    forbidden_patterns = ["*test_metric*", "*internal_test*", "*test_prediction*",
                           "*cologne_metric*", "*cologne_evaluation*", "*cologne_prediction*"]
    stray = []
    for base in [REPORTS, ROOT / "data" / "modeling", ROOT / "models"]:
        if base.exists():
            for pat in forbidden_patterns:
                stray.extend(base.rglob(pat))
    check("no stray test/Cologne metrics or prediction artifacts exist", len(stray) == 0)

    # ---- no Cologne / post-Cologne ids in the split at all ----
    em = pd.read_csv(INTERIM / "evaluation_manifest.csv")
    cologne_ids = set(em.loc[em["evaluation_group"] == "cologne_2026", "match_id"])
    post_ids = set(em.loc[em["evaluation_group"] == "post_cologne", "match_id"])
    check("no cologne_2026 match_id in the split", set(split["match_id"]).isdisjoint(cologne_ids))
    check("no post_cologne match_id in the split", set(split["match_id"]).isdisjoint(post_ids))

    # ---- probabilities finite/[0,1]; native reload reproduces them ----
    features_df = pd.read_parquet(FEATURES_PATH, engine="fastparquet")
    df = features_df.merge(split[["match_id", "split"]], on="match_id", how="inner")
    val_raw = df[df["split"] == "validation"].reset_index(drop=True)
    X_val, names = transform(val_raw, preproc)
    check("transform produces 19 columns on real validation data", X_val.shape[1] == 19)
    check("validation prediction count == 1419", X_val.shape[0] == EXPECTED_VAL_N)

    model = load_model(XGB_MODEL_PATH)
    proba = model.predict_proba(X_val)[:, 1]
    check("validation probabilities finite", np.isfinite(proba).all())
    check("validation probabilities in [0,1]", (proba >= 0).all() and (proba <= 1).all())

    model2 = load_model(XGB_MODEL_PATH)
    proba2 = model2.predict_proba(X_val)[:, 1]
    check("reloading the saved model reproduces validation probabilities (tol=1e-9)",
          np.allclose(proba, proba2, atol=1e-9))
    check("reloaded artifact exposes predict_proba (classifier interface, not bare Booster)",
          hasattr(model, "predict_proba"))

    # ---- NaN genuinely preserved end-to-end on real data ----
    check("NaN preserved in real validation matrix (native XGBoost handling)",
          int(np.isnan(X_val).sum()) == meta.get("nan_count_validation"))

    # ---- data/raw untouched ----
    raw_dir = ROOT / "data" / "raw"
    check("data/raw/ still present and readable", raw_dir.exists() and any(raw_dir.iterdir()))

    n_pass = sum(1 for _, ok in CHECKS if ok)
    n_total = len(CHECKS)
    print(f"\n{n_pass}/{n_total} checks passed")
    if n_pass != n_total:
        sys.exit(1)


if __name__ == "__main__":
    main()
