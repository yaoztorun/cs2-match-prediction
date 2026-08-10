"""
Phase 4A.1 validation (artifact-level). Read-only. Exits non-zero on failure.
"""

import ast
import hashlib
import json
import sys

import numpy as np
import pandas as pd
import yaml

from _common import ROOT, INTERIM, REPORTS
from models.logistic_regression_scratch import predict_proba
from preprocessing_logistic_v1 import transform
from logistic_regression_tuning_v2 import (
    select_winner, LAMBDA_GRID, ALPHA, MAX_ITERATIONS, LOG_LOSS_EQUIVALENCE_EPSILON,
)

CONFIG_PATH = ROOT / "config" / "series_features_v1.yaml"
FEATURES_PATH = ROOT / "data" / "features" / "series_features_v1.parquet"
SPLIT_PATH = ROOT / "data" / "modeling" / "series_split_v1.csv"
CV_FOLDS_PATH = ROOT / "data" / "modeling" / "random_forest_cv_folds_v2.csv"
TUNING_CSV = REPORTS / "tables" / "logistic_regression_tuning_v2.csv"
SELECTED_CONFIG = ROOT / "data" / "modeling" / "logistic_regression_v2_selected_config.json"
V2_PREPROC = ROOT / "data" / "modeling" / "logistic_preprocessing_v2.json"
V2_NPZ = ROOT / "models" / "logistic_regression_scratch_v2.npz"
V2_META = ROOT / "models" / "logistic_regression_scratch_v2.json"
TUNING_SOURCE = ROOT / "scripts" / "logistic_regression_tuning_v2.py"
TRAIN_SOURCE = ROOT / "scripts" / "train_logistic_regression_v2.py"
SCRATCH_CORE = ROOT / "scripts" / "models" / "logistic_regression_scratch.py"
SRC_DIR = ROOT / "src"

EXPECTED_TRAIN_N, EXPECTED_VAL_N, EXPECTED_TEST_N = 6619, 1419, 1418
EXPECTED_AUGMENTED_N = 13238
N_FOLDS = 4

# sha256 captured read-only before any Phase 4A.1 work began.
BASELINE_HASHES = {
    "models/logistic_regression_scratch_v1.npz": "504c7f83d9e3162daa0680aeaaa2bf9e7051882e3c5cf17dc05cf9ab494402a3",
    "models/logistic_regression_scratch_v1.json": "584bb916f6260276d09245c8804dca386d32eecbba49691b193f819a6a0c0046",
    "data/modeling/logistic_preprocessing_v1.json": "d8dda783e3f029c31e9d03112c0d676a1947665d6409edcd272030211c09f972",
    "data/modeling/series_split_v1.csv": "fe1b947a3dd9829f1fd9b3e8ac8cc8ae796b8426ef728f609523ae8c48c0c253",
    "data/modeling/random_forest_cv_folds_v2.csv": "152864c64ef558139af8b588d80e94102a13f52786275dc386357b52ac524247",
    "models/random_forest_v1.joblib": "05b4cdd377694ad10a5ee8c163cfbaa3daa542c50802538031cf12ac85d051f7",
    "models/random_forest_v1.json": "8e1e11137fe7972d9a1f55de020d32e2f655b1733736dbfd1d11a99824412ffb",
    "models/random_forest_v2.joblib": "e26e97fd8f1ea7676659605af2d9abd4d4e4cb0c5b767d1df506fb0a9cfac4a9",
    "models/random_forest_v2.json": "c5e527161925758718e5597b8ff730e67cdbb4626c3dee0065968be78499456d",
    "data/modeling/random_forest_v2_selected_config.json": "3666622740fe27a8cb51647133d2707e421010e865ecc56ce04916ed2b422934",
    "models/xgboost_v1.json": "9e9719a62b10b07b422683057a6f59ae5cd6a7ef367883e07f828ac41ec38794",
    "models/xgboost_v1_metadata.json": "42d47c175116eb0c3baa4e737b4b930db04fdd9808377bae3fca6ea1044f6a73",
    "models/xgboost_v2.json": "9479c6d4fcd660967ca01b38316afbdda8ef4470ac850b267cfaac870d258e62",
    "models/xgboost_v2_metadata.json": "01d783835fd3c1b3bb871e15a881c433aba517a354edfb5c706d1ff643414d34",
    "data/modeling/xgboost_v2_selected_config.json": "7cd50900c23551f11d9aeffa7cfcf1bb52dc0c409c5d9629cb875711e9b5ded2",
    "reports/tables/logistic_regression_coefficients_v1.csv": "b8ee4d26190222ac7b650a149cab142f3e697c654fae90dd93b0941bcfa6e0ae",
    "scripts/models/logistic_regression_scratch.py": "cb31baffff3d248f1109729c3f78e66523c588178757ce9382a376f56cd1d44f",
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
                    seg = ast.unparse(arg)
                    if any(n in seg for n in needles):
                        return True
    return False


def main():
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    model_features = cfg["model_features"]

    # ---- every frozen artifact untouched (incl. the scratch core itself) ----
    for rel, expected in BASELINE_HASHES.items():
        p = ROOT / rel
        check(f"unchanged: {rel}", (sha256(p) if p.exists() else None) == expected)

    src_files = [p for p in SRC_DIR.rglob("*") if p.is_file()] if SRC_DIR.exists() else []
    check("src/ remains empty", len(src_files) == 0)
    raw_dir, ref_dir = ROOT / "data" / "raw", ROOT / "reference"
    check("data/raw/ present and readable", raw_dir.exists() and any(raw_dir.iterdir()))
    check("reference/ present and readable", ref_dir.exists() and any(ref_dir.iterdir()))

    # ---- from-scratch guarantee ----
    core = SCRATCH_CORE.read_text(encoding="utf-8")
    code, in_doc = [], False
    for line in core.splitlines():
        if line.strip().startswith('"""'):
            in_doc = not in_doc
            continue
        if not in_doc and not line.strip().startswith("#"):
            code.append(line)
    core_code = "\n".join(code)
    check("scratch core has no sklearn import", "import sklearn" not in core_code and "from sklearn" not in core_code)
    check("scratch core never references sklearn LogisticRegression", "LogisticRegression" not in core_code)
    train_src = TRAIN_SOURCE.read_text(encoding="utf-8")
    check("refit script does not use sklearn.linear_model", "sklearn.linear_model" not in train_src)
    check("refit script uses the frozen scratch primitives",
          "from models.logistic_regression_scratch import" in train_src)

    # ---- split ----
    split = pd.read_csv(SPLIT_PATH)
    counts = split["split"].value_counts()
    check("train == 6619", counts.get("train", 0) == EXPECTED_TRAIN_N)
    check("validation == 1419", counts.get("validation", 0) == EXPECTED_VAL_N)
    check("test == 1418", counts.get("test", 0) == EXPECTED_TEST_N)
    train_ids = set(split.loc[split.split == "train", "match_id"])
    val_ids = set(split.loc[split.split == "validation", "match_id"])
    test_ids = set(split.loc[split.split == "test", "match_id"])

    # ---- CV folds: TRAIN-only, chronology ----
    cv = pd.read_csv(CV_FOLDS_PATH, parse_dates=["datetime"])
    cv_ids = set(cv["match_id"])
    check("tuning used only global TRAIN ids", cv_ids <= train_ids)
    check("no main-validation id in CV folds", cv_ids.isdisjoint(val_ids))
    check("no TEST id in CV folds", cv_ids.isdisjoint(test_ids))
    em = pd.read_csv(INTERIM / "evaluation_manifest.csv")
    check("no cologne_2026 id in CV folds",
          cv_ids.isdisjoint(set(em.loc[em.evaluation_group == "cologne_2026", "match_id"])))
    check("no post_cologne id in CV folds",
          cv_ids.isdisjoint(set(em.loc[em.evaluation_group == "post_cologne", "match_id"])))
    check("fold chronology holds for every fold", all(
        cv.loc[(cv.fold == f) & (cv.role == "train"), "datetime"].max()
        < cv.loc[(cv.fold == f) & (cv.role == "validation"), "datetime"].min()
        for f in range(1, N_FOLDS + 1)))

    # ---- main validation structurally absent from tuning ----
    tuning_src = TUNING_SOURCE.read_text(encoding="utf-8")
    check("tuning script never READS the main split manifest",
          not reads_path(tuning_src, ["SPLIT_PATH", "series_split_v1"]))

    # ---- tuning table ----
    tuning = pd.read_csv(TUNING_CSV)
    fold_rows = tuning[tuning.row_type == "fold"]
    agg_rows = tuning[tuning.row_type == "aggregate"].reset_index(drop=True)
    check(f"tuning CSV has {len(LAMBDA_GRID) * N_FOLDS} fold rows",
          len(fold_rows) == len(LAMBDA_GRID) * N_FOLDS)
    check(f"tuning CSV has {len(LAMBDA_GRID)} aggregate rows", len(agg_rows) == len(LAMBDA_GRID))
    check("tuning CSV lambdas match the predefined grid",
          sorted(agg_rows["lambda"].tolist()) == sorted(LAMBDA_GRID))
    check("lambda=0.0 was evaluated (eligible reference)", 0.0 in set(agg_rows["lambda"]))
    check("tuning CSV records the convergence diagnostics",
          {"converged", "converged_by", "final_gradient_norm", "final_relative_improvement"}
          <= set(fold_rows.columns))
    check("tuning CSV contains no wall-clock timing column (byte-reproducible artifact)",
          "elapsed_seconds" not in tuning.columns)

    # ---- selection independently recomputed ----
    selected = json.loads(SELECTED_CONFIG.read_text(encoding="utf-8"))
    recomputed_lambda, recomputed_stage = select_winner(agg_rows)
    check("recomputed selection matches the frozen lambda",
          float(recomputed_lambda) == float(selected["selected_lambda"]))
    check("recomputed selection stage matches", recomputed_stage == selected["selection_stage"])
    check("selected lambda is in the predefined grid", float(selected["selected_lambda"]) in LAMBDA_GRID)
    check("log-loss epsilon recorded as 0.002",
          selected["log_loss_epsilon"] == LOG_LOSS_EQUIVALENCE_EPSILON)
    check("selected config records main_validation_used_in_selection == False",
          selected.get("main_validation_used_in_selection") is False)
    check("selected lambda converged in all 4 folds", selected.get("all_folds_converged") is True)

    sel_folds = fold_rows[fold_rows["lambda"] == float(selected["selected_lambda"])]
    check("selected lambda has 4 fold rows, all converged",
          len(sel_folds) == N_FOLDS and bool(sel_folds["converged"].all()))

    # ---- optimization settings fixed, not searched ----
    check("frozen alpha matches the fixed optimization setting", selected["alpha"] == ALPHA)
    check("frozen max_iterations matches the fixed optimization setting",
          selected["max_iterations"] == MAX_ITERATIONS)

    # ---- final refit uses the frozen config, no validation-based stopping ----
    meta = json.loads(V2_META.read_text(encoding="utf-8"))
    check("V2 model lambda matches frozen config", meta.get("lambda_") == selected["selected_lambda"])
    check("V2 model alpha matches frozen config", meta.get("alpha") == selected["alpha"])
    check("V2 model max_iterations matches frozen config", meta.get("max_iterations") == selected["max_iterations"])
    check("V2 metadata declares validation_based_stopping_used == False",
          meta.get("validation_based_stopping_used") is False)
    check("refit script passes no eval_set", not uses_kwarg(train_src, "eval_set"))
    check("refit script passes no early_stopping_rounds", not uses_kwarg(train_src, "early_stopping_rounds"))
    check("V2 final refit converged", meta.get("converged") is True)

    # ---- mirroring accounting ----
    check("unique_training_matches == 6619", meta.get("unique_training_matches") == EXPECTED_TRAIN_N)
    check("augmented_training_observations == 13238",
          meta.get("augmented_training_observations") == EXPECTED_AUGMENTED_N)
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

    # ---- preprocessing / features ----
    preproc = json.loads(V2_PREPROC.read_text(encoding="utf-8"))
    check("V2 preprocessing whitelist matches the YAML config",
          preproc["original_model_feature_names"] == model_features)
    check("V2 feature_count == 19", meta.get("feature_count") == 19)
    check("V2 preprocessing has train-only standardization stats",
          {"train_means", "train_stds", "train_medians"} <= set(preproc.keys()))

    # ---- parameters finite, probabilities valid, artifacts reload ----
    npz = np.load(V2_NPZ, allow_pickle=True)
    w, b, names, J = npz["w"], float(npz["b"][0]), npz["feature_names"], npz["J_history"]
    check("learned weights finite", np.isfinite(w).all())
    check("learned bias finite", np.isfinite(b))
    check("cost history finite", np.isfinite(J).all())
    check("saved feature_names has 19 entries", len(names) == 19)

    feats = pd.read_parquet(FEATURES_PATH, engine="fastparquet")
    df = feats.merge(split[["match_id", "split"]], on="match_id", how="inner")
    val_raw = df[df["split"] == "validation"].reset_index(drop=True)
    X_val, tnames = transform(val_raw, preproc)
    check("transform yields 19 columns on real validation data", X_val.shape[1] == 19)
    check("validation row count == 1419", X_val.shape[0] == EXPECTED_VAL_N)

    proba = predict_proba(X_val, w, b)
    check("validation probabilities finite", np.isfinite(proba).all())
    check("validation probabilities in [0,1]", (proba >= 0).all() and (proba <= 1).all())

    # reloaded artifacts reproduce the recorded validation metrics
    from sklearn.metrics import roc_auc_score, log_loss
    y_val = val_raw[cfg["target"]].to_numpy(dtype=float)
    check("reloaded model reproduces recorded validation ROC-AUC (tol=1e-9)",
          abs(float(roc_auc_score(y_val, proba)) - meta["validation_metrics"]["roc_auc"]) < 1e-9)
    check("reloaded model reproduces recorded validation log loss (tol=1e-9)",
          abs(float(log_loss(y_val, proba, labels=[0, 1])) - meta["validation_metrics"]["log_loss"]) < 1e-9)

    n_pass = sum(1 for _, ok in CHECKS if ok)
    print(f"\n{n_pass}/{len(CHECKS)} checks passed")
    if n_pass != len(CHECKS):
        sys.exit(1)


if __name__ == "__main__":
    main()
