"""
Phase 4A validation (artifact-level, like validate_phase2/3.py). Read-only.
Exits non-zero if any check fails.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from _common import ROOT, INTERIM, REPORTS
from models.logistic_regression_scratch import predict_proba
from preprocessing_logistic_v1 import build_augmented_training_raw, fit_preprocessing

CONFIG_PATH = ROOT / "config" / "series_features_v1.yaml"
FEATURES_PATH = ROOT / "data" / "features" / "series_features_v1.parquet"
SPLIT_PATH = ROOT / "data" / "modeling" / "series_split_v1.csv"
PREPROC_PATH = ROOT / "data" / "modeling" / "logistic_preprocessing_v1.json"
MODEL_JSON_PATH = ROOT / "models" / "logistic_regression_scratch_v1.json"
MODEL_NPZ_PATH = ROOT / "models" / "logistic_regression_scratch_v1.npz"
MODEL_SOURCE_PATH = ROOT / "scripts" / "models" / "logistic_regression_scratch.py"

CHECKS = []


def check(name, condition):
    CHECKS.append((name, bool(condition)))
    print(f"[{'PASS' if condition else 'FAIL'}] {name}")


def strip_docstrings(source):
    in_doc, out = False, []
    for line in source.splitlines():
        if line.strip().startswith('"""'):
            in_doc = not in_doc
            continue
        if not in_doc:
            out.append(line)
    return "\n".join(out)


def main():
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    model_features = cfg["model_features"]

    # ---- required artifacts exist ----
    for p in [CONFIG_PATH, FEATURES_PATH, SPLIT_PATH, PREPROC_PATH, MODEL_JSON_PATH, MODEL_NPZ_PATH]:
        check(f"artifact exists: {p.relative_to(ROOT)}", p.exists())

    source = MODEL_SOURCE_PATH.read_text(encoding="utf-8")
    code = strip_docstrings(source)
    check("scratch model source has no sklearn import/usage",
          "import sklearn" not in code and "from sklearn" not in code and "sklearn." not in code)

    preproc = json.loads(PREPROC_PATH.read_text(encoding="utf-8"))
    check("preprocessing artifact's original feature list matches config whitelist exactly",
          preproc["original_model_feature_names"] == model_features)

    split = pd.read_csv(SPLIT_PATH, parse_dates=["datetime"])
    train_split = split[split["split"] == "train"]
    val_split = split[split["split"] == "validation"]
    test_split = split[split["split"] == "test"]

    # ---- chronology ----
    check("train strictly before validation", train_split["datetime"].max() < val_split["datetime"].min())
    check("validation strictly before test", val_split["datetime"].max() < test_split["datetime"].min())

    per_dt = split.groupby("datetime")["split"].nunique()
    check("no datetime group crosses a split boundary", (per_dt == 1).all())

    # ---- no Cologne / post-Cologne in the split (structurally guaranteed by Phase 3, checked anyway) ----
    em = pd.read_csv(INTERIM / "evaluation_manifest.csv")
    cologne_ids = set(em.loc[em["evaluation_group"] == "cologne_2026", "match_id"])
    post_cologne_ids = set(em.loc[em["evaluation_group"] == "post_cologne", "match_id"])
    check("no cologne_2026 match_id in the split file", set(split["match_id"]).isdisjoint(cologne_ids))
    check("no post_cologne match_id in the split file", set(split["match_id"]).isdisjoint(post_cologne_ids))

    # ---- preprocessing stats independently recomputed from train split (real reproducibility check) ----
    features = pd.read_parquet(FEATURES_PATH, engine="fastparquet")
    df = features.merge(split[["match_id", "split"]], on="match_id", how="inner")
    train_raw = df[df["split"] == "train"].reset_index(drop=True)
    val_raw = df[df["split"] == "validation"].reset_index(drop=True)

    augmented_train_raw = build_augmented_training_raw(train_raw)
    recomputed = fit_preprocessing(augmented_train_raw, model_features)
    means_match = all(abs(recomputed["train_means"][k] - preproc["train_means"][k]) < 1e-9 for k in preproc["train_means"])
    stds_match = all(abs(recomputed["train_stds"][k] - preproc["train_stds"][k]) < 1e-9 for k in preproc["train_stds"])
    medians_match = all(abs(recomputed["train_medians"][k] - preproc["train_medians"][k]) < 1e-9 for k in preproc["train_medians"])
    check("preprocessing means recomputed from train split match the saved artifact exactly", means_match)
    check("preprocessing stds recomputed from train split match the saved artifact exactly", stds_match)
    check("preprocessing medians recomputed from train split match the saved artifact exactly", medians_match)

    # ---- mirrored augmentation: train-only, exact 50/50 ----
    model_meta = json.loads(MODEL_JSON_PATH.read_text(encoding="utf-8"))
    check("training_rows_before_mirroring matches the split file's train row count",
          model_meta["training_rows_before_mirroring"] == len(train_split))
    check("training_rows_after_mirroring == 2x training_rows_before_mirroring",
          model_meta["training_rows_after_mirroring"] == 2 * model_meta["training_rows_before_mirroring"])
    check("mirrored training target mean is exactly 0.5",
          abs(model_meta["mirrored_train_target_mean"] - 0.5) < 1e-9)

    # ---- validation/test never mirrored ----
    check("validation metrics row count matches the (unmirrored) validation split size",
          model_meta["validation_metrics"]["n"] == len(val_split))
    check("train metrics row count matches the (unmirrored) train split size (not the augmented 2x set)",
          model_meta["train_metrics"]["n"] == len(train_split))

    # ---- test/Cologne must never have been scored ----
    check("no test_metrics key present in model metadata", "test_metrics" not in model_meta)
    check("no cologne_metrics key present in model metadata", "cologne_metrics" not in model_meta)
    check("model metadata declares test_status == SEALED", model_meta.get("test_status", "").startswith("SEALED"))
    check("model metadata declares cologne_status == UNTOUCHED", model_meta.get("cologne_status", "").startswith("UNTOUCHED"))

    forbidden_patterns = ["*test_metric*", "*internal_test*", "*cologne_metric*", "*cologne_evaluation*"]
    stray_files = []
    for base in [REPORTS, ROOT / "data" / "modeling", ROOT / "models"]:
        if base.exists():
            for pat in forbidden_patterns:
                stray_files.extend(base.rglob(pat))
    check("no stray test-metrics/cologne-metrics artifact files exist", len(stray_files) == 0)

    # ---- weights/bias/cost finite; probabilities in [0,1] ----
    npz = np.load(MODEL_NPZ_PATH, allow_pickle=True)
    w, b, feature_names, J_history = npz["w"], npz["b"], npz["feature_names"], npz["J_history"]
    check("learned weights are finite", np.isfinite(w).all())
    check("learned bias is finite", np.isfinite(b).all())
    check("cost history is finite throughout", np.isfinite(J_history).all())

    from preprocessing_logistic_v1 import transform
    X_val, _ = transform(val_raw, preproc)
    proba_val = predict_proba(X_val, w, float(b[0]))
    check("predicted probabilities on validation are all within [0,1]",
          (proba_val >= 0).all() and (proba_val <= 1).all())

    n_pass = sum(1 for _, ok in CHECKS if ok)
    n_total = len(CHECKS)
    print(f"\n{n_pass}/{n_total} checks passed")
    if n_pass != n_total:
        sys.exit(1)


if __name__ == "__main__":
    main()
