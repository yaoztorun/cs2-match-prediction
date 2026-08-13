"""
Phase 7 validation. Read-only. Exits non-zero on failure.

CRITICAL: this validator NEVER rescopes or rescoring TEST. It inspects the
saved prediction artifact, receipt, protocol, reports, and code structure
only - the canonical TEST predictions are immutable, and this script proves
that fact rather than reproducing it.
"""

import ast
import hashlib
import json
import sys

import numpy as np
import pandas as pd

from _common import ROOT

EVAL_DIR = ROOT / "data" / "evaluation"
PROTOCOL_PATH = EVAL_DIR / "phase7_test_protocol_v1.json"
PRED_PATH = EVAL_DIR / "map_test_predictions_v1.parquet"
RECEIPT_PATH = EVAL_DIR / "phase7_test_open_receipt_v1.json"

SCRIPTS_DIR = ROOT / "scripts"
TABLES_DIR = ROOT / "reports" / "tables"
FIGURES_DIR = ROOT / "reports" / "figures"

FROZEN_PHASE1_6D_ARTIFACTS = [
    "models/map_xgboost_v3_final.json", "models/map_xgboost_v3_final_metadata.json",
    "data/modeling/map_xgboost_v3_final_preprocessing.json", "data/modeling/map_xgboost_v3_final_config.json",
    "config/map_features_v3_modern_map.yaml", "data/features/map_features_v3_modern_map.parquet",
    "data/modeling/map_split_v1.csv", "data/modeling/map_cv_folds_v1.csv",
    "reports/tables/map_xgboost_v3_final_oof_metrics.csv",
    "reports/tables/map_model_validation_metrics_v1.csv",
    "reports/tables/map_xgboost_v3_final_feature_importance.csv",
    "reports/tables/map_xgboost_v3_final_group_importance.csv",
    "reports/phase6d_final_xgboost_v3_results.md",
]
PHASE7_SOLE_TEST_READER = "evaluate_phase7_test_once.py"
PHASE7_DOWNSTREAM_SCRIPTS = ["phase7_test_reports.py", "phase7_test_bootstrap.py",
                              "phase7_test_visualizations.py"]

EXPECTED_TEST_ROWS = 1427
EXPECTED_N_BOOTSTRAP = 2000
EXPECTED_RANDOM_STATE = 42

CHECKS = []


def check(name, cond):
    CHECKS.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")


def sha256(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def reads_any(source, needles):
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            f = node.func
            fname = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else "")
            if fname in {"read_csv", "read_parquet"}:
                for arg in list(node.args) + [kw.value for kw in node.keywords]:
                    if any(n in ast.unparse(arg) for n in needles):
                        return True
    return False


def has_fit_call(source):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            attr = f.attr if isinstance(f, ast.Attribute) else ""
            if attr in {"fit", "fit_transform", "fit_preprocessing"}:
                return True
    return False


def main():
    print("=== capturing hashes of Phase 1-6D frozen artifacts ===")
    baseline = {}
    for rel in FROZEN_PHASE1_6D_ARTIFACTS:
        p = ROOT / rel
        baseline[rel] = sha256(p) if p.exists() else None
        check(f"Phase 1-6D artifact present: {rel}", baseline[rel] is not None)

    print("\n=== 1. repo structure ===")
    src_dir = ROOT / "src"
    check("src/ remains empty", not any(p.is_file() for p in src_dir.rglob("*")) if src_dir.exists() else True)
    check("data/raw/ present and non-empty", any((ROOT / "data" / "raw").rglob("*")))
    check("reference/ present and non-empty", any((ROOT / "reference").rglob("*")))

    print("\n=== 2. protocol frozen, no TEST outcome inside it ===")
    check("protocol artifact exists", PROTOCOL_PATH.exists())
    if not PROTOCOL_PATH.exists():
        n_pass = sum(1 for _, ok in CHECKS if ok)
        print(f"\n{n_pass}/{len(CHECKS)} checks passed (stopped early - protocol missing)")
        sys.exit(1)
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    check("protocol records main model id == map_xgboost_v3_final",
          protocol["model_under_evaluation"] == "map_xgboost_v3_final")
    check("protocol records threshold == 0.5", protocol["threshold"] == 0.5)
    check("protocol records expected TEST row count == 1427", protocol["expected_test_row_count"] == 1427)
    check("protocol records bootstrap contract (2000, seed 42, 95% CI, match_id)",
          protocol["bootstrap"] == {"n_bootstrap": 2000, "random_state": 42, "ci": 0.95,
                                     "cluster_key": "match_id"})
    check("protocol records the fixed 10-bin calibration contract",
          protocol["calibration_bins"]["n_bins"] == 10
          and protocol["calibration_bins"]["edges"] == [round(i / 10, 1) for i in range(11)]
          and protocol["calibration_bins"]["last_bin_closed_at_one"] is True)
    check("protocol contains no TEST outcome fields",
          "no_test_outcome_present" in protocol and protocol["no_test_outcome_present"] is True)
    text = json.dumps(protocol)
    check("protocol JSON contains no obvious result field",
          not any(k in text for k in ["p_xgb_v3_final", "y_pred_xgb_v3_final", "\"accuracy\":", "roc_auc_test"]))

    print("\n=== 3. frozen model/config/preprocessing hashes match the protocol ===")
    hash_targets = {
        "final_model": ROOT / "models" / "map_xgboost_v3_final.json",
        "final_model_metadata": ROOT / "models" / "map_xgboost_v3_final_metadata.json",
        "final_preprocessing": ROOT / "data" / "modeling" / "map_xgboost_v3_final_preprocessing.json",
        "final_xgb_config": ROOT / "data" / "modeling" / "map_xgboost_v3_final_config.json",
        "v3_feature_config": ROOT / "config" / "map_features_v3_modern_map.yaml",
        "v3_feature_parquet": ROOT / "data" / "features" / "map_features_v3_modern_map.parquet",
        "test_split_manifest": ROOT / "data" / "modeling" / "map_split_v1.csv",
    }
    for label, p in hash_targets.items():
        check(f"protocol hash matches current file: {label}",
              protocol["artifact_hashes"].get(label) == sha256(p))

    print("\n=== 4. sole-TEST-reader static guard ===")
    reader_path = SCRIPTS_DIR / PHASE7_SOLE_TEST_READER
    reader_src = reader_path.read_text(encoding="utf-8")
    check(f"{PHASE7_SOLE_TEST_READER} contains the split=='test' filter",
          'split"] == "test"' in reader_src.replace("'", '"'))
    check(f"{PHASE7_SOLE_TEST_READER} contains no model-fitting call", not has_fit_call(reader_src))
    for name in PHASE7_DOWNSTREAM_SCRIPTS:
        src = (SCRIPTS_DIR / name).read_text(encoding="utf-8")
        check(f"{name} does NOT filter split=='test'", 'split"] == "test"' not in src.replace("'", '"'))
        check(f"{name} never reopens map_features_v3_modern_map.parquet or map_split_v1.csv",
              not reads_any(src, ["map_features_v3_modern_map", "map_split_v1"]))
        check(f"{name} contains no model-fitting call", not has_fit_call(src))

    print("\n=== 5. canonical prediction artifact ===")
    check("canonical prediction artifact exists", PRED_PATH.exists())
    if PRED_PATH.exists():
        pred = pd.read_parquet(PRED_PATH, engine="fastparquet")
        check(f"row count == {EXPECTED_TEST_ROWS}", len(pred) == EXPECTED_TEST_ROWS)
        check("unique (match_id, game_id) count == row count",
              pred.drop_duplicates(subset=["match_id", "game_id"]).shape[0] == len(pred))
        check("no duplicate rows", pred.duplicated().sum() == 0)
        em = pd.read_csv(ROOT / "data" / "interim" / "evaluation_manifest.csv")
        cologne_ids = set(em.loc[em["evaluation_group"].isin(["cologne_2026", "post_cologne"]), "match_id"])
        check("no Cologne/post-Cologne match_id in the prediction artifact",
              set(pred["match_id"]).isdisjoint(cologne_ids))
        check("target is binary", set(pd.unique(pred["y_true"])) <= {0, 1})
        prob_cols = ["p_xgb_v3_final", "p_constant_05", "p_overall_elo", "p_map_elo", "p_xgb_v3_mirrored"]
        for c in prob_cols:
            vals = pred[c].to_numpy(dtype=float)
            check(f"{c} finite and in [0,1]", bool(np.isfinite(vals).all() and (vals >= 0).all()
                                                    and (vals <= 1).all()))
        check("threshold-derived label matches threshold 0.5 exactly",
              (pred["y_pred_xgb_v3_final"].to_numpy() == (pred["p_xgb_v3_final"].to_numpy() >= 0.5).astype(int)).all())

        print("\n=== 6. final model / preprocessing provenance (never by rescoring TEST) ===")
        # The prediction artifact deliberately does not carry raw model input features (only
        # identity/coverage columns + probabilities), so provenance is checked via metadata flags
        # and the AST fit-call check above - NOT by reloading the model against TEST rows, which
        # would mean reopening a raw TEST source and effectively rescoring TEST.
        import preprocessing_xgboost_map_v3 as prep_xgb
        params = prep_xgb.load_preprocessing(hash_targets["final_preprocessing"])
        meta = json.loads(hash_targets["final_model_metadata"].read_text(encoding="utf-8"))
        check("model metadata records threshold == 0.5", meta["threshold"] == 0.5)
        check("model metadata records calibration_applied == False", meta["calibration_applied"] is False)
        check("preprocessing artifact has exactly 131 transformed feature names",
              len(params["transformed_feature_names"]) == 131)

    print("\n=== 7. TEST-open receipt ===")
    check("receipt exists", RECEIPT_PATH.exists())
    if RECEIPT_PATH.exists() and PRED_PATH.exists():
        receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
        check("receipt records protocol_hash matching the protocol file",
              receipt["protocol_hash"] == protocol["protocol_hash"])
        check("receipt records the correct prediction-artifact hash",
              receipt["prediction_artifact_hash"] == sha256(PRED_PATH))
        check("receipt records TEST row count == 1427", receipt["test_row_count"] == 1427)
        check("receipt has no wall-clock timestamp field",
              not any("time" in k.lower() or "date" == k.lower() for k in receipt))

    print("\n=== 8. reports/tables/figures exist ===")
    expected_tables = [
        "map_test_metrics_v1.csv", "map_test_baseline_comparison_v1.csv", "map_test_bootstrap_ci_v1.csv",
        "map_test_paired_bootstrap_deltas_v1.csv", "map_test_series_macro_v1.csv", "map_test_per_map_v1.csv",
        "map_test_bestof_v1.csv", "map_test_tier_v1.csv", "map_test_coverage_v1.csv",
        "map_test_calibration_bins_v1.csv", "map_development_vs_test_v1.csv",
    ]
    for t in expected_tables:
        check(f"table exists: {t}", (TABLES_DIR / t).exists())
    for r in ["phase7_internal_test_results.md", "phase7_internal_test_uncertainty.md"]:
        check(f"report exists: {r}", (ROOT / "reports" / r).exists())
    core_figures = ["map_xgb_v3_test_roc.png", "map_xgb_v3_test_calibration.png",
                     "map_xgb_v3_test_probability_distribution.png",
                     "map_xgb_v3_test_confusion_matrix.png", "map_test_poster_summary.png"]
    for f in core_figures:
        check(f"figure exists: {f}", (FIGURES_DIR / f).exists())

    print("\n=== 9. bootstrap procedure matches the predefined contract ===")
    ci_path = TABLES_DIR / "map_test_bootstrap_ci_v1.csv"
    if ci_path.exists():
        ci_df = pd.read_csv(ci_path)
        check("bootstrap CI table uses n_bootstrap == 2000 for every metric",
              (ci_df["n_bootstrap"] == EXPECTED_N_BOOTSTRAP).all())
        check("bootstrap CI table has all four metrics", set(ci_df["metric"]) == {"accuracy", "roc_auc",
                                                                                     "log_loss", "brier"})
    boot_src = (SCRIPTS_DIR / "phase7_test_bootstrap.py").read_text(encoding="utf-8")
    check("bootstrap script hardcodes RANDOM_STATE = 42", "RANDOM_STATE = 42" in boot_src)
    check("bootstrap script hardcodes N_BOOTSTRAP = 2000", "N_BOOTSTRAP = 2000" in boot_src)

    print("\n=== 10. no calibration / new ensemble / threshold search artifact ===")
    check("no calibration artifact anywhere under data/evaluation or models",
          not list(EVAL_DIR.glob("*calibrat*")) and not list((ROOT / "models").glob("*calibrat*")))
    check("no threshold-search artifact", not list(EVAL_DIR.glob("*threshold*")))
    check("no new Phase 7 ensemble artifact", not list(EVAL_DIR.glob("*ensemble*")))
    check("no TEST-trained model artifact (only map_xgboost_v3_final.json exists as the model)",
          not list((ROOT / "models").glob("*test*")))

    print("\n=== 11. Cologne untouched (structural) ===")
    if PRED_PATH.exists():
        check("prediction artifact contains zero Cologne rows (re-verified)",
              set(pred["match_id"]).isdisjoint(cologne_ids))

    print("\n=== 12. Phase 1-6D artifacts unchanged ===")
    for rel, expected in baseline.items():
        p = ROOT / rel
        check(f"unchanged: {rel}", p.exists() and sha256(p) == expected)

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
