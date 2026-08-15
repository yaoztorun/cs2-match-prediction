"""
Phase 9C validation. Read-only. Exits non-zero on failure. Never imports a
test module (amendment #7) - Phase 9B regression is re-verified by running
the ACTUAL pytest suite and validate_phase9b.py as real subprocess commands,
exactly as a human would. Passing this script means: the feature-group
mapping has 100% exact-set-equality coverage for both models, RF/XGB
attribution reconstructs their respective model outputs within frozen
tolerances, the XGB tree-range/feature-order contract holds, DP reach/
leverage invariants hold, outputs are JSON-safe and deterministic, no
fitting/training call exists anywhere in the explanation code, and the
sealed Phase 7 TEST partition is never touched.
"""

import ast
import json
import os
import subprocess
import sys

import pandas as pd
import yaml

from _common import ROOT
import application_inference as ai
import application_explanations as ae

CHECKS = []


def check(name, cond):
    CHECKS.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")


def module_reads_forbidden(path, forbidden_substrings):
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(path))
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if any(f in alias.name for f in forbidden_substrings):
                    hits.append(alias.name)
        if isinstance(node, ast.ImportFrom) and node.module:
            if any(f in node.module for f in forbidden_substrings):
                hits.append(node.module)
    return hits


def main():
    print("\n=== 1. explanation registry ===")
    reg = ae.get_explanation_metadata()
    check("causal = false", reg["causal"] is False)
    check("explanation_type = model_feature_attribution", reg["explanation_type"] == "model_feature_attribution")
    check("RF method = saabas_path_decomposition (never called SHAP)",
          reg["models"]["series_random_forest_v2"]["attribution_method"] == "saabas_path_decomposition")
    check("XGB method = xgboost_native_treeshap",
          reg["models"]["map_xgboost_v3_final"]["attribution_method"] == "xgboost_native_treeshap")

    print("\n=== 2. feature-group coverage (exact set equality, amendment #10) ===")
    fg = yaml.safe_load((ROOT / "config" / "application_explanation_feature_groups_v1.yaml")
                         .read_text(encoding="utf-8"))
    ctx = ai.get_context("deployment_post_cologne_v1")
    rf_expected = set(ctx.rf_context.preprocessing["transformed_feature_names"])
    rf_mapped = {f["transformed_feature"] for f in fg["rf_v2"]["features"]}
    check("RF: exact set equality, 19 features", rf_mapped == rf_expected and len(rf_mapped) == 19)
    xgb_expected = set(ctx.xgb_preprocessing["transformed_feature_names"])
    xgb_mapped = {f["transformed_feature"] for f in fg["map_xgboost_v3_final"]["features"]}
    check("XGB: exact set equality, 131 features", xgb_mapped == xgb_expected and len(xgb_mapped) == 131)
    for model_key in ("rf_v2", "map_xgboost_v3_final"):
        names = [f["transformed_feature"] for f in fg[model_key]["features"]]
        check(f"{model_key}: no duplicate feature mapping", len(names) == len(set(names)))

    print("\n=== 3. RF attribution reconstruction (broad fixture set) ===")
    import feature_engine as fe
    import preprocessing_random_forest_v1 as prep_rf
    import random
    store = ctx.rf_context.store
    prep = ctx.rf_context.preprocessing
    teams = list(store.teams.keys())
    random.seed(42)
    max_rf_err = 0.0
    for _ in range(200):
        t1, t2 = random.sample(teams, 2)
        bo = random.choice([1, 3, 5])
        tier = random.choice(["tier1", "tier2", "tier3"])
        raw = fe.build_features(store, t1, t2, ctx.state_cutoff, bo, tier=tier)
        df = pd.DataFrame([{k: raw[k] for k in prep["original_model_feature_names"]}])
        X, _ = prep_rf.transform(df, prep)
        base, contrib = ae._rf_saabas_contributions(ctx.rf_context.model, X[0])
        recon = base + contrib.sum()
        actual = ctx.rf_context.model.predict_proba(X)[0, 1]
        max_rf_err = max(max_rf_err, abs(recon - actual))
    check(f"RF additivity holds within 1e-9 across 200 fixtures (max_err={max_rf_err:.2e})", max_rf_err < 1e-9)

    print("\n=== 4. XGB attribution reconstruction + tree-range/feature-order contract ===")
    booster = ctx.xgb_model.get_booster()
    check("XGB has no best_iteration/early-stopping metadata (uses all trees, same as predict_proba)",
          booster.attributes() == {} and getattr(ctx.xgb_model, "best_iteration", None) is None)
    max_xgb_err = 0.0
    for map_name in [m["canonical_name"] for m in ctx.map_registry["model_supported_maps"]]:
        r = ae.explain_map("deployment_post_cologne_v1", "Team Vitality", "Team Falcons", map_name, 3)
        chk = r["explanation"]["reconstruction_check"]
        max_xgb_err = max(max_xgb_err, abs(chk["sigmoid_reconstructed"] - r["prediction"]["probability_team_a"]))
    check(f"XGB additivity holds within {ae.XGB_RECONSTRUCTION_TOLERANCE} across 9 maps "
          f"(max_err={max_xgb_err:.2e})", max_xgb_err < ae.XGB_RECONSTRUCTION_TOLERANCE)

    print("\n=== 5. DP reach / leverage invariants ===")
    p1, p2, p3 = 0.6, 0.4, 0.5
    reach = ae._reach_probabilities([p1, p2, p3], 3)
    check("BO3 analytic reach probability contract",
          reach[0] == 1.0 and reach[1] == 1.0 and
          abs(reach[2] - (p1 * (1 - p2) + (1 - p1) * p2)) < 1e-12)
    probs5 = [0.6, 0.5, 0.4, 0.55, 0.5]
    reach5 = ae._reach_probabilities(probs5, 5)
    check("BO5 reach probabilities monotonic non-increasing and in [0,1]",
          all(0.0 <= r <= 1.0 for r in reach5) and all(reach5[i + 1] <= reach5[i] + 1e-12 for i in range(4)))
    leverage5 = ae._series_composition_leverage(probs5, 5)
    check("leverage[last_map] == reach[last_map] (mathematical identity)",
          abs(leverage5[-1] - reach5[-1]) < 1e-9)
    lev_a = ae._series_composition_leverage([0.1, 0.4, 0.6], 3)[0]
    lev_b = ae._series_composition_leverage([0.9, 0.4, 0.6], 3)[0]
    check("leverage independent of the original p_i value", abs(lev_a - lev_b) < 1e-12)

    print("\n=== 6. JSON safety / determinism ===")
    r1 = ae.explain_series_unknown_maps("deployment_post_cologne_v1", "Team Vitality", "Team Falcons", 3)
    r2 = ae.explain_series_unknown_maps("deployment_post_cologne_v1", "Team Vitality", "Team Falcons", 3)
    try:
        json.dumps(r1)
        json_ok = True
    except TypeError:
        json_ok = False
    check("explanation output is JSON-serializable", json_ok)
    check("repeated identical explanation call is deterministic",
          r1["explanation"]["base_value"] == r2["explanation"]["base_value"] and
          [g["rank"] for g in r1["explanation"]["grouped_factors"]] ==
          [g["rank"] for g in r2["explanation"]["grouped_factors"]])

    print("\n=== 7. no model-fitting imports / no research TEST reopening ===")
    ae_path = ROOT / "scripts" / "application_explanations.py"
    hits = module_reads_forbidden(ae_path, ["sklearn.model_selection", "hyperopt", "optuna", "shap", "eli5"])
    check("application_explanations.py imports no tuning/fitting/unavailable-library module (AST)",
          len(hits) == 0)
    src = ae_path.read_text(encoding="utf-8")
    check("application_explanations.py never calls .fit(", ".fit(" not in src)
    for forbidden_path in ("phase7_test_protocol_v1.json", "map_test_predictions_v1.parquet",
                            "map_split_v1.csv", "series_split_v1.csv"):
        check(f"application_explanations.py never references {forbidden_path}", forbidden_path not in src)

    print("\n=== 8. Phase 9B regression gate - REAL commands, not test-module imports (amendment #7) ===")
    env = {"PYTHONIOENCODING": "utf-8", **os.environ}
    result = subprocess.run([sys.executable, "-m", "pytest", "tests/test_phase9b_application_inference.py", "-q"],
                             cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", env=env)
    check("full Phase 9B pytest suite passes (real subprocess run)", result.returncode == 0)
    if result.returncode != 0:
        print(result.stdout[-2000:])
    result2 = subprocess.run([sys.executable, "validate_phase9b.py"], cwd=str(ROOT / "scripts"),
                              capture_output=True, text=True, encoding="utf-8", env=env)
    check("scripts/validate_phase9b.py passes (real subprocess run)", result2.returncode == 0)
    if result2.returncode != 0:
        print(result2.stdout[-2000:])

    print("\n=== 9. fixture manifest / receipt ===")
    fixture_manifest_path = ROOT / "config" / "application_explanation_fixtures_v1.yaml"
    check("fixture manifest exists", fixture_manifest_path.exists())
    fixtures_dir = ROOT / "data" / "deployment" / "phase9c_fixtures"
    check("fixture outputs exist (3 files)", len(list(fixtures_dir.glob("*.json"))) == 3)

    print("\n=== 10. report exists with required markers ===")
    report_path = ROOT / "reports" / "phase9c_application_explanation_core.md"
    check("phase9c report exists", report_path.exists())
    if report_path.exists():
        text = report_path.read_text(encoding="utf-8")
        for marker in ["APPLICATION EXPLANATION CORE = IMPLEMENTED", "EXPLANATIONS = NON-CAUSAL",
                       "PHASE 9B PREDICTIONS = UNCHANGED", "RF V2 = UNCHANGED", "XGB V3 = UNCHANGED"]:
            check(f"report contains marker: {marker!r}", marker in text)

    _finish()


def _finish():
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
