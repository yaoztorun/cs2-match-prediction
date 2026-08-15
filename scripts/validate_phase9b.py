"""
Phase 9B validation. Read-only. Exits non-zero on failure. Passing this
script means: the context registry hashes the full executable pipeline
(not just a model file), every hash matches the file on disk, the
historical RF wrapper reproduces all 2,976 frozen Phase 8D matrix rows
exactly, the known-map wrapper matches the direct frozen chain, the
composer/DP/validation contracts hold, inference is deterministic and
state-immutable, and no Phase 9B code fits/trains a model or reopens the
sealed Phase 7 TEST partition.
"""

import ast
import json
import sys

import pandas as pd
import yaml

from _common import ROOT
import application_inference as ai
import build_application_registries as bar

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
    print("\n=== 1. context registry ===")
    registry = yaml.safe_load((ROOT / "config" / "application_inference_contexts_v1.yaml").read_bytes())
    check("exactly 2 contexts registered",
          set(registry["contexts"]) == {"historical_cologne_pre_event", "deployment_post_cologne_v1"})
    check("historical context locked to the frozen Phase 8D cutoff",
          registry["contexts"]["historical_cologne_pre_event"]["state_cutoff"] == "2026-06-02T13:30:00")
    check("deployment context cutoff matches the Phase 9A receipt",
          registry["contexts"]["deployment_post_cologne_v1"]["state_cutoff"] == "2026-06-28T20:00:00")

    print("\n=== 2. registry hashes the full pipeline, not just a model file ===")
    fresh_rf = bar.hash_group(bar.RF_PIPELINE)
    fresh_xgb = bar.hash_group(bar.XGB_PIPELINE)
    check(f"RF pipeline hashes match files on disk ({len(bar.RF_PIPELINE)} items)",
          fresh_rf == registry["rf_unknown_map_pipeline"] and len(fresh_rf) == len(bar.RF_PIPELINE))
    check(f"XGB pipeline hashes match files on disk ({len(bar.XGB_PIPELINE)} items)",
          fresh_xgb == registry["xgb_known_map_pipeline"] and len(fresh_xgb) == len(bar.XGB_PIPELINE))
    for ctx_id in ("historical_cologne_pre_event", "deployment_post_cologne_v1"):
        expected = bar.HISTORICAL_STATE if ctx_id.startswith("historical") else bar.DEPLOYMENT_STATE
        fresh_state = bar.hash_group(expected)
        check(f"{ctx_id}: 5 state hashes match files on disk",
              fresh_state == registry["contexts"][ctx_id]["state_hashes"] and len(fresh_state) == 5)
    check("deployment context references the Phase 9A deployment receipt hash",
          bar.sha256_file(bar.DEPLOYMENT_RECEIPT) ==
          registry["contexts"]["deployment_post_cologne_v1"]["phase9a_deployment_receipt_hash"])

    print("\n=== 3. historical RF parity - ALL 2,976 rows (hard gate) ===")
    df = pd.read_parquet(ROOT / "data" / "evaluation" / "cologne_2026_pre_event_matchup_probabilities_v1.parquet",
                          engine="fastparquet")
    check("frozen matrix has 2,976 rows", len(df) == 2976)
    n_exact = 0
    first_mismatch = None
    for row in df.itertuples(index=False):
        r = ai.predict_series_unknown_maps("historical_cologne_pre_event", row.team_a, row.team_b,
                                            int(row.best_of))
        if abs(r["probability_team_a"] - row.probability_team_a) <= 1e-9:
            n_exact += 1
        elif first_mismatch is None:
            first_mismatch = (row.team_a, row.team_b, row.best_of, r["probability_team_a"],
                               row.probability_team_a)
    check(f"2976/2976 exact parity (n_exact={n_exact})", n_exact == 2976)
    if first_mismatch:
        print(f"  first mismatch: {first_mismatch}")

    print("\n=== 4. known-map wrapper parity vs. the direct frozen chain ===")
    import rich_modern_map_feature_composer as rmmc
    import preprocessing_xgboost_map_v3 as prep_xgb
    ctx = ai.get_context("deployment_post_cologne_v1")
    n_ok = 0
    for map_name in [m["canonical_name"] for m in ctx.map_registry["model_supported_maps"]]:
        raw = rmmc.build_future_modern_rich_map_features(
            "Team Vitality", "Team Falcons", 3, map_name, ctx.state_cutoff,
            ctx.rf_context.store, ctx.map_state, ctx.form_state, ctx.roster_state, ctx.modern_map_state,
            tier="tier1")
        dfr = pd.DataFrame([{k: raw[k] for k in ctx.xgb_preprocessing["original_model_feature_names"]}])
        X, _ = prep_xgb.transform(dfr, ctx.xgb_preprocessing, ctx.xgb_roles)
        p_direct = float(ctx.xgb_model.predict_proba(X)[:, 1][0])
        r = ai.predict_map("deployment_post_cologne_v1", "Team Vitality", "Team Falcons", map_name, 3)
        n_ok += int(abs(r["probability_team_a"] - p_direct) <= 1e-12)
    check(f"known-map wrapper matches direct chain for all 9 model-supported maps ({n_ok}/9)", n_ok == 9)

    print("\n=== 5. map support contract ===")
    supported = {m["canonical_name"] for m in ctx.map_registry["model_supported_maps"]}
    check("exactly 9 model-supported maps", len(supported) == 9)
    check("Cache is not a model-supported map", "Cache" not in supported)
    try:
        ai.predict_map("deployment_post_cologne_v1", "Team Vitality", "Team Falcons", "Cache", 3)
        cache_rejected = False
    except ai.ApplicationInferenceError as e:
        cache_rejected = (e.error_code == "unsupported_map")
    check("Cache raises unsupported_map (not silently bucketed)", cache_rejected)

    print("\n=== 6. DP invariants ===")
    from math import comb
    def analytic(p, bo):
        need = (bo + 1) // 2
        return sum(comb(need - 1 + j, j) * (p ** need) * ((1 - p) ** j) for j in range(need))
    dp_ok = all(abs(ai.compose_series_probability([p] * bo, bo) - analytic(p, bo)) < 1e-9
                for bo in (1, 3, 5) for p in (0.3, 0.5, 0.7, 0.9))
    check("DP matches closed-form race-to-N for BO1/3/5 across several p", dp_ok)
    check("DP conserves probability (p_a + p_b == 1) for an asymmetric BO3 case",
          abs(ai.compose_series_probability([0.8, 0.3, 0.6], 3) +
              ai.compose_series_probability([0.2, 0.7, 0.4], 3) - 1.0) < 1e-9)

    print("\n=== 7. determinism / state isolation ===")
    r1 = ai.predict_series_unknown_maps("deployment_post_cologne_v1", "Team Vitality", "Team Falcons", 3)
    r2 = ai.predict_series_unknown_maps("deployment_post_cologne_v1", "Team Vitality", "Team Falcons", 3)
    check("repeated identical RF call is byte-identical", r1["probability_team_a"] == r2["probability_team_a"])
    check("deployment RF model has n_jobs=1", ctx.rf_context.model.n_jobs == 1)
    check("historical and deployment contexts hold distinct StateStore objects",
          ai.get_context("historical_cologne_pre_event").rf_context.store is not ctx.rf_context.store)

    print("\n=== 8. JSON serialization ===")
    try:
        json.dumps(r1)
        json_ok = True
    except TypeError:
        json_ok = False
    check("prediction output is JSON-serializable", json_ok)

    print("\n=== 9. no model fitting / no research TEST reopening ===")
    ai_path = ROOT / "scripts" / "application_inference.py"
    hits_fit = module_reads_forbidden(ai_path, ["sklearn.model_selection", "hyperopt", "optuna"])
    check("application_inference.py imports no tuning/fitting module (AST)", len(hits_fit) == 0)
    src = ai_path.read_text(encoding="utf-8")
    check("application_inference.py never calls .fit(", ".fit(" not in src)
    for forbidden_path in ("phase7_test_protocol_v1.json", "map_test_predictions_v1.parquet", "map_split_v1.csv",
                           "series_split_v1.csv"):
        check(f"application_inference.py never references {forbidden_path}", forbidden_path not in src)

    print("\n=== 10. report exists with required markers ===")
    report_path = ROOT / "reports" / "phase9b_application_inference_core.md"
    check("phase9b report exists", report_path.exists())
    if report_path.exists():
        text = report_path.read_text(encoding="utf-8")
        for marker in ["APPLICATION INFERENCE CORE = IMPLEMENTED", "HISTORICAL REPLAY PARITY = VERIFIED",
                       "RF V2 = UNCHANGED", "XGB V3 = UNCHANGED", "through June 28, 2026"]:
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
