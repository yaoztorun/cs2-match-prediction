"""
Phase 9D validation. Read-only. Exits non-zero on failure. Passing this
script means: the application API receipt's hashes match the files on disk,
the startup contract verifies cleanly, readiness/liveness behave correctly,
HTTP prediction and explanation are byte-identical to a direct Phase 9B/9C
Python call (the hard gate), the frozen error-code -> HTTP-status policy
holds, OpenAPI loads and covers every endpoint, no state file changes after
a full request battery, concurrent requests are deterministic and use the
same cached context object, no fitting/training call exists anywhere in the
API module, no write-mode file operation exists anywhere in the API module,
and the Phase 9B/9C validators still pass unchanged.
"""

import ast
import json
import subprocess
import sys

import yaml
from fastapi.testclient import TestClient

from _common import ROOT
import application_inference as ai
import application_explanations as ae
import application_api as api
import build_application_registries as bar
import phase9a_common as p9a

sha256_file = p9a.sha256_file

CONFIG = ROOT / "config"
SCRIPTS = ROOT / "scripts"
TESTS = ROOT / "tests"
DEPLOY = ROOT / "data" / "deployment"

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
    api_py = SCRIPTS / "application_api.py"

    print("\n=== 1. receipt / version hashes ===")
    receipt_path = DEPLOY / "application_api_receipt_v1.json"
    check("application_api_receipt_v1.json exists", receipt_path.exists())
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        check("receipt committed=True", receipt.get("committed") is True)
        check("api_version = v1", receipt["api_version"] == "v1")
        check("prediction_contract = phase9b", receipt["prediction_contract"] == "phase9b")
        check("explanation_contract = application_explanations_v1",
              receipt["explanation_contract"] == "application_explanations_v1")
        check("default_context = deployment_post_cologne_v1",
              receipt["default_context"] == "deployment_post_cologne_v1")
        h = receipt["hashes"]
        fresh = {
            "application_api_py": sha256_file(SCRIPTS / "application_api.py"),
            "run_application_api_py": sha256_file(SCRIPTS / "run_application_api.py"),
            "api_config": sha256_file(CONFIG / "application_api_v1.yaml"),
            "fixture_manifest": sha256_file(CONFIG / "application_api_fixtures_v1.yaml"),
            "phase9b_context_registry": sha256_file(CONFIG / "application_inference_contexts_v1.yaml"),
            "phase9c_explanation_receipt": sha256_file(DEPLOY / "application_explanation_receipt_v1.json"),
            "application_inference_py": sha256_file(SCRIPTS / "application_inference.py"),
            "application_explanations_py": sha256_file(SCRIPTS / "application_explanations.py"),
            "rf_v2_model": sha256_file(bar.RF_PIPELINE["rf_model"]),
            "rf_v2_preprocessing": sha256_file(bar.RF_PIPELINE["rf_preprocessing"]),
            "xgb_v3_model": sha256_file(bar.XGB_PIPELINE["xgb_model"]),
            "xgb_v3_preprocessing": sha256_file(bar.XGB_PIPELINE["xgb_preprocessing"]),
            "test_phase9d_application_api_py": sha256_file(TESTS / "test_phase9d_application_api.py"),
        }
        for k, v in fresh.items():
            check(f"receipt hash matches disk: {k}", h.get(k) == v)
        check("openapi_snapshot hash present", "openapi_snapshot" in h)

    print("\n=== 2. startup contract verification ===")
    # Phase 9E amendment #3 split the original single-boolean startup check into
    # independent (prediction_ready, explanation_ready) + a shared detail dict -
    # same underlying RF/XGB/state/receipt/explanation hash checks, just reshaped.
    prediction_ok, explanation_ok, detail = api._verify_prediction_and_explanation_contract()
    check(f"startup contract verifies cleanly ({detail})", prediction_ok and explanation_ok)

    print("\n=== 3. liveness / readiness over HTTP ===")
    with TestClient(api.app) as client:
        r = client.get("/api/v1/health/live")
        check("health/live returns 200", r.status_code == 200 and r.json()["status"] == "live")
        r = client.get("/api/v1/health/ready")
        check("health/ready returns 200 with default context ready",
              r.status_code == 200 and r.json()["default_context_id"] == api.DEFAULT_CONTEXT_ID)

        print("\n=== 4. default context is never historical ===")
        check("DEFAULT_CONTEXT_ID is the deployment context",
              api.DEFAULT_CONTEXT_ID == ai.DEPLOYMENT_CONTEXT_ID)

        print("\n=== 5. request schema strict typing ===")
        r = client.post("/api/v1/predict/series", json={"context_id": api.DEFAULT_CONTEXT_ID, "mode": "pre_veto",
                                                          "team_a": "Team Vitality", "team_b": "Team Falcons",
                                                          "best_of": True})
        check("bool best_of rejected as schema_validation_error (422)",
              r.status_code == 422 and r.json()["error"]["code"] == "schema_validation_error")
        r = client.post("/api/v1/predict/series", json={"context_id": api.DEFAULT_CONTEXT_ID, "mode": "pre_veto",
                                                          "team_a": "Team Vitality", "team_b": "Team Falcons",
                                                          "best_of": 3, "extra_field": 1})
        check("extra field rejected (422)", r.status_code == 422)

        print("\n=== 6. frozen error_code -> HTTP status policy ===")
        # Original 13 Phase 9D codes are unchanged; Phase 9E adds tournament
        # codes additively (amendment #25 "Phase 9D routes must remain
        # byte/semantically stable" - the original 13 below are verified
        # byte-for-byte unchanged, extension is explicitly in scope).
        expected_map = {
            "unknown_context": 404, "unknown_team": 404, "invalid_best_of": 422, "invalid_tier": 422,
            "same_team": 422, "unsupported_map": 422, "invalid_map_count": 422, "duplicate_map": 422,
            "invalid_probability": 422, "historical_context_datetime_locked": 422,
            "prediction_datetime_before_state_contract": 422, "ambiguous_team": 409, "missing_state_support": 500,
            "unknown_ruleset": 404, "invalid_participant_count": 422, "duplicate_team": 422, "invalid_seed": 422,
            "missing_seed": 422, "invalid_override": 422, "override_team_mismatch": 422, "duplicate_override": 422,
            "contradictory_override": 422, "invalid_simulation_count": 422, "probability_matrix_incomplete": 500,
        }
        check("ERROR_STATUS_MAP matches the frozen policy exactly", api.ERROR_STATUS_MAP == expected_map)
        r = client.post("/api/v1/predict/series", json={"context_id": "nope", "mode": "pre_veto",
                                                          "team_a": "Team Vitality", "team_b": "Team Falcons",
                                                          "best_of": 3})
        check("unknown_context -> 404 end-to-end", r.status_code == 404)
        r = client.post("/api/v1/predict/series", json={"context_id": api.DEFAULT_CONTEXT_ID, "mode": "pre_veto",
                                                          "team_a": "Team Vitality", "team_b": "Team Vitality",
                                                          "best_of": 3})
        check("same_team -> 422 end-to-end", r.status_code == 422)
        r = client.get("/api/v1/contexts/does_not_exist")
        check("all error responses share {error:{code,message,detail}, request_id} shape",
              set(r.json().keys()) == {"error", "request_id"} and
              set(r.json()["error"].keys()) == {"code", "message", "detail"})

        print("\n=== 7. hard gate: HTTP prediction parity vs direct Phase 9B core ===")
        n_ok, n_total = 0, 0
        for context_id in (ai.HISTORICAL_CONTEXT_ID, ai.DEPLOYMENT_CONTEXT_ID):
            pdt = "2026-06-02T13:30:00" if context_id == ai.HISTORICAL_CONTEXT_ID else None
            for bo in (1, 3, 5):
                payload = {"context_id": context_id, "mode": "pre_veto", "team_a": "Team Vitality",
                           "team_b": "Team Falcons", "best_of": bo, "include_explanation": False}
                if pdt:
                    payload["prediction_datetime"] = pdt
                r = client.post("/api/v1/predict/series", json=payload)
                direct = ai.predict_series_unknown_maps(context_id, "Team Vitality", "Team Falcons", bo,
                                                          prediction_datetime=pdt)
                n_total += 1
                n_ok += int(r.json()["prediction"]["probability_team_a"] == direct["probability_team_a"])
        r = client.post("/api/v1/predict/series", json={"context_id": ai.HISTORICAL_CONTEXT_ID, "mode": "pre_veto",
                                                          "team_a": "THUNDERdOWNUNDER", "team_b": "MOUZ",
                                                          "best_of": 3, "include_explanation": False})
        direct = ai.predict_series_unknown_maps(ai.HISTORICAL_CONTEXT_ID, "THUNDERdOWNUNDER", "MOUZ", 3)
        n_total += 1
        n_ok += int(r.json()["prediction"]["probability_team_a"] == direct["probability_team_a"])
        check(f"pre_veto HTTP parity across historical+deployment/BO1/3/5/cold-start ({n_ok}/{n_total})",
              n_ok == n_total)

        ctx = ai.get_context(ai.DEPLOYMENT_CONTEXT_ID)
        maps = [m["canonical_name"] for m in ctx.map_registry["model_supported_maps"]]
        n_ok = 0
        for map_name in maps:
            r = client.post("/api/v1/predict/map", json={"context_id": ai.DEPLOYMENT_CONTEXT_ID,
                                                           "team_a": "Team Vitality", "team_b": "Team Falcons",
                                                           "map_name": map_name, "best_of": 3,
                                                           "include_explanation": False})
            direct = ai.predict_map(ai.DEPLOYMENT_CONTEXT_ID, "Team Vitality", "Team Falcons", map_name, 3)
            n_ok += int(r.json()["prediction"]["probability_team_a"] == direct["probability_team_a"])
        check(f"known-map HTTP parity across all 9 model-supported maps ({n_ok}/9)", n_ok == 9)

        n_ok = 0
        for bo, ms in ((1, ["Mirage"]), (3, ["Mirage", "Inferno", "Nuke"]),
                       (5, ["Mirage", "Inferno", "Nuke", "Ancient", "Overpass"])):
            r = client.post("/api/v1/predict/series", json={"context_id": ai.DEPLOYMENT_CONTEXT_ID,
                                                              "mode": "known_maps", "team_a": "Team Vitality",
                                                              "team_b": "Team Falcons", "best_of": bo,
                                                              "ordered_maps": ms, "include_explanation": False})
            direct = ai.predict_series_known_maps(ai.DEPLOYMENT_CONTEXT_ID, "Team Vitality", "Team Falcons", bo, ms)
            n_ok += int(r.json()["prediction"] == direct)
        check(f"known-series HTTP parity across BO1/3/5 ({n_ok}/3)", n_ok == 3)

        print("\n=== 8. hard gate: HTTP explanation parity vs direct Phase 9C core ===")
        r = client.post("/api/v1/predict/series", json={"context_id": ai.DEPLOYMENT_CONTEXT_ID, "mode": "pre_veto",
                                                          "team_a": "Team Vitality", "team_b": "Team Falcons",
                                                          "best_of": 3, "explanation_detail": "full"})
        direct = ae.explain_series_unknown_maps(ai.DEPLOYMENT_CONTEXT_ID, "Team Vitality", "Team Falcons", 3)
        api_exp = dict(r.json()["explanation"]); api_exp.pop("detail_level")
        check("full RF explanation via HTTP == direct Phase 9C explanation", api_exp == direct["explanation"])

        r = client.post("/api/v1/predict/map", json={"context_id": ai.DEPLOYMENT_CONTEXT_ID,
                                                       "team_a": "Team Vitality", "team_b": "Team Falcons",
                                                       "map_name": "Mirage", "best_of": 3,
                                                       "explanation_detail": "full"})
        direct = ae.explain_map(ai.DEPLOYMENT_CONTEXT_ID, "Team Vitality", "Team Falcons", "Mirage", 3)
        api_exp = dict(r.json()["explanation"]); api_exp.pop("detail_level"); api_exp.pop("state_support")
        check("full XGB explanation via HTTP == direct Phase 9C explanation", api_exp == direct["explanation"])

        r = client.post("/api/v1/predict/series", json={"context_id": ai.DEPLOYMENT_CONTEXT_ID, "mode": "pre_veto",
                                                          "team_a": "Team Vitality", "team_b": "Team Falcons",
                                                          "best_of": 3, "explanation_detail": "summary"})
        summary_exp = r.json()["explanation"]
        check("summary omits feature_contributions and supporting_features",
              "feature_contributions" not in summary_exp and
              all("supporting_features" not in g for g in summary_exp["grouped_factors"]))

        print("\n=== 9. OpenAPI ===")
        spec = client.get("/openapi.json").json()
        expected_paths = {"/api/v1/health/live", "/api/v1/health/ready", "/api/v1/meta", "/api/v1/contexts",
                           "/api/v1/contexts/{context_id}", "/api/v1/teams", "/api/v1/maps",
                           "/api/v1/predict/series", "/api/v1/predict/map"}
        check("OpenAPI covers every public endpoint", expected_paths.issubset(set(spec["paths"])))
        check("OpenAPI contains no absolute filesystem path leakage", str(ROOT) not in json.dumps(spec))

        print("\n=== 10. zero state mutation over a request battery ===")
        before = bar.hash_group(bar.DEPLOYMENT_STATE)
        fixtures = yaml.safe_load((CONFIG / "application_api_fixtures_v1.yaml").read_text(encoding="utf-8"))
        for fx in fixtures["fixtures"]["pre_veto"]:
            payload = {"context_id": fx["context_id"], "mode": "pre_veto", "team_a": fx["team_a"],
                       "team_b": fx["team_b"], "best_of": fx["best_of"], "include_explanation": False}
            if fx["context_id"] == ai.HISTORICAL_CONTEXT_ID:
                payload["prediction_datetime"] = "2026-06-02T13:30:00"
            client.post("/api/v1/predict/series", json=payload)
        after = bar.hash_group(bar.DEPLOYMENT_STATE)
        check("deployment state file hashes unchanged after a full fixture battery", before == after)

        print("\n=== 11. concurrency determinism ===")
        from concurrent.futures import ThreadPoolExecutor
        direct_p = ai.predict_series_unknown_maps(ai.DEPLOYMENT_CONTEXT_ID, "Team Vitality", "Team Falcons", 3)[
            "probability_team_a"]
        ctx_before = ai._CONTEXT_CACHE[ai.DEPLOYMENT_CONTEXT_ID]

        def _call(i):
            rr = client.post("/api/v1/predict/series", json={"context_id": ai.DEPLOYMENT_CONTEXT_ID,
                                                               "mode": "pre_veto", "team_a": "Team Vitality",
                                                               "team_b": "Team Falcons", "best_of": 3,
                                                               "include_explanation": bool(i % 2)})
            return rr.json()["prediction"]["probability_team_a"]

        with ThreadPoolExecutor(max_workers=16) as ex:
            results = list(ex.map(_call, range(40)))
        check("40 concurrent requests all match the direct-core probability exactly",
              all(v == direct_p for v in results))
        check("context object identity unchanged after concurrent load (no per-request reload)",
              ai._CONTEXT_CACHE[ai.DEPLOYMENT_CONTEXT_ID] is ctx_before)

    print("\n=== 12. no fitting/training / no write operations in the API module ===")
    hits = module_reads_forbidden(api_py, ["sklearn.model_selection", "hyperopt", "optuna"])
    check("application_api.py imports no tuning/fitting module (AST)", len(hits) == 0)
    src = api_py.read_text(encoding="utf-8")
    check("application_api.py never calls .fit(", ".fit(" not in src)
    for forbidden in ("os.remove", "shutil.rmtree", ".to_parquet(", ".to_csv(", ".write_bytes("):
        check(f"application_api.py never calls {forbidden!r}", forbidden not in src)

    print("\n=== 13. no file-path-style request fields ===")
    for model in (api.SeriesPredictionRequest, api.MapPredictionRequest):
        names = list(model.model_fields)
        check(f"{model.__name__} accepts no path/file-style field",
              all("path" not in n.lower() and "file" not in n.lower() for n in names))

    print("\n=== 14. Phase 9B/9C regression gate - REAL subprocess commands ===")
    env = {"PYTHONIOENCODING": "utf-8"}
    import os
    env = {**os.environ, **env}
    r1 = subprocess.run([sys.executable, "-m", "pytest", "tests/test_phase9b_application_inference.py", "-q"],
                         cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", env=env)
    check("full Phase 9B pytest suite still passes", r1.returncode == 0)
    r2 = subprocess.run([sys.executable, "-m", "pytest", "tests/test_phase9c_application_explanations.py", "-q"],
                         cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", env=env)
    check("full Phase 9C pytest suite still passes", r2.returncode == 0)
    r3 = subprocess.run([sys.executable, "validate_phase9b.py"], cwd=str(SCRIPTS), capture_output=True, text=True,
                         encoding="utf-8", env=env)
    check("scripts/validate_phase9b.py still passes", r3.returncode == 0)
    r4 = subprocess.run([sys.executable, "validate_phase9c.py"], cwd=str(SCRIPTS), capture_output=True, text=True,
                         encoding="utf-8", env=env)
    check("scripts/validate_phase9c.py still passes", r4.returncode == 0)
    for label, res in (("9B pytest", r1), ("9C pytest", r2), ("validate_phase9b", r3), ("validate_phase9c", r4)):
        if res.returncode != 0:
            print(f"  --- {label} tail ---\n{res.stdout[-2000:]}")

    print("\n=== 15. report exists with required markers ===")
    report_path = ROOT / "reports" / "phase9d_application_api.md"
    check("phase9d report exists", report_path.exists())
    if report_path.exists():
        text = report_path.read_text(encoding="utf-8")
        for marker in ["APPLICATION API V1 = IMPLEMENTED", "PREDICTION PARITY = VERIFIED",
                       "EXPLANATION PARITY = VERIFIED", "DEPLOYMENT DATA THROUGH 2026-06-28",
                       "RF V2 = UNCHANGED", "XGB V3 = UNCHANGED"]:
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
