"""
Phase 9D application API version-freeze receipt. Refuses to overwrite an
already-committed valid receipt (mirrors the Phase 8E/9A/9B/9C commit-marker
discipline). Also snapshots /openapi.json to a versioned artifact so the
receipt can hash a concrete API surface, not just source files.
"""

import json

from fastapi.testclient import TestClient

from _common import ROOT
import phase9a_common as p9a
import build_application_registries as bar
import application_api as api

sha256_file = p9a.sha256_file

DEPLOY = ROOT / "data" / "deployment"
RECEIPT_PATH = DEPLOY / "application_api_receipt_v1.json"
OPENAPI_SNAPSHOT_PATH = DEPLOY / "phase9d_openapi_snapshot_v1.json"
CONFIG = ROOT / "config"
SCRIPTS = ROOT / "scripts"
TESTS = ROOT / "tests"


def preflight():
    if RECEIPT_PATH.exists():
        existing = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
        if existing.get("committed") is True:
            raise RuntimeError(f"STOP: a valid Phase 9D application API receipt already exists at "
                                f"{RECEIPT_PATH} (committed=True) - refusing to overwrite.")


def _write_openapi_snapshot():
    with TestClient(api.app) as c:
        spec = c.get("/openapi.json").json()
    OPENAPI_SNAPSHOT_PATH.write_text(json.dumps(spec, indent=2, sort_keys=True), encoding="utf-8")
    return OPENAPI_SNAPSHOT_PATH


def build():
    preflight()
    openapi_path = _write_openapi_snapshot()

    receipt = {
        "api_version": api.API_VERSION,
        "prediction_contract": api.PREDICTION_CONTRACT,
        "explanation_contract": api.EXPLANATION_VERSION,
        "default_context": api.DEFAULT_CONTEXT_ID,
        "committed": True,
        "hashes": {
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
            "openapi_snapshot": sha256_file(openapi_path),
        },
    }
    RECEIPT_PATH.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(f"wrote {RECEIPT_PATH} (COMMIT MARKER)")
    print(f"wrote {openapi_path}")
    return receipt


if __name__ == "__main__":
    build()
