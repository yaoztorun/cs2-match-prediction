"""
Phase 9E tournament service version-freeze receipt. Refuses to overwrite an
already-committed valid receipt (mirrors the Phase 8D/8E/9B/9C/9D
commit-marker discipline). Written LAST, after tests/validator pass.
"""

import json

from _common import ROOT
import feature_engineering.state.phase9a_common as p9a

sha256_file = p9a.sha256_file

DEPLOY = ROOT / "data" / "deployment"
RECEIPT_PATH = DEPLOY / "application_tournament_service_receipt_v1.json"
CONFIG = ROOT / "config"
TESTS = ROOT / "tests"
EVAL = ROOT / "data" / "evaluation"


def preflight():
    if RECEIPT_PATH.exists():
        existing = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
        if existing.get("committed") is True:
            raise RuntimeError(f"STOP: a valid Phase 9E tournament service receipt already exists at "
                                f"{RECEIPT_PATH} (committed=True) - refusing to overwrite.")


def build():
    preflight()

    receipt = {
        "service_version": "application_tournament_service_v1",
        "default_prediction_context": "deployment_post_cologne_v1",
        "historical_cologne_contract": "frozen_phase8d_phase8e",
        "default_ruleset_id": "iem_cologne_major_2026_format_v1",
        "committed": True,
        "hashes": {
            "application_tournament_service_py": sha256_file(ROOT / "application" / "tournament" / "application_tournament_service.py"),
            "application_tournament_router_py": sha256_file(ROOT / "application" / "tournament" / "application_tournament_router.py"),
            "ruleset_registry": sha256_file(CONFIG / "application" / "application_tournament_rulesets_v1.yaml"),
            "fixture_manifest": sha256_file(CONFIG / "application" / "application_tournament_fixtures_v1.yaml"),
            "tournament_engine_py": sha256_file(ROOT / "tournament" / "engine" / "tournament_engine.py"),
            "phase9b_context_registry": sha256_file(CONFIG / "application" / "application_inference_contexts_v1.yaml"),
            "phase9d_api_receipt": sha256_file(DEPLOY / "application_api_receipt_v1.json"),
            "phase8d_simulation_receipt": sha256_file(EVAL / "cologne_2026_pre_event_simulation_receipt_v1.json"),
            "phase8e_evaluation_receipt": sha256_file(EVAL / "cologne_2026_simulation_vs_reality_receipt_v1.json"),
            "test_phase9e_py": sha256_file(TESTS / "application" / "test_phase9e_application_tournament_service.py"),
        },
    }
    RECEIPT_PATH.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(f"wrote {RECEIPT_PATH} (COMMIT MARKER)")
    return receipt


if __name__ == "__main__":
    build()
