"""
Phase 9C explanation-version freeze receipt. Refuses to overwrite an
already-committed valid receipt (mirrors the Phase 8E/9A/9B commit-marker
discipline).
"""

import json

from _common import ROOT
import phase9a_common as p9a
import build_application_registries as bar

sha256_file = p9a.sha256_file

DEPLOY = ROOT / "data" / "deployment"
RECEIPT_PATH = DEPLOY / "application_explanation_receipt_v1.json"
CONFIG = ROOT / "config"
SCRIPTS = ROOT / "scripts"
FIXTURES_DIR = DEPLOY / "phase9c_fixtures"


def preflight():
    if RECEIPT_PATH.exists():
        existing = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
        if existing.get("committed") is True:
            raise RuntimeError(f"STOP: a valid Phase 9C explanation receipt already exists at "
                                f"{RECEIPT_PATH} (committed=True) - refusing to overwrite.")


def build():
    preflight()

    fixture_files = sorted(FIXTURES_DIR.glob("*.json"))

    receipt = {
        "explanation_version": "application_explanations_v1",
        "prediction_contract": "phase9b",
        "causal": False,
        "committed": True,
        "hashes": {
            "application_explanations_py": sha256_file(SCRIPTS / "application_explanations.py"),
            "explanation_registry": sha256_file(CONFIG / "application_explanations_v1.yaml"),
            "feature_groups_registry": sha256_file(CONFIG / "application_explanation_feature_groups_v1.yaml"),
            "fixture_manifest": sha256_file(CONFIG / "application_explanation_fixtures_v1.yaml"),
            "application_inference_py": sha256_file(SCRIPTS / "application_inference.py"),
            "phase9b_context_registry": sha256_file(CONFIG / "application_inference_contexts_v1.yaml"),
            "rf_v2_model": sha256_file(bar.RF_PIPELINE["rf_model"]),
            "rf_v2_preprocessing": sha256_file(bar.RF_PIPELINE["rf_preprocessing"]),
            "xgb_v3_model": sha256_file(bar.XGB_PIPELINE["xgb_model"]),
            "xgb_v3_preprocessing": sha256_file(bar.XGB_PIPELINE["xgb_preprocessing"]),
            "fixture_outputs": {f.name: sha256_file(f) for f in fixture_files},
        },
    }
    RECEIPT_PATH.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(f"wrote {RECEIPT_PATH} (COMMIT MARKER)")
    return receipt


if __name__ == "__main__":
    build()
