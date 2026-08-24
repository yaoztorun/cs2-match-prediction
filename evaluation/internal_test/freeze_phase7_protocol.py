"""
Phase 7, Stage A: freeze the TEST evaluation protocol BEFORE TEST is opened.

Reads ZERO TEST content. Every hash below is a plain file hash
(hashlib.sha256 of raw bytes) - none of these reads ever open
data/modeling/map_split_v1.csv as a dataframe or filter it by `split`, so
this script trivially satisfies the "only evaluation/internal_test/evaluate_phase7_test_once.py
may select split=='test'" rule (brief section 30) - it never even loads that
file as a dataframe, just hashes its bytes.

Must be run AFTER evaluation/internal_test/evaluate_phase7_test_once.py,
evaluation/internal_test/phase7_test_reports.py, evaluation/uncertainty/phase7_test_bootstrap.py and
evaluation/internal_test/phase7_test_visualizations.py are written (so their source hashes can
be captured) but BEFORE any of them is ever executed against real data - this
is what "protocol frozen before evaluator execution" means in practice.

Writes:
    data/evaluation/phase7_test_protocol_v1.json
"""

import hashlib
import json

from _common import ROOT

EVAL_DIR = ROOT / "data" / "evaluation"

FROZEN_ARTIFACTS = {
    "final_model": ROOT / "models" / "map" / "map_xgboost_v3_final.json",
    "final_model_metadata": ROOT / "models" / "map" / "map_xgboost_v3_final_metadata.json",
    "final_preprocessing": ROOT / "data" / "modeling" / "map_xgboost_v3_final_preprocessing.json",
    "final_xgb_config": ROOT / "data" / "modeling" / "map_xgboost_v3_final_config.json",
    "v3_feature_config": ROOT / "config" / "features" / "map_features_v3_modern_map.yaml",
    "v3_feature_parquet": ROOT / "data" / "features" / "map_features_v3_modern_map.parquet",
    "test_split_manifest": ROOT / "data" / "modeling" / "map_split_v1.csv",   # FILE HASH ONLY - never opened as a dataframe here
}
EVAL_CODE = {
    "evaluate_phase7_test_once": ROOT / "evaluation" / "internal_test" / "evaluate_phase7_test_once.py",
    "phase7_test_reports": ROOT / "evaluation" / "internal_test" / "phase7_test_reports.py",
    "phase7_test_bootstrap": ROOT / "evaluation" / "uncertainty" / "phase7_test_bootstrap.py",
    "phase7_test_visualizations": ROOT / "evaluation" / "internal_test" / "phase7_test_visualizations.py",
    "phase7_evaluation_protocol_config": ROOT / "config" / "evaluation" / "phase7_test_evaluation_protocol.yaml",
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    for label, p in {**FROZEN_ARTIFACTS, **EVAL_CODE}.items():
        if not p.exists():
            raise RuntimeError(f"cannot freeze the protocol: {label} does not exist yet at {p}")

    protocol = {
        "protocol_version": "1.0.0",
        "model_under_evaluation": "map_xgboost_v3_final",
        "threshold": 0.5,
        "artifact_hashes": {label: sha256(p) for label, p in FROZEN_ARTIFACTS.items()},
        "evaluation_code_hashes": {label: sha256(p) for label, p in EVAL_CODE.items()},
        "expected_test_row_count": 1427,
        "bootstrap": {"n_bootstrap": 2000, "random_state": 42, "ci": 0.95, "cluster_key": "match_id"},
        "calibration_bins": {"n_bins": 10, "edges": [round(i / 10, 1) for i in range(11)],
                              "last_bin_closed_at_one": True},
        "no_test_outcome_present": True,
    }
    protocol["protocol_hash"] = hashlib.sha256(
        json.dumps({k: v for k, v in protocol.items() if k != "protocol_hash"},
                    sort_keys=True, default=str).encode("utf-8")).hexdigest()

    EVAL_DIR.mkdir(exist_ok=True, parents=True)
    out_path = EVAL_DIR / "phase7_test_protocol_v1.json"
    out_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    print(f"Wrote {out_path}")
    print(f"protocol_hash = {protocol['protocol_hash']}")
    print("Artifact hashes:")
    for label, h in protocol["artifact_hashes"].items():
        print(f"  {label}: {h[:16]}...")
    print("Evaluation code hashes:")
    for label, h in protocol["evaluation_code_hashes"].items():
        print(f"  {label}: {h[:16]}...")
    print("\nPROTOCOL FROZEN. TEST has not been opened by this script.")


if __name__ == "__main__":
    main()
