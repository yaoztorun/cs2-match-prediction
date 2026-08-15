"""
Phase 8E transactional commit: validate everything, THEN write the receipt
last, as the commit marker (mirrors run_phase8d_pipeline.py's stage ->
validate -> promote -> receipt-last discipline). Refuses to silently
overwrite an already-valid receipt. All Phase 8E artifacts in this session
were generated directly to their final paths in one continuous run (no
concurrent writers), so the safety property that matters - no receipt
without a fully passing validation pass immediately before it - is enforced
by calling the validator's checks in-process before writing anything.
"""

import hashlib
import json
import subprocess
import sys

from _common import ROOT
import phase8e_common as p8e

EVAL = ROOT / "data" / "evaluation"
RECEIPT_PATH = EVAL / "cologne_2026_simulation_vs_reality_receipt_v1.json"

EVALUATION_ARTIFACTS = {
    "protocol": ROOT / "config" / "phase8e_cologne_simulation_vs_reality_protocol.yaml",
    "result_source_manifest": ROOT / "data" / "tournaments" / "iem_cologne_major_2026_actual_results_sources.json",
    "result_source_snapshot": ROOT / "data" / "tournaments" / "iem_cologne_major_2026_actual_result_source_snapshot_v1.json",
    "reconciliation_table": EVAL / "cologne_2026_result_reconciliation_v1.csv",
    "actual_series_results": EVAL / "cologne_2026_actual_series_results_v1.parquet",
    "gate2_replay_artifact": EVAL / "cologne_2026_actual_engine_replay_v1.json",
    "actual_match_predictions": EVAL / "cologne_2026_actual_match_predictions_v1.parquet",
    "milestone_comparison": EVAL / "cologne_2026_actual_milestone_comparison_v1.csv",
    "metrics_detail": EVAL / "cologne_2026_phase8e_metrics_detail_v1.json",
    "simulation_vs_reality_summary": EVAL / "cologne_2026_simulation_vs_reality_summary_v1.json",
    "report": ROOT / "reports" / "phase8e_cologne_simulation_vs_reality.md",
}

EVALUATION_SCRIPTS = [
    "scripts/build_phase8e_protocol.py", "scripts/phase8e_common.py",
    "scripts/reconcile_cologne_actual_results.py", "scripts/phase8e_actual_outcome_provider.py",
    "scripts/build_phase8e_gate2_artifact.py", "scripts/build_phase8e_predictions.py",
    "scripts/phase8e_metrics.py", "scripts/build_phase8e_summary.py",
    "scripts/build_phase8e_source_snapshot.py", "scripts/build_phase8e_metrics_detail.py",
    "scripts/phase8e_figures.py", "scripts/validate_phase8e.py", "scripts/build_phase8e_receipt.py",
]

FIGURES_DIR = ROOT / "reports" / "figures" / "phase8e"


def preflight_check():
    if RECEIPT_PATH.exists():
        existing = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
        if existing.get("committed") is True:
            raise RuntimeError(f"STOP: a valid Phase 8E receipt already exists at {RECEIPT_PATH} "
                                f"(committed=True) - refusing to overwrite. Delete it explicitly first "
                                f"if regeneration is genuinely intended.")


def run_validator():
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / "validate_phase8e.py")],
                             cwd=str(ROOT / "scripts"), capture_output=True, text=True,
                             encoding="utf-8", env={"PYTHONIOENCODING": "utf-8", **__import__("os").environ})
    print(result.stdout[-2000:])
    if result.returncode != 0:
        raise RuntimeError("STOP: scripts/validate_phase8e.py failed - refusing to write the receipt. "
                            f"stderr: {result.stderr[-2000:]}")


def hash_dir(path, exts=(".png", ".pdf")):
    out = {}
    for p in sorted(path.glob("*")):
        if p.suffix in exts:
            out[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def build():
    preflight_check()
    print("=== running scripts/validate_phase8e.py before committing the receipt ===")
    run_validator()

    phase8d_receipt = json.loads((EVAL / "cologne_2026_pre_event_simulation_receipt_v1.json")
                                  .read_text(encoding="utf-8"))

    hashes = {
        "immutable_pre_event_record": p8e.hash_immutable_pre_event_record(),
        "evaluation_artifacts": {name: p8e.sha256_file(path) for name, path in EVALUATION_ARTIFACTS.items()},
        "evaluation_scripts": {name: p8e.sha256_file(ROOT / name) for name in EVALUATION_SCRIPTS},
        "figures": hash_dir(FIGURES_DIR),
    }

    receipt = {
        "event_id": "iem_cologne_major_2026",
        "phase": "phase8e_simulation_vs_reality",
        "committed": True,
        "phase8d_simulation_receipt_reference": {
            "path": "data/evaluation/cologne_2026_pre_event_simulation_receipt_v1.json",
            "protocol_hash": phase8d_receipt.get("protocol_hash"),
            "n_simulations": phase8d_receipt.get("n_simulations"),
            "created_before_results_opened": phase8d_receipt.get("created_before_results_opened"),
            "cologne_results_status_at_phase8d_time": phase8d_receipt.get("cologne_results_status"),
            "own_file_hash": p8e.sha256_file(EVAL / "cologne_2026_pre_event_simulation_receipt_v1.json"),
        },
        "reconciliation": {"original_rows": 107, "official_rows": 106,
                            "excluded_row": "Team Germany vs Team Poland showmatch, 2026-06-21"},
        "engine_replay": {"total_matched": "106/106", "unused": 0, "duplicate": 0, "ambiguous": 0, "missing": 0},
        "hashes": hashes,
        "policy_declarations": {
            "no_retraining": True, "no_recalibration": True, "no_feature_engineering": True,
            "no_threshold_adjustment": True, "no_model_replacement": True, "no_state_update": True,
            "no_cologne_training_ingestion_yet": True, "no_api_yet": True, "no_pwa_yet": True,
            "rf_v2_never_loaded_in_phase8e": True,
        },
    }

    RECEIPT_PATH.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(f"\nwrote {RECEIPT_PATH} (COMMIT MARKER)")
    return receipt


if __name__ == "__main__":
    build()
