"""
Phase 9A transactional commit: writes data/deployment/deployment_state_receipt_v1.json
LAST, as the commit marker (mirrors Phase 8D/8E's stage -> validate ->
promote -> receipt-last discipline). Refuses to overwrite an already-valid
receipt (see run_phase9a_pipeline.preflight, which calls this only after
that check and after validate_phase9a.py has passed).
"""

import json

import pandas as pd

from _common import ROOT
import feature_engineering.state.phase9a_common as p9a

RECEIPT_PATH = p9a.DEPLOY / "deployment_state_receipt_v1.json"


def build(historical_replay_before_hashes=None):
    manifest = pd.read_parquet(p9a.NEW_DEPLOYMENT_ARTIFACTS["deployment_history_manifest"], engine="fastparquet")
    included = manifest[manifest["history_status"] == "included"]
    deployment_cutoff = str(included["datetime"].max())
    latest_row = included.loc[included["datetime"].idxmax()]

    audit = pd.read_csv(p9a.NEW_DEPLOYMENT_ARTIFACTS["deployment_state_consumption_audit"])
    per_engine = {}
    for state_type, g in audit.groupby("state_type"):
        per_engine[state_type] = {
            "candidate_rows": int(len(g)), "eligible_rows": int(g["eligible_for_state"].sum()),
            "consumed_rows": int(g["consumed_by_state"].sum()),
            "eligible_but_not_consumed": int((g["eligible_for_state"] & ~g["consumed_by_state"]).sum()),
        }

    historical_replay_after = p9a.hash_historical_replay_record()
    if historical_replay_before_hashes is not None:
        unchanged_within_run = historical_replay_before_hashes == historical_replay_after
    else:
        unchanged_within_run = None  # standalone run - no true before/after available

    receipt = {
        "event_id": "iem_cologne_major_2026", "phase": "phase9a_post_cologne_deployment_state",
        "committed": True,

        "cutoffs": {
            "historical_evaluation_cutoff": "pre_cologne (2026-06-02T13:30:00, see Phase 8D/8D.1)",
            "deployment_history_cutoff": deployment_cutoff,
            "real_world_current_date": None,
            "note": "The dataset's latest locally available historical state ends at "
                    "deployment_history_cutoff. This is NOT the real-world current date and must never be "
                    "interpreted as live/current state by any later API code.",
        },

        "deployment_history": {
            "total_raw_rows": int(len(manifest)), "included": int(len(included)),
            "excluded_showmatch": int((manifest["history_status"] == "excluded_showmatch").sum()),
            "excluded_existing_reject": int((manifest["history_status"] == "excluded_existing_reject").sum()),
            "official_cologne_rows_included": 106,
            "post_cologne_rows_included": 32,
            "latest_included_match_id": int(latest_row["match_id"]),
            "latest_included_tournament": latest_row["tournament"],
        },

        "consumption_by_engine": per_engine,

        "active_map_pool_limitation": "Deployment history ends 2026-06-28, predating any later Active Duty "
                                       "change (e.g. a Cache re-addition). The modern-map state contains no "
                                       "legitimate post-change map experience. The engine itself has no "
                                       "map-name allowlist (any map_name string is accepted generically) - "
                                       "this is a data-coverage limitation, not an engine mechanism, and is "
                                       "NOT fixed in Phase 9A.",

        "historical_replay_unchanged_within_run": unchanged_within_run,

        "hashes": {
            "historical_replay_record": historical_replay_after,
            "deployment_build_inputs": p9a.hash_deployment_build_inputs(),
            "new_deployment_artifacts": {name: p9a.sha256_file(path)
                                          for name, path in p9a.NEW_DEPLOYMENT_ARTIFACTS.items()},
        },

        "policy_declarations": {
            "no_model_fitting": True, "no_retraining": True, "no_calibration": True,
            "no_train_set_modification": True, "no_phase7_test_reopening": True,
            "no_historical_replay_modification": True, "rf_v2_unchanged": True, "known_map_xgb_v3_unchanged": True,
        },
    }

    RECEIPT_PATH.write_text(json.dumps(receipt, indent=2, default=str), encoding="utf-8")
    print(f"wrote {RECEIPT_PATH} (COMMIT MARKER)")
    return receipt


if __name__ == "__main__":
    build()
