"""
Phase 7, Stage B: THE ONE-SHOT TEST READ.

This is the ONLY script in Phase 7 permitted to filter
data/modeling/map_split_v1.csv by split == "test" or merge it against
data/features/map_features_v3_modern_map.parquet for TEST target rows.
validation/validate_phase7.py enforces this as an executable AST check, not just
a convention.

Contains NO model-fitting call anywhere (also AST-checked by the validator
and by tests/evaluation/test_phase7_internal_test.py).

Sequence (brief section 6):
  1. load the frozen protocol, recompute every artifact hash fresh, assert
     equality (abort on any mismatch - proves which frozen system is scored);
  2. load the TEST partition, assert exactly 1,427 rows, no duplicate
     (match_id, game_id), no match_id crosses a partition;
  3. merge onto the V3 feature table (validate="one_to_one", must be total);
  4. assert zero Cologne/post-Cologne match_id (group-label lookup only -
     not "opening Cologne data");
  5-9. load (never fit) the frozen preprocessing + model, transform TEST
     exactly once, predict, compute baseline probabilities, compute the
     mirrored-inference diagnostic;
  10. ATOMIC commit: abort if the canonical parquet already exists; write to
      a temp file in the same directory; verify the temp file's row
      count/uniqueness/columns/probability domain; os.replace() the temp file
      into the canonical path; hash the now-committed canonical file;
  11. write the TEST-open receipt, also atomically.

If step 2, 3 or 4 fails, the script raises BEFORE any prediction is made.
"""

import hashlib
import json
import os

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from _common import ROOT
from feature_engineering.preprocessing.preprocessing_common_map_v3 import load_map_v3_roles
from feature_engineering.preprocessing.preprocessing_common_map_v2 import mirror_raw_rows
import feature_engineering.preprocessing.preprocessing_xgboost_map_v3 as prep_xgb
from training.map_models.map_modeling_common import baseline_probabilities

EVAL_DIR = ROOT / "data" / "evaluation"
PROTOCOL_PATH = EVAL_DIR / "phase7_test_protocol_v1.json"
CANONICAL_PATH = EVAL_DIR / "map_test_predictions_v1.parquet"
TEMP_PATH = EVAL_DIR / "map_test_predictions_v1.parquet.tmp"
RECEIPT_PATH = EVAL_DIR / "phase7_test_open_receipt_v1.json"
RECEIPT_TEMP_PATH = EVAL_DIR / "phase7_test_open_receipt_v1.json.tmp"

CONFIG_PATH = ROOT / "config" / "features" / "map_features_v3_modern_map.yaml"
FEATURES_PATH = ROOT / "data" / "features" / "map_features_v3_modern_map.parquet"
SPLIT_PATH = ROOT / "data" / "modeling" / "map_split_v1.csv"
MODEL_PATH = ROOT / "models" / "map" / "map_xgboost_v3_final.json"
MODEL_META_PATH = ROOT / "models" / "map" / "map_xgboost_v3_final_metadata.json"
PREP_PATH = ROOT / "data" / "modeling" / "map_xgboost_v3_final_preprocessing.json"
FINAL_CONFIG_PATH = ROOT / "data" / "modeling" / "map_xgboost_v3_final_config.json"

EXPECTED_TEST_ROWS = 1427

IDENTITY_COLS = ["match_id", "game_id", "series_datetime", "map_name", "bestOf", "tier",
                  "team1_canonical", "team2_canonical"]
COVERAGE_COLS = [
    "both_teams_have_recent_selected_map_history", "map_adjusted_history_mass_min",
    "roster_map_players_with_history_min", "current_core_map_continuity_min",
    "selected_map_in_both_recent_pools", "both_teams_have_map_history", "both_teams_have_5_map_matches",
]
REQUIRED_ARTIFACT_COLUMNS = (
    IDENTITY_COLS + ["y_true", "p_xgb_v3_final", "y_pred_xgb_v3_final", "p_constant_05", "p_overall_elo",
                      "p_map_elo", "p_xgb_v3_mirrored"] + COVERAGE_COLS
)
PROBABILITY_COLS = ["p_xgb_v3_final", "p_constant_05", "p_overall_elo", "p_map_elo", "p_xgb_v3_mirrored"]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_protocol_hashes(protocol):
    current = {
        "final_model": sha256(MODEL_PATH),
        "final_model_metadata": sha256(MODEL_META_PATH),
        "final_preprocessing": sha256(PREP_PATH),
        "final_xgb_config": sha256(FINAL_CONFIG_PATH),
        "v3_feature_config": sha256(CONFIG_PATH),
        "v3_feature_parquet": sha256(FEATURES_PATH),
        "test_split_manifest": sha256(SPLIT_PATH),
    }
    mismatches = {k: (protocol["artifact_hashes"][k], current[k])
                  for k in current if protocol["artifact_hashes"][k] != current[k]}
    if mismatches:
        raise RuntimeError(f"FROZEN ARTIFACT HASH MISMATCH - the system on disk differs from what the protocol "
                            f"froze. Aborting before any TEST row is read. Mismatches: {mismatches}")
    print("Artifact-hash verification PASSED - the exact frozen system matches the protocol.")


def main():
    if not PROTOCOL_PATH.exists():
        raise RuntimeError(f"{PROTOCOL_PATH} does not exist - run evaluation/internal_test/freeze_phase7_protocol.py first.")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    verify_protocol_hashes(protocol)

    # ---- ABORT-BEFORE-SCORING GUARD: canonical artifact must not already exist ----
    if CANONICAL_PATH.exists():
        raise RuntimeError(f"{CANONICAL_PATH} already exists - refusing to overwrite or regenerate the "
                            "canonical one-shot TEST prediction artifact. Aborting immediately.")
    # A stale temp file may be cleaned ONLY because we just proved the canonical file is absent.
    if TEMP_PATH.exists():
        print(f"Removing stale temp file from a previous failed run: {TEMP_PATH}")
        TEMP_PATH.unlink()
    if RECEIPT_PATH.exists():
        raise RuntimeError(f"{RECEIPT_PATH} already exists but {CANONICAL_PATH} does not - inconsistent prior "
                            "state. Aborting rather than guessing.")
    if RECEIPT_TEMP_PATH.exists():
        RECEIPT_TEMP_PATH.unlink()

    # =====================================================================
    # 2. THE ONE PLACE IN PHASE 7 THAT FILTERS split == "test"
    # =====================================================================
    split = pd.read_csv(SPLIT_PATH)
    crossing = split.groupby("match_id")["split"].nunique()
    if not (crossing == 1).all():
        raise RuntimeError("a match_id crosses more than one partition - STOPPING before scoring.")
    test_ids = split[split["split"] == "test"][["match_id", "game_id"]]
    if len(test_ids) != EXPECTED_TEST_ROWS:
        raise RuntimeError(f"TEST partition has {len(test_ids)} rows, expected {EXPECTED_TEST_ROWS} - "
                            "STOPPING before scoring.")
    if test_ids.duplicated(subset=["match_id", "game_id"]).any():
        raise RuntimeError("duplicate (match_id, game_id) in the TEST partition - STOPPING before scoring.")
    print(f"TEST partition: {len(test_ids)} rows (assertion PASSED)")

    # =====================================================================
    # 3. merge onto the V3 feature table - must be total
    # =====================================================================
    features = pd.read_parquet(FEATURES_PATH, engine="fastparquet")
    test_raw = test_ids.merge(features, on=["match_id", "game_id"], how="left", validate="one_to_one")
    if test_raw["team1_map_win"].isna().any() or len(test_raw) != EXPECTED_TEST_ROWS:
        raise RuntimeError("TEST merge onto the V3 feature table was not total - STOPPING before scoring.")
    if not set(pd.unique(test_raw["team1_map_win"])) <= {0, 1}:
        raise RuntimeError("TEST target is not binary - STOPPING before scoring.")
    print(f"TEST merge onto map_features_v3_modern_map.parquet: total, {len(test_raw)} rows.")

    # =====================================================================
    # 4. zero Cologne / post-Cologne match_id (group-label lookup only)
    # =====================================================================
    em = pd.read_csv(ROOT / "data" / "interim" / "evaluation_manifest.csv")
    cologne_ids = set(em.loc[em["evaluation_group"].isin(["cologne_2026", "post_cologne"]), "match_id"])
    if set(test_raw["match_id"]) & cologne_ids:
        raise RuntimeError("a Cologne/post-Cologne match_id was found in TEST - STOPPING before scoring.")
    print("Cologne/post-Cologne cross-check PASSED: zero overlap.")

    # =====================================================================
    # 5-9. load (never fit) preprocessing + model; predict; baselines; mirrored diagnostic
    # =====================================================================
    roles = load_map_v3_roles(CONFIG_PATH)
    params = prep_xgb.load_preprocessing(PREP_PATH)
    model = XGBClassifier()
    model.load_model(str(MODEL_PATH))   # load only - no .fit anywhere in this script

    X_test, _ = prep_xgb.transform(test_raw, params, roles)   # exactly once
    p_final = model.predict_proba(X_test)[:, 1]
    y_pred_final = (p_final >= 0.5).astype(int)

    p_const = baseline_probabilities(test_raw, "half")
    p_overall_elo = baseline_probabilities(test_raw, "overall_elo")
    p_map_elo = baseline_probabilities(test_raw, "map_elo")

    mirrored = mirror_raw_rows(test_raw, roles)   # diagnostic transform only - no target flip is used downstream
    X_mirrored, _ = prep_xgb.transform(mirrored, params, roles)
    p_mirrored = model.predict_proba(X_mirrored)[:, 1]

    out = pd.DataFrame({c: test_raw[c].to_numpy() for c in IDENTITY_COLS})
    out["y_true"] = test_raw["team1_map_win"].to_numpy(dtype=int)
    out["p_xgb_v3_final"] = p_final
    out["y_pred_xgb_v3_final"] = y_pred_final
    out["p_constant_05"] = p_const
    out["p_overall_elo"] = p_overall_elo
    out["p_map_elo"] = p_map_elo
    out["p_xgb_v3_mirrored"] = p_mirrored
    for c in COVERAGE_COLS:
        out[c] = test_raw[c].to_numpy()

    missing_cols = [c for c in REQUIRED_ARTIFACT_COLUMNS if c not in out.columns]
    assert not missing_cols, f"prediction artifact missing required columns: {missing_cols}"

    # =====================================================================
    # 10. ATOMIC commit: temp file -> verify -> os.replace into the canonical path
    # =====================================================================
    out.to_parquet(TEMP_PATH, engine="fastparquet", index=False)

    verify = pd.read_parquet(TEMP_PATH, engine="fastparquet")
    assert len(verify) == EXPECTED_TEST_ROWS, f"temp artifact has {len(verify)} rows, expected {EXPECTED_TEST_ROWS}"
    assert verify.duplicated(subset=["match_id", "game_id"]).sum() == 0, "duplicate rows in the temp artifact"
    assert not [c for c in REQUIRED_ARTIFACT_COLUMNS if c not in verify.columns], "temp artifact missing columns"
    assert set(pd.unique(verify["y_true"])) <= {0, 1}, "temp artifact target is not binary"
    for c in PROBABILITY_COLS:
        vals = verify[c].to_numpy(dtype=float)
        assert np.isfinite(vals).all() and (vals >= 0).all() and (vals <= 1).all(), \
            f"temp artifact column {c} has a non-finite or out-of-[0,1] value"
    print("Temp artifact verification PASSED (row count, uniqueness, columns, target, probability domain).")

    os.replace(TEMP_PATH, CANONICAL_PATH)   # atomic on the same filesystem
    assert CANONICAL_PATH.exists() and not TEMP_PATH.exists()
    canonical_hash = sha256(CANONICAL_PATH)
    print(f"Committed canonical TEST prediction artifact: {CANONICAL_PATH} (sha256 {canonical_hash[:16]}...)")

    # =====================================================================
    # 11. TEST-open receipt - also written atomically
    # =====================================================================
    receipt = {
        "protocol_hash": protocol["protocol_hash"],
        "input_artifact_hashes": protocol["artifact_hashes"],
        "prediction_artifact_hash": canonical_hash,
        "prediction_artifact_path": "data/evaluation/map_test_predictions_v1.parquet",
        "test_row_count": int(len(out)),
        "distinct_match_count": int(out["match_id"].nunique()),
        "probability_range": {c: [float(out[c].min()), float(out[c].max())] for c in PROBABILITY_COLS},
        "target_balance": {"team1_wins": int(out["y_true"].sum()), "team2_wins": int((1 - out["y_true"]).sum()),
                            "team1_win_rate": float(out["y_true"].mean())},
    }
    RECEIPT_TEMP_PATH.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    os.replace(RECEIPT_TEMP_PATH, RECEIPT_PATH)
    print(f"Wrote {RECEIPT_PATH}")
    print("\nTEST OPENED. Exactly once. The canonical prediction artifact is now immutable.")


if __name__ == "__main__":
    main()
