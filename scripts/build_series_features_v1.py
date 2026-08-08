"""
Phase 3 Mode A: development dataset generation.

Dataset-specific orchestration on top of the reusable engine in
feature_engine.py. This script owns all "Kaggle"/"development" concepts - the
engine itself knows nothing about them.

Writes:
    data/interim/phase3_identity_excluded_matches.csv
    data/features/series_features_v1.parquet
    data/features/series_team_states_v1.parquet   (audit/debug only)
    data/features/series_team_state_v1_full.json  (full re-loadable state)

Read-only against data/raw/ and data/interim/ inputs.
"""

from pathlib import Path

import pandas as pd
import yaml

from _common import INTERIM, ROOT, raw_file_hashes
from feature_engine import StateStore, process_chronological_stream, ENGINE_VERSION

SOURCE = "kaggle_ektarr"
FEATURES_DIR = ROOT / "data" / "features"
FEATURES_DIR.mkdir(exist_ok=True, parents=True)
CONFIG_PATH = ROOT / "config" / "series_features_v1.yaml"


def load_config():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def canonical_eligibility_map():
    policy = pd.read_csv(INTERIM / "team_identity_policy.csv")
    return policy.groupby("canonical_team_name")["identity_feature_eligible"].min().astype(bool).to_dict()


def build_stream_rows(dev):
    canon_elig = canonical_eligibility_map()

    rows = dev.copy()
    rows["source"] = SOURCE
    rows["source_match_id"] = rows["match_id"].astype(str)
    rows["canonical_match_uid"] = SOURCE + ":" + rows["source_match_id"]
    rows["best_of"] = rows["bestOf"]
    rows["score1"] = rows["score1_match"]
    rows["score2"] = rows["score2_match"]
    rows["team1_win"] = rows["team1_series_win"]

    t1_elig = rows["team1_canonical"].map(canon_elig)
    t2_elig = rows["team2_canonical"].map(canon_elig)
    if t1_elig.isna().any() or t2_elig.isna().any():
        missing = set(rows.loc[t1_elig.isna(), "team1_canonical"]) | set(rows.loc[t2_elig.isna(), "team2_canonical"])
        raise ValueError(f"canonical names missing from team_identity_policy.csv: {missing}")
    rows["team1_eligible"] = t1_elig
    rows["team2_eligible"] = t2_elig

    return rows


def main():
    hashes_before = raw_file_hashes()
    cfg = load_config()

    sb = pd.read_parquet(INTERIM / "series_base.parquet", engine="fastparquet")
    em = pd.read_csv(INTERIM / "evaluation_manifest.csv")
    merged = sb.merge(em, on="match_id", how="inner")
    dev = merged[merged["evaluation_group"] == "development"].copy()
    print(f"development pool (series_base retained rows labeled development): {len(dev)}")

    rows = build_stream_rows(dev)

    store = StateStore()
    processed_rows, excluded_rows = process_chronological_stream(store, rows)

    print(f"processed (eligible-pair, got a feature row): {len(processed_rows)}")
    print(f"excluded (>=1 identity-ineligible side): {len(excluded_rows)}")
    assert len(processed_rows) + len(excluded_rows) == len(dev)

    # ---- phase3_identity_excluded_matches.csv ----
    excl_df = pd.DataFrame(excluded_rows)
    excl_out = excl_df[["match_id", "team1", "team2", "datetime", "reason"]].sort_values("datetime")
    excl_out.to_csv(INTERIM / "phase3_identity_excluded_matches.csv", index=False, encoding="utf-8")

    # ---- series_features_v1.parquet (model-ready) ----
    proc_df = pd.DataFrame(processed_rows)
    metadata_cols = cfg["metadata"] + [cfg["target"]]
    model_cols = cfg["model_features"]
    features_df = proc_df[metadata_cols + model_cols].sort_values(["datetime", "match_id"]).reset_index(drop=True)
    features_df.to_parquet(FEATURES_DIR / "series_features_v1.parquet", engine="fastparquet", index=False)

    # ---- series_team_states_v1.parquet (AUDIT/DEBUG ONLY - not the modelling table) ----
    audit_cols = ["match_id", "datetime", "team1_canonical", "team2_canonical"] + \
        [c for c in proc_df.columns if c.startswith("team1_") or c.startswith("team2_")]
    audit_cols = list(dict.fromkeys(audit_cols))  # de-dup preserving order
    audit_df = proc_df[audit_cols].sort_values(["datetime", "match_id"]).reset_index(drop=True)
    audit_df.to_parquet(FEATURES_DIR / "series_team_states_v1.parquet", engine="fastparquet", index=False)

    # ---- full re-loadable state ----
    meta = {
        "mode": "A_development_full",
        "source": SOURCE,
        "cutoff_datetime": str(dev["datetime"].max()),
        "n_input_rows": len(dev),
        "n_feature_rows": len(processed_rows),
        "n_excluded_rows": len(excluded_rows),
        "generated_at": str(pd.Timestamp.now()),
    }
    store.to_json(FEATURES_DIR / "series_team_state_v1_full.json", meta=meta)

    hashes_after = raw_file_hashes()
    assert hashes_before == hashes_after, "data/raw/ was modified during Phase 3 build - this must never happen"

    print(f"Wrote {FEATURES_DIR / 'series_features_v1.parquet'} ({len(features_df)} rows)")
    print(f"Wrote {FEATURES_DIR / 'series_team_states_v1.parquet'} (AUDIT ONLY, {len(audit_df)} rows)")
    print(f"Wrote {FEATURES_DIR / 'series_team_state_v1_full.json'} (engine v{ENGINE_VERSION}, "
          f"{len(store.teams)} teams, {len(store.processed_match_uids)} matches processed)")
    print(f"Wrote {INTERIM / 'phase3_identity_excluded_matches.csv'} ({len(excl_out)} rows)")


if __name__ == "__main__":
    main()
