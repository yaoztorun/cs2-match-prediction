"""
Phase 3 Mode B: frozen pre-Cologne team-state snapshot.

Processes the exact SAME full development stream as Mode A (same rows, same
identity-eligibility/update rules - including single-eligible matches updating
the eligible team's independent history) through a FRESH StateStore, further
restricted to datetime < the first Cologne 2026 match. This is a separate
what-if run, not a continuation of Mode A's store.

The Cologne cutoff is derived PRIMARILY from data/interim/evaluation_manifest.csv
(evaluation_group == "cologne_2026"), not from the tournament name string - the
tournament name is only used as a secondary sanity cross-check.

Writes:
    data/features/pre_cologne_team_state_v1.parquet  (flat/scalar summary only)
    data/features/pre_cologne_team_state_v1_full.json (full re-loadable state)

Read-only against data/raw/ and data/interim/ inputs. Asserts no Cologne
match ever contributes to this snapshot.
"""

import pandas as pd

from _common import INTERIM, load_games_tiered, raw_file_hashes
from feature_engineering.series.feature_engine import StateStore, process_chronological_stream, ENGINE_VERSION
from feature_engineering.series.build_series_features_v1 import SOURCE, build_stream_rows, FEATURES_DIR


def cologne_match_ids_and_cutoff(em):
    """Primary definition: evaluation_manifest's cologne_2026 group. Tournament
    name is only used below as a secondary sanity cross-check, never as the
    sole definition of the holdout."""
    cologne_ids = set(em.loc[em["evaluation_group"] == "cologne_2026", "match_id"])

    raw = load_games_tiered()
    tt = raw[raw["is_total"] == "True"].copy()
    tt["match_id_num"] = pd.to_numeric(tt["match_id"]).astype("Int64")
    tt["dt"] = pd.to_datetime(tt["datetime"], format="%Y-%m-%d %H:%M:%S", errors="coerce")

    cologne_rows = tt[tt["match_id_num"].isin(cologne_ids)]
    if len(cologne_rows) != len(cologne_ids):
        raise AssertionError(f"expected {len(cologne_ids)} cologne rows in raw data, found {len(cologne_rows)}")

    non_cologne_name = cologne_rows[cologne_rows["tournament"] != "IEM Cologne Major 2026"]
    if len(non_cologne_name) > 0:
        print(f"WARNING: {len(non_cologne_name)} evaluation_manifest cologne_2026 match_ids do NOT carry the "
              f"'IEM Cologne Major 2026' tournament name in raw data - manifest label still governs the holdout, "
              f"this is only a sanity cross-check: {non_cologne_name['match_id'].tolist()[:10]}")

    cutoff = cologne_rows["dt"].min()
    return cologne_ids, cutoff


def main():
    hashes_before = raw_file_hashes()

    sb = pd.read_parquet(INTERIM / "series_base.parquet", engine="fastparquet")
    em = pd.read_csv(INTERIM / "evaluation_manifest.csv")
    merged = sb.merge(em, on="match_id", how="inner")
    dev = merged[merged["evaluation_group"] == "development"].copy()

    cologne_ids, cutoff = cologne_match_ids_and_cutoff(em)
    print(f"Cologne match_ids (from evaluation_manifest): {len(cologne_ids)}; cutoff datetime: {cutoff}")

    rows = build_stream_rows(dev)

    # Mode B additional restriction: datetime strictly before the Cologne cutoff.
    # Development, by the Phase 2 manifest definition, already excludes Cologne
    # itself - so this is the only extra filter needed.
    pre_cologne_rows = rows[rows["datetime"] < cutoff].copy()
    print(f"full development pool: {len(rows)} | pre-Cologne subset (datetime < cutoff): {len(pre_cologne_rows)}")

    assert set(pre_cologne_rows["match_id"]).isdisjoint(cologne_ids), \
        "a Cologne match_id leaked into the pre-Cologne input stream"
    assert (pre_cologne_rows["datetime"] < cutoff).all()

    store = StateStore()
    processed_rows, excluded_rows = process_chronological_stream(store, pre_cologne_rows)
    print(f"processed (eligible-pair, contributed to state): {len(processed_rows)}")
    print(f"excluded (>=1 identity-ineligible side, eligible side may still have partially updated): {len(excluded_rows)}")

    cologne_uids = {f"{SOURCE}:{mid}" for mid in cologne_ids}
    assert store.processed_match_uids.isdisjoint(cologne_uids), \
        "a Cologne match contributed to the pre-Cologne snapshot's state"
    for ts in store.teams.values():
        for h in ts.history:
            assert h.canonical_match_uid not in cologne_uids, \
                f"Cologne match {h.canonical_match_uid} found in {ts.canonical_name}'s history"
            assert h.dt < cutoff, f"history entry at/after Cologne cutoff found in {ts.canonical_name}'s history"

    # ---- flat/scalar parquet summary (fastparquet-safe: no lists/nested cols) ----
    summary = store.snapshot_summary_df()
    summary["engine_version"] = ENGINE_VERSION
    summary["cutoff_datetime"] = str(cutoff)
    summary["source_dataset"] = SOURCE
    summary["n_matches_processed_total"] = len(store.processed_match_uids)
    summary = summary.sort_values("canonical_team_name").reset_index(drop=True)
    summary.to_parquet(FEATURES_DIR / "pre_cologne_team_state_v1.parquet", engine="fastparquet", index=False)

    # ---- full re-loadable state ----
    meta = {
        "mode": "B_pre_cologne_frozen_snapshot",
        "source": SOURCE,
        "cutoff_datetime": str(cutoff),
        "cologne_match_ids_excluded": len(cologne_ids),
        "n_input_rows": len(pre_cologne_rows),
        "n_feature_rows_discarded_by_design": len(processed_rows),  # Mode B doesn't need feature rows, only final state
        "generated_at": str(pd.Timestamp.now()),
    }
    store.to_json(FEATURES_DIR / "pre_cologne_team_state_v1_full.json", meta=meta)

    hashes_after = raw_file_hashes()
    assert hashes_before == hashes_after, "data/raw/ was modified during Phase 3 build - this must never happen"

    print(f"Wrote {FEATURES_DIR / 'pre_cologne_team_state_v1.parquet'} ({len(summary)} teams)")
    print(f"Wrote {FEATURES_DIR / 'pre_cologne_team_state_v1_full.json'}")
    print("Assertions passed: zero Cologne match_ids contributed to this snapshot.")


if __name__ == "__main__":
    main()
