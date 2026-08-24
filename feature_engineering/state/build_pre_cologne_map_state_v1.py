"""
Phase 5A - freeze the map state as it stood immediately before IEM Cologne 2026.

This is the state a model would have had at the moment the Major started. It
exists so a future Cologne case study can be run WITHOUT any Cologne match ever
having touched the pipeline. No model is trained here and no Cologne row is
read for anything but deriving the cutoff.

WHY THE STATE IS REBUILT FROM THE CANONICAL MAP STREAM
-------------------------------------------------------
It would be tempting to replay `map_features_v1` rows instead. That would be
WRONG, because that table contains only maps where BOTH identities are trusted,
whereas the identity policy says an eligible team's OWN map history is real
evidence even when its opponent's identity is not trustworthy:

    own history (win/loss, margin, recency, pool membership)
        -> updates for each ELIGIBLE side, regardless of the opponent
        -> the entry is flagged opponent_identity_trusted=False
    map ELO (pair-dependent, moves BOTH ratings)
        -> updates only when BOTH identities are trusted

Replaying the training table would therefore silently erase real history. The
script measures exactly how much (`entries_from_untrusted_opponent_matches`)
so the distinction is evidenced, not merely asserted.

No post-Cologne deployment snapshot is produced (out of scope for Phase 5A).

Writes:
    data/interim/pre_cologne_map_state_v1.parquet   (flat scalar summary)
    data/interim/pre_cologne_map_state_v1.json      (full re-loadable state)
"""

import json

import pandas as pd

from _common import INTERIM, raw_file_hashes
from feature_engineering.maps.map_feature_engine import (
    MAP_ENGINE_VERSION, MAP_POOL_LOOKBACK_DAYS, MapStateStore, process_combined_stream,
)
from feature_engineering.maps.map_stream_common import load_map_stream, cologne_cutoff


def main():
    hashes_before = raw_file_hashes()

    # ---- 1. cutoff from the manifest (primary source), name only cross-checked ----
    cologne_dt, cologne_ids = cologne_cutoff()
    sb = pd.read_parquet(INTERIM / "series_base.parquet", engine="fastparquet",
                         columns=["match_id", "datetime", "tournament"])
    names = sorted(set(sb.loc[sb["match_id"].isin(cologne_ids), "tournament"]))
    print(f"Cologne cutoff (from evaluation_manifest.csv): {cologne_dt}")
    print(f"cross-check, tournament names in that group  : {names}")

    # ---- 2. canonical map stream, strictly before the cutoff ----
    stream, info = load_map_stream(evaluation_groups=("development",),
                                   max_exclusive_datetime=cologne_dt)
    print(json.dumps(info, indent=2))

    # ---- 3. replay into a FRESH store; no feature rows are emitted ----
    store = MapStateStore()
    process_combined_stream(store, stream, series_requests=None, emit_map_features=False)

    # ---- 4. independent contamination assertions ----
    assert stream["series_datetime"].max() < cologne_dt, "a source map is not strictly before Cologne"
    assert not (set(stream["match_id"]) & cologne_ids), "a Cologne match_id entered the stream"
    seen_matches, seen_games = set(), set()
    for st in store.states.values():
        for h in st.history:
            assert pd.Timestamp(h.series_dt) < cologne_dt, f"history entry at/after cutoff: {h.match_id}"
            seen_matches.add(int(h.match_id))
            seen_games.add(int(h.game_id))
    assert not (seen_matches & cologne_ids), "a Cologne match reached a team's map history"

    # ---- 5. evidence for the identity distinction described in the docstring ----
    untrusted_entries = sum(
        1 for st in store.states.values() for h in st.history if not h.opponent_identity_trusted)
    both_eligible = stream["team1_eligible"] & stream["team2_eligible"]
    maps_with_an_ineligible_side = int((~both_eligible).sum())

    summary = store.snapshot_summary_df().sort_values(
        ["canonical_team_name", "map_name"]).reset_index(drop=True)
    summary.to_parquet(INTERIM / "pre_cologne_map_state_v1.parquet",
                       engine="fastparquet", index=False)

    meta = {
        "map_engine_version": MAP_ENGINE_VERSION,
        "snapshot_id": "pre_cologne_map_state_v1",
        "state_source": "canonical development map stream + team_identity_policy.csv",
        "not_rebuilt_from": "map_features_v1.parquet (would drop untrusted-opponent history)",
        "cutoff_rule": "series_datetime < cologne_first_datetime (strict)",
        "cologne_first_datetime": str(cologne_dt),
        "cologne_match_ids_excluded": len(cologne_ids),
        "map_pool_lookback_days": MAP_POOL_LOOKBACK_DAYS,
        "source_maps_replayed": int(len(stream)),
        "source_matches_replayed": int(stream["match_id"].nunique()),
        "max_source_series_datetime": str(stream["series_datetime"].max()),
        "team_map_states": int(len(store.states)),
        "distinct_teams": int(len(store.teams())),
        "distinct_maps": int(summary["map_name"].nunique()),
        "maps_with_an_ineligible_side": maps_with_an_ineligible_side,
        "entries_from_untrusted_opponent_matches": untrusted_entries,
        "post_cologne_snapshot_built": False,
    }
    store.to_json(INTERIM / "pre_cologne_map_state_v1.json", meta=meta)

    assert hashes_before == raw_file_hashes(), "data/raw/ was modified during the build"

    print(f"\nteam-map states : {len(store.states)} ({len(store.teams())} teams, "
          f"{summary['map_name'].nunique()} maps)")
    print(f"maps replayed   : {len(stream)}")
    print(f"history entries recorded against an untrusted opponent: {untrusted_entries} "
          f"(from {maps_with_an_ineligible_side} such maps) - these would have been LOST "
          f"had the snapshot been rebuilt from map_features_v1")
    print(f"Wrote {INTERIM / 'pre_cologne_map_state_v1.parquet'} ({len(summary)} rows)")
    print(f"Wrote {INTERIM / 'pre_cologne_map_state_v1.json'}")


if __name__ == "__main__":
    main()
