"""
Phase 6C - frozen pre-Cologne modern-map state snapshot for future inference.

Mirrors scripts/build_pre_cologne_map_state_v1.py /
build_pre_cologne_form_state_v1.py / build_pre_cologne_player_roster_state_v1.py:
replays the canonical modern-map streams strictly before the Cologne cutoff
into a FRESH ModernMapStateStore, independently re-verifies that no Cologne
information reached the stream or any ledger, and writes the full reloadable
state (.json).

Only the NEW state type (ModernMapStateStore) is (re)built here.
map_state/player_roster_state at inference time reuse the EXISTING, already
frozen pre_cologne_map_state_v1.json / pre_cologne_player_roster_state_v1.json
snapshots (Phase 5A/5C) unchanged - both were already built with the identical
strict-cutoff discipline, and ModernMapStateStore's own ledgers are, by
construction (see modern_map_feature_engine.py's module docstring),
order-independent, so building it from the SAME cutoff-restricted stream as
those two produces a jointly-consistent trio.

Read-only against data/raw/ and data/interim/.
"""

import pandas as pd

from _common import INTERIM, raw_file_hashes
from map_stream_common import cologne_cutoff
from modern_map_feature_engine import (
    MODERN_MAP_ENGINE_VERSION, MAP_FORM_HALF_LIFE_DAYS,
    ModernMapStateStore, apply_selected_map_team_result, apply_selected_map_player_observation,
)
from modern_map_stream_common import load_modern_map_streams


def main():
    hashes_before = raw_file_hashes()

    cologne_dt, cologne_ids = cologne_cutoff()
    print(f"Cologne cutoff: {cologne_dt} ({len(cologne_ids)} Cologne match_ids)")

    map_rows, player_rows, info = load_modern_map_streams(
        evaluation_groups=("development",), max_exclusive_datetime=cologne_dt)
    assert map_rows["series_datetime"].max() < cologne_dt, "a map row is not strictly before Cologne"
    assert player_rows["series_datetime"].max() < cologne_dt, "a player row is not strictly before Cologne"
    cologne_ids_str = {str(i) for i in cologne_ids}
    assert not (set(map_rows["match_id"].astype(str)) & cologne_ids_str), \
        "a Cologne match_id entered the modern map stream"
    assert not (set(player_rows["match_id"].astype(str)) & cologne_ids_str), \
        "a Cologne match_id entered the modern player stream"
    print(f"map stream: {len(map_rows)} rows | player stream: {len(player_rows)} rows, "
          f"max series_datetime {max(map_rows['series_datetime'].max(), player_rows['series_datetime'].max())}")

    store = ModernMapStateStore()
    for _, r in map_rows.iterrows():
        apply_selected_map_team_result(store, r)
    for _, r in player_rows.iterrows():
        apply_selected_map_player_observation(store, r)

    # ---- independent contamination re-derivation from the STORE itself ----
    for (_team, _map), hist in store.team_map.items():
        for h in hist:
            assert pd.Timestamp(h.series_dt) < cologne_dt, f"team_map entry at/after cutoff: {h.game_id}"
            assert str(h.match_id) not in cologne_ids_str, f"Cologne match reached team_map: {h.match_id}"
    for (_pid, _map), hist in store.player_map.items():
        for h in hist:
            assert pd.Timestamp(h.series_dt) < cologne_dt, f"player_map entry at/after cutoff: {h.game_id}"
    for (_team, _map), apps in store.team_map_roster.items():
        for a in apps:
            assert pd.Timestamp(a.series_dt) < cologne_dt, f"team_map_roster entry at/after cutoff: {a.game_id}"

    # ---- structural invariants ----
    for key, hist in store.team_map.items():
        keys = [(h.game_id,) for h in hist]
        assert len(keys) == len(set(keys)), f"team_map {key} has duplicate game_id entries"
    for key, hist in store.player_map.items():
        keys = [(h.game_id,) for h in hist]
        assert len(keys) == len(set(keys)), f"player_map {key} has duplicate game_id entries"

    meta = {
        "modern_map_engine_version": MODERN_MAP_ENGINE_VERSION,
        "snapshot_id": "pre_cologne_modern_map_state_v1",
        "state_source": "modern_map_stream_common.load_modern_map_streams (canonical stream, cutoff-restricted)",
        "companion_snapshots_reused_unchanged": [
            "data/interim/pre_cologne_map_state_v1.json",
            "data/interim/pre_cologne_player_roster_state_v1.json",
        ],
        "cutoff_rule": "authoritative series_datetime < cologne_first_datetime (strict)",
        "cologne_first_datetime": str(cologne_dt),
        "cologne_match_ids_excluded": len(cologne_ids),
        "max_source_series_datetime": str(max(map_rows["series_datetime"].max(),
                                                player_rows["series_datetime"].max())),
        "map_form_half_life_days": MAP_FORM_HALF_LIFE_DAYS,
        "team_map_ledgers": len(store.team_map),
        "player_map_ledgers": len(store.player_map),
        "team_map_roster_ledgers": len(store.team_map_roster),
        "maps_processed": len(store.processed_map_uids),
        "cologne_contamination": 0,
        "post_cologne_snapshot_built": False,
        "stream_info": info,
        "generated_at": str(pd.Timestamp.now()),
    }
    store.to_json(INTERIM / "pre_cologne_modern_map_state_v1.json", meta=meta)

    assert hashes_before == raw_file_hashes(), "data/raw/ was modified during the build"
    print(f"Wrote {INTERIM / 'pre_cologne_modern_map_state_v1.json'} "
          f"({len(store.team_map)} team-map, {len(store.player_map)} player-map, "
          f"{len(store.team_map_roster)} team-map-roster ledgers)")
    print("Cologne contamination: ZERO (independently re-derived from the store)")


if __name__ == "__main__":
    main()
