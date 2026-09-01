"""
Phase 5C - frozen pre-Cologne player/roster state snapshot for future inference.

Mirrors feature_engineering/state/build_pre_cologne_map_state_v1.py and
feature_engineering/state/build_pre_cologne_form_state_v1.py: replays the canonical player
observation stream strictly before the Cologne cutoff into a FRESH
PlayerRosterStateStore, independently re-verifies that no Cologne information
reached the stream or any player/team state, and writes both a flat scalar
summary (.parquet) and the full reloadable state (.json).

Built from the canonical stream + identity policy, NOT by replaying
series_features_v4_roster.parquet - that table contains only both-eligible
series and no player-level detail at all, so replaying it would be impossible
as well as lossy.

No post-Cologne deployment snapshot is built here - out of scope.

Read-only against data/raw/ and data/interim/.
"""

import pandas as pd

from _common import INTERIM, raw_file_hashes
from feature_engineering.maps.map_stream_common import cologne_cutoff
from feature_engineering.roster.player_roster_feature_engine import (
    PLAYER_ROSTER_ENGINE_VERSION, ROSTER_LOOKBACK_DAYS, PLAYER_FORM_HALF_LIFE_DAYS,
    PlayerRosterStateStore, process_player_roster_stream,
)
from feature_engineering.roster.player_roster_stream_common import load_player_roster_stream


def main():
    hashes_before = raw_file_hashes()

    cologne_dt, cologne_ids = cologne_cutoff()
    print(f"Cologne cutoff: {cologne_dt} ({len(cologne_ids)} Cologne match_ids)")

    stream, info = load_player_roster_stream(evaluation_groups=("development",),
                                              max_exclusive_datetime=cologne_dt)
    assert stream["series_datetime"].max() < cologne_dt, \
        "a source observation is not strictly before Cologne"
    cologne_ids_str = {str(i) for i in cologne_ids}
    assert not (set(stream["match_id"].astype(str)) & cologne_ids_str), \
        "a Cologne match_id entered the player stream"
    print(f"stream: {len(stream)} player observations, max series_datetime {stream['series_datetime'].max()}")

    store = PlayerRosterStateStore()
    process_player_roster_stream(store, stream, series_requests=None, emit_features=False)

    # ---- independent contamination re-derivation from the STORE itself ----
    seen_match_ids = set()
    for st in store.players.values():
        for h in st.history:
            assert pd.Timestamp(h.series_dt) < cologne_dt, \
                f"player history entry at/after cutoff: {h.game_id}"
            seen_match_ids.add(str(h.match_id))
    assert not (seen_match_ids & cologne_ids_str), "a Cologne match reached a player's history"
    for team, st in store.teams.items():
        for a in st.appearances:
            assert pd.Timestamp(a.series_dt) < cologne_dt, \
                f"appearance at/after cutoff for {team}: {a.game_id}"

    # ---- structural invariants ----
    for pid, st in store.players.items():
        keys = [(h.game_id, pid) for h in st.history]
        assert len(keys) == len(set(keys)), f"player {pid} has duplicate (game_id, player_id) entries"
    for team, st in store.teams.items():
        keys = [(a.game_id, a.player_id) for a in st.appearances]
        assert len(keys) == len(set(keys)), f"team {team} has duplicate appearances"

    n_transfers = sum(1 for st in store.players.values()
                       if len({h.team_canonical for h in st.history}) > 1)

    meta = {
        "player_roster_engine_version": PLAYER_ROSTER_ENGINE_VERSION,
        "snapshot_id": "pre_cologne_player_roster_state_v1",
        "state_source": ("canonical player observation stream "
                          "(player_roster_stream_common.load_player_roster_stream), "
                          "NOT series_features_v4_roster.parquet"),
        "not_rebuilt_from": ["series_features_v4_roster.parquet"],
        "cutoff_rule": "authoritative series_datetime < cologne_first_datetime (strict)",
        "cologne_first_datetime": str(cologne_dt),
        "cologne_match_ids_excluded": len(cologne_ids),
        "max_source_series_datetime": str(stream["series_datetime"].max()),
        "roster_lookback_days": ROSTER_LOOKBACK_DAYS,
        "player_form_half_life_days": PLAYER_FORM_HALF_LIFE_DAYS,
        "player_states": len(store.players),
        "team_states": len(store.teams),
        "player_map_entries": len(store.player_map_keys),
        "appearance_entries": len(store.appearance_keys),
        "maps_processed": len(store.processed_map_uids),
        "players_with_multiple_teams_observed": n_transfers,
        "cologne_contamination": 0,
        "post_cologne_snapshot_built": False,
        "stream_info": info,
        "generated_at": str(pd.Timestamp.now()),
    }

    summary_df = store.snapshot_summary_df().sort_values("player_id").reset_index(drop=True)
    summary_df.to_parquet(INTERIM / "pre_cologne_player_roster_state_v1.parquet",
                          engine="fastparquet", index=False)
    store.to_json(INTERIM / "pre_cologne_player_roster_state_v1.json", meta=meta)

    assert hashes_before == raw_file_hashes(), "data/raw/ was modified during the build"
    print(f"Wrote {INTERIM / 'pre_cologne_player_roster_state_v1.parquet'} ({len(summary_df)} players)")
    print(f"Wrote {INTERIM / 'pre_cologne_player_roster_state_v1.json'} "
          f"({len(store.players)} players, {len(store.teams)} teams, "
          f"{len(store.player_map_keys)} player-map entries, {n_transfers} players seen on >1 team)")
    print("Cologne contamination: ZERO (independently re-derived from the store)")


if __name__ == "__main__":
    main()
