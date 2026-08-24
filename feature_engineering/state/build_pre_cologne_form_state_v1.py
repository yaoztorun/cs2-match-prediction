"""
Phase 5B.2 - frozen pre-Cologne team-form state snapshot for future inference.

Mirrors feature_engineering/state/build_pre_cologne_map_state_v1.py: replays the canonical
series stream strictly before the Cologne cutoff into a FRESH
TeamFormStateStore, independently re-verifies no Cologne information reached
the stream or any team's history, and writes both a flat scalar summary
(.parquet) and the full reloadable state (.json). Built from the canonical
stream + identity policy, NOT by replaying series_features_v3_form.parquet -
that table only contains both-eligible-pair rows and would silently drop
legitimate own-history evidence recorded against an untrusted opponent (same
reasoning as build_pre_cologne_map_state_v1.py's own docstring).

No post-Cologne deployment snapshot is built here - out of scope.

Read-only against data/raw/ and data/interim/.
"""

import pandas as pd

from _common import INTERIM, raw_file_hashes
from feature_engineering.maps.map_stream_common import cologne_cutoff
from feature_engineering.form.team_form_engine import FORM_ENGINE_VERSION, TeamFormStateStore, process_form_stream
from feature_engineering.form.team_form_stream_common import load_series_form_stream


def main():
    hashes_before = raw_file_hashes()

    cologne_dt, cologne_ids = cologne_cutoff()
    print(f"Cologne cutoff: {cologne_dt} ({len(cologne_ids)} Cologne match_ids)")

    stream, info = load_series_form_stream(evaluation_groups=("development",), max_exclusive_datetime=cologne_dt)
    assert stream["datetime"].max() < cologne_dt, "a source series is not strictly before Cologne"
    cologne_ids_str = {str(i) for i in cologne_ids}
    assert not (set(stream["match_id"].astype(str)) & cologne_ids_str), "a Cologne match_id entered the stream"
    print(f"stream: {len(stream)} rows, max datetime {stream['datetime'].max()}")

    store = TeamFormStateStore()
    process_form_stream(store, stream, emit_features=False)

    seen_source_match_ids = set()
    for st in store.teams.values():
        for h in st.history:
            assert pd.Timestamp(h.dt) < cologne_dt, f"history entry at/after cutoff: {h.canonical_match_uid}"
            seen_source_match_ids.add(h.source_match_id)
    assert not (seen_source_match_ids & cologne_ids_str), "a Cologne match reached a team's form history"

    n_trusted_entries = sum(1 for st in store.teams.values() for h in st.history if h.opponent_identity_trusted)
    n_untrusted_entries = sum(1 for st in store.teams.values() for h in st.history if not h.opponent_identity_trusted)

    meta = {
        "form_engine_version": FORM_ENGINE_VERSION,
        "snapshot_id": "pre_cologne_form_state_v1",
        "state_source": ("canonical series stream (team_form_stream_common.load_series_form_stream), "
                          "NOT series_features_v3_form.parquet"),
        "not_rebuilt_from": ["series_features_v3_form.parquet"],
        "cutoff_rule": "datetime < cologne_first_datetime (strict)",
        "cologne_first_datetime": str(cologne_dt),
        "cologne_match_ids_excluded": len(cologne_ids),
        "max_source_series_datetime": str(stream["datetime"].max()),
        "team_states": len(store.teams),
        "matches_processed": len(store.processed_match_uids),
        "history_entries_trusted_opponent": n_trusted_entries,
        "history_entries_untrusted_opponent": n_untrusted_entries,
        "post_cologne_snapshot_built": False,
        "generated_at": str(pd.Timestamp.now()),
    }

    summary_df = store.snapshot_summary_df().sort_values("canonical_team_name").reset_index(drop=True)
    summary_df.to_parquet(INTERIM / "pre_cologne_form_state_v1.parquet", engine="fastparquet", index=False)
    store.to_json(INTERIM / "pre_cologne_form_state_v1.json", meta=meta)

    assert hashes_before == raw_file_hashes(), "data/raw/ was modified during the build"
    print(f"Wrote {INTERIM / 'pre_cologne_form_state_v1.parquet'} ({len(summary_df)} teams)")
    print(f"Wrote {INTERIM / 'pre_cologne_form_state_v1.json'} "
          f"({len(store.teams)} teams, {len(store.processed_match_uids)} matches, "
          f"{n_trusted_entries} trusted / {n_untrusted_entries} untrusted history entries)")


if __name__ == "__main__":
    main()
