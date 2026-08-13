"""
Phase 5A - build data/features/map_features_v1.parquet.

Prediction task: KNOWN MAP. The map identity is given before the map is played,
so `map_name` and every map-specific historical statistic are legitimate inputs
here - the exact opposite of the pre-veto series task.

One row per historical development map. Predictors are:
  * 9 map-specific directional + 5 symmetric features from the map engine;
  * the 17 Phase 3 series features joined by match_id (valid: they are strictly
    prior to the same series cutoff);
  * map_name / bestOf / tier as task context.

Read-only against data/raw/ and data/interim/.
"""

import json

import pandas as pd
import yaml

from _common import INTERIM, ROOT, raw_file_hashes
from map_feature_engine import MAP_ENGINE_VERSION, MapStateStore, process_combined_stream
from map_stream_common import load_map_stream, cologne_cutoff

FEATURES_DIR = ROOT / "data" / "features"
FEATURES_DIR.mkdir(exist_ok=True, parents=True)
CONFIG_PATH = ROOT / "config" / "map_features_v1.yaml"

SERIES_JOIN_FEATURES = [
    "elo_diff", "overall_win_rate_diff", "win_rate_last_5_diff", "win_rate_last_10_diff",
    "format_win_rate_diff", "avg_series_margin_last_5_diff", "avg_series_margin_last_10_diff",
    "matches_last_30_days_diff", "days_since_last_match_diff", "total_matches_before_diff",
    "history_matches_min", "history_matches_sum", "both_teams_have_history",
    "both_teams_have_5_matches", "both_teams_have_10_matches",
]


def main():
    hashes_before = raw_file_hashes()
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # ---- 1. canonical development map stream (Cologne + post-Cologne excluded) ----
    stream, info = load_map_stream(evaluation_groups=("development",))
    print(json.dumps(info, indent=2))

    cologne_dt, cologne_ids = cologne_cutoff()
    assert not (set(stream["match_id"]) & cologne_ids), "Cologne match leaked into the development map stream"

    # ---- 2. run the series-atomic engine ----
    store = MapStateStore()
    emitted, _ = process_combined_stream(store, stream, series_requests=None)
    emitted_df = pd.DataFrame(emitted)
    print(f"map rows in stream          : {len(stream)}")
    print(f"feature rows emitted        : {len(emitted_df)} (both identities trusted)")
    print(f"skipped, identity-ineligible: {len(stream) - len(emitted_df)}")

    # ---- 3. join the Phase 3 series features ----
    series = pd.read_parquet(FEATURES_DIR / "series_features_v1.parquet", engine="fastparquet")
    before_join = len(emitted_df)
    out = emitted_df.merge(series[["match_id"] + SERIES_JOIN_FEATURES], on="match_id", how="inner")
    dropped = before_join - len(out)
    print(f"dropped, match absent from series_features_v1: {dropped}")

    # ---- 4. assemble exactly the whitelisted columns ----
    meta_cols = cfg["metadata_columns"]
    feature_cols = cfg["directional_features"] + cfg["symmetric_features"] + cfg["categorical_context"]
    target = cfg["target"] if isinstance(cfg["target"], str) else list(cfg["target"])[0]

    missing = [c for c in meta_cols + feature_cols + [target] if c not in out.columns]
    if missing:
        raise ValueError(f"columns promised by the config are absent from the built frame: {missing}")

    final = out[meta_cols + [target] + feature_cols].sort_values(
        ["series_datetime", "match_id", "game_id"]).reset_index(drop=True)

    # ---- 5. guard rails before writing ----
    forbidden = {"score1_game", "score2_game", "map_id", "team1_map_elo_raw"}
    leaked = forbidden & set(final.columns)
    assert not leaked, f"forbidden column reached the modelling table: {leaked}"
    player_tokens = ("player", "kill", "death", "assist", "adr", "kast", "rating",
                     "headshot", "flash", "clutch", "damage")
    bad = [c for c in final.columns if any(t in c.lower() for t in player_tokens)]
    assert not bad, f"player-level column reached the modelling table: {bad}"
    assert final["evaluation_group"].unique().tolist() == ["development"]
    # NOTE: `development` means "not Cologne and not later than Cologne's LAST
    # match" (Phase 2 definition), so it legitimately contains non-Cologne events
    # running CONCURRENTLY with the Major - its max datetime is after Cologne's
    # first. What must hold is that no Cologne match_id and no Cologne map ever
    # enters the stream, which is asserted above and re-checked independently in
    # validate_phase5a.py. The strict `< cologne_first_datetime` rule applies to
    # the frozen pre-Cologne snapshot, not to this table. Identical to Phase 3.
    assert not (set(final["match_id"]) & cologne_ids)
    assert not final.duplicated(subset=["match_id", "game_id"]).any()

    final.to_parquet(FEATURES_DIR / "map_features_v1.parquet", engine="fastparquet", index=False)

    summary = {
        "map_engine_version": MAP_ENGINE_VERSION,
        "task_id": "map_features_v1",
        "prediction_task": "known_map",
        "rows": int(len(final)),
        "distinct_matches": int(final["match_id"].nunique()),
        "distinct_maps": int(final["map_name"].nunique()),
        "stream_info": info,
        "rows_skipped_identity_ineligible": int(len(stream) - before_join),
        "rows_dropped_no_series_features": int(dropped),
        "cologne_first_datetime": str(cologne_dt),
        "target_positive_rate": float(final[target].mean()),
        "n_directional_features": len(cfg["directional_features"]),
        "n_symmetric_features": len(cfg["symmetric_features"]),
    }
    with open(INTERIM / "map_features_v1_build_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    assert hashes_before == raw_file_hashes(), "data/raw/ was modified during the build - this must never happen"
    print(f"\nWrote {FEATURES_DIR / 'map_features_v1.parquet'} ({len(final)} rows x {len(final.columns)} cols)")
    print(f"Wrote {INTERIM / 'map_features_v1_build_summary.json'}")


if __name__ == "__main__":
    main()
