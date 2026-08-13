"""
Phase 5A - build data/features/series_features_v2_map_pool.parquet.

Prediction task: PRE-VETO SERIES. The maps that will be played are UNKNOWN at
prediction time, so nothing about the target match's maps may enter. Map
knowledge appears only as summaries of the two teams' OWN PRIOR map results.

Contract with V1 (enforced here and re-checked in validate_phase5a.py):
  * identical match_id universe (9,456), identical targets, identical ordering;
  * every V1 feature column preserved value-for-value;
  * map-pool columns appended.

Every one of the 9,456 series gets pool features - including the ~4,504 that
have no map rows of their own, because pool features come from the teams'
history, not from the target match. That is why the build drives the engine
with an explicit series-request frame rather than off the map stream.

Read-only against data/raw/ and data/interim/.
"""

import json

import numpy as np
import pandas as pd
import yaml

from _common import INTERIM, ROOT, raw_file_hashes
from map_feature_engine import (
    MAP_ENGINE_VERSION, MAP_POOL_LOOKBACK_DAYS, MapStateStore, process_combined_stream,
    SERIES_MAP_POOL_DIRECTIONAL, SERIES_MAP_POOL_SYMMETRIC,
)
from map_stream_common import canonical_eligibility_map, load_map_stream, cologne_cutoff

FEATURES_DIR = ROOT / "data" / "features"
CONFIG_PATH = ROOT / "config" / "series_features_v2_map_pool.yaml"

V1_FEATURES = [
    "elo_diff", "overall_win_rate_diff", "win_rate_last_5_diff", "win_rate_last_10_diff",
    "format_win_rate_diff", "avg_series_margin_last_5_diff", "avg_series_margin_last_10_diff",
    "matches_last_30_days_diff", "days_since_last_match_diff", "total_matches_before_diff",
    "history_matches_min", "history_matches_sum", "both_teams_have_history",
    "both_teams_have_5_matches", "both_teams_have_10_matches", "bestOf", "tier",
]
V1_METADATA = ["match_id", "datetime", "tournament", "team1", "team2",
               "team1_canonical", "team2_canonical", "source", "canonical_match_uid"]


def main():
    hashes_before = raw_file_hashes()
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    v1 = pd.read_parquet(FEATURES_DIR / "series_features_v1.parquet", engine="fastparquet")
    print(f"series_features_v1: {len(v1)} rows")

    # ---- 1. the same development map stream that map_features_v1 uses ----
    stream, info = load_map_stream(evaluation_groups=("development",))
    cologne_dt, cologne_ids = cologne_cutoff()
    assert not (set(stream["match_id"]) & cologne_ids)

    # ---- 2. one PRE-VETO request per V1 series, whether or not it has map rows ----
    canon_elig = canonical_eligibility_map()
    reqs = pd.DataFrame({
        "match_id": v1["match_id"],
        "series_datetime": v1["datetime"],
        "team1_canonical": v1["team1_canonical"],
        "team2_canonical": v1["team2_canonical"],
        "bestOf": v1["bestOf"],
        "tier": v1["tier"],
        "team1_eligible": v1["team1_canonical"].map(canon_elig),
        "team2_eligible": v1["team2_canonical"].map(canon_elig),
    })
    # V1 already excluded identity-ineligible pairs, so every request must be eligible;
    # if that ever stops holding the merge below would silently lose rows.
    assert reqs["team1_eligible"].all() and reqs["team2_eligible"].all(), \
        "series_features_v1 contains an identity-ineligible pair - V1/V2 coverage would diverge"

    matches_with_own_maps = int(v1["match_id"].isin(set(stream["match_id"])).sum())
    print(f"series requests: {len(reqs)} ({matches_with_own_maps} have own map rows, "
          f"{len(reqs) - matches_with_own_maps} do not)")

    # ---- 3. run the engine (map feature emission off - this task does not need it) ----
    store = MapStateStore()
    _, series_rows = process_combined_stream(store, stream, reqs, emit_map_features=False)
    pool = pd.DataFrame(series_rows)
    print(f"pool feature rows emitted: {len(pool)}")

    pool_cols = SERIES_MAP_POOL_DIRECTIONAL + SERIES_MAP_POOL_SYMMETRIC
    assert set(pool_cols) == set(cfg["directional_features"] + cfg["symmetric_features"]) - set(V1_FEATURES), \
        "engine pool columns and the config whitelist disagree"

    # ---- 4. append to V1, preserving V1 exactly ----
    merged = v1.merge(pool[["match_id"] + pool_cols], on="match_id", how="left", validate="one_to_one")
    assert len(merged) == len(v1)
    assert merged["match_id"].tolist() == v1["match_id"].tolist(), "row ordering diverged from V1"
    for c in V1_FEATURES + V1_METADATA + ["team1_series_win"]:
        a, b = v1[c], merged[c]
        if pd.api.types.is_float_dtype(a):
            assert np.allclose(a.to_numpy(), b.to_numpy(), rtol=0, atol=0, equal_nan=True), c
        else:
            assert a.equals(b), c
    missing_pool = int(merged[pool_cols[0]].isna().sum())
    assert missing_pool == 0, f"{missing_pool} series received no pool features"

    final = merged[V1_METADATA + ["team1_series_win"] + V1_FEATURES + pool_cols]

    # ---- 5. guard rails: nothing about the target match's maps may be present ----
    forbidden = {"map_name", "map_id", "game_id", "score1_game", "score2_game",
                 "team1_map_win", "maps_played", "map_list", "veto"}
    leaked = forbidden & set(final.columns)
    assert not leaked, f"veto-leaking column reached the pre-veto table: {leaked}"
    player_tokens = ("player", "kill", "death", "assist", "adr", "kast",
                     "headshot", "flash", "clutch", "damage")
    bad = [c for c in final.columns if any(t in c.lower() for t in player_tokens)]
    assert not bad, f"player-level column reached the pre-veto table: {bad}"
    numeric = final[pool_cols].select_dtypes(include=[np.number])
    assert np.isfinite(numeric.to_numpy()).all(), "non-finite value in a pool feature"

    final.to_parquet(FEATURES_DIR / "series_features_v2_map_pool.parquet",
                     engine="fastparquet", index=False)

    cold = int((final["union_map_count"] == 0).sum())
    summary = {
        "map_engine_version": MAP_ENGINE_VERSION,
        "task_id": "series_features_v2_map_pool",
        "prediction_task": "pre_veto_series",
        "map_pool_lookback_days": MAP_POOL_LOOKBACK_DAYS,
        "rows": int(len(final)),
        "v1_rows": int(len(v1)),
        "matches_with_own_map_rows": matches_with_own_maps,
        "matches_without_own_map_rows": int(len(reqs) - matches_with_own_maps),
        "rows_with_empty_union_pool": cold,
        "n_pool_directional": len(SERIES_MAP_POOL_DIRECTIONAL),
        "n_pool_symmetric": len(SERIES_MAP_POOL_SYMMETRIC),
        "n_total_features": len(V1_FEATURES) + len(pool_cols),
        "stream_info": info,
        "cologne_first_datetime": str(cologne_dt),
    }
    with open(INTERIM / "series_features_v2_build_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    assert hashes_before == raw_file_hashes(), "data/raw/ was modified during the build"
    print(f"\nWrote {FEATURES_DIR / 'series_features_v2_map_pool.parquet'} "
          f"({len(final)} rows x {len(final.columns)} cols)")
    print(f"rows with a completely empty recent map pool (cold start): {cold}")
    print(f"Wrote {INTERIM / 'series_features_v2_build_summary.json'}")


if __name__ == "__main__":
    main()
