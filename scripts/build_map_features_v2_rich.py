"""
Phase 6A - build data/features/map_features_v2_rich.parquet.

KNOWN-MAP prediction task: given Team A, Team B, bestOf and a user-selected
map, predict who wins that map. map_name is a LEGITIMATE predictor here (the
user explicitly selects it) - unlike the pre-veto series task.

No new temporal/streaming engine is needed: map_features_v1.parquet (Phase
5A) is already one-row-per-map with a proven leakage-safe cutoff
(`series_datetime`), and series_features_v4_roster.parquet is already
one-row-per-series computed at that exact same cutoff. Broadcasting one V4
row across all maps of a series via a many-to-one join on match_id IS the
same-series snapshot contract: it is structurally impossible for Map 1 to see
Map 2's result through this join, because V4 was frozen before any map of the
series was played.

HARD GATE: before writing anything, this script asserts (a) every map's
match_id resolves in V4 and (b) `series_datetime == V4's datetime` for EVERY
row - the formal, exhaustive proof that every joined V4 feature is safe for
every map of that series.

Contract with map_features_v1 (enforced here and re-checked in validate_phase6a.py):
  * identical row count (10,318), identical row order, identical target;
  * map_features_v1's own baked-in copy of the 17 original V1 series features
    (+ bestOf/tier) is DROPPED in favor of V4's copy of the same columns -
    verified numerically identical below, so nothing is lost.

Read-only against data/raw/ and data/interim/.
"""

import json

import numpy as np
import pandas as pd
import yaml

from _common import ROOT, INTERIM, raw_file_hashes
from player_roster_feature_engine import ROSTER_PERFORMANCE_DIFFS

FEATURES_DIR = ROOT / "data" / "features"
CONFIG_PATH = ROOT / "config" / "map_features_v2_rich.yaml"
MAP_V1_PATH = FEATURES_DIR / "map_features_v1.parquet"
SERIES_V4_PATH = FEATURES_DIR / "series_features_v4_roster.parquet"

EXPECTED_ROWS = 10318

# The 17 columns map_features_v1 already carries from Phase 3 (10 directional +
# 5 symmetric + bestOf + tier), which V4 also carries unchanged since V1. These
# are DROPPED from map_features_v1's side and taken from V4 instead.
V1_REDUNDANT_WITH_V4 = [
    "elo_diff", "overall_win_rate_diff", "win_rate_last_5_diff", "win_rate_last_10_diff",
    "format_win_rate_diff", "avg_series_margin_last_5_diff", "avg_series_margin_last_10_diff",
    "matches_last_30_days_diff", "days_since_last_match_diff", "total_matches_before_diff",
    "history_matches_min", "history_matches_sum", "both_teams_have_history",
    "both_teams_have_5_matches", "both_teams_have_10_matches",
    "bestOf", "tier",
]

MAP_V1_METADATA = ["match_id", "game_id", "series_datetime", "map_datetime", "tournament",
                    "team1", "team2", "team1_canonical", "team2_canonical", "evaluation_group",
                    "team1_map_matches_before", "team2_map_matches_before"]
MAP_V1_OWN_FEATURES = [
    # map-specific directional (9)
    "map_elo_diff", "map_smoothed_win_rate_diff", "map_win_rate_last_5_diff", "map_win_rate_last_10_diff",
    "map_normalized_margin_all_diff", "map_normalized_margin_last_5_diff", "map_normalized_margin_last_10_diff",
    "map_matches_before_diff", "days_since_map_played_diff",
    # map-specific symmetric (5)
    "map_matches_before_min", "map_matches_before_sum", "both_teams_have_map_history",
    "both_teams_have_5_map_matches", "both_teams_have_10_map_matches",
]
V4_FEATURE_COLS = [
    # V2 map-pool depth (14) + same-map matchup (6)
    "map_pool_size_diff", "map_pool_total_matches_diff", "map_pool_experienced_maps_diff",
    "map_pool_mean_elo_diff", "map_pool_best_elo_diff", "map_pool_second_best_elo_diff",
    "map_pool_third_best_elo_diff", "map_pool_worst_elo_diff", "map_pool_mean_smoothed_wr_diff",
    "map_pool_best_smoothed_wr_diff", "map_pool_second_best_smoothed_wr_diff",
    "map_pool_third_best_smoothed_wr_diff", "map_pool_worst_smoothed_wr_diff",
    "map_pool_mean_normalized_margin_diff",
    "map_matchup_mean_elo_advantage", "map_matchup_median_elo_advantage",
    "map_matchup_midrange_elo_advantage", "map_matchup_positive_advantage_balance",
    "map_matchup_mean_smoothed_wr_advantage", "map_matchup_median_smoothed_wr_advantage",
    "map_pool_size_min", "map_pool_total_matches_min", "both_teams_have_map_pool_history",
    "both_teams_have_3_recent_maps", "both_teams_have_5_experienced_maps", "union_map_count",
    "shared_recent_map_count", "shared_experienced_map_count", "map_matchup_shared_coverage",
    "map_matchup_elo_advantage_range",
    # V3 form (8 directional + 4 symmetric)
    "avg_opponent_elo_last_5_diff", "avg_opponent_elo_last_10_diff",
    "performance_residual_last_5_diff", "performance_residual_last_10_diff", "performance_residual_all_diff",
    "time_weighted_win_rate_diff", "time_weighted_performance_residual_diff", "time_weighted_series_margin_diff",
    "opponent_adjusted_history_min", "both_teams_have_5_adjusted_matches",
    "both_teams_have_10_adjusted_matches", "time_weighted_history_mass_min",
    # V4 roster (15 directional + 6 symmetric)
    "roster_mean_adr_diff", "roster_top_adr_diff", "roster_bottom_adr_diff",
    "roster_mean_kast_diff", "roster_top_kast_diff", "roster_bottom_kast_diff",
    "roster_mean_kd_balance_diff", "roster_top_kd_balance_diff", "roster_bottom_kd_balance_diff",
    "roster_mean_assists_per_round_diff",
    "recent_unique_players_10_maps_diff", "recent_unique_players_20_maps_diff",
    "core5_appearance_concentration_90d_diff", "core5_continuity_last_10_diff",
    "roster_mean_player_history_mass_diff",
    "roster_size_min", "both_teams_have_5_inferred_players", "roster_min_player_history_mass",
    "roster_core_concentration_min", "roster_core_continuity_last10_min", "roster_form_players_min",
    # inherited V1 series-level + categorical context (taken from V4, dropped from map_v1's own copy)
] + V1_REDUNDANT_WITH_V4


def main():
    hashes_before = raw_file_hashes()
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    mv1 = pd.read_parquet(MAP_V1_PATH, engine="fastparquet")
    v4 = pd.read_parquet(SERIES_V4_PATH, engine="fastparquet")
    print(f"map_features_v1: {len(mv1)} rows x {len(mv1.columns)} cols "
          f"({mv1['match_id'].nunique()} matches)")
    print(f"series_features_v4_roster: {len(v4)} rows x {len(v4.columns)} cols")
    assert len(mv1) == EXPECTED_ROWS

    # ---- 1. join totality + HARD GATE: pre-series-safety proof ----
    assert mv1["match_id"].isin(set(v4["match_id"])).all(), \
        "a map's match_id is absent from series_features_v4_roster - cannot enrich it safely"

    merged = mv1.merge(v4, on="match_id", how="left", suffixes=("", "_v4"), validate="many_to_one")
    assert len(merged) == len(mv1), "join changed row count"
    assert merged["match_id"].tolist() == mv1["match_id"].tolist(), "join changed row order"

    cutoff_equal = (merged["series_datetime"] == merged["datetime"]).all()
    if not cutoff_equal:
        bad = merged.loc[merged["series_datetime"] != merged["datetime"], ["match_id", "game_id"]]
        raise AssertionError(
            f"series_datetime != V4 datetime for {len(bad)} rows - V4 state would NOT be "
            f"strictly pre-series for these maps. STOPPING before any artifact is written:\n{bad.head(10)}")
    print("HARD GATE PASSED: series_datetime == V4's datetime for all "
          f"{len(merged)} rows - every joined V4 feature is proven pre-series-safe.")

    # ---- 2. cross-check the 17 shared V1 columns are numerically identical before dropping map_v1's copy ----
    for c in V1_REDUNDANT_WITH_V4:
        a, b = merged[c], merged[f"{c}_v4"]
        if pd.api.types.is_numeric_dtype(a):
            assert np.array_equal(a.to_numpy(dtype=float), b.to_numpy(dtype=float), equal_nan=True), \
                f"{c}: map_features_v1's own copy disagrees with V4's copy - NOT safe to drop"
        else:
            assert a.equals(b), f"{c}: map_features_v1's own copy disagrees with V4's copy"
    print(f"Cross-check PASSED: all {len(V1_REDUNDANT_WITH_V4)} columns shared by name are "
          "numerically identical between map_features_v1 and V4 - safe to take V4's copy.")

    # V4 columns carry a "_v4" suffix only where the name collided with map_features_v1;
    # the 63 genuinely-new V2/V3/V4 columns (map_pool_*, map_matchup_*, avg_opponent_*,
    # performance_residual_*, time_weighted_*, roster_*) did NOT collide, so they keep
    # their plain names in `merged` already.
    rename_map = {f"{c}_v4": c for c in V1_REDUNDANT_WITH_V4}
    for src, dst in rename_map.items():
        merged[dst] = merged[src]

    final = merged[MAP_V1_METADATA + [cfg["target"]] + MAP_V1_OWN_FEATURES
                   + V4_FEATURE_COLS + ["map_name"]]
    # reorder to exactly the config's declared column order
    declared = (MAP_V1_METADATA + [cfg["target"]] + cfg["directional_features"]
                + cfg["symmetric_features"] + cfg["categorical_context"])
    assert sorted(final.columns) == sorted(declared), \
        f"assembled columns disagree with the config whitelist: {set(final.columns) ^ set(declared)}"
    final = final[declared]

    # ---- 3. forbidden-column / leakage guard rails ----
    forbidden = {"score1_game", "score2_game", "map_id", "kills", "deaths", "assists", "adr",
                 "kast", "kddiff", "player_id", "team1_win", "team1_series_win"}
    leaked = forbidden & set(final.columns)
    assert not leaked, f"forbidden column reached the table: {leaked}"
    bad_tokens = ("player1", "player2", "player3", "player4", "player5", "_kills", "_deaths", "kddiff")
    bad = [c for c in final.columns if any(t in c.lower() for t in bad_tokens)]
    assert not bad, f"raw player identity/box-score column reached the table: {bad}"

    # ---- 4. NaN contract: NaN is permitted, but only where documented ----
    always_finite_predictive = [c for c in cfg["directional_features"] + cfg["symmetric_features"]
                                 if c not in ("days_since_map_played_diff", "days_since_last_match_diff")
                                 and c not in ROSTER_PERFORMANCE_DIFFS]
    non_finite = {}
    for c in always_finite_predictive:
        if pd.api.types.is_numeric_dtype(final[c]):
            bad_n = int((~np.isfinite(final[c].to_numpy(dtype=float))).sum())
            if bad_n:
                non_finite[c] = bad_n
    assert not non_finite, f"unexpected non-finite value(s) outside the documented NaN contract: {non_finite}"

    days_map_isna = final["days_since_map_played_diff"].isna()
    cold_map = (merged["team1_map_matches_before"] == 0) | (merged["team2_map_matches_before"] == 0)
    assert days_map_isna.equals(cold_map), \
        "days_since_map_played_diff NaN pattern does not match the documented map cold-start condition"

    roster_no_evidence = final["roster_form_players_min"] == 0
    for c in ROSTER_PERFORMANCE_DIFFS:
        assert final[c].isna().equals(roster_no_evidence), \
            f"{c}: NaN pattern does not match roster_form_players_min == 0"

    final.to_parquet(FEATURES_DIR / "map_features_v2_rich.parquet", engine="fastparquet", index=False)

    n_directional_new = len(cfg["directional_features"])
    n_symmetric_new = len(cfg["symmetric_features"])
    summary = {
        "task_id": "map_features_v2_rich",
        "prediction_task": "known_map",
        "rows": int(len(final)),
        "map_v1_rows": int(len(mv1)),
        "matches": int(final["match_id"].nunique()),
        "maps_represented": sorted(final["map_name"].unique().tolist()),
        "n_directional": n_directional_new,
        "n_symmetric": n_symmetric_new,
        "n_categorical": len(cfg["categorical_context"]),
        "n_total_predictive": n_directional_new + n_symmetric_new + len(cfg["categorical_context"]),
        "cold_start_rows_days_since_map_played": int(days_map_isna.sum()),
        "cold_start_rows_roster_form": int(roster_no_evidence.sum()),
    }
    with open(INTERIM / "map_features_v2_build_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    assert hashes_before == raw_file_hashes(), "data/raw/ was modified during the build"
    print(f"\nWrote {FEATURES_DIR / 'map_features_v2_rich.parquet'} ({len(final)} rows x {len(final.columns)} cols)")
    print(f"total predictive inputs: {summary['n_total_predictive']} "
          f"({n_directional_new} directional + {n_symmetric_new} symmetric + "
          f"{len(cfg['categorical_context'])} categorical)")
    print(f"Wrote {INTERIM / 'map_features_v2_build_summary.json'}")


ROSTER_PERFORMANCE_DIFFS = [
    "roster_mean_adr_diff", "roster_top_adr_diff", "roster_bottom_adr_diff",
    "roster_mean_kast_diff", "roster_top_kast_diff", "roster_bottom_kast_diff",
    "roster_mean_kd_balance_diff", "roster_top_kd_balance_diff", "roster_bottom_kd_balance_diff",
    "roster_mean_assists_per_round_diff",
]


if __name__ == "__main__":
    main()
