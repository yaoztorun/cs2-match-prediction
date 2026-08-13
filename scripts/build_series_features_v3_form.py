"""
Phase 5B.2 - build data/features/series_features_v3_form.parquet.

Extends V2 (series_features_v2_map_pool.parquet) with leakage-safe
opponent-adjusted, recency-weighted team-form features via a NEW, independent
state replay (scripts/team_form_engine.py) of the same series stream Phase 3
used - see that module's docstring for why an independent replay is
necessary (Phase 3's own frozen state doesn't record opponent ELO).

HARD GATE: before writing ANY output file, this script exhaustively verifies
(all 9,456 rows, not a sample) that the new engine's independently-computed
pre-match team1_elo - team2_elo matches V1/V2's own elo_diff column within a
strict floating-point tolerance. This proves the new engine's timestamp
batching, eligibility gating, cold start, and ELO-update ordering are
identical to Phase 3's. If parity fails, the script raises before any V3
artifact is written or finalized.

Contract with V2 (enforced here and re-checked in validate_phase5b2.py):
  * identical match_id universe (9,456), identical targets, identical ordering;
  * every V2 feature column preserved value-for-value;
  * team-form columns appended.

Read-only against data/raw/ and data/interim/.
"""

import json

import numpy as np
import pandas as pd
import yaml

from _common import INTERIM, ROOT, raw_file_hashes
from team_form_engine import (
    FORM_ENGINE_VERSION, FORM_HALF_LIFE_DAYS, TeamFormStateStore, process_form_stream,
    FORM_DIRECTIONAL_FEATURES, FORM_SYMMETRIC_FEATURES,
)
from team_form_stream_common import load_series_form_stream

FEATURES_DIR = ROOT / "data" / "features"
CONFIG_PATH = ROOT / "config" / "series_features_v3_form.yaml"

V2_FEATURES = [
    # directional (30)
    "map_pool_size_diff", "map_pool_total_matches_diff", "map_pool_experienced_maps_diff",
    "map_pool_mean_elo_diff", "map_pool_best_elo_diff", "map_pool_second_best_elo_diff",
    "map_pool_third_best_elo_diff", "map_pool_worst_elo_diff", "map_pool_mean_smoothed_wr_diff",
    "map_pool_best_smoothed_wr_diff", "map_pool_second_best_smoothed_wr_diff",
    "map_pool_third_best_smoothed_wr_diff", "map_pool_worst_smoothed_wr_diff",
    "map_pool_mean_normalized_margin_diff",
    "map_matchup_mean_elo_advantage", "map_matchup_median_elo_advantage",
    "map_matchup_midrange_elo_advantage", "map_matchup_positive_advantage_balance",
    "map_matchup_mean_smoothed_wr_advantage", "map_matchup_median_smoothed_wr_advantage",
    "elo_diff", "overall_win_rate_diff", "win_rate_last_5_diff", "win_rate_last_10_diff",
    "format_win_rate_diff", "avg_series_margin_last_5_diff", "avg_series_margin_last_10_diff",
    "matches_last_30_days_diff", "days_since_last_match_diff", "total_matches_before_diff",
    # symmetric (15)
    "map_pool_size_min", "map_pool_total_matches_min", "both_teams_have_map_pool_history",
    "both_teams_have_3_recent_maps", "both_teams_have_5_experienced_maps", "union_map_count",
    "shared_recent_map_count", "shared_experienced_map_count", "map_matchup_shared_coverage",
    "map_matchup_elo_advantage_range",
    "history_matches_min", "history_matches_sum", "both_teams_have_history",
    "both_teams_have_5_matches", "both_teams_have_10_matches",
    # categorical
    "bestOf", "tier",
]
V2_METADATA = ["match_id", "datetime", "tournament", "team1", "team2",
               "team1_canonical", "team2_canonical", "source", "canonical_match_uid"]

ELO_PARITY_TOLERANCE = 1e-9


def main():
    hashes_before = raw_file_hashes()
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    v2 = pd.read_parquet(FEATURES_DIR / "series_features_v2_map_pool.parquet", engine="fastparquet")
    print(f"series_features_v2_map_pool: {len(v2)} rows")

    # ---- 1. rebuild the exact same series stream Phase 3 used ----
    stream, info = load_series_form_stream(evaluation_groups=("development",))
    print(f"series stream: {len(stream)} rows ({info})")

    # ---- 2. run the new, independent form engine ----
    store = TeamFormStateStore()
    processed_rows, excluded_rows = process_form_stream(store, stream, emit_features=True)
    print(f"form engine processed (eligible-pair): {len(processed_rows)}, excluded: {len(excluded_rows)}")
    assert len(processed_rows) == len(v2) == 9456, \
        f"eligible-pair universe diverged from V2: engine={len(processed_rows)} v2={len(v2)}"

    form_df = pd.DataFrame(processed_rows)
    form_cols = FORM_DIRECTIONAL_FEATURES + FORM_SYMMETRIC_FEATURES
    assert set(form_cols) == set(cfg["directional_features"] + cfg["symmetric_features"]) - set(V2_FEATURES), \
        "engine form columns and the config whitelist disagree"

    # ---- 3. HARD GATE: exhaustive Phase-3 ELO parity, ALL 9,456 rows ----
    # (per Phase 5B.2 correction #2 - not a sample, and no artifact may be
    # written or finalized until this passes.)
    form_df["form_elo_diff"] = form_df["team1_elo_before"] - form_df["team2_elo_before"]
    parity = v2[["match_id", "elo_diff"]].merge(
        form_df[["match_id", "form_elo_diff"]], on="match_id", how="inner", validate="one_to_one")
    assert len(parity) == len(v2), "parity check lost or duplicated rows during merge"
    parity["abs_diff"] = (parity["elo_diff"] - parity["form_elo_diff"]).abs()
    max_diff = float(parity["abs_diff"].max())
    n_bad = int((parity["abs_diff"] >= ELO_PARITY_TOLERANCE).sum())
    if n_bad > 0:
        worst = parity.sort_values("abs_diff", ascending=False).head(10)
        print("ELO PARITY CHECK FAILED - divergence between the independent form engine and Phase 3's elo_diff:")
        print(worst.to_string(index=False))
        raise AssertionError(
            f"{n_bad}/{len(parity)} rows diverge from Phase 3's elo_diff by >= {ELO_PARITY_TOLERANCE} "
            f"(max abs diff {max_diff:.3e}). STOPPING before any V3 artifact is written or finalized - "
            "the divergence must be understood and fixed first.")
    print(f"ELO parity check PASSED: all {len(parity)}/{len(parity)} rows match Phase 3's elo_diff "
          f"within {ELO_PARITY_TOLERANCE:.0e} (max abs diff observed: {max_diff:.3e})")

    # ---- 4. append to V2, preserving V2 exactly ----
    merged = v2.merge(form_df[["match_id"] + form_cols], on="match_id", how="left", validate="one_to_one")
    assert len(merged) == len(v2)
    assert merged["match_id"].tolist() == v2["match_id"].tolist(), "row ordering diverged from V2"
    for c in V2_FEATURES + V2_METADATA + ["team1_series_win"]:
        a, b = v2[c], merged[c]
        if pd.api.types.is_float_dtype(a):
            assert np.allclose(a.to_numpy(dtype=float), b.to_numpy(dtype=float), rtol=0, atol=0, equal_nan=True), c
        else:
            assert a.equals(b), c
    missing_form = int(merged[form_cols[0]].isna().sum())
    assert missing_form == 0, f"{missing_form} series received no form features"

    final = merged[V2_METADATA + ["team1_series_win"] + V2_FEATURES + form_cols]

    # ---- 5. guard rails: nothing about the target series may be present ----
    forbidden = {"map_name", "map_id", "game_id", "score1_game", "score2_game",
                 "team1_map_win", "maps_played", "map_list", "veto",
                 "score1", "score2", "score1_match", "score2_match", "team1_win"}
    leaked = forbidden & set(final.columns)
    assert not leaked, f"target-series-leaking column reached the table: {leaked}"
    player_tokens = ("player", "kill", "death", "assist", "adr", "kast",
                     "headshot", "flash", "clutch", "damage")
    bad = [c for c in final.columns if any(t in c.lower() for t in player_tokens)]
    assert not bad, f"player-level column reached the table: {bad}"
    numeric = final[form_cols].select_dtypes(include=[np.number])
    assert np.isfinite(numeric.to_numpy()).all(), "non-finite value in a form feature"

    final.to_parquet(FEATURES_DIR / "series_features_v3_form.parquet", engine="fastparquet", index=False)

    # ---- 6. audit-only parquet (raw team1_*/team2_* form values) ----
    audit_cols = ["match_id", "datetime", "team1_canonical", "team2_canonical"] + \
        [c for c in form_df.columns if c.startswith("team1_") or c.startswith("team2_")]
    audit_cols = list(dict.fromkeys(audit_cols))
    audit_df = form_df[audit_cols].sort_values(["datetime", "match_id"]).reset_index(drop=True)
    audit_df.to_parquet(FEATURES_DIR / "series_team_form_states_v1.parquet", engine="fastparquet", index=False)

    cold = int((final["opponent_adjusted_history_min"] == 0).sum())
    summary = {
        "form_engine_version": FORM_ENGINE_VERSION,
        "task_id": "series_features_v3_form",
        "prediction_task": "pre_veto_series",
        "form_half_life_days": FORM_HALF_LIFE_DAYS,
        "rows": int(len(final)),
        "v2_rows": int(len(v2)),
        "elo_parity_check": {"tolerance": ELO_PARITY_TOLERANCE, "max_abs_diff": max_diff,
                              "n_rows_checked": int(len(parity)), "n_rows_failed": n_bad},
        "rows_with_zero_trusted_opponent_history": cold,
        "n_form_directional": len(FORM_DIRECTIONAL_FEATURES),
        "n_form_symmetric": len(FORM_SYMMETRIC_FEATURES),
        "n_total_features": len(V2_FEATURES) + len(form_cols),
        "stream_info": info,
    }
    with open(INTERIM / "series_features_v3_build_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    assert hashes_before == raw_file_hashes(), "data/raw/ was modified during the build"
    print(f"\nWrote {FEATURES_DIR / 'series_features_v3_form.parquet'} ({len(final)} rows x {len(final.columns)} cols)")
    print(f"Wrote {FEATURES_DIR / 'series_team_form_states_v1.parquet'} (AUDIT ONLY, {len(audit_df)} rows)")
    print(f"rows with zero trusted-opponent-adjusted history (cold start): {cold}")
    print(f"Wrote {INTERIM / 'series_features_v3_build_summary.json'}")


if __name__ == "__main__":
    main()
