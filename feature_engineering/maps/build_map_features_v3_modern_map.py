"""
Phase 6C - build data/features/map_features_v3_modern_map.parquet.

Extends map_features_v2_rich.parquet (Phase 6A, frozen, untouched) with 25 new
selected-map features (18 directional + 7 symmetric/confidence) from the new,
independent feature_engineering/maps/modern_map_feature_engine.py. Every V2-rich column is
preserved value-for-value; the same 10,318 rows, same order, same target.

Runs, in order, before writing anything:
  1. the map-order audit (brief section 19) - real evidence, real conclusion,
     NO map_slot is ever added regardless of what the audit finds (per the
     Phase 6C approval corrections, the audit's own findings independently
     support this: no per-map timestamp exists in this dataset at all);
  2. the exhaustive (10,318-row, not a sample) pre-series-ELO join-parity HARD
     GATE required by the Phase 6C approval corrections - team1_pre_series_elo
     - team2_pre_series_elo must equal map_features_v2_rich's own inherited
     `elo_diff` within floating-point tolerance for every row, or the script
     stops before building or finalizing anything;
  3. the combined feature build (modern_map_feature_engine.process_modern_map_stream
     for the order-dependent map_elo/pool pieces + the two order-independent
     ledgers, built once, upfront, over the FULL development history).

Read-only against data/raw/ and data/interim/; never modifies map_features_v2_rich.parquet.
"""

import json

import numpy as np
import pandas as pd
import yaml

from _common import INTERIM, ROOT, raw_file_hashes
from feature_engineering.maps.map_feature_engine import MapStateStore
from feature_engineering.roster.player_roster_feature_engine import PlayerRosterStateStore, apply_player_observation
from feature_engineering.maps.modern_map_feature_engine import (
    MODERN_MAP_ENGINE_VERSION, MAP_FORM_HALF_LIFE_DAYS,
    MODERN_MAP_DIRECTIONAL_FEATURES, MODERN_MAP_SYMMETRIC_FEATURES, MODERN_MAP_NAN_CAPABLE_FEATURES,
    ModernMapStateStore, apply_selected_map_team_result, apply_selected_map_player_observation,
    process_modern_map_stream,
)
from feature_engineering.maps.modern_map_stream_common import load_modern_map_streams

FEATURES_DIR = ROOT / "data" / "features"
CONFIG_PATH = ROOT / "config" / "features" / "map_features_v3_modern_map.yaml"
V2_PATH = FEATURES_DIR / "map_features_v2_rich.parquet"

ELO_PARITY_TOLERANCE = 1e-6

FORBIDDEN_TOKENS = ("score1_game", "score2_game", "map_id", "kddiff", "player1", "player2",
                     "player3", "player4", "player5", "_kills", "_deaths", "lineup", "veto")


# ---------------------------------------------------------------------------
# 1. Map-order audit (brief section 19) - executed for real, never assumed.
# ---------------------------------------------------------------------------

def run_map_order_audit():
    mb = pd.read_parquet(INTERIM / "map_base.parquet", engine="fastparquet",
                          columns=["match_id", "game_id", "datetime", "team1_map_win"])
    multi = mb.groupby("match_id").filter(lambda d: len(d) > 1)

    n_distinct_dt = multi.groupby("match_id")["datetime"].nunique()
    all_share_one_datetime = bool((n_distinct_dt == 1).all())

    def is_contiguous(d):
        g = sorted(d["game_id"].tolist())
        return all(g[i + 1] - g[i] == 1 for i in range(len(g) - 1))

    contiguous_frac = float(multi.groupby("match_id").apply(is_contiguous).mean())

    def last_game_team1_win(d):
        return d.sort_values("game_id").iloc[-1]["team1_map_win"]

    last_game_win_rate = float(multi.groupby("match_id").apply(last_game_team1_win).mean())
    overall_win_rate = float(mb["team1_map_win"].mean())

    reliable = False   # per the Phase 6C approval corrections: game_id monotonicity/contiguity is NOT
                        # proof of true chronological order, and outcome correlation is NOT used as evidence
                        # either way (series length is itself a function of the result). The single decisive
                        # finding is the absence of any independent per-map timestamp.
    conclusion = "Map order cannot be independently verified from the current dataset."

    return {
        "n_multi_map_matches": int(multi["match_id"].nunique()),
        "all_maps_of_a_series_share_one_raw_datetime": all_share_one_datetime,
        "pct_matches_with_contiguous_game_id": contiguous_frac,
        "team1_win_rate_on_last_game_id_of_series": last_game_win_rate,
        "overall_team1_map_win_rate": overall_win_rate,
        "note_on_win_rate_comparison": (
            "reported descriptively only - NOT used as evidence for or against reliability, "
            "since series length is itself a function of the result"),
        "map_slot_added": reliable,
        "conclusion": conclusion,
    }


def main():
    hashes_before = raw_file_hashes()
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert "map_slot" not in cfg["categorical_context"], "config must not declare map_slot"

    audit = run_map_order_audit()
    print("Map-order audit:", json.dumps(audit, indent=2))
    assert audit["map_slot_added"] is False, "map_slot must not be added in Phase 6C"

    v2 = pd.read_parquet(V2_PATH, engine="fastparquet")
    print(f"map_features_v2_rich: {len(v2)} rows x {len(v2.columns)} cols")

    # ---- 2. load the modern streams (map rows carry team1/2_pre_series_elo) ----
    map_rows, player_rows, stream_info = load_modern_map_streams(evaluation_groups=("development",))
    print(f"modern map stream: {len(map_rows)} rows | player stream: {len(player_rows)} rows")

    # ---- HARD GATE: exhaustive pre-series-ELO join parity (all 10,318 v2 rows) ----
    parity = v2[["match_id", "game_id", "elo_diff"]].merge(
        map_rows[["match_id", "game_id", "team1_pre_series_elo", "team2_pre_series_elo", "pre_series_elo_known"]],
        on=["match_id", "game_id"], how="left", validate="one_to_one")
    n_unmatched = int(parity["team1_pre_series_elo"].isna().sum())
    if n_unmatched:
        raise AssertionError(
            f"{n_unmatched}/{len(v2)} map_features_v2_rich rows found no modern-stream match - "
            "STOPPING before building or finalizing anything.")
    n_unknown_elo = int((~parity["pre_series_elo_known"]).sum())
    if n_unknown_elo:
        raise AssertionError(
            f"{n_unknown_elo}/{len(v2)} map_features_v2_rich rows have no pre-series ELO evidence - "
            "STOPPING before building or finalizing anything (every V2-rich row is expected to resolve).")
    parity["form_elo_diff"] = parity["team1_pre_series_elo"] - parity["team2_pre_series_elo"]
    parity["abs_diff"] = (parity["elo_diff"] - parity["form_elo_diff"]).abs()
    max_diff = float(parity["abs_diff"].max())
    n_bad = int((parity["abs_diff"] >= ELO_PARITY_TOLERANCE).sum())
    if n_bad > 0:
        worst = parity.sort_values("abs_diff", ascending=False).head(10)
        print("ELO PARITY CHECK FAILED:")
        print(worst.to_string(index=False))
        raise AssertionError(
            f"{n_bad}/{len(parity)} rows diverge from map_features_v2_rich's own elo_diff by >= "
            f"{ELO_PARITY_TOLERANCE} (max abs diff {max_diff:.3e}). STOPPING before building/finalizing "
            "anything - the divergence must be understood and fixed first.")
    print(f"ELO parity check PASSED: all {len(parity)}/{len(parity)} rows match map_features_v2_rich's own "
          f"elo_diff within {ELO_PARITY_TOLERANCE:.0e} (max abs diff observed: {max_diff:.3e})")

    # ---- 3. build the two order-independent ledgers upfront (any order) ----
    player_roster_state = PlayerRosterStateStore()
    for _, r in player_rows.iterrows():
        apply_player_observation(player_roster_state, r)
    print(f"player_roster_state: {len(player_roster_state.players)} players, "
          f"{len(player_roster_state.teams)} teams")

    modern_state = ModernMapStateStore()
    for _, r in map_rows.iterrows():
        apply_selected_map_team_result(modern_state, r)
    for _, r in player_rows.iterrows():
        apply_selected_map_player_observation(modern_state, r)
    print(f"modern_state: {len(modern_state.team_map)} team-map ledgers, "
          f"{len(modern_state.player_map)} player-map ledgers, "
          f"{len(modern_state.team_map_roster)} team-map-roster ledgers")

    # ---- 4. run the order-dependent driver (map_elo/pool pieces), emit all 25 new features ----
    map_state = MapStateStore()
    emitted = process_modern_map_stream(map_state, modern_state, player_roster_state, map_rows,
                                         emit_features=True)
    emitted_df = pd.DataFrame(emitted)
    print(f"modern-map feature rows emitted (both identities eligible): {len(emitted_df)}")

    new_cols = MODERN_MAP_DIRECTIONAL_FEATURES + MODERN_MAP_SYMMETRIC_FEATURES
    merged = v2.merge(emitted_df[["match_id", "game_id"] + new_cols], on=["match_id", "game_id"],
                       how="left", validate="one_to_one")
    n_missing_new = int(merged["map_recent_history_mass_min"].isna().sum())
    if n_missing_new:
        raise AssertionError(f"{n_missing_new} map_features_v2_rich rows found no modern-map feature match")
    assert len(merged) == len(v2), "row count changed during the modern-map join"
    assert merged["match_id"].tolist() == v2["match_id"].tolist(), "row ordering diverged from V2"
    assert merged["game_id"].tolist() == v2["game_id"].tolist(), "row ordering diverged from V2"

    # ---- 5. every V2-rich column preserved value-for-value ----
    for c in v2.columns:
        a, b = v2[c], merged[c]
        if pd.api.types.is_numeric_dtype(a):
            assert np.array_equal(a.to_numpy(dtype=float), b.to_numpy(dtype=float), equal_nan=True), c
        else:
            assert a.equals(b), c
    print("V2-rich column preservation check PASSED (value-for-value, all columns)")

    meta_cols = cfg["metadata_columns"]
    target = cfg["target"] if isinstance(cfg["target"], str) else list(cfg["target"])[0]
    v2_feature_cols = [c for c in v2.columns if c not in meta_cols + [target]]
    feature_cols = v2_feature_cols + new_cols
    config_feature_cols = cfg["directional_features"] + cfg["symmetric_features"] + cfg["categorical_context"]
    assert sorted(feature_cols) == sorted(config_feature_cols), (
        "assembled feature columns disagree with the config whitelist: "
        f"only-in-built={sorted(set(feature_cols) - set(config_feature_cols))} "
        f"only-in-config={sorted(set(config_feature_cols) - set(feature_cols))}")

    final = merged[meta_cols + [target] + feature_cols].reset_index(drop=True)
    assert len(final) == 10318
    assert len(final.columns) == len(meta_cols) + 1 + len(feature_cols)

    # ---- 6. forbidden-column scan ----
    leaked_exact = {"score1_game", "score2_game", "map_id"} & set(final.columns)
    assert not leaked_exact, f"forbidden column reached the table: {leaked_exact}"
    bad = [c for c in final.columns if any(t in c.lower() for t in FORBIDDEN_TOKENS)]
    assert not bad, f"forbidden token in a column reaching the table: {bad}"

    # ---- 7. NaN contract: exact match, not just "no worse than documented" ----
    for c in new_cols:
        if c in MODERN_MAP_NAN_CAPABLE_FEATURES:
            continue
        n_nan = int(final[c].isna().sum()) if pd.api.types.is_float_dtype(final[c]) else 0
        assert n_nan == 0, f"{c}: not declared NaN-capable but has {n_nan} missing values"
    nan_pattern_gate = (final["roster_map_players_with_history_min"] == 0)
    for c in MODERN_MAP_NAN_CAPABLE_FEATURES:
        assert final[c].isna().equals(nan_pattern_gate), (
            f"{c}: NaN pattern does not exactly match roster_map_players_with_history_min == 0")
    print(f"NaN contract check PASSED: {int(nan_pattern_gate.sum())} rows carry the documented "
          f"roster_map_players_with_history_min == 0 cold-start NaN on {MODERN_MAP_NAN_CAPABLE_FEATURES}")

    non_nan_capable_numeric = [c for c in new_cols if c not in MODERN_MAP_NAN_CAPABLE_FEATURES]
    numeric = final[non_nan_capable_numeric].select_dtypes(include=[np.number])
    assert np.isfinite(numeric.to_numpy()).all(), "a non-NaN-capable new feature is non-finite"

    final.to_parquet(FEATURES_DIR / "map_features_v3_modern_map.parquet", engine="fastparquet", index=False)

    summary = {
        "modern_map_engine_version": MODERN_MAP_ENGINE_VERSION,
        "map_form_half_life_days": MAP_FORM_HALF_LIFE_DAYS,
        "task_id": "map_features_v3_modern_map",
        "prediction_task": "known_map",
        "rows": int(len(final)),
        "v2_rows": int(len(v2)),
        "n_new_directional": len(MODERN_MAP_DIRECTIONAL_FEATURES),
        "n_new_symmetric": len(MODERN_MAP_SYMMETRIC_FEATURES),
        "n_total_directional": len(cfg["directional_features"]),
        "n_total_symmetric": len(cfg["symmetric_features"]),
        "n_total_predictive_inputs": len(cfg["directional_features"]) + len(cfg["symmetric_features"])
        + len(cfg["categorical_context"]),
        "elo_parity_check": {"tolerance": ELO_PARITY_TOLERANCE, "max_abs_diff": max_diff,
                              "n_rows_checked": int(len(parity)), "n_rows_failed": n_bad},
        "rows_with_roster_map_history_nan": int(nan_pattern_gate.sum()),
        "map_order_audit": audit,
        "stream_info": stream_info,
    }
    with open(INTERIM / "map_features_v3_build_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    assert hashes_before == raw_file_hashes(), "data/raw/ was modified during the build"
    print(f"\nWrote {FEATURES_DIR / 'map_features_v3_modern_map.parquet'} "
          f"({len(final)} rows x {len(final.columns)} cols)")
    print(f"total predictive inputs: {summary['n_total_predictive_inputs']} "
          f"({summary['n_total_directional']} directional + {summary['n_total_symmetric']} symmetric + "
          f"{len(cfg['categorical_context'])} categorical)")
    print(f"Wrote {INTERIM / 'map_features_v3_build_summary.json'}")


if __name__ == "__main__":
    main()
