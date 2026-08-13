"""
Phase 5B.2 validation (artifact-level). Read-only. Exits non-zero on failure.

Checks fall into groups:
  1. frozen artifacts (Phase 1-5B.1 + the engines/scripts this phase reuses) untouched;
  2. V2 -> V3 superset contract (9,456 rows, order, target, every V2 column preserved);
  3. config whitelist matches the real parquet columns;
  4. forbidden columns absent;
  5. exhaustive PRE-MATCH ELO parity vs. Phase 3's own elo_diff, ALL rows;
  6. side-swap symmetry on real rows;
  7. independent recomputation of sampled rows from a fresh state replay;
  8. same-timestamp isolation on real data;
  9. Cologne absence (parquet + snapshot);
  10. cold-start / confidence-flag correspondence;
  11. deterministic rebuild (pure feature engineering - exact byte-identical hash,
      same returncode==0-gated pattern used by validate_phase5a.py).
"""

import hashlib
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml

from _common import ROOT, INTERIM, REPORTS
from map_stream_common import cologne_cutoff
from team_form_engine import (
    FORM_DIRECTIONAL_FEATURES, FORM_SYMMETRIC_FEATURES, FORM_HALF_LIFE_DAYS,
    TeamFormStateStore, process_form_stream, build_future_team_form_features, apply_form_result,
)
from team_form_stream_common import load_series_form_stream

CONFIG_PATH = ROOT / "config" / "series_features_v3_form.yaml"
FEATURES_V2 = ROOT / "data" / "features" / "series_features_v2_map_pool.parquet"
FEATURES_V3 = ROOT / "data" / "features" / "series_features_v3_form.parquet"
AUDIT_V3 = ROOT / "data" / "features" / "series_team_form_states_v1.parquet"
SNAPSHOT_PARQUET = INTERIM / "pre_cologne_form_state_v1.parquet"
SNAPSHOT_JSON = INTERIM / "pre_cologne_form_state_v1.json"
BUILD_SCRIPT = ROOT / "scripts" / "build_series_features_v3_form.py"

EXPECTED_ROWS = 9456
ELO_PARITY_TOLERANCE = 1e-9

# Frozen artifacts (Phase 1-5B.1 + engines/scripts this phase reuses, never modifies) - must not change.
FROZEN_PATHS = [
    "scripts/feature_engine.py",
    "scripts/build_series_features_v1.py",
    "scripts/map_feature_engine.py",
    "scripts/map_stream_common.py",
    "scripts/build_series_features_v2_map_pool.py",
    "data/features/series_features_v1.parquet",
    "data/features/series_features_v2_map_pool.parquet",
    "data/features/map_features_v1.parquet",
    "config/series_features_v1.yaml",
    "config/series_features_v2_map_pool.yaml",
    "config/map_features_v1.yaml",
    "data/modeling/random_forest_v2_selected_config.json",
    "data/modeling/xgboost_v2_selected_config.json",
    "data/modeling/random_forest_cv_folds_v2.csv",
    "models/random_forest_v2.json",
    "models/xgboost_v2.json",
    "reports/phase5b1_series_map_pool_cv_results.md",
    "reports/tables/series_feature_v1_v2_cv_comparison.csv",
    "data/interim/pre_cologne_map_state_v1.json",
    "data/interim/pre_cologne_map_state_v1.parquet",
]

CHECKS = []


def check(name, cond):
    CHECKS.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")


def sha256(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    print("=== capturing pre-run hashes of frozen artifacts ===")
    baseline = {}
    for rel in FROZEN_PATHS:
        p = ROOT / rel
        baseline[rel] = sha256(p) if p.exists() else None
        check(f"frozen artifact present: {rel}", baseline[rel] is not None)

    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    v2 = pd.read_parquet(FEATURES_V2, engine="fastparquet")
    v3 = pd.read_parquet(FEATURES_V3, engine="fastparquet")
    audit = pd.read_parquet(AUDIT_V3, engine="fastparquet")

    # ---------------------------------------------------------------------
    print("\n=== 1. V2 -> V3 superset contract ===")
    check(f"V2 rows == {EXPECTED_ROWS}", len(v2) == EXPECTED_ROWS)
    check(f"V3 rows == {EXPECTED_ROWS}", len(v3) == EXPECTED_ROWS)
    check("V2/V3 match_id order identical", v2["match_id"].tolist() == v3["match_id"].tolist())
    check("V2/V3 target identical", v2["team1_series_win"].equals(v3["team1_series_win"]))
    check("V2/V3 datetime identical", v2["datetime"].equals(v3["datetime"]))
    numeric_ok = True
    for c in v2.columns:
        if pd.api.types.is_numeric_dtype(v2[c]):
            if not np.array_equal(v2[c].to_numpy(dtype=float), v3[c].to_numpy(dtype=float), equal_nan=True):
                numeric_ok = False
        else:
            if not v2[c].equals(v3[c]):
                numeric_ok = False
    check("every V2 column preserved in V3 value-for-value", numeric_ok)

    # ---------------------------------------------------------------------
    print("\n=== 2. config whitelist matches the real parquet columns ===")
    m_declared = (cfg["directional_features"] + cfg["symmetric_features"]
                  + cfg["categorical_context"] + cfg["metadata_columns"] + [cfg["target"]])
    check("series_features_v3_form columns == config whitelist", sorted(v3.columns) == sorted(m_declared))
    check("form_half_life_days recorded as 60.0",
          cfg["constants"]["form_half_life_days"] == FORM_HALF_LIFE_DAYS == 60.0)
    check("8 new directional + 30 inherited == 38 directional total", len(cfg["directional_features"]) == 38)
    check("4 new symmetric + 15 inherited == 19 symmetric total", len(cfg["symmetric_features"]) == 19)
    new_directional = set(FORM_DIRECTIONAL_FEATURES)
    new_symmetric = set(FORM_SYMMETRIC_FEATURES)
    check("new directional features declared in config", new_directional <= set(cfg["directional_features"]))
    check("new symmetric features declared in config", new_symmetric <= set(cfg["symmetric_features"]))
    check("exactly 12 new features total (8 directional + 4 symmetric)",
          len(FORM_DIRECTIONAL_FEATURES) == 8 and len(FORM_SYMMETRIC_FEATURES) == 4)

    # ---------------------------------------------------------------------
    print("\n=== 3. forbidden / target-leaking columns absent ===")
    forbidden = {"map_name", "map_id", "game_id", "score1_game", "score2_game", "team1_map_win",
                 "score1", "score2", "score1_match", "score2_match", "team1_win"}
    check("no target-series-leaking column in V3", not (forbidden & set(v3.columns)))
    player_tokens = ("player", "kill", "death", "assist", "adr", "kast", "headshot", "flash", "clutch", "damage")
    bad = [c for c in v3.columns if any(t in c.lower() for t in player_tokens)]
    check("no player-level column in V3", not bad)
    numeric_form = v3[FORM_DIRECTIONAL_FEATURES + FORM_SYMMETRIC_FEATURES].select_dtypes(include=[np.number])
    check("all new form features finite", bool(np.isfinite(numeric_form.to_numpy()).all()))
    check("no missing value in any new form feature", int(v3[FORM_DIRECTIONAL_FEATURES + FORM_SYMMETRIC_FEATURES].isna().sum().sum()) == 0)

    # ---------------------------------------------------------------------
    print("\n=== 4. exhaustive PRE-MATCH ELO parity vs. Phase 3's elo_diff (all rows) ===")
    merged = v2[["match_id", "elo_diff"]].merge(
        audit[["match_id", "team1_elo_before", "team2_elo_before"]],
        on="match_id", how="inner", validate="one_to_one")
    check("audit parquet covers every V2 row", len(merged) == len(v2))
    diff = (merged["elo_diff"] - (merged["team1_elo_before"] - merged["team2_elo_before"])).abs()
    check(f"form-engine elo_diff matches Phase 3's elo_diff for all {len(merged)} rows (tol {ELO_PARITY_TOLERANCE:.0e})",
          float(diff.max()) < ELO_PARITY_TOLERANCE)

    # ---------------------------------------------------------------------
    print("\n=== 5. side-swap symmetry + independent recomputation on real rows ===")
    stream, _ = load_series_form_stream(evaluation_groups=("development",))
    store = TeamFormStateStore()
    process_form_stream(store, stream, emit_features=False)

    v3_sorted = v3.sort_values(["datetime", "match_id"]).reset_index(drop=True)
    sample_idx = [2000, 5000, 8000]
    dir_ok, sym_ok, recompute_ok = True, True, True
    for idx in sample_idx:
        row = v3_sorted.iloc[idx]
        cutoff = pd.Timestamp(row["datetime"])
        t1, t2 = row["team1_canonical"], row["team2_canonical"]

        a = build_future_team_form_features(store, t1, t2, cutoff)
        b = build_future_team_form_features(store, t2, t1, cutoff)
        for k in FORM_DIRECTIONAL_FEATURES:
            if not np.isclose(a[k], -b[k], atol=1e-9):
                dir_ok = False
        for k in FORM_SYMMETRIC_FEATURES:
            if not np.isclose(a[k], b[k], atol=1e-9):
                sym_ok = False

        # independent recomputation from a store rebuilt from scratch, strictly before cutoff.
        # Must replay in the same chronological (datetime, canonical_match_uid) order
        # process_form_stream itself uses - applying rows in raw file order would compute
        # each match's pre-match ELO against the wrong prior state.
        prior = stream[stream["datetime"] < cutoff].sort_values(["datetime", "canonical_match_uid"])
        fresh_store = TeamFormStateStore()
        for _, r in prior.iterrows():
            apply_form_result(fresh_store, r)
        want = build_future_team_form_features(fresh_store, t1, t2, cutoff)
        for k in FORM_DIRECTIONAL_FEATURES + FORM_SYMMETRIC_FEATURES:
            got = float(row[k])
            if not np.isclose(float(want[k]), got, atol=1e-6):
                recompute_ok = False

    check("real-history directional form features negate under swap (3 sampled rows)", dir_ok)
    check("real-history symmetric/confidence form features unchanged under swap (3 sampled rows)", sym_ok)
    check("sampled V3 rows reproduce under independent from-scratch recomputation", recompute_ok)

    # ---------------------------------------------------------------------
    print("\n=== 6. same-timestamp isolation on real data ===")
    ts_counts = stream["datetime"].value_counts()
    shared_ts = ts_counts[ts_counts > 1].index
    check("dataset contains real same-timestamp groups to test", len(shared_ts) > 0)
    if len(shared_ts) > 0:
        # every match in a shared-timestamp group must see the SAME store state
        # (already exhaustively guaranteed by the ELO-parity check above being
        # bit-identical to Phase 3's own two-phase driver; spot-check one group
        # by rebuilding state fresh up to (not including) that timestamp and
        # confirming the emitted features for every match in the group match a
        # rebuild ending exactly at the group start, independent of intra-group order).
        ts0 = shared_ts[0]
        prior = stream[stream["datetime"] < ts0]
        group = stream[stream["datetime"] == ts0]
        fresh1 = TeamFormStateStore()
        for _, r in prior.iterrows():
            apply_form_result(fresh1, r)
        pre_group_snapshot = {name: ts.elo for name, ts in fresh1.teams.items()}
        # process the whole group via the real two-phase driver starting from this exact state
        processed, _ = process_form_stream(fresh1, group)
        isolation_ok = all(
            abs(processed[i]["team1_elo_before"] - pre_group_snapshot.get(processed[i]["team1_canonical"], 1500.0)) < 1e-9
            for i in range(len(processed))
        )
        check(f"every match in a real shared-timestamp group (t={ts0}) saw pre-group state, not intra-group updates",
              isolation_ok)

    # ---------------------------------------------------------------------
    print("\n=== 7. Cologne absence ===")
    cologne_dt, cologne_ids = cologne_cutoff()
    check("zero Cologne rows in V3", not (set(v3["match_id"]) & cologne_ids))
    snap_meta = json.loads(SNAPSHOT_JSON.read_text(encoding="utf-8"))["meta"]
    check("snapshot cutoff equals the manifest-derived Cologne start",
          pd.Timestamp(snap_meta["cologne_first_datetime"]) == cologne_dt)
    check("snapshot max source datetime strictly before Cologne",
          pd.Timestamp(snap_meta["max_source_series_datetime"]) < cologne_dt)
    check("no post-Cologne deployment snapshot was built", snap_meta["post_cologne_snapshot_built"] is False)
    check("snapshot was NOT rebuilt from series_features_v3_form.parquet",
          "series_features_v3_form.parquet" in snap_meta["not_rebuilt_from"])
    snap_payload = json.loads(SNAPSHOT_JSON.read_text(encoding="utf-8"))
    cologne_ids_str = {str(i) for i in cologne_ids}
    hist_ok = all(str(h["source_match_id"]) not in cologne_ids_str
                  for t in snap_payload["teams"].values() for h in t["history"])
    check("no Cologne match_id in any team's snapshot history", hist_ok)

    # ---------------------------------------------------------------------
    print("\n=== 8. cold-start / confidence-flag correspondence ===")
    both_join = v3[["match_id", "opponent_adjusted_history_min", "both_teams_have_5_adjusted_matches",
                     "both_teams_have_10_adjusted_matches", "time_weighted_history_mass_min"]].merge(
        audit[["match_id", "team1_adjusted_matches_before", "team2_adjusted_matches_before"]],
        on="match_id", how="inner", validate="one_to_one")
    expected_min = both_join[["team1_adjusted_matches_before", "team2_adjusted_matches_before"]].min(axis=1)
    check("opponent_adjusted_history_min matches min(team1,team2) trusted-adjusted counts",
          (both_join["opponent_adjusted_history_min"] == expected_min).all())
    check("both_teams_have_5_adjusted_matches consistent with the min count",
          ((both_join["both_teams_have_5_adjusted_matches"] == 1) == (expected_min >= 5)).all())
    check("both_teams_have_10_adjusted_matches consistent with the min count",
          ((both_join["both_teams_have_10_adjusted_matches"] == 1) == (expected_min >= 10)).all())
    check("time_weighted_history_mass_min is non-negative everywhere", (both_join["time_weighted_history_mass_min"] >= 0).all())
    zero_flag_rows = both_join[both_join["opponent_adjusted_history_min"] == 0]
    check("every opponent_adjusted_history_min==0 row genuinely has >=1 side with zero trusted history",
          bool(((zero_flag_rows["team1_adjusted_matches_before"] == 0)
                | (zero_flag_rows["team2_adjusted_matches_before"] == 0)).all()) if len(zero_flag_rows) else True)

    # ---------------------------------------------------------------------
    print("\n=== 9. deterministic rebuild ===")
    targets = {"series_features_v3_form.parquet": FEATURES_V3,
               "series_team_form_states_v1.parquet": AUDIT_V3}
    before_hashes = {k: sha256(p) for k, p in targets.items()}
    env_scripts = str(ROOT / "scripts")
    r = subprocess.run([sys.executable, str(BUILD_SCRIPT)], capture_output=True, text=True,
                        env={**__import__("os").environ, "PYTHONPATH": env_scripts, "PYTHONIOENCODING": "utf-8"})
    ok = r.returncode == 0
    check("rebuild of series_features_v3_form.parquet succeeded", ok)
    if not ok:
        print(f"    --- stdout ---\n{r.stdout}")
        print(f"    --- stderr ---\n{r.stderr}")
        for k in targets:
            check(f"byte-identical after re-run: {k} (skipped: rebuild failed)", False)
    else:
        for k, p in targets.items():
            check(f"byte-identical after re-run: {k}", sha256(p) == before_hashes[k])

    # ---------------------------------------------------------------------
    print("\n=== 10. frozen artifacts still byte-unchanged after this validator's own run ===")
    for rel in FROZEN_PATHS:
        p = ROOT / rel
        check(f"unchanged: {rel}", p.exists() and sha256(p) == baseline[rel])
    src_dir = ROOT / "src"
    src_files = [p for p in src_dir.rglob("*") if p.is_file()] if src_dir.exists() else []
    check("src/ remains empty (no repo restructuring)", len(src_files) == 0)

    n_pass = sum(1 for _, ok in CHECKS if ok)
    print(f"\n{n_pass}/{len(CHECKS)} checks passed")
    if n_pass != len(CHECKS):
        print("FAILED:")
        for name, ok in CHECKS:
            if not ok:
                print(f"  - {name}")
        sys.exit(1)


if __name__ == "__main__":
    main()
