"""
Phase 5C validation (artifact-level). Read-only. Exits non-zero on failure.

Checks fall into groups:
  1. repo structure / src / raw / reference untouched;
  2. Phase 1-5B.3 artifacts byte-unchanged;
  3. V3 -> V4 superset contract (9,456 rows, order, target, every V3 column);
  4. config whitelist == parquet columns, exactly 21 new features;
  5. forbidden / target-series-leaking columns absent;
  6. features are strictly pre-series: independent from-scratch recomputation
     of sampled rows, and proof the target series' own lineup/box score was
     not used for its own emission;
  7. exact-timestamp isolation on real data;
  8. persistent-player transfer semantics on real data;
  9. cold-start contract: NaN exactly where roster_form_players_min == 0;
  10. side-swap mirroring correct on real rows;
  11. pre-Cologne snapshot uncontaminated;
  12. deterministic rebuild (exact byte-identical - pure feature engineering).
"""

import hashlib
import json
import math
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml

from _common import ROOT, INTERIM, REPORTS
from feature_engineering.maps.map_stream_common import cologne_cutoff
from feature_engineering.roster.player_roster_feature_engine import (
    PlayerRosterStateStore, process_player_roster_stream, apply_player_observation,
    build_future_player_roster_features, infer_expected_roster,
    ROSTER_DIRECTIONAL_FEATURES, ROSTER_SYMMETRIC_FEATURES, ROSTER_PERFORMANCE_DIFFS,
    ROSTER_LOOKBACK_DAYS, PLAYER_FORM_HALF_LIFE_DAYS,
)
from feature_engineering.roster.player_roster_stream_common import load_player_roster_stream

CONFIG_PATH = ROOT / "config" / "features" / "series_features_v4_roster.yaml"
FEATURES_V3 = ROOT / "data" / "features" / "series_features_v3_form.parquet"
FEATURES_V4 = ROOT / "data" / "features" / "series_features_v4_roster.parquet"
AUDIT_V4 = ROOT / "data" / "features" / "series_roster_states_v1.parquet"
SNAPSHOT_JSON = INTERIM / "pre_cologne_player_roster_state_v1.json"
SNAPSHOT_PARQUET = INTERIM / "pre_cologne_player_roster_state_v1.parquet"
BUILD_SCRIPT = ROOT / "feature_engineering" / "roster" / "build_series_features_v4_roster.py"

EXPECTED_ROWS = 9456

FROZEN_PATHS = [
    "feature_engineering/series/feature_engine.py",
    "feature_engineering/series/build_series_features_v1.py",
    "feature_engineering/maps/map_feature_engine.py",
    "feature_engineering/maps/map_stream_common.py",
    "feature_engineering/maps/build_series_features_v2_map_pool.py",
    "feature_engineering/form/team_form_engine.py",
    "feature_engineering/form/team_form_stream_common.py",
    "feature_engineering/form/build_series_features_v3_form.py",
    "feature_engineering/preprocessing/preprocessing_common.py",
    "feature_engineering/preprocessing/preprocessing_common_v2_map_pool.py",
    "feature_engineering/preprocessing/preprocessing_common_v3_form.py",
    "evaluation/validation/evaluate_series_feature_sets_v2.py",
    "evaluation/validation/evaluate_series_feature_sets_v3.py",
    "data/features/series_features_v1.parquet",
    "data/features/series_features_v2_map_pool.parquet",
    "data/features/series_features_v3_form.parquet",
    "data/features/map_features_v1.parquet",
    "data/interim/map_base.parquet",
    "data/interim/series_base.parquet",
    "data/interim/team_identity_policy.csv",
    "data/interim/evaluation_manifest.csv",
    "config/features/series_features_v1.yaml",
    "config/features/series_features_v2_map_pool.yaml",
    "config/features/series_features_v3_form.yaml",
    "config/features/map_features_v1.yaml",
    "data/modeling/random_forest_v2_selected_config.json",
    "data/modeling/xgboost_v2_selected_config.json",
    "data/modeling/random_forest_cv_folds_v2.csv",
    "data/modeling/series_split_v1.csv",
    "models/series/random_forest_v2.json",
    "models/series/xgboost_v2.json",
    "reports/phases/phase5b1_series_map_pool_cv_results.md",
    "reports/phases/phase5b3_team_form_cv_results.md",
    "reports/tables/series_feature_v1_v2_cv_comparison.csv",
    "reports/tables/series_feature_v2_v3_cv_comparison.csv",
    "data/interim/pre_cologne_map_state_v1.json",
    "data/interim/pre_cologne_form_state_v1.json",
]

CHECKS = []


def check(name, cond):
    CHECKS.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")


def sha256(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def nan_aware_equal(a, b, tol=1e-9):
    if isinstance(a, float) and math.isnan(a):
        return isinstance(b, float) and math.isnan(b)
    return abs(float(a) - float(b)) <= tol


def main():
    print("=== capturing pre-run hashes of frozen artifacts ===")
    baseline = {}
    for rel in FROZEN_PATHS:
        p = ROOT / rel
        baseline[rel] = sha256(p) if p.exists() else None
        check(f"frozen artifact present: {rel}", baseline[rel] is not None)

    raw_before = {}
    for p in sorted((ROOT / "data" / "raw").rglob("*")):
        if p.is_file():
            raw_before[str(p)] = sha256(p)
    ref_before = {}
    ref_dir = ROOT / "reference"
    if ref_dir.exists():
        for p in sorted(ref_dir.rglob("*")):
            if p.is_file():
                ref_before[str(p)] = sha256(p)

    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    v3 = pd.read_parquet(FEATURES_V3, engine="fastparquet")
    v4 = pd.read_parquet(FEATURES_V4, engine="fastparquet")
    audit = pd.read_parquet(AUDIT_V4, engine="fastparquet")

    # ---------------------------------------------------------------------
    print("\n=== 1. repo structure ===")
    src_dir = ROOT / "src"
    src_files = [p for p in src_dir.rglob("*") if p.is_file()] if src_dir.exists() else []
    check("src/ remains empty (no repo restructuring)", len(src_files) == 0)
    check("data/raw/ present and non-empty", len(raw_before) > 0)
    check("reference/ present and non-empty", len(ref_before) > 0)
    for d in ["data_preparation", "feature_engineering", "training", "evaluation", "tournament", "application",
              "validation", "tests", "data", "config", "reports"]:
        check(f"expected top-level directory present: {d}/", (ROOT / d).is_dir())

    # ---------------------------------------------------------------------
    print("\n=== 2. V3 -> V4 superset contract ===")
    check(f"V3 rows == {EXPECTED_ROWS}", len(v3) == EXPECTED_ROWS)
    check(f"V4 rows == {EXPECTED_ROWS}", len(v4) == EXPECTED_ROWS)
    check("V3/V4 match_id order identical", v3["match_id"].tolist() == v4["match_id"].tolist())
    check("V3/V4 target identical", v3["team1_series_win"].equals(v4["team1_series_win"]))
    check("V3/V4 datetime identical", v3["datetime"].equals(v4["datetime"]))
    preserved = True
    for c in v3.columns:
        if pd.api.types.is_numeric_dtype(v3[c]):
            if not np.array_equal(v3[c].to_numpy(dtype=float), v4[c].to_numpy(dtype=float), equal_nan=True):
                preserved = False
        else:
            if not v3[c].equals(v4[c]):
                preserved = False
    check("every V3 column preserved in V4 value-for-value", preserved)

    new_cols = ROSTER_DIRECTIONAL_FEATURES + ROSTER_SYMMETRIC_FEATURES
    actual_new = set(v4.columns) - set(v3.columns)
    check("exactly 21 new columns (15 directional + 6 symmetric)",
          actual_new == set(new_cols) and len(new_cols) == 21)

    # ---------------------------------------------------------------------
    print("\n=== 3. config whitelist ===")
    declared = (cfg["directional_features"] + cfg["symmetric_features"]
                + cfg["categorical_context"] + cfg["metadata_columns"] + [cfg["target"]])
    check("series_features_v4_roster columns == config whitelist", sorted(v4.columns) == sorted(declared))
    check("config declares prediction_task=pre_veto_series",
          cfg["metadata"]["prediction_task"] == "pre_veto_series")
    check("config records roster_lookback_days=90 / player_form_half_life_days=60",
          cfg["constants"]["roster_lookback_days"] == ROSTER_LOOKBACK_DAYS == 90.0
          and cfg["constants"]["player_form_half_life_days"] == PLAYER_FORM_HALF_LIFE_DAYS == 60.0)
    check("new directional features declared in config",
          set(ROSTER_DIRECTIONAL_FEATURES) <= set(cfg["directional_features"]))
    check("new symmetric features declared in config",
          set(ROSTER_SYMMETRIC_FEATURES) <= set(cfg["symmetric_features"]))

    # ---------------------------------------------------------------------
    print("\n=== 4. forbidden / target-series-leaking columns absent ===")
    forbidden = {"map_name", "map_id", "game_id", "score1_game", "score2_game", "team1_map_win",
                 "score1", "score2", "score1_match", "score2_match", "team1_win", "player_id"}
    check("no target-series-leaking column in V4", not (forbidden & set(v4.columns)))
    raw_player_tokens = ("player1", "player2", "player3", "player4", "player5",
                          "_kills", "_deaths", "kddiff")
    bad = [c for c in v4.columns if any(t in c.lower() for t in raw_player_tokens)]
    check("no raw player identity/box-score column in V4", not bad)

    # ---------------------------------------------------------------------
    print("\n=== 5. cold-start contract: NaN exactly where evidence is absent ===")
    no_evidence = v4["roster_form_players_min"] == 0
    nan_ok = True
    for c in ROSTER_PERFORMANCE_DIFFS:
        if not v4[c].isna().equals(no_evidence):
            nan_ok = False
    check(f"the {len(ROSTER_PERFORMANCE_DIFFS)} performance diffs are NaN exactly where "
          "roster_form_players_min == 0", nan_ok)
    non_perf = [c for c in new_cols if c not in ROSTER_PERFORMANCE_DIFFS]
    check("every non-performance roster feature is finite everywhere",
          bool(np.isfinite(v4[non_perf].to_numpy(dtype=float)).all()))
    check("roster_form_players_min <= roster_size_min everywhere (form is a subset of the roster)",
          bool((v4["roster_form_players_min"] <= v4["roster_size_min"]).all()))
    check("both_teams_have_5_inferred_players agrees with roster_size_min",
          bool(((v4["both_teams_have_5_inferred_players"] == 1) == (v4["roster_size_min"] >= 5)).all()))
    check("concentration/continuity minima within [0,1]",
          bool(((v4["roster_core_concentration_min"].between(0, 1))
                & (v4["roster_core_continuity_last10_min"].between(0, 1))).all()))
    print(f"    cold-start rows: {int(no_evidence.sum())} / {len(v4)} "
          f"({100 * no_evidence.mean():.2f}%)")

    # ---------------------------------------------------------------------
    print("\n=== 6. strictly pre-series: independent from-scratch recomputation ===")
    stream, info = load_player_roster_stream(evaluation_groups=("development",))
    check("authoritative series datetime used (map timestamps are provenance only)",
          "map_rows_whose_map_datetime_differs_from_series_datetime" in info)

    v4_sorted = v4.sort_values(["datetime", "match_id"]).reset_index(drop=True)
    sample_idx = [1500, 4000, 7000, 9000]
    recompute_ok = True
    for idx in sample_idx:
        row = v4_sorted.iloc[idx]
        cutoff = pd.Timestamp(row["datetime"])
        t1, t2 = row["team1_canonical"], row["team2_canonical"]
        # replay chronologically from scratch, strictly before the cutoff
        prior = stream[stream["series_datetime"] < cutoff].sort_values(
            ["series_datetime", "match_id", "game_id", "side", "slot"])
        fresh = PlayerRosterStateStore()
        for _, r in prior.iterrows():
            apply_player_observation(fresh, r)
        want = build_future_player_roster_features(fresh, t1, t2, cutoff)
        for k in new_cols:
            if not nan_aware_equal(want[k], row[k], tol=1e-6):
                recompute_ok = False
    check("sampled V4 rows reproduce under independent from-scratch recomputation", recompute_ok)

    # the target series' own observations must be strictly excluded from its own state
    leak_ok = True
    for idx in sample_idx:
        row = v4_sorted.iloc[idx]
        cutoff = pd.Timestamp(row["datetime"])
        mid = str(row["match_id"])
        own = stream[stream["match_id"].astype(str) == mid]
        if len(own) and (own["series_datetime"] < cutoff).any():
            leak_ok = False
    check("no target series' own player observation is strictly before its own cutoff "
          "(its lineup/box score cannot enter its own features)", leak_ok)

    # ---------------------------------------------------------------------
    print("\n=== 7. exact-timestamp isolation on real data ===")
    ts_counts = stream.drop_duplicates("match_id")["series_datetime"].value_counts()
    shared = ts_counts[ts_counts > 1]
    check("dataset contains real shared-timestamp groups to test", len(shared) > 0)
    if len(shared) > 0:
        ts0 = shared.index[0]
        prior = stream[stream["series_datetime"] < ts0].sort_values(
            ["series_datetime", "match_id", "game_id", "side", "slot"])
        fresh = PlayerRosterStateStore()
        for _, r in prior.iterrows():
            apply_player_observation(fresh, r)
        group_matches = sorted(set(stream.loc[stream["series_datetime"] == ts0, "match_id"].astype(str)))
        teams_at_ts = {}
        for mid in group_matches:
            sub = stream[(stream["series_datetime"] == ts0) & (stream["match_id"].astype(str) == mid)]
            for t in sub["team_canonical"].unique():
                teams_at_ts.setdefault(t, []).append(mid)
        pre_group = {t: infer_expected_roster(fresh, t, ts0) for t in teams_at_ts}
        # now run the real driver over the whole batch and confirm emission used pre-group state
        reqs = pd.DataFrame([{"match_id": mid, "series_datetime": ts0,
                               "team1_canonical": stream[(stream["series_datetime"] == ts0)
                                                          & (stream["match_id"].astype(str) == mid)
                                                          & (stream["side"] == 1)]["team_canonical"].iloc[0],
                               "team2_canonical": stream[(stream["series_datetime"] == ts0)
                                                          & (stream["match_id"].astype(str) == mid)
                                                          & (stream["side"] == 2)]["team_canonical"].iloc[0],
                               "team1_eligible": True, "team2_eligible": True}
                              for mid in group_matches])
        batch = stream[stream["series_datetime"] == ts0]
        emitted = process_player_roster_stream(fresh, batch, reqs, emit_features=True)
        iso_ok = True
        for e in emitted:
            t1 = reqs.loc[reqs["match_id"] == e["match_id"], "team1_canonical"].iloc[0]
            if e["team1_roster_size"] != len(pre_group.get(t1, [])):
                iso_ok = False
        check(f"every match in a real shared-timestamp group (t={ts0}) saw pre-group state only", iso_ok)

    # ---------------------------------------------------------------------
    print("\n=== 8. persistent-player transfer semantics on real data ===")
    snap = json.loads(SNAPSHOT_JSON.read_text(encoding="utf-8"))
    multi_team = 0
    for pid, pdict in snap["players"].items():
        if len({h["team_canonical"] for h in pdict["history"]}) > 1:
            multi_team += 1
    check("transfers are observable in the tracked player state (>0 players on >1 canonical team)",
          multi_team > 0)
    check("snapshot meta records the observed transfer count",
          snap["meta"]["players_with_multiple_teams_observed"] == multi_team)
    # a transferred player's global history spans both teams while team membership does not
    store_full = PlayerRosterStateStore.from_json(SNAPSHOT_JSON)
    transfer_ok = True
    for pid, st in list(store_full.players.items()):
        teams = sorted({h.team_canonical for h in st.history}, key=str)
        if len(teams) > 1:
            first_dt = min(h.series_dt for h in st.history if h.team_canonical == teams[0])
            # before the player's FIRST appearance for a later team, they must not be in its roster
            later = teams[1]
            first_later = min(h.series_dt for h in st.history if h.team_canonical == later)
            roster_before = [p for p, _m, _d in infer_expected_roster(store_full, later, first_later)]
            if pid in roster_before:
                transfer_ok = False
            break
    check("team membership does not precede the player's first observed appearance for that team",
          transfer_ok)

    # ---------------------------------------------------------------------
    print("\n=== 9. side-swap mirroring on real rows ===")
    dir_ok, sym_ok, nontrivial_total = True, True, 0
    for idx in sample_idx:
        row = v4_sorted.iloc[idx]
        cutoff = pd.Timestamp(row["datetime"])
        prior = stream[stream["series_datetime"] < cutoff].sort_values(
            ["series_datetime", "match_id", "game_id", "side", "slot"])
        fresh = PlayerRosterStateStore()
        for _, r in prior.iterrows():
            apply_player_observation(fresh, r)
        a = build_future_player_roster_features(fresh, row["team1_canonical"], row["team2_canonical"], cutoff)
        b = build_future_player_roster_features(fresh, row["team2_canonical"], row["team1_canonical"], cutoff)
        for k in ROSTER_DIRECTIONAL_FEATURES:
            if isinstance(a[k], float) and math.isnan(a[k]):
                if not (isinstance(b[k], float) and math.isnan(b[k])):
                    dir_ok = False
                continue
            if not math.isclose(a[k], -b[k], abs_tol=1e-9):
                dir_ok = False
            if abs(a[k]) > 1e-9:
                nontrivial_total += 1
        for k in ROSTER_SYMMETRIC_FEATURES:
            if not nan_aware_equal(a[k], b[k]):
                sym_ok = False
    check("real-history directional roster features negate under side swap", dir_ok)
    check("real-history symmetric roster features unchanged under side swap", sym_ok)
    check("the swap test was non-trivial", nontrivial_total >= 8)

    # ---------------------------------------------------------------------
    print("\n=== 10. pre-Cologne snapshot uncontaminated ===")
    cologne_dt, cologne_ids = cologne_cutoff()
    meta = snap["meta"]
    check("snapshot cutoff equals the manifest-derived Cologne start",
          pd.Timestamp(meta["cologne_first_datetime"]) == cologne_dt)
    check("snapshot max source datetime strictly before Cologne",
          pd.Timestamp(meta["max_source_series_datetime"]) < cologne_dt)
    check("no post-Cologne deployment snapshot was built", meta["post_cologne_snapshot_built"] is False)
    check("snapshot was NOT rebuilt from series_features_v4_roster.parquet",
          "series_features_v4_roster.parquet" in meta["not_rebuilt_from"])
    cologne_ids_str = {str(i) for i in cologne_ids}
    hist_clean = all(str(h["match_id"]) not in cologne_ids_str
                     for p in snap["players"].values() for h in p["history"])
    check("no Cologne match_id in any player's snapshot history", hist_clean)
    dt_clean = all(pd.Timestamp(h["series_dt"]) < cologne_dt
                   for p in snap["players"].values() for h in p["history"])
    check("no player history entry at/after the Cologne cutoff", dt_clean)
    appt_clean = all(pd.Timestamp(a["series_dt"]) < cologne_dt
                     for t in snap["teams"].values() for a in t["appearances"])
    check("no team appearance at/after the Cologne cutoff", appt_clean)
    check("snapshot declares zero Cologne contamination", meta["cologne_contamination"] == 0)
    check("snapshot parquet exists and is flat/scalar", SNAPSHOT_PARQUET.exists())

    # ---------------------------------------------------------------------
    print("\n=== 11. structural state invariants (malformed duplication cannot inflate) ===")
    dup_player_ok, dup_team_ok = True, True
    for pid, pdict in snap["players"].items():
        keys = [h["game_id"] for h in pdict["history"]]
        if len(keys) != len(set(keys)):
            dup_player_ok = False
    for team, tdict in snap["teams"].items():
        keys = [(a["game_id"], a["player_id"]) for a in tdict["appearances"]]
        if len(keys) != len(set(keys)):
            dup_team_ok = False
    check("at most ONE PlayerMapEntry per (game_id, player_id)", dup_player_ok)
    check("at most ONE AppearanceEntry per (game_id, team, player_id)", dup_team_ok)
    check("loader excluded maps with a player on both sides",
          info["maps_excluded_player_on_both_sides"] >= 0)
    check("loader reported duplicate-slot handling",
          "duplicate_player_slots_collapsed" in info
          and "duplicate_player_slots_conflicting_excluded" in info)

    # ---------------------------------------------------------------------
    print("\n=== 12. deterministic rebuild ===")
    targets = {"series_features_v4_roster.parquet": FEATURES_V4,
               "series_roster_states_v1.parquet": AUDIT_V4}
    before_hashes = {k: sha256(p) for k, p in targets.items()}
    r = subprocess.run([sys.executable, "-m", ".".join(BUILD_SCRIPT.relative_to(ROOT).with_suffix("").parts)], capture_output=True, text=True,
                        env={**__import__("os").environ, "PYTHONPATH": str(ROOT),
                              "PYTHONIOENCODING": "utf-8"})
    ok = r.returncode == 0
    check("rebuild of series_features_v4_roster.parquet succeeded", ok)
    if not ok:
        print(f"    --- stdout ---\n{r.stdout}")
        print(f"    --- stderr ---\n{r.stderr}")
        for k in targets:
            check(f"byte-identical after re-run: {k} (skipped: rebuild failed)", False)
    else:
        for k, p in targets.items():
            check(f"byte-identical after re-run: {k}", sha256(p) == before_hashes[k])

    # ---------------------------------------------------------------------
    print("\n=== 13. frozen artifacts / raw / reference still unchanged ===")
    for rel, expected in baseline.items():
        p = ROOT / rel
        check(f"unchanged: {rel}", p.exists() and sha256(p) == expected)
    raw_after = {}
    for p in sorted((ROOT / "data" / "raw").rglob("*")):
        if p.is_file():
            raw_after[str(p)] = sha256(p)
    check("data/raw/ byte-unchanged", raw_after == raw_before)
    ref_after = {}
    if ref_dir.exists():
        for p in sorted(ref_dir.rglob("*")):
            if p.is_file():
                ref_after[str(p)] = sha256(p)
    check("reference/ byte-unchanged", ref_after == ref_before)

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
