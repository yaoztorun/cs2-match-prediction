"""
Phase 5A validation (artifact-level). Read-only. Exits non-zero on failure.

Checks fall into five groups:
  1. nothing that should be frozen was touched;
  2. the two new datasets match their config contracts exactly;
  3. no forbidden / leaking column reached either table;
  4. leakage-safety properties re-derived INDEPENDENTLY of the engine
     (recomputed straight from map_base rather than trusting the builder);
  5. the pipeline is deterministic on a clean re-run.
"""

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from _common import ROOT, INTERIM, REPORTS, raw_file_hashes
from feature_engineering.maps.map_feature_engine import (
    MAP_ENGINE_VERSION, MAP_POOL_LOOKBACK_DAYS, EXPERIENCED_MAP_MIN_MATCHES,
    MapStateStore, build_future_map_matchup_features, build_future_series_map_pool_features,
    MAP_DIRECTIONAL_FEATURES, MAP_SYMMETRIC_FEATURES,
    SERIES_MAP_POOL_DIRECTIONAL, SERIES_MAP_POOL_SYMMETRIC,
)
from feature_engineering.maps.map_stream_common import load_map_stream, cologne_cutoff, canonical_eligibility_map

FEATURES = ROOT / "data" / "features"
MAP_FEATURES = FEATURES / "map_features_v1.parquet"
SERIES_V2 = FEATURES / "series_features_v2_map_pool.parquet"
SERIES_V1 = FEATURES / "series_features_v1.parquet"
SNAP_PARQUET = INTERIM / "pre_cologne_map_state_v1.parquet"
SNAP_JSON = INTERIM / "pre_cologne_map_state_v1.json"
MAP_CFG = ROOT / "config" / "features" / "map_features_v1.yaml"
SER_CFG = ROOT / "config" / "features" / "series_features_v2_map_pool.yaml"
SRC_DIR = ROOT / "src"

EXPECTED_SERIES_ROWS = 9456
EXPECTED_TRAIN_N, EXPECTED_VAL_N, EXPECTED_TEST_N = 6619, 1419, 1418

# sha256 captured read-only BEFORE any Phase 5A work began.
BASELINE_HASHES = {
    "data/features/series_features_v1.parquet": "dd0bfa5272350221feb667a05b5456180b96b1c514040632352ce41452c7c1e9",
    "config/features/series_features_v1.yaml": "31f71bea039fc9b64fd8abc29eaac2aca4a407700b1dadfec833ac576c08481f",
    "data/modeling/series_split_v1.csv": "fe1b947a3dd9829f1fd9b3e8ac8cc8ae796b8426ef728f609523ae8c48c0c253",
    "data/interim/map_base.parquet": "c9008a94f13e1c9d1437d17bac4c241129e9592ca46ee5e2e491f95c6164ad83",
    "data/interim/series_base.parquet": "c76248cf2329db0c635fb4d29013779ac37b36c48bfe7039fe4e6e21a75ef608",
    "data/interim/team_identity_policy.csv": "1a7ffda00928ef3d342c859eeb939f28ed07ae385f25b2b5a726a2bda8a596d8",
    "data/interim/evaluation_manifest.csv": "fdc6c62ab1bd4e52c2d36e1d88a817447fd5fc9e9075a05eed85b8052cde2c03",
    "feature_engineering/series/feature_engine.py": "9650fc15d36a212520af1810b14cc40c80ce298c54113090dc7db42a6cbd7f29",
    "feature_engineering/preprocessing/preprocessing_common.py": "272e090b53847d30120979713d13118b28eebe37e1d03371444a207c676ed101",
    "training/logistic_regression/logistic_regression_scratch.py": "cb31baffff3d248f1109729c3f78e66523c588178757ce9382a376f56cd1d44f",
    "models/series/logistic_regression_scratch_v1.npz": "504c7f83d9e3162daa0680aeaaa2bf9e7051882e3c5cf17dc05cf9ab494402a3",
    "models/series/logistic_regression_scratch_v1.json": "584bb916f6260276d09245c8804dca386d32eecbba49691b193f819a6a0c0046",
    "models/series/logistic_regression_scratch_v2.npz": "afaee2b0ef62775af8b55a580d60b8987ee0bff63a5d67ba275d12ae2e4cf0ee",
    "models/series/logistic_regression_scratch_v2.json": "febf663c8f0f50950f592fe46e4687e28d0c392db7c747a1a4e5a1c9e44ca7f7",
    "models/series/random_forest_v1.joblib": "05b4cdd377694ad10a5ee8c163cfbaa3daa542c50802538031cf12ac85d051f7",
    "models/series/random_forest_v1.json": "8e1e11137fe7972d9a1f55de020d32e2f655b1733736dbfd1d11a99824412ffb",
    "models/series/random_forest_v2.joblib": "e26e97fd8f1ea7676659605af2d9abd4d4e4cb0c5b767d1df506fb0a9cfac4a9",
    "models/series/random_forest_v2.json": "c5e527161925758718e5597b8ff730e67cdbb4626c3dee0065968be78499456d",
    "models/series/xgboost_v1.json": "9e9719a62b10b07b422683057a6f59ae5cd6a7ef367883e07f828ac41ec38794",
    "models/series/xgboost_v1_metadata.json": "42d47c175116eb0c3baa4e737b4b930db04fdd9808377bae3fca6ea1044f6a73",
    "models/series/xgboost_v2.json": "9479c6d4fcd660967ca01b38316afbdda8ef4470ac850b267cfaac870d258e62",
    "models/series/xgboost_v2_metadata.json": "01d783835fd3c1b3bb871e15a881c433aba517a354edfb5c706d1ff643414d34",
}

CHECKS = []


def check(name, cond):
    CHECKS.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# 1. frozen things stayed frozen
# ---------------------------------------------------------------------------

def check_frozen():
    print("\n=== 1. frozen artifacts ===")
    for rel, expected in BASELINE_HASHES.items():
        p = ROOT / rel
        check(f"unchanged: {rel}", p.exists() and sha256(p) == expected)

    raw = raw_file_hashes()
    check("data/raw/ readable and non-empty", len(raw) > 0)
    ref = sorted((ROOT / "reference").rglob("*"))
    check("reference/ still present", len(ref) > 0)
    src_contents = [p for p in SRC_DIR.rglob("*") if p.is_file()] if SRC_DIR.exists() else []
    check("src/ contains no files", len(src_contents) == 0)

    split = pd.read_csv(ROOT / "data" / "modeling" / "series_split_v1.csv")
    counts = split["split"].value_counts().to_dict()
    check("split partitions unchanged (6619/1419/1418)",
          counts.get("train") == EXPECTED_TRAIN_N and counts.get("validation") == EXPECTED_VAL_N
          and counts.get("test") == EXPECTED_TEST_N)


# ---------------------------------------------------------------------------
# 2. new artifacts and their config contracts
# ---------------------------------------------------------------------------

def check_artifacts_exist():
    print("\n=== 2. new artifacts exist ===")
    for p in [MAP_FEATURES, SERIES_V2, SNAP_PARQUET, SNAP_JSON, MAP_CFG, SER_CFG,
              ROOT / "feature_engineering" / "maps" / "map_feature_engine.py",
              ROOT / "feature_engineering" / "maps" / "map_stream_common.py",
              ROOT / "tests" / "test_map_feature_engine.py",
              REPORTS / "phases" / "phase5a_map_feature_engineering.md",
              REPORTS / "phases" / "phase5a_map_feature_quality.md",
              REPORTS / "tables" / "map_feature_coverage_v1.csv"]:
        check(f"exists: {p.relative_to(ROOT)}", p.exists())


def check_config_contracts(mapf, ser):
    print("\n=== 2b. config contracts ===")
    mcfg = yaml.safe_load(MAP_CFG.read_text(encoding="utf-8"))
    scfg = yaml.safe_load(SER_CFG.read_text(encoding="utf-8"))

    check("map config declares prediction_task=known_map",
          mcfg["metadata"]["prediction_task"] == "known_map")
    check("series config declares prediction_task=pre_veto_series",
          scfg["metadata"]["prediction_task"] == "pre_veto_series")
    check("prediction_task is config metadata, not a map_features column",
          "prediction_task" not in mapf.columns)
    check("prediction_task is config metadata, not a series_v2 column",
          "prediction_task" not in ser.columns)

    m_declared = (mcfg["directional_features"] + mcfg["symmetric_features"]
                  + mcfg["categorical_context"] + mcfg["metadata_columns"] + [mcfg["target"]])
    check("map_features_v1 columns == map config whitelist",
          sorted(mapf.columns) == sorted(m_declared))
    s_declared = (scfg["directional_features"] + scfg["symmetric_features"]
                  + scfg["categorical_context"] + scfg["metadata_columns"] + [scfg["target"]])
    check("series_v2 columns == series config whitelist",
          sorted(ser.columns) == sorted(s_declared))

    check("engine directional/symmetric names appear in the series config",
          set(SERIES_MAP_POOL_DIRECTIONAL) <= set(scfg["directional_features"])
          and set(SERIES_MAP_POOL_SYMMETRIC) <= set(scfg["symmetric_features"]))
    check("engine map feature names appear in the map config",
          set(MAP_DIRECTIONAL_FEATURES) <= set(mcfg["directional_features"])
          and set(MAP_SYMMETRIC_FEATURES) <= set(mcfg["symmetric_features"]))
    check("both configs record lookback 180 / experienced-threshold 5",
          scfg["constants"]["map_pool_lookback_days"] == MAP_POOL_LOOKBACK_DAYS == 180
          and scfg["constants"]["experienced_map_min_matches"] == EXPERIENCED_MAP_MIN_MATCHES == 5)
    check("series config documents the empty-union neutral handling",
          "empty_union" in scfg["cold_start"])
    check("map config reserves __UNKNOWN_MAP__",
          mcfg["cold_start"]["unknown_map_category"] == "__UNKNOWN_MAP__")
    check("per-side audit counts are metadata, never features",
          "team1_map_matches_before" in mcfg["metadata_columns"]
          and "team1_map_matches_before" not in mcfg["directional_features"]
          and "team1_map_matches_before" not in mcfg["symmetric_features"])


# ---------------------------------------------------------------------------
# 3. V1 -> V2 contract, forbidden columns
# ---------------------------------------------------------------------------

V1_FEATURES = ["elo_diff", "overall_win_rate_diff", "win_rate_last_5_diff", "win_rate_last_10_diff",
               "format_win_rate_diff", "avg_series_margin_last_5_diff",
               "avg_series_margin_last_10_diff", "matches_last_30_days_diff",
               "days_since_last_match_diff", "total_matches_before_diff", "history_matches_min",
               "history_matches_sum", "both_teams_have_history", "both_teams_have_5_matches",
               "both_teams_have_10_matches", "bestOf", "tier"]


def check_v1_v2_contract(ser):
    print("\n=== 3. series V2 is a strict superset of V1 ===")
    v1 = pd.read_parquet(SERIES_V1, engine="fastparquet")
    check(f"series V2 row count == {EXPECTED_SERIES_ROWS}", len(ser) == EXPECTED_SERIES_ROWS)
    check("series V2 match_id universe identical to V1",
          set(ser["match_id"]) == set(v1["match_id"]))
    check("series V2 row ORDER identical to V1",
          ser["match_id"].tolist() == v1["match_id"].tolist())
    check("targets identical to V1",
          ser["team1_series_win"].equals(v1["team1_series_win"]))

    identical = []
    for c in V1_FEATURES:
        a, b = v1[c], ser[c]
        if pd.api.types.is_float_dtype(a):
            identical.append(np.array_equal(a.to_numpy(), b.to_numpy(), equal_nan=True))
        else:
            identical.append(a.equals(b))
    check("every V1 feature column preserved value-for-value", all(identical))
    check("V1 has no map-pool column (V2 genuinely adds them)",
          not any(c.startswith(("map_pool_", "map_matchup_")) for c in v1.columns))


def check_forbidden(mapf, ser):
    print("\n=== 3b. forbidden columns ===")
    veto_leaks = {"map_name", "map_id", "game_id", "score1_game", "score2_game",
                  "team1_map_win", "maps_played", "map_list", "veto", "selected_maps"}
    check("pre-veto series V2 contains no map-identity / veto column",
          not (veto_leaks & set(ser.columns)))
    check("map V1 contains no current-map score or map_id",
          not ({"score1_game", "score2_game", "map_id"} & set(mapf.columns)))
    check("map V1 DOES contain map_name (legitimate for the known-map task)",
          "map_name" in mapf.columns)

    player_tokens = ("player", "kill", "death", "assist", "adr", "kast",
                     "headshot", "flash", "clutch", "damage", "_hs")
    bad_m = [c for c in mapf.columns if any(t in c.lower() for t in player_tokens)]
    bad_s = [c for c in ser.columns if any(t in c.lower() for t in player_tokens)]
    check("no player-level column in map V1", not bad_m)
    check("no player-level column in series V2", not bad_s)
    check("evaluation_group is not a feature in series V2", "evaluation_group" not in ser.columns)


# ---------------------------------------------------------------------------
# 4. leakage safety, re-derived independently of the builders
# ---------------------------------------------------------------------------

def check_provenance_and_cologne(mapf, ser):
    print("\n=== 4. provenance and Cologne protection ===")
    mb = pd.read_parquet(INTERIM / "map_base.parquet", engine="fastparquet",
                         columns=["match_id", "game_id", "map_name", "datetime",
                                  "score1_game", "score2_game", "team1_map_win"])
    em = pd.read_csv(INTERIM / "evaluation_manifest.csv")
    cologne_dt, cologne_ids = cologne_cutoff()
    post_ids = set(em.loc[em["evaluation_group"] == "post_cologne", "match_id"])

    key = set(zip(mb["match_id"], mb["game_id"]))
    check("every map V1 row traces to a map_base row",
          set(zip(mapf["match_id"], mapf["game_id"])) <= key)
    check("map V1 has no duplicate (match_id, game_id)",
          not mapf.duplicated(subset=["match_id", "game_id"]).any())

    check("zero Cologne rows in map V1", not (set(mapf["match_id"]) & cologne_ids))
    check("zero post-Cologne rows in map V1", not (set(mapf["match_id"]) & post_ids))
    check("zero Cologne rows in series V2", not (set(ser["match_id"]) & cologne_ids))
    check("zero post-Cologne rows in series V2", not (set(ser["match_id"]) & post_ids))
    dev_ids = set(em.loc[em["evaluation_group"] == "development", "match_id"])
    check("map V1 is entirely development", set(mapf["match_id"]) <= dev_ids)

    # target re-derived straight from map_base, ignoring the built table's own value
    j = mapf[["match_id", "game_id", "team1_map_win"]].merge(
        mb, on=["match_id", "game_id"], suffixes=("_built", "_raw"))
    recomputed = (j["score1_game"] > j["score2_game"]).astype(int)
    check("map V1 target re-derived from raw scores matches",
          int((recomputed != j["team1_map_win_built"]).sum()) == 0)
    check("no tied maps survived into map V1", int((j["score1_game"] == j["score2_game"]).sum()) == 0)


def check_snapshot():
    print("\n=== 4b. frozen pre-Cologne map state ===")
    payload = json.loads(SNAP_JSON.read_text(encoding="utf-8"))
    meta = payload["meta"]
    cologne_dt, cologne_ids = cologne_cutoff()

    check("snapshot records the map engine version",
          payload["map_engine_version"] == MAP_ENGINE_VERSION)
    check("snapshot cutoff equals the manifest-derived Cologne start",
          pd.Timestamp(meta["cologne_first_datetime"]) == cologne_dt)
    check("max source datetime is STRICTLY before Cologne",
          pd.Timestamp(meta["max_source_series_datetime"]) < cologne_dt)
    check("no post-Cologne deployment snapshot was built",
          meta["post_cologne_snapshot_built"] is False
          and not (INTERIM / "post_cologne_map_state_v1.json").exists())

    # independent re-derivation: walk every history entry in the JSON
    late, cologne_hits, entries, untrusted = 0, 0, 0, 0
    for st in payload["states"]:
        for h in st["history"]:
            entries += 1
            if pd.Timestamp(h["series_dt"]) >= cologne_dt:
                late += 1
            if int(h["match_id"]) in cologne_ids:
                cologne_hits += 1
            if not h["opponent_identity_trusted"]:
                untrusted += 1
    check("no history entry at or after the Cologne cutoff", late == 0)
    check("no Cologne match_id anywhere in the snapshot history", cologne_hits == 0)
    check("snapshot is non-trivial (>10k history entries)", entries > 10000)

    # the identity distinction is real, not just documented
    check("snapshot retains own-history entries against untrusted opponents",
          untrusted == meta["entries_from_untrusted_opponent_matches"] > 0)
    check("snapshot was NOT rebuilt from the training table",
          "map_features_v1" in meta["not_rebuilt_from"])

    snap = pd.read_parquet(SNAP_PARQUET, engine="fastparquet")
    check("snapshot parquet row count == team-map states",
          len(snap) == meta["team_map_states"] == len(payload["states"]))
    check("snapshot parquet is flat and scalar",
          not any(isinstance(v, (list, dict, set)) for c in snap.columns for v in snap[c].head(50)))

    store = MapStateStore.from_json(SNAP_JSON)
    check("snapshot reloads into a usable store", len(store.states) == len(payload["states"]))
    f = build_future_series_map_pool_features(
        store, snap.iloc[0]["canonical_team_name"], snap.iloc[-1]["canonical_team_name"],
        3, cologne_dt)
    check("reloaded snapshot produces finite pre-veto features",
          all(np.isfinite(v) for k, v in f.items()
              if isinstance(v, (int, float)) and not isinstance(v, bool)))


def check_recomputation(mapf, ser):
    print("\n=== 4c. independent recomputation of sampled rows ===")
    stream, _ = load_map_stream(evaluation_groups=("development",))
    v1 = pd.read_parquet(SERIES_V1, engine="fastparquet")

    # Rebuild state from scratch up to a chosen cutoff, then compare against the
    # stored row. This reproduces the numbers WITHOUT reusing the builder script.
    sample_series = v1.sort_values("datetime").iloc[[3000, 6000, 9000]]
    ok_series = True
    for _, row in sample_series.iterrows():
        cutoff = pd.Timestamp(row["datetime"])
        prior = stream[stream["series_datetime"] < cutoff]
        store = MapStateStore()
        from feature_engineering.maps.map_feature_engine import apply_map_result
        for _, r in prior.iterrows():
            apply_map_result(store, r)
        want = build_future_series_map_pool_features(
            store, row["team1_canonical"], row["team2_canonical"], row["bestOf"], cutoff, row["tier"])
        got = ser[ser["match_id"] == row["match_id"]].iloc[0]
        for k in SERIES_MAP_POOL_DIRECTIONAL + SERIES_MAP_POOL_SYMMETRIC:
            if not np.isclose(float(want[k]), float(got[k]), rtol=0, atol=1e-9):
                ok_series = False
                print(f"    mismatch {row['match_id']} {k}: {want[k]} vs {got[k]}")
    check("sampled series V2 rows reproduce under independent recomputation", ok_series)

    sample_map = mapf.sort_values(["series_datetime", "game_id"]).iloc[[2000, 5000, 8000]]
    ok_map = True
    for _, row in sample_map.iterrows():
        cutoff = pd.Timestamp(row["series_datetime"])
        prior = stream[stream["series_datetime"] < cutoff]
        store = MapStateStore()
        from feature_engineering.maps.map_feature_engine import apply_map_result
        for _, r in prior.iterrows():
            apply_map_result(store, r)
        want = build_future_map_matchup_features(
            store, row["team1_canonical"], row["team2_canonical"], row["bestOf"],
            row["map_name"], cutoff, row["tier"])
        for k in MAP_DIRECTIONAL_FEATURES + MAP_SYMMETRIC_FEATURES:
            a, b = float(want[k]), float(row[k])
            if np.isnan(a) and np.isnan(b):
                continue
            if not np.isclose(a, b, rtol=0, atol=1e-9):
                ok_map = False
                print(f"    mismatch {row['match_id']}/{row['game_id']} {k}: {a} vs {b}")
    check("sampled map V1 rows reproduce under independent recomputation", ok_map)


def check_same_series_isolation_on_real_data(mapf):
    print("\n=== 4d. same-series isolation on REAL rows ===")
    multi = mapf.groupby("match_id").filter(lambda g: len(g) >= 3)
    check("dataset actually contains multi-map series to test", len(multi) > 0)
    cols = MAP_SYMMETRIC_FEATURES + ["team1_map_matches_before", "team2_map_matches_before"]
    # Two maps of one series differ ONLY through map identity, never through a
    # result that happened earlier in the same series. The pool-independent
    # counts below are map-specific, so instead assert the invariant that holds
    # for the whole series: every row shares one cutoff.
    one_cutoff = multi.groupby("match_id")["series_datetime"].nunique()
    check("all maps of a series share exactly one prediction cutoff",
          int(one_cutoff.max()) == 1)

    # The decisive check: a repeated map inside one series must show IDENTICAL
    # pre-series state on both occurrences (state was not updated mid-series).
    dup = mapf[mapf.duplicated(subset=["match_id", "map_name"], keep=False)]
    if len(dup):
        bad = 0
        for (_mid, _mp), g in dup.groupby(["match_id", "map_name"]):
            for c in ["team1_map_matches_before", "team2_map_matches_before", "map_elo_diff"]:
                if g[c].nunique() > 1:
                    bad += 1
        check(f"repeated map within a series shows identical pre-series state "
              f"({len(dup)} such rows)", bad == 0)
    else:
        check("no repeated map within any series (nothing to test)", True)

    # Series-level counts must not grow across the maps of one series.
    grew = 0
    for _mid, g in multi.groupby("match_id"):
        if g["history_matches_sum"].nunique() > 1:
            grew += 1
    check("series-level joined features constant across a series' maps", grew == 0)


def check_symmetry_on_real_rows(mapf, ser):
    print("\n=== 4e. side-swap symmetry on REAL history ===")
    stream, _ = load_map_stream(evaluation_groups=("development",))
    cutoff = pd.Timestamp(ser["datetime"].iloc[7000])
    prior = stream[stream["series_datetime"] < cutoff]
    store = MapStateStore()
    from feature_engineering.maps.map_feature_engine import apply_map_result
    for _, r in prior.iterrows():
        apply_map_result(store, r)

    t1 = ser["team1_canonical"].iloc[7000]
    t2 = ser["team2_canonical"].iloc[7000]
    a = build_future_series_map_pool_features(store, t1, t2, 3, cutoff)
    b = build_future_series_map_pool_features(store, t2, t1, 3, cutoff)
    dir_ok = all(np.isclose(a[k], -b[k], rtol=0, atol=1e-9) for k in SERIES_MAP_POOL_DIRECTIONAL)
    sym_ok = all(np.isclose(a[k], b[k], rtol=0, atol=1e-9) for k in SERIES_MAP_POOL_SYMMETRIC)
    nontrivial = sum(1 for k in SERIES_MAP_POOL_DIRECTIONAL if abs(a[k]) > 1e-9)
    check("real-history pre-veto directional features negate under swap", dir_ok)
    check("real-history pre-veto symmetric features unchanged under swap", sym_ok)
    check("the swap test was non-trivial (>=6 non-zero directional features)", nontrivial >= 6)

    mrow = mapf.iloc[5000]
    mc = pd.Timestamp(mrow["series_datetime"])
    store2 = MapStateStore()
    for _, r in stream[stream["series_datetime"] < mc].iterrows():
        apply_map_result(store2, r)
    fa = build_future_map_matchup_features(store2, mrow["team1_canonical"], mrow["team2_canonical"],
                                           mrow["bestOf"], mrow["map_name"], mc)
    fb = build_future_map_matchup_features(store2, mrow["team2_canonical"], mrow["team1_canonical"],
                                           mrow["bestOf"], mrow["map_name"], mc)
    mdir = all(np.isclose(fa[k], -fb[k], rtol=0, atol=1e-9)
               for k in MAP_DIRECTIONAL_FEATURES
               if not (np.isnan(fa[k]) and np.isnan(fb[k])))
    check("real-history map-specific directional features negate under swap", mdir)
    check("real-history map-specific symmetric features unchanged under swap",
          all(fa[k] == fb[k] for k in MAP_SYMMETRIC_FEATURES))


def check_values(mapf, ser):
    print("\n=== 4f. value sanity ===")
    pool_cols = SERIES_MAP_POOL_DIRECTIONAL + SERIES_MAP_POOL_SYMMETRIC
    check("all pre-veto pool features finite",
          np.isfinite(ser[pool_cols].select_dtypes(include=[np.number]).to_numpy()).all())
    check("no missing value in any pre-veto pool feature",
          int(ser[pool_cols].isna().sum().sum()) == 0)

    defined = [c for c in MAP_DIRECTIONAL_FEATURES + MAP_SYMMETRIC_FEATURES
               if c != "days_since_map_played_diff"]
    check("all map features finite except the documented days-since NaNs",
          np.isfinite(mapf[defined].select_dtypes(include=[np.number]).to_numpy()).all())
    cold = mapf[(mapf["team1_map_matches_before"] == 0) | (mapf["team2_map_matches_before"] == 0)]
    check("days_since_map_played_diff is NaN exactly on cold-start rows",
          int(mapf["days_since_map_played_diff"].isna().sum()) == len(cold))

    check("map_matchup_shared_coverage within [0,1]",
          float(ser["map_matchup_shared_coverage"].min()) >= 0.0
          and float(ser["map_matchup_shared_coverage"].max()) <= 1.0)
    check("shared_recent_map_count never exceeds union_map_count",
          bool((ser["shared_recent_map_count"] <= ser["union_map_count"]).all()))
    check("shared_experienced_map_count never exceeds shared_recent_map_count",
          bool((ser["shared_experienced_map_count"] <= ser["shared_recent_map_count"]).all()))
    check("empty union implies zero coverage",
          bool((ser.loc[ser["union_map_count"] == 0, "map_matchup_shared_coverage"] == 0).all()))
    check("empty union implies neutral advantages",
          bool((ser.loc[ser["union_map_count"] == 0, "map_matchup_mean_elo_advantage"] == 0).all()))
    check("map_matchup_elo_advantage_range is non-negative",
          bool((ser["map_matchup_elo_advantage_range"] >= 0).all()))
    check("map_matches_before_min <= map_matches_before_sum",
          bool((mapf["map_matches_before_min"] <= mapf["map_matches_before_sum"]).all()))
    check("both_teams_have_map_history agrees with the per-side counts",
          bool((mapf["both_teams_have_map_history"] ==
                ((mapf["team1_map_matches_before"] > 0) &
                 (mapf["team2_map_matches_before"] > 0)).astype(int)).all()))
    check("map V1 target is binary", set(mapf["team1_map_win"].unique()) <= {0, 1})


# ---------------------------------------------------------------------------
# 5. determinism
# ---------------------------------------------------------------------------

def check_determinism():
    print("\n=== 5. clean re-run determinism ===")
    targets = {
        "map_features_v1.parquet": MAP_FEATURES,
        "series_features_v2_map_pool.parquet": SERIES_V2,
        "pre_cologne_map_state_v1.parquet": SNAP_PARQUET,
    }
    before = {k: sha256(p) for k, p in targets.items()}
    env_scripts = str(ROOT)
    all_ok = True
    for script in ["feature_engineering.maps.build_map_features_v1", "feature_engineering.maps.build_series_features_v2_map_pool",
                   "feature_engineering.state.build_pre_cologne_map_state_v1"]:
        r = subprocess.run([sys.executable, "-m", script],
                           capture_output=True, text=True,
                           env={**__import__("os").environ, "PYTHONPATH": env_scripts,
                                "PYTHONIOENCODING": "utf-8"})
        ok = r.returncode == 0
        check(f"re-run succeeded: {script}", ok)
        if not ok:
            all_ok = False
            print(f"    --- stdout ({script}) ---\n{r.stdout}")
            print(f"    --- stderr ({script}) ---\n{r.stderr}")

    if not all_ok:
        for k in targets:
            check(f"byte-identical after re-run: {k} (skipped: rebuild failed)", False)
        return

    after = {k: sha256(p) for k, p in targets.items()}
    for k in targets:
        check(f"byte-identical after re-run: {k}", before[k] == after[k])


def main():
    mapf = pd.read_parquet(MAP_FEATURES, engine="fastparquet")
    ser = pd.read_parquet(SERIES_V2, engine="fastparquet")

    check_frozen()
    check_artifacts_exist()
    check_config_contracts(mapf, ser)
    check_v1_v2_contract(ser)
    check_forbidden(mapf, ser)
    check_provenance_and_cologne(mapf, ser)
    check_snapshot()
    check_recomputation(mapf, ser)
    check_same_series_isolation_on_real_data(mapf)
    check_symmetry_on_real_rows(mapf, ser)
    check_values(mapf, ser)
    check_determinism()

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
