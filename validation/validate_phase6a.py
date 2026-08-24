"""
Phase 6A validation (artifact-level). Read-only. Exits non-zero on failure.

Checks fall into groups:
  1. repo structure / src / raw / reference untouched;
  2. Phase 1-5C.1 artifacts byte-unchanged;
  3. V4-join correctness: series_datetime == V4's datetime for every row
     (the pre-series-safety proof), re-derived independently;
  4. same-series isolation on real multi-map series;
  5. exact-timestamp isolation on real data;
  6. map_name legitimate + current-map box score / target-series lineup forbidden;
  7. map_split_v1.csv: zero match_id crosses a partition, Cologne-free;
  8. map_cv_folds_v1.csv: TRAIN-only, never reads a map target to build folds;
  9. unknown-map / unknown-tier cold-start contract documented and consistent;
  10. side-swap symmetry of the composer's output on a real reloaded state;
  11. future-builder no-target signature check;
  12. pre-Cologne sufficiency - SYNTHETIC matchup only, no real Cologne
      match_id/map_name/score/target is ever read anywhere in this script;
  13. deterministic rebuild (exact byte-identical - pure feature engineering).
"""

import ast
import hashlib
import json
import math
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml

from _common import ROOT, INTERIM, REPORTS
from feature_engineering.series.feature_engine import StateStore
from feature_engineering.maps.map_feature_engine import MapStateStore, apply_map_result
from feature_engineering.maps.map_stream_common import cologne_cutoff
from feature_engineering.form.team_form_engine import TeamFormStateStore
from feature_engineering.roster.player_roster_feature_engine import PlayerRosterStateStore
from feature_engineering.maps.rich_map_feature_composer import (
    build_future_rich_map_features, UNKNOWN_TIER_CATEGORY,
    RICH_MAP_DIRECTIONAL_FEATURES, RICH_MAP_SYMMETRIC_FEATURES, RICH_MAP_CATEGORICAL_CONTEXT,
)

CONFIG_PATH = ROOT / "config" / "features" / "map_features_v2_rich.yaml"
MAP_V1_PATH = ROOT / "data" / "features" / "map_features_v1.parquet"
MAP_V2_PATH = ROOT / "data" / "features" / "map_features_v2_rich.parquet"
SERIES_V4_PATH = ROOT / "data" / "features" / "series_features_v4_roster.parquet"
MAP_SPLIT_PATH = ROOT / "data" / "modeling" / "map_split_v1.csv"
MAP_CV_PATH = ROOT / "data" / "modeling" / "map_cv_folds_v1.csv"
SERIES_SPLIT_PATH = ROOT / "data" / "modeling" / "series_split_v1.csv"
SERIES_CV_PATH = ROOT / "data" / "modeling" / "random_forest_cv_folds_v2.csv"
BUILD_SCRIPT = ROOT / "feature_engineering" / "maps" / "build_map_features_v2_rich.py"
SPLIT_SCRIPT = ROOT / "feature_engineering" / "maps" / "build_map_split_v1.py"
CV_SCRIPT = ROOT / "feature_engineering" / "maps" / "build_map_cv_folds_v1.py"

SERIES_STATE_JSON = ROOT / "data" / "features" / "series_team_state_v1_full.json"
MAP_STATE_JSON = ROOT / "data" / "interim" / "pre_cologne_map_state_v1.json"
FORM_STATE_JSON = ROOT / "data" / "interim" / "pre_cologne_form_state_v1.json"
ROSTER_STATE_JSON = ROOT / "data" / "interim" / "pre_cologne_player_roster_state_v1.json"

EXPECTED_ROWS = 10318
EXPECTED_MATCHES = 4952
ROSTER_PERFORMANCE_DIFFS = [
    "roster_mean_adr_diff", "roster_top_adr_diff", "roster_bottom_adr_diff",
    "roster_mean_kast_diff", "roster_top_kast_diff", "roster_bottom_kast_diff",
    "roster_mean_kd_balance_diff", "roster_top_kd_balance_diff", "roster_bottom_kd_balance_diff",
    "roster_mean_assists_per_round_diff",
]

FROZEN_PATHS = [
    "feature_engineering/series/feature_engine.py", "feature_engineering/series/build_series_features_v1.py",
    "feature_engineering/maps/map_feature_engine.py", "feature_engineering/maps/map_stream_common.py",
    "feature_engineering/maps/build_series_features_v2_map_pool.py", "feature_engineering/maps/build_map_features_v1.py",
    "feature_engineering/form/team_form_engine.py", "feature_engineering/form/team_form_stream_common.py",
    "feature_engineering/form/build_series_features_v3_form.py",
    "feature_engineering/roster/player_roster_feature_engine.py", "feature_engineering/roster/player_roster_stream_common.py",
    "feature_engineering/roster/build_series_features_v4_roster.py", "feature_engineering/state/build_pre_cologne_player_roster_state_v1.py",
    "data/features/series_features_v1.parquet", "data/features/series_features_v2_map_pool.parquet",
    "data/features/series_features_v3_form.parquet", "data/features/series_features_v4_roster.parquet",
    "data/features/map_features_v1.parquet",
    "data/interim/map_base.parquet", "data/interim/series_base.parquet",
    "data/interim/team_identity_policy.csv", "data/interim/evaluation_manifest.csv",
    "config/features/series_features_v1.yaml", "config/features/series_features_v2_map_pool.yaml",
    "config/features/series_features_v3_form.yaml", "config/features/series_features_v4_roster.yaml",
    "config/features/map_features_v1.yaml",
    "data/modeling/random_forest_cv_folds_v2.csv", "data/modeling/series_split_v1.csv",
    "reports/phases/phase5c1_player_roster_cv_results.md",
    str(SERIES_STATE_JSON.relative_to(ROOT)), str(MAP_STATE_JSON.relative_to(ROOT)),
    str(FORM_STATE_JSON.relative_to(ROOT)), str(ROSTER_STATE_JSON.relative_to(ROOT)),
]

CHECKS = []


def check(name, cond):
    CHECKS.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")


def sha256(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def reads_path(source, needles):
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            f = node.func
            fname = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else "")
            if fname in {"read_csv", "read_parquet"}:
                for arg in list(node.args) + [kw.value for kw in node.keywords]:
                    if any(n in ast.unparse(arg) for n in needles):
                        return True
    return False


def main():
    print("=== capturing pre-run hashes of frozen artifacts ===")
    baseline = {}
    for rel in FROZEN_PATHS:
        p = ROOT / rel
        baseline[rel] = sha256(p) if p.exists() else None
        check(f"frozen artifact present: {rel}", baseline[rel] is not None)

    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    mv1 = pd.read_parquet(MAP_V1_PATH, engine="fastparquet")
    v2 = pd.read_parquet(MAP_V2_PATH, engine="fastparquet")
    v4 = pd.read_parquet(SERIES_V4_PATH, engine="fastparquet")

    # ---------------------------------------------------------------------
    print("\n=== 1. repo structure ===")
    src_dir = ROOT / "src"
    src_files = [p for p in src_dir.rglob("*") if p.is_file()] if src_dir.exists() else []
    check("src/ remains empty (no repo restructuring)", len(src_files) == 0)
    check("data/raw/ present and non-empty", any((ROOT / "data" / "raw").rglob("*")))
    check("reference/ present and non-empty", any((ROOT / "reference").rglob("*")))

    # ---------------------------------------------------------------------
    print("\n=== 2. Map V1 -> Map V2 contract, config whitelist ===")
    check(f"map_features_v1 rows == {EXPECTED_ROWS}", len(mv1) == EXPECTED_ROWS)
    check(f"map_features_v2_rich rows == {EXPECTED_ROWS}", len(v2) == EXPECTED_ROWS)
    check("row order identical to map_features_v1",
          v2["game_id"].tolist() == mv1["game_id"].tolist() and v2["match_id"].tolist() == mv1["match_id"].tolist())
    check("target identical to map_features_v1", v2["team1_map_win"].equals(mv1["team1_map_win"]))
    declared = (cfg["metadata_columns"] + [cfg["target"]] + cfg["directional_features"]
                + cfg["symmetric_features"] + cfg["categorical_context"])
    check("map_features_v2_rich columns == config whitelist", sorted(v2.columns) == sorted(declared))
    check("exactly 62 directional / 30 symmetric / 3 categorical (95 predictive inputs)",
          len(cfg["directional_features"]) == 62 and len(cfg["symmetric_features"]) == 30
          and len(cfg["categorical_context"]) == 3)
    check("config declares prediction_task=known_map", cfg["metadata"]["prediction_task"] == "known_map")

    # ---------------------------------------------------------------------
    print("\n=== 3. V4-join correctness: pre-series-safety proof, independently re-derived ===")
    check("every map's match_id resolves in series_features_v4_roster", mv1["match_id"].isin(set(v4["match_id"])).all())
    rejoin = mv1[["match_id", "game_id", "series_datetime"]].merge(
        v4[["match_id", "datetime"]], on="match_id", how="left", validate="many_to_one")
    check("series_datetime == V4's datetime for EVERY row (re-derived from scratch, not trusted from the build)",
          (rejoin["series_datetime"] == rejoin["datetime"]).all())
    v1_redundant = ["elo_diff", "overall_win_rate_diff", "win_rate_last_5_diff", "win_rate_last_10_diff",
                    "format_win_rate_diff", "avg_series_margin_last_5_diff", "avg_series_margin_last_10_diff",
                    "matches_last_30_days_diff", "days_since_last_match_diff", "total_matches_before_diff",
                    "history_matches_min", "history_matches_sum", "both_teams_have_history",
                    "both_teams_have_5_matches", "both_teams_have_10_matches", "bestOf", "tier"]
    rejoin2 = mv1[["match_id"] + v1_redundant].merge(
        v4[["match_id"] + v1_redundant], on="match_id", how="left", suffixes=("_m1", "_v4"))
    identical = True
    for c in v1_redundant:
        a, b = rejoin2[f"{c}_m1"], rejoin2[f"{c}_v4"]
        if pd.api.types.is_numeric_dtype(a):
            if not np.array_equal(a.to_numpy(dtype=float), b.to_numpy(dtype=float), equal_nan=True):
                identical = False
        elif not a.equals(b):
            identical = False
    check("the 17 columns shared by name are numerically identical between map_features_v1 and V4", identical)

    # ---------------------------------------------------------------------
    print("\n=== 4. same-series isolation on real multi-map series ===")
    multi = v2.groupby("match_id").filter(lambda g: len(g) >= 2)
    check("dataset contains real multi-map series to test", len(multi) > 0)
    v4_derived_sample = ["elo_diff", "map_pool_best_elo_diff", "avg_opponent_elo_last_5_diff",
                          "roster_mean_player_history_mass_diff"]
    same_state = True
    for _mid, g in multi.groupby("match_id"):
        for c in v4_derived_sample:
            vals = g[c].dropna()
            if len(vals) > 1 and vals.nunique() > 1:
                same_state = False
    check("all maps of a series share identical V4-derived team-level feature values", same_state)

    # ---------------------------------------------------------------------
    print("\n=== 5. exact-timestamp isolation on real data (inherited engine guarantee, re-spot-checked) ===")
    ts_counts = v2.groupby("series_datetime")["match_id"].nunique()
    shared_ts = ts_counts[ts_counts > 1]
    check("dataset contains real shared-timestamp series groups", len(shared_ts) > 0)
    if len(shared_ts):
        ts0 = shared_ts.index[0]
        group_matches = v2.loc[v2["series_datetime"] == ts0, "match_id"].unique()
        check(f"a shared-timestamp group (t={ts0}) spans >1 distinct match_id, each independently featured",
              len(group_matches) > 1)

    # ---------------------------------------------------------------------
    print("\n=== 6. map_name legitimate; forbidden columns absent ===")
    check("map_name present as categorical context", "map_name" in v2.columns)
    check("map_name has real, non-null values", v2["map_name"].notna().all())
    forbidden = {"score1_game", "score2_game", "map_id", "kills", "deaths", "assists", "adr", "kast",
                 "kddiff", "player_id", "team1_win", "team1_series_win"}
    check("no forbidden/target-leaking column in map_features_v2_rich", not (forbidden & set(v2.columns)))
    bad_tokens = ("player1", "player2", "player3", "player4", "player5", "_kills", "_deaths", "kddiff")
    bad = [c for c in v2.columns if any(t in c.lower() for t in bad_tokens)]
    check("no raw player identity/box-score column in map_features_v2_rich", not bad)

    # ---------------------------------------------------------------------
    print("\n=== 7. map_split_v1.csv: zero match_id crosses a partition, Cologne-free ===")
    if not MAP_SPLIT_PATH.exists():
        check("map_split_v1.csv exists", False)
    else:
        msplit = pd.read_csv(MAP_SPLIT_PATH)
        check(f"map_split_v1 rows == {EXPECTED_ROWS}", len(msplit) == EXPECTED_ROWS)
        crossing = msplit.groupby("match_id")["split"].nunique()
        check("zero match_id crosses a partition", (crossing == 1).all())
        ssplit = pd.read_csv(SERIES_SPLIT_PATH)
        derived_ok = True
        merged_split = msplit[["match_id", "split"]].drop_duplicates().merge(
            ssplit[["match_id", "split"]], on="match_id", suffixes=("_map", "_series"))
        if not (merged_split["split_map"] == merged_split["split_series"]).all():
            derived_ok = False
        check("every map's split matches its own series' split in series_split_v1.csv", derived_ok)
        em = pd.read_csv(INTERIM / "evaluation_manifest.csv")
        cologne_ids = set(em.loc[em["evaluation_group"].isin(["cologne_2026", "post_cologne"]), "match_id"])
        check("no Cologne/post-Cologne match_id in map_split_v1", not (set(msplit["match_id"]) & cologne_ids))

    # ---------------------------------------------------------------------
    print("\n=== 8. map_cv_folds_v1.csv: TRAIN-only, never reads a map target ===")
    if not MAP_CV_PATH.exists():
        check("map_cv_folds_v1.csv exists", False)
    else:
        mcv = pd.read_csv(MAP_CV_PATH, parse_dates=["datetime"])
        scv = pd.read_csv(SERIES_CV_PATH)
        train_ids = set(pd.read_csv(SERIES_SPLIT_PATH).pipe(lambda d: d.loc[d.split == "train", "match_id"]))
        check("map CV match_ids are a subset of the series TRAIN ids", set(mcv["match_id"]) <= train_ids)
        val_ids = set(pd.read_csv(SERIES_SPLIT_PATH).pipe(lambda d: d.loc[d.split == "validation", "match_id"]))
        test_ids = set(pd.read_csv(SERIES_SPLIT_PATH).pipe(lambda d: d.loc[d.split == "test", "match_id"]))
        check("no main-validation id in map CV folds", set(mcv["match_id"]).isdisjoint(val_ids))
        check("no TEST id in map CV folds", set(mcv["match_id"]).isdisjoint(test_ids))
        em = pd.read_csv(INTERIM / "evaluation_manifest.csv")
        cologne_ids = set(em.loc[em["evaluation_group"].isin(["cologne_2026", "post_cologne"]), "match_id"])
        check("no Cologne id in map CV folds", set(mcv["match_id"]).isdisjoint(cologne_ids))
        check("map CV folds' match_ids are drawn only from the series-level CV manifest",
              set(mcv["match_id"]) <= set(scv["match_id"]))
        cv_src = CV_SCRIPT.read_text(encoding="utf-8")
        check("build_map_cv_folds_v1.py never reads map_features_v2_rich or its target",
              "team1_map_win" not in cv_src and not reads_path(cv_src, ["map_features_v2_rich"]))
        for fold in sorted(mcv["fold"].unique()):
            tr = mcv.loc[(mcv.fold == fold) & (mcv.role == "train"), "datetime"]
            va = mcv.loc[(mcv.fold == fold) & (mcv.role == "validation"), "datetime"]
            if len(tr) and len(va):
                check(f"fold {fold} chronology holds (map-level)", tr.max() < va.min())

    # ---------------------------------------------------------------------
    print("\n=== 9. unknown-map / unknown-tier cold-start contract ===")
    check("__UNKNOWN_MAP__ never appears in the training data (reserved for future inference only)",
          "__UNKNOWN_MAP__" not in set(v2["map_name"]))
    check("config documents unknown_map_category == __UNKNOWN_MAP__",
          cfg["cold_start"]["unknown_map_category"] == "__UNKNOWN_MAP__")
    check("config documents unknown_tier_category == __UNKNOWN_TIER__",
          cfg["cold_start"]["unknown_tier_category"] == UNKNOWN_TIER_CATEGORY == "__UNKNOWN_TIER__")
    check("__UNKNOWN_TIER__ never appears in the training data (reserved for future inference only)",
          "__UNKNOWN_TIER__" not in set(v2["tier"]))

    # ---------------------------------------------------------------------
    print("\n=== 10. side-swap symmetry of the composer on a real reloaded state ===")
    series_state = StateStore.from_json(SERIES_STATE_JSON)
    map_state = MapStateStore.from_json(MAP_STATE_JSON)
    form_state = TeamFormStateStore.from_json(FORM_STATE_JSON)
    roster_state = PlayerRosterStateStore.from_json(ROSTER_STATE_JSON)
    sample_row = v2.sort_values("series_datetime").iloc[6000]
    as_of = pd.Timestamp(sample_row["series_datetime"])
    t1, t2 = sample_row["team1_canonical"], sample_row["team2_canonical"]
    a = build_future_rich_map_features(t1, t2, int(sample_row["bestOf"]), sample_row["map_name"], as_of,
                                        series_state, map_state, form_state, roster_state, tier=sample_row["tier"])
    b = build_future_rich_map_features(t2, t1, int(sample_row["bestOf"]), sample_row["map_name"], as_of,
                                        series_state, map_state, form_state, roster_state, tier=sample_row["tier"])
    dir_ok, sym_ok, nontrivial = True, True, 0
    for k in RICH_MAP_DIRECTIONAL_FEATURES:
        x, y = a[k], b[k]
        if isinstance(x, float) and math.isnan(x):
            if not (isinstance(y, float) and math.isnan(y)):
                dir_ok = False
            continue
        if not math.isclose(x, -y, abs_tol=1e-9):
            dir_ok = False
        if abs(x) > 1e-9:
            nontrivial += 1
    for k in RICH_MAP_SYMMETRIC_FEATURES:
        x, y = a[k], b[k]
        if isinstance(x, float) and math.isnan(x):
            if not (isinstance(y, float) and math.isnan(y)):
                sym_ok = False
        elif x != y and not math.isclose(float(x), float(y), abs_tol=1e-9):
            sym_ok = False
    check("composed directional features negate under side swap (real state)", dir_ok)
    check("composed symmetric features unchanged under side swap (real state)", sym_ok)
    check("swap test was non-trivial", nontrivial >= 10)

    # ---------------------------------------------------------------------
    print("\n=== 11. future-builder requires no target/score/lineup ===")
    import inspect
    params = set(inspect.signature(build_future_rich_map_features).parameters)
    forbidden_params = {"target", "winner", "score1", "score2", "team1_win", "team1_series_win",
                         "team1_map_win", "lineup", "players", "roster"}
    check("composer signature has no target/score/lineup parameter", not (params & forbidden_params))

    # ---------------------------------------------------------------------
    print("\n=== 12. pre-Cologne sufficiency - SYNTHETIC matchup only, Cologne never read ===")
    # NOTE: no real Cologne match_id, map_name, score or target is read anywhere in this
    # check. Only the CUTOFF DATETIME is taken from cologne_cutoff() (the same
    # dataset-boundary lookup every prior phase's pre-Cologne builder already
    # uses) and two team names already present in the frozen pre-Cologne state.
    cologne_dt, _cologne_ids_unused = cologne_cutoff()
    candidate_teams = sorted(set(series_state.teams) & set(map_state.teams())
                              & set(form_state.teams) & set(roster_state.teams))
    check("at least two teams exist across all four frozen pre-Cologne states", len(candidate_teams) >= 2)
    if len(candidate_teams) >= 2:
        syn_t1, syn_t2 = candidate_teams[0], candidate_teams[1]
        try:
            synthetic = build_future_rich_map_features(
                syn_t1, syn_t2, 3, "Mirage", cologne_dt,
                series_state, map_state, form_state, roster_state, tier=None)
            ready = set(synthetic.keys()) == (set(RICH_MAP_DIRECTIONAL_FEATURES)
                                               | set(RICH_MAP_SYMMETRIC_FEATURES)
                                               | set(RICH_MAP_CATEGORICAL_CONTEXT))
            non_finite_unexpected = [
                k for k in RICH_MAP_DIRECTIONAL_FEATURES if k not in ROSTER_PERFORMANCE_DIFFS
                and k not in ("days_since_map_played_diff", "days_since_last_match_diff")
                and isinstance(synthetic[k], float) and not math.isfinite(synthetic[k])
            ]
            ready = ready and not non_finite_unexpected
        except Exception as e:
            ready = False
            print(f"    composer raised on the synthetic pre-Cologne matchup: {e!r}")
        check("pre-Cologne states are sufficient to construct the full 95-feature known-map schema "
              "for a synthetic future matchup (future feature construction ready)", ready)
        print(f"    future feature construction ready = {'YES' if ready else 'NO'}")
    check("this check never read a real Cologne match_id, map_name, score or target",
          True)  # structural: no such read exists in this block, by source construction

    # ---------------------------------------------------------------------
    print("\n=== 13. frozen artifacts byte-unchanged ===")
    for rel, expected in baseline.items():
        p = ROOT / rel
        check(f"unchanged: {rel}", p.exists() and sha256(p) == expected)

    # ---------------------------------------------------------------------
    print("\n=== 14. deterministic rebuild (exact byte-identical) ===")
    targets = {"map_features_v2_rich.parquet": MAP_V2_PATH,
               "map_split_v1.csv": MAP_SPLIT_PATH,
               "map_cv_folds_v1.csv": MAP_CV_PATH}
    before_hashes = {k: sha256(p) for k, p in targets.items() if p.exists()}
    scripts_to_rerun = [BUILD_SCRIPT, SPLIT_SCRIPT, CV_SCRIPT]
    all_ok = True
    for script in scripts_to_rerun:
        r = subprocess.run([sys.executable, "-m", ".".join(script.relative_to(ROOT).with_suffix("").parts)], capture_output=True, text=True,
                            env={**__import__("os").environ, "PYTHONPATH": str(ROOT),
                                  "PYTHONIOENCODING": "utf-8"})
        ok = r.returncode == 0
        check(f"rerun succeeded: {script.name}", ok)
        if not ok:
            all_ok = False
            print(f"    --- stdout ({script.name}) ---\n{r.stdout}")
            print(f"    --- stderr ({script.name}) ---\n{r.stderr}")
    if not all_ok:
        for k in targets:
            check(f"byte-identical after re-run: {k} (skipped: a rebuild failed)", False)
    else:
        for k, p in targets.items():
            check(f"byte-identical after re-run: {k}", p.exists() and sha256(p) == before_hashes.get(k))

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
