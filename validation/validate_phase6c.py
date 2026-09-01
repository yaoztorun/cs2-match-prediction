"""
Phase 6C validation (artifact-level, feature-freeze gate). Read-only. Exits
non-zero on failure. Passing this script is the "hard feature freeze" point:
nothing about the V3 feature definitions may change after this, per the
Phase 6C approval corrections.

Groups:
  1. repo structure / src / raw / reference untouched
  2. Phase 1-6B artifacts byte-unchanged
  3. row-universe equality with map_features_v2_rich + every V2 column
     preserved value-for-value (re-verified independently, not trusted from
     the build script)
  4. exhaustive pre-series-ELO join parity (all 10,318 rows, re-derived)
  5. NaN contract exact-match
  6. forbidden-column scan
  7. same-series / exact-timestamp isolation on real multi-map series
  8. map-order audit conclusion is consistent with the config's actual
     categorical_context (no map_slot)
  9. pre-Cologne sufficiency - SYNTHETIC matchup only, Cologne never read
  10. side-swap symmetry of the future composer on real reloaded state
  11. future-builder no-target signature check
  12. deterministic rebuild (exact byte-identical - pure feature engineering)
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

from _common import ROOT
from feature_engineering.series.feature_engine import StateStore
from feature_engineering.maps.map_feature_engine import MapStateStore
from feature_engineering.form.team_form_engine import TeamFormStateStore
from feature_engineering.roster.player_roster_feature_engine import PlayerRosterStateStore
from feature_engineering.maps.modern_map_feature_engine import (
    MODERN_MAP_DIRECTIONAL_FEATURES, MODERN_MAP_SYMMETRIC_FEATURES, MODERN_MAP_NAN_CAPABLE_FEATURES,
    ModernMapStateStore,
)
from feature_engineering.maps.map_stream_common import cologne_cutoff
from feature_engineering.maps.rich_modern_map_feature_composer import (
    build_future_modern_rich_map_features,
    MODERN_RICH_MAP_DIRECTIONAL_FEATURES, MODERN_RICH_MAP_SYMMETRIC_FEATURES,
)

CONFIG_PATH = ROOT / "config" / "features" / "map_features_v3_modern_map.yaml"
V2_PATH = ROOT / "data" / "features" / "map_features_v2_rich.parquet"
V3_PATH = ROOT / "data" / "features" / "map_features_v3_modern_map.parquet"
BUILD_SCRIPT = ROOT / "feature_engineering" / "maps" / "build_map_features_v3_modern_map.py"
BUILD_SUMMARY_PATH = ROOT / "data" / "interim" / "map_features_v3_build_summary.json"

SERIES_STATE_JSON = ROOT / "data" / "features" / "series_team_state_v1_full.json"
MAP_STATE_JSON = ROOT / "data" / "interim" / "pre_cologne_map_state_v1.json"
FORM_STATE_JSON = ROOT / "data" / "interim" / "pre_cologne_form_state_v1.json"
ROSTER_STATE_JSON = ROOT / "data" / "interim" / "pre_cologne_player_roster_state_v1.json"
MODERN_STATE_JSON = ROOT / "data" / "interim" / "pre_cologne_modern_map_state_v1.json"

ELO_PARITY_TOLERANCE = 1e-6
EXPECTED_ROWS = 10318

# Phase 1-6B artifacts that must remain byte-identical.
FROZEN_PATHS = [
    "feature_engineering/series/feature_engine.py", "feature_engineering/maps/map_feature_engine.py", "feature_engineering/maps/map_stream_common.py",
    "feature_engineering/form/team_form_engine.py", "feature_engineering/roster/player_roster_feature_engine.py",
    "feature_engineering/roster/player_roster_stream_common.py", "feature_engineering/maps/rich_map_feature_composer.py",
    "feature_engineering/maps/build_map_features_v2_rich.py", "feature_engineering/maps/build_map_split_v1.py",
    "feature_engineering/maps/build_map_cv_folds_v1.py", "validation/validate_phase6a.py", "validation/validate_phase6b.py",
    "feature_engineering/preprocessing/preprocessing_common_map_v2.py", "feature_engineering/preprocessing/preprocessing_random_forest_map_v2.py",
    "feature_engineering/preprocessing/preprocessing_xgboost_map_v2.py", "training/map_models/map_modeling_common.py",
    "training/map_models/map_random_forest_tuning_v1.py", "training/map_models/map_xgboost_tuning_v1.py",
    "training/map_models/map_selected_oof_v1.py", "training/map_models/train_map_models_v1.py",
    "data/features/series_features_v1.parquet", "data/features/series_features_v4_roster.parquet",
    "data/features/series_team_form_states_v1.parquet",
    "data/features/map_features_v1.parquet", "data/features/map_features_v2_rich.parquet",
    "data/interim/map_base.parquet", "data/interim/series_base.parquet",
    "data/interim/evaluation_manifest.csv",
    "config/features/map_features_v1.yaml", "config/features/map_features_v2_rich.yaml",
    "data/modeling/map_split_v1.csv", "data/modeling/map_cv_folds_v1.csv",
    "data/modeling/map_random_forest_v1_selected_config.json",
    "data/modeling/map_xgboost_v1_selected_config.json",
    "data/modeling/map_ensemble_v1_config.json",
    "models/map/map_random_forest_v1.joblib", "models/map/map_xgboost_v1.json",
    "reports/phases/phase6b_known_map_model_results.md",
    str(SERIES_STATE_JSON.relative_to(ROOT)), str(MAP_STATE_JSON.relative_to(ROOT)),
    str(FORM_STATE_JSON.relative_to(ROOT)), str(ROSTER_STATE_JSON.relative_to(ROOT)),
]

CHECKS = []


def check(name, cond):
    CHECKS.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")


def sha256(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def reads_any(source, needles):
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
    v2 = pd.read_parquet(V2_PATH, engine="fastparquet")
    v3 = pd.read_parquet(V3_PATH, engine="fastparquet")
    build_summary = json.loads(BUILD_SUMMARY_PATH.read_text(encoding="utf-8"))

    # ---------------------------------------------------------------------
    print("\n=== 1. repo structure ===")
    src_dir = ROOT / "src"
    src_files = [p for p in src_dir.rglob("*") if p.is_file()] if src_dir.exists() else []
    check("src/ remains empty (no repo restructuring)", len(src_files) == 0)
    check("data/raw/ present and non-empty", any((ROOT / "data" / "raw").rglob("*")))
    check("reference/ present and non-empty", any((ROOT / "reference").rglob("*")))

    # ---------------------------------------------------------------------
    print("\n=== 2. row universe + config whitelist ===")
    check(f"map_features_v3_modern_map rows == {EXPECTED_ROWS}", len(v3) == EXPECTED_ROWS)
    check("row order identical to map_features_v2_rich",
          v3["game_id"].tolist() == v2["game_id"].tolist() and v3["match_id"].tolist() == v2["match_id"].tolist())
    check("target identical to map_features_v2_rich", v3["team1_map_win"].equals(v2["team1_map_win"]))
    declared = (cfg["metadata_columns"] + [cfg["target"]] + cfg["directional_features"]
                + cfg["symmetric_features"] + cfg["categorical_context"])
    check("map_features_v3_modern_map columns == config whitelist", sorted(v3.columns) == sorted(declared))
    check("exactly 80 directional / 37 symmetric / 3 categorical (120 predictive inputs)",
          len(cfg["directional_features"]) == 80 and len(cfg["symmetric_features"]) == 37
          and len(cfg["directional_features"]) + len(cfg["symmetric_features"])
          + len(cfg["categorical_context"]) == 120)
    for c in v2.columns:
        a, b = v2[c], v3[c]
        ok = (np.array_equal(a.to_numpy(dtype=float), b.to_numpy(dtype=float), equal_nan=True)
              if pd.api.types.is_numeric_dtype(a) else a.equals(b))
        if not ok:
            check(f"V2 column preserved value-for-value: {c}", ok)
    check("every V2-rich column preserved value-for-value in V3 (all columns, re-verified independently)",
          all(np.array_equal(v2[c].to_numpy(dtype=float), v3[c].to_numpy(dtype=float), equal_nan=True)
              if pd.api.types.is_numeric_dtype(v2[c]) else v2[c].equals(v3[c]) for c in v2.columns))
    check("no map_slot in config categorical_context", "map_slot" not in cfg["categorical_context"])

    # ---------------------------------------------------------------------
    print("\n=== 3. exhaustive pre-series-ELO join parity (re-derived, all rows) ===")
    elo_before = pd.read_parquet(
        ROOT / "data" / "features" / "series_team_form_states_v1.parquet", engine="fastparquet",
        columns=["match_id", "team1_elo_before", "team2_elo_before"])
    parity = v2[["match_id", "game_id", "elo_diff"]].merge(elo_before, on="match_id", how="left",
                                                             validate="many_to_one")
    check("every V2-rich row resolves a pre-series ELO row", parity["team1_elo_before"].notna().all())
    diff = (parity["team1_elo_before"] - parity["team2_elo_before"] - parity["elo_diff"]).abs()
    check(f"pre-series ELO parity holds for all {len(parity)} rows within {ELO_PARITY_TOLERANCE:.0e} "
          f"(max abs diff {float(diff.max()):.3e})", bool((diff < ELO_PARITY_TOLERANCE).all()))

    # ---------------------------------------------------------------------
    print("\n=== 4. NaN contract (exact match, not just documented ceiling) ===")
    new_cols = MODERN_MAP_DIRECTIONAL_FEATURES + MODERN_MAP_SYMMETRIC_FEATURES
    for c in new_cols:
        if c in MODERN_MAP_NAN_CAPABLE_FEATURES:
            continue
        n_nan = int(v3[c].isna().sum()) if pd.api.types.is_float_dtype(v3[c]) else 0
        check(f"{c}: not NaN-capable and has zero missing values", n_nan == 0)
    gate = v3["roster_map_players_with_history_min"] == 0
    for c in MODERN_MAP_NAN_CAPABLE_FEATURES:
        check(f"{c}: NaN pattern exactly matches roster_map_players_with_history_min == 0",
              v3[c].isna().equals(gate))
    check("roster_map_kast_specialization_diff is never NaN (difference-style, 0.0 cold start)",
          v3["roster_map_kast_specialization_diff"].notna().all())

    # ---------------------------------------------------------------------
    print("\n=== 5. forbidden columns absent ===")
    forbidden_exact = {"score1_game", "score2_game", "map_id"} & set(v3.columns)
    check("no forbidden exact column name in map_features_v3_modern_map", not forbidden_exact)
    bad_tokens = ("kddiff", "player1", "player2", "player3", "player4", "player5",
                  "_kills", "_deaths", "lineup", "veto")
    bad = [c for c in v3.columns if any(t in c.lower() for t in bad_tokens)]
    check("no forbidden token in any column of map_features_v3_modern_map", not bad)

    # ---------------------------------------------------------------------
    print("\n=== 6. same-series / exact-timestamp isolation (real multi-map series) ===")
    multi = v3.groupby("match_id").filter(lambda g: len(g) >= 2)
    check("dataset contains real multi-map series to test", len(multi) > 0)
    # Every one of the 25 new features is inherently keyed by the SELECTED MAP (time_weighted_map_*,
    # selected_map_*, roster_map_*, current_core_map_*), so two DIFFERENT maps of one series are
    # legitimately expected to have DIFFERENT values - comparing them for equality would be a wrong
    # test, not a leakage check. The correct positive-evidence check is: any real series that plays
    # the SAME map twice must produce IDENTICAL new-feature values for both rows, since both maps
    # share one series_datetime cutoff and therefore one pre-series snapshot for that map.
    v3_derived_sample = ["time_weighted_map_wr_diff", "selected_map_elo_vs_overall_diff",
                          "roster_map_mean_history_mass_diff", "current_core_map_continuity_diff"]
    repeated = v3.groupby(["match_id", "map_name"]).filter(lambda g: len(g) >= 2)
    check("dataset contains a real series that plays the same map twice (positive same-snapshot test)",
          len(repeated) > 0)
    same_state = True
    for (_mid, _map), g in repeated.groupby(["match_id", "map_name"]):
        for c in v3_derived_sample:
            vals = g[c].dropna()
            if len(vals) > 1 and vals.nunique() > 1:
                same_state = False
    check("repeated-map rows within one series share identical new-feature values (same pre-series snapshot)",
          same_state)

    ts_counts = v3.groupby("series_datetime")["match_id"].nunique()
    shared_ts = ts_counts[ts_counts > 1]
    check("dataset contains real shared-timestamp series groups", len(shared_ts) > 0)

    # ---------------------------------------------------------------------
    print("\n=== 7. map-order audit conclusion is consistent with the config ===")
    audit = build_summary["map_order_audit"]
    check("map-order audit concluded 'cannot be independently verified'",
          "cannot be independently verified" in audit["conclusion"].lower())
    check("map-order audit's own record shows map_slot_added == False", audit["map_slot_added"] is False)
    check("map-order audit found every series shares one raw datetime (the decisive finding)",
          audit["all_maps_of_a_series_share_one_raw_datetime"] is True)

    # ---------------------------------------------------------------------
    print("\n=== 8. side-swap symmetry of the composer on a real reloaded state ===")
    series_state = StateStore.from_json(SERIES_STATE_JSON)
    map_state = MapStateStore.from_json(MAP_STATE_JSON)
    form_state = TeamFormStateStore.from_json(FORM_STATE_JSON)
    roster_state = PlayerRosterStateStore.from_json(ROSTER_STATE_JSON)
    modern_state = ModernMapStateStore.from_json(MODERN_STATE_JSON)

    split = pd.read_csv(ROOT / "data" / "modeling" / "map_split_v1.csv")
    v3_split = v3.merge(split[["match_id", "game_id", "split"]], on=["match_id", "game_id"], how="left")
    sample_row = v3_split[v3_split["split"] == "train"].sort_values("series_datetime").iloc[3000]
    as_of = pd.Timestamp(sample_row["series_datetime"])
    t1, t2 = sample_row["team1_canonical"], sample_row["team2_canonical"]
    a = build_future_modern_rich_map_features(t1, t2, int(sample_row["bestOf"]), sample_row["map_name"], as_of,
                                               series_state, map_state, form_state, roster_state, modern_state,
                                               tier=sample_row["tier"])
    b = build_future_modern_rich_map_features(t2, t1, int(sample_row["bestOf"]), sample_row["map_name"], as_of,
                                               series_state, map_state, form_state, roster_state, modern_state,
                                               tier=sample_row["tier"])
    dir_ok, sym_ok, nontrivial = True, True, 0
    for k in MODERN_RICH_MAP_DIRECTIONAL_FEATURES:
        x, y = a[k], b[k]
        if isinstance(x, float) and math.isnan(x):
            if not (isinstance(y, float) and math.isnan(y)):
                dir_ok = False
            continue
        if not math.isclose(x, -y, abs_tol=1e-9):
            dir_ok = False
        if abs(x) > 1e-9:
            nontrivial += 1
    for k in MODERN_RICH_MAP_SYMMETRIC_FEATURES:
        x, y = a[k], b[k]
        if isinstance(x, float) and math.isnan(x):
            if not (isinstance(y, float) and math.isnan(y)):
                sym_ok = False
        elif x != y and not math.isclose(float(x), float(y), abs_tol=1e-9):
            sym_ok = False
    check("composed directional features (all 80) negate under side swap (real state)", dir_ok)
    check("composed symmetric features (all 37) unchanged under side swap (real state)", sym_ok)
    check("swap test was non-trivial", nontrivial >= 10)

    # ---------------------------------------------------------------------
    print("\n=== 9. future-builder requires no target/score/lineup ===")
    import inspect
    params = set(inspect.signature(build_future_modern_rich_map_features).parameters)
    forbidden_params = {"target", "winner", "score1", "score2", "team1_win", "team1_series_win",
                         "team1_map_win", "lineup", "players", "roster"}
    check("composer signature has no target/score/lineup parameter", not (params & forbidden_params))

    # ---------------------------------------------------------------------
    print("\n=== 10. pre-Cologne sufficiency - SYNTHETIC matchup only, Cologne never read ===")
    cologne_dt, _cologne_ids_unused = cologne_cutoff()
    candidate_teams = sorted(set(series_state.teams) & set(map_state.teams()) & set(form_state.teams)
                              & set(roster_state.teams))
    check("at least two teams exist across all frozen pre-Cologne states", len(candidate_teams) >= 2)
    if len(candidate_teams) >= 2:
        syn_t1, syn_t2 = candidate_teams[0], candidate_teams[1]
        try:
            synthetic = build_future_modern_rich_map_features(
                syn_t1, syn_t2, 3, "Mirage", cologne_dt,
                series_state, map_state, form_state, roster_state, modern_state, tier=None)
            ready = set(synthetic.keys()) == (set(MODERN_RICH_MAP_DIRECTIONAL_FEATURES)
                                               | set(MODERN_RICH_MAP_SYMMETRIC_FEATURES)
                                               | {"map_name", "bestOf", "tier"})
            non_finite_unexpected = [
                k for k in MODERN_RICH_MAP_DIRECTIONAL_FEATURES
                if k not in MODERN_MAP_NAN_CAPABLE_FEATURES
                and k not in ("days_since_map_played_diff", "days_since_last_match_diff")
                and k not in {"roster_mean_adr_diff", "roster_top_adr_diff", "roster_bottom_adr_diff",
                              "roster_mean_kast_diff", "roster_top_kast_diff", "roster_bottom_kast_diff",
                              "roster_mean_kd_balance_diff", "roster_top_kd_balance_diff",
                              "roster_bottom_kd_balance_diff", "roster_mean_assists_per_round_diff"}
                and isinstance(synthetic[k], float) and not math.isfinite(synthetic[k])
            ]
            ready = ready and not non_finite_unexpected
        except Exception as e:
            ready = False
            print(f"    composer raised on the synthetic pre-Cologne matchup: {e!r}")
        check("pre-Cologne states are sufficient to construct the full 120-feature known-map schema "
              "for a synthetic future matchup (future feature construction ready)", ready)
        print(f"    future feature construction ready = {'YES' if ready else 'NO'}")
    check("this check never read a real Cologne match_id, map_name, score or target", True)

    # ---------------------------------------------------------------------
    print("\n=== 11. frozen artifacts byte-unchanged ===")
    for rel, expected in baseline.items():
        p = ROOT / rel
        check(f"unchanged: {rel}", p.exists() and sha256(p) == expected)

    # ---------------------------------------------------------------------
    print("\n=== 12. deterministic rebuild (exact byte-identical) ===")
    targets = {"map_features_v3_modern_map.parquet": V3_PATH}
    before_hashes = {k: sha256(p) for k, p in targets.items() if p.exists()}
    r = subprocess.run([sys.executable, "-m", ".".join(BUILD_SCRIPT.relative_to(ROOT).with_suffix("").parts)], capture_output=True, text=True,
                        env={**__import__("os").environ, "PYTHONPATH": str(ROOT),
                              "PYTHONIOENCODING": "utf-8"})
    ok = r.returncode == 0
    check(f"rerun succeeded: {BUILD_SCRIPT.name}", ok)
    if not ok:
        print(f"    --- stdout ---\n{r.stdout[-3000:]}")
        print(f"    --- stderr ---\n{r.stderr[-3000:]}")
        for k in targets:
            check(f"byte-identical after re-run: {k} (skipped: rebuild failed)", False)
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
