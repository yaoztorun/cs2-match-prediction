"""
Phase 3 validation (artifact-level, like validate_phase2.py). Read-only.
Checks the real generated artifacts for the structural/leakage properties
that can be verified post-hoc (letters D, E, J from the spec, plus
reconciliation and Cologne-contamination checks). The synthetic
algorithmic proofs (A, B, C, F, G, H, I + the batch/single-call
equivalence) live in tests/features/test_feature_engine.py (run via pytest).
Exits non-zero if any check fails.
"""

import json
import sys

import pandas as pd
import yaml

from _common import INTERIM, ROOT, raw_file_hashes

FEATURES_DIR = ROOT / "data" / "features"
CONFIG_PATH = ROOT / "config" / "features" / "series_features_v1.yaml"

FORBIDDEN_MODEL_FEATURES = {
    "score1_match", "score2_match", "score1_game", "score2_game", "team1_win",
    "team1_id", "team2_id", "team1", "team2", "match_id", "tournament",
}

CHECKS = []


def check(name, condition):
    CHECKS.append((name, bool(condition)))
    print(f"[{'PASS' if condition else 'FAIL'}] {name}")


def main():
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    model_features = set(cfg["model_features"])
    metadata_cols = set(cfg["metadata"])
    target_col = cfg["target"]

    features = pd.read_parquet(FEATURES_DIR / "series_features_v1.parquet", engine="fastparquet")
    audit = pd.read_parquet(FEATURES_DIR / "series_team_states_v1.parquet", engine="fastparquet")
    excluded = pd.read_csv(INTERIM / "phase3_identity_excluded_matches.csv")
    policy = pd.read_csv(INTERIM / "team_identity_policy.csv")
    em = pd.read_csv(INTERIM / "evaluation_manifest.csv")
    sb = pd.read_parquet(INTERIM / "series_base.parquet", engine="fastparquet")

    # ---- E: forbidden columns never in the model feature whitelist ----
    check("config model_features contains none of the forbidden columns",
          model_features.isdisjoint(FORBIDDEN_MODEL_FEATURES))
    check("config metadata/model_features/target are mutually exclusive",
          model_features.isdisjoint(metadata_cols) and target_col not in model_features
          and target_col not in metadata_cols)

    # ---- whitelist vs actual columns cross-check ----
    expected_cols = metadata_cols | model_features | {target_col}
    actual_cols = set(features.columns)
    check("series_features_v1.parquet columns exactly match config (metadata+model_features+target)",
          actual_cols == expected_cols)

    # ---- D: no Cologne/post-Cologne leakage ----
    cologne_ids = set(em.loc[em["evaluation_group"].isin(["cologne_2026", "post_cologne"]), "match_id"])
    check("series_features_v1.parquet contains zero cologne_2026/post_cologne match_ids",
          set(features["match_id"]).isdisjoint(cologne_ids))

    # ---- J: identity exclusion enforcement ----
    check("no excluded match_id appears in series_features_v1.parquet",
          set(excluded["match_id"]).isdisjoint(set(features["match_id"])))

    canon_elig = policy.groupby("canonical_team_name")["identity_feature_eligible"].min().to_dict()
    sb_idx = sb.set_index("match_id")
    excl_has_ineligible_side = []
    for mid in excluded["match_id"]:
        if mid not in sb_idx.index:
            excl_has_ineligible_side.append(False)
            continue
        row = sb_idx.loc[mid]
        t1_elig = canon_elig.get(row["team1_canonical"], 1)
        t2_elig = canon_elig.get(row["team2_canonical"], 1)
        excl_has_ineligible_side.append((t1_elig == 0) or (t2_elig == 0))
    check("every excluded match genuinely has >=1 identity-ineligible side",
          all(excl_has_ineligible_side))

    check("every match_id in series_features_v1.parquet has both sides identity-eligible",
          features.apply(lambda r: canon_elig.get(sb_idx.loc[r["match_id"], "team1_canonical"], 1) == 1
                          and canon_elig.get(sb_idx.loc[r["match_id"], "team2_canonical"], 1) == 1, axis=1).all())

    # ---- reconciliation ----
    merged = sb.merge(em, on="match_id", how="inner")
    dev_pool = merged[merged["evaluation_group"] == "development"]
    check("feature rows + excluded rows == development pool size",
          len(features) + len(excluded) == len(dev_pool))
    check("series_team_states_v1.parquet (audit) has the same row count as series_features_v1.parquet",
          len(audit) == len(features))

    # ---- target/feature sanity ----
    check("target column is binary {0,1} only", set(features[target_col].unique()) <= {0, 1})
    check("bestOf values valid (1,3,5 only)", set(features["bestOf"].unique()) <= {1, 3, 5})
    check("tier values valid", set(features["tier"].unique()) <= {"tier1", "tier2", "tier3"})
    check("elo_diff is finite for every row", features["elo_diff"].apply(lambda x: pd.notna(x) and abs(x) < 10000).all())
    check("both_teams_have_history/5/10 are monotone-consistent (10 implies 5 implies history)",
          ((features["both_teams_have_10_matches"] <= features["both_teams_have_5_matches"])
           & (features["both_teams_have_5_matches"] <= features["both_teams_have_history"])).all())

    # ---- pre-Cologne snapshot: independent re-verification of Cologne contamination ----
    with open(FEATURES_DIR / "pre_cologne_team_state_v1_full.json", encoding="utf-8") as f:
        pre_cologne_state = json.load(f)
    cologne_match_ids_only = set(em.loc[em["evaluation_group"] == "cologne_2026", "match_id"])
    cologne_uids = {f"kaggle_ektarr:{mid}" for mid in cologne_match_ids_only}
    all_uids_in_snapshot = set()
    for team in pre_cologne_state["teams"].values():
        for h in team["history"]:
            all_uids_in_snapshot.add(h["canonical_match_uid"])
    check("pre_cologne_team_state_v1: zero Cologne match uids in any team's history",
          all_uids_in_snapshot.isdisjoint(cologne_uids))

    pre_cologne_summary = pd.read_parquet(FEATURES_DIR / "pre_cologne_team_state_v1.parquet", engine="fastparquet")
    check("pre_cologne_team_state_v1.parquet is flat/scalar (no list/object-of-list columns)",
          all(pre_cologne_summary[c].apply(lambda v: not isinstance(v, (list, dict))).all()
              for c in pre_cologne_summary.columns))
    check("pre_cologne_team_state_v1.parquet carries versioning metadata columns",
          {"engine_version", "cutoff_datetime", "source_dataset", "n_matches_processed_total"} <= set(pre_cologne_summary.columns))

    # ---- raw data untouched ----
    hashes = raw_file_hashes()
    check("data/raw/: files still present and readable", len(hashes) > 0)

    n_pass = sum(1 for _, ok in CHECKS if ok)
    n_total = len(CHECKS)
    print(f"\n{n_pass}/{n_total} checks passed")
    if n_pass != n_total:
        sys.exit(1)


if __name__ == "__main__":
    main()
