"""
Phase 6C reports.

Writes:
    reports/phase6c_modern_map_feature_quality.md   (TRAIN-only descriptive diagnostics)

DISCIPLINE
  * computed on the GLOBAL TRAIN partition only (data/modeling/map_split_v1.csv),
    the same discipline every prior phase's quality report used;
  * validation is never summarized, test is never opened, Cologne is never read;
  * NO feature-vs-target association is reported anywhere.
"""

import numpy as np
import pandas as pd
import yaml

from _common import ROOT
from feature_engineering.maps.modern_map_feature_engine import MODERN_MAP_DIRECTIONAL_FEATURES, MODERN_MAP_SYMMETRIC_FEATURES

FEATURES_DIR = ROOT / "data" / "features"
REPORTS = ROOT / "reports"
CONFIG_PATH = ROOT / "config" / "features" / "map_features_v3_modern_map.yaml"

NEW_DIRECTIONAL = list(MODERN_MAP_DIRECTIONAL_FEATURES)
NEW_SYMMETRIC = list(MODERN_MAP_SYMMETRIC_FEATURES)
NEW_ALL = NEW_DIRECTIONAL + NEW_SYMMETRIC


def md_table(df, floatfmt="{:.4f}"):
    def fmt(v):
        if isinstance(v, float):
            return "n/a" if pd.isna(v) else floatfmt.format(v)
        return str(v)
    head = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep] + body)


def main():
    v3 = pd.read_parquet(FEATURES_DIR / "map_features_v3_modern_map.parquet", engine="fastparquet")
    split = pd.read_csv(ROOT / "data" / "modeling" / "map_split_v1.csv")
    merged = v3.merge(split[["match_id", "game_id", "split"]], on=["match_id", "game_id"],
                       how="left", validate="one_to_one")
    assert merged["split"].notna().all()
    train = merged[merged["split"] == "train"].drop(columns=["split"]).reset_index(drop=True)

    md = []
    md.append("# Phase 6C - Modern Selected-Map Feature Quality (descriptive)\n")
    md.append("**Scope discipline.** Every number below is computed on the **global TRAIN partition** "
              "(`data/modeling/map_split_v1.csv`), the same discipline every prior phase's quality report used. "
              "Validation is not summarized, the test partition is not opened, and Cologne is not read. "
              "**No feature-vs-target association is reported anywhere.**\n")
    md.append(f"TRAIN map rows: **{len(train):,}** of {len(v3):,}.\n")

    md.append("## 1. Coverage - team-level recent selected-map history\n")
    rows = [
        ("both teams have recent selected-map history (both_teams_have_recent_selected_map_history==1)",
         int((train["both_teams_have_recent_selected_map_history"] == 1).sum())),
        ("both teams have trusted-opponent-adjusted recent map history "
         "(map_adjusted_history_mass_min > 0)",
         int((train["map_adjusted_history_mass_min"] > 0).sum())),
        ("selected map present in BOTH teams' recent map-pool (selected_map_in_both_recent_pools==1)",
         int((train["selected_map_in_both_recent_pools"] == 1).sum())),
    ]
    cov = pd.DataFrame(rows, columns=["quantity", "n"])
    cov["pct_of_train"] = 100 * cov["n"] / len(train)
    md.append(md_table(cov, "{:.2f}"))
    md.append("")

    md.append("## 2. Coverage - current-roster selected-map performance\n")
    rows2 = [
        ("roster_map_players_with_history_min >= 1 (at least one evidenced player per side)",
         int((train["roster_map_players_with_history_min"] >= 1).sum())),
        ("roster_map_players_with_history_min >= 3",
         int((train["roster_map_players_with_history_min"] >= 3).sum())),
        ("roster_map_players_with_history_min == 0 (cold start - the NaN gate)",
         int((train["roster_map_players_with_history_min"] == 0).sum())),
        ("current_core_map_continuity_min > 0 (at least some prior-core overlap on this map, both sides)",
         int((train["current_core_map_continuity_min"] > 0).sum())),
    ]
    cov2 = pd.DataFrame(rows2, columns=["quantity", "n"])
    cov2["pct_of_train"] = 100 * cov2["n"] / len(train)
    md.append(md_table(cov2, "{:.2f}"))
    md.append("")

    md.append("## 3. Missingness (TRAIN)\n")
    md.append("Every NaN below is a documented cold-start contract, never a fabricated value - the four "
              "roster-map performance diffs are NaN exactly when `roster_map_players_with_history_min == 0`. "
              "No other new feature carries any NaN.\n")
    miss = []
    for c in NEW_ALL:
        n_missing = int(train[c].isna().sum())
        if n_missing:
            miss.append((c, n_missing, 100 * n_missing / len(train)))
    mdf = pd.DataFrame(miss, columns=["feature", "n_missing", "pct_missing"])
    md.append(md_table(mdf, "{:.2f}") if len(mdf) else "No new feature has any missing value.")
    md.append("")

    md.append("## 4. History-mass distributions (TRAIN)\n")
    mass_cols = ["map_recent_history_mass_min", "map_adjusted_history_mass_min",
                 "roster_map_history_mass_min", "current_core_map_continuity_min"]
    dist_rows = []
    for c in mass_cols:
        s = train[c]
        dist_rows.append((c, float(s.mean()), float(s.std()), float(s.min()), float(s.median()), float(s.max())))
    md.append(md_table(pd.DataFrame(dist_rows, columns=["feature", "mean", "std", "min", "median", "max"]),
                        "{:.4f}"))
    md.append("")

    md.append("## 5. Selected-map recency and specialization distributions (TRAIN)\n")
    spec_cols = ["time_weighted_map_wr_diff", "time_weighted_map_performance_residual_diff",
                 "time_weighted_map_opponent_elo_diff", "selected_map_elo_vs_overall_diff",
                 "selected_map_elo_vs_pool_mean_diff", "selected_map_wr_vs_pool_mean_diff",
                 "selected_map_rank_percentile_diff", "roster_map_kast_specialization_diff",
                 "current_core_map_continuity_diff"]
    dist_rows2 = []
    for c in spec_cols:
        s = train[c].dropna()
        dist_rows2.append((c, float(s.mean()), float(s.std()), float(s.min()), float(s.median()), float(s.max())))
    md.append(md_table(pd.DataFrame(dist_rows2, columns=["feature", "mean", "std", "min", "median", "max"]),
                        "{:.4f}"))
    md.append("")

    md.append("## 6. Correlation among the new features (TRAIN, descriptive only)\n")
    sample_cols = ["time_weighted_map_wr_diff", "selected_map_elo_vs_overall_diff",
                   "selected_map_elo_vs_pool_mean_diff", "roster_map_mean_kast_diff",
                   "roster_map_kast_specialization_diff", "current_core_map_continuity_diff"]
    corr = train[sample_cols].corr()
    corr_df = corr.reset_index().rename(columns={"index": "feature"})
    md.append(md_table(corr_df, "{:.2f}"))
    md.append("\nDescriptive only - no feature is added, removed or reweighted based on this table.\n")

    md.append("## 7. Map-by-map coverage (TRAIN)\n")
    by_map = train.groupby("map_name").agg(
        rows=("match_id", "size"),
        both_recent_history_pct=("both_teams_have_recent_selected_map_history", lambda s: 100 * s.mean()),
        both_pool_membership_pct=("selected_map_in_both_recent_pools", lambda s: 100 * s.mean()),
        roster_map_coverage_pct=("roster_map_players_with_history_min",
                                  lambda s: 100 * (s >= 1).mean()),
    ).sort_values("rows", ascending=False).reset_index()
    md.append(md_table(by_map, "{:.2f}"))
    md.append("")

    md.append("## 8. What this report does not claim\n")
    md.append("Nothing here says any feature is useful. Coverage, spread and completeness are properties of "
              "the data; predictive value is assessed separately, TRAIN-only, in "
              "`evaluation/validation/evaluate_map_feature_sets_v3.py` (Stage B) - never here.\n")

    (REPORTS / "phases" / "phase6c_modern_map_feature_quality.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {REPORTS / 'phase6c_modern_map_feature_quality.md'}")


if __name__ == "__main__":
    main()
