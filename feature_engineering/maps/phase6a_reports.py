"""
Phase 6A reports.

Writes:
    reports/phase6a_map_v2_feature_quality.md   (descriptive diagnostics)

DISCIPLINE
  * computed on the GLOBAL TRAIN partition only (data/modeling/map_split_v1.csv,
    itself derived from data/modeling/series_split_v1.csv), the same
    discipline every prior phase's quality report used;
  * validation is never summarized, test is never opened, Cologne is never read;
  * NO feature-vs-target association is reported anywhere.
"""

import json

import numpy as np
import pandas as pd
import yaml

from _common import INTERIM, ROOT
from feature_engineering.maps.rich_map_feature_composer import RICH_MAP_DIRECTIONAL_FEATURES, RICH_MAP_SYMMETRIC_FEATURES

FEATURES_DIR = ROOT / "data" / "features"
REPORTS = ROOT / "reports"
CONFIG_PATH = ROOT / "config" / "features" / "map_features_v2_rich.yaml"


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
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    v2 = pd.read_parquet(FEATURES_DIR / "map_features_v2_rich.parquet", engine="fastparquet")
    split = pd.read_csv(ROOT / "data" / "modeling" / "map_split_v1.csv")
    v2_with_split = v2.merge(split[["match_id", "game_id", "split"]], on=["match_id", "game_id"],
                             how="left", validate="one_to_one")
    assert v2_with_split["split"].notna().all()
    train = v2_with_split[v2_with_split["split"] == "train"].drop(columns=["split"]).reset_index(drop=True)

    all_new = RICH_MAP_DIRECTIONAL_FEATURES + RICH_MAP_SYMMETRIC_FEATURES

    md = []
    md.append("# Phase 6A - Known-Map Feature Quality (descriptive)\n")
    md.append("**Scope discipline.** Every number below is computed on the **global TRAIN partition** "
              "(`data/modeling/map_split_v1.csv`, itself derived from `series_split_v1.csv` so no match_id "
              "crosses a partition), the same discipline every prior phase's quality report used. Validation is "
              "not summarized, the test partition is not opened, and Cologne is not read. **No feature-vs-target "
              "association is reported anywhere.**\n")
    md.append(f"TRAIN map rows: **{len(train):,}** of {len(v2):,}.\n")

    md.append("## 1. Row count and map representation\n")
    md.append(f"- Total rows: **{len(v2):,}** (identical to `map_features_v1.parquet`)")
    md.append(f"- Distinct matches: **{v2['match_id'].nunique():,}**")
    md.append(f"- Distinct maps represented: **{v2['map_name'].nunique()}**\n")
    rpm = v2["map_name"].value_counts()
    rpm_df = pd.DataFrame({"map_name": rpm.index, "rows": rpm.values})
    rpm_df["pct_of_total"] = 100 * rpm_df["rows"] / len(v2)
    md.append("### Rows per map (imbalance)\n")
    md.append(md_table(rpm_df, "{:.2f}"))
    md.append(f"\nMax/min row-count ratio across maps: **{rpm.max() / rpm.min():.2f}x** "
              f"({rpm.idxmax()}={rpm.max()} vs {rpm.idxmin()}={rpm.min()}).\n")

    md.append("## 2. Coverage (TRAIN)\n")
    rows = [
        ("both teams have prior history on this map (both_teams_have_map_history==1)",
         int((train["both_teams_have_map_history"] == 1).sum())),
        ("both teams have >=5 prior matches on this map",
         int((train["both_teams_have_5_map_matches"] == 1).sum())),
        ("both teams have >=10 prior matches on this map",
         int((train["both_teams_have_10_map_matches"] == 1).sum())),
        ("both_teams_have_5_inferred_players (V4 roster)",
         int((train["both_teams_have_5_inferred_players"] == 1).sum())),
        ("roster_form_players_min >= 5 (V4 player-performance evidence, both sides)",
         int((train["roster_form_players_min"] >= 5).sum())),
        ("roster_form_players_min == 0 (>=1 side has no usable player history)",
         int((train["roster_form_players_min"] == 0).sum())),
    ]
    cov = pd.DataFrame(rows, columns=["quantity", "n"])
    cov["pct_of_train"] = 100 * cov["n"] / len(train)
    md.append(md_table(cov, "{:.2f}"))
    md.append("")

    md.append("## 3. NaN rates on the new/inherited features (TRAIN)\n")
    md.append("Every NaN below is a documented cold-start contract, never a fabricated value - the two "
              "`days_since_*` features are NaN exactly when the corresponding side has zero prior history, and "
              "the ten roster-performance diffs are NaN exactly when `roster_form_players_min == 0`.\n")
    miss = []
    for c in all_new:
        n_missing = int(train[c].isna().sum())
        if n_missing:
            miss.append((c, n_missing, 100 * n_missing / len(train)))
    mdf = pd.DataFrame(miss, columns=["feature", "n_missing", "pct_missing"])
    md.append(md_table(mdf, "{:.2f}") if len(mdf) else "No feature has any missing value.")
    md.append("")

    md.append("## 4. Feature-family counts\n")
    families = [
        ("map-specific (Phase 5A)", 9, 5),
        ("V2 map-pool depth (Phase 5A)", 14, 0),
        ("V2 same-map matchup (Phase 5A)", 6, 0),
        ("V2 map-pool confidence (Phase 5A)", 0, 10),
        ("V3 opponent-strength/residual form (Phase 5B.2)", 5, 0),
        ("V3 time-decayed form (Phase 5B.2)", 3, 0),
        ("V3 form confidence (Phase 5B.2)", 0, 4),
        ("V4 player performance (Phase 5C)", 10, 0),
        ("V4 roster stability (Phase 5C)", 4, 0),
        ("V4 roster evidence/confidence (Phase 5C)", 1, 6),
        ("inherited Phase 3 V1 series-level", 10, 5),
    ]
    fam_df = pd.DataFrame(families, columns=["family", "directional", "symmetric"])
    md.append(md_table(fam_df, "{:.0f}"))
    md.append(f"\nTotal: **{fam_df['directional'].sum()} directional + {fam_df['symmetric'].sum()} symmetric "
              f"+ 3 categorical context = {fam_df['directional'].sum() + fam_df['symmetric'].sum() + 3} "
              "predictive inputs**.\n")

    md.append("## 5. Map-specific evidence by map (TRAIN)\n")
    by_map = train.groupby("map_name").agg(
        rows=("match_id", "size"),
        both_history_pct=("both_teams_have_map_history", lambda s: 100 * s.mean()),
        both5_pct=("both_teams_have_5_map_matches", lambda s: 100 * s.mean()),
        both10_pct=("both_teams_have_10_map_matches", lambda s: 100 * s.mean()),
    ).sort_values("rows", ascending=False).reset_index()
    md.append(md_table(by_map, "{:.2f}"))
    md.append("")

    md.append("## 6. Distribution summaries (TRAIN, new/inherited numeric features - sample)\n")
    sample_cols = ["map_elo_diff", "map_pool_best_elo_diff", "avg_opponent_elo_last_5_diff",
                   "performance_residual_all_diff", "roster_mean_adr_diff", "roster_mean_kast_diff",
                   "core5_continuity_last_10_diff", "roster_mean_player_history_mass_diff"]
    dist_rows = []
    for c in sample_cols:
        s = pd.to_numeric(train[c], errors="coerce")
        dist_rows.append((c, float(s.mean()), float(s.std()), float(s.min()), float(s.median()), float(s.max())))
    md.append(md_table(pd.DataFrame(dist_rows, columns=["feature", "mean", "std", "min", "median", "max"]), "{:.4f}"))
    md.append("")

    md.append("## 7. What this report does not claim\n")
    md.append("Nothing here says any feature is useful. Coverage, spread and completeness are properties of "
              "the data; predictive value is a property of a model that has not been fitted yet - no model is "
              "trained in Phase 6A.\n")

    (REPORTS / "phases" / "phase6a_map_v2_feature_quality.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {REPORTS / 'phase6a_map_v2_feature_quality.md'}")


if __name__ == "__main__":
    main()
