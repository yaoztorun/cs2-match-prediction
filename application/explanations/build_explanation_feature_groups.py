"""
Builds config/application_explanation_feature_groups_v1.yaml (Phase 9C).

RF V2: the 19 transformed model-input features are the LOW-LEVEL attribution
surface (RF trees split on these, never on the 17 raw features directly -
amendment #1/#2). Each transformed feature maps to exactly one of the 17 raw
concepts and to exactly one product factor_group, hand-authored here (no
pre-existing RF family table exists) but validated for 100% coverage against
the frozen preprocessing contract before writing anything.

XGB V3: reuses the EXISTING, already-published Phase 6D family taxonomy
(reports/tables/map_xgboost_v3_final_feature_importance.csv, cross-referenced
against .../map_xgboost_v3_final_group_importance.csv for family labels) as
the low-level grouping - not reinvented. Every one of the 131 transformed
features already has exactly one family letter (A-P); this script only adds
the family -> product-facing factor_group collapse, frozen explicitly below
with a documented rationale, particularly for the K/M/N split (amendment
#11). Exact SET EQUALITY between the family table and the frozen
transformed_feature_names is asserted before writing (amendment #10) - the
131-count sum alone is not treated as sufficient.
"""

import json

import pandas as pd
import yaml

from _common import ROOT

MODELING = ROOT / "data" / "modeling"
TABLES = ROOT / "reports" / "tables"
CONFIG = ROOT / "config"

RF_PREP_PATH = MODELING / "random_forest_preprocessing_v2.json"
XGB_PREP_PATH = MODELING / "map_xgboost_v3_final_preprocessing.json"
XGB_FAMILY_CSV = TABLES / "map_xgboost_v3_final_feature_importance.csv"
XGB_GROUP_CSV = TABLES / "map_xgboost_v3_final_group_importance.csv"

# ---------------------------------------------------------------------------
# RF V2: 17 raw -> 19 transformed -> product factor_group (hand-authored,
# since no pre-existing family table exists for RF; validated below for
# 100% coverage of the frozen preprocessing contract).
# `direction_semantics`: "higher value favors team_a" describes the RAW
# feature's natural sign convention (all *_diff features are team1-team2,
# i.e. team_a-team_b under application orientation - see amendment #8 of
# Phase 9B). Confidence/count/categorical features have no natural
# favors-a-or-b direction and are marked accordingly.
RF_RAW_FEATURE_TAXONOMY = {
    "elo_diff": ("overall_strength", "ELO rating difference",
                 "Difference in each team's overall ELO rating.", "higher_favors_team_a"),
    "overall_win_rate_diff": ("overall_strength", "Overall win-rate difference",
                               "Difference in each team's smoothed all-time win rate.", "higher_favors_team_a"),
    "win_rate_last_5_diff": ("recent_performance", "Last-5-match win-rate difference",
                              "Difference in each team's win rate over its last 5 series.", "higher_favors_team_a"),
    "win_rate_last_10_diff": ("recent_performance", "Last-10-match win-rate difference",
                               "Difference in each team's win rate over its last 10 series.", "higher_favors_team_a"),
    "format_win_rate_diff": ("recent_performance", "Format-specific win-rate difference",
                              "Difference in each team's win rate at this series' best-of format.",
                              "higher_favors_team_a"),
    "avg_series_margin_last_5_diff": ("recent_performance", "Last-5 average series margin difference",
                                       "Difference in each team's average map-score margin over its last 5 "
                                       "series.", "higher_favors_team_a"),
    "avg_series_margin_last_10_diff": ("recent_performance", "Last-10 average series margin difference",
                                        "Difference in each team's average map-score margin over its last "
                                        "10 series.", "higher_favors_team_a"),
    "matches_last_30_days_diff": ("recent_performance", "Recent activity difference",
                                   "Difference in how many series each team played in the last 30 days.",
                                   "context_dependent"),
    "days_since_last_match_diff": ("recent_performance", "Rest/inactivity difference",
                                    "Difference in days since each team's last series.", "context_dependent"),
    "total_matches_before_diff": ("historical_experience", "Total prior match-count difference",
                                   "Difference in each team's total historical series count.",
                                   "context_dependent"),
    "history_matches_min": ("historical_experience", "Minimum historical match count",
                             "The smaller of the two teams' historical series counts (data-sufficiency "
                             "signal, not team-specific).", "neutral_confidence"),
    "history_matches_sum": ("historical_experience", "Combined historical match count",
                             "The sum of both teams' historical series counts (data-sufficiency signal).",
                             "neutral_confidence"),
    "both_teams_have_history": ("historical_experience", "Both teams have any history",
                                 "Whether both teams have at least one prior series (confidence flag).",
                                 "neutral_confidence"),
    "both_teams_have_5_matches": ("historical_experience", "Both teams have >=5 matches",
                                   "Whether both teams have at least 5 prior series (confidence flag).",
                                   "neutral_confidence"),
    "both_teams_have_10_matches": ("historical_experience", "Both teams have >=10 matches",
                                    "Whether both teams have at least 10 prior series (confidence flag).",
                                    "neutral_confidence"),
    "bestOf": ("event_context", "Series format", "The series best-of format (1/3/5).", "categorical"),
    "tier": ("event_context", "Tournament tier", "The series' tournament tier.", "categorical"),
}

RF_PRODUCT_GROUPS = {
    "overall_strength": "Overall team strength (ELO, all-time win rate).",
    "recent_performance": "Recent-form performance, activity, and rest.",
    "historical_experience": "How much historical data exists for this matchup.",
    "event_context": "Series format and tournament tier context.",
}
# Explicitly NOT created for RF (amendment #4 of Phase 9B / section 4 of Phase 9C):
RF_GROUPS_NOT_APPLICABLE = ["opponent_strength", "map_pool", "selected_map_strength", "map_experience",
                             "player_strength", "roster_stability", "roster_map_familiarity"]


def build_rf_feature_groups():
    prep = json.loads(RF_PREP_PATH.read_text(encoding="utf-8"))
    raw_names = prep["original_model_feature_names"]
    transformed_names = prep["transformed_feature_names"]
    if set(raw_names) != set(RF_RAW_FEATURE_TAXONOMY):
        raise ValueError(f"STOP: RF raw feature taxonomy does not match the frozen preprocessing contract. "
                          f"missing={set(raw_names) - set(RF_RAW_FEATURE_TAXONOMY)}, "
                          f"extra={set(RF_RAW_FEATURE_TAXONOMY) - set(raw_names)}")

    # explicit raw -> transformed expansion, derived from the frozen contract itself, not guessed
    raw_to_transformed = {}
    for raw in raw_names:
        if raw == "bestOf":
            raw_to_transformed[raw] = [t for t in transformed_names if t.startswith("bestOf_")]
        elif raw == "tier":
            raw_to_transformed[raw] = [t for t in transformed_names if t.startswith("tier_")]
        else:
            if raw not in transformed_names:
                raise ValueError(f"STOP: raw feature {raw!r} has no 1:1 transformed counterpart")
            raw_to_transformed[raw] = [raw]

    covered_transformed = sorted({t for ts in raw_to_transformed.values() for t in ts})
    if covered_transformed != sorted(transformed_names):
        raise ValueError(f"STOP: RF raw->transformed expansion does not cover the frozen transformed feature "
                          f"list exactly. missing={set(transformed_names) - set(covered_transformed)}, "
                          f"extra={set(covered_transformed) - set(transformed_names)}")

    features = []
    for raw, transformed_list in raw_to_transformed.items():
        group, label, desc, direction = RF_RAW_FEATURE_TAXONOMY[raw]
        for t in transformed_list:
            features.append({
                "model_id": "series_random_forest_v2", "raw_feature": raw, "transformed_feature": t,
                "factor_group": group, "display_label": label, "description": desc,
                "direction_semantics": direction,
            })

    seen = [f["transformed_feature"] for f in features]
    if sorted(seen) != sorted(transformed_names) or len(seen) != len(set(seen)):
        raise ValueError("STOP: RF feature-group mapping has a duplicate or missing transformed feature")

    return {
        "model_id": "series_random_forest_v2", "raw_feature_count": len(raw_names),
        "transformed_feature_count": len(transformed_names),
        "product_factor_groups": RF_PRODUCT_GROUPS,
        "factor_groups_not_applicable_to_this_model": RF_GROUPS_NOT_APPLICABLE,
        "features": features,
    }


# ---------------------------------------------------------------------------
# XGB V3: reuse the existing Phase 6D family taxonomy, collapse families to
# product groups explicitly (amendment #11 - no vague "likely" mapping).
# ---------------------------------------------------------------------------

FAMILY_TO_PRODUCT_GROUP = {
    "A": ("overall_strength", "original series V1 signal (ELO/win-rate/activity), reused unchanged as raw "
          "input to the known-map model."),
    "B": ("map_pool", "map-pool depth and order statistics (breadth/quality of the team's map pool)."),
    "C": ("map_pool", "same-map matchup advantage across the pool."),
    "D": ("map_pool", "map-pool confidence/coverage flags."),
    "E": ("opponent_strength", "opponent-strength / residual form (strength of schedule, over/under-performance "
          "vs. ELO-expected outcome)."),
    "F": ("recent_form", "time-decayed (recency-weighted) form."),
    "G": ("recent_form", "form confidence flags."),
    "H": ("player_strength", "individual player performance statistics (ADR/KAST/KD/assists)."),
    "I": ("roster_stability", "roster/lineup stability and continuity."),
    "J": ("roster_stability", "roster/player confidence flags."),
    "K": ("selected_map_strength", "the team's own RAW historical performance on the specific selected map "
          "(map ELO, win rate, margin, recency-of-last-play, rank-percentile vs. own pool) - the direct "
          "strength signal on this map."),
    "L": ("event_context", "categorical map/bestOf/tier context."),
    "M": ("map_experience", "recency- and opponent-adjusted selected-map TEAM features - a comparative/"
          "contextual view of map performance, distinct from K's raw strength numbers."),
    "N": ("map_experience", "map specialization relative to the team's overall/pool strength - also a "
          "comparative view, grouped with M rather than K's raw-strength numbers."),
    "O": ("roster_map_familiarity", "current-roster player performance ON the selected map."),
    "P": ("roster_map_familiarity", "current-core lineup continuity ON the selected map."),
}


def build_xgb_feature_groups():
    prep = json.loads(XGB_PREP_PATH.read_text(encoding="utf-8"))
    transformed_names = prep["transformed_feature_names"]

    fam_df = pd.read_csv(XGB_FAMILY_CSV)
    group_df = pd.read_csv(XGB_GROUP_CSV).set_index("family")

    if set(fam_df["feature"]) != set(transformed_names):
        raise ValueError(f"STOP: XGB family table does not exactly set-match the frozen transformed feature "
                          f"list. missing={set(transformed_names) - set(fam_df['feature'])}, "
                          f"extra={set(fam_df['feature']) - set(transformed_names)}")
    if fam_df["feature"].duplicated().any():
        raise ValueError("STOP: XGB family table has a duplicate feature name")
    if len(fam_df) != 131:
        raise ValueError(f"STOP: expected 131 XGB family rows, got {len(fam_df)}")
    if set(fam_df["family"]) != set(FAMILY_TO_PRODUCT_GROUP):
        raise ValueError(f"STOP: family letters in the CSV do not match the frozen collapse mapping. "
                          f"csv={set(fam_df['family'])}, mapping={set(FAMILY_TO_PRODUCT_GROUP)}")

    features = []
    for row in fam_df.itertuples(index=False):
        group, rationale = FAMILY_TO_PRODUCT_GROUP[row.family]
        features.append({
            "model_id": "map_xgboost_v3_final", "transformed_feature": row.feature,
            "family": row.family, "family_label": group_df.loc[row.family, "family_label"],
            "factor_group": group, "factor_group_rationale": rationale,
        })

    product_groups = sorted({v[0] for v in FAMILY_TO_PRODUCT_GROUP.values()})
    return {
        "model_id": "map_xgboost_v3_final", "transformed_feature_count": len(transformed_names),
        "source_family_taxonomy": "reports/tables/map_xgboost_v3_final_feature_importance.csv + "
                                   "reports/tables/map_xgboost_v3_final_group_importance.csv (Phase 6D, "
                                   "reused unmodified)",
        "product_factor_groups": product_groups,
        "family_to_product_group": {k: v[0] for k, v in FAMILY_TO_PRODUCT_GROUP.items()},
        "features": features,
    }


def main():
    rf = build_rf_feature_groups()
    xgb = build_xgb_feature_groups()
    out = {"config_version": "application_explanation_feature_groups_v1",
           "note": "Every transformed feature actually consumed by each frozen model maps to exactly one "
                   "product factor_group - no silently unmapped model features (verified programmatically "
                   "before this file was written, not asserted).",
           "rf_v2": rf, "map_xgboost_v3_final": xgb}
    out_path = CONFIG / "application" / "application_explanation_feature_groups_v1.yaml"
    out_path.write_text(yaml.safe_dump(out, sort_keys=False, default_flow_style=False), encoding="utf-8")
    print(f"wrote {out_path}: RF {len(rf['features'])} features, XGB {len(xgb['features'])} features")
    return out


if __name__ == "__main__":
    main()
