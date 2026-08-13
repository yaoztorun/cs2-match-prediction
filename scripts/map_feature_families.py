"""
[PROJECT ADDITION - Phase 6B]

The feature-family taxonomy from brief section 35, mapped onto the 106
TRANSFORMED columns of the known-map model matrix. Every transformed column
belongs to exactly one family, and the families partition the matrix exactly -
asserted at import-time-of-use, so a future feature addition cannot silently
fall outside the taxonomy.

    A  original series V1 (Phase 3 ELO / win rates / activity)
    B  map-pool depth / order statistics
    C  same-map matchup advantages
    D  map-pool confidence
    E  opponent-strength / residual form
    F  time-decayed form
    G  form confidence
    H  player performance
    I  roster stability
    J  roster / player confidence
    K  MAP-SPECIFIC historical strength  <- the family the phase's scientific
                                            question is about
    L  categorical map/context dummies
"""

FAMILY_LABELS = {
    "A": "original series V1 (ELO / win rate / activity)",
    "B": "map-pool depth and order statistics",
    "C": "same-map matchup advantage",
    "D": "map-pool confidence",
    "E": "opponent-strength / residual form",
    "F": "time-decayed form",
    "G": "form confidence",
    "H": "player performance",
    "I": "roster stability",
    "J": "roster / player confidence",
    "K": "map-specific historical strength (selected map)",
    "L": "categorical map / bestOf / tier context",
}

# Raw (pre-encoding) membership, stated explicitly rather than pattern-matched
# on names - a prefix rule would silently misfile any future feature.
_RAW_FAMILIES = {
    "K": [
        "map_elo_diff", "map_smoothed_win_rate_diff", "map_win_rate_last_5_diff",
        "map_win_rate_last_10_diff", "map_normalized_margin_all_diff",
        "map_normalized_margin_last_5_diff", "map_normalized_margin_last_10_diff",
        "map_matches_before_diff", "days_since_map_played_diff",
        "map_matches_before_min", "map_matches_before_sum", "both_teams_have_map_history",
        "both_teams_have_5_map_matches", "both_teams_have_10_map_matches",
    ],
    "B": [
        "map_pool_size_diff", "map_pool_total_matches_diff", "map_pool_experienced_maps_diff",
        "map_pool_mean_elo_diff", "map_pool_best_elo_diff", "map_pool_second_best_elo_diff",
        "map_pool_third_best_elo_diff", "map_pool_worst_elo_diff", "map_pool_mean_smoothed_wr_diff",
        "map_pool_best_smoothed_wr_diff", "map_pool_second_best_smoothed_wr_diff",
        "map_pool_third_best_smoothed_wr_diff", "map_pool_worst_smoothed_wr_diff",
        "map_pool_mean_normalized_margin_diff",
    ],
    "C": [
        "map_matchup_mean_elo_advantage", "map_matchup_median_elo_advantage",
        "map_matchup_midrange_elo_advantage", "map_matchup_positive_advantage_balance",
        "map_matchup_mean_smoothed_wr_advantage", "map_matchup_median_smoothed_wr_advantage",
    ],
    "D": [
        "map_pool_size_min", "map_pool_total_matches_min", "both_teams_have_map_pool_history",
        "both_teams_have_3_recent_maps", "both_teams_have_5_experienced_maps", "union_map_count",
        "shared_recent_map_count", "shared_experienced_map_count", "map_matchup_shared_coverage",
        "map_matchup_elo_advantage_range",
    ],
    "E": [
        "avg_opponent_elo_last_5_diff", "avg_opponent_elo_last_10_diff",
        "performance_residual_last_5_diff", "performance_residual_last_10_diff",
        "performance_residual_all_diff",
    ],
    "F": [
        "time_weighted_win_rate_diff", "time_weighted_performance_residual_diff",
        "time_weighted_series_margin_diff",
    ],
    "G": [
        "opponent_adjusted_history_min", "both_teams_have_5_adjusted_matches",
        "both_teams_have_10_adjusted_matches", "time_weighted_history_mass_min",
    ],
    "H": [
        "roster_mean_adr_diff", "roster_top_adr_diff", "roster_bottom_adr_diff",
        "roster_mean_kast_diff", "roster_top_kast_diff", "roster_bottom_kast_diff",
        "roster_mean_kd_balance_diff", "roster_top_kd_balance_diff", "roster_bottom_kd_balance_diff",
        "roster_mean_assists_per_round_diff",
    ],
    "I": [
        "recent_unique_players_10_maps_diff", "recent_unique_players_20_maps_diff",
        "core5_appearance_concentration_90d_diff", "core5_continuity_last_10_diff",
    ],
    "J": [
        "roster_mean_player_history_mass_diff", "roster_size_min", "both_teams_have_5_inferred_players",
        "roster_min_player_history_mass", "roster_core_concentration_min",
        "roster_core_continuity_last10_min", "roster_form_players_min",
    ],
    "A": [
        "elo_diff", "overall_win_rate_diff", "win_rate_last_5_diff", "win_rate_last_10_diff",
        "format_win_rate_diff", "avg_series_margin_last_5_diff", "avg_series_margin_last_10_diff",
        "matches_last_30_days_diff", "days_since_last_match_diff", "total_matches_before_diff",
        "history_matches_min", "history_matches_sum", "both_teams_have_history",
        "both_teams_have_5_matches", "both_teams_have_10_matches",
    ],
}


def feature_family_map(roles, transformed_feature_names):
    """Returns {"groups": {family: [transformed columns]},
                "by_feature": {transformed column: family}}.

    Every one-hot dummy (map_name_*, bestOf_BO*, tier_*) lands in family L. The
    partition is asserted exact against both the config's 92 numeric predictive
    inputs and the full transformed column list."""
    raw_declared = [c for cols in _RAW_FAMILIES.values() for c in cols]
    assert len(raw_declared) == len(set(raw_declared)), "a raw feature was assigned to two families"
    expected_raw = set(roles["directional"]) | set(roles["symmetric"])
    assert set(raw_declared) == expected_raw, (
        "family taxonomy disagrees with the config: "
        f"missing={sorted(expected_raw - set(raw_declared))} extra={sorted(set(raw_declared) - expected_raw)}")

    by_feature, groups = {}, {k: [] for k in FAMILY_LABELS}
    raw_to_family = {c: fam for fam, cols in _RAW_FAMILIES.items() for c in cols}
    for name in transformed_feature_names:
        fam = raw_to_family.get(name)
        if fam is None:
            assert name.startswith(("map_name_", "bestOf_BO", "tier_")), \
                f"transformed column outside the family taxonomy: {name}"
            fam = "L"
        by_feature[name] = fam
        groups[fam].append(name)

    assert sum(len(v) for v in groups.values()) == len(transformed_feature_names)
    return {"groups": groups, "by_feature": by_feature}
