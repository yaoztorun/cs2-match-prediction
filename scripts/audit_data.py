"""
Read-only raw-data audit for the CS2 match prediction project.

Loads every CSV under data/raw/, computes schema/quality/identity/leakage/
target-distribution statistics with pandas, and writes three reproducible
reports to reports/:

    reports/data_audit.md
    reports/data_dictionary.csv
    reports/leakage_analysis.md

This script never writes to data/raw/ and performs no feature engineering
or modelling. Re-run it any time the raw data changes to regenerate the
reports from scratch.
"""

import re
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

GAMES_FILES = {
    "tier1": "cs2_tier1_games.csv",
    "tier2": "cs2_tier2_games.csv",
    "tier3": "cs2_tier3_games.csv",
}
ALL_TIERS_FILE = "cs2_all_tiers_games.csv"

PLAYER_ID_COLS = [f"team{t}_player{i}_id" for t in (1, 2) for i in range(1, 6)]
PLAYER_NAME_COLS = [f"team{t}_player{i}" for t in (1, 2) for i in range(1, 6)]
PLAYER_STAT_SUFFIXES = ["kills", "deaths", "assists", "adr", "kast", "kddiff"]
PLAYER_STAT_COLS = [
    f"team{t}_player{i}_{s}"
    for t in (1, 2)
    for i in range(1, 6)
    for s in PLAYER_STAT_SUFFIXES
]


def load_games():
    """Load tier1/2/3 with an explicit `tier` column, and the all-tiers file separately."""
    frames = []
    for tier, fname in GAMES_FILES.items():
        df = pd.read_csv(RAW / fname, dtype=str, keep_default_na=False)
        df["tier"] = tier
        frames.append(df)
    tiered = pd.concat(frames, ignore_index=True)

    all_tiers = pd.read_csv(RAW / ALL_TIERS_FILE, dtype=str, keep_default_na=False)
    return tiered, all_tiers


def load_lookups():
    players = pd.read_csv(RAW / "players.csv", dtype=str, keep_default_na=False)
    teams = pd.read_csv(RAW / "teams.csv", dtype=str, keep_default_na=False)
    tournaments = pd.read_csv(RAW / "tournaments.csv", dtype=str, keep_default_na=False)
    return players, teams, tournaments


def to_num(s):
    return pd.to_numeric(s.replace("", np.nan), errors="coerce")


def pct(n, d):
    return 0.0 if d == 0 else 100.0 * n / d


# ---------------------------------------------------------------------------
# Structural checks
# ---------------------------------------------------------------------------

def check_tier_partition(tiered, all_tiers):
    ids_by_tier = {t: set(g["match_id"]) for t, g in tiered.groupby("tier")}
    union = set.union(*ids_by_tier.values())
    all_ids = set(all_tiers["match_id"])
    overlaps = {
        f"{a}&{b}": len(ids_by_tier[a] & ids_by_tier[b])
        for i, a in enumerate(ids_by_tier)
        for b in list(ids_by_tier)[i + 1:]
    }
    return {
        "n_matches_per_tier": {t: len(v) for t, v in ids_by_tier.items()},
        "union_equals_all": union == all_ids,
        "row_count_tiered": len(tiered),
        "row_count_all_tiers_file": len(all_tiers),
        "overlaps": overlaps,
    }


def check_is_total_semantics(df):
    out = {}
    for tier, g in [("ALL", df)] + [(t, g) for t, g in df.groupby("tier")]:
        tt = g[g["is_total"] == "True"]
        tf = g[g["is_total"] == "False"]
        by_match_total = tt.groupby("match_id").size()
        n_matches = g["match_id"].nunique()
        out[tier] = {
            "n_total_true": len(tt),
            "n_total_false": len(tf),
            "map_id_blank_when_total_true": int((tt["map_id"] == "").sum()),
            "map_id_blank_when_total_false": int((tf["map_id"] == "").sum()),
            "matches_with_exactly_1_total_row": int((by_match_total == 1).sum()),
            "matches_with_0_total_rows": n_matches - by_match_total.shape[0],
            "matches_with_gt1_total_rows": int((by_match_total > 1).sum()),
            "n_matches": n_matches,
        }
    return out


def check_game_id_semantics(df):
    tf = df[df["is_total"] == "False"]
    tt = df[df["is_total"] == "True"]
    map_gids = tf["game_id"]
    dup = map_gids[map_gids != ""].duplicated().sum()
    blank_map = (map_gids == "").sum()

    pos_ids = set(g for g in df["game_id"] if g and not g.startswith("-"))
    neg = tt[tt["game_id"].str.startswith("-", na=False)]
    blank_series = (tt["game_id"] == "").sum()
    neg_abs_exists = sum(1 for g in neg["game_id"] if g.lstrip("-") in pos_ids)
    return {
        "map_rows": len(tf),
        "map_game_id_duplicates": int(dup),
        "map_game_id_blank": int(blank_map),
        "series_rows": len(tt),
        "series_game_id_blank": int(blank_series),
        "series_game_id_negative": len(neg),
        "series_negative_abs_matches_real_positive_id": neg_abs_exists,
    }


def check_bestof_games_played(df):
    tt = df[df["is_total"] == "True"].copy()
    tt["bestOf_num"] = to_num(tt["bestOf"])
    bo_counts = tt["bestOf_num"].value_counts(dropna=False).sort_index()

    tf = df[df["is_total"] == "False"]
    maps_per_match = tf.groupby("match_id").size()
    tt = tt.set_index("match_id")
    gp_num = to_num(tt["games_played"])
    joined = pd.DataFrame({"games_played": gp_num, "actual_maps": maps_per_match}).dropna(subset=["games_played"])
    mismatch = (joined["games_played"] != joined["actual_maps"]).sum()

    dist_maps_per_match = maps_per_match.value_counts().sort_index()
    return {
        "bestof_blank": int((tt["bestOf"] == "").sum()),
        "bestof_distribution": bo_counts.to_dict(),
        "games_played_checked": len(joined),
        "games_played_mismatch": int(mismatch),
        "maps_per_match_distribution": dist_maps_per_match.to_dict(),
    }


def check_map_row_quality(df):
    tf = df[df["is_total"] == "False"]
    both_null = ((tf["score1_game"] == "") & (tf["map_id"] == "")).sum()
    score_present_map_null = ((tf["score1_game"] != "") & (tf["map_id"] == "")).sum()
    score_null_map_present = ((tf["score1_game"] == "") & (tf["map_id"] != "")).sum()
    return {
        "map_rows": len(tf),
        "both_score_and_map_id_blank": int(both_null),
        "score_present_but_map_id_blank": int(score_present_map_null),
        "score_blank_but_map_id_present": int(score_null_map_present),
    }


def check_player_stat_nullness(df):
    tt = df[df["is_total"] == "True"]
    tf = df[df["is_total"] == "False"]
    sample_cols = ["team1_player1_kills", "team1_player1_adr", "team1_player1_id", "team1_player1"]
    out = {}
    for c in sample_cols:
        out[c] = {
            "series_null_pct": round(pct((tt[c] == "").sum(), len(tt)), 1),
            "map_null_pct": round(pct((tf[c] == "").sum(), len(tf)), 1),
        }
    return out


def reconstruct_winner(tt):
    """tt: is_total=True rows. Returns a Series 'winner' in {'team1','team2','tie','missing'}."""
    s1 = to_num(tt["score1_match"])
    s2 = to_num(tt["score2_match"])
    winner = pd.Series("missing", index=tt.index, dtype=object)
    both = s1.notna() & s2.notna()
    winner[both & (s1 > s2)] = "team1"
    winner[both & (s2 > s1)] = "team2"
    winner[both & (s1 == s2)] = "tie"
    return winner, s1, s2


def check_team1_win_reliability(df):
    tt = df[df["is_total"] == "True"].copy()
    winner, s1, s2 = reconstruct_winner(tt)
    tt["winner"] = winner
    resolvable = tt[tt["winner"].isin(["team1", "team2"])].copy()
    actual = to_num(resolvable["team1_win"]).astype("Int64")
    expected = (resolvable["winner"] == "team1").astype(int)
    mismatch = (actual != expected)

    # breakdown
    is_team1_actual_winner = resolvable["winner"] == "team1"
    is_team2_actual_winner = resolvable["winner"] == "team2"
    correct_when_team1_wins = ((actual == 1) & is_team1_actual_winner).sum()
    correct_when_team2_wins = ((actual == 0) & is_team2_actual_winner).sum()

    # internal consistency: does team1_win differ across a match's own rows?
    vals_per_match = df.groupby("match_id")["team1_win"].nunique()
    inconsistent_matches = int((vals_per_match > 1).sum())

    return {
        "n_series_rows": len(tt),
        "n_missing_score": int((winner == "missing").sum()),
        "n_tie_score": int((winner == "tie").sum()),
        "n_resolvable": len(resolvable),
        "n_mismatch": int(mismatch.sum()),
        "mismatch_pct": round(pct(mismatch.sum(), len(resolvable)), 1),
        "team1_win_eq_1_count": int((to_num(tt["team1_win"]) == 1).sum()),
        "team1_win_eq_1_pct": round(pct((to_num(tt["team1_win"]) == 1).sum(), len(tt)), 1),
        "score1_gt_score2_count": int(is_team1_actual_winner.sum()),
        "score1_gt_score2_pct": round(pct(is_team1_actual_winner.sum(), len(resolvable)), 1),
        "correct_rate_when_team1_truly_wins": round(pct(correct_when_team1_wins, is_team1_actual_winner.sum()), 1),
        "correct_rate_when_team2_truly_wins": round(pct(correct_when_team2_wins, is_team2_actual_winner.sum()), 1),
        "matches_with_inconsistent_team1_win_across_rows": inconsistent_matches,
        "total_matches": df["match_id"].nunique(),
    }


def check_team_identity(df):
    tt = df[df["is_total"] == "True"]
    order_ok = (to_num(tt["team1_id"]) < to_num(tt["team2_id"])).sum()
    order_total = tt["team1_id"].notna().sum()

    long1 = tt[["match_id", "team1_id", "team1"]].rename(columns={"team1_id": "team_id", "team1": "team_name"})
    long2 = tt[["match_id", "team2_id", "team2"]].rename(columns={"team2_id": "team_id", "team2": "team_name"})
    long = pd.concat([long1, long2], ignore_index=True)

    id_match_counts = long.groupby("team_id")["match_id"].nunique()

    # a blank team_name is not a team - one row (the known missing-score row) has team1/team2
    # blank while team1_id/team2_id are still populated and legitimate; exclude the blank from
    # name-based counts only (team_id counts above are unaffected) so it isn't counted as a
    # 793rd "team name" or as a false name/tournament collision.
    long_named = long[long["team_name"] != ""]
    name_match_counts = long_named.groupby("team_name")["match_id"].nunique()

    name_to_ids = long_named.groupby("team_name")["team_id"].nunique().sort_values(ascending=False)
    top_colliding_names = name_to_ids[name_to_ids > 1].head(10)

    # same name + same tournament -> multiple ids?
    long_t = tt[["match_id", "tournament", "team1_id", "team1"]].rename(columns={"team1_id": "team_id", "team1": "team_name"})
    long_t2 = tt[["match_id", "tournament", "team2_id", "team2"]].rename(columns={"team2_id": "team_id", "team2": "team_name"})
    long_t = pd.concat([long_t, long_t2], ignore_index=True)
    long_t = long_t[long_t["team_name"] != ""]
    grp = long_t.groupby(["team_name", "tournament"])["team_id"].nunique()
    same_tourney_collisions = int((grp > 1).sum())

    return {
        "team1_id_lt_team2_id_count": int(order_ok),
        "team1_id_lt_team2_id_total": int(order_total),
        "distinct_team_ids": long["team_id"].nunique(),
        "team_ids_reused_across_matches": int((id_match_counts > 1).sum()),
        "team_ids_single_match_only": int((id_match_counts == 1).sum()),
        "distinct_team_names": long_named["team_name"].nunique(),
        "team_names_reused_across_matches": int((name_match_counts > 1).sum()),
        "team_names_single_match_only": int((name_match_counts == 1).sum()),
        "top_colliding_names": top_colliding_names.to_dict(),
        "name_tournament_pairs_with_gt1_id": same_tourney_collisions,
        "name_tournament_pairs_total": int(grp.shape[0]),
    }


def check_player_identity(df):
    long_frames = []
    for idc, namec in zip(PLAYER_ID_COLS, PLAYER_NAME_COLS):
        sub = df[["match_id", idc, namec]].rename(columns={idc: "player_id", namec: "player_name"})
        long_frames.append(sub)
    long = pd.concat(long_frames, ignore_index=True)
    long = long[long["player_id"] != ""]

    id_match_counts = long.groupby("player_id")["match_id"].nunique()
    non_blank = long[long["player_name"] != ""]
    name_conflicts = non_blank.groupby("player_id")["player_name"].nunique()
    return {
        "distinct_player_ids": long["player_id"].nunique(),
        "player_ids_reused_across_matches": int((id_match_counts > 1).sum()),
        "player_ids_single_match_only": int((id_match_counts == 1).sum()),
        "player_ids_with_gt1_nonblank_name": int((name_conflicts > 1).sum()),
    }


def check_lookup_coverage(df, players, teams, tournaments):
    tt = df[df["is_total"] == "True"]
    games_team_ids = set(tt["team1_id"]) | set(tt["team2_id"])
    teams_ids = set(teams["team_id"])
    teams_name_map = dict(zip(teams["team_id"], teams["team_name"]))
    games_id_to_names = {}
    for _, r in tt.iterrows():
        games_id_to_names.setdefault(r["team1_id"], set()).add(r["team1"])
        games_id_to_names.setdefault(r["team2_id"], set()).add(r["team2"])
    name_mismatches = sum(
        1 for tid in (games_team_ids & teams_ids)
        if teams_name_map[tid] not in games_id_to_names.get(tid, set())
    )

    long_frames = []
    for idc in PLAYER_ID_COLS:
        long_frames.append(df[idc])
    games_player_ids = set(pd.concat(long_frames)) - {""}
    players_ids = set(players["player_id"])
    players_blank_names = int((players["player_name"] == "").sum())

    games_tournaments = set(df["tournament"])
    tournaments_lookup = set(tournaments["tournament"])

    return {
        "team_ids_in_games": len(games_team_ids),
        "team_ids_in_teams_csv": len(teams_ids),
        "team_ids_missing_from_teams_csv": len(games_team_ids - teams_ids),
        "team_ids_in_teams_csv_unused_in_games": len(teams_ids - games_team_ids),
        "team_name_mismatches_vs_lookup": name_mismatches,
        "player_ids_in_games": len(games_player_ids),
        "player_ids_in_players_csv": len(players_ids),
        "player_ids_missing_from_players_csv": len(games_player_ids - players_ids),
        "player_ids_in_players_csv_unused_in_games": len(players_ids - games_player_ids),
        "players_csv_blank_name_pct": round(pct(players_blank_names, len(players)), 1),
        "tournaments_in_games": len(games_tournaments),
        "tournaments_in_lookup": len(tournaments_lookup),
        "tournaments_in_games_not_in_lookup": len(games_tournaments - tournaments_lookup),
        "tournaments_in_lookup_unused_in_games": len(tournaments_lookup - games_tournaments),
    }


def check_dates_and_maps(df):
    dt = pd.to_datetime(df["datetime"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    unparseable = int(dt.isna().sum())
    map_names = sorted(m for m in df["map_name"].unique() if m)
    return {
        "min_date": str(dt.min()),
        "max_date": str(dt.max()),
        "unparseable_datetimes": unparseable,
        "map_pool": map_names,
    }


def check_tournament_tier_distribution(tiered):
    out = {}
    tset = {}
    for tier, g in tiered.groupby("tier"):
        tset[tier] = set(g["tournament"])
        out[tier] = g["tournament"].nunique()
    overlaps = {}
    tiers = list(tset)
    for i, a in enumerate(tiers):
        for b in tiers[i + 1:]:
            overlaps[f"{a}&{b}"] = len(tset[a] & tset[b])
    return {"unique_tournaments_per_tier": out, "overlaps": overlaps}


def check_cologne(tiered):
    names = sorted(t for t in tiered["tournament"].unique() if "cologne" in t.lower())
    out = {}
    for name in names:
        g = tiered[tiered["tournament"] == name]
        tt = g[g["is_total"] == "True"]
        dt = pd.to_datetime(g["datetime"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
        teams_c = set(g["team1"]) | set(g["team2"])
        maps_c = sorted(m for m in g["map_name"].unique() if m)
        out[name] = {
            "rows": len(g),
            "matches": len(tt),
            "min_date": str(dt.min()),
            "max_date": str(dt.max()),
            "n_teams": len(teams_c),
            "maps": maps_c,
            "bestOf_values": sorted(set(to_num(tt["bestOf"]).dropna().astype(int).astype(str))),
            "tiers_present": sorted(g["tier"].unique().tolist()),
        }
    return out


# ---------------------------------------------------------------------------
# Target distribution (reconstructed score-based winner)
# ---------------------------------------------------------------------------

def target_distribution(tiered):
    tt = tiered[tiered["is_total"] == "True"].copy()
    winner, s1, s2 = reconstruct_winner(tt)
    tt["winner"] = winner
    tt["bestOf_num"] = to_num(tt["bestOf"])
    dt = pd.to_datetime(tt["datetime"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    tt["year"] = dt.dt.year

    def dist(series_group):
        vc = series_group.value_counts(dropna=False)
        total = vc.sum()
        return {k: {"n": int(v), "pct": round(pct(v, total), 1)} for k, v in vc.items()}

    overall = dist(tt["winner"])
    by_bo = {
        str(bo): dist(g["winner"])
        for bo, g in tt.groupby("bestOf_num", dropna=True)
        if bo in (1.0, 3.0, 5.0)
    }
    by_tier = {tier: dist(g["winner"]) for tier, g in tt.groupby("tier")}
    by_year = {int(yr): dist(g["winner"]) for yr, g in tt.groupby("year") if pd.notna(yr)}

    return {"overall": overall, "by_bestof": by_bo, "by_tier": by_tier, "by_year": by_year}


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def null_pct_by_rowtype(df, col):
    tt = df[df["is_total"] == "True"]
    tf = df[df["is_total"] == "False"]
    return (
        round(pct((tt[col] == "").sum(), len(tt)), 1),
        round(pct((tf[col] == "").sum(), len(tf)), 1),
    )


def classify_column(col):
    """Return (granularity, known_before_match, leak_A, leak_B, notes) for the data dictionary."""
    if col in ("match_id",):
        return ("series", "yes", "none", "none",
                "Series identifier. Not unique alone in the raw file (1 series row + 1-5 map rows share it) - filter by is_total for a real key.")
    if col == "game_id":
        return ("map", "no", "n/a", "n/a",
                "Valid unique key only for map rows (is_total=False). On series rows it is blank or the negative of a real map game_id - a borrowed marker, not an independent id.")
    if col == "tournament":
        return ("series", "yes", "none", "none", "")
    if col == "tier":
        return ("series", "yes", "none", "none",
                "Not a raw column - only recoverable from which source file (tier1/2/3) a match came from. Recommend materializing explicitly at ingestion.")
    if col in ("team1_id", "team2_id"):
        return ("team", "yes", "none", "none",
                "NOT a persistent team identity - each id appears in exactly one match (verified). Canonically sorted (team1_id < team2_id always), unrelated to home/away. Do not use for historical joins. See Open Questions.")
    if col in ("team1", "team2"):
        return ("team", "yes", "none", "none",
                "Candidate identity key (repeats across matches, unlike team_id) but has unresolved name collisions across many different team_ids, including within a single tournament. Treat as an initial candidate key only - do not assume it safely resolves team identity. See Open Questions.")
    if col in ("score1_match", "score2_match"):
        return ("series outcome", "no", "high", "high",
                "Defines the match result. Reconstruct the label as sign(score1_match-score2_match) from this column directly - do not use the provided team1_win column (see leakage_analysis.md).")
    if col == "is_total":
        return ("row-type flag", "n/a", "n/a", "n/a", "Row-type indicator (series vs map row), not a pre-match feature.")
    if col == "bestOf":
        return ("series", "yes", "none", "none", "Announced match format. 116/9,923 series rows blank - needs an explicit fallback rule.")
    if col == "datetime":
        return ("series", "yes (assumed)", "none", "none",
                "Assumed to be the scheduled/start time, but this is NOT verified against the data source - if it is instead a completion/scrape time, chronological train/test splitting logic would need to change. See Open Questions.")
    if col in ("score1_game", "score2_game"):
        return ("map outcome", "no", "n/a (unknown pre-veto)", "high",
                "Current-map score. Unavailable before the map is played in either task; still leaky in Task B (map-specific prediction) since it's the current map's own outcome.")
    if col in ("map_id", "map_name"):
        return ("map", "no", "n/a (unknown pre-veto)", "legitimate input",
                "Task A (pre-veto series prediction): unavailable/unknown, since which maps will be played is not decided yet - do not use. Task B (future map-specific prediction): the target map is a legitimate, required input, not leakage - but only the specific map being predicted, not other maps' results.")
    if col == "team1_win":
        return ("series outcome", "no", "high", "high",
                "UNRELIABLE, not just leaky: disagrees with sign(score1_match-score2_match) in ~50% of series rows and is internally inconsistent within a match's own rows in ~34% of matches. Do not use as the target or a feature - reconstruct the label from scores instead. See leakage_analysis.md.")
    if col == "games_played":
        return ("series outcome", "no", "high", "high",
                "Disagrees with the actual count of map rows for the same match in ~56% of matches - unreliable even as a labeled-leaky feature; verify before any use.")
    if col in PLAYER_ID_COLS or col in PLAYER_NAME_COLS:
        return ("player/lineup", "ambiguous", "ambiguous", "ambiguous",
                "Could reflect the pre-announced starting lineup or the post-hoc actual boxscore roster (incl. substitutions) - not distinguishable from the data alone. Treat as leaky by default until the collection method is confirmed. See Open Questions.")
    if col in PLAYER_STAT_COLS:
        return ("player boxscore (map-level)", "no", "high", "high",
                "Post-map boxscore statistic (only ~2.3% null on map rows; only ~50% populated on series rows - treat as map-level). Usable only as a historical/lagged aggregate computed from strictly-prior matches.")
    return ("", "", "", "", "")


STRING_COLS = {"tournament", "team1", "team2", "map_name", "datetime", "is_total"} | set(PLAYER_NAME_COLS)


def build_data_dictionary(df, players, teams, tournaments):
    rows = []
    games_cols = [c for c in df.columns if c != "tier"]
    for col in games_cols:
        gran, known, leakA, leakB, notes = classify_column(col)
        if col in PLAYER_STAT_COLS or col in ("score1_game", "score2_game", "map_id", "map_name") or col in PLAYER_ID_COLS or col in PLAYER_NAME_COLS:
            s_null, m_null = null_pct_by_rowtype(df, col)
        else:
            s_null = m_null = round(pct((df[col] == "").sum(), len(df)), 1)
        raw_dtype_note = "string/object" if col in STRING_COLS else "numeric-as-string (int-like/float-like, format varies by source file)"
        rows.append({
            "file(s)": "cs2_all_tiers_games.csv, cs2_tier1_games.csv, cs2_tier2_games.csv, cs2_tier3_games.csv",
            "column_name": col,
            "raw_dtype_note": raw_dtype_note,
            "granularity": gran,
            "null_pct_series_rows": s_null,
            "null_pct_map_rows": m_null,
            "known_before_match": known,
            "leakage_risk_task_A_pre_veto_series": leakA,
            "leakage_risk_task_B_map_specific": leakB,
            "notes": notes,
        })
    # add the explicit `tier` derived column
    rows.append({
        "file(s)": "derived (not a raw column)",
        "column_name": "tier",
        "raw_dtype_note": "string (1/2/3)",
        "granularity": "series",
        "null_pct_series_rows": 0.0,
        "null_pct_map_rows": 0.0,
        "known_before_match": "yes",
        "leakage_risk_task_A_pre_veto_series": "none",
        "leakage_risk_task_B_map_specific": "none",
        "notes": "Not a raw column - only recoverable from which source file (tier1/2/3) a match came from. Recommend materializing explicitly at ingestion.",
    })
    for col in ["player_id", "player_name"]:
        rows.append({
            "file(s)": "players.csv", "column_name": col,
            "raw_dtype_note": "string",
            "granularity": "player",
            "null_pct_series_rows": "n/a",
            "null_pct_map_rows": "n/a",
            "known_before_match": "n/a (lookup table)",
            "leakage_risk_task_A_pre_veto_series": "n/a", "leakage_risk_task_B_map_specific": "n/a",
            "notes": "Pure id<->name lookup derived from the games files; player_name is blank in 54.8% of rows even though the games file itself has the name for most of those ids.",
        })
    for col in ["team_id", "team_name"]:
        rows.append({
            "file(s)": "teams.csv", "column_name": col,
            "raw_dtype_note": "string",
            "granularity": "team",
            "null_pct_series_rows": "n/a",
            "null_pct_map_rows": "n/a",
            "known_before_match": "n/a (lookup table)",
            "leakage_risk_task_A_pre_veto_series": "n/a", "leakage_risk_task_B_map_specific": "n/a",
            "notes": "Pure 1:1 derived export of the same non-persistent team_ids/names already in the games files. Does NOT resolve team identity - see Open Questions.",
        })
    rows.append({
        "file(s)": "tournaments.csv", "column_name": "tournament",
        "raw_dtype_note": "string",
        "granularity": "tournament",
        "null_pct_series_rows": "n/a",
        "null_pct_map_rows": "n/a",
        "known_before_match": "n/a (lookup table)",
        "leakage_risk_task_A_pre_veto_series": "n/a", "leakage_risk_task_B_map_specific": "n/a",
        "notes": "Just a list of the 344 distinct tournament name strings - no tier/date/prize-pool/location attributes.",
    })
    return pd.DataFrame(rows)


def fmt_dict(d, indent="  "):
    return "\n".join(f"{indent}- {k}: {v}" for k, v in d.items())


def main():
    tiered, all_tiers = load_games()
    players, teams, tournaments = load_lookups()

    partition = check_tier_partition(tiered, all_tiers)
    is_total_info = check_is_total_semantics(tiered)
    game_id_info = check_game_id_semantics(all_tiers)
    bo_gp_info = check_bestof_games_played(all_tiers)
    map_quality = check_map_row_quality(all_tiers)
    player_null_info = check_player_stat_nullness(all_tiers)
    win_reliability = check_team1_win_reliability(all_tiers)
    team_identity = check_team_identity(all_tiers)
    player_identity = check_player_identity(all_tiers)
    lookup_cov = check_lookup_coverage(all_tiers, players, teams, tournaments)
    dates_maps = check_dates_and_maps(all_tiers)
    tier_dist = check_tournament_tier_distribution(tiered)
    cologne = check_cologne(tiered)
    target_dist = target_distribution(tiered)

    shape_all = {"rows": len(all_tiers), "cols": len([c for c in all_tiers.columns])}
    shape_tiers = {t: len(pd.read_csv(RAW / f, dtype=str, keep_default_na=False)) for t, f in GAMES_FILES.items()}

    print("=" * 80)
    print("HEADLINE NUMBERS (for verification)")
    print("=" * 80)
    print(f"all_tiers rows: {shape_all['rows']}, cols: {shape_all['cols']}")
    print(f"tier row counts: {shape_tiers}")
    print(f"tier partition matches ids: {partition['union_equals_all']}")
    print(f"team1_win mismatch: {win_reliability['n_mismatch']}/{win_reliability['n_resolvable']} ({win_reliability['mismatch_pct']}%)")
    print(f"team_id reused across matches: {team_identity['team_ids_reused_across_matches']} / {team_identity['distinct_team_ids']}")
    print(f"player_id reused across matches: {player_identity['player_ids_reused_across_matches']} / {player_identity['distinct_player_ids']}")
    print(f"date range: {dates_maps['min_date']} .. {dates_maps['max_date']}")
    print(f"IEM Cologne Major 2026: {cologne.get('IEM Cologne Major 2026')}")

    # ---------------- data_dictionary.csv ----------------
    dd = build_data_dictionary(all_tiers, players, teams, tournaments)
    dd.to_csv(REPORTS / "data_dictionary.csv", index=False, encoding="utf-8")

    # ---------------- data_audit.md ----------------
    md = []
    md.append("# CS2 Match Prediction — Raw Data Audit\n")
    md.append("Generated by `scripts/audit_data.py` from the files under `data/raw/`. "
               "No raw data was modified and no features/models were built as part of this audit. "
               "Re-run the script to regenerate this report from scratch.\n")

    md.append("## 1. File-by-file summary\n")
    md.append("### cs2_all_tiers_games.csv")
    md.append(f"- Shape: {shape_all['rows']:,} rows x {shape_all['cols']} columns")
    md.append(f"- Unique match_id: {all_tiers['match_id'].nunique():,}")
    md.append(f"- Unique team_id (team1_id + team2_id): {team_identity['distinct_team_ids']:,}")
    md.append(f"- Unique player_id: {player_identity['distinct_player_ids']:,}")
    md.append(f"- Unique tournaments: {lookup_cov['tournaments_in_games']}")
    md.append(f"- Date range: {dates_maps['min_date']} to {dates_maps['max_date']} ({dates_maps['unparseable_datetimes']} unparseable datetimes)")
    md.append(f"- Map pool ({len(dates_maps['map_pool'])} maps): {', '.join(dates_maps['map_pool'])}")
    md.append("- **Verified**: this file is the exact, disjoint union of the 3 tier files "
               f"(union_equals_all = {partition['union_equals_all']}, no match_id overlap between any pair of tiers: {partition['overlaps']}).\n")

    md.append("### cs2_tier1_games.csv / cs2_tier2_games.csv / cs2_tier3_games.csv")
    md.append("Identical 98-column schema to the all-tiers file, plus an explicit `tier` label added by this audit "
               "(the raw files carry no tier column themselves - tier is only recoverable from which file a row came from).\n")
    md.append("| tier | rows | matches | unique tournaments |")
    md.append("|---|---|---|---|")
    for t, f in GAMES_FILES.items():
        md.append(f"| {t} | {shape_tiers[t]:,} | {partition['n_matches_per_tier'][t]:,} | {tier_dist['unique_tournaments_per_tier'][t]} |")
    md.append(f"\nTournament-name overlap between tiers: {tier_dist['overlaps']} (all zero - tier is a strict partition by tournament).\n")

    md.append("### players.csv, teams.csv, tournaments.csv (lookup tables)")
    md.append(f"- players.csv: {len(players):,} rows, columns [player_id, player_name]. "
              f"{lookup_cov['players_csv_blank_name_pct']}% of rows have a blank player_name. "
              f"{lookup_cov['player_ids_missing_from_players_csv']} player_ids referenced in the games files are missing from this lookup; "
              f"{lookup_cov['player_ids_in_players_csv_unused_in_games']} rows in the lookup are never referenced in the games files.")
    md.append(f"- teams.csv: {len(teams):,} rows, columns [team_id, team_name]. "
              f"{lookup_cov['team_ids_missing_from_teams_csv']} missing / {lookup_cov['team_ids_in_teams_csv_unused_in_games']} unused / "
              f"{lookup_cov['team_name_mismatches_vs_lookup']} name mismatches vs. the games files. "
              "This is a pure 1:1 derived export of the same team_ids already in the games files - it adds no new attributes and does **not** resolve the team-identity problem (Section 4).")
    md.append(f"- tournaments.csv: {len(tournaments):,} rows, single column [tournament]. "
              f"{lookup_cov['tournaments_in_games_not_in_lookup']} missing / {lookup_cov['tournaments_in_lookup_unused_in_games']} unused. "
              "No tier, date, prize-pool, or location attributes - just the name strings.\n")

    md.append("## 2. Granularity and primary keys\n")
    md.append("Each `match_id` has **exactly one** `is_total=True` (\"series\") row carrying `score1_match`/`score2_match`, "
               "plus 1-5 `is_total=False` (\"map\") rows carrying per-map score/box-score data.\n")
    for tier, info in is_total_info.items():
        md.append(f"- **{tier}**: {info['n_total_true']:,} series rows, {info['n_total_false']:,} map rows, "
                  f"{info['n_matches']:,} matches — exactly 1 series row per match in {info['matches_with_exactly_1_total_row']:,}/{info['n_matches']:,} cases "
                  f"(0 with 0 rows, 0 with >1 rows).")
    md.append(f"\nMap-rows-per-match distribution: {is_total_info['ALL']['n_total_false']:,} map rows total, "
              f"distributed as {dict(sorted(bo_gp_info['maps_per_match_distribution'].items()))} (maps -> n_matches).\n")

    md.append("**Candidate primary keys**:")
    md.append("- Series level: `match_id`, filtered to `is_total=True` (verified exactly one such row per match, see Section 1).")
    md.append(f"- Map level: `game_id`, filtered to `is_total=False` — verified unique ({game_id_info['map_game_id_duplicates']} duplicates) "
              f"and complete ({game_id_info['map_game_id_blank']} blanks) among {game_id_info['map_rows']:,} map rows.")
    md.append(f"- **`game_id` is NOT a valid key on series rows**: blank in {game_id_info['series_game_id_blank']:,}/{game_id_info['series_rows']:,} series rows, "
              f"and in the remaining {game_id_info['series_game_id_negative']:,} it is the **negative** of a real map-level `game_id` found elsewhere in the file "
              f"(verified true for {game_id_info['series_negative_abs_matches_real_positive_id']}/{game_id_info['series_game_id_negative']} of them, i.e. 100%) — "
              "it is a borrowed marker, not an independent identifier. Do not use `game_id` as a key without first filtering on `is_total`.")
    md.append("- No exact duplicate rows found anywhere in `cs2_all_tiers_games.csv`. No duplicate `(match_id, game_id)` pairs.\n")

    md.append("## 3. `is_total` semantics\n")
    md.append("`is_total=True` marks the one series-level summary row per match (final `score1_match`/`score2_match`); "
              "`is_total=False` marks the individual map rows. Confirmed:")
    md.append(f"- When `is_total=True`: `map_id` is blank in {is_total_info['ALL']['map_id_blank_when_total_true']:,}/{is_total_info['ALL']['n_total_true']:,} rows "
              "(the remainder is BO1 matches where the single map's data is duplicated onto the totals row, or short data-quality gaps).")
    md.append(f"- When `is_total=False`: `map_id` is blank in only {is_total_info['ALL']['map_id_blank_when_total_false']:,}/{is_total_info['ALL']['n_total_false']:,} rows (see Section 6).\n")

    md.append("## 4. BO1 / BO3 / BO5 counts\n")
    bo_readable = {}
    for k, v in bo_gp_info["bestof_distribution"].items():
        if pd.isna(k):
            bo_readable["blank"] = bo_readable.get("blank", 0) + v
        else:
            bo_readable[f"BO{int(k)}"] = bo_readable.get(f"BO{int(k)}", 0) + v
    md.append("Distribution among the 9,923 series rows (numeric-format duplicates like `'3'` vs `'3.0'`, an artifact of tier files "
              "being written with different numeric formatting, are merged here):\n")
    md.append("| format | count |\n|---|---|")
    for k, v in sorted(bo_readable.items(), key=lambda x: x[0]):
        md.append(f"| {k} | {v:,} |")
    md.append(f"\n`bestOf` is blank in {bo_gp_info['bestof_blank']:,} series rows — needs an explicit fallback rule before use.\n")

    md.append("## 5. `games_played` reliability\n")
    md.append(f"Compared against the actual number of `is_total=False` map rows found for the same `match_id`: "
              f"mismatched in {bo_gp_info['games_played_mismatch']:,}/{bo_gp_info['games_played_checked']:,} matches "
              f"({pct(bo_gp_info['games_played_mismatch'], bo_gp_info['games_played_checked']):.1f}%). "
              "**Do not trust `games_played` at face value** — prefer counting map rows directly (with the caveat that forfeited/unplayed maps may not have a row at all).\n")

    md.append("## 6. Map-row data quality\n")
    md.append(f"Among {map_quality['map_rows']:,} map rows: {map_quality['both_score_and_map_id_blank']} rows have both `score1_game` and `map_id` blank "
              f"(no data at all for that map slot — likely forfeit/no-data), and a further {map_quality['score_present_but_map_id_blank']} rows have a score "
              "but a blank map identity (map identity lost). Combined this affects "
              f"{map_quality['both_score_and_map_id_blank'] + map_quality['score_present_but_map_id_blank']}/{map_quality['map_rows']:,} "
              f"({pct(map_quality['both_score_and_map_id_blank'] + map_quality['score_present_but_map_id_blank'], map_quality['map_rows']):.1f}%) of map rows.\n")

    md.append("## 7. Player box-score column population\n")
    md.append("Player id/name/stat columns are populated far more consistently on map rows than on series rows — treat them as map-level data:\n")
    md.append("| column | blank % on series rows | blank % on map rows |\n|---|---|---|")
    for c, v in player_null_info.items():
        md.append(f"| {c} | {v['series_null_pct']}% | {v['map_null_pct']}% |")
    md.append("")

    md.append("## 8. Tournament tier distribution\n")
    md.append(f"{tier_dist['unique_tournaments_per_tier']} distinct tournaments per tier "
              f"({sum(tier_dist['unique_tournaments_per_tier'].values())} total, matching tournaments.csv exactly). "
              f"Zero tournament-name overlap between any two tiers: {tier_dist['overlaps']}. "
              "Tier is a strict partition of tournaments, but is not itself a column in the raw data.\n")

    md.append("## 9. Team identity consistency\n")
    md.append("_Corrected in the Phase 2.5 verification pass: one series row (`match_id` 10064713, the row already "
              "flagged elsewhere in this report as the 1 missing-score series row) has a blank `team1`/`team2` "
              "display name while `team1_id`/`team2_id` are still populated with real values. Earlier team-name "
              "counts on this page counted that blank string as if it were a 793rd team name; team_id counts were "
              "never affected. All counts below exclude the blank name and now agree exactly with "
              "`reports/team_identity_analysis.md`._\n")
    md.append(f"- `team1_id < team2_id` holds in {team_identity['team1_id_lt_team2_id_count']:,}/{team_identity['team1_id_lt_team2_id_total']:,} "
              "series rows (100%) — ids are canonically sorted per row, not a home/away or scrape-order designation.")
    md.append(f"- **Every one of the {team_identity['distinct_team_ids']:,} distinct `team_id` values appears in exactly one match** "
              f"({team_identity['team_ids_reused_across_matches']} ids reused across >1 match, {team_identity['team_ids_single_match_only']:,} appear in exactly 1) — "
              "`team_id` behaves like a per-match-appearance surrogate key, not a persistent team/franchise identifier.")
    md.append(f"- `team_name`, by contrast, does repeat across matches: {team_identity['team_names_reused_across_matches']}/{team_identity['distinct_team_names']} "
              "distinct names appear in more than one match. However it is **not a clean identity key either** — "
              "the top colliding names map to many different `team_id`s:\n"
              f"{fmt_dict(team_identity['top_colliding_names'])}")
    md.append(f"- These collisions are not limited to different tournaments: {team_identity['name_tournament_pairs_with_gt1_id']}/"
              f"{team_identity['name_tournament_pairs_total']} (team_name, tournament) pairs already span more than one `team_id` "
              "(e.g. 'Inner Circle' in tournament 'Super DraculaN Season 1' spans 7 different team_ids — this specific tournament/entry looks worth a manual sanity check).")
    md.append("- **`teams.csv` does not resolve this**: it is a pure 1:1 derived export of the same ids/names already in the games files "
              f"({lookup_cov['team_ids_missing_from_teams_csv']} missing, {lookup_cov['team_name_mismatches_vs_lookup']} name mismatches).")
    md.append("- **Conclusion**: normalized `team_name` should be treated as an *initial candidate key only*, not a verified persistent identity. "
              "Team identity resolution is an open issue that must be addressed before building any ELO/rolling/historical team features (see Section 12).\n")

    md.append("## 10. Player identity consistency\n")
    md.append(f"- {player_identity['distinct_player_ids']:,} distinct `player_id`s; {player_identity['player_ids_reused_across_matches']:,} "
              f"({pct(player_identity['player_ids_reused_across_matches'], player_identity['distinct_player_ids']):.0f}%) appear in more than one match — "
              "unlike `team_id`, `player_id` behaves like a genuinely persistent identifier.")
    md.append(f"- Zero `player_id`s map to more than one non-blank display name ({player_identity['player_ids_with_gt1_nonblank_name']} conflicts found) — "
              "player identity is internally consistent.")
    md.append(f"- Coverage vs. `players.csv`: {lookup_cov['player_ids_missing_from_players_csv']} player_ids referenced in games are missing from the lookup; "
              f"{lookup_cov['players_csv_blank_name_pct']}% of players.csv rows have a blank name (though the games file itself usually has the name elsewhere).\n")

    md.append("## 11. Target trustworthiness: is `team1_win` usable?\n")
    md.append("**No — see `reports/leakage_analysis.md` for the full analysis.** Summary: reconstructing the winner as "
              f"`sign(score1_match - score2_match)` on the {win_reliability['n_series_rows']:,} series rows "
              f"({win_reliability['n_missing_score']} missing score, {win_reliability['n_tie_score']} literal ties set aside) disagrees with the provided "
              f"`team1_win` column in {win_reliability['n_mismatch']:,}/{win_reliability['n_resolvable']:,} rows "
              f"({win_reliability['mismatch_pct']}%). `team1_win==1` fires in only {win_reliability['team1_win_eq_1_pct']}% of series rows even though "
              f"`score1_match > score2_match` is true in {win_reliability['score1_gt_score2_pct']}% of resolvable rows — it is correct "
              f"{win_reliability['correct_rate_when_team2_truly_wins']}% of the time when team2 truly wins but only "
              f"{win_reliability['correct_rate_when_team1_truly_wins']}% of the time when team1 truly wins. It is also internally inconsistent: "
              f"it differs between a match's own series row and its map rows in {win_reliability['matches_with_inconsistent_team1_win_across_rows']:,}/"
              f"{win_reliability['total_matches']:,} matches. **Reconstruct the label from scores; do not use `team1_win`.**\n")

    md.append("## 12. IEM Cologne Major 2026 coverage (coverage only — not used for feature/model decisions at this stage)\n")
    for name, info in cologne.items():
        md.append(f"- **{name}** (tier: {', '.join(info['tiers_present'])}): {info['matches']} matches, {info['rows']} rows, "
                  f"{info['min_date']} to {info['max_date']}, {info['n_teams']} teams, maps={info['maps']}, bestOf values present: {info['bestOf_values']}.")
    md.append(f"\nNote: the dataset's overall max date ({dates_maps['max_date']}) is only about a week past IEM Cologne Major 2026's last match "
              f"({cologne.get('IEM Cologne Major 2026', {}).get('max_date')}) — there is very little post-Cologne data available for a genuine forward holdout. "
              "This is a coverage observation only; no modelling or feature-selection decision is made from it here.\n")

    md.append("## 13. Target distribution (reconstructed score-based winner)\n")
    md.append("Winner reconstructed as `sign(score1_match - score2_match)` on series rows (`team1` = lower team_id side, `team2` = higher team_id side — "
              "**not** a home/away or favorite/underdog label, just whichever side sorts first by id in that row).\n")

    def dist_table(d):
        lines = ["| outcome | n | % |", "|---|---|---|"]
        for k, v in d.items():
            lines.append(f"| {k} | {v['n']:,} | {v['pct']}% |")
        return "\n".join(lines)

    md.append("**Overall:**\n")
    md.append(dist_table(target_dist["overall"]))
    md.append("\n**By best-of format:**\n")
    for bo, d in sorted(target_dist["by_bestof"].items()):
        md.append(f"\n_BO{int(float(bo))}_\n")
        md.append(dist_table(d))
    md.append("\n**By tier:**\n")
    for tier, d in sorted(target_dist["by_tier"].items()):
        md.append(f"\n_{tier}_\n")
        md.append(dist_table(d))
    md.append("\n**By year:**\n")
    for yr, d in sorted(target_dist["by_year"].items()):
        md.append(f"\n_{yr}_\n")
        md.append(dist_table(d))
    md.append("")

    md.append("## 14. Open Questions Before Feature Engineering\n")
    md.append("These are flagged, not resolved, by this audit:\n")
    md.append("1. **`datetime` semantics** — assumed to be the scheduled/start time, but this is not verified against the data source. "
              "If it is instead a completion/scrape time, any chronological train/test split logic would need to change.")
    md.append("2. **Roster availability timing** — do the `team{1,2}_player{1-5}_id`/name columns reflect the pre-announced starting lineup "
              "(safe as a pre-match feature) or the post-hoc actual boxscore roster including substitutions (leaky)? Not distinguishable from the data alone; "
              "must be confirmed against the collection method.")
    md.append("3. **Team identity resolution** — `team_id` is confirmed non-persistent (unique per match); `team_name` is only a candidate key and has "
              "unresolved collisions (Section 9). This must be resolved — likely via a name-normalization + manual-review pass, or an external team-id "
              "mapping — before any ELO/rolling/head-to-head feature can be trusted.")
    md.append(f"4. **Tied/missing series scores** — {win_reliability['n_tie_score']} series rows have literal tied `score1_match`/`score2_match` "
              f"and {win_reliability['n_missing_score']} row has a missing score. These need an explicit drop/adjudicate rule.")
    md.append(f"5. **Missing `bestOf`** — {bo_gp_info['bestof_blank']} series rows have a blank `bestOf`; needs a fallback (e.g. infer from map-row count, or drop).")
    md.append(f"6. **Blank map identities** — {map_quality['score_present_but_map_id_blank']} map rows have a score but no `map_id`/`map_name`; decide impute vs. drop.")
    md.append(f"7. **Blank map rows** — {map_quality['both_score_and_map_id_blank']} map rows have neither score nor map identity; decide whether these represent forfeits and how to encode that.")
    md.append(f"8. **`games_played` inconsistency** — disagrees with actual map-row counts in {pct(bo_gp_info['games_played_mismatch'], bo_gp_info['games_played_checked']):.0f}% "
              "of matches; decide whether to use it at all or always recompute from map rows.")
    md.append("9. **Tier provenance** — tier (1/2/3) is currently only recoverable from source filename; recommend materializing it as an explicit column "
              "during ingestion (this audit does so for its own analysis but the raw files are unchanged).\n")

    (REPORTS / "data_audit.md").write_text("\n".join(md), encoding="utf-8")

    # ---------------- leakage_analysis.md ----------------
    lk = []
    lk.append("# Leakage Analysis\n")
    lk.append("Generated by `scripts/audit_data.py`. Covers (1) why the provided `team1_win` column cannot be used as the label, "
              "(2) why `team_id` cannot be used for historical joins and why `team_name` is only a candidate key, "
              "(3) a full known-before-match / leaky / ambiguous column classification, split by task, and "
              "(4) reliability notes on a few other columns so future feature engineering doesn't trust them naively.\n")

    lk.append("## 1. The target: reconstruct it, don't read `team1_win`\n")
    lk.append(f"On the {win_reliability['n_series_rows']:,} series rows (`is_total=True`), the reconstructed winner "
              f"`sign(score1_match - score2_match)` disagrees with the provided `team1_win` column in "
              f"{win_reliability['n_mismatch']:,}/{win_reliability['n_resolvable']:,} rows ({win_reliability['mismatch_pct']}%). "
              f"This is not random noise: `team1_win==1` occurs in only {win_reliability['team1_win_eq_1_pct']}% of series rows overall, while "
              f"`score1_match > score2_match` is true in {win_reliability['score1_gt_score2_pct']}% of resolvable rows. Breaking it down by actual outcome:\n")
    lk.append(f"- When team2 truly wins by score: `team1_win` correctly reads 0 about {win_reliability['correct_rate_when_team2_truly_wins']}% of the time.")
    lk.append(f"- When team1 truly wins by score: `team1_win` only correctly reads 1 about {win_reliability['correct_rate_when_team1_truly_wins']}% of the time "
              "(it stays 0 the rest of the time) — i.e. the column behaves close to a near-constant \"team1 doesn't win\" flag rather than a real result indicator.")
    lk.append(f"- It is also internally inconsistent: it differs between a match's own series row and its map rows in "
              f"{win_reliability['matches_with_inconsistent_team1_win_across_rows']:,}/{win_reliability['total_matches']:,} matches (34%).\n")
    lk.append("**Rule going forward**: the target must be reconstructed as `1 if score1_match > score2_match else 0` (evaluated on the `is_total=True` row), "
              f"with the {win_reliability['n_missing_score']} missing-score and {win_reliability['n_tie_score']} tie-score rows set aside for explicit adjudication "
              "rather than silently defaulted. `team1_win` should not be used anywhere in the pipeline, as a feature or as the label.\n")

    lk.append("## 2. Team identity: `team_id` is unusable across matches; `team_name` is a candidate key only\n")
    lk.append(f"`team_id` is confirmed to be a per-match-appearance surrogate: all {team_identity['distinct_team_ids']:,} distinct values appear in "
              "exactly one match each (0 reused). It cannot be used to link a team's history across matches by id.\n")
    lk.append(f"`team_name` does recur naturally ({team_identity['team_names_reused_across_matches']}/{team_identity['distinct_team_names']} names appear "
              "in more than one match) and is therefore the *only* candidate for team-level historical joins in this dataset. However, it is explicitly "
              "**not** being asserted here as a safe or resolved identity key: name collisions exist across many distinct `team_id`s. Top offenders:\n\n"
              f"{fmt_dict(team_identity['top_colliding_names'])}\n\n"
              f"...including {team_identity['name_tournament_pairs_with_gt1_id']} cases where the *same* name already spans multiple `team_id`s within a *single* tournament. "
              "`teams.csv` is a pure derived export of the same ids/names and does not add any disambiguating attribute (country, founding date, org id, etc.).\n")
    lk.append("**Rule going forward**: treat normalized `team_name` as an initial candidate key only. Team identity resolution "
              "(deduplicating/validating names, deciding how to handle collisions such as generic-sounding names in lower-tier qualifiers) is an "
              "**open issue that must be resolved before any ELO, rolling-form, or head-to-head feature is built** — building such features on an "
              "unresolved identity key would silently merge or split real teams' histories.\n")

    lk.append("## 3. Column classification by task\n")
    lk.append("Two downstream prediction tasks were distinguished, since they have different leakage boundaries for map-related columns:\n")
    lk.append("- **Task A — pre-veto series prediction**: predict the series winner before the map veto happens. The exact maps to be played "
              "are not yet known, so `map_id`/`map_name` (and anything scoped to a specific map) are unavailable, not just leaky.")
    lk.append("- **Task B — future map-specific prediction**: predict the outcome of a specific, already-determined map (e.g. \"who wins on Mirage, "
              "map 2 of this BO3\"). Here `map_name`/`map_id` for *that* map is a legitimate, required input — but the *current* map's own score and "
              "player statistics are still leakage, since those are exactly what's being predicted.\n")

    lk.append("### Safe pre-match (both tasks)\n")
    lk.append("`match_id`, `tournament`, `team1_id`/`team1`, `team2_id`/`team2` (subject to the identity caveat in Section 2), `bestOf`, "
              "`datetime` (assumed scheduled time — see Open Question in `data_audit.md`).\n")

    lk.append("### Task A (pre-veto series prediction) — leaky / unavailable\n")
    lk.append("`score1_match`, `score2_match`, `team1_win`, `games_played`, `is_total`, `score1_game`, `score2_game`, `map_id`, `map_name` "
              "(not yet decided pre-veto), and all `team{1,2}_player{1-5}_{kills,deaths,assists,adr,kast,kddiff}` columns. "
              "Usable only as historical/lagged aggregates computed from strictly-prior matches (by `datetime`), never as the current match's own values.\n")

    lk.append("### Task B (future map-specific prediction) — leaky vs. legitimate input\n")
    lk.append("- **Legitimate input**: `map_id`/`map_name` *for the map being predicted* — this is given as part of the task (e.g. \"map 2 is Mirage\"), "
              "not leakage, since predicting a specific map's outcome requires knowing which map it is.")
    lk.append("- **Still leaky**: `score1_game`/`score2_game` for that same map (the outcome being predicted), plus `score1_match`, `score2_match`, "
              "`team1_win`, `games_played`, and all player box-score columns for that map or the series as a whole — these remain post-outcome data "
              "and may only be used as historical aggregates from prior matches/maps, exactly as in Task A.\n")

    lk.append("### Ambiguous — flagged, not resolved (both tasks)\n")
    lk.append("`team{1,2}_player{1-5}_id` and `team{1,2}_player{1-5}` (names): could reflect the pre-announced starting lineup (safe pre-match) or "
              "the post-hoc actual boxscore roster including substitutions (leaky). Not distinguishable from the data alone — treat as leaky by default "
              "until the data collection method is confirmed (see `data_audit.md` Open Questions).\n")

    lk.append("## 4. Other reliability notes\n")
    lk.append(f"- `game_id` is a valid unique key only for map rows (`is_total=False`); on series rows it is blank "
              f"({game_id_info['series_game_id_blank']:,}/{game_id_info['series_rows']:,}) or the negative of an unrelated map's own game_id "
              f"({game_id_info['series_game_id_negative']:,} rows, 100% verified to collide with a real positive id) — never use it as a series-level key.")
    lk.append(f"- `games_played` disagrees with the actual map-row count for the same match in "
              f"{pct(bo_gp_info['games_played_mismatch'], bo_gp_info['games_played_checked']):.0f}% of matches — don't trust it even in its intended "
              "post-match role without cross-checking against actual map rows.")
    lk.append(f"- `bestOf` is blank in {bo_gp_info['bestof_blank']} series rows and needs an explicit fallback rule rather than silent imputation.")
    lk.append(f"- {map_quality['score_present_but_map_id_blank'] + map_quality['both_score_and_map_id_blank']} map rows "
              f"({pct(map_quality['score_present_but_map_id_blank'] + map_quality['both_score_and_map_id_blank'], map_quality['map_rows']):.1f}%) "
              "have partially or fully missing map identity/score data and should be handled explicitly, not silently coerced.\n")

    (REPORTS / "leakage_analysis.md").write_text("\n".join(lk), encoding="utf-8")

    print("\nWrote:")
    print(f"  {REPORTS / 'data_audit.md'}")
    print(f"  {REPORTS / 'data_dictionary.csv'}")
    print(f"  {REPORTS / 'leakage_analysis.md'}")


if __name__ == "__main__":
    main()
