"""
Phase 8E reconciliation (Gate 1) + canonical actual-results builder.

Reconciles the 107 cologne_2026-tagged series_base.parquet rows against the
frozen Phase 8B 32-team roster and independently-sourced Liquipedia
round-by-round structure, producing:

  - data/evaluation/cologne_2026_result_reconciliation_v1.csv   (all 107 rows)
  - data/evaluation/cologne_2026_actual_series_results_v1.parquet (106 official rows)

Winner is ALWAYS derived from score1_match > score2_match (never the broken
team1_series_win field). Stage is assigned from independently-verified
calendar-date windows (Phase 8B pre-event structure + Liquipedia raw
wikitext cross-check, see reports/phase8e_cologne_simulation_vs_reality.md
section B). Swiss round_number and record_group are derived by replaying
real chronological match order per stage - never by asking the Phase 8C
engine "where would this fit" (amendment #3: avoids circularity, since the
engine independently re-derives the same structure in Gate 2 from winners
alone, and is compared against THIS table, not the source of it).
"""

import json

import pandas as pd
import yaml

from _common import INTERIM, ROOT

TOURNAMENT_YAML = ROOT / "config" / "tournaments" / "iem_cologne_major_2026_pre_event.yaml"

STAGE_WINDOWS = [
    ("stage_1", "2026-06-02", "2026-06-05"),
    ("stage_2", "2026-06-06", "2026-06-09"),
    ("stage_3", "2026-06-11", "2026-06-15"),
    ("playoffs", "2026-06-18", "2026-06-21"),
]
PLAYOFF_ROUND_LABEL = {1: "quarterfinal", 2: "semifinal", 3: "grand_final"}


def load_canonical_roster():
    cfg = yaml.safe_load(TOURNAMENT_YAML.read_bytes())
    p = cfg["participants"]
    roster = {}
    for group, stage in [("stage_1_entrants", "stage_1"), ("stage_2_direct_entrants", "stage_2"),
                          ("stage_3_direct_entrants", "stage_3")]:
        for t in p[group]:
            roster[t["canonical_model_name"]] = {"starting_stage": stage,
                                                   "pre_event_seed": t["pre_event_seed"],
                                                   "display_name": t["display_name"]}
    if len(roster) != 32:
        raise ValueError(f"expected 32 canonical Cologne teams from Phase 8B YAML, got {len(roster)}")
    return roster


def load_raw_cologne_rows():
    em = pd.read_csv(INTERIM / "evaluation_manifest.csv")
    cologne_ids = set(em.loc[em["evaluation_group"] == "cologne_2026", "match_id"])
    sb = pd.read_parquet(INTERIM / "series_base.parquet", engine="fastparquet")
    col = sb[sb["match_id"].isin(cologne_ids)].sort_values(["datetime", "match_id"]).reset_index(drop=True)
    if len(col) != 107:
        raise ValueError(f"expected 107 cologne_2026-tagged rows, got {len(col)}")
    return col


def derive_winner(row):
    s1, s2 = row["score1_match"], row["score2_match"]
    if pd.isna(s1) or pd.isna(s2):
        raise ValueError(f"match_id={row['match_id']}: missing score")
    if s1 == s2:
        raise ValueError(f"match_id={row['match_id']}: tied score {s1}-{s2}")
    return row["team1_canonical"] if s1 > s2 else row["team2_canonical"]


def assign_stage(dt):
    d = dt.date().isoformat()
    for stage, start, end in STAGE_WINDOWS:
        if start <= d <= end:
            return stage
    return None


def build_reconciliation_rows(raw, roster):
    roster_ids = set(roster.keys())
    rows = []
    for _, r in raw.iterrows():
        t1, t2 = r["team1_canonical"], r["team2_canonical"]
        both_rostered = (t1 in roster_ids) and (t2 in roster_ids)
        winner = derive_winner(r)
        loser = t2 if winner == t1 else t1
        stage_guess = assign_stage(r["datetime"])
        base = {
            "source_match_id": int(r["match_id"]),
            "datetime": r["datetime"].isoformat(),
            "team1": t1, "team2": t2, "best_of": int(r["bestOf"]),
            "score1_match": int(r["score1_match"]), "score2_match": int(r["score2_match"]),
            "derived_winner": winner, "derived_loser": loser,
            "tournament_name": r["tournament"],
        }
        if not both_rostered:
            unrostered = [t for t in (t1, t2) if t not in roster_ids]
            base.update({
                "included_in_official_event": False,
                "candidate_stage": None, "candidate_round": None, "record_group": None,
                "reconciliation_status": "non_tournament_showmatch",
                "reconciliation_reason": (
                    f"{unrostered} not present in the frozen Phase 8B 32-team Cologne roster "
                    "(config/tournaments/iem_cologne_major_2026_pre_event.yaml participants). "
                    "Independently identified as a national all-star exhibition ('Team Germany vs. "
                    "Team Poland Showmatch') played 2026-06-21, the same day as the Grand Final, "
                    "unrelated to official Swiss/playoff standings."
                ),
                "supporting_evidence": (
                    "HLTV match page titled 'Team Germany vs. Team Poland at Showmatch CS' "
                    "(https://www.hltv.org/matches/2395350/team-germany-vs-team-poland-showmatch-cs); "
                    "corroborated by press coverage (cryptobriefing.com) explicitly describing it as a "
                    "national CS2 showmatch distinct from the main Major bracket."
                ),
                "confidence": "high",
            })
        else:
            base.update({
                "included_in_official_event": True,
                "candidate_stage": stage_guess,
                "candidate_round": None,   # filled in below, after chronological replay
                "record_group": None,      # filled in below (swiss only)
                "reconciliation_status": "official_tournament_match",
                "reconciliation_reason": (
                    f"both teams present in the frozen Phase 8B 32-team roster; datetime "
                    f"{r['datetime'].date().isoformat()} falls inside the {stage_guess} window "
                    "independently corroborated by Liquipedia's raw per-round wikitext."
                ),
                "supporting_evidence": "Liquipedia raw wikitext (Stage_1/Stage_2/Stage_3/Playoffs, "
                                        "action=raw) team-pairing cross-check; see result source manifest.",
                "confidence": "high",
            })
        rows.append(base)
    return rows


def assign_rounds_and_records(rows):
    """Chronological per-stage replay: round_number = 1 + prior matches played
    by both paired teams this stage (must agree - asserted). Swiss record_group
    is the shared pre-match (wins, losses) of both teams, real winners only."""
    by_stage = {}
    for row in rows:
        if not row["included_in_official_event"]:
            continue
        by_stage.setdefault(row["candidate_stage"], []).append(row)

    for stage, stage_rows in by_stage.items():
        stage_rows.sort(key=lambda r: (r["datetime"], r["source_match_id"]))
        games_played, wins, losses = {}, {}, {}
        for row in stage_rows:
            t1, t2 = row["team1"], row["team2"]
            g1, g2 = games_played.get(t1, 0), games_played.get(t2, 0)
            if g1 != g2:
                raise ValueError(f"round-derivation mismatch in {stage}: {t1} played {g1} prior matches, "
                                  f"{t2} played {g2} (match_id={row['source_match_id']})")
            round_number = g1 + 1
            if stage == "playoffs":
                # round_number stays an int (1/2/3), matching tournament_engine.py's own
                # MatchSpec.round_number convention for playoffs exactly - not a string label.
                row["candidate_round"] = round_number
                row["playoff_round_label"] = PLAYOFF_ROUND_LABEL.get(round_number, f"round_{round_number}")
                row["record_group"] = "playoffs"
            else:
                w1, l1 = wins.get(t1, 0), losses.get(t1, 0)
                w2, l2 = wins.get(t2, 0), losses.get(t2, 0)
                if (w1, l1) != (w2, l2):
                    raise ValueError(f"record-group mismatch in {stage} round {round_number}: "
                                      f"{t1}={w1}-{l1} vs {t2}={w2}-{l2} (match_id={row['source_match_id']})")
                row["candidate_round"] = round_number
                row["record_group"] = f"{w1}-{l1}"
                winner, loser = row["derived_winner"], row["derived_loser"]
                wins[winner] = wins.get(winner, 0) + 1
                losses[loser] = losses.get(loser, 0) + 1
            games_played[t1] = g1 + 1
            games_played[t2] = g2 + 1
    return rows


def main():
    roster = load_canonical_roster()
    raw = load_raw_cologne_rows()
    rows = build_reconciliation_rows(raw, roster)
    rows = assign_rounds_and_records(rows)

    recon_df = pd.DataFrame(rows)
    n_official = int(recon_df["included_in_official_event"].sum())
    n_excluded = len(recon_df) - n_official
    print(f"reconciliation: {len(recon_df)} raw rows -> {n_official} official, {n_excluded} excluded")
    if n_official != 106:
        raise ValueError(f"STOP: expected exactly 106 official rows after reconciliation, got {n_official}")

    per_stage_counts = recon_df[recon_df["included_in_official_event"]]["candidate_stage"].value_counts().to_dict()
    expected_counts = {"stage_1": 33, "stage_2": 33, "stage_3": 33, "playoffs": 7}
    for stage, expected in expected_counts.items():
        actual = per_stage_counts.get(stage, 0)
        if actual != expected:
            raise ValueError(f"STOP: {stage} has {actual} official matches, expected {expected}")
        print(f"  {stage}: {actual}/{expected}")

    recon_out = ROOT / "data" / "evaluation" / "cologne_2026_result_reconciliation_v1.csv"
    recon_df.to_csv(recon_out, index=False)
    print(f"wrote {recon_out}")

    official = recon_df[recon_df["included_in_official_event"]].copy()
    canonical = pd.DataFrame({
        "event_id": "iem_cologne_major_2026",
        "source_match_id": official["source_match_id"],
        "stage": official["candidate_stage"],
        "round_number": official["candidate_round"],
        "record_group": official["record_group"],
        "team_1": official["team1"], "team_2": official["team2"],
        "best_of": official["best_of"],
        "score_team_1": official["score1_match"], "score_team_2": official["score2_match"],
        "winner": official["derived_winner"], "loser": official["derived_loser"],
        "datetime_source": official["datetime"],
        "result_source": "dataset(kaggle_ektarr) scores, cross-validated vs Liquipedia raw wikitext "
                          "structure and independent press coverage (see result source manifest)",
    }).sort_values(["stage", "datetime_source"]).reset_index(drop=True)

    if canonical["source_match_id"].duplicated().any():
        raise ValueError("STOP: duplicate source_match_id in canonical actual results")
    if len(canonical) != 106:
        raise ValueError(f"STOP: canonical table has {len(canonical)} rows, expected 106")

    canon_out = ROOT / "data" / "evaluation" / "cologne_2026_actual_series_results_v1.parquet"
    canonical.to_parquet(canon_out, engine="fastparquet", index=False)
    print(f"wrote {canon_out}")
    return recon_df, canonical


if __name__ == "__main__":
    main()
