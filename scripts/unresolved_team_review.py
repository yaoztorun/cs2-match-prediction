"""
Phase 2.5 item 3: deep-dive review of the 25 `unresolved` team names from
data/interim/team_aliases.csv.

For each name: appearance timeline, tournaments, chronological roster-change
events (via persistent player_ids), every close-in-time zero-overlap event
with full dates/rosters, an evidence-based assessment, and a PROPOSED (not
applied) decision. No identity is split or merged here - this only writes
reports/unresolved_team_review.md.

Read-only against data/raw/ and data/interim/team_aliases.csv.
"""

import re

import pandas as pd

from _common import load_games_tiered, INTERIM, REPORTS

CLOSE_DAYS = 60
PID_T1 = [f"team1_player{i}_id" for i in range(1, 6)]
PID_T2 = [f"team2_player{i}_id" for i in range(1, 6)]
GENERIC_PATTERN = re.compile(r"^(mix\d*|tbd|team \d+[a-z]*|unknown|amateur team|qualifier)$", re.I)

ESTABLISHED_MATCH_THRESHOLD = 50  # matches at/above this are treated as a recurring, recognizable org


def roster(row, side):
    cols = PID_T1 if side == 1 else PID_T2
    return frozenset(x for x in (row[c] for c in cols) if x)


def build_appearances(tt, name):
    rows = []
    for _, r in tt.iterrows():
        if r["team1"] == name:
            rows.append({"team_id": r["team1_id"], "match_id": r["match_id"], "tournament": r["tournament"],
                         "datetime": r["datetime"], "roster": roster(r, 1)})
        if r["team2"] == name:
            rows.append({"team_id": r["team2_id"], "match_id": r["match_id"], "tournament": r["tournament"],
                         "datetime": r["datetime"], "roster": roster(r, 2)})
    df = pd.DataFrame(rows)
    df["dt"] = pd.to_datetime(df["datetime"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    return df.sort_values("dt").reset_index(drop=True)


def build_eras(app):
    """Group appearances into consecutive-in-time 'eras' of an identical roster.
    Appearances with an unknown (empty) roster are skipped for era-clustering."""
    known = app[app["roster"].apply(lambda r: len(r) > 0)].sort_values("dt")
    eras = []
    for _, row in known.iterrows():
        if eras and eras[-1]["roster"] == row["roster"]:
            eras[-1]["max_dt"] = row["dt"]
            eras[-1]["team_ids"].append(row["team_id"])
        else:
            eras.append({"roster": row["roster"], "min_dt": row["dt"], "max_dt": row["dt"],
                         "team_ids": [row["team_id"]]})
    return eras


def eras_sequential(eras):
    """True if every era's appearances are chronologically disjoint from every other era's
    (i.e. one roster cleanly hands off to the next, never interleaved)."""
    sorted_eras = sorted(eras, key=lambda e: e["min_dt"])
    for i in range(len(sorted_eras) - 1):
        if sorted_eras[i]["max_dt"] >= sorted_eras[i + 1]["min_dt"]:
            return False
    return True


def close_zero_events(app, eras):
    """Zero-roster-overlap events reported at the ERA level (one line per distinct-roster
    pair that came close in time), not the raw appearance level - two eras with dozens of
    matches each would otherwise produce a combinatorial explosion of near-duplicate lines
    for what is really a single roster-transition event."""
    events = []
    for i in range(len(eras)):
        for j in range(i + 1, len(eras)):
            a, b = eras[i], eras[j]
            if a["roster"] & b["roster"]:
                continue  # some overlap - not a zero-overlap event
            gap_days = (b["min_dt"] - a["max_dt"]).days if b["min_dt"] >= a["max_dt"] else (a["min_dt"] - b["max_dt"]).days
            if gap_days <= CLOSE_DAYS:
                events.append({
                    "era_a_roster": sorted(a["roster"]), "era_a_window": (a["min_dt"], a["max_dt"]),
                    "era_b_roster": sorted(b["roster"]), "era_b_window": (b["min_dt"], b["max_dt"]),
                    "gap_days": gap_days,
                })
    return events


def assess(name, app, eras, close_events, generic_flagged):
    n_matches = len(app)
    n_eras = len(eras)
    sequential = eras_sequential(eras) if n_eras > 1 else True

    if generic_flagged and n_matches >= ESTABLISHED_MATCH_THRESHOLD and not close_events:
        return ("a", "KEEP_AS_SINGLE_TEAM",
                "Matched the generic-name regex on spelling alone (e.g. 'Team <digits>' pattern), but this deeper "
                "pass finds no close-in-time zero-roster-overlap evidence at all and a high match count - likely a "
                "regex false positive on a real, established team name, not a placeholder identity.")

    if generic_flagged and n_matches < 15:
        return ("b", "EXCLUDE_FROM_IDENTITY_DEPENDENT_FEATURES",
                "Generic/placeholder-looking name with a low match count - consistent with an ad-hoc qualifier "
                "stand-in roster rather than a persistent organization.")

    if not close_events:
        # no close-in-time collision was found on re-check (may have been flagged via a different
        # pairing at analysis time) - treat as adequately evidenced, default to keep.
        return ("a", "KEEP_AS_SINGLE_TEAM",
                "No close-in-time zero-roster-overlap event confirmed in this deeper pass; existing evidence does "
                "not contradict treating this as one continuous team.")

    if n_matches >= ESTABLISHED_MATCH_THRESHOLD:
        if sequential:
            return ("a", "KEEP_AS_SINGLE_TEAM",
                    f"{n_matches} matches under this name with clean sequential roster eras (each era's "
                    "appearances are chronologically disjoint from the next - a roster hand-off pattern, not "
                    "simultaneous usage). Per the review's own rule, zero overlap between two eras of an "
                    "established, high-volume name is NOT by itself evidence of a different organization - full "
                    "roster turnover over a multi-year dataset is expected for real orgs. Recommend keeping as a "
                    "single identity, with the era boundaries available for anyone who later wants roster-level "
                    "(not org-level) granularity.")
        else:
            return ("c", "MANUAL_REVIEW",
                    f"{n_matches} matches, but roster eras are NOT cleanly sequential (at least two distinct, "
                    "non-overlapping rosters were both in use during overlapping time windows) - this is a "
                    "higher-volume name, so outright exclusion seems too aggressive, but the interleaving pattern "
                    "is not explained by simple roster turnover either. Needs a human look at the specific events "
                    "listed below before deciding.")

    # lower match count, close-zero events exist, not generic-flagged
    if sequential:
        return ("c", "MANUAL_REVIEW",
                f"Only {n_matches} matches under this name; roster eras are sequential (consistent with turnover) "
                "but the sample is too small to be confident this isn't actually two different low-tier squads "
                "that happened to reuse a common-sounding name. Needs manual review.")
    else:
        return ("b", "NEEDS_EPISODE_SPLIT",
                f"Only {n_matches} matches under this name, with non-sequential (interleaved) rosters - two or "
                "more clearly distinct rosters appear to have been active under the same name in overlapping "
                "windows. If confirmed, this name would need to be split into separate identities per era rather "
                "than treated as one team.")


def main():
    df = load_games_tiered()
    tt = df[df["is_total"] == "True"]
    aliases = pd.read_csv(INTERIM / "team_aliases.csv", dtype=str, keep_default_na=False)
    unresolved = aliases[aliases["resolution_type"] == "unresolved"]["original_team_name"].tolist()

    md = []
    md.append("# Unresolved Team Name Review\n")
    md.append(f"Deep-dive on the {len(unresolved)} `unresolved` names from `data/interim/team_aliases.csv` "
              "(Phase 2 team-identity pass). **No identity is split or merged here** - this is a proposal for "
              "human review only; `team_aliases.csv` is not modified by this script. Per explicit guidance for "
              "this pass: for established professional organization names, zero roster overlap between two "
              "team_id instances is *not*, by itself, sufficient evidence that they are different organizations - "
              "full roster turnover is expected for real orgs over a multi-year dataset. That is reflected in the "
              "proposed decisions below.\n")
    md.append("**Proposed decision categories** (proposed only, not applied): `KEEP_AS_SINGLE_TEAM`, "
              "`NEEDS_EPISODE_SPLIT`, `EXCLUDE_FROM_IDENTITY_DEPENDENT_FEATURES`, `MANUAL_REVIEW`.\n")

    decisions_summary = []

    for name in sorted(unresolved):
        app = build_appearances(tt, name)
        if app.empty:
            md.append(f"## `{name}`\n\nNo appearances found (unexpected) - skipped.\n")
            continue

        first_dt, last_dt = app["dt"].min(), app["dt"].max()
        tournaments = sorted(app["tournament"].unique().tolist())
        n_matches = app["match_id"].nunique()
        generic_flagged = bool(GENERIC_PATTERN.match(name))

        eras = build_eras(app)
        close_events = close_zero_events(app, eras)
        letter, decision, rationale = assess(name, app, eras, close_events, generic_flagged)
        decisions_summary.append((name, n_matches, letter, decision))

        md.append(f"## `{name}`\n")
        md.append(f"- First appearance: {first_dt}  |  Last appearance: {last_dt}")
        md.append(f"- Matches: {n_matches}  |  Tournaments ({len(tournaments)}): {tournaments}")
        md.append(f"- Generic-name pattern match: {generic_flagged}")
        md.append("")

        md.append("**Chronological roster eras** (consecutive appearances sharing an identical roster; "
                   "appearances with an unknown/blank roster are omitted from era-clustering):\n")
        if eras:
            md.append("| era | roster (player_ids) | first seen | last seen | team_ids |")
            md.append("|---|---|---|---|---|")
            for i, e in enumerate(eras, 1):
                md.append(f"| {i} | {sorted(e['roster'])} | {e['min_dt']} | {e['max_dt']} | {e['team_ids']} |")
            md.append(f"\nEras chronologically sequential (no overlap between different rosters' active windows): "
                      f"{eras_sequential(eras) if len(eras) > 1 else 'n/a (only one roster era)'}\n")
        else:
            md.append("_No appearances with a known roster - cannot build an era timeline._\n")

        md.append(f"**Close-in-time zero-roster-overlap events** (era-to-era, gap <= {CLOSE_DAYS} days or "
                  "overlapping - reported once per distinct-roster pair, not once per match):\n")
        if close_events:
            for ev in close_events:
                overlap_note = " (OVERLAPPING/interleaved, not sequential)" if ev["gap_days"] < 0 else f" ({ev['gap_days']}-day gap)"
                md.append(f"- roster {ev['era_a_roster']} (active {ev['era_a_window'][0]} to {ev['era_a_window'][1]}) "
                          f"vs roster {ev['era_b_roster']} (active {ev['era_b_window'][0]} to {ev['era_b_window'][1]}){overlap_note}")
        else:
            md.append("- None found at the era level in this deeper, name-specific re-check.")
        md.append("")

        md.append(f"**Assessment**: ({letter}) "
                  f"{'same organization with roster turnover' if letter == 'a' else 'likely recycled/ambiguous team name' if letter == 'b' else 'insufficient evidence'}")
        md.append(f"**Proposed decision**: `{decision}`")
        md.append(f"**Rationale**: {rationale}\n")

    md.append("## Summary table\n")
    md.append("| team_name | matches | assessment | proposed decision |")
    md.append("|---|---|---|---|")
    for name, n_matches, letter, decision in sorted(decisions_summary, key=lambda x: -x[1]):
        md.append(f"| {name} | {n_matches} | ({letter}) | `{decision}` |")

    counts = pd.Series([d for *_, d in decisions_summary]).value_counts().to_dict()
    md.append(f"\nProposed-decision counts: {counts}\n")
    md.append("**None of these proposed decisions have been applied.** `data/interim/team_aliases.csv` is "
              "unchanged by this script. Applying any of them (splitting an identity into eras, excluding a name "
              "from identity-dependent features, etc.) is a Phase 3+ decision requiring human sign-off.\n")

    (REPORTS / "unresolved_team_review.md").write_text("\n".join(md), encoding="utf-8")

    decisions_df = pd.DataFrame(decisions_summary, columns=["team_name", "matches", "assessment_letter", "proposed_decision"])
    decisions_df.to_csv(INTERIM / "unresolved_team_decisions.csv", index=False, encoding="utf-8")

    print(f"Reviewed {len(unresolved)} unresolved names. Proposed-decision counts: {counts}")
    print("Wrote reports/unresolved_team_review.md and data/interim/unresolved_team_decisions.csv")


if __name__ == "__main__":
    main()
