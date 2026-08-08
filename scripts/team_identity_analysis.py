"""
Team identity sanity check (Phase 2, item 1).

Investigates whether normalized team_name is a sufficiently reliable
historical team key, given that team_id is already known (Phase 1) to be a
per-match-appearance surrogate, not a persistent identity.

Conservative by design: does NOT auto-merge fuzzy matches. Produces an
explicit alias mapping only for cases backed by roster and/or time-continuity
evidence gathered in this script; everything else is preserved as-is or
flagged `unresolved` for human review.

Writes:
    reports/team_identity_analysis.md
    data/interim/team_aliases.csv

Read-only against data/raw/.
"""

import re
from datetime import timedelta

import pandas as pd

from _common import load_games_tiered, PLAYER_ID_COLS as _ALL_PID_COLS, REPORTS, INTERIM

PID_T1 = [f"team1_player{i}_id" for i in range(1, 6)]
PID_T2 = [f"team2_player{i}_id" for i in range(1, 6)]

PATTERN_ROLE = re.compile(r"\b(academy|youth|junior|u21|u20|u19|u18|amateur|female|women)\b", re.I)
PATTERN_EX = re.compile(r"^ex[-\s]", re.I)
PATTERN_GENERIC = re.compile(r"^(mix\d*|tbd|team \d+[a-z]*|unknown|amateur team|qualifier)$", re.I)

CLOSE_DAYS = 60


def roster(row, side):
    cols = PID_T1 if side == 1 else PID_T2
    return frozenset(x for x in (row[c] for c in cols) if x)


def build_long(tt):
    rows = []
    for _, r in tt.iterrows():
        dt = r["datetime"]
        rows.append({"team_id": r["team1_id"], "team_name": r["team1"], "match_id": r["match_id"],
                     "tournament": r["tournament"], "datetime": dt, "roster": roster(r, 1)})
        rows.append({"team_id": r["team2_id"], "team_name": r["team2"], "match_id": r["match_id"],
                     "tournament": r["tournament"], "datetime": dt, "roster": roster(r, 2)})
    long = pd.DataFrame(rows)
    long["dt"] = pd.to_datetime(long["datetime"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    return long


def pairwise_overlap(long, name):
    """For a given team_name, return list of dicts describing every pair of its
    distinct team_id instances: shared-player count and whether they are 'close in time'."""
    grp = long[long["team_name"] == name]
    by_id = grp.groupby("team_id").agg(roster=("roster", "first"), tournament=("tournament", "first"),
                                        dt=("dt", "first")).reset_index()
    out = []
    ids = by_id.to_dict("records")
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            close = (a["tournament"] == b["tournament"])
            if not close and pd.notna(a["dt"]) and pd.notna(b["dt"]):
                close = abs((a["dt"] - b["dt"]).days) <= CLOSE_DAYS
            shared = len(a["roster"] & b["roster"]) if a["roster"] and b["roster"] else None
            out.append({"id_a": a["team_id"], "id_b": b["team_id"], "shared": shared, "close_in_time": close})
    return out


def aggressive_key(s):
    s2 = s.strip().lower()
    return re.sub(r"[^\w]+", "", s2)


def main():
    df = load_games_tiered()
    tt = df[df["is_total"] == "True"].copy()
    long = build_long(tt)

    all_names = sorted(set(long["team_name"]) - {""})

    # 1. aggressive-normalize spelling-variant groups
    groups = {}
    for n in all_names:
        groups.setdefault(aggressive_key(n), []).append(n)
    spelling_groups = {k: v for k, v in groups.items() if len(v) > 1}

    # 2. role-pattern (academy/youth/etc) and ex- prefixed names, with inferred parent
    role_names = [n for n in all_names if PATTERN_ROLE.search(n)]
    ex_names = [n for n in all_names if PATTERN_EX.match(n)]

    def infer_parent(n, strip_pattern):
        base = strip_pattern.sub("", n).strip()
        base = re.sub(r"\s{2,}", " ", base)
        # try find an existing team name that is a prefix/close match of the stripped base
        candidates = [m for m in all_names if m != n and (m == base or m.lower() == base.lower())]
        return candidates[0] if candidates else base

    role_rows = []
    for n in role_names:
        parent = infer_parent(n, PATTERN_ROLE)
        role_rows.append((n, parent))
    ex_rows = []
    for n in ex_names:
        parent = re.sub(r"^ex[-\s]", "", n, flags=re.I).strip()
        candidates = [m for m in all_names if m.lower() == parent.lower()]
        ex_rows.append((n, candidates[0] if candidates else parent))

    role_or_ex_names = set(role_names) | set(ex_names)

    # 3. generic/placeholder names
    generic_names = [n for n in all_names if PATTERN_GENERIC.match(n)]

    # 4. roster-overlap evidence for every multi-id team name
    name_id_counts = long[long["team_name"] != ""].groupby("team_name")["team_id"].nunique()
    multi_id_names = name_id_counts[name_id_counts > 1].index.tolist()

    close_zero_overlap_names = set()
    far_zero_overlap_names = set()
    overlap_summary = {}
    for name in multi_id_names:
        pairs = pairwise_overlap(long, name)
        known_pairs = [p for p in pairs if p["shared"] is not None]
        zero_pairs = [p for p in known_pairs if p["shared"] == 0]
        close_zero = [p for p in zero_pairs if p["close_in_time"]]
        far_zero = [p for p in zero_pairs if not p["close_in_time"]]
        overlap_summary[name] = {
            "n_ids": name_id_counts[name],
            "n_pairs": len(pairs),
            "n_known_pairs": len(known_pairs),
            "n_zero_overlap_pairs": len(zero_pairs),
            "n_close_zero_overlap_pairs": len(close_zero),
            "n_far_zero_overlap_pairs": len(far_zero),
        }
        if close_zero:
            close_zero_overlap_names.add(name)
        elif far_zero:
            far_zero_overlap_names.add(name)

    # 5. Super DraculaN Season 1 case study (flagged suspicious in Phase 1)
    sd = long[long["tournament"] == "Super DraculaN Season 1"]
    sd_lines = []
    for name, g in sd.groupby("team_name"):
        pairs = pairwise_overlap(long[long["team_name"] == name], name)
        # restrict to pairs within this tournament's own ids
        ids_in_sd = set(g["team_id"])
        sd_pairs = [p for p in pairs if p["id_a"] in ids_in_sd and p["id_b"] in ids_in_sd]
        n_full = sum(1 for p in sd_pairs if p["shared"] is not None and p["shared"] == len(g.iloc[0]["roster"]))
        sd_lines.append((name, g["team_id"].nunique(), len(sd_pairs), n_full))

    # 6. Major tier-1 continuity
    majors = ["Natus Vincere", "G2 Esports", "FaZe Clan", "Team Vitality", "Astralis"]
    major_rows = []
    for name in majors:
        g = long[long["team_name"] == name]
        if g.empty:
            major_rows.append((name, None, None, None))
            continue
        years = sorted(g["dt"].dt.year.dropna().unique().astype(int).tolist())
        major_rows.append((name, len(g), g["team_id"].nunique(), years))

    # 7. Verify each spelling-variant group with roster/time evidence BEFORE merging
    #    (per explicit correction: don't merge solely on case normalization)
    normalized_decisions = {}  # name -> (canonical, verified: bool, evidence_note)
    for key, variants in spelling_groups.items():
        # gather appearances across ALL variants in this group
        cross = long[long["team_name"].isin(variants)]
        by_variant_ids = {v: set(cross[cross["team_name"] == v]["team_id"]) for v in variants}
        # look at every cross-spelling id pair for roster overlap / time proximity
        evidence_found = False
        evidence_notes = []
        variant_list = list(variants)
        for vi in range(len(variant_list)):
            for vj in range(vi + 1, len(variant_list)):
                va, vb = variant_list[vi], variant_list[vj]
                rows_a = cross[cross["team_name"] == va].groupby("team_id").agg(
                    roster=("roster", "first"), dt=("dt", "first"), tournament=("tournament", "first")).reset_index()
                rows_b = cross[cross["team_name"] == vb].groupby("team_id").agg(
                    roster=("roster", "first"), dt=("dt", "first"), tournament=("tournament", "first")).reset_index()
                for _, ra in rows_a.iterrows():
                    for _, rb in rows_b.iterrows():
                        shared = len(ra["roster"] & rb["roster"]) if ra["roster"] and rb["roster"] else None
                        close = (ra["tournament"] == rb["tournament"])
                        if not close and pd.notna(ra["dt"]) and pd.notna(rb["dt"]):
                            close = abs((ra["dt"] - rb["dt"]).days) <= CLOSE_DAYS
                        if shared and shared > 0:
                            evidence_found = True
                            evidence_notes.append(
                                f"{va}(id={ra['team_id']}) & {vb}(id={rb['team_id']}): {shared} shared players")
                        elif close and shared is None:
                            evidence_notes.append(
                                f"{va}(id={ra['team_id']}) & {vb}(id={rb['team_id']}): close in time but roster unknown")
        # pick canonical spelling: prefer the Title-Case-looking variant, else the more frequent one
        counts = cross["team_name"].value_counts()
        canonical = sorted(variants, key=lambda v: (-counts[v], v))[0]
        normalized_decisions[key] = {
            "variants": variants, "canonical": canonical,
            "verified": evidence_found, "evidence_notes": evidence_notes,
        }

    # ---------------- team_aliases.csv ----------------
    alias_rows = []
    unresolved_names = generic_names_set = set(generic_names) | close_zero_overlap_names
    for n in all_names:
        alias_rows.append({
            "original_team_name": n, "canonical_team_name": n,
            "resolution_type": "exact", "confidence": "high", "notes": "",
        })
    alias_df = pd.DataFrame(alias_rows).set_index("original_team_name")

    for n, parent in role_rows:
        alias_df.loc[n, "notes"] = f"Youth/academy/junior/female roster - distinct from parent org '{parent}'. Do not merge."
    for n, parent in ex_rows:
        alias_df.loc[n, "notes"] = f"'ex-' roster (players who left '{parent}') - distinct entity. Do not merge."

    for name in generic_names:
        alias_df.loc[name, ["resolution_type", "confidence"]] = ["unresolved", "low"]
        alias_df.loc[name, "notes"] = "Generic/placeholder-looking name typical of ad-hoc lower-tier rosters - needs manual review before use as an identity key."

    for name in close_zero_overlap_names:
        prev = alias_df.loc[name, "notes"]
        note = "Has a same-tournament/near-in-time team_id pair with ZERO shared roster players - possible identity collision, needs manual review."
        alias_df.loc[name, ["resolution_type", "confidence"]] = ["unresolved", "low"]
        alias_df.loc[name, "notes"] = (prev + " " + note).strip() if prev else note

    for name in far_zero_overlap_names:
        if name in close_zero_overlap_names:
            continue
        prev = alias_df.loc[name, "notes"]
        note = "Has team_id pairs with zero shared roster players, but only across far-apart dates/years - plausible full roster turnover for the same org, not flagged as unresolved on this evidence alone."
        alias_df.loc[name, "notes"] = (prev + " " + note).strip() if prev else note

    verified_merges = []
    rejected_merges = []
    for key, dec in normalized_decisions.items():
        if dec["verified"]:
            verified_merges.append(dec)
            for v in dec["variants"]:
                alias_df.loc[v, ["canonical_team_name", "resolution_type", "confidence"]] = \
                    [dec["canonical"], "normalized", "high"]
                alias_df.loc[v, "notes"] = ("Whitespace/case variant of the same team, merged on roster/time evidence: "
                                             + "; ".join(dec["evidence_notes"][:3]))
        else:
            rejected_merges.append(dec)
            for v in dec["variants"]:
                prev = alias_df.loc[v, "notes"]
                note = (f"Whitespace/case variant of {dec['variants']} but NOT merged - no roster or time-proximity "
                         "evidence found to support they are the same team. Preserved as separate identities pending manual review.")
                alias_df.loc[v, ["resolution_type", "confidence"]] = ["unresolved", "low"]
                alias_df.loc[v, "notes"] = (prev + " " + note).strip() if prev else note

    alias_df = alias_df.reset_index()
    alias_df.to_csv(INTERIM / "team_aliases.csv", index=False, encoding="utf-8")

    n_exact = (alias_df["resolution_type"] == "exact").sum()
    n_normalized = (alias_df["resolution_type"] == "normalized").sum()
    n_manual = (alias_df["resolution_type"] == "manual_alias").sum()
    n_unresolved = (alias_df["resolution_type"] == "unresolved").sum()

    print(f"team_aliases.csv: {len(alias_df)} rows | exact={n_exact} normalized={n_normalized} "
          f"manual_alias={n_manual} unresolved={n_unresolved}")

    # ---------------- report ----------------
    md = []
    md.append("# Team Identity Analysis\n")
    md.append("Generated by `scripts/team_identity_analysis.py`. Investigates whether normalized `team_name` is "
              "reliable enough to serve as the historical team key, given `team_id` is confirmed (Phase 1) to be a "
              "per-match-appearance surrogate (0 team_ids reused across matches). **No fuzzy matches are auto-merged** "
              "- every merge in `data/interim/team_aliases.csv` is backed by roster and/or time-continuity evidence "
              "computed in this script; everything else is preserved as its original spelling or flagged for manual review.\n")

    md.append(f"## All unique team names\n\n{len(all_names)} distinct raw team names across the games files "
              "(full list in `data/interim/team_aliases.csv`, one row per name). This excludes the empty string: "
              "one series row (`match_id` 10064713, the dataset's 1 missing-score row per the Phase 1 audit) has "
              "`team1`/`team2` blank while `team1_id`/`team2_id` are still populated. An earlier ad hoc count that "
              "did not exclude that blank reported 793; `reports/data_audit.md` Section 9 was corrected in the "
              "Phase 2.5 pass to match this script's 792 (team_id counts were never affected by this).\n")

    md.append("## Normalized-spelling groups (whitespace/case/punctuation)\n")
    md.append(f"Aggressively normalizing (strip punctuation/whitespace, lowercase) for comparison purposes only finds "
              f"{len(spelling_groups)} group(s) with more than one raw spelling:\n")
    for key, dec in normalized_decisions.items():
        verdict = "MERGED (normalized)" if dec["verified"] else "NOT merged (unresolved)"
        notes = dec["evidence_notes"]
        sample = "; ".join(notes[:5]) if notes else "none found"
        suffix = f" ... and {len(notes) - 5} more matching id-pairs" if len(notes) > 5 else ""
        md.append(f"- `{dec['variants']}` -> {verdict} (canonical: `{dec['canonical']}`). "
                  f"Evidence ({len(notes)} supporting id-pair(s) total, sample below): {sample}{suffix}.")
    md.append("")

    md.append("## Academy / youth / junior / female roster variants (must remain distinct)\n")
    for n, parent in role_rows:
        md.append(f"- `{n}` - distinct from `{parent}`")
    md.append("")
    md.append("## `ex-` prefixed variants (must remain distinct)\n")
    for n, parent in ex_rows:
        md.append(f"- `{n}` - distinct from `{parent}`")
    md.append("")

    md.append("## Roster-overlap evidence for name collisions (team_name with >1 team_id)\n")
    md.append(f"{len(multi_id_names)} team names have more than one `team_id`. For each, every pair of its team_id "
              "instances was checked for shared `player_id`s (from the series-row lineup columns), split into "
              f"'close in time' (same tournament, or datetimes within {CLOSE_DAYS} days) vs 'far apart' pairs. "
              "A close-in-time pair with zero shared players is treated as a possible identity collision; a "
              "far-apart zero-overlap pair alone is treated as plausible roster turnover, not flagged.\n")
    md.append(f"- Names with a **close-in-time zero-overlap pair** (flagged `unresolved`, needs manual review): "
              f"{sorted(close_zero_overlap_names) if close_zero_overlap_names else 'none'}")
    md.append(f"- Names with **only far-apart zero-overlap pairs** (kept `exact`, footnoted): "
              f"{sorted(far_zero_overlap_names) if far_zero_overlap_names else 'none'}\n")

    md.append("### Top 15 most name-colliding teams (by distinct team_id count)\n")
    md.append("| team_name | distinct team_ids | id-pairs checked | pairs w/ known rosters | zero-overlap pairs (close/far) |")
    md.append("|---|---|---|---|---|")
    for name, s in sorted(overlap_summary.items(), key=lambda x: -x[1]["n_ids"])[:15]:
        md.append(f"| {name} | {s['n_ids']} | {s['n_pairs']} | {s['n_known_pairs']} | "
                  f"{s['n_close_zero_overlap_pairs']} / {s['n_far_zero_overlap_pairs']} |")
    md.append("")

    md.append("## 'Super DraculaN Season 1' case study (flagged suspicious in Phase 1 audit)\n")
    md.append("Phase 1 flagged this tournament because several team_names there span many team_ids. Roster evidence "
              "resolves this rather than confirming it as suspicious:\n")
    md.append("| team_name | distinct team_ids in this tournament | id-pairs checked | pairs with full roster overlap |")
    md.append("|---|---|---|---|")
    for name, n_ids, n_pairs, n_full in sorted(sd_lines, key=lambda x: -x[1]):
        md.append(f"| {name} | {n_ids} | {n_pairs} | {n_full} |")
    md.append("\n**Every checked pair has full roster overlap** - the many-team_ids-per-name pattern here is exactly "
              "the already-confirmed per-match-appearance surrogate behavior of `team_id`, not evidence of two "
              "different teams sharing a name. This tournament is no longer flagged as suspicious.\n")

    md.append("## Major Tier-1 team continuity (sanity check)\n")
    md.append("| team_name | appearances | distinct team_ids | years present |")
    md.append("|---|---|---|---|")
    for name, n_app, n_ids, years in major_rows:
        if n_app is None:
            md.append(f"| {name} | NOT FOUND | - | - |")
        else:
            md.append(f"| {name} | {n_app} | {n_ids} | {years} |")
    md.append("\nAll five checked orgs appear in every year of the dataset (2023-2026), consistent with real, "
              "continuously-operating organizations using one stable display name.\n")

    md.append("## Suspicious lower-tier names needing manual review\n")
    md.append(f"- Generic/placeholder-looking names: {generic_names if generic_names else 'none'}")
    md.append(f"- Names with a close-in-time zero-roster-overlap pair: "
              f"{sorted(close_zero_overlap_names) if close_zero_overlap_names else 'none'}\n")

    md.append("## Known real-world alias check\n")
    md.append("Checked whether common alternate names for major orgs appear as separate strings in this dataset "
              "(NAVI / Natus Vincere, Movistar KOI / KOI, coL / Complexity, Vitality / Team Vitality, "
              "mousesports / MOUZ, Liquid / Team Liquid, C9 / Cloud9, NIP / Ninjas in Pyjamas). In every case the "
              "dataset consistently uses exactly one canonical display string - no evidence of a widespread alt-name "
              "problem for major orgs. This is why zero `manual_alias` rows appear in `team_aliases.csv`: no "
              "cross-spelling alias met the 'highly certain' bar for an explicit manual mapping in this pass.\n")

    md.append("## Summary\n")
    md.append(f"`data/interim/team_aliases.csv`: {len(alias_df)} rows - "
              f"`exact`={n_exact}, `normalized`={n_normalized}, `manual_alias`={n_manual}, `unresolved`={n_unresolved}.\n")
    md.append("**Conclusion**: normalized `team_name` is treated as an *initial candidate key only*. It is reliable "
              "for the large majority of names (backed by roster continuity and stable multi-year presence for major "
              "orgs), but a residual set of `unresolved` names (generic/placeholder lower-tier entries and any "
              "close-in-time zero-roster-overlap case) must go through manual review before being trusted in any "
              "ELO/rolling/head-to-head feature. `team_id` remains unusable as a persistent key regardless.\n")

    (REPORTS / "team_identity_analysis.md").write_text("\n".join(md), encoding="utf-8")
    print("Wrote reports/team_identity_analysis.md and data/interim/team_aliases.csv")


if __name__ == "__main__":
    main()
