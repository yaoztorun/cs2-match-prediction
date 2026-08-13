"""
Phase 5C reports.

Writes:
    reports/phase5c_player_roster_feature_engineering.md   (design / rules / decisions)
    reports/phase5c_player_roster_feature_quality.md       (descriptive diagnostics)

DISCIPLINE FOR THE QUALITY REPORT
  * computed on the GLOBAL TRAIN partition only (data/modeling/series_split_v1.csv),
    the same discipline scripts/phase5a_reports.py and phase5b2_reports.py use;
  * validation is never summarized, test is never opened, Cologne is never read;
  * NO feature-vs-target correlations, NO predictive-performance metric of any
    kind, NO tuning of the 90-day / 60-day constants.
"""

import json

import numpy as np
import pandas as pd

from _common import INTERIM, ROOT
from player_roster_feature_engine import (
    PLAYER_ROSTER_ENGINE_VERSION, ROSTER_LOOKBACK_DAYS, PLAYER_FORM_HALF_LIFE_DAYS, ROSTER_SIZE,
    ROSTER_DIRECTIONAL_FEATURES, ROSTER_SYMMETRIC_FEATURES, ROSTER_PERFORMANCE_DIFFS,
)
from player_roster_stream_common import MIN_ROUNDS_FOR_VALID_MAP

FEATURES_DIR = ROOT / "data" / "features"
REPORTS = ROOT / "reports"


def md_table(df, floatfmt="{:.4f}"):
    def fmt(v):
        if isinstance(v, float):
            return "n/a" if pd.isna(v) else floatfmt.format(v)
        return str(v)
    head = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep] + body)


def write_engineering_report(summary):
    info = summary["stream_info"]
    md = []
    md.append("# Phase 5C - Leakage-Safe Player Form and Roster Stability (Design)\n")
    md.append("**No model is trained in this phase.** No validation, test or Cologne metric is "
              "computed anywhere. The deliverable is a reusable player/roster state engine plus one "
              "extended feature dataset built with it.\n")

    md.append("## 1. Why player features may add signal\n")
    md.append("Team ELO, map pool and team-level form all describe a team as an indivisible unit. A "
              "CS2 team is five individuals whose recent individual output varies, who change "
              "employers, and who are sometimes replaced by stand-ins. Two teams with identical ELO "
              "can differ sharply in the current form of their star, the form of their weakest "
              "player, and whether they are even fielding a settled lineup. Phase 5C tries to make "
              "that visible pre-match.\n")

    md.append("## 2. Two distinct state types (never conflated)\n")
    md.append("- **Global player performance**, keyed by persistent `player_id`. Individual form "
              "FOLLOWS THE PLAYER across team changes - a transfer must not pretend a strong player "
              "has no history. This state never asks which team the player was on.")
    md.append("- **Team roster / appearance**, keyed by canonical team identity. Which players have "
              "recently REPRESENTED THAT TEAM. Team membership does NOT transfer with the player: a "
              "transferred player enters the new team's inferred roster only after actually "
              "appearing for it. Both halves of that distinction are pinned by explicit tests.\n")

    md.append("## 3. Persistent player identity\n")
    md.append("The schema audit found `player_id` to be by far the most trustworthy identifier in "
              f"this dataset: {info['distinct_players']:,} distinct ids in the development stream, "
              "nullable `Int64` with no `0`/`-1` sentinels, and **zero** id-to-name conflicts in "
              "either direction. By contrast `team_id` is not an organisation key at all (92% of "
              "players change it), so every team-keyed structure here uses `team{1,2}_canonical`. "
              "Player NAME columns are never read: 1,316 slots carry an id with a blank name, so the "
              "id is the strictly broader signal.\n")

    md.append("## 4. Historical roster inference (and why the target lineup is forbidden)\n")
    md.append(f"For team X at cutoff T: take that team's appearances in the half-open window "
              f"`[T - {ROSTER_LOOKBACK_DAYS:.0f}d, T)`, weight each by "
              f"`0.5 ** (age_days / {PLAYER_FORM_HALF_LIFE_DAYS:.0f})`, and take the top "
              f"{ROSTER_SIZE} players by (mass, then most recent appearance, then lowest "
              "`player_id`). Fewer than five available -> use what exists; players are never "
              "fabricated.\n")
    md.append("The target series' actual lineup is forbidden as a predictor for two independent "
              "reasons, both pointing the same way:\n")
    md.append("1. **Application contract.** The planned single-page app asks a user to choose two "
              "TEAMS, not ten player ids. A training-time feature depending on the real lineup could "
              "never be reproduced at inference time.")
    md.append("2. **This repo's own standing ruling.** `reports/data_audit.md` open question 2 and "
              "`reports/leakage_analysis.md` record that it is *unknown* whether the lineup columns "
              "are a pre-announced starting five or a post-hoc box-score roster including "
              "substitutions, and rule that they must be **treated as leaky by default until the "
              "collection method is confirmed**. Inferring from prior appearances only is what keeps "
              "Phase 5C compliant with that ruling.\n")
    md.append("The actual lineup is used ONLY in Phase B, after the series' own feature row has been "
              "emitted, to update state for LATER series. A dedicated test replaces the target "
              "series' five players with five completely different players and asserts that not one "
              "emitted feature moves.\n")

    md.append("## 5. Authoritative series datetime\n")
    md.append("The cutoff is the authoritative series start taken from the canonical series source "
              "(`series_base.parquet`, the same column V1/V2/V3 use), joined onto every "
              "map/player observation of that `match_id`. It is deliberately **not** "
              "`groupby(match_id).datetime.min()` over map rows - map timestamps are provenance "
              "only. In this export "
              f"{info['map_rows_whose_map_datetime_differs_from_series_datetime']} map rows carry a "
              "map timestamp differing from the authoritative series start; a test constructs a "
              "series whose maps carry deliberately different map timestamps and proves the emitted "
              "pre-series features are unchanged.\n")

    md.append("## 6. Exact-timestamp and same-series isolation\n")
    md.append("```\nfor each authoritative series_datetime batch:\n"
              "    PHASE A (read):  emit ONE pre-series vector per requested series,\n"
              "                     from the state as it was BEFORE the batch\n"
              "    PHASE B (write): only then apply every player observation in the batch\n```\n")
    md.append("Map 1's box score therefore cannot reach Map 2 of the same series, a series cannot "
              "see its own maps, and two series sharing an instant cannot see each other. Input row "
              "order within a batch is irrelevant to the output.\n")

    md.append("## 7. Player-statistic normalization\n")
    md.append("- `kd_balance = (kills - deaths) / max(kills + deaths, 1)` - bounded in [-1, 1], "
              "preferred over a raw K/D ratio because it cannot explode when deaths are small.")
    md.append("- `assists_per_round = assists / max(rounds, 1)`, where `rounds = score1_game + "
              "score2_game`. Included only because the audit proved that denominator clean (0 nulls, "
              "median 21, and a kills-per-round distribution tightly concentrated at the theoretical "
              "~6.6).")
    md.append("- `adr` and `kast` are **already round-normalized rates** in the source and are used "
              "as-is - never divided again. `kast` is on a 0-100 scale.")
    md.append("- `kddiff` is **never read**: the audit found it is 100% collinear with "
              "`kills - deaths` (0 mismatches in 102,243 slots).\n")

    md.append("## 8. Time weighting\n")
    md.append(f"Player form uses `weight = 0.5 ** (age_days / {PLAYER_FORM_HALF_LIFE_DAYS:.0f})` over "
              "**all** of a player's strictly-prior maps (the 90-day window applies to roster "
              "inference only). `player_history_mass = sum(weights)` is carried alongside every form "
              "statistic so the model can tell a well-evidenced estimate from a thin one. Both "
              "constants are fixed engineering choices for this phase and are **not tuned** against "
              "any metric.\n")

    md.append("## 9. Malformed source rows\n")
    md.append("The audit found two structural defects that would silently corrupt a per-player "
              "state. Both are handled at load time and counted:\n")
    md.append(f"- **Same player on BOTH sides of one map** ({info['maps_excluded_player_on_both_sides']} "
              "maps): that player would receive a win and a loss for one map and both teams would "
              "record them as a member -> the **entire map** is excluded from performance and "
              "appearance updates.")
    md.append(f"- **Same player in more than one slot on one side** "
              f"({info['duplicate_player_slot_groups']} groups): collapsed to a single observation "
              f"when the duplicated slots agree ({info['duplicate_player_slots_collapsed']}), and the "
              f"player's entry excluded from that map when they disagree "
              f"({info['duplicate_player_slots_conflicting_excluded']}).\n")
    md.append("Independently of the loader, the store enforces two structural invariants by "
              "construction, so duplication can never inflate history mass even if a future data "
              "source presents it: **at most one `PlayerMapEntry` per (game_id, player_id)** and "
              "**at most one `AppearanceEntry` per (game_id, team, player_id)**.\n")

    md.append("## 10. Identity policy for the two states\n")
    md.append("Following the Phase 3 / 5A / 5B.2 principle (\"a team's own result is a real fact\") "
              "applied at the correct key granularity: **player performance** is player-keyed and "
              "updates whenever that player's box score is usable, regardless of either team's "
              "identity eligibility - the player's own performance is a real fact about the player. "
              "**Team appearance** is team-keyed and updates only when that canonical team is "
              "identity-eligible, exactly like every other team-keyed state in this project.\n")
    md.append("Note that canonical team identity deliberately spans roster turnover "
              "(`KEEP_AS_SINGLE_TEAM` in the Phase 2.5 policy, whose stated rationale is that full "
              "roster turnover is expected for real organisations over a multi-year dataset). That "
              "is what makes turnover a measurable signal here rather than a hidden identity split.\n")

    md.append("## 11. Cold start\n")
    md.append("| quantity | cold start | rationale |\n|---|---|---|\n"
              "| roster performance aggregates | **NaN** | genuinely missing; never a sentinel and "
              "never a population-wide mean, which would import information from outside the "
              "strictly-prior window |\n"
              "| player_history_mass | 0.0 | a true absence of evidence, not \"neutral\" |\n"
              "| roster_size, stability counts/ratios, flags | 0 | |\n")
    md.append("Aggregates are computed over the inferred roster players that have at least one "
              "usable prior observation, and are NaN only when NO inferred roster player has any. "
              "The build asserts the exact invariant that the ten performance diffs are NaN "
              "**exactly** where `roster_form_players_min == 0`. Downstream this is handled the same "
              "way `days_since_last_match_diff` already is: preserved natively by XGBoost, "
              "train-fold-only median imputation for Random Forest.\n")
    md.append("`roster_size_min` and `roster_form_players_min` are deliberately different quantities: "
              "a team can have five INFERRED players while only some of them have usable prior box "
              "scores.\n")

    md.append("## 12. Feature inventory\n")
    md.append(f"**{len(ROSTER_DIRECTIONAL_FEATURES)} directional + "
              f"{len(ROSTER_SYMMETRIC_FEATURES)} symmetric = "
              f"{len(ROSTER_DIRECTIONAL_FEATURES) + len(ROSTER_SYMMETRIC_FEATURES)} new features**, "
              f"appended to V3's 57 predictive columns for a V4 total of "
              f"{summary['n_total_features']}.\n")
    md.append("Deliberately compact and CS2-meaningful rather than exhaustive: lineup quality "
              "(mean), star form (top), weak-link form (bottom) for ADR/KAST/KD-balance; support "
              "contribution (assists per round); roster churn and continuity; and the evidence "
              "behind all of it.\n")
    md.append("Directional: " + ", ".join(f"`{c}`" for c in ROSTER_DIRECTIONAL_FEATURES) + "\n")
    md.append("Symmetric/confidence: " + ", ".join(f"`{c}`" for c in ROSTER_SYMMETRIC_FEATURES) + "\n")

    md.append("## 13. Future application contract\n")
    md.append("`build_future_player_roster_features(store, team1, team2, as_of_datetime)` is pure "
              "and read-only and takes **no target, no score and no lineup argument**, so the target "
              "series' five players cannot enter even by accident (asserted by signature "
              "inspection). It is the exact function the offline builder calls, so training and "
              "inference cannot silently diverge. The state is deliberately keyed by persistent "
              "player id and canonical team name only, so a future provider (e.g. GRID) could update "
              "it without any feature definition changing. No such integration is done now.\n")

    md.append("## 14. Pre-Cologne state\n")
    md.append("`scripts/build_pre_cologne_player_roster_state_v1.py` replays the canonical stream "
              "strictly before the Cologne cutoff into a fresh store and independently re-derives, "
              "from the store itself, that no history entry or appearance is at/after the cutoff and "
              "that no Cologne `match_id` appears anywhere. Written as a flat scalar parquet plus the "
              "full reloadable JSON. No post-Cologne deployment state is built.\n")

    md.append("## 15. Limitations (read before interpreting any later ablation)\n")
    md.append("- **Player data covers only about half the series universe.** `map_base` carries "
              f"player rows for {summary['matches_with_own_player_rows']:,} of the "
              f"{summary['rows']:,} series in V4 "
              f"({summary['matches_without_own_player_rows']:,} have none of their own), and map "
              "coverage is strongly tier-skewed (~74% tier1 vs ~36% tier2 and ~26% tier3). "
              f"**{summary['rows_with_no_usable_player_history']:,} of {summary['rows']:,} V4 rows "
              f"({100 * summary['rows_with_no_usable_player_history'] / summary['rows']:.1f}%) have "
              "no usable prior player history on at least one side** and therefore carry NaN "
              "performance features. This is the binding constraint on how much these features can "
              "possibly contribute, and it is a property of the source data, not of the engine.")
    md.append("- The inferred roster is a *prediction* of the lineup, not the lineup. When a team "
              "fields an unexpected stand-in, the features describe who was expected to play.")
    md.append("- Individual statistics are opponent-unadjusted: ADR earned against weak opposition "
              "counts the same as ADR against elite opposition. The opponent-adjustment machinery "
              "built in Phase 5B.2 operates at team level only.")
    md.append("- `ex-<Org>` canonical variants register as separate teams, so a rebrand looks like a "
              "brand-new team with no roster history.")
    md.append("- No post-Cologne deployment state; no GRID or other live provider integration.\n")

    md.append("## 16. What Phase 5C deliberately does NOT do\n")
    md.append("- No model is trained or tuned; no validation, test or Cologne metric is computed.\n"
              "- No tuning of the 90-day roster window or the 60-day half-life.\n"
              "- No feature selection, and no feature-vs-target association is reported anywhere.\n"
              "- No use of the target series' lineup, player ids, player names or box score.\n"
              "- Nothing under `data/raw/`, `reference/` or `src/` is touched, and no Phase 1-5B.3 "
              "artifact is modified.\n")

    (REPORTS / "phase5c_player_roster_feature_engineering.md").write_text("\n".join(md), encoding="utf-8")


def write_quality_report(summary):
    info = summary["stream_info"]
    v4 = pd.read_parquet(FEATURES_DIR / "series_features_v4_roster.parquet", engine="fastparquet")
    audit = pd.read_parquet(FEATURES_DIR / "series_roster_states_v1.parquet", engine="fastparquet")
    split = pd.read_csv(ROOT / "data" / "modeling" / "series_split_v1.csv")
    train_ids = set(split.loc[split["split"] == "train", "match_id"])
    train = v4[v4["match_id"].isin(train_ids)].reset_index(drop=True)

    audit["_mid"] = audit["match_id"].astype(str)
    train_audit = audit[audit["_mid"].isin({str(m) for m in train_ids})].reset_index(drop=True)

    new_cols = ROSTER_DIRECTIONAL_FEATURES + ROSTER_SYMMETRIC_FEATURES

    md = []
    md.append("# Phase 5C - Player / Roster Feature Quality (descriptive)\n")
    md.append("**Scope discipline.** Every number below is computed on the **global TRAIN "
              "partition** (`data/modeling/series_split_v1.csv`), the same discipline "
              "`scripts/phase5a_reports.py` and `phase5b2_reports.py` established. Validation is not "
              "summarized, the test partition is not opened, and Cologne is not read. **No "
              "feature-vs-target association is reported** - no correlations with the target, no "
              "rankings, no predictive-performance metric, no tuning of the 90/60-day constants.\n")
    md.append(f"TRAIN series: **{len(train):,}** of {len(v4):,}.\n")

    md.append("## 1. Player identity and stat coverage (development stream)\n")
    rows = [
        ("distinct persistent player_ids", info["distinct_players"]),
        ("distinct maps contributing player observations", info["distinct_maps"]),
        ("distinct matches contributing player observations", info["distinct_matches"]),
        ("player-slot observations after cleaning", info["player_slot_observations_after_cleaning"]),
        ("observations with a usable box score", info["observations_with_usable_stats"]),
        ("observations with an id but no usable box score", info["observations_without_usable_stats"]),
        ("maps excluded (same player on both sides)", info["maps_excluded_player_on_both_sides"]),
        ("duplicate player-slot groups collapsed", info["duplicate_player_slots_collapsed"]),
        ("duplicate player-slot groups excluded (conflicting stats)",
         info["duplicate_player_slots_conflicting_excluded"]),
        ("map rows whose map timestamp differs from the authoritative series start",
         info["map_rows_whose_map_datetime_differs_from_series_datetime"]),
    ]
    md.append(md_table(pd.DataFrame(rows, columns=["quantity", "n"]), "{:.0f}"))
    md.append("")
    usable_pct = 100 * info["observations_with_usable_stats"] / max(
        info["player_slot_observations_after_cleaning"], 1)
    md.append(f"Usable-box-score rate among retained player observations: **{usable_pct:.2f}%**.\n")

    md.append("## 2. Series-level player coverage\n")
    rows = [
        ("V4 series with their own player rows", summary["matches_with_own_player_rows"]),
        ("V4 series with NO player rows of their own", summary["matches_without_own_player_rows"]),
    ]
    cov = pd.DataFrame(rows, columns=["quantity", "n"])
    cov["pct_of_v4"] = 100 * cov["n"] / summary["rows"]
    md.append(md_table(cov, "{:.2f}"))
    md.append("\nA series without player rows of its own still receives roster features - they "
              "describe the two teams' PRIOR history, not the target match.\n")

    md.append("## 3. Inferred-roster completeness and cold start (TRAIN)\n")
    rows = [
        ("both inferred rosters contain five players",
         int((train["both_teams_have_5_inferred_players"] == 1).sum())),
        ("roster_size_min == 0 (>=1 side has no inferred roster at all)",
         int((train["roster_size_min"] == 0).sum())),
        ("roster_form_players_min == 0 (>=1 side has no usable player history) -> NaN performance",
         int((train["roster_form_players_min"] == 0).sum())),
        ("roster_form_players_min >= 5 (both sides fully evidenced)",
         int((train["roster_form_players_min"] >= 5).sum())),
        ("roster_min_player_history_mass == 0 (>=1 inferred player with no evidence)",
         int((train["roster_min_player_history_mass"] == 0).sum())),
    ]
    comp = pd.DataFrame(rows, columns=["quantity", "n"])
    comp["pct_of_train"] = 100 * comp["n"] / len(train)
    md.append(md_table(comp, "{:.2f}"))
    md.append("")
    md.append("**`roster_size_min` and `roster_form_players_min` are different quantities**: the gap "
              "between the two rows above is exactly the population of series where a five-player "
              "lineup could be inferred but some of those players have no usable prior box score.\n")

    md.append("## 4. Player-form distributions (TRAIN, per-side inferred-roster aggregates)\n")
    md.append("Descriptive only - to confirm nothing is degenerate or absurdly scaled. `adr` and "
              "`kast` are the source's own round-normalized rates (KAST on 0-100); `kd_balance` is "
              "bounded in [-1, 1].\n")
    side_cols = [("team1_roster_mean_adr", "roster mean ADR"),
                 ("team1_roster_top_adr", "roster top ADR"),
                 ("team1_roster_bottom_adr", "roster bottom ADR"),
                 ("team1_roster_mean_kast", "roster mean KAST"),
                 ("team1_roster_mean_kd_balance", "roster mean KD-balance"),
                 ("team1_roster_mean_assists_per_round", "roster mean assists/round"),
                 ("team1_roster_mean_player_history_mass", "roster mean player history mass"),
                 ("team1_roster_min_player_history_mass", "roster min player history mass")]
    drows = []
    for col, label in side_cols:
        if col not in train_audit.columns:
            continue
        s = pd.to_numeric(train_audit[col], errors="coerce")
        drows.append((label, int(s.notna().sum()), float(s.mean()), float(s.std()),
                      float(s.min()), float(s.median()), float(s.max())))
    md.append(md_table(pd.DataFrame(
        drows, columns=["quantity", "n_defined", "mean", "std", "min", "median", "max"]), "{:.4f}"))
    md.append("")

    md.append("## 5. Roster-stability distributions (TRAIN, team1 side)\n")
    srows = []
    for col, label in [("team1_recent_unique_players_10_maps", "unique players in last 10 maps"),
                        ("team1_recent_unique_players_20_maps", "unique players in last 20 maps"),
                        ("team1_core5_appearance_concentration_90d", "core-5 appearance concentration (90d)"),
                        ("team1_core5_continuity_last_10", "core-5 continuity over last 10 maps"),
                        ("team1_roster_size", "inferred roster size")]:
        if col not in train_audit.columns:
            continue
        s = pd.to_numeric(train_audit[col], errors="coerce")
        srows.append((label, float(s.mean()), float(s.std()), float(s.min()),
                      float(s.median()), float(s.max())))
    md.append(md_table(pd.DataFrame(
        srows, columns=["quantity", "mean", "std", "min", "median", "max"]), "{:.4f}"))
    md.append("\nA perfectly settled five-player roster scores 1.0 on both concentration and "
              "continuity; frequent stand-ins or turnover push both down.\n")

    md.append("## 6. Observed player mobility (team changes)\n")
    snap_path = INTERIM / "pre_cologne_player_roster_state_v1.json"
    if snap_path.exists():
        meta = json.loads(snap_path.read_text(encoding="utf-8"))["meta"]
        md.append(f"In the strictly pre-Cologne state: **{meta['players_with_multiple_teams_observed']:,}** "
                  f"of {meta['player_states']:,} tracked players "
                  f"({100 * meta['players_with_multiple_teams_observed'] / max(meta['player_states'], 1):.1f}%) "
                  "were observed playing for more than one canonical team - i.e. transfers are "
                  "directly measurable, which is exactly why global player form is tracked "
                  "separately from team membership.\n")

    md.append("## 7. Missingness of the new features (TRAIN)\n")
    miss = []
    for c in new_cols:
        n_missing = int(train[c].isna().sum())
        miss.append((c, n_missing, 100 * n_missing / len(train)))
    mdf = pd.DataFrame(miss, columns=["feature", "n_missing", "pct_missing"])
    md.append(md_table(mdf[mdf["n_missing"] > 0], "{:.2f}") if (mdf["n_missing"] > 0).any()
              else "No new feature has any missing value.")
    md.append("")
    md.append(f"Exactly the {len(ROSTER_PERFORMANCE_DIFFS)} performance diffs carry NaN, and only "
              "where `roster_form_players_min == 0` - the documented cold-start contract, asserted "
              "at build time and re-checked by `scripts/validate_phase5c.py`. The remaining "
              f"{len(new_cols) - len(ROSTER_PERFORMANCE_DIFFS)} features are always defined.\n")

    md.append("## 8. Distribution summaries of the new features (TRAIN)\n")
    dist_rows = []
    for c in new_cols:
        s = pd.to_numeric(train[c], errors="coerce")
        dist_rows.append((c, float(s.mean()), float(s.std()), float(s.min()),
                           float(s.median()), float(s.max())))
    md.append(md_table(pd.DataFrame(
        dist_rows, columns=["feature", "mean", "std", "min", "median", "max"]), "{:.4f}"))
    md.append("")

    md.append("## 9. Feature-feature redundancy (descriptive, NOT target correlation)\n")
    md.append("Correlation among the 15 new directional features only, to describe overlap without "
              "ever touching the target:\n")
    corr = train[ROSTER_DIRECTIONAL_FEATURES].corr(numeric_only=True)
    pairs = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if not pd.isna(corr.iloc[i, j]):
                pairs.append((cols[i], cols[j], corr.iloc[i, j]))
    pairs.sort(key=lambda t: -abs(t[2]))
    md.append("Top 12 |corr| pairs among the new directional features:\n")
    md.append("| feature A | feature B | r |\n|---|---|---|")
    for a, b, r in pairs[:12]:
        md.append(f"| {a} | {b} | {r:+.3f} |")
    md.append("")

    md.append("## 10. What this report does not claim\n")
    md.append("Nothing here says any feature is useful. Coverage, spread and completeness are "
              "properties of the data; predictive value is a property of a model that has not been "
              "fitted yet. A separate paired V3-vs-V4 ablation will measure that, and V4 will not be "
              "modified in response to it.\n")

    (REPORTS / "phase5c_player_roster_feature_quality.md").write_text("\n".join(md), encoding="utf-8")


def main():
    with open(INTERIM / "series_features_v4_build_summary.json", encoding="utf-8") as f:
        summary = json.load(f)
    write_engineering_report(summary)
    write_quality_report(summary)
    print(f"Wrote {REPORTS / 'phase5c_player_roster_feature_engineering.md'}")
    print(f"Wrote {REPORTS / 'phase5c_player_roster_feature_quality.md'}")


if __name__ == "__main__":
    main()
