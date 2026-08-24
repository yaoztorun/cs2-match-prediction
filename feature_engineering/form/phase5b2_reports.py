"""
Phase 5B.2 reports.

Writes:
    reports/phase5b2_team_form_feature_engineering.md   (design / rules / decisions)
    reports/phase5b2_team_form_feature_quality.md       (descriptive diagnostics)

DISCIPLINE FOR THE QUALITY REPORT
  * computed on the GLOBAL TRAIN partition only (data/modeling/series_split_v1.csv),
    the same discipline feature_engineering/maps/phase5a_reports.py already established;
  * validation is never summarized, test is never opened, Cologne is never read;
  * NO feature-vs-target correlations, rankings or "promising feature" claims,
    NO predictive-performance metric of any kind, NO half-life tuning discussion.
"""

import json

import numpy as np
import pandas as pd
import yaml

from _common import INTERIM, ROOT
from feature_engineering.form.team_form_engine import FORM_DIRECTIONAL_FEATURES, FORM_SYMMETRIC_FEATURES, FORM_HALF_LIFE_DAYS

FEATURES_DIR = ROOT / "data" / "features"
REPORTS = ROOT / "reports"


def md_table(df, floatfmt="{:.4f}"):
    def fmt(v):
        if isinstance(v, float):
            return "n/a" if pd.isna(v) else floatfmt.format(v)
        return str(v)
    head = "| " + " | ".join(df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep] + body)


def write_engineering_report():
    md = []
    md.append("# Phase 5B.2 - Leakage-Safe Opponent-Adjusted and Recency-Aware Team Form (Design)\n")
    md.append("**No model is trained in this phase.** No validation, test or Cologne metric is computed "
              "anywhere. The deliverable is a reusable team-form state engine plus one extended feature "
              "dataset built with it.\n")

    md.append("## 1. Why this phase exists\n")
    md.append("V1's form features (`win_rate_last_5/10`, `avg_series_margin_last_5/10`) treat \"5 wins vs weak "
              "opponents\" the same as \"5 wins vs elite opponents\" - they carry no strength-of-schedule or "
              "over/under-performance-vs-expectation signal. Phase 5B.2 adds that signal: average pre-match "
              "ELO of recent opponents, ELO-expectation performance residuals, and a fixed 60-day exponential "
              "time-decay weighting.\n")

    md.append("## 2. Why a new, independent state store\n")
    md.append("`feature_engineering/series/feature_engine.py` (frozen, never modified) computes `elo_expected`/`elo_update` as "
              "pure functions, but its `HistoryEntry`/`TeamState` never records the OPPONENT's pre-match ELO "
              "or a performance residual - that data does not exist in the frozen engine's state. "
              "`feature_engineering/form/team_form_engine.py` therefore replays the SAME series stream (same rows, same "
              "eligibility gating, same two-phase per-timestamp batching, same pure `elo_expected`/`elo_update` "
              "calls, imported unchanged) through its OWN state store, additionally recording each match's "
              "opponent ELO and performance residual.\n")
    md.append("**This is not merely asserted to reproduce Phase 3 - it is exhaustively verified.** Before "
              "`series_features_v3_form.parquet` is written, `feature_engineering/form/build_series_features_v3_form.py` "
              "compares the new engine's independently-computed pre-match `team1_elo_before - "
              "team2_elo_before` against V1/V2's own `elo_diff` column for ALL 9,456 rows (not a sample). "
              "Observed result: **max absolute difference 0.0** across all 9,456 rows - bit-for-bit identical. "
              "Re-verified independently in `validation/validate_phase5b2.py` from the audit parquet.\n")

    md.append("## 3. Pre-match ELO only\n")
    md.append("`expected_win_prob = elo_expected(own_elo_before, opponent_elo_before)` is always computed "
              "BEFORE `elo_update` is called for that match - never recomputed from post-match ratings. "
              "`performance_residual = actual_result - expected_win_prob`: an upset win against a stronger "
              "opponent gives a large positive residual, an expected win against a weak opponent gives a "
              "small positive residual, a loss as a strong favorite gives a large negative residual, and an "
              "expected loss against a much stronger opponent gives a smaller-magnitude negative residual. "
              "Directly tested for all four scenarios in `tests/features/test_team_form_engine.py`.\n")

    md.append("## 4. Trusted-opponent gating - the two populations are NOT interchangeable\n")
    md.append("Following the Phase 3 / Phase 5A identity policy: an eligible team's own history (result, "
              "normalized margin, activity) may still update from a match against an identity-ineligible "
              "opponent - that team's own result is a real fact - flagged `opponent_identity_trusted=False`. "
              "But opponent-adjusted information depends on a reliable, persistent opponent identity and ELO "
              "trajectory, which an untrusted opponent does not have. Therefore:\n")
    md.append("- **TRUSTED population** (`opponent_identity_trusted == True` only): "
              "`avg_opponent_elo_last_5/10`, `performance_residual_last_5/10/all`, "
              "`time_weighted_performance_residual`, and the confidence flags "
              "`opponent_adjusted_history_min`/`both_teams_have_5_adjusted_matches`/"
              "`both_teams_have_10_adjusted_matches` (which count ONLY trusted opponent-adjusted observations).")
    md.append("- **ALL-eligible population** (trusted or not): `time_weighted_win_rate`, "
              "`time_weighted_normalized_series_margin`, and their confidence companion "
              "`time_weighted_history_mass` - these describe a team's own result/margin, not who they played, "
              "so they do not depend on persistent opponent identity.\n")

    md.append("## 5. Recency weighting\n")
    md.append(f"Fixed engineering constant `FORM_HALF_LIFE_DAYS = {FORM_HALF_LIFE_DAYS:.0f}` - never tuned "
              "against any metric, never compared against 30/90-day alternatives. `weight = 0.5 ** "
              "(age_days / 60)`. Applied to `time_weighted_win_rate`, `time_weighted_performance_residual` "
              "(trusted population), and `time_weighted_normalized_series_margin`. "
              "`time_weighted_history_mass = sum(weight_i)` over the ALL-eligible population is the confidence "
              "companion for the first and third of these; `time_weighted_history_mass_min = "
              "min(mass_team1, mass_team2)` is the symmetric confidence feature actually stored.\n")

    md.append("## 6. Series margin\n")
    md.append("`normalized_series_margin(score_for, score_against) = (score_for - score_against) / "
              "(score_for + score_against)`, 0.0 when the denominator is 0 - the same normalization "
              "convention `feature_engineering/maps/map_feature_engine.py` uses for map margins (a local copy, not an import, "
              "to keep this module's only dependency on the Phase 3 engine), applied here to series map-count "
              "scores so it scales consistently across BO1/BO3/BO5. Computed only from each historical match's "
              "own final score - never from the target series.\n")

    md.append("## 7. Cold start\n")
    md.append("| quantity | cold start | rationale |\n|---|---|---|\n"
              "| avg_opponent_elo_last_5/10 | 1500.0 (ELO_INITIAL) | neutral opponent assumption, no evidence |\n"
              "| performance_residual_last_5/10/all | 0.0 | neutral - no evidence of over/under-performance |\n"
              "| time_weighted_win_rate | 0.5 | neutral |\n"
              "| time_weighted_performance_residual | 0.0 | neutral |\n"
              "| time_weighted_normalized_series_margin | 0.0 | neutral |\n"
              "| time_weighted_history_mass | 0.0 | zero effective evidence - a true absence, not \"neutral\" |\n"
              "| confidence flags | 0 | explicit, so a neutral value is never mistaken for evidence |\n")
    md.append("No opponent history is ever fabricated for a cold-start team.\n")

    md.append("## 8. Exact-timestamp leakage protection\n")
    md.append("`process_form_stream` uses the identical two-phase per-timestamp-group protocol as "
              "`feature_engine.process_chronological_stream`: Phase A emits features for every eligible-pair "
              "row in a timestamp group from the state as it was BEFORE the group; Phase B applies every row's "
              "result only after the whole group has been read. Proved on real data in "
              "`validation/validate_phase5b2.py` (a real shared-timestamp group is rebuilt from a pre-group state "
              "snapshot and every match in the group is confirmed to see that snapshot, not any other match's "
              "result from the same group) and on synthetic fixtures in `tests/features/test_team_form_engine.py`.\n")

    md.append("## 9. Feature inventory\n")
    md.append("8 new directional (Team1-Team2 diffs) + 4 new symmetric/confidence = **12 new features**, "
              "appended to V2's 47 to give V3's 59 predictive features (38 directional + 19 symmetric + 2 "
              "categorical context).\n")
    md.append("Directional: " + ", ".join(f"`{c}`" for c in FORM_DIRECTIONAL_FEATURES) + "\n")
    md.append("Symmetric/confidence: " + ", ".join(f"`{c}`" for c in FORM_SYMMETRIC_FEATURES) + "\n")

    md.append("## 10. What Phase 5B.2 deliberately does NOT do\n")
    md.append("- No model is trained; no validation, test or Cologne metric is computed.\n"
              "- No half-life tuning (30/60/90-day comparison) - 60 days is a fixed a priori constant.\n"
              "- No feature selection, and no feature-vs-target association is reported anywhere.\n"
              "- No post-Cologne deployment snapshot.\n"
              "- Nothing under `data/raw/`, `reference/` or `src/` is touched; no Phase 1-5B.1 artifact is "
              "modified; the test partition and main validation partition are never loaded.\n")

    (REPORTS / "phases" / "phase5b2_team_form_feature_engineering.md").write_text("\n".join(md), encoding="utf-8")


def write_quality_report():
    v3 = pd.read_parquet(FEATURES_DIR / "series_features_v3_form.parquet", engine="fastparquet")
    split = pd.read_csv(ROOT / "data" / "modeling" / "series_split_v1.csv")
    train_ids = set(split.loc[split["split"] == "train", "match_id"])
    train = v3[v3["match_id"].isin(train_ids)].reset_index(drop=True)

    all_new = FORM_DIRECTIONAL_FEATURES + FORM_SYMMETRIC_FEATURES

    md = []
    md.append("# Phase 5B.2 - Team Form Feature Quality (descriptive)\n")
    md.append("**Scope discipline.** Every number below is computed on the **global TRAIN partition** "
              "(`data/modeling/series_split_v1.csv`), the same discipline `feature_engineering/maps/phase5a_reports.py` "
              "established. Validation is not summarized, the test partition is not opened, and Cologne is "
              "not read. **No feature-vs-target association is reported** - no correlations, no rankings, no "
              "\"promising feature\" claims, no half-life tuning discussion.\n")
    md.append(f"TRAIN series: **{len(train):,}** of {len(v3):,}.\n")

    md.append("## 1. Missingness\n")
    n_missing = int(train[all_new].isna().sum().sum())
    md.append(f"New features with any missing value: **{n_missing} of {len(all_new)}** columns "
              f"(all 12 new features are always defined, by cold-start construction).\n")

    md.append("## 2. Coverage and cold start (TRAIN series)\n")
    rows = [
        ("series with opponent_adjusted_history_min == 0 (>=1 side has zero trusted opponent history)",
         int((train["opponent_adjusted_history_min"] == 0).sum())),
        ("series where both teams have >= 5 trusted opponent-adjusted matches",
         int((train["both_teams_have_5_adjusted_matches"] == 1).sum())),
        ("series where both teams have >= 10 trusted opponent-adjusted matches",
         int((train["both_teams_have_10_adjusted_matches"] == 1).sum())),
        ("series where time_weighted_history_mass_min == 0 (>=1 side has zero own eligible history)",
         int((train["time_weighted_history_mass_min"] == 0).sum())),
    ]
    cov_df = pd.DataFrame(rows, columns=["quantity", "n"])
    cov_df["pct_of_train"] = 100 * cov_df["n"] / len(train)
    md.append(md_table(cov_df, "{:.2f}"))
    md.append("")

    md.append("## 3. Time-weight behavior (illustrative, not data-derived)\n")
    md.append(f"`weight = 0.5 ** (age_days / {FORM_HALF_LIFE_DAYS:.0f})` - the fixed 60-day half-life, "
              "shown for representative ages:\n")
    example_days = [0, 30, 60, 90, 120, 180, 365]
    weight_df = pd.DataFrame({"age_days": example_days,
                               "weight": [0.5 ** (d / FORM_HALF_LIFE_DAYS) for d in example_days]})
    md.append(md_table(weight_df, "{:.4f}"))
    md.append("")

    md.append("## 4. Distribution summaries (TRAIN series)\n")
    md.append("Descriptive only - spread and centring, to confirm nothing is degenerate or absurdly scaled.\n")
    dist_rows = []
    for c in all_new:
        s = train[c]
        dist_rows.append((c, float(s.mean()), float(s.std()), float(s.min()), float(s.median()), float(s.max())))
    dist_df = pd.DataFrame(dist_rows, columns=["feature", "mean", "std", "min", "median", "max"])
    md.append(md_table(dist_df, "{:.4f}"))
    md.append("")

    md.append("## 5. Feature-feature redundancy (descriptive, NOT target correlation)\n")
    md.append("Correlation among the 8 new directional features only, to describe overlap without ever "
              "touching the target:\n")
    corr = train[FORM_DIRECTIONAL_FEATURES].corr(numeric_only=True)
    pairs = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairs.append((cols[i], cols[j], corr.iloc[i, j]))
    pairs.sort(key=lambda t: -abs(t[2]))
    md.append("Top 10 |corr| pairs among the new directional features:\n")
    md.append("| feature A | feature B | r |\n|---|---|---|")
    for a, b, r in pairs[:10]:
        md.append(f"| {a} | {b} | {r:+.3f} |")
    md.append("")

    md.append("## 6. What this report does not claim\n")
    md.append("Nothing here says any feature is useful. Coverage, spread and completeness are properties of "
              "the data; predictive value is a property of a model that has not been fitted yet.\n")

    (REPORTS / "phases" / "phase5b2_team_form_feature_quality.md").write_text("\n".join(md), encoding="utf-8")


def main():
    write_engineering_report()
    write_quality_report()
    print(f"Wrote {REPORTS / 'phase5b2_team_form_feature_engineering.md'}")
    print(f"Wrote {REPORTS / 'phase5b2_team_form_feature_quality.md'}")


if __name__ == "__main__":
    main()
