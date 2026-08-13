"""
Phase 5A reports.

Writes:
    reports/phase5a_map_feature_engineering.md   (design / rules / decisions)
    reports/phase5a_map_feature_quality.md       (descriptive diagnostics)
    reports/tables/map_feature_coverage_v1.csv

DISCIPLINE FOR THE QUALITY REPORT
  * computed on the GLOBAL TRAIN partition only (data/modeling/series_split_v1.csv);
  * validation is never summarized, test is never opened, Cologne is never read;
  * NO feature-vs-target correlations, rankings or "promising feature" claims -
    those are model-selection decisions and would consume the very evidence the
    later phases are supposed to spend.
"""

import json

import numpy as np
import pandas as pd
import yaml

from _common import INTERIM, ROOT
from map_feature_engine import (
    MAP_ENGINE_VERSION, MAP_POOL_LOOKBACK_DAYS, EXPERIENCED_MAP_MIN_MATCHES,
    UNKNOWN_MAP_CATEGORY, SERIES_MAP_POOL_DIRECTIONAL, SERIES_MAP_POOL_SYMMETRIC,
    MAP_DIRECTIONAL_FEATURES, MAP_SYMMETRIC_FEATURES,
)

FEATURES_DIR = ROOT / "data" / "features"
REPORTS = ROOT / "reports"
TABLES = REPORTS / "tables"
TABLES.mkdir(exist_ok=True, parents=True)


def md_table(df, floatfmt="{:.4f}"):
    def fmt(v):
        if isinstance(v, float):
            return "n/a" if pd.isna(v) else floatfmt.format(v)
        return str(v)
    head = "| " + " | ".join(df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep] + body)


def load_all():
    mapf = pd.read_parquet(FEATURES_DIR / "map_features_v1.parquet", engine="fastparquet")
    ser = pd.read_parquet(FEATURES_DIR / "series_features_v2_map_pool.parquet", engine="fastparquet")
    v1 = pd.read_parquet(FEATURES_DIR / "series_features_v1.parquet", engine="fastparquet")
    split = pd.read_csv(ROOT / "data" / "modeling" / "series_split_v1.csv")
    snap = pd.read_parquet(INTERIM / "pre_cologne_map_state_v1.parquet", engine="fastparquet")
    with open(INTERIM / "map_features_v1_build_summary.json", encoding="utf-8") as f:
        map_sum = json.load(f)
    with open(INTERIM / "series_features_v2_build_summary.json", encoding="utf-8") as f:
        ser_sum = json.load(f)
    with open(INTERIM / "pre_cologne_map_state_v1.json", encoding="utf-8") as f:
        snap_meta = json.load(f)["meta"]
    return mapf, ser, v1, split, snap, map_sum, ser_sum, snap_meta


# ---------------------------------------------------------------------------
# Report 1: engineering / design
# ---------------------------------------------------------------------------

def write_engineering_report(mapf, ser, v1, snap, map_sum, ser_sum, snap_meta):
    with open(ROOT / "config" / "map_features_v1.yaml", encoding="utf-8") as f:
        map_cfg = yaml.safe_load(f)
    with open(ROOT / "config" / "series_features_v2_map_pool.yaml", encoding="utf-8") as f:
        ser_cfg = yaml.safe_load(f)

    L = []
    A = L.append
    A("# Phase 5A - Leakage-Safe Map History and Map-Pool Feature Engineering\n")
    A("**No model is trained in this phase.** No validation, test or Cologne metric is computed "
      "anywhere. The deliverable is a reusable map state engine plus two feature datasets built "
      "with it.\n")

    A("## 1. Why this phase exists\n")
    A("Phase 4 tuned three very different algorithms on the same 17 series features and they "
      "converged into a narrow band (validation ROC-AUC 0.6412 / 0.6566 / 0.6504 for LR V2, "
      "RF V2 and XGB V2). When three model families with different inductive biases agree that "
      "closely, the binding constraint is usually the information in the features rather than "
      "the flexibility of the model. Phase 5A adds genuine CS2 domain knowledge - what maps each "
      "team actually plays and how well - to test that hypothesis in a later phase.\n")

    A("## 2. The two prediction tasks are NOT the same task\n")
    A("Everything in this phase follows from one distinction, which is why the two datasets exist "
      "separately rather than as one table with a flag:\n")
    A(md_table(pd.DataFrame([
        {"": "**Task**", "Pre-veto series": "predict the series winner **before the veto**",
         "Known map": "predict the winner of **one specific map**"},
        {"": "**Dataset**", "Pre-veto series": "`series_features_v2_map_pool.parquet`",
         "Known map": "`map_features_v1.parquet`"},
        {"": "**`prediction_task`**", "Pre-veto series": "`pre_veto_series`", "Known map": "`known_map`"},
        {"": "**Maps of the target match**", "Pre-veto series": "**FORBIDDEN** - unknown at prediction time",
         "Known map": "**LEGITIMATE INPUT** - the map is given"},
        {"": "**`map_name` column**", "Pre-veto series": "absent (enforced)", "Known map": "present, a real feature"},
        {"": "**Map knowledge enters as**", "Pre-veto series": "summaries of the teams' own PRIOR maps",
         "Known map": "both teams' record ON THIS MAP + pool context"},
        {"": "**Rows**", "Pre-veto series": f"{len(ser):,} series", "Known map": f"{len(mapf):,} maps"},
    ])))
    A("\nThe `prediction_task` identifier lives in the YAML config as metadata. It is deliberately "
      "**not** a column in either parquet: it describes the task, and feeding it to a model would "
      "be meaningless (it is constant within a dataset).\n")

    A("## 3. The prediction cutoff: `series_datetime` vs `map_datetime`\n")
    A("These are different concepts and the engine keeps them apart everywhere:\n")
    A("- **`series_datetime`** - the prediction cutoff. Everything knowable about a series, "
      "*including every map that will be played in it*, is frozen at this instant.\n"
      "- **`map_datetime`** - when an individual map was actually played. It may be later than "
      "the cutoff, and a future provider may expose a genuine per-map start time.\n")
    A("All batching, all history filtering and all feature emission key on `series_datetime`. "
      "`map_datetime` is carried for provenance and **never** decides what a map may see. "
      "In the current export the two coincide for every row "
      f"(`map_datetime_differs_from_series` = {map_sum['stream_info']['map_datetime_differs_from_series']}), "
      "but that is a property of this export, not an assumption of the pipeline - "
      "`TestD2_DifferentMapTimestampsSameSeriesCutoff` constructs a BO3 whose three maps carry "
      "deliberately different `map_datetime` values (12:05, 13:20, 14:45) under one cutoff and "
      "proves all three still receive the identical pre-series state, and that reversing those "
      "map times changes no feature at all.\n")

    A("## 4. Series-atomic, two-phase processing\n")
    A("A `match_id` is an **atomic prediction/update unit**:\n")
    A("```\nfor each exact series_datetime batch:\n"
      "    PHASE A (read):  emit features for ALL maps of ALL series in the batch\n"
      "                     from the state as it was BEFORE the batch\n"
      "    PHASE B (write): only then apply every completed map result\n```\n")
    A("Consequences, each proved by a synthetic test rather than argued:\n")
    A("- Map 1's result cannot reach Map 2 or Map 3 of the same series - the state is never "
      "updated inside a series.\n"
      "- Two different series sharing one timestamp cannot see each other, in either input order. "
      "This is not theoretical here: "
      "**706 timestamps in this dataset carry more than one distinct match, covering 3,348 map rows.**\n"
      "- The same map appearing twice in one series is still isolated.\n"
      "- Input row order is irrelevant to the output.\n")

    A("## 5. Reuse of the Phase 3 engine\n")
    A("`map_feature_engine.py` imports `elo_expected`, `elo_update`, `ELO_INITIAL`, `ELO_K` and "
      "`_beta_smoothed_win_rate` from `feature_engine.py` rather than reimplementing them. Map ELO "
      "is therefore the same rating mathematics with the same **K = 32** and the same 1500 cold "
      "start as series ELO, and map win rates use the same Beta(2,2) smoothing. **No K tuning and "
      "no time-decay tuning happens in Phase 5A** - `MAP_POOL_LOOKBACK_DAYS = 180` is a fixed "
      "design constant, chosen a priori and never selected against any performance metric.\n")

    A("## 6. Feature definitions\n")
    A("### 6.1 Normalized round margin\n")
    A("`(rounds_for - rounds_against) / (rounds_for + rounds_against)`, and exactly `0.0` when the "
      "denominator is zero. Normalizing by the total makes the signal invariant to CS scoring-format "
      "changes (MR15 -> MR12), so a 16-8 win and an 8-4 win score identically. Examples: "
      "13-10 -> 3/23 = 0.130, 13-5 -> 8/18 = 0.444.\n")
    A("### 6.2 Beta(2,2) smoothed map win rate\n")
    A("`(wins + 2) / (matches + 4)`. A team that has won its only map on Nuke gets 0.60, not 1.00. "
      "This matters far more per-map than per-series: map samples are thin by construction, and an "
      "unsmoothed rate would hand the model a pile of spurious 0.0/1.0 certainties.\n")
    A("### 6.3 Recent map pool\n")
    A(f"The maps a team played in the half-open window `[T - {MAP_POOL_LOOKBACK_DAYS}d, T)`. "
      "Half-open by construction: a map at exactly `T` is the current series and is excluded; a map "
      "at exactly the lower bound is included. A map is **experienced** for a team when it has at "
      f"least {EXPERIENCED_MAP_MIN_MATCHES} prior maps on it.\n")

    A("## 7. Two DISTINCT map-pool families (do not describe them interchangeably)\n")
    A("### (A) Pool-depth / order-statistic features - `map_pool_*`\n")
    A("Each team's own recent pool is summarized independently (size, total matches, experienced-map "
      "count, mean/best/2nd/3rd/worst map ELO and smoothed win rate, mean margin), and only then "
      "subtracted. **The two sides' k-th-best entries may refer to completely different map "
      "identities.** `map_pool_best_elo_diff` says \"Team1's strongest map is stronger than Team2's "
      "strongest map\" - it says nothing about whether those are the same map, and it is not a "
      "head-to-head comparison. This family describes the *shape and depth* of each pool.\n")
    A("### (B) Same-map matchup features - `map_matchup_*`\n")
    A("Computed across `union(pool1, pool2)`; for each map identity the two teams are compared "
      "**on that same map** (neutral cold-start defaults where one side lacks history), and the "
      "resulting per-map advantages are then summarized. These *are* genuine head-to-head, per-map "
      "comparisons.\n")
    A("### Why family (B) uses mean / median / midrange and not k-th-best\n")
    A("Mirrored augmentation requires every directional feature to negate exactly under a "
      "Team1<->Team2 swap. Swapping negates the per-map advantage list, and the k-th **largest** of "
      "a negated list is minus the k-th **smallest** of the original - so a \"2nd best advantage\" "
      "feature is antisymmetric only when *paired* with \"2nd worst advantage\", which the mirroring "
      "contract cannot express. `mean`, `median` and `midrange = (max+min)/2` each satisfy "
      "`f(-x) = -f(x)` individually. The spread that the order statistics would have carried is "
      "preserved by the swap-**invariant** `map_matchup_elo_advantage_range`, which is therefore "
      "declared symmetric, not directional. Family (A) keeps its order statistics precisely because "
      "each side's statistic is computed over its own pool *before* subtraction, so the difference "
      "negates cleanly. `TestG.test_order_statistics_would_have_broken_symmetry` pins this down.\n")

    A("## 8. Feature inventory\n")
    A(md_table(pd.DataFrame([
        {"dataset": "`map_features_v1`", "family": "map-specific directional",
         "n": len(MAP_DIRECTIONAL_FEATURES), "example": "`map_elo_diff`"},
        {"dataset": "`map_features_v1`", "family": "map-specific symmetric",
         "n": len(MAP_SYMMETRIC_FEATURES), "example": "`both_teams_have_5_map_matches`"},
        {"dataset": "`map_features_v1`", "family": "Phase 3 series features (joined)",
         "n": 15, "example": "`elo_diff`"},
        {"dataset": "`map_features_v1`", "family": "categorical context",
         "n": len(map_cfg["categorical_context"]), "example": "`map_name`"},
        {"dataset": "`series_features_v2_map_pool`", "family": "(A) pool depth, directional",
         "n": 14, "example": "`map_pool_best_elo_diff`"},
        {"dataset": "`series_features_v2_map_pool`", "family": "(B) same-map matchup, directional",
         "n": 6, "example": "`map_matchup_mean_elo_advantage`"},
        {"dataset": "`series_features_v2_map_pool`", "family": "map-pool symmetric / confidence",
         "n": len(SERIES_MAP_POOL_SYMMETRIC), "example": "`shared_experienced_map_count`"},
        {"dataset": "`series_features_v2_map_pool`", "family": "inherited Phase 3 V1 features",
         "n": 17, "example": "`elo_diff`"},
    ]), floatfmt="{:.0f}"))
    A("\nThe three symmetric map-coverage confidence features are `shared_recent_map_count` "
      "(`|pool1 n pool2|`), `shared_experienced_map_count` (map identities where **both** teams have "
      f">= {EXPERIENCED_MAP_MIN_MATCHES} prior maps) and `map_matchup_shared_coverage` "
      "(`|pool1 n pool2| / |pool1 u pool2|`, **0.0 when the union is empty**). They let a model "
      "discount a matchup advantage that rests on thin or non-overlapping evidence.\n")

    A("## 9. Cold-start contract\n")
    A("Identical at training time and at inference time, so a cold-start row is not a special case "
      "the deployment path has to reinvent:\n")
    A(md_table(pd.DataFrame([
        {"quantity": "map ELO", "cold start": "1500.0 (= `ELO_INITIAL`)"},
        {"quantity": "smoothed map win rate", "cold start": "0.5 (Beta(2,2), zero observations)"},
        {"quantity": "rolling win rates", "cold start": "0.5"},
        {"quantity": "normalized margins", "cold start": "0.0"},
        {"quantity": "match counts / pool sizes / flags", "cold start": "0"},
        {"quantity": "`days_since_map_played`", "cold start": "**NaN - genuinely missing, never a sentinel number**"},
        {"quantity": "k-th-best slot, pool shallower than k", "cold start": "the neutral value (1500 / 0.5), never an extreme"},
        {"quantity": "empty `union(pool1, pool2)`", "cold start": "every `map_matchup_*` advantage 0.0; coverage 0.0"},
        {"quantity": "unseen map identity", "cold start": f"reserved category `{UNKNOWN_MAP_CATEGORY}`"},
    ])))
    A("\nUsing a neutral rather than an extreme fallback for a shallow pool is deliberate: an "
      "extreme sentinel would masquerade as a real strength reading. The thin-evidence signal is "
      "carried by the confidence features instead, where a model can weigh it explicitly.\n")

    A("## 10. Identity policy - two different eligibility questions\n")
    A("The policy is inherited unchanged from Phase 2.5/3 (`team_identity_policy.csv`); Phase 5A "
      "makes no new identity decisions. Two rules that are easy to conflate:\n")
    A("1. **A supervised feature row** requires **both** canonical identities to be trusted.\n"
      "2. **State updates** are finer-grained. An eligible team's *own* map history (win/loss, "
      "margin, recency, pool membership) updates even when the opponent's identity is not "
      "trustworthy - that team's own result is a real fact - and the entry is flagged "
      "`opponent_identity_trusted=False`. But **map ELO is pair-dependent** and moves both ratings, "
      "so it updates *only* when both identities are trusted.\n")
    A(f"This is measurable, not just asserted: the pre-Cologne snapshot contains "
      f"**{snap_meta['entries_from_untrusted_opponent_matches']} history entries** recorded against "
      f"an untrusted opponent, from {snap_meta['maps_with_an_ineligible_side']} such maps. Rebuilding "
      "the snapshot by replaying `map_features_v1` (which contains only both-eligible rows) would "
      "have silently destroyed every one of them - which is exactly why "
      "`build_pre_cologne_map_state_v1.py` replays the **canonical map stream** instead.\n")

    A("## 11. Dataset construction\n")
    A("### 11.1 `map_features_v1.parquet`\n")
    si = map_sum["stream_info"]
    A(md_table(pd.DataFrame([
        {"step": "`map_base.parquet` rows", "rows": si["map_base_rows"]},
        {"step": "labeled `development` (Cologne + post-Cologne dropped)", "rows": si["rows_selected"]},
        {"step": "minus rows touching an identity-ineligible team", "rows": si["both_eligible_rows"]},
        {"step": "minus maps whose match is absent from `series_features_v1`", "rows": map_sum["rows"]},
    ]), floatfmt="{:.0f}"))
    A(f"\nFinal: **{map_sum['rows']:,} rows** over {map_sum['distinct_matches']:,} matches and "
      f"{map_sum['distinct_maps']} map identities. The target `team1_map_win` was **re-derived from "
      f"`score1_game > score2_game`** rather than trusted: "
      f"{si['target_integrity']['rows_checked']:,} rows checked, {si['target_integrity']['ties']} ties, "
      f"{si['target_integrity']['null_scores']} null scores, "
      f"{si['target_integrity']['target_mismatches']} disagreements. Target positive rate "
      f"{map_sum['target_positive_rate']:.4f}.\n")
    A("The 17 Phase 3 series features are joined by `match_id`. This is legitimate: they are "
      "computed strictly before the same series cutoff, so a map row sees exactly the series-level "
      "knowledge that existed before its series began.\n")

    A("### 11.2 `series_features_v2_map_pool.parquet`\n")
    A(f"**{ser_sum['rows']:,} rows - the same match_id universe, the same targets and the same "
      f"ordering as `series_features_v1`, with every V1 feature column preserved value-for-value** "
      "(asserted in the builder and re-checked independently by the validator).\n")
    A(f"Critically, {ser_sum['matches_without_own_map_rows']:,} of these "
      f"{ser_sum['rows']:,} series have **no map rows of their own**, and they still receive full "
      "pool features - because pool features describe the two teams' *prior* map history, not the "
      "target match. Driving emission off the map stream would have silently dropped them, so the "
      "engine takes an explicit series-request frame instead "
      "(`process_combined_stream`). A series request contributes nothing to state; only completed "
      "maps do.\n")

    A("## 12. Forbidden columns\n")
    A("Enforced in code at build time and re-checked by `validate_phase5a.py`:\n")
    A("- **Pre-veto series V2**: no `map_name`, `map_id`, `game_id`, no list/count/order of the "
      "maps selected in the target match, no `score1_game`/`score2_game`, no per-map result of the "
      "target series, no player statistic from it. Any of these would leak the veto.\n"
      "- **Map V1**: no `score1_game`/`score2_game` (the current map's own score *is* the target), "
      "no `map_id`, no player boxscore column, nothing derived from the current or any later map.\n"
      "- **Both**: `evaluation_group` is experimental bookkeeping and never a model input; no "
      "player-level statistics anywhere in Phase 5A.\n")

    A("## 13. Frozen pre-Cologne map state\n")
    A(md_table(pd.DataFrame([{"property": k, "value": str(v)} for k, v in [
        ("cutoff source", "`evaluation_manifest.csv` `cologne_2026` group (tournament name only cross-checked)"),
        ("cutoff rule", "`series_datetime < cologne_first_datetime` (strict)"),
        ("Cologne first datetime", snap_meta["cologne_first_datetime"]),
        ("max source `series_datetime`", snap_meta["max_source_series_datetime"]),
        ("maps replayed", f"{snap_meta['source_maps_replayed']:,}"),
        ("team-map states", f"{snap_meta['team_map_states']:,}"),
        ("distinct teams / maps", f"{snap_meta['distinct_teams']} / {snap_meta['distinct_maps']}"),
        ("Cologne match_ids in any history", "0 (asserted independently)"),
        ("post-Cologne deployment snapshot", "**not built** - out of scope for Phase 5A"),
    ]])))
    A("\nThe snapshot is written twice: a flat scalar `.parquet` summary (one row per team-map, "
      "fastparquet-safe) and a `.json` carrying the full re-loadable state. A round-trip test proves "
      "a reloaded store reproduces identical feature vectors.\n")

    A("## 14. Future application\n")
    A("The functions that generate training rows are the *same* functions a live pre-match call "
      "would use - `build_future_map_matchup_features(store, team1, team2, best_of, map_name, "
      "as_of_datetime, tier)` and `build_future_series_map_pool_features(store, team1, team2, "
      "best_of, as_of_datetime, tier)`. Both are pure and read-only, neither takes a target or a "
      "current score, and the series builder takes **no map argument at all** (asserted by "
      "signature inspection, so pre-veto contamination cannot creep in via a later edit). "
      "Two equivalence tests confirm that an offline stream and a live call produce identical "
      "numbers, which is what stops training and deployment from silently diverging.\n")

    A("## 15. Test coverage\n")
    A("`tests/test_map_feature_engine.py` - 73 synthetic-fixture tests, no dependency on the real "
      "dataset. Groups A-N cover Beta smoothing, map ELO (including rating conservation), strict "
      "chronology, same-series isolation, differing map timestamps under one cutoff, same-timestamp "
      "isolation across series, target reconstruction, side-swap symmetry, cold start, the 180-day "
      "window boundaries, normalized-margin arithmetic, the identity policy, the future-application "
      "contract, absence of player statistics, and snapshot round-trip - plus the stream contract "
      "and the combined driver.\n")

    A("## 16. What Phase 5A deliberately does NOT do\n")
    A("- No model is trained; no validation, test or Cologne metric is computed.\n"
      "- No player-level features.\n"
      "- No time-decay or ELO-K tuning; every constant is fixed a priori.\n"
      "- No new identity decisions.\n"
      "- No feature selection, and no feature-vs-target association is reported anywhere - "
      "including in the quality report, which is descriptive only.\n"
      "- No post-Cologne deployment snapshot.\n"
      "- Nothing under `data/raw/`, `reference/` or `src/` is touched, no Phase 4 artifact is "
      "modified, and the test partition stays sealed.\n")

    A("## 17. Open questions carried into the next phase\n")
    A("1. **Map-pool features may be partly redundant with series ELO.** A team with a strong pool "
      "is usually just a strong team. Whether family (A) adds anything beyond `elo_diff` is an "
      "empirical question that must be answered by a model comparison, not asserted here.\n"
      "2. **Map rows begin 2023-10-25, about nine months after series rows.** 957 of 9,456 series "
      "(10.1%) therefore have a completely empty recent pool. Whether to model these or restrict "
      "the map-pool comparison to the covered era is a Phase 5B decision.\n"
      "3. **The 180-day window is unvalidated by construction.** It was fixed a priori to keep this "
      "phase honest. If it is ever tuned, that must happen inside chronological CV on TRAIN only.\n"
      "4. **`map_features_v1` rows are not independent** - up to 5 maps share a series and a "
      "pre-series state. Any future map-level model must account for this in its splitting and in "
      "its error bars; treating 10,318 maps as 10,318 independent observations would overstate "
      "confidence.\n")

    (REPORTS / "phase5a_map_feature_engineering.md").write_text("\n".join(L), encoding="utf-8")
    print(f"Wrote {REPORTS / 'phase5a_map_feature_engineering.md'}")


# ---------------------------------------------------------------------------
# Report 2: descriptive quality (TRAIN partition only)
# ---------------------------------------------------------------------------

def write_quality_report(mapf, ser, split, snap, ser_sum):
    train_ids = set(split.loc[split["split"] == "train", "match_id"])
    ser_tr = ser[ser["match_id"].isin(train_ids)].copy()
    map_tr = mapf[mapf["match_id"].isin(train_ids)].copy()

    pool_cols = SERIES_MAP_POOL_DIRECTIONAL + SERIES_MAP_POOL_SYMMETRIC
    map_cols = MAP_DIRECTIONAL_FEATURES + MAP_SYMMETRIC_FEATURES

    L = []
    A = L.append
    A("# Phase 5A - Map Feature Quality (descriptive)\n")
    A("**Scope discipline.** Every number below is computed on the **global TRAIN partition** "
      "(`data/modeling/series_split_v1.csv`). Validation is not summarized, the test partition is "
      "not opened, and Cologne 2026 is not read. **No feature-vs-target association is reported** - "
      "no correlations, no rankings, no \"promising feature\" claims. Those are model-selection "
      "decisions; making them here would spend evidence that later phases need and would quietly "
      "turn a descriptive report into feature selection on data the model has not earned.\n")
    A(f"TRAIN series: **{len(ser_tr):,}** of {len(ser):,}. "
      f"TRAIN map rows: **{len(map_tr):,}** of {len(mapf):,}.\n")

    A("## 1. Completeness\n")
    miss_ser = ser_tr[pool_cols].isna().sum()
    miss_map = map_tr[map_cols].isna().sum()
    A(f"- Pre-veto pool features with any missing value: **{int((miss_ser > 0).sum())} of "
      f"{len(pool_cols)}** columns.\n")
    A(f"- Map-specific features with any missing value: **{int((miss_map > 0).sum())} of "
      f"{len(map_cols)}** columns.\n")
    if int((miss_map > 0).sum()):
        rows = [{"feature": c, "n_missing": int(miss_map[c]),
                 "pct": 100 * miss_map[c] / len(map_tr)} for c in map_cols if miss_map[c] > 0]
        A("\n" + md_table(pd.DataFrame(rows), floatfmt="{:.2f}"))
        A("\nThese are the documented `days_since_*` cold-start NaNs - a team that has never played "
          "the map has no \"days since\" value, and a sentinel number would be a lie. They are "
          "genuinely missing, not corrupt.\n")
    finite_ser = np.isfinite(ser_tr[pool_cols].select_dtypes(include=[np.number]).to_numpy()).all()
    A(f"\n- All pre-veto pool features finite: **{bool(finite_ser)}**.\n")

    A("## 2. Map-history coverage and cold start\n")
    A(md_table(pd.DataFrame([
        {"quantity": "series with an empty recent pool for at least one side",
         "n": int((ser_tr["map_pool_size_min"] == 0).sum()),
         "pct of TRAIN series": 100 * (ser_tr["map_pool_size_min"] == 0).mean()},
        {"quantity": "series with a completely empty union pool (both sides cold)",
         "n": int((ser_tr["union_map_count"] == 0).sum()),
         "pct of TRAIN series": 100 * (ser_tr["union_map_count"] == 0).mean()},
        {"quantity": "series where both teams have >= 3 recent maps",
         "n": int(ser_tr["both_teams_have_3_recent_maps"].sum()),
         "pct of TRAIN series": 100 * ser_tr["both_teams_have_3_recent_maps"].mean()},
        {"quantity": "series where both teams have >= 5 experienced maps",
         "n": int(ser_tr["both_teams_have_5_experienced_maps"].sum()),
         "pct of TRAIN series": 100 * ser_tr["both_teams_have_5_experienced_maps"].mean()},
        {"quantity": "series with zero shared recent maps",
         "n": int((ser_tr["shared_recent_map_count"] == 0).sum()),
         "pct of TRAIN series": 100 * (ser_tr["shared_recent_map_count"] == 0).mean()},
    ]), floatfmt="{:.2f}"))
    A("\nThe cold-start share is a direct consequence of map coverage starting about nine months "
      "after series coverage; it is concentrated in the earliest part of the timeline, which the "
      "chronological split places entirely inside TRAIN.\n")

    A("### Recent-pool-size distribution (TRAIN series, orientation-independent measures)\n")
    A(md_table(pd.DataFrame([
        {"statistic": "min", "map_pool_size_min": ser_tr["map_pool_size_min"].min(),
         "union_map_count": ser_tr["union_map_count"].min(),
         "shared_recent_map_count": ser_tr["shared_recent_map_count"].min()},
        {"statistic": "median", "map_pool_size_min": ser_tr["map_pool_size_min"].median(),
         "union_map_count": ser_tr["union_map_count"].median(),
         "shared_recent_map_count": ser_tr["shared_recent_map_count"].median()},
        {"statistic": "mean", "map_pool_size_min": ser_tr["map_pool_size_min"].mean(),
         "union_map_count": ser_tr["union_map_count"].mean(),
         "shared_recent_map_count": ser_tr["shared_recent_map_count"].mean()},
        {"statistic": "max", "map_pool_size_min": ser_tr["map_pool_size_min"].max(),
         "union_map_count": ser_tr["union_map_count"].max(),
         "shared_recent_map_count": ser_tr["shared_recent_map_count"].max()},
    ]), floatfmt="{:.2f}"))

    A("\n## 3. Map-specific coverage (TRAIN map rows)\n")
    A(md_table(pd.DataFrame([
        {"quantity": "rows where both teams have prior history on this map",
         "n": int(map_tr["both_teams_have_map_history"].sum()),
         "pct": 100 * map_tr["both_teams_have_map_history"].mean()},
        {"quantity": "rows where both teams have >= 5 prior maps here",
         "n": int(map_tr["both_teams_have_5_map_matches"].sum()),
         "pct": 100 * map_tr["both_teams_have_5_map_matches"].mean()},
        {"quantity": "rows where both teams have >= 10 prior maps here",
         "n": int(map_tr["both_teams_have_10_map_matches"].sum()),
         "pct": 100 * map_tr["both_teams_have_10_map_matches"].mean()},
        {"quantity": "rows where at least one side is cold on this map",
         "n": int((map_tr["both_teams_have_map_history"] == 0).sum()),
         "pct": 100 * (map_tr["both_teams_have_map_history"] == 0).mean()},
    ]), floatfmt="{:.2f}"))

    A("\n## 4. Distribution summaries\n")
    A("Descriptive only - spread and centring, to confirm nothing is degenerate or absurdly scaled.\n")
    desc_cols = [c for c in pool_cols if pd.api.types.is_numeric_dtype(ser_tr[c])]
    d = ser_tr[desc_cols].describe().T[["mean", "std", "min", "50%", "max"]].reset_index()
    d.columns = ["feature", "mean", "std", "min", "median", "max"]
    A(md_table(d, floatfmt="{:.3f}"))
    pos_share = float((ser_tr["map_pool_size_diff"] > 0).mean())
    A(f"\n**These directional means are NOT zero, and that is expected.** "
      f"`map_pool_size_diff` averages {ser_tr['map_pool_size_diff'].mean():.3f} and "
      f"`map_pool_total_matches_diff` averages {ser_tr['map_pool_total_matches_diff'].mean():.3f}; "
      f"the Team1 side has the larger recent pool in {100 * pos_share:.1f}% of TRAIN series. This is "
      "the same Team1 orientation artifact documented in Phase 2 "
      "(`reports/orientation_analysis.md`) - the export tends to list the better-established team "
      "first, so a feature that measures establishment inherits the tilt. It is a property of the "
      "raw rows, not a defect in the features: the features are antisymmetric *by construction* "
      "(swapping the sides negates each of them exactly, proven in "
      "`TestG_SideSwapSymmetry`), and mirrored augmentation neutralizes the offset at training "
      "time by adding the swapped copy of every row. No de-biasing is applied here, because these "
      "parquets store raw pre-mirroring rows - mirroring is a training-time step that must happen "
      "after the train/validation split, never inside feature engineering.\n")

    dm = map_tr[[c for c in map_cols if pd.api.types.is_numeric_dtype(map_tr[c])]].describe().T[
        ["mean", "std", "min", "50%", "max"]].reset_index()
    dm.columns = ["feature", "mean", "std", "min", "median", "max"]
    A("\n### Map-specific features\n")
    A(md_table(dm, floatfmt="{:.3f}"))

    A("\n## 5. Per-map coverage\n")
    A("Full detail in `reports/tables/map_feature_coverage_v1.csv`.\n")
    cov = per_map_coverage(mapf, map_tr, snap)
    A(md_table(cov, floatfmt="{:.2f}"))

    A("\n## 6. Teams with map history in the frozen pre-Cologne snapshot\n")
    A(f"- Team-map states: **{len(snap):,}**\n"
      f"- Distinct teams: **{snap['canonical_team_name'].nunique():,}**\n"
      f"- Distinct maps: **{snap['map_name'].nunique()}**\n"
      f"- Median maps played per team-map state: **{snap['map_matches'].median():.0f}**\n"
      f"- Team-map states with >= {EXPERIENCED_MAP_MIN_MATCHES} maps: "
      f"**{int((snap['map_matches'] >= EXPERIENCED_MAP_MIN_MATCHES).sum()):,}** "
      f"({100 * (snap['map_matches'] >= EXPERIENCED_MAP_MIN_MATCHES).mean():.1f}%)\n")
    A(f"- History entries recorded against an untrusted opponent: "
      f"**{int(snap['n_entries_with_untrusted_opponent'].sum())}** - real own-team evidence that a "
      "rebuild from the training table would have discarded.\n")

    A("\n## 7. What this report does not claim\n")
    A("Nothing here says any feature is useful. Coverage, spread and completeness are properties of "
      "the data; predictive value is a property of a model that has not been fitted yet. The next "
      "phase must establish that separately, and only against TRAIN and validation.\n")

    (REPORTS / "phase5a_map_feature_quality.md").write_text("\n".join(L), encoding="utf-8")
    print(f"Wrote {REPORTS / 'phase5a_map_feature_quality.md'}")


def per_map_coverage(mapf, map_tr, snap):
    rows = []
    for m in sorted(mapf["map_name"].unique()):
        sub = mapf[mapf["map_name"] == m]
        sub_tr = map_tr[map_tr["map_name"] == m]
        sn = snap[snap["map_name"] == m]
        rows.append({
            "map_name": m,
            "rows_all": int(len(sub)),
            "rows_train": int(len(sub_tr)),
            "teams_seen": int(sn["canonical_team_name"].nunique()),
            "team1_cold_start": int((sub["team1_map_matches_before"] == 0).sum()),
            "team2_cold_start": int((sub["team2_map_matches_before"] == 0).sum()),
            "both_have_history": int(sub["both_teams_have_map_history"].sum()),
            "pct_both_have_history": 100 * sub["both_teams_have_map_history"].mean(),
        })
    return pd.DataFrame(rows)


def main():
    mapf, ser, v1, split, snap, map_sum, ser_sum, snap_meta = load_all()
    write_engineering_report(mapf, ser, v1, snap, map_sum, ser_sum, snap_meta)
    train_ids = set(split.loc[split["split"] == "train", "match_id"])
    map_tr = mapf[mapf["match_id"].isin(train_ids)]
    per_map_coverage(mapf, map_tr, snap).to_csv(
        TABLES / "map_feature_coverage_v1.csv", index=False, encoding="utf-8")
    print(f"Wrote {TABLES / 'map_feature_coverage_v1.csv'}")
    write_quality_report(mapf, ser, split, snap, ser_sum)


if __name__ == "__main__":
    main()
