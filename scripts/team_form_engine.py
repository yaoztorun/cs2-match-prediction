"""
Reusable, dataset-agnostic SERIES-level opponent-adjusted team-form engine
(Phase 5B.2).

Mirrors the design of scripts/feature_engine.py (Phase 3) and
scripts/map_feature_engine.py (Phase 5A) and deliberately REUSES Phase 3's
pure ELO primitives rather than reimplementing them: `ELO_INITIAL`, `ELO_K`,
`elo_expected`, `elo_update`. Neither feature_engine.py nor map_feature_engine.py
is modified anywhere in this file.

WHY A NEW, INDEPENDENT STATE STORE
-----------------------------------
feature_engine.py's own state (HistoryEntry/TeamState) never records the
OPPONENT's pre-match ELO or a performance-vs-expectation residual - that data
does not exist in the frozen engine's state, and the engine cannot be
modified. This module therefore replays the SAME series stream (same rows,
same eligibility gating, same two-phase per-timestamp batching, same pure
elo_expected/elo_update calls) through its OWN state store, which - by
construction, given identical inputs - reproduces Phase 3's own `elo_diff`
trajectory exactly, while ADDITIONALLY recording each match's opponent ELO
and performance residual. This is independently, exhaustively verified (not
merely asserted) in scripts/build_series_features_v3_form.py before any V3
artifact is written, and re-verified in scripts/validate_phase5b2.py.

PRE-MATCH ELO ONLY
-------------------
`expected_win_prob = elo_expected(own_elo_before, opponent_elo_before)` is
always computed BEFORE `elo_update` is called for that match. Never
recomputed from post-match ratings.

TRUSTED-OPPONENT GATING FOR OPPONENT-ADJUSTED FORM (read this before touching
compute_team_form_features)
-------------------------------------------------------------------------
Following the Phase 3 / Phase 5A identity policy: an eligible team's OWN
independent history (result, normalized series margin, activity/timing) may
still update from a match against an identity-ineligible opponent - that
team's own result is a real fact - flagged `opponent_identity_trusted=False`.
But OPPONENT-ADJUSTED information (who they played, how strong that opponent
was) depends on a reliable, persistent opponent identity and ELO trajectory,
which an untrusted opponent does not have. Therefore:

  * `avg_opponent_elo_last_5/10`, `performance_residual_last_5/10/all`, and
    `time_weighted_performance_residual` are computed ONLY over history
    entries with `opponent_identity_trusted == True` ("trusted" population).
  * `time_weighted_win_rate` and `time_weighted_normalized_series_margin` use
    ALL of the team's own eligible history (trusted or not) - they describe
    the team's own recent results, not anything about who they played, so
    they do not depend on persistent opponent identity ("all" population).
  * `time_weighted_history_mass` (the confidence companion for the "all"
    population above) is likewise computed over ALL eligible history, not
    trust-filtered.
  * The confidence/evidence flags `opponent_adjusted_history_min`,
    `both_teams_have_5_adjusted_matches`, `both_teams_have_10_adjusted_matches`
    count only TRUSTED opponent-adjusted observations - never all history -
    so they can never overstate the evidence behind the trusted-population
    stats above.

STATE-HISTORY ELIGIBILITY vs TRAINING ELIGIBILITY (not the same thing)
------------------------------------------------------------------------
  * A supervised TRAINING ROW requires BOTH canonical identities trusted.
  * ELO is pair-dependent and updates ONLY when both are eligible.
  * Opponent ELO / expected win probability / performance residual are always
    COMPUTED and STORED on every history entry (they are well-defined numbers
    regardless of trust - "opponent_elo_before" is just whatever ELO that
    team-name is currently tracked at); it is the READ-TIME AGGREGATION in
    compute_team_form_features that filters by `opponent_identity_trusted`,
    per the gating rule above.

EXACT-TIMESTAMP LEAKAGE PROTECTION
------------------------------------
`process_form_stream` uses the identical two-phase per-timestamp-group
protocol as `feature_engine.process_chronological_stream`: Phase A emits
features for every eligible-pair row in a timestamp group from the state as
it was BEFORE the group; Phase B applies every row's result only after the
whole group has been read. No match at time T can see another match at T.
"""

import json
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

from feature_engine import ELO_INITIAL, ELO_K, elo_expected, elo_update, REQUIRED_STREAM_COLUMNS

FORM_ENGINE_VERSION = "1.0.0"

FORM_HALF_LIFE_DAYS = 60.0   # fixed engineering constant for this phase - NOT tuned, never read from a search artifact

# Cold-start contract (documented once, applied everywhere)
COLD_START_OPPONENT_ELO = ELO_INITIAL   # 1500.0 - neutral opponent assumption, no evidence
COLD_START_RESIDUAL = 0.0               # neutral - no evidence of over/under-performance
COLD_START_WIN_RATE = 0.5
COLD_START_MARGIN = 0.0
COLD_START_HISTORY_MASS = 0.0           # zero effective evidence, not "neutral" - a true absence
COLD_START_MATCHES = 0


def normalized_series_margin(score_for, score_against):
    """(for - against) / (for + against); 0.0 when the denominator is 0. Same
    normalization convention scripts/map_feature_engine.py uses for map
    margins (normalized_round_margin), applied here to series map-count
    scores so it scales consistently across BO1/BO3/BO5 without inventing a
    new concept. A local copy, not an import, to keep this module's only
    dependency on the Phase 3 engine (map_feature_engine.py is a sibling
    Phase 5A module, not a dependency of this one)."""
    total = float(score_for) + float(score_against)
    if total <= 0:
        return 0.0
    return (float(score_for) - float(score_against)) / total


def _recency_weight(as_of, dt, half_life_days=FORM_HALF_LIFE_DAYS):
    age_days = (as_of - dt).total_seconds() / 86400.0
    return 0.5 ** (age_days / half_life_days)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class FormHistoryEntry:
    dt: pd.Timestamp
    source: str
    source_match_id: str
    canonical_match_uid: str
    opponent_canonical: str
    opponent_identity_trusted: bool
    win: int
    own_elo_before: float
    opponent_elo_before: float
    expected_win_prob: float       # elo_expected(own_elo_before, opponent_elo_before) - PRE-MATCH only
    performance_residual: float    # win - expected_win_prob
    normalized_margin: float

    def to_dict(self):
        d = asdict(self)
        d["dt"] = self.dt.isoformat()
        return d

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        d["dt"] = pd.Timestamp(d["dt"])
        return cls(**d)


@dataclass
class TeamFormState:
    canonical_name: str
    elo: float = ELO_INITIAL
    history: list = field(default_factory=list)   # list[FormHistoryEntry]

    def to_dict(self):
        return {"canonical_name": self.canonical_name, "elo": self.elo,
                "history": [h.to_dict() for h in self.history]}

    @classmethod
    def from_dict(cls, d):
        return cls(canonical_name=d["canonical_name"], elo=d["elo"],
                    history=[FormHistoryEntry.from_dict(h) for h in d["history"]])


class TeamFormStateStore:
    """canonical_team_name -> TeamFormState. Entries are created lazily only
    when a real match result is applied (apply_form_result); feature
    computation never creates or mutates state."""

    def __init__(self):
        self.teams: dict[str, TeamFormState] = {}
        self.processed_match_uids: set[str] = set()

    def get(self, canonical_name):
        return self.teams.get(canonical_name)

    def get_or_create(self, canonical_name):
        if canonical_name not in self.teams:
            self.teams[canonical_name] = TeamFormState(canonical_name=canonical_name)
        return self.teams[canonical_name]

    def to_json(self, path, meta=None):
        payload = {
            "form_engine_version": FORM_ENGINE_VERSION,
            "n_teams": len(self.teams),
            "n_matches_processed": len(self.processed_match_uids),
            "meta": meta or {},
            "teams": {name: ts.to_dict() for name, ts in self.teams.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)

    @classmethod
    def from_json(cls, path):
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        store = cls()
        for name, tdict in payload["teams"].items():
            store.teams[name] = TeamFormState.from_dict(tdict)
        for ts in store.teams.values():
            for h in ts.history:
                store.processed_match_uids.add(h.canonical_match_uid)
        store.loaded_meta = payload.get("meta", {})
        return store

    def snapshot_summary_df(self):
        """Flat, scalar-only, one row per team - fastparquet-safe. No
        time-weighted stats here (those are inherently relative to a query
        time, not to a state dump); full per-match history is only available
        via to_json/from_json."""
        rows = []
        for st in self.teams.values():
            hist_all = sorted(st.history, key=lambda h: h.dt)
            hist_trusted = [h for h in hist_all if h.opponent_identity_trusted]
            rows.append({
                "canonical_team_name": st.canonical_name,
                "elo": st.elo,
                "total_matches": len(hist_all),
                "trusted_opponent_matches": len(hist_trusted),
                "wins": sum(h.win for h in hist_all),
                "avg_opponent_elo_all_trusted": (
                    float(np.mean([h.opponent_elo_before for h in hist_trusted]))
                    if hist_trusted else COLD_START_OPPONENT_ELO),
                "performance_residual_all_trusted": (
                    float(np.mean([h.performance_residual for h in hist_trusted]))
                    if hist_trusted else COLD_START_RESIDUAL),
                "last_match_datetime": hist_all[-1].dt if hist_all else pd.NaT,
            })
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Strictly-prior per-team statistics
# ---------------------------------------------------------------------------

def compute_team_form_features(state, as_of):
    """One team's strictly-prior (`h.dt < as_of`) form stats. `state` may be
    None (never seen before) -> documented cold start. See the module
    docstring's TRUSTED-OPPONENT GATING section for which stats use the
    "trusted" population (opponent_identity_trusted == True) vs. the "all"
    population (every own eligible entry)."""
    as_of = pd.Timestamp(as_of)
    history = state.history if state is not None else []
    hist_all = sorted([h for h in history if h.dt < as_of], key=lambda h: h.dt)
    hist_trusted = [h for h in hist_all if h.opponent_identity_trusted]

    trusted_last5 = hist_trusted[-5:]
    trusted_last10 = hist_trusted[-10:]
    n_trusted = len(hist_trusted)

    avg_opp_elo_5 = (float(np.mean([h.opponent_elo_before for h in trusted_last5]))
                      if trusted_last5 else COLD_START_OPPONENT_ELO)
    avg_opp_elo_10 = (float(np.mean([h.opponent_elo_before for h in trusted_last10]))
                       if trusted_last10 else COLD_START_OPPONENT_ELO)
    perf_resid_5 = (float(np.mean([h.performance_residual for h in trusted_last5]))
                     if trusted_last5 else COLD_START_RESIDUAL)
    perf_resid_10 = (float(np.mean([h.performance_residual for h in trusted_last10]))
                      if trusted_last10 else COLD_START_RESIDUAL)
    perf_resid_all = (float(np.mean([h.performance_residual for h in hist_trusted]))
                       if hist_trusted else COLD_START_RESIDUAL)

    # time-weighted win rate / margin / history mass: ALL own eligible history
    if hist_all:
        weights_all = [_recency_weight(as_of, h.dt) for h in hist_all]
        wsum_all = sum(weights_all)
        tw_win_rate = sum(w * h.win for w, h in zip(weights_all, hist_all)) / wsum_all
        tw_margin = sum(w * h.normalized_margin for w, h in zip(weights_all, hist_all)) / wsum_all
        history_mass = wsum_all
    else:
        tw_win_rate = COLD_START_WIN_RATE
        tw_margin = COLD_START_MARGIN
        history_mass = COLD_START_HISTORY_MASS

    # time-weighted performance residual: TRUSTED population only
    if hist_trusted:
        weights_tr = [_recency_weight(as_of, h.dt) for h in hist_trusted]
        wsum_tr = sum(weights_tr)
        tw_perf_residual = sum(w * h.performance_residual for w, h in zip(weights_tr, hist_trusted)) / wsum_tr
    else:
        tw_perf_residual = COLD_START_RESIDUAL

    return {
        "elo_before": state.elo if state is not None else ELO_INITIAL,
        "avg_opponent_elo_last_5": avg_opp_elo_5,
        "avg_opponent_elo_last_10": avg_opp_elo_10,
        "performance_residual_last_5": perf_resid_5,
        "performance_residual_last_10": perf_resid_10,
        "performance_residual_all": perf_resid_all,
        "time_weighted_win_rate": tw_win_rate,
        "time_weighted_performance_residual": tw_perf_residual,
        "time_weighted_normalized_series_margin": tw_margin,
        "time_weighted_history_mass": history_mass,
        "adjusted_matches_before": n_trusted,
    }


FORM_DIRECTIONAL_FEATURES = [
    "avg_opponent_elo_last_5_diff", "avg_opponent_elo_last_10_diff",
    "performance_residual_last_5_diff", "performance_residual_last_10_diff", "performance_residual_all_diff",
    "time_weighted_win_rate_diff", "time_weighted_performance_residual_diff", "time_weighted_series_margin_diff",
]
FORM_SYMMETRIC_FEATURES = [
    "opponent_adjusted_history_min", "both_teams_have_5_adjusted_matches",
    "both_teams_have_10_adjusted_matches", "time_weighted_history_mass_min",
]


def build_future_team_form_features(store, team1, team2, as_of_datetime):
    """PURE / READ-ONLY. Needs no target and no current-series score. Produces
    the exact schema used for training, so future inference cannot diverge."""
    as_of = pd.Timestamp(as_of_datetime)
    f1 = compute_team_form_features(store.get(team1), as_of)
    f2 = compute_team_form_features(store.get(team2), as_of)

    out = {}
    for k, v in f1.items():
        out[f"team1_{k}"] = v
    for k, v in f2.items():
        out[f"team2_{k}"] = v

    out["avg_opponent_elo_last_5_diff"] = f1["avg_opponent_elo_last_5"] - f2["avg_opponent_elo_last_5"]
    out["avg_opponent_elo_last_10_diff"] = f1["avg_opponent_elo_last_10"] - f2["avg_opponent_elo_last_10"]
    out["performance_residual_last_5_diff"] = f1["performance_residual_last_5"] - f2["performance_residual_last_5"]
    out["performance_residual_last_10_diff"] = f1["performance_residual_last_10"] - f2["performance_residual_last_10"]
    out["performance_residual_all_diff"] = f1["performance_residual_all"] - f2["performance_residual_all"]
    out["time_weighted_win_rate_diff"] = f1["time_weighted_win_rate"] - f2["time_weighted_win_rate"]
    out["time_weighted_performance_residual_diff"] = (
        f1["time_weighted_performance_residual"] - f2["time_weighted_performance_residual"])
    out["time_weighted_series_margin_diff"] = (
        f1["time_weighted_normalized_series_margin"] - f2["time_weighted_normalized_series_margin"])

    n1, n2 = f1["adjusted_matches_before"], f2["adjusted_matches_before"]
    out["opponent_adjusted_history_min"] = min(n1, n2)
    out["both_teams_have_5_adjusted_matches"] = int(n1 >= 5 and n2 >= 5)
    out["both_teams_have_10_adjusted_matches"] = int(n1 >= 10 and n2 >= 10)
    out["time_weighted_history_mass_min"] = min(f1["time_weighted_history_mass"], f2["time_weighted_history_mass"])

    return out


# ---------------------------------------------------------------------------
# State updates
# ---------------------------------------------------------------------------

def apply_form_result(store, row):
    """Apply ONE completed series to the store. Must only be called in Phase B.

    Identity policy (Phase 3 / Phase 5A rules, restated): each eligible
    team's own history updates regardless of the opponent's eligibility
    (its own result is a real fact), flagged `opponent_identity_trusted`.
    ELO is pair-dependent and updates only when both are eligible.
    `opponent_elo_before`/`expected_win_prob`/`performance_residual` are
    always computed here (well-defined regardless of trust); the
    TRUSTED-only filtering happens at READ time in compute_team_form_features.
    """
    t1, t2 = row["team1_canonical"], row["team2_canonical"]
    e1, e2 = bool(row["team1_eligible"]), bool(row["team2_eligible"])
    dt = pd.Timestamp(row["datetime"])
    win1 = int(row["team1_win"])
    win2 = 1 - win1
    score1, score2 = row["score1"], row["score2"]
    uid = row["canonical_match_uid"]

    s1, s2 = store.get(t1), store.get(t2)
    own_elo1 = s1.elo if s1 is not None else ELO_INITIAL
    own_elo2 = s2.elo if s2 is not None else ELO_INITIAL
    # PRE-MATCH expectation only - computed BEFORE elo_update is ever called.
    expected1 = elo_expected(own_elo1, own_elo2)
    expected2 = 1.0 - expected1
    residual1 = win1 - expected1
    residual2 = win2 - expected2
    margin1 = normalized_series_margin(score1, score2)
    margin2 = normalized_series_margin(score2, score1)

    if e1:
        st = store.get_or_create(t1)
        st.history.append(FormHistoryEntry(
            dt=dt, source=row["source"], source_match_id=str(row["source_match_id"]),
            canonical_match_uid=uid, opponent_canonical=t2, opponent_identity_trusted=e2,
            win=win1, own_elo_before=own_elo1, opponent_elo_before=own_elo2,
            expected_win_prob=expected1, performance_residual=residual1, normalized_margin=margin1))
    if e2:
        st = store.get_or_create(t2)
        st.history.append(FormHistoryEntry(
            dt=dt, source=row["source"], source_match_id=str(row["source_match_id"]),
            canonical_match_uid=uid, opponent_canonical=t1, opponent_identity_trusted=e1,
            win=win2, own_elo_before=own_elo2, opponent_elo_before=own_elo1,
            expected_win_prob=expected2, performance_residual=residual2, normalized_margin=margin2))
    if e1 and e2:
        a, b = store.get_or_create(t1), store.get_or_create(t2)
        a.elo, b.elo = elo_update(a.elo, b.elo, float(win1))

    store.processed_match_uids.add(uid)


# ---------------------------------------------------------------------------
# Two-phase chronological driver (identical protocol to
# feature_engine.process_chronological_stream)
# ---------------------------------------------------------------------------

def process_form_stream(store, rows, emit_features=True):
    """
    rows: DataFrame with at least REQUIRED_STREAM_COLUMNS (imported unchanged
    from feature_engine.py - the exact same stream contract as Phase 3).

    Phase A (read):  build_future_team_form_features() for every eligible-pair
                      row in the timestamp group, using state exactly as it
                      was before this group started. No mutation.
    Phase B (write): only after every row in the group has been read, apply
                      apply_form_result() for every row in the group (in
                      deterministic canonical_match_uid order).

    `emit_features=False` skips Phase A entirely (used by the pre-Cologne
    snapshot builder, which only needs final state).

    Returns (processed_rows, excluded_rows).
    """
    missing = set(REQUIRED_STREAM_COLUMNS) - set(rows.columns)
    if missing:
        raise ValueError(f"process_form_stream: missing required columns {sorted(missing)}")
    if rows["canonical_match_uid"].duplicated().any():
        dupes = rows.loc[rows["canonical_match_uid"].duplicated(), "canonical_match_uid"].tolist()
        raise ValueError(f"duplicate canonical_match_uid in stream: {dupes[:5]}")

    ordered = rows.sort_values(["datetime", "canonical_match_uid"]).reset_index(drop=True)
    processed_rows, excluded_rows = [], []

    for ts, group in ordered.groupby("datetime", sort=True):
        group = group.sort_values("canonical_match_uid")

        # Phase A: read only.
        batch_features = {}
        if emit_features:
            for _, row in group.iterrows():
                uid = row["canonical_match_uid"]
                t1_elig, t2_elig = bool(row["team1_eligible"]), bool(row["team2_eligible"])
                if t1_elig and t2_elig:
                    batch_features[uid] = build_future_team_form_features(
                        store, row["team1_canonical"], row["team2_canonical"], ts)
                else:
                    reasons = []
                    if not t1_elig:
                        reasons.append(f"team1_not_identity_eligible:{row['team1_canonical']}")
                    if not t2_elig:
                        reasons.append(f"team2_not_identity_eligible:{row['team2_canonical']}")
                    excluded = row.to_dict()
                    excluded["reason"] = ";".join(reasons)
                    excluded_rows.append(excluded)

        # Phase B: write, only after every read above for this timestamp group.
        for _, row in group.iterrows():
            uid = row["canonical_match_uid"]
            apply_form_result(store, row)
            if uid in batch_features:
                out_row = row.to_dict()
                out_row.update(batch_features[uid])
                processed_rows.append(out_row)

    return processed_rows, excluded_rows
