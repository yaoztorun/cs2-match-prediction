"""
Reusable, dataset-agnostic temporal team-state / feature engine (Phase 3 V1).

No "development", "Cologne", or Kaggle-specific concepts appear in this file.
Dataset scope/filtering (which matches to feed in, which cutoff to use, which
teams are identity-trustworthy) is entirely an orchestration-layer concern -
see scripts/build_series_features_v1.py (Mode A) and
scripts/build_pre_cologne_snapshot.py (Mode B). A future Mode C (single-match
prediction) uses the exact same build_features()/update_state() calls, one
match at a time - see tests/test_feature_engine.py's equivalence test.

Core objects
------------
HistoryEntry   - one completed match from ONE team's perspective.
TeamState      - a team's ELO rating + its list of HistoryEntry.
StateStore     - canonical_team_name -> TeamState, with JSON save/load (the
                 full, re-loadable state) and a flat, scalar-only summary
                 export (parquet-safe - fastparquet, the active engine here,
                 does not want nested/list columns).

Core functions
--------------
elo_expected, elo_update          - pure ELO math, no side effects.
compute_team_features             - one team's pre-match feature dict as of a
                                     given timestamp (defensively filters to
                                     strictly-prior history even if the caller
                                     mis-orders calls).
build_features                    - read-only Team1-vs-Team2 matchup feature
                                     dict. Never mutates the store.
update_state                      - mutates the store after a match RESULT is
                                     known. update_team1/update_team2/update_elo
                                     flags implement the identity-exclusion
                                     policy (ELO is the only pair-dependent
                                     piece; each team's own win/loss/margin/
                                     activity history is independent).
process_chronological_stream      - the shared two-phase batch driver used by
                                     every orchestration mode. Guarantees
                                     history_time < current_match_time and that
                                     matches sharing a timestamp never see each
                                     other's result.

Match identity: every processed match carries `source`, `source_match_id`, and
a globally-unique `canonical_match_uid` (e.g. f"{source}:{source_match_id}").
Raw provider match ids are NOT assumed unique across sources - only
canonical_match_uid is relied on for de-duplication/ordering.
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import timedelta

import pandas as pd

ENGINE_VERSION = "1.0.0"

ELO_INITIAL = 1500.0
ELO_K = 32.0
BETA_PRIOR_ALPHA = 2.0
BETA_PRIOR_BETA = 2.0
ROLLING_WINDOWS = (5, 10)
ACTIVITY_WINDOW_DAYS = 30

REQUIRED_STREAM_COLUMNS = [
    "datetime", "source", "source_match_id", "canonical_match_uid",
    "team1_canonical", "team2_canonical", "best_of", "tier",
    "score1", "score2", "team1_win", "team1_eligible", "team2_eligible",
]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class HistoryEntry:
    dt: pd.Timestamp
    source: str
    source_match_id: str
    canonical_match_uid: str
    opponent_canonical: str
    opponent_identity_trusted: bool
    best_of: int
    win: int
    margin: int

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
class TeamState:
    canonical_name: str
    elo: float = ELO_INITIAL
    history: list = field(default_factory=list)  # list[HistoryEntry]

    def to_dict(self):
        return {"canonical_name": self.canonical_name, "elo": self.elo,
                "history": [h.to_dict() for h in self.history]}

    @classmethod
    def from_dict(cls, d):
        return cls(canonical_name=d["canonical_name"], elo=d["elo"],
                    history=[HistoryEntry.from_dict(h) for h in d["history"]])


class StateStore:
    """canonical_team_name -> TeamState. Cold-start teams are created lazily
    (get_or_create) only when a real match result is applied via update_state -
    build_features/compute_team_features never create or mutate entries."""

    def __init__(self):
        self.teams: dict[str, TeamState] = {}
        self.processed_match_uids: set[str] = set()

    def get(self, canonical_name):
        return self.teams.get(canonical_name)

    def get_or_create(self, canonical_name):
        if canonical_name not in self.teams:
            self.teams[canonical_name] = TeamState(canonical_name=canonical_name)
        return self.teams[canonical_name]

    def to_json(self, path, meta=None):
        payload = {
            "engine_version": ENGINE_VERSION,
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
            store.teams[name] = TeamState.from_dict(tdict)
        for ts in store.teams.values():
            for h in ts.history:
                store.processed_match_uids.add(h.canonical_match_uid)
        store.loaded_meta = payload.get("meta", {})
        store.loaded_engine_version = payload.get("engine_version")
        return store

    def snapshot_summary_df(self):
        """Flat, scalar-only, one row per team - safe for a fastparquet export.
        Full per-match history is only ever available via to_json/from_json."""
        rows = [team_summary_row(ts) for ts in self.teams.values()]
        df = pd.DataFrame(rows)
        fmt_cols = [c for c in df.columns if c.startswith("format_bo")]
        if fmt_cols:
            df[fmt_cols] = df[fmt_cols].fillna(0).astype("int64")
        return df


# ---------------------------------------------------------------------------
# ELO - pure functions
# ---------------------------------------------------------------------------

def elo_expected(r_a, r_b):
    """P(A beats B) given current ratings. No side effects."""
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))


def elo_update(r_a, r_b, s_a):
    """S_a in {0,1} (1 if A won). Returns (new_r_a, new_r_b). Pure - does not
    touch a TeamState. V1 is deliberately simple: no tier/BO/time-decay/margin
    weighting, no K-factor tuning."""
    e_a = elo_expected(r_a, r_b)
    e_b = 1.0 - e_a
    s_b = 1.0 - s_a
    new_a = r_a + ELO_K * (s_a - e_a)
    new_b = r_b + ELO_K * (s_b - e_b)
    return new_a, new_b


# ---------------------------------------------------------------------------
# Shared stat helpers (used by both the per-match feature path and the
# flat summary-snapshot path, so the two can never drift apart)
# ---------------------------------------------------------------------------

def _sorted_before(history, as_of):
    """as_of=None means no time filtering (used only by the summary/debug
    snapshot path, where the caller has already bounded the input stream)."""
    if as_of is None:
        entries = list(history)
    else:
        entries = [h for h in history if h.dt < as_of]
    return sorted(entries, key=lambda h: h.dt)


def _win_rate(entries):
    return (sum(h.win for h in entries) / len(entries)) if entries else 0.5


def _avg_margin(entries):
    return (sum(h.margin for h in entries) / len(entries)) if entries else 0.0


def _beta_smoothed_win_rate(entries):
    wins = sum(h.win for h in entries)
    n = len(entries)
    return (wins + BETA_PRIOR_ALPHA) / (n + BETA_PRIOR_ALPHA + BETA_PRIOR_BETA)


# ---------------------------------------------------------------------------
# Per-match feature computation
# ---------------------------------------------------------------------------

def compute_team_features(state, as_of, current_best_of):
    """One team's pre-match feature dict as of `as_of` (exclusive). `state` may
    be None (a team never seen before - cold start). Defensively filters to
    history strictly before `as_of` regardless of what the caller has or hasn't
    already applied - this is what makes the strict-chronology guarantee hold
    even under an out-of-order call, not just under a well-behaved orchestrator."""
    as_of = pd.Timestamp(as_of)
    history = state.history if state is not None else []
    hist = _sorted_before(history, as_of)
    n = len(hist)

    last5 = hist[-5:]
    last10 = hist[-10:]
    fmt_hist = [h for h in hist if h.best_of == current_best_of]

    if n > 0:
        days_since_last = (as_of - hist[-1].dt).total_seconds() / 86400.0
    else:
        days_since_last = float("nan")

    thirty_days_ago = as_of - timedelta(days=ACTIVITY_WINDOW_DAYS)
    matches_last_30_days = sum(1 for h in hist if h.dt >= thirty_days_ago)

    return {
        "elo_before": state.elo if state is not None else ELO_INITIAL,
        "overall_win_rate_smoothed": _beta_smoothed_win_rate(hist),
        "win_rate_last_5": _win_rate(last5),
        "win_rate_last_10": _win_rate(last10),
        "format_win_rate_smoothed": _beta_smoothed_win_rate(fmt_hist),
        "avg_series_margin_last_5": _avg_margin(last5),
        "avg_series_margin_last_10": _avg_margin(last10),
        "total_matches_before": n,
        "matches_last_30_days": matches_last_30_days,
        "days_since_last_match": days_since_last,
    }


def build_features(store, team1, team2, prediction_time, best_of, tier=None):
    """Read-only. Never mutates `store`. Returns a dict with raw team1_*/team2_*
    per-team values (for audit), the 10 Team1-Team2 diff features, 5
    side-symmetric confidence features, and bestOf/tier passthrough."""
    prediction_time = pd.Timestamp(prediction_time)
    f1 = compute_team_features(store.get(team1), prediction_time, best_of)
    f2 = compute_team_features(store.get(team2), prediction_time, best_of)

    out = {}
    for k, v in f1.items():
        out[f"team1_{k}"] = v
    for k, v in f2.items():
        out[f"team2_{k}"] = v

    out["elo_diff"] = f1["elo_before"] - f2["elo_before"]
    out["overall_win_rate_diff"] = f1["overall_win_rate_smoothed"] - f2["overall_win_rate_smoothed"]
    out["win_rate_last_5_diff"] = f1["win_rate_last_5"] - f2["win_rate_last_5"]
    out["win_rate_last_10_diff"] = f1["win_rate_last_10"] - f2["win_rate_last_10"]
    out["format_win_rate_diff"] = f1["format_win_rate_smoothed"] - f2["format_win_rate_smoothed"]
    out["avg_series_margin_last_5_diff"] = f1["avg_series_margin_last_5"] - f2["avg_series_margin_last_5"]
    out["avg_series_margin_last_10_diff"] = f1["avg_series_margin_last_10"] - f2["avg_series_margin_last_10"]
    out["matches_last_30_days_diff"] = f1["matches_last_30_days"] - f2["matches_last_30_days"]
    out["days_since_last_match_diff"] = f1["days_since_last_match"] - f2["days_since_last_match"]
    out["total_matches_before_diff"] = f1["total_matches_before"] - f2["total_matches_before"]

    n1, n2 = f1["total_matches_before"], f2["total_matches_before"]
    out["history_matches_min"] = min(n1, n2)
    out["history_matches_sum"] = n1 + n2
    out["both_teams_have_history"] = int(n1 > 0 and n2 > 0)
    out["both_teams_have_5_matches"] = int(n1 >= 5 and n2 >= 5)
    out["both_teams_have_10_matches"] = int(n1 >= 10 and n2 >= 10)

    out["bestOf"] = best_of
    out["tier"] = tier
    return out


def update_state(store, team1, team2, winner, source, source_match_id, canonical_match_uid,
                  prediction_time, best_of, score1, score2,
                  update_team1=True, update_team2=True, update_elo=True):
    """Mutates `store` after a match result is known. `winner` in {"team1","team2"}.

    ELO is the only pair-dependent piece of state - update_elo requires both
    update_team1 and update_team2 to be True (enforced below). Each team's own
    win/loss/margin/activity history is independent and may update even when
    the opponent's identity is untrusted; in that case the new HistoryEntry is
    still recorded (the eligible team's own result is a real fact) but flagged
    opponent_identity_trusted=False so future opponent-dependent features
    (H2H etc.) can exclude it."""
    if update_elo and not (update_team1 and update_team2):
        raise ValueError("update_elo=True requires both update_team1 and update_team2 to be True")

    prediction_time = pd.Timestamp(prediction_time)
    win1 = 1 if winner == "team1" else 0
    win2 = 1 - win1
    margin1 = int(score1) - int(score2)
    margin2 = int(score2) - int(score1)

    if update_team1:
        s1 = store.get_or_create(team1)
        s1.history.append(HistoryEntry(
            dt=prediction_time, source=source, source_match_id=str(source_match_id),
            canonical_match_uid=canonical_match_uid, opponent_canonical=team2,
            opponent_identity_trusted=bool(update_team2), best_of=int(best_of),
            win=win1, margin=margin1,
        ))
    if update_team2:
        s2 = store.get_or_create(team2)
        s2.history.append(HistoryEntry(
            dt=prediction_time, source=source, source_match_id=str(source_match_id),
            canonical_match_uid=canonical_match_uid, opponent_canonical=team1,
            opponent_identity_trusted=bool(update_team1), best_of=int(best_of),
            win=win2, margin=margin2,
        ))
    if update_elo:
        s1 = store.get_or_create(team1)
        s2 = store.get_or_create(team2)
        s1.elo, s2.elo = elo_update(s1.elo, s2.elo, float(win1))

    store.processed_match_uids.add(canonical_match_uid)


# ---------------------------------------------------------------------------
# Flat summary snapshot (for parquet export - no lists/nested structures)
# ---------------------------------------------------------------------------

def team_summary_row(state):
    hist = _sorted_before(state.history, as_of=None)
    n = len(hist)
    last5, last10 = hist[-5:], hist[-10:]

    row = {
        "canonical_team_name": state.canonical_name,
        "elo": state.elo,
        "total_matches": n,
        "wins": sum(h.win for h in hist),
        "overall_win_rate_smoothed": _beta_smoothed_win_rate(hist),
        "win_rate_last_5": _win_rate(last5),
        "matches_in_last_5": len(last5),
        "win_rate_last_10": _win_rate(last10),
        "matches_in_last_10": len(last10),
        "avg_series_margin_last_5": _avg_margin(last5),
        "avg_series_margin_last_10": _avg_margin(last10),
        "last_match_datetime": hist[-1].dt if hist else pd.NaT,
    }
    for bo in sorted(set(h.best_of for h in hist)):
        sub = [h for h in hist if h.best_of == bo]
        row[f"format_bo{bo}_matches"] = len(sub)
        row[f"format_bo{bo}_wins"] = sum(h.win for h in sub)
    return row


# ---------------------------------------------------------------------------
# Shared chronological-stream driver (used by every orchestration mode)
# ---------------------------------------------------------------------------

def process_chronological_stream(store, rows):
    """
    rows: DataFrame with at least REQUIRED_STREAM_COLUMNS. Extra columns are
    carried through into the returned dicts unchanged (metadata passthrough),
    so orchestration can attach match_id/tournament/display names/etc. freely.

    CRITICAL two-phase per-timestamp-group protocol:
        Phase A (read):  build_features() for every eligible-pair row in the
                          group, using state exactly as it was before this
                          group started. No mutation.
        Phase B (write):  only after every row in the group has been read,
                          apply update_state() for every row in the group (in
                          deterministic canonical_match_uid order), respecting
                          each row's eligibility flags.
    This guarantees history_time < current_match_time strictly, and that two
    matches sharing an exact timestamp never see each other's result.

    Returns (processed_rows, excluded_rows): both are lists of dicts.
    processed_rows = original row + build_features() output, one per
    eligible-pair match (i.e. the ones that got a feature row).
    excluded_rows = original row + "reason", one per match with >=1
    identity-ineligible side (no feature row, but state may still have been
    partially updated for the eligible side - see update_state).
    """
    missing = set(REQUIRED_STREAM_COLUMNS) - set(rows.columns)
    if missing:
        raise ValueError(f"process_chronological_stream: missing required columns {sorted(missing)}")
    if rows["canonical_match_uid"].duplicated().any():
        dupes = rows.loc[rows["canonical_match_uid"].duplicated(), "canonical_match_uid"].tolist()
        raise ValueError(f"duplicate canonical_match_uid in stream: {dupes[:5]}")

    ordered = rows.sort_values(["datetime", "canonical_match_uid"]).reset_index(drop=True)

    processed_rows = []
    excluded_rows = []

    for ts, group in ordered.groupby("datetime", sort=True):
        group = group.sort_values("canonical_match_uid")

        # Phase A: read only.
        batch_features = {}
        for _, row in group.iterrows():
            uid = row["canonical_match_uid"]
            t1_elig, t2_elig = bool(row["team1_eligible"]), bool(row["team2_eligible"])
            if t1_elig and t2_elig:
                batch_features[uid] = build_features(
                    store, row["team1_canonical"], row["team2_canonical"], ts, row["best_of"], row.get("tier"))
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
            t1_elig, t2_elig = bool(row["team1_eligible"]), bool(row["team2_eligible"])
            update_state(
                store, row["team1_canonical"], row["team2_canonical"],
                winner=("team1" if row["team1_win"] == 1 else "team2"),
                source=row["source"], source_match_id=row["source_match_id"],
                canonical_match_uid=uid, prediction_time=ts, best_of=row["best_of"],
                score1=row["score1"], score2=row["score2"],
                update_team1=t1_elig, update_team2=t2_elig, update_elo=(t1_elig and t2_elig),
            )
            if uid in batch_features:
                out_row = row.to_dict()
                out_row.update(batch_features[uid])
                processed_rows.append(out_row)

    return processed_rows, excluded_rows
