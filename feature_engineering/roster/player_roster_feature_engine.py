"""
Reusable, dataset-agnostic PLAYER-FORM / ROSTER-STABILITY engine (Phase 5C).

Mirrors the design of scripts/feature_engine.py (Phase 3),
scripts/map_feature_engine.py (Phase 5A) and scripts/team_form_engine.py
(Phase 5B.2). None of those files is modified anywhere in Phase 5C.

TWO DISTINCT STATE TYPES - NEVER CONFLATE THEM
-----------------------------------------------
A. GLOBAL PLAYER PERFORMANCE STATE, keyed by persistent `player_id`.
   A player's individual form FOLLOWS THE PLAYER across team changes: if a
   strong player transfers from Team A to Team B we must not pretend they
   have no history. This state never asks which team the player was on.

B. TEAM ROSTER / APPEARANCE STATE, keyed by canonical team identity.
   Which players have recently REPRESENTED THAT TEAM. This is what the
   expected-current-roster inference and every roster-stability feature read.
   Team membership does NOT transfer with the player: a transferred player
   only enters Team B's roster after they have actually appeared for Team B.

THE TARGET SERIES' ACTUAL LINEUP IS FORBIDDEN AS A PREDICTOR
-------------------------------------------------------------
The application asks a user to pick two TEAMS, not ten player ids, so a
training-time feature that depended on knowing the target series' real five
players could never be reproduced at inference time. Independently, this
repo's own standing ruling (reports/data_audit.md open question 2,
reports/leakage_analysis.md) is that the lineup columns may be a post-hoc
box-score roster including substitutions and must be "treated as leaky by
default until the data collection method is confirmed". Both reasons point
the same way: the expected roster is INFERRED from strictly-prior team
appearances only (see `infer_expected_roster`), and the target series'
lineup is used only in Phase B, after that series' feature row has already
been emitted, to update state for LATER series.

AUTHORITATIVE SERIES DATETIME
------------------------------
Everything keys on `series_datetime` - the authoritative series start taken
from the canonical series source (see player_roster_stream_common.py), never
a map-derived timestamp. All maps of a series share one cutoff, so no map of
a series can influence its own series' features, and Map 1's box score can
never reach Map 2.

IDENTITY POLICY FOR THE TWO STATES
-----------------------------------
Following the Phase 3 / 5A / 5B.2 principle ("a team's own result is a real
fact") applied at the correct key granularity:
  * PLAYER PERFORMANCE is player-keyed, so it updates whenever that player's
    box score is usable, regardless of either team's identity eligibility -
    the player's own performance is a real fact about the player.
  * TEAM APPEARANCE is team-keyed, so it updates only when that canonical
    team is identity-eligible, exactly like every other team-keyed state in
    this project.
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import timedelta

import numpy as np
import pandas as pd

PLAYER_ROSTER_ENGINE_VERSION = "1.0.0"

# Fixed engineering constants for Phase 5C - NEVER tuned, never selected
# against any metric.
ROSTER_LOOKBACK_DAYS = 90.0
PLAYER_FORM_HALF_LIFE_DAYS = 60.0
ROSTER_SIZE = 5

# Cold-start contract (documented once, applied everywhere).
# Performance aggregates are genuinely MISSING when no inferred roster player
# has any prior observation - never a fabricated number, and deliberately
# never a population-wide mean (that would import information from outside
# the strictly-prior window).
COLD_START_PERFORMANCE = float("nan")
COLD_START_HISTORY_MASS = 0.0      # a true absence of evidence, not "neutral"
COLD_START_COUNT = 0
COLD_START_RATIO = 0.0

REQUIRED_PLAYER_STREAM_COLUMNS = [
    "match_id", "game_id", "series_datetime", "side", "slot",
    "team_canonical", "team_eligible", "player_id",
    "adr", "kast", "kd_balance", "assists_per_round", "has_usable_stats",
]
SERIES_REQUEST_COLUMNS = [
    "match_id", "series_datetime", "team1_canonical", "team2_canonical",
    "team1_eligible", "team2_eligible",
]


def recency_weight(as_of, dt, half_life_days=PLAYER_FORM_HALF_LIFE_DAYS):
    age_days = (as_of - dt).total_seconds() / 86400.0
    return 0.5 ** (age_days / half_life_days)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class PlayerMapEntry:
    """One completed map from ONE player's perspective. `team_canonical` is
    provenance only - player form is team-independent by construction."""
    series_dt: pd.Timestamp
    game_id: str
    match_id: str
    team_canonical: str
    adr: float
    kast: float
    kd_balance: float
    assists_per_round: float

    def to_dict(self):
        d = asdict(self)
        d["series_dt"] = self.series_dt.isoformat()
        return d

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        d["series_dt"] = pd.Timestamp(d["series_dt"])
        return cls(**d)


@dataclass
class AppearanceEntry:
    """One appearance of a player FOR A SPECIFIC TEAM on one map."""
    series_dt: pd.Timestamp
    game_id: str
    player_id: int

    def to_dict(self):
        d = asdict(self)
        d["series_dt"] = self.series_dt.isoformat()
        return d

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        d["series_dt"] = pd.Timestamp(d["series_dt"])
        d["player_id"] = int(d["player_id"])
        return cls(**d)


@dataclass
class PlayerPerformanceState:
    player_id: int
    history: list = field(default_factory=list)     # list[PlayerMapEntry]

    def to_dict(self):
        return {"player_id": self.player_id, "history": [h.to_dict() for h in self.history]}

    @classmethod
    def from_dict(cls, d):
        return cls(player_id=int(d["player_id"]),
                    history=[PlayerMapEntry.from_dict(h) for h in d["history"]])


@dataclass
class TeamRosterState:
    canonical_name: str
    appearances: list = field(default_factory=list)  # list[AppearanceEntry]

    def to_dict(self):
        return {"canonical_name": self.canonical_name,
                "appearances": [a.to_dict() for a in self.appearances]}

    @classmethod
    def from_dict(cls, d):
        return cls(canonical_name=d["canonical_name"],
                    appearances=[AppearanceEntry.from_dict(a) for a in d["appearances"]])


class PlayerRosterStateStore:
    """Holds BOTH state types. Entries are created lazily only when a real
    completed map is applied; feature computation never creates or mutates
    state.

    Two structural invariants are enforced by construction, so malformed
    source duplication can never inflate history mass:
        at most ONE PlayerMapEntry  per (game_id, player_id)
        at most ONE AppearanceEntry per (game_id, team_canonical, player_id)
    """

    def __init__(self):
        self.players: dict[int, PlayerPerformanceState] = {}
        self.teams: dict[str, TeamRosterState] = {}
        self.processed_map_uids: set = set()
        self.player_map_keys: set = set()      # (game_id, player_id)
        self.appearance_keys: set = set()      # (game_id, team_canonical, player_id)

    def get_player(self, player_id):
        return self.players.get(int(player_id))

    def get_team(self, canonical_name):
        return self.teams.get(canonical_name)

    def _get_or_create_player(self, player_id):
        pid = int(player_id)
        if pid not in self.players:
            self.players[pid] = PlayerPerformanceState(player_id=pid)
        return self.players[pid]

    def _get_or_create_team(self, canonical_name):
        if canonical_name not in self.teams:
            self.teams[canonical_name] = TeamRosterState(canonical_name=canonical_name)
        return self.teams[canonical_name]

    def to_json(self, path, meta=None):
        payload = {
            "player_roster_engine_version": PLAYER_ROSTER_ENGINE_VERSION,
            "n_players": len(self.players),
            "n_teams": len(self.teams),
            "n_maps_processed": len(self.processed_map_uids),
            "meta": meta or {},
            "players": {str(pid): st.to_dict() for pid, st in self.players.items()},
            "teams": {name: st.to_dict() for name, st in self.teams.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)

    @classmethod
    def from_json(cls, path):
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        store = cls()
        for pid, pdict in payload["players"].items():
            st = PlayerPerformanceState.from_dict(pdict)
            store.players[int(pid)] = st
            for h in st.history:
                store.player_map_keys.add((h.game_id, st.player_id))
                store.processed_map_uids.add(h.game_id)
        for name, tdict in payload["teams"].items():
            st = TeamRosterState.from_dict(tdict)
            store.teams[name] = st
            for a in st.appearances:
                store.appearance_keys.add((a.game_id, name, a.player_id))
                store.processed_map_uids.add(a.game_id)
        store.loaded_meta = payload.get("meta", {})
        return store

    def snapshot_summary_df(self):
        """Flat, scalar-only, one row per player - fastparquet-safe. Full
        per-map history lives only in the JSON snapshot. No time-weighted
        statistic appears here: those are relative to a query time, not to a
        state dump."""
        rows = []
        for pid, st in self.players.items():
            hist = sorted(st.history, key=lambda h: h.series_dt)
            rows.append({
                "player_id": pid,
                "maps": len(hist),
                "distinct_teams": len({h.team_canonical for h in hist}),
                "mean_adr": float(np.mean([h.adr for h in hist])) if hist else np.nan,
                "mean_kast": float(np.mean([h.kast for h in hist])) if hist else np.nan,
                "mean_kd_balance": float(np.mean([h.kd_balance for h in hist])) if hist else np.nan,
                "mean_assists_per_round": float(np.mean([h.assists_per_round for h in hist])) if hist else np.nan,
                "first_map_datetime": hist[0].series_dt if hist else pd.NaT,
                "last_map_datetime": hist[-1].series_dt if hist else pd.NaT,
            })
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Strictly-prior player form
# ---------------------------------------------------------------------------

def compute_player_form(state, as_of, half_life_days=PLAYER_FORM_HALF_LIFE_DAYS):
    """One player's strictly-prior (`series_dt < as_of`) time-weighted form,
    over ALL of their prior maps regardless of which team they played for.
    The 90-day window applies to ROSTER INFERENCE only, never here."""
    as_of = pd.Timestamp(as_of)
    hist = [h for h in (state.history if state is not None else []) if h.series_dt < as_of]
    if not hist:
        return {
            "time_weighted_adr": COLD_START_PERFORMANCE,
            "time_weighted_kast": COLD_START_PERFORMANCE,
            "time_weighted_kd_balance": COLD_START_PERFORMANCE,
            "time_weighted_assists_per_round": COLD_START_PERFORMANCE,
            "player_history_mass": COLD_START_HISTORY_MASS,
            "prior_maps": 0,
        }
    weights = [recency_weight(as_of, h.series_dt, half_life_days) for h in hist]
    wsum = sum(weights)
    def wmean(attr):
        return sum(w * getattr(h, attr) for w, h in zip(weights, hist)) / wsum
    return {
        "time_weighted_adr": wmean("adr"),
        "time_weighted_kast": wmean("kast"),
        "time_weighted_kd_balance": wmean("kd_balance"),
        "time_weighted_assists_per_round": wmean("assists_per_round"),
        "player_history_mass": wsum,
        "prior_maps": len(hist),
    }


# ---------------------------------------------------------------------------
# Expected-roster inference (strictly prior team appearances only)
# ---------------------------------------------------------------------------

def _appearance_masses(store, team, as_of, lookback_days=ROSTER_LOOKBACK_DAYS,
                        half_life_days=PLAYER_FORM_HALF_LIFE_DAYS):
    """{player_id: (weighted_appearance_mass, last_appearance_dt)} over the
    half-open window [as_of - lookback_days, as_of). Half-open by
    construction: an appearance exactly at `as_of` belongs to the current
    series and is excluded."""
    state = store.get_team(team)
    if state is None:
        return {}
    as_of = pd.Timestamp(as_of)
    lower = as_of - timedelta(days=lookback_days)
    out = {}
    for a in state.appearances:
        if lower <= a.series_dt < as_of:
            w = recency_weight(as_of, a.series_dt, half_life_days)
            if a.player_id in out:
                mass, last = out[a.player_id]
                out[a.player_id] = (mass + w, max(last, a.series_dt))
            else:
                out[a.player_id] = (w, a.series_dt)
    return out


def infer_expected_roster(store, team, as_of, lookback_days=ROSTER_LOOKBACK_DAYS,
                           half_life_days=PLAYER_FORM_HALF_LIFE_DAYS, roster_size=ROSTER_SIZE):
    """The team's expected current lineup, inferred STRICTLY from prior
    appearances for that team. Deterministic ranking:
        1. highest recency-weighted appearance mass
        2. most recent appearance
        3. lowest player_id (final deterministic tie-break)
    Returns a list of (player_id, mass, last_dt), at most `roster_size` long.
    Fewer than `roster_size` available -> returns what exists; players are
    NEVER fabricated."""
    masses = _appearance_masses(store, team, as_of, lookback_days, half_life_days)
    ranked = sorted(masses.items(), key=lambda kv: (-kv[1][0], -kv[1][1].value, kv[0]))
    return [(pid, mass, last) for pid, (mass, last) in ranked[:roster_size]]


def _recent_team_maps(store, team, as_of, n):
    """The team's most recent `n` distinct game_ids strictly before `as_of`,
    newest first, as a list of (game_id, series_dt)."""
    state = store.get_team(team)
    if state is None:
        return []
    as_of = pd.Timestamp(as_of)
    seen = {}
    for a in state.appearances:
        if a.series_dt < as_of and a.game_id not in seen:
            seen[a.game_id] = a.series_dt
    ordered = sorted(seen.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    return ordered[:n]


def _players_in_map(store, team, game_id):
    state = store.get_team(team)
    if state is None:
        return set()
    return {a.player_id for a in state.appearances if a.game_id == game_id}


# ---------------------------------------------------------------------------
# Per-team roster feature block
# ---------------------------------------------------------------------------

def compute_team_roster_features(store, team, as_of):
    """One team's strictly-prior inferred-roster profile. Read-only: never
    creates or mutates state."""
    as_of = pd.Timestamp(as_of)
    roster = infer_expected_roster(store, team, as_of)
    roster_pids = [pid for pid, _m, _d in roster]

    forms = [compute_player_form(store.get_player(pid), as_of) for pid in roster_pids]
    with_history = [f for f in forms if f["player_history_mass"] > 0.0]

    def agg(key, fn):
        vals = [f[key] for f in with_history]
        return float(fn(vals)) if vals else COLD_START_PERFORMANCE

    # history mass is defined over the FULL inferred roster (0.0 for a player
    # with no prior observation) so it measures evidence, not just the
    # evidenced subset.
    masses = [f["player_history_mass"] for f in forms]
    mean_mass = float(np.mean(masses)) if masses else COLD_START_HISTORY_MASS
    min_mass = float(np.min(masses)) if masses else COLD_START_HISTORY_MASS

    # ---- roster stability ----
    last10 = _recent_team_maps(store, team, as_of, 10)
    last20 = _recent_team_maps(store, team, as_of, 20)
    unique10 = len({p for gid, _dt in last10 for p in _players_in_map(store, team, gid)})
    unique20 = len({p for gid, _dt in last20 for p in _players_in_map(store, team, gid)})

    masses_all = _appearance_masses(store, team, as_of)
    total_mass = sum(m for m, _d in masses_all.values())
    core_mass = sum(masses_all[pid][0] for pid in roster_pids if pid in masses_all)
    concentration = (core_mass / total_mass) if total_mass > 0 else COLD_START_RATIO

    if last10 and roster_pids:
        fracs = []
        for gid, _dt in last10:
            present = _players_in_map(store, team, gid)
            fracs.append(len([p for p in roster_pids if p in present]) / len(roster_pids))
        continuity = float(np.mean(fracs))
    else:
        continuity = COLD_START_RATIO

    return {
        "roster_size": len(roster_pids),
        "roster_form_player_count": len(with_history),
        "roster_mean_adr": agg("time_weighted_adr", np.mean),
        "roster_top_adr": agg("time_weighted_adr", np.max),
        "roster_bottom_adr": agg("time_weighted_adr", np.min),
        "roster_mean_kast": agg("time_weighted_kast", np.mean),
        "roster_top_kast": agg("time_weighted_kast", np.max),
        "roster_bottom_kast": agg("time_weighted_kast", np.min),
        "roster_mean_kd_balance": agg("time_weighted_kd_balance", np.mean),
        "roster_top_kd_balance": agg("time_weighted_kd_balance", np.max),
        "roster_bottom_kd_balance": agg("time_weighted_kd_balance", np.min),
        "roster_mean_assists_per_round": agg("time_weighted_assists_per_round", np.mean),
        "roster_mean_player_history_mass": mean_mass,
        "roster_min_player_history_mass": min_mass,
        "recent_unique_players_10_maps": unique10,
        "recent_unique_players_20_maps": unique20,
        "core5_appearance_concentration_90d": concentration,
        "core5_continuity_last_10": continuity,
    }


ROSTER_DIRECTIONAL_FEATURES = [
    "roster_mean_adr_diff", "roster_top_adr_diff", "roster_bottom_adr_diff",
    "roster_mean_kast_diff", "roster_top_kast_diff", "roster_bottom_kast_diff",
    "roster_mean_kd_balance_diff", "roster_top_kd_balance_diff", "roster_bottom_kd_balance_diff",
    "roster_mean_assists_per_round_diff",
    "recent_unique_players_10_maps_diff", "recent_unique_players_20_maps_diff",
    "core5_appearance_concentration_90d_diff", "core5_continuity_last_10_diff",
    "roster_mean_player_history_mass_diff",
]
ROSTER_SYMMETRIC_FEATURES = [
    "roster_size_min", "both_teams_have_5_inferred_players", "roster_min_player_history_mass",
    "roster_core_concentration_min", "roster_core_continuity_last10_min", "roster_form_players_min",
]

# The 10 PERFORMANCE diffs below are the only features that may be NaN, and
# they are NaN exactly when `roster_form_players_min == 0` (i.e. at least one
# side has no inferred roster player with any prior observation). Asserted in
# the builder and re-checked in scripts/validate_phase5c.py.
ROSTER_PERFORMANCE_DIFFS = [
    "roster_mean_adr_diff", "roster_top_adr_diff", "roster_bottom_adr_diff",
    "roster_mean_kast_diff", "roster_top_kast_diff", "roster_bottom_kast_diff",
    "roster_mean_kd_balance_diff", "roster_top_kd_balance_diff", "roster_bottom_kd_balance_diff",
    "roster_mean_assists_per_round_diff",
]


def build_future_player_roster_features(store, team1, team2, as_of_datetime):
    """PURE / READ-ONLY future-application API.

    Takes only a state store, two TEAM identities and a cutoff datetime -
    no target, no score, and deliberately NO LINEUP ARGUMENT, so the target
    series' actual five players cannot enter even by accident. This is the
    exact function the offline builder uses, so training and inference
    cannot silently diverge."""
    as_of = pd.Timestamp(as_of_datetime)
    f1 = compute_team_roster_features(store, team1, as_of)
    f2 = compute_team_roster_features(store, team2, as_of)

    out = {}
    for k, v in f1.items():
        out[f"team1_{k}"] = v
    for k, v in f2.items():
        out[f"team2_{k}"] = v

    for base in ["roster_mean_adr", "roster_top_adr", "roster_bottom_adr",
                  "roster_mean_kast", "roster_top_kast", "roster_bottom_kast",
                  "roster_mean_kd_balance", "roster_top_kd_balance", "roster_bottom_kd_balance",
                  "roster_mean_assists_per_round",
                  "recent_unique_players_10_maps", "recent_unique_players_20_maps",
                  "core5_appearance_concentration_90d", "core5_continuity_last_10",
                  "roster_mean_player_history_mass"]:
        out[f"{base}_diff"] = f1[base] - f2[base]

    out["roster_size_min"] = min(f1["roster_size"], f2["roster_size"])
    out["both_teams_have_5_inferred_players"] = int(f1["roster_size"] >= ROSTER_SIZE
                                                     and f2["roster_size"] >= ROSTER_SIZE)
    out["roster_min_player_history_mass"] = min(f1["roster_min_player_history_mass"],
                                                 f2["roster_min_player_history_mass"])
    out["roster_core_concentration_min"] = min(f1["core5_appearance_concentration_90d"],
                                                f2["core5_appearance_concentration_90d"])
    out["roster_core_continuity_last10_min"] = min(f1["core5_continuity_last_10"],
                                                    f2["core5_continuity_last_10"])
    # Distinct from roster_size_min: a team can have five INFERRED players but
    # usable prior box scores for only some of them.
    out["roster_form_players_min"] = min(f1["roster_form_player_count"], f2["roster_form_player_count"])
    return out


# ---------------------------------------------------------------------------
# State updates (Phase B only)
# ---------------------------------------------------------------------------

def apply_player_observation(store, row):
    """Apply ONE (game_id, side, player) observation. Must only be called in
    Phase B. Idempotent per structural key, so malformed source duplication
    can never inflate history mass or appearance counts."""
    pid = int(row["player_id"])
    gid = str(row["game_id"])
    team = row["team_canonical"]
    series_dt = pd.Timestamp(row["series_datetime"])

    # B. team-keyed appearance: only when that canonical team is eligible
    if bool(row["team_eligible"]):
        akey = (gid, team, pid)
        if akey not in store.appearance_keys:
            store.appearance_keys.add(akey)
            store._get_or_create_team(team).appearances.append(
                AppearanceEntry(series_dt=series_dt, game_id=gid, player_id=pid))

    # A. player-keyed performance: independent of team eligibility
    if bool(row["has_usable_stats"]):
        pkey = (gid, pid)
        if pkey not in store.player_map_keys:
            store.player_map_keys.add(pkey)
            store._get_or_create_player(pid).history.append(PlayerMapEntry(
                series_dt=series_dt, game_id=gid, match_id=str(row["match_id"]),
                team_canonical=team, adr=float(row["adr"]), kast=float(row["kast"]),
                kd_balance=float(row["kd_balance"]),
                assists_per_round=float(row["assists_per_round"])))

    store.processed_map_uids.add(gid)


# ---------------------------------------------------------------------------
# Two-phase chronological driver
# ---------------------------------------------------------------------------

def process_player_roster_stream(store, player_rows, series_requests=None, emit_features=True):
    """
    player_rows: long observations (one row per game_id/side/player) carrying
    the AUTHORITATIVE `series_datetime`.
    series_requests: the series needing pre-match features, supplied
    SEPARATELY - roughly half of the series universe has no map/player rows
    of its own but still needs roster features from the two teams' PRIOR
    history, and driving emission off the player stream alone would silently
    drop every such match.

    Batching is on `series_datetime`:
        PHASE A (read):  emit ONE pre-series feature vector per requested
                          series in the batch, from the state exactly as it
                          was BEFORE the batch. No mutation.
        PHASE B (write): only after every read above, apply every player
                          observation in the batch.
    Guarantees a series never sees its own maps, Map 1 never reaches Map 2,
    and two series sharing a timestamp never see each other.
    """
    missing = set(REQUIRED_PLAYER_STREAM_COLUMNS) - set(player_rows.columns)
    if missing:
        raise ValueError(f"process_player_roster_stream: missing required columns {sorted(missing)}")

    obs = player_rows.sort_values(
        ["series_datetime", "match_id", "game_id", "side", "slot"]).reset_index(drop=True)

    if series_requests is None:
        reqs = pd.DataFrame(columns=SERIES_REQUEST_COLUMNS)
    else:
        rmissing = set(SERIES_REQUEST_COLUMNS) - set(series_requests.columns)
        if rmissing:
            raise ValueError(f"process_player_roster_stream: series_requests missing {sorted(rmissing)}")
        reqs = series_requests.sort_values(["series_datetime", "match_id"]).reset_index(drop=True)

    timestamps = sorted(set(obs["series_datetime"]) | set(reqs["series_datetime"]))
    obs_by_ts = {ts: g for ts, g in obs.groupby("series_datetime", sort=False)}
    req_by_ts = {ts: g for ts, g in reqs.groupby("series_datetime", sort=False)}

    emitted = []
    for ts in timestamps:
        # ---------------- PHASE A: read-only emission ----------------
        if emit_features and ts in req_by_ts:
            for _, q in req_by_ts[ts].iterrows():
                if not (bool(q["team1_eligible"]) and bool(q["team2_eligible"])):
                    continue
                feats = build_future_player_roster_features(
                    store, q["team1_canonical"], q["team2_canonical"], ts)
                emitted.append({"match_id": q["match_id"], "series_datetime": ts, **feats})

        # ---------------- PHASE B: apply results (only now) ----------------
        if ts in obs_by_ts:
            for _, r in obs_by_ts[ts].iterrows():
                apply_player_observation(store, r)

    return emitted
