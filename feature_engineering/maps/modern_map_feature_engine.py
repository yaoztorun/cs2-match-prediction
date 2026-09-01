"""
Reusable, dataset-agnostic MODERN SELECTED-MAP state / feature engine (Phase 6C).

Adds exactly the state Phase 6B's feature-family importance results were
missing: RECENT / opponent-adjusted selected-map performance, selected-map
strength relative to a team's general strength, and CURRENT inferred-roster
experience on the selected map. Mirrors the design of every prior engine
(feature_engine.py, map_feature_engine.py, team_form_engine.py,
player_roster_feature_engine.py) and reuses their primitives directly rather
than reimplementing them - none of those four files is modified anywhere in
this module.

THREE INDEPENDENT, ORDER-INDEPENDENT LEDGERS
---------------------------------------------
Unlike map_feature_engine.MapStateStore (whose `map_elo` is a MUTATED RUNNING
SCALAR, genuinely order-dependent) or team_form_engine.TeamFormStateStore
(same reason), every store this module owns is a flat, dated ledger with NO
cumulative scalar:

    TeamSelectedMapState   (team_canonical, map_name) -> [SelectedMapHistoryEntry]
    PlayerSelectedMapState (player_id, map_name)       -> [PlayerSelectedMapEntry]
    TeamSelectedMapRosterState (team_canonical, map_name) -> [SelectedMapAppearanceEntry]

Every query function below filters strictly by `series_dt < as_of` INSIDE the
query itself (exactly like player_roster_feature_engine's own
`compute_player_form`/`_appearance_masses`), so it is safe and CORRECT to
build these three ledgers via a single full pass over the ENTIRE historical
stream in ANY order, then query them with any historical `as_of` - the
strictly-prior filtering happens at read time, not from replay order. This is
proven, not assumed: `apply_selected_map_team_result` inserts only STATIC,
ALREADY-COMPUTED values (`own_pre_series_elo`/`opponent_pre_series_elo`, read
from the frozen `series_team_form_states_v1.parquet` audit artifact - see
feature_engineering/maps/modern_map_stream_common.py) rather than a locally mutated ELO, so
even the opponent-adjusted residual introduces no order dependency.

The ONE genuinely order-dependent quantity this phase needs - raw per-team
`map_elo` and recent-map-pool statistics, both for section 13's
map-vs-overall/map-vs-pool comparisons - is read from a properly
series_datetime-batched replay of the FROZEN, UNMODIFIED
map_feature_engine.MapStateStore (via its own public `apply_map_result`,
`compute_team_map_features`, `recent_map_pool`), driven by
`process_modern_map_stream` below.

TRUSTED-OPPONENT GATING (same principle as team_form_engine.py, restated here
at map granularity, per the Phase 6C approval corrections)
--------------------------------------------------------------------------
An eligible team's OWN selected-map history (result, margin) may update from
a map played against an identity-ineligible opponent - that team's own result
is a real fact - flagged `opponent_identity_trusted`. But OPPONENT-ADJUSTED
information requires a reliable, persistent opponent identity:
  * `weighted_map_wr` / `weighted_map_margin` / `map_recent_history_mass` use
    ALL of the team's own eligible selected-map history (trusted or not).
  * `weighted_map_performance_residual` / `weighted_map_opponent_elo` /
    `map_adjusted_history_mass` use ONLY entries with
    `opponent_identity_trusted == True`.

DIFFERENCE-STYLE SPECIALIZATION, NOT RATIOS (per the Phase 6C approval
corrections)
--------------------------------------------------------------------------
Every "is this map unusually strong/weak for this team" comparison is a
SUBTRACTION, never a division: `selected_map_elo_vs_overall`,
`selected_map_elo_vs_pool_mean`, `selected_map_wr_vs_pool_mean`, and
`roster_map_kast_specialization` (per-player, before roster aggregation) are
all differences. Their cold-start neutral value is `0.0` - never a fabricated
ratio fallback. `selected_map_rank_percentile` is the ONE feature that is
naturally scaled to [0,1] rather than a difference; its cold start (map
absent from the team's recent pool, or the pool itself is empty) is `0.5`,
paired with the `selected_map_in_both_recent_pools` confidence flag.

NO MAP_SLOT
-----------
The map-order audit (see feature_engineering/maps/build_map_features_v3_modern_map.py) found
no independent signal - every map of a series shares one identical raw
`datetime` in this dataset - so map order cannot be independently verified.
`map_slot` is therefore never computed or emitted anywhere in this module.
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import timedelta

import numpy as np
import pandas as pd

from feature_engineering.series.feature_engine import ELO_INITIAL, elo_expected
from feature_engineering.maps.map_feature_engine import (
    COLD_START_MAP_ELO, COLD_START_SMOOTHED_WR,
    compute_team_map_features, recent_map_pool, normalized_round_margin,
)
from feature_engineering.roster.player_roster_feature_engine import infer_expected_roster, compute_player_form

MODERN_MAP_ENGINE_VERSION = "1.0.0"

MAP_FORM_HALF_LIFE_DAYS = 90.0   # fixed engineering constant for this phase - NOT tuned

# Cold-start contract (documented once, applied everywhere)
COLD_START_MAP_WR = 0.5
COLD_START_MAP_MARGIN = 0.0
COLD_START_MAP_RESIDUAL = 0.0
COLD_START_MAP_OPPONENT_ELO = ELO_INITIAL   # 1500.0
COLD_START_MASS = 0.0
COLD_START_SPECIALIZATION = 0.0             # difference-style, never a ratio fallback
COLD_START_RANK_PERCENTILE = 0.5
COLD_START_PLAYER_MAP_STAT = float("nan")   # genuinely missing - never fabricated
COLD_START_CONTINUITY = 0.0


def recency_weight(as_of, dt, half_life_days=MAP_FORM_HALF_LIFE_DAYS):
    age_days = (as_of - dt).total_seconds() / 86400.0
    return 0.5 ** (age_days / half_life_days)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SelectedMapHistoryEntry:
    """One completed selected-map result from ONE team's perspective.
    `own_pre_series_elo`/`opponent_pre_series_elo` are STATIC values already
    computed by the frozen Phase 5B.2 form-engine replay (never recomputed
    here), so this entry introduces no order dependency of its own."""
    series_dt: pd.Timestamp
    match_id: str
    game_id: str
    win: int
    normalized_margin: float
    opponent_identity_trusted: bool
    own_pre_series_elo: float
    opponent_pre_series_elo: float
    expected_win_prob: float
    performance_residual: float

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
class PlayerSelectedMapEntry:
    series_dt: pd.Timestamp
    game_id: str
    team_canonical: str    # provenance only - player-map form is team-independent
    adr: float
    kast: float
    kd_balance: float

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
class SelectedMapAppearanceEntry:
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


class ModernMapStateStore:
    """Holds all three ledgers. Entries are created lazily only when a real
    completed observation is applied; feature computation never creates or
    mutates state."""

    def __init__(self):
        self.team_map: dict[tuple, list] = {}          # (team, map_name) -> [SelectedMapHistoryEntry]
        self.player_map: dict[tuple, list] = {}         # (player_id, map_name) -> [PlayerSelectedMapEntry]
        self.team_map_roster: dict[tuple, list] = {}    # (team, map_name) -> [SelectedMapAppearanceEntry]
        self.processed_map_uids: set = set()
        self.player_map_keys: set = set()               # (game_id, player_id, map_name)
        self.appearance_keys: set = set()                # (game_id, team, player_id, map_name)

    def get_team_map(self, team, map_name):
        return self.team_map.get((team, map_name), [])

    def get_player_map(self, player_id, map_name):
        return self.player_map.get((int(player_id), map_name), [])

    def get_team_map_roster(self, team, map_name):
        return self.team_map_roster.get((team, map_name), [])

    def to_json(self, path, meta=None):
        payload = {
            "modern_map_engine_version": MODERN_MAP_ENGINE_VERSION,
            "n_team_map_ledgers": len(self.team_map),
            "n_player_map_ledgers": len(self.player_map),
            "n_team_map_roster_ledgers": len(self.team_map_roster),
            "n_maps_processed": len(self.processed_map_uids),
            "meta": meta or {},
            "team_map": {f"{t}||{m}": [e.to_dict() for e in hist] for (t, m), hist in self.team_map.items()},
            "player_map": {f"{p}||{m}": [e.to_dict() for e in hist] for (p, m), hist in self.player_map.items()},
            "team_map_roster": {f"{t}||{m}": [e.to_dict() for e in hist]
                                 for (t, m), hist in self.team_map_roster.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)

    @classmethod
    def from_json(cls, path):
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        store = cls()
        for key, hist in payload["team_map"].items():
            t, m = key.split("||", 1)
            store.team_map[(t, m)] = [SelectedMapHistoryEntry.from_dict(e) for e in hist]
            for e in hist:
                store.processed_map_uids.add(e["game_id"])
        for key, hist in payload["player_map"].items():
            p, m = key.split("||", 1)
            store.player_map[(int(p), m)] = [PlayerSelectedMapEntry.from_dict(e) for e in hist]
            for e in hist:
                store.player_map_keys.add((e["game_id"], int(p), m))
        for key, hist in payload["team_map_roster"].items():
            t, m = key.split("||", 1)
            store.team_map_roster[(t, m)] = [SelectedMapAppearanceEntry.from_dict(e) for e in hist]
            for e in hist:
                store.appearance_keys.add((e["game_id"], t, int(e["player_id"]), m))
        store.loaded_meta = payload.get("meta", {})
        return store


# ---------------------------------------------------------------------------
# Strictly-prior team-level selected-map statistics (sections 8-11)
# ---------------------------------------------------------------------------

def compute_team_selected_map_recent_features(history, as_of, half_life_days=MAP_FORM_HALF_LIFE_DAYS):
    """`history` is a flat list[SelectedMapHistoryEntry] (possibly containing
    entries at or after `as_of` - filtered out here, not by the caller)."""
    as_of = pd.Timestamp(as_of)
    hist_all = sorted([h for h in history if h.series_dt < as_of], key=lambda h: h.series_dt)
    hist_trusted = [h for h in hist_all if h.opponent_identity_trusted]

    if hist_all:
        w_all = [recency_weight(as_of, h.series_dt, half_life_days) for h in hist_all]
        wsum_all = sum(w_all)
        weighted_map_wr = (sum(w * h.win for w, h in zip(w_all, hist_all)) + 2) / (wsum_all + 4)
        weighted_map_margin = sum(w * h.normalized_margin for w, h in zip(w_all, hist_all)) / wsum_all
        map_recent_history_mass = wsum_all
    else:
        weighted_map_wr = COLD_START_MAP_WR
        weighted_map_margin = COLD_START_MAP_MARGIN
        map_recent_history_mass = COLD_START_MASS

    if hist_trusted:
        w_tr = [recency_weight(as_of, h.series_dt, half_life_days) for h in hist_trusted]
        wsum_tr = sum(w_tr)
        weighted_map_performance_residual = sum(
            w * h.performance_residual for w, h in zip(w_tr, hist_trusted)) / wsum_tr
        weighted_map_opponent_elo = sum(
            w * h.opponent_pre_series_elo for w, h in zip(w_tr, hist_trusted)) / wsum_tr
        map_adjusted_history_mass = wsum_tr
    else:
        weighted_map_performance_residual = COLD_START_MAP_RESIDUAL
        weighted_map_opponent_elo = COLD_START_MAP_OPPONENT_ELO
        map_adjusted_history_mass = COLD_START_MASS

    return {
        "weighted_map_wr": weighted_map_wr,
        "weighted_map_margin": weighted_map_margin,
        "weighted_map_performance_residual": weighted_map_performance_residual,
        "weighted_map_opponent_elo": weighted_map_opponent_elo,
        "map_recent_history_mass": map_recent_history_mass,
        "map_adjusted_history_mass": map_adjusted_history_mass,
    }


# ---------------------------------------------------------------------------
# Map specialization relative to general team strength (section 13) -
# DIFFERENCES ONLY, never ratios. `map_state` must be a MapStateStore
# positioned at "state strictly before `as_of`" by the caller (this function
# is pure and does no positioning of its own).
# ---------------------------------------------------------------------------

def compute_map_specialization_features(map_state, team, map_name, overall_elo, weighted_map_wr, as_of):
    as_of = pd.Timestamp(as_of)
    map_elo = compute_team_map_features(map_state.get(team, map_name), as_of)["map_elo"]
    pool = recent_map_pool(map_state, team, as_of)

    if pool:
        # sorted(pool), never raw set iteration - a bare Python set's iteration order depends on
        # PYTHONHASHSEED (randomized per interpreter start for strings), which would otherwise make
        # np.mean's floating-point summation order - and therefore the exact byte output of the
        # feature-engineering rebuild - nondeterministic across runs. Matches
        # map_feature_engine._pool_profile's own established convention exactly.
        ordered_pool = sorted(pool)
        elo_by_map = {m: compute_team_map_features(map_state.get(team, m), as_of)["map_elo"]
                      for m in ordered_pool}
        wr_by_map = {m: compute_team_map_features(map_state.get(team, m), as_of)["map_smoothed_win_rate"]
                     for m in ordered_pool}
        pool_mean_elo = float(np.mean([elo_by_map[m] for m in ordered_pool]))
        pool_mean_wr = float(np.mean([wr_by_map[m] for m in ordered_pool]))
    else:
        elo_by_map = {}
        pool_mean_elo = COLD_START_MAP_ELO
        pool_mean_wr = COLD_START_SMOOTHED_WR

    if map_name in pool:
        ranked = sorted(pool, key=lambda m: (-elo_by_map[m], m))
        idx = ranked.index(map_name)
        rank_percentile = 1.0 if len(pool) == 1 else 1.0 - idx / (len(pool) - 1)
        in_pool = True
    else:
        rank_percentile = COLD_START_RANK_PERCENTILE
        in_pool = False

    return {
        "elo_vs_overall": map_elo - overall_elo,
        "elo_vs_pool_mean": map_elo - pool_mean_elo,
        "wr_vs_pool_mean": weighted_map_wr - pool_mean_wr,
        "rank_percentile": rank_percentile,
        "in_pool": in_pool,
    }


# ---------------------------------------------------------------------------
# Current inferred roster on the selected map (sections 14-16)
# ---------------------------------------------------------------------------

def compute_player_selected_map_form(history, as_of, half_life_days=MAP_FORM_HALF_LIFE_DAYS):
    """One player's strictly-prior time-weighted stats on ONE selected map."""
    as_of = pd.Timestamp(as_of)
    hist = [h for h in history if h.series_dt < as_of]
    if not hist:
        return {"time_weighted_map_adr": COLD_START_PLAYER_MAP_STAT,
                "time_weighted_map_kast": COLD_START_PLAYER_MAP_STAT,
                "time_weighted_map_kd_balance": COLD_START_PLAYER_MAP_STAT,
                "player_map_history_mass": COLD_START_MASS}
    weights = [recency_weight(as_of, h.series_dt, half_life_days) for h in hist]
    wsum = sum(weights)

    def wmean(attr):
        return sum(w * getattr(h, attr) for w, h in zip(weights, hist)) / wsum

    return {"time_weighted_map_adr": wmean("adr"), "time_weighted_map_kast": wmean("kast"),
            "time_weighted_map_kd_balance": wmean("kd_balance"), "player_map_history_mass": wsum}


def compute_current_roster_map_features(player_roster_state, modern_state, team, map_name, as_of):
    """`player_roster_state` (order-independent, see module docstring) and
    `modern_state` are both READ-ONLY here."""
    as_of = pd.Timestamp(as_of)
    roster = infer_expected_roster(player_roster_state, team, as_of)
    roster_pids = [pid for pid, _m, _d in roster]

    map_forms = [compute_player_selected_map_form(modern_state.get_player_map(pid, map_name), as_of)
                 for pid in roster_pids]
    global_forms = [compute_player_form(player_roster_state.get_player(pid), as_of) for pid in roster_pids]
    with_history = [f for f in map_forms if f["player_map_history_mass"] > 0.0]

    def agg(key, fn):
        vals = [f[key] for f in with_history]
        return float(fn(vals)) if vals else COLD_START_PLAYER_MAP_STAT

    masses = [f["player_map_history_mass"] for f in map_forms]
    mean_mass = float(np.mean(masses)) if masses else COLD_START_MASS

    spec_vals = []
    for mform, gform in zip(map_forms, global_forms):
        if mform["player_map_history_mass"] > 0.0 and gform["player_history_mass"] > 0.0:
            spec_vals.append(mform["time_weighted_map_kast"] - gform["time_weighted_kast"])
        else:
            spec_vals.append(COLD_START_SPECIALIZATION)   # neutral, difference-style - never a ratio fallback
    roster_map_kast_specialization = float(np.mean(spec_vals)) if spec_vals else COLD_START_SPECIALIZATION

    return {
        "roster_map_mean_kast": agg("time_weighted_map_kast", np.mean),
        "roster_map_bottom_kast": agg("time_weighted_map_kast", np.min),
        "roster_map_mean_adr": agg("time_weighted_map_adr", np.mean),
        "roster_map_mean_kd_balance": agg("time_weighted_map_kd_balance", np.mean),
        "roster_map_mean_history_mass": mean_mass,
        "roster_map_players_with_history": len(with_history),
        "roster_map_kast_specialization": roster_map_kast_specialization,
    }


# ---------------------------------------------------------------------------
# Current-core selected-map continuity (section 17)
# ---------------------------------------------------------------------------

def compute_current_core_map_continuity_features(player_roster_state, modern_state, team, map_name, as_of,
                                                   half_life_days=MAP_FORM_HALF_LIFE_DAYS):
    as_of = pd.Timestamp(as_of)
    roster_pids = {pid for pid, _m, _d in infer_expected_roster(player_roster_state, team, as_of)}
    appearances = [a for a in modern_state.get_team_map_roster(team, map_name) if a.series_dt < as_of]

    if not appearances or not roster_pids:
        return {"continuity": COLD_START_CONTINUITY, "mass": COLD_START_MASS}

    by_game: dict[str, dict] = {}
    for a in appearances:
        g = by_game.setdefault(a.game_id, {"dt": a.series_dt, "players": set()})
        g["players"].add(a.player_id)
        g["dt"] = min(g["dt"], a.series_dt)

    weighted_overlap_sum, weight_sum = 0.0, 0.0
    for g in by_game.values():
        w = recency_weight(as_of, g["dt"], half_life_days)
        overlap = len(roster_pids & g["players"]) / len(roster_pids)
        weighted_overlap_sum += w * overlap
        weight_sum += w

    continuity = (weighted_overlap_sum / weight_sum) if weight_sum > 0 else COLD_START_CONTINUITY
    return {"continuity": continuity, "mass": weight_sum}


# ---------------------------------------------------------------------------
# Composed future-application feature block (25: 18 directional + 7 symmetric)
# ---------------------------------------------------------------------------

MODERN_MAP_DIRECTIONAL_FEATURES = [
    # team-level recent / opponent-adjusted selected-map form (5)
    "time_weighted_map_wr_diff", "time_weighted_map_margin_diff",
    "time_weighted_map_performance_residual_diff", "time_weighted_map_opponent_elo_diff",
    "map_recent_history_mass_diff",
    # map specialization relative to general team strength (4)
    "selected_map_elo_vs_overall_diff", "selected_map_elo_vs_pool_mean_diff",
    "selected_map_wr_vs_pool_mean_diff", "selected_map_rank_percentile_diff",
    # current-roster selected-map performance (6)
    "roster_map_mean_kast_diff", "roster_map_bottom_kast_diff", "roster_map_mean_adr_diff",
    "roster_map_mean_kd_balance_diff", "roster_map_mean_history_mass_diff",
    "roster_map_players_with_history_diff",
    # player map specialization (1)
    "roster_map_kast_specialization_diff",
    # current-core selected-map continuity (2)
    "current_core_map_continuity_diff", "current_core_map_history_mass_diff",
]
MODERN_MAP_SYMMETRIC_FEATURES = [
    "map_recent_history_mass_min", "both_teams_have_recent_selected_map_history",
    "map_adjusted_history_mass_min", "selected_map_in_both_recent_pools",
    "roster_map_players_with_history_min", "roster_map_history_mass_min",
    "current_core_map_continuity_min",
]
assert len(MODERN_MAP_DIRECTIONAL_FEATURES) == 18
assert len(MODERN_MAP_SYMMETRIC_FEATURES) == 7

# NaN is permitted only here, and only exactly where roster_map_players_with_history_min == 0
# (mirrors Phase 5C's ROSTER_PERFORMANCE_DIFFS contract precisely). Every other new feature -
# including the specialization difference - is a documented finite neutral, never NaN.
MODERN_MAP_NAN_CAPABLE_FEATURES = [
    "roster_map_mean_kast_diff", "roster_map_bottom_kast_diff",
    "roster_map_mean_adr_diff", "roster_map_mean_kd_balance_diff",
]


def build_future_modern_map_state_features(map_state, modern_state, player_roster_state,
                                            team1, team2, map_name, as_of_datetime,
                                            team1_overall_elo, team2_overall_elo):
    """PURE / READ-ONLY. `map_state` must already reflect state strictly
    before `as_of_datetime` (positioned by the caller - see
    process_modern_map_stream for the historical build, or the future
    composer for live application). No target, no score, no lineup
    parameter anywhere in this call chain."""
    as_of = pd.Timestamp(as_of_datetime)

    r1 = compute_team_selected_map_recent_features(modern_state.get_team_map(team1, map_name), as_of)
    r2 = compute_team_selected_map_recent_features(modern_state.get_team_map(team2, map_name), as_of)

    s1 = compute_map_specialization_features(map_state, team1, map_name, team1_overall_elo,
                                              r1["weighted_map_wr"], as_of)
    s2 = compute_map_specialization_features(map_state, team2, map_name, team2_overall_elo,
                                              r2["weighted_map_wr"], as_of)

    c1 = compute_current_roster_map_features(player_roster_state, modern_state, team1, map_name, as_of)
    c2 = compute_current_roster_map_features(player_roster_state, modern_state, team2, map_name, as_of)

    d1 = compute_current_core_map_continuity_features(player_roster_state, modern_state, team1, map_name, as_of)
    d2 = compute_current_core_map_continuity_features(player_roster_state, modern_state, team2, map_name, as_of)

    out = {}
    out["time_weighted_map_wr_diff"] = r1["weighted_map_wr"] - r2["weighted_map_wr"]
    out["time_weighted_map_margin_diff"] = r1["weighted_map_margin"] - r2["weighted_map_margin"]
    out["time_weighted_map_performance_residual_diff"] = (
        r1["weighted_map_performance_residual"] - r2["weighted_map_performance_residual"])
    out["time_weighted_map_opponent_elo_diff"] = r1["weighted_map_opponent_elo"] - r2["weighted_map_opponent_elo"]
    out["map_recent_history_mass_diff"] = r1["map_recent_history_mass"] - r2["map_recent_history_mass"]
    out["map_recent_history_mass_min"] = min(r1["map_recent_history_mass"], r2["map_recent_history_mass"])
    out["both_teams_have_recent_selected_map_history"] = int(
        r1["map_recent_history_mass"] > 0.0 and r2["map_recent_history_mass"] > 0.0)
    out["map_adjusted_history_mass_min"] = min(r1["map_adjusted_history_mass"], r2["map_adjusted_history_mass"])

    out["selected_map_elo_vs_overall_diff"] = s1["elo_vs_overall"] - s2["elo_vs_overall"]
    out["selected_map_elo_vs_pool_mean_diff"] = s1["elo_vs_pool_mean"] - s2["elo_vs_pool_mean"]
    out["selected_map_wr_vs_pool_mean_diff"] = s1["wr_vs_pool_mean"] - s2["wr_vs_pool_mean"]
    out["selected_map_rank_percentile_diff"] = s1["rank_percentile"] - s2["rank_percentile"]
    out["selected_map_in_both_recent_pools"] = int(s1["in_pool"] and s2["in_pool"])

    out["roster_map_mean_kast_diff"] = c1["roster_map_mean_kast"] - c2["roster_map_mean_kast"]
    out["roster_map_bottom_kast_diff"] = c1["roster_map_bottom_kast"] - c2["roster_map_bottom_kast"]
    out["roster_map_mean_adr_diff"] = c1["roster_map_mean_adr"] - c2["roster_map_mean_adr"]
    out["roster_map_mean_kd_balance_diff"] = c1["roster_map_mean_kd_balance"] - c2["roster_map_mean_kd_balance"]
    out["roster_map_mean_history_mass_diff"] = c1["roster_map_mean_history_mass"] - c2["roster_map_mean_history_mass"]
    out["roster_map_players_with_history_diff"] = (
        c1["roster_map_players_with_history"] - c2["roster_map_players_with_history"])
    out["roster_map_kast_specialization_diff"] = (
        c1["roster_map_kast_specialization"] - c2["roster_map_kast_specialization"])
    out["roster_map_players_with_history_min"] = min(
        c1["roster_map_players_with_history"], c2["roster_map_players_with_history"])
    out["roster_map_history_mass_min"] = min(c1["roster_map_mean_history_mass"], c2["roster_map_mean_history_mass"])

    out["current_core_map_continuity_diff"] = d1["continuity"] - d2["continuity"]
    out["current_core_map_history_mass_diff"] = d1["mass"] - d2["mass"]
    out["current_core_map_continuity_min"] = min(d1["continuity"], d2["continuity"])

    expected_keys = set(MODERN_MAP_DIRECTIONAL_FEATURES) | set(MODERN_MAP_SYMMETRIC_FEATURES)
    assert set(out.keys()) == expected_keys, \
        f"composed modern-map feature dict disagrees with the declared schema: {set(out.keys()) ^ expected_keys}"
    return out


# ---------------------------------------------------------------------------
# State updates (apply only - never called during feature emission)
# ---------------------------------------------------------------------------

def apply_selected_map_team_result(store, row):
    """Apply ONE completed map's team-level selected-map observation. `row`
    must carry `team1_pre_series_elo`/`team2_pre_series_elo` - STATIC values
    already joined in by feature_engineering/maps/modern_map_stream_common.py from the frozen
    Phase 5B.2 audit artifact, never recomputed here - and a
    `pre_series_elo_known` boolean, False for the small number of map rows
    whose containing match_id has no series-level ELO row at all (see that
    module's docstring). When False, the joined ELO values are a harmless
    fillna placeholder; `opponent_identity_trusted` is ANDed with this flag
    so the placeholder can never enter a trusted-population aggregate, while
    each eligible side's own win/margin fact still updates (matching
    map_feature_engine's identity policy exactly). Order-independent: safe
    to call in any order over the full historical stream (see module
    docstring)."""
    t1, t2 = row["team1_canonical"], row["team2_canonical"]
    e1, e2 = bool(row["team1_eligible"]), bool(row["team2_eligible"])
    elo_known = bool(row.get("pre_series_elo_known", True))
    map_name = row["map_name"]
    series_dt = pd.Timestamp(row["series_datetime"])
    s1, s2 = int(row["score1_game"]), int(row["score2_game"])
    win1 = int(row["team1_map_win"])
    elo1, elo2 = float(row["team1_pre_series_elo"]), float(row["team2_pre_series_elo"])

    expected1 = elo_expected(elo1, elo2)
    expected2 = 1.0 - expected1
    residual1 = win1 - expected1
    residual2 = (1 - win1) - expected2

    if e1:
        store.team_map.setdefault((t1, map_name), []).append(SelectedMapHistoryEntry(
            series_dt=series_dt, match_id=str(row["match_id"]), game_id=str(row["game_id"]),
            win=win1, normalized_margin=normalized_round_margin(s1, s2),
            opponent_identity_trusted=e2 and elo_known,
            own_pre_series_elo=elo1, opponent_pre_series_elo=elo2,
            expected_win_prob=expected1, performance_residual=residual1))
    if e2:
        store.team_map.setdefault((t2, map_name), []).append(SelectedMapHistoryEntry(
            series_dt=series_dt, match_id=str(row["match_id"]), game_id=str(row["game_id"]),
            win=1 - win1, normalized_margin=normalized_round_margin(s2, s1),
            opponent_identity_trusted=e1 and elo_known,
            own_pre_series_elo=elo2, opponent_pre_series_elo=elo1,
            expected_win_prob=expected2, performance_residual=residual2))

    store.processed_map_uids.add(str(row["game_id"]))


def apply_selected_map_player_observation(store, row):
    """Apply ONE (game_id, side, player) selected-map observation. `row` must
    carry `map_name` (joined in by feature_engineering/maps/modern_map_stream_common.py) in
    addition to player_roster_stream_common's own columns. Idempotent per
    structural key. Order-independent."""
    pid = int(row["player_id"])
    gid = str(row["game_id"])
    team = row["team_canonical"]
    map_name = row["map_name"]
    series_dt = pd.Timestamp(row["series_datetime"])

    if bool(row["team_eligible"]):
        akey = (gid, team, pid, map_name)
        if akey not in store.appearance_keys:
            store.appearance_keys.add(akey)
            store.team_map_roster.setdefault((team, map_name), []).append(
                SelectedMapAppearanceEntry(series_dt=series_dt, game_id=gid, player_id=pid))

    if bool(row["has_usable_stats"]):
        pkey = (gid, pid, map_name)
        if pkey not in store.player_map_keys:
            store.player_map_keys.add(pkey)
            store.player_map.setdefault((pid, map_name), []).append(PlayerSelectedMapEntry(
                series_dt=series_dt, game_id=gid, team_canonical=team,
                adr=float(row["adr"]), kast=float(row["kast"]), kd_balance=float(row["kd_balance"])))


# ---------------------------------------------------------------------------
# Two-phase chronological driver for the ONE order-dependent store (map_elo)
# ---------------------------------------------------------------------------

def process_modern_map_stream(map_state, modern_state, player_roster_state, map_rows, emit_features=True):
    """
    map_rows: the enriched map stream from feature_engineering/maps/modern_map_stream_common.py
    (map_stream_common.load_map_stream's own columns +
    team1_pre_series_elo/team2_pre_series_elo).

    `modern_state` and `player_roster_state` must ALREADY be fully built
    (order-independent - see module docstring) before this is called; this
    driver never mutates them. Only `map_state`
    (map_feature_engine.MapStateStore) is advanced here, via the frozen,
    unmodified `map_feature_engine.apply_map_result`, in the SAME
    series_datetime-batched two-phase discipline every Phase 5/6 engine uses:
    PHASE A emits every eligible map's 25 features from the state as it was
    BEFORE the batch; PHASE B applies the batch's map results only after
    every read above. No mutation happens between Phase A and Phase B, so no
    map can see another map played at or after its own series_datetime.
    """
    from feature_engineering.maps.map_feature_engine import apply_map_result, REQUIRED_MAP_STREAM_COLUMNS
    required = set(REQUIRED_MAP_STREAM_COLUMNS) | {
        "team1_pre_series_elo", "team2_pre_series_elo", "pre_series_elo_known"}
    missing = required - set(map_rows.columns)
    if missing:
        raise ValueError(f"process_modern_map_stream: missing required columns {sorted(missing)}")

    ordered = map_rows.sort_values(["series_datetime", "match_id", "game_id"]).reset_index(drop=True)
    out = []

    for ts, batch in ordered.groupby("series_datetime", sort=True):
        # ---------------- PHASE A: read-only emission ----------------
        if emit_features:
            for _, r in batch.iterrows():
                if not (bool(r["team1_eligible"]) and bool(r["team2_eligible"])):
                    continue
                feats = build_future_modern_map_state_features(
                    map_state, modern_state, player_roster_state,
                    r["team1_canonical"], r["team2_canonical"], r["map_name"], ts,
                    float(r["team1_pre_series_elo"]), float(r["team2_pre_series_elo"]))
                out_row = r.to_dict()
                out_row.update(feats)
                out.append(out_row)

        # ---------------- PHASE B: apply results (only now) ----------------
        for _, r in batch.iterrows():
            apply_map_result(map_state, r)

    return out
