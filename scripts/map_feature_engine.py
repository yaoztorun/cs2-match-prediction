"""
Reusable, dataset-agnostic MAP state / feature engine (Phase 5A).

Mirrors the design of scripts/feature_engine.py (Phase 3) and deliberately
REUSES its primitives rather than reimplementing them:
`elo_expected`, `elo_update`, `ELO_INITIAL`, `ELO_K`, `_beta_smoothed_win_rate`.
Map ELO therefore uses exactly the same rating mathematics and the same fixed
K-factor as series ELO. No K tuning happens in Phase 5A.

THE TWO TIMESTAMP CONCEPTS (do not conflate)
--------------------------------------------
`series_datetime` : the PREDICTION CUTOFF. Everything a model may know about
                    a series - including every map that will be played in it -
                    is frozen at this instant.
`map_datetime`    : when an individual map was actually played. It may be
                    later than `series_datetime`, and richer future data
                    sources may expose real per-map start times.

The engine keys ALL batching and all feature emission on `series_datetime`.
`map_datetime` is carried for provenance only and is NEVER used to decide what
a map may see. This is what makes the engine correct for the real application
task: when a BO3's maps are known before the series starts, every one of those
maps must be predicted from the same pre-series knowledge state.

SERIES-ATOMIC, TWO-PHASE PROCESSING
-----------------------------------
A series (match_id) is an ATOMIC prediction/update unit:

    for each exact series_datetime batch:
        PHASE A (read):  emit features for ALL maps of ALL series in the batch
                         using the state as it was BEFORE the batch
        PHASE B (write): only then apply every completed map result

State is never updated between Map 1 / Map 2 / Map 3 of the same series, and
never between two series sharing a `series_datetime`. Correctness does NOT
depend on the current Kaggle property that a series' map rows happen to share
one timestamp - maps may carry differing `map_datetime` values and still
receive identical pre-series state.

STATE-HISTORY ELIGIBILITY vs TRAINING ELIGIBILITY (not the same thing)
----------------------------------------------------------------------
Following the Phase 3 identity policy exactly:
  * A supervised TRAINING ROW requires BOTH canonical identities trusted.
  * An eligible team's OWN independent map history (win/loss, margins,
    recency, pool membership) may still update from a map played against an
    identity-ineligible opponent - that team's own result is a real fact -
    but the entry is flagged `opponent_identity_trusted=False` so future
    opponent-dependent features can exclude it.
  * MAP ELO is pair-dependent and updates ONLY when both identities are
    trusted.
This is why the pre-Cologne state snapshot is built from the canonical map
stream + identity policy, and NOT by replaying map_features_v1 training rows.
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import timedelta

import numpy as np
import pandas as pd

from feature_engine import (
    ELO_INITIAL, ELO_K, elo_expected, elo_update, _beta_smoothed_win_rate,
)

MAP_ENGINE_VERSION = "1.0.0"

MAP_POOL_LOOKBACK_DAYS = 180          # feature-engineering constant, NOT a tuned hyperparameter
ROLLING_WINDOWS = (5, 10)
EXPERIENCED_MAP_MIN_MATCHES = 5
UNKNOWN_MAP_CATEGORY = "__UNKNOWN_MAP__"

# Cold-start contract (documented once, applied everywhere)
COLD_START_MAP_ELO = ELO_INITIAL      # 1500.0
COLD_START_SMOOTHED_WR = 0.5
COLD_START_WIN_RATE = 0.5
COLD_START_MARGIN = 0.0
COLD_START_MATCHES = 0
COLD_START_DAYS_SINCE = float("nan")  # explicit missing - never a fake sentinel

REQUIRED_MAP_STREAM_COLUMNS = [
    "series_datetime", "map_datetime", "match_id", "game_id", "map_name",
    "team1_canonical", "team2_canonical", "score1_game", "score2_game",
    "team1_map_win", "team1_eligible", "team2_eligible",
]


def normalized_round_margin(score_for, score_against):
    """(for - against) / (for + against); 0.0 when the denominator is 0.
    Scale-normalized so historical CS scoring-format changes (MR15 -> MR12)
    do not distort the signal. 13-10 -> 3/23, 13-5 -> 8/18."""
    total = float(score_for) + float(score_against)
    if total <= 0:
        return 0.0
    return (float(score_for) - float(score_against)) / total


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class MapHistoryEntry:
    series_dt: pd.Timestamp          # PREDICTION CUTOFF of the containing series
    map_dt: pd.Timestamp             # when the map itself was played (provenance only)
    match_id: str
    game_id: str
    map_name: str
    opponent_canonical: str
    opponent_identity_trusted: bool
    win: int
    score_for: int
    score_against: int
    normalized_margin: float

    def to_dict(self):
        d = asdict(self)
        d["series_dt"] = self.series_dt.isoformat()
        d["map_dt"] = self.map_dt.isoformat()
        return d

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        d["series_dt"] = pd.Timestamp(d["series_dt"])
        d["map_dt"] = pd.Timestamp(d["map_dt"])
        return cls(**d)


@dataclass
class TeamMapState:
    canonical_name: str
    map_name: str
    map_elo: float = COLD_START_MAP_ELO
    history: list = field(default_factory=list)   # list[MapHistoryEntry]

    def to_dict(self):
        return {"canonical_name": self.canonical_name, "map_name": self.map_name,
                "map_elo": self.map_elo, "history": [h.to_dict() for h in self.history]}

    @classmethod
    def from_dict(cls, d):
        return cls(canonical_name=d["canonical_name"], map_name=d["map_name"],
                    map_elo=d["map_elo"],
                    history=[MapHistoryEntry.from_dict(h) for h in d["history"]])


class MapStateStore:
    """(team_canonical, map_name) -> TeamMapState. Entries are created lazily
    only when a real completed map is applied; feature computation never
    creates or mutates state."""

    def __init__(self):
        self.states: dict[tuple, TeamMapState] = {}
        self.processed_map_uids: set[str] = set()

    def get(self, team, map_name):
        return self.states.get((team, map_name))

    def get_or_create(self, team, map_name):
        key = (team, map_name)
        if key not in self.states:
            self.states[key] = TeamMapState(canonical_name=team, map_name=map_name)
        return self.states[key]

    def maps_for_team(self, team):
        return [m for (t, m) in self.states if t == team]

    def teams(self):
        return sorted({t for (t, _m) in self.states})

    def to_json(self, path, meta=None):
        payload = {
            "map_engine_version": MAP_ENGINE_VERSION,
            "n_team_map_states": len(self.states),
            "n_maps_processed": len(self.processed_map_uids),
            "meta": meta or {},
            "states": [s.to_dict() for s in self.states.values()],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)

    @classmethod
    def from_json(cls, path):
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        store = cls()
        for sd in payload["states"]:
            st = TeamMapState.from_dict(sd)
            store.states[(st.canonical_name, st.map_name)] = st
            for h in st.history:
                store.processed_map_uids.add(f"{h.match_id}:{h.game_id}")
        store.loaded_meta = payload.get("meta", {})
        return store

    def snapshot_summary_df(self):
        """Flat, scalar-only, one row per (team, map) - fastparquet-safe.
        Full per-map history lives only in the JSON snapshot."""
        rows = []
        for st in self.states.values():
            hist = sorted(st.history, key=lambda h: h.series_dt)
            n = len(hist)
            last5, last10 = hist[-5:], hist[-10:]
            rows.append({
                "canonical_team_name": st.canonical_name,
                "map_name": st.map_name,
                "map_elo": st.map_elo,
                "map_matches": n,
                "map_wins": sum(h.win for h in hist),
                "map_losses": n - sum(h.win for h in hist),
                "map_smoothed_win_rate": _beta_smoothed_win_rate(hist),
                "map_win_rate_last_5": (sum(h.win for h in last5) / len(last5)) if last5 else COLD_START_WIN_RATE,
                "map_win_rate_last_10": (sum(h.win for h in last10) / len(last10)) if last10 else COLD_START_WIN_RATE,
                "map_normalized_margin_all": float(np.mean([h.normalized_margin for h in hist])) if hist else COLD_START_MARGIN,
                "map_normalized_margin_last_5": float(np.mean([h.normalized_margin for h in last5])) if last5 else COLD_START_MARGIN,
                "map_normalized_margin_last_10": float(np.mean([h.normalized_margin for h in last10])) if last10 else COLD_START_MARGIN,
                "last_map_series_datetime": hist[-1].series_dt if hist else pd.NaT,
                "n_entries_with_untrusted_opponent": sum(1 for h in hist if not h.opponent_identity_trusted),
            })
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Strictly-prior per-(team, map) statistics
# ---------------------------------------------------------------------------

def _history_before(state, as_of):
    """Entries strictly before the prediction cutoff. Filtering is on
    `series_dt` (the cutoff), never `map_dt`."""
    if state is None:
        return []
    as_of = pd.Timestamp(as_of)
    return sorted([h for h in state.history if h.series_dt < as_of], key=lambda h: h.series_dt)


def compute_team_map_features(state, as_of):
    """One team's strictly-prior stats on ONE map, as of the prediction cutoff.
    `state` may be None (never played this map) -> documented cold start."""
    as_of = pd.Timestamp(as_of)
    hist = _history_before(state, as_of)
    n = len(hist)
    last5, last10 = hist[-5:], hist[-10:]

    if n > 0:
        days_since = (as_of - hist[-1].series_dt).total_seconds() / 86400.0
    else:
        days_since = COLD_START_DAYS_SINCE

    return {
        "map_elo": state.map_elo if state is not None else COLD_START_MAP_ELO,
        "map_smoothed_win_rate": _beta_smoothed_win_rate(hist),
        "map_win_rate_last_5": (sum(h.win for h in last5) / len(last5)) if last5 else COLD_START_WIN_RATE,
        "map_win_rate_last_10": (sum(h.win for h in last10) / len(last10)) if last10 else COLD_START_WIN_RATE,
        "map_normalized_margin_all": float(np.mean([h.normalized_margin for h in hist])) if hist else COLD_START_MARGIN,
        "map_normalized_margin_last_5": float(np.mean([h.normalized_margin for h in last5])) if last5 else COLD_START_MARGIN,
        "map_normalized_margin_last_10": float(np.mean([h.normalized_margin for h in last10])) if last10 else COLD_START_MARGIN,
        "map_matches_before": n,
        "days_since_map_played": days_since,
    }


def recent_map_pool(store, team, as_of, lookback_days=MAP_POOL_LOOKBACK_DAYS):
    """Maps the team played in [as_of - lookback_days, as_of).
    Half-open by construction: a map exactly at `as_of` is excluded (it is the
    current series), a map exactly at the lower bound is included."""
    as_of = pd.Timestamp(as_of)
    lower = as_of - timedelta(days=lookback_days)
    pool = set()
    for m in store.maps_for_team(team):
        st = store.get(team, m)
        for h in st.history:
            if lower <= h.series_dt < as_of:
                pool.add(m)
                break
    return pool


# ---------------------------------------------------------------------------
# MAP-SPECIFIC matchup features (task: known map)
# ---------------------------------------------------------------------------

MAP_DIRECTIONAL_FEATURES = [
    "map_elo_diff", "map_smoothed_win_rate_diff", "map_win_rate_last_5_diff",
    "map_win_rate_last_10_diff", "map_normalized_margin_all_diff",
    "map_normalized_margin_last_5_diff", "map_normalized_margin_last_10_diff",
    "map_matches_before_diff", "days_since_map_played_diff",
]
MAP_SYMMETRIC_FEATURES = [
    "map_matches_before_min", "map_matches_before_sum", "both_teams_have_map_history",
    "both_teams_have_5_map_matches", "both_teams_have_10_map_matches",
]


def build_future_map_matchup_features(store, team1, team2, best_of, map_name, as_of_datetime, tier=None):
    """PURE / READ-ONLY. Needs no target and no current-map score. Produces the
    exact schema used for training, so future inference cannot diverge."""
    as_of = pd.Timestamp(as_of_datetime)
    f1 = compute_team_map_features(store.get(team1, map_name), as_of)
    f2 = compute_team_map_features(store.get(team2, map_name), as_of)

    out = {}
    for k, v in f1.items():
        out[f"team1_{k}"] = v
    for k, v in f2.items():
        out[f"team2_{k}"] = v

    out["map_elo_diff"] = f1["map_elo"] - f2["map_elo"]
    out["map_smoothed_win_rate_diff"] = f1["map_smoothed_win_rate"] - f2["map_smoothed_win_rate"]
    out["map_win_rate_last_5_diff"] = f1["map_win_rate_last_5"] - f2["map_win_rate_last_5"]
    out["map_win_rate_last_10_diff"] = f1["map_win_rate_last_10"] - f2["map_win_rate_last_10"]
    out["map_normalized_margin_all_diff"] = f1["map_normalized_margin_all"] - f2["map_normalized_margin_all"]
    out["map_normalized_margin_last_5_diff"] = f1["map_normalized_margin_last_5"] - f2["map_normalized_margin_last_5"]
    out["map_normalized_margin_last_10_diff"] = f1["map_normalized_margin_last_10"] - f2["map_normalized_margin_last_10"]
    out["map_matches_before_diff"] = f1["map_matches_before"] - f2["map_matches_before"]
    out["days_since_map_played_diff"] = f1["days_since_map_played"] - f2["days_since_map_played"]

    n1, n2 = f1["map_matches_before"], f2["map_matches_before"]
    out["map_matches_before_min"] = min(n1, n2)
    out["map_matches_before_sum"] = n1 + n2
    out["both_teams_have_map_history"] = int(n1 > 0 and n2 > 0)
    out["both_teams_have_5_map_matches"] = int(n1 >= 5 and n2 >= 5)
    out["both_teams_have_10_map_matches"] = int(n1 >= 10 and n2 >= 10)

    out["map_name"] = map_name
    out["bestOf"] = best_of
    out["tier"] = tier
    return out


# ---------------------------------------------------------------------------
# PRE-VETO series map-pool features (task: pre-veto series)
# ---------------------------------------------------------------------------
# Two DISTINCT feature families - never describe them interchangeably:
#
#  (A) POOL-DEPTH / ORDER-STATISTIC features
#      "team1's best map strength vs team2's best map strength". The two sides'
#      k-th-best entries may refer to DIFFERENT map identities. These describe
#      the shape/depth of each team's own pool.
#
#  (B) SAME-MAP MATCHUP features
#      computed across union(pool1, pool2); for each map identity the two teams
#      are compared ON THAT SAME MAP (cold-start defaults where a side lacks
#      history), then summarized.

SERIES_MAP_POOL_DIRECTIONAL = [
    "map_pool_size_diff", "map_pool_total_matches_diff", "map_pool_experienced_maps_diff",
    "map_pool_mean_elo_diff", "map_pool_best_elo_diff", "map_pool_second_best_elo_diff",
    "map_pool_third_best_elo_diff", "map_pool_worst_elo_diff",
    "map_pool_mean_smoothed_wr_diff", "map_pool_best_smoothed_wr_diff",
    "map_pool_second_best_smoothed_wr_diff", "map_pool_third_best_smoothed_wr_diff",
    "map_pool_worst_smoothed_wr_diff", "map_pool_mean_normalized_margin_diff",
    "map_matchup_mean_elo_advantage", "map_matchup_median_elo_advantage",
    "map_matchup_midrange_elo_advantage", "map_matchup_positive_advantage_balance",
    "map_matchup_mean_smoothed_wr_advantage", "map_matchup_median_smoothed_wr_advantage",
]
SERIES_MAP_POOL_SYMMETRIC = [
    "map_pool_size_min", "map_pool_total_matches_min", "both_teams_have_map_pool_history",
    "both_teams_have_3_recent_maps", "both_teams_have_5_experienced_maps", "union_map_count",
    "shared_recent_map_count", "shared_experienced_map_count", "map_matchup_shared_coverage",
    "map_matchup_elo_advantage_range",
]

# WHY THE (B) FAMILY USES mean / median / midrange AND NOT k-TH-BEST
# ------------------------------------------------------------------
# Mirrored training augmentation requires every directional feature to negate
# exactly under a Team1<->Team2 swap. Swapping negates the per-map advantage
# list, and the k-th LARGEST of a negated list is minus the k-th SMALLEST of
# the original - so "2nd best advantage" maps onto "2nd worst advantage", not
# onto itself. Such a feature is antisymmetric only as a PAIR, which the
# mirroring contract cannot express. mean, median and midrange ((max+min)/2)
# each satisfy f(-x) = -f(x) individually, and the spread information that the
# order statistics carried is preserved by the swap-INVARIANT range (max-min),
# which is therefore declared symmetric.
#
# The (A) pool-depth family keeps its k-th-best order statistics because each
# team's statistic is computed independently over its OWN pool and only then
# subtracted: (p1.kth - p2.kth) negates cleanly under a swap.


def _pool_profile(store, team, pool, as_of):
    """(A) POOL-DEPTH profile for one team over its OWN recent pool."""
    elos, wrs, margins, total_matches, experienced = [], [], [], 0, 0
    for m in sorted(pool):
        f = compute_team_map_features(store.get(team, m), as_of)
        elos.append(f["map_elo"])
        wrs.append(f["map_smoothed_win_rate"])
        margins.append(f["map_normalized_margin_all"])
        total_matches += f["map_matches_before"]
        if f["map_matches_before"] >= EXPERIENCED_MAP_MIN_MATCHES:
            experienced += 1

    def order_stat(values, k, neutral):
        """k-th best (0-based) descending; documented NEUTRAL cold-start
        fallback when the pool is too shallow - never an extreme sentinel."""
        s = sorted(values, reverse=True)
        return s[k] if len(s) > k else neutral

    return {
        "size": len(pool),
        "total_matches": total_matches,
        "experienced_maps": experienced,
        "mean_elo": float(np.mean(elos)) if elos else COLD_START_MAP_ELO,
        "best_elo": order_stat(elos, 0, COLD_START_MAP_ELO),
        "second_best_elo": order_stat(elos, 1, COLD_START_MAP_ELO),
        "third_best_elo": order_stat(elos, 2, COLD_START_MAP_ELO),
        "worst_elo": (min(elos) if elos else COLD_START_MAP_ELO),
        "mean_wr": float(np.mean(wrs)) if wrs else COLD_START_SMOOTHED_WR,
        "best_wr": order_stat(wrs, 0, COLD_START_SMOOTHED_WR),
        "second_best_wr": order_stat(wrs, 1, COLD_START_SMOOTHED_WR),
        "third_best_wr": order_stat(wrs, 2, COLD_START_SMOOTHED_WR),
        "worst_wr": (min(wrs) if wrs else COLD_START_SMOOTHED_WR),
        "mean_margin": float(np.mean(margins)) if margins else COLD_START_MARGIN,
    }


def build_future_series_map_pool_features(store, team1, team2, best_of, as_of_datetime, tier=None,
                                           lookback_days=MAP_POOL_LOOKBACK_DAYS):
    """PURE / READ-ONLY PRE-VETO features. Uses ONLY historical map evidence -
    never the maps selected for the target match. Needs no target."""
    as_of = pd.Timestamp(as_of_datetime)
    pool1 = recent_map_pool(store, team1, as_of, lookback_days)
    pool2 = recent_map_pool(store, team2, as_of, lookback_days)

    p1 = _pool_profile(store, team1, pool1, as_of)
    p2 = _pool_profile(store, team2, pool2, as_of)

    out = {}
    # ---- (A) pool-depth / order-statistic diffs (may compare different maps) ----
    out["map_pool_size_diff"] = p1["size"] - p2["size"]
    out["map_pool_total_matches_diff"] = p1["total_matches"] - p2["total_matches"]
    out["map_pool_experienced_maps_diff"] = p1["experienced_maps"] - p2["experienced_maps"]
    for a, b in [("mean_elo", "mean_elo"), ("best_elo", "best_elo"),
                  ("second_best_elo", "second_best_elo"), ("third_best_elo", "third_best_elo"),
                  ("worst_elo", "worst_elo")]:
        out[f"map_pool_{a}_diff"] = p1[a] - p2[b]
    out["map_pool_mean_smoothed_wr_diff"] = p1["mean_wr"] - p2["mean_wr"]
    out["map_pool_best_smoothed_wr_diff"] = p1["best_wr"] - p2["best_wr"]
    out["map_pool_second_best_smoothed_wr_diff"] = p1["second_best_wr"] - p2["second_best_wr"]
    out["map_pool_third_best_smoothed_wr_diff"] = p1["third_best_wr"] - p2["third_best_wr"]
    out["map_pool_worst_smoothed_wr_diff"] = p1["worst_wr"] - p2["worst_wr"]
    out["map_pool_mean_normalized_margin_diff"] = p1["mean_margin"] - p2["mean_margin"]

    # ---- (B) same-map matchup advantages across union(pool1, pool2) ----
    union = sorted(pool1 | pool2)
    elo_adv, wr_adv = [], []
    pos = neg = 0
    for m in union:
        f1 = compute_team_map_features(store.get(team1, m), as_of)
        f2 = compute_team_map_features(store.get(team2, m), as_of)
        d_elo = f1["map_elo"] - f2["map_elo"]
        elo_adv.append(d_elo)
        wr_adv.append(f1["map_smoothed_win_rate"] - f2["map_smoothed_win_rate"])
        if d_elo > 0:
            pos += 1
        elif d_elo < 0:
            neg += 1

    # Empty union -> neutral 0.0 advantage everywhere (documented cold start).
    out["map_matchup_mean_elo_advantage"] = float(np.mean(elo_adv)) if elo_adv else 0.0
    out["map_matchup_median_elo_advantage"] = float(np.median(elo_adv)) if elo_adv else 0.0
    out["map_matchup_midrange_elo_advantage"] = ((max(elo_adv) + min(elo_adv)) / 2.0) if elo_adv else 0.0
    out["map_matchup_positive_advantage_balance"] = pos - neg
    out["map_matchup_mean_smoothed_wr_advantage"] = float(np.mean(wr_adv)) if wr_adv else 0.0
    out["map_matchup_median_smoothed_wr_advantage"] = float(np.median(wr_adv)) if wr_adv else 0.0
    # swap-INVARIANT spread (see the note beside SERIES_MAP_POOL_SYMMETRIC)
    out["map_matchup_elo_advantage_range"] = (max(elo_adv) - min(elo_adv)) if elo_adv else 0.0

    # ---- symmetric confidence / evidence features ----
    shared = pool1 & pool2
    exp1 = {m for m in pool1
            if compute_team_map_features(store.get(team1, m), as_of)["map_matches_before"] >= EXPERIENCED_MAP_MIN_MATCHES}
    exp2 = {m for m in pool2
            if compute_team_map_features(store.get(team2, m), as_of)["map_matches_before"] >= EXPERIENCED_MAP_MIN_MATCHES}

    out["map_pool_size_min"] = min(p1["size"], p2["size"])
    out["map_pool_total_matches_min"] = min(p1["total_matches"], p2["total_matches"])
    out["both_teams_have_map_pool_history"] = int(p1["size"] > 0 and p2["size"] > 0)
    out["both_teams_have_3_recent_maps"] = int(p1["size"] >= 3 and p2["size"] >= 3)
    out["both_teams_have_5_experienced_maps"] = int(p1["experienced_maps"] >= 5 and p2["experienced_maps"] >= 5)
    out["union_map_count"] = len(union)
    out["shared_recent_map_count"] = len(shared)
    out["shared_experienced_map_count"] = len(exp1 & exp2)
    # Jaccard overlap; 0.0 when the union is empty (documented neutral handling)
    out["map_matchup_shared_coverage"] = (len(shared) / len(union)) if union else 0.0

    out["bestOf"] = best_of
    out["tier"] = tier
    return out


# ---------------------------------------------------------------------------
# State updates
# ---------------------------------------------------------------------------

def apply_map_result(store, row):
    """Apply ONE completed map to the store. Must only be called in Phase B.

    Identity policy (Phase 3 rules, restated):
      * each eligible team's OWN history updates regardless of the opponent's
        eligibility (its own result is a real fact), flagged with
        `opponent_identity_trusted`;
      * MAP ELO is pair-dependent and updates ONLY if both are eligible.
    """
    t1, t2 = row["team1_canonical"], row["team2_canonical"]
    e1, e2 = bool(row["team1_eligible"]), bool(row["team2_eligible"])
    map_name = row["map_name"]
    series_dt = pd.Timestamp(row["series_datetime"])
    map_dt = pd.Timestamp(row["map_datetime"])
    s1, s2 = int(row["score1_game"]), int(row["score2_game"])
    win1 = int(row["team1_map_win"])

    uid = f"{row['match_id']}:{row['game_id']}"

    if e1:
        st = store.get_or_create(t1, map_name)
        st.history.append(MapHistoryEntry(
            series_dt=series_dt, map_dt=map_dt, match_id=str(row["match_id"]),
            game_id=str(row["game_id"]), map_name=map_name, opponent_canonical=t2,
            opponent_identity_trusted=e2, win=win1, score_for=s1, score_against=s2,
            normalized_margin=normalized_round_margin(s1, s2)))
    if e2:
        st = store.get_or_create(t2, map_name)
        st.history.append(MapHistoryEntry(
            series_dt=series_dt, map_dt=map_dt, match_id=str(row["match_id"]),
            game_id=str(row["game_id"]), map_name=map_name, opponent_canonical=t1,
            opponent_identity_trusted=e1, win=1 - win1, score_for=s2, score_against=s1,
            normalized_margin=normalized_round_margin(s2, s1)))
    if e1 and e2:
        a = store.get_or_create(t1, map_name)
        b = store.get_or_create(t2, map_name)
        a.map_elo, b.map_elo = elo_update(a.map_elo, b.map_elo, float(win1))

    store.processed_map_uids.add(uid)


# ---------------------------------------------------------------------------
# Series-atomic, two-phase chronological driver
# ---------------------------------------------------------------------------

def process_map_stream(store, rows, emit_map_features=True, emit_series_features=False,
                        lookback_days=MAP_POOL_LOOKBACK_DAYS):
    """
    rows: DataFrame with at least REQUIRED_MAP_STREAM_COLUMNS. Extra columns are
    carried through into the emitted dicts unchanged.

    Batching is on `series_datetime` (the prediction cutoff), and a series is
    atomic: every map of every series in a batch is scored from the SAME
    pre-batch state, and no result is applied until the whole batch is read.
    Deliberately independent of `map_datetime`, which may differ per map.

    Returns (map_feature_rows, series_feature_rows).
    """
    missing = set(REQUIRED_MAP_STREAM_COLUMNS) - set(rows.columns)
    if missing:
        raise ValueError(f"process_map_stream: missing required columns {sorted(missing)}")

    ordered = rows.sort_values(["series_datetime", "match_id", "game_id"]).reset_index(drop=True)
    map_out, series_out = [], []

    for series_dt, batch in ordered.groupby("series_datetime", sort=True):
        # ---------------- PHASE A: read-only emission ----------------
        if emit_series_features:
            for match_id, mgrp in batch.groupby("match_id", sort=True):
                head = mgrp.iloc[0]
                if not (bool(head["team1_eligible"]) and bool(head["team2_eligible"])):
                    continue
                feats = build_future_series_map_pool_features(
                    store, head["team1_canonical"], head["team2_canonical"],
                    head.get("bestOf"), series_dt, head.get("tier"), lookback_days)
                row = {"match_id": match_id, "series_datetime": series_dt, **feats}
                series_out.append(row)

        if emit_map_features:
            for _, r in batch.iterrows():
                if not (bool(r["team1_eligible"]) and bool(r["team2_eligible"])):
                    continue   # supervised training row requires BOTH identities trusted
                feats = build_future_map_matchup_features(
                    store, r["team1_canonical"], r["team2_canonical"],
                    r.get("bestOf"), r["map_name"], series_dt, r.get("tier"))
                out_row = r.to_dict()
                out_row.update(feats)
                map_out.append(out_row)

        # ---------------- PHASE B: apply results (only now) ----------------
        for _, r in batch.iterrows():
            apply_map_result(store, r)

    return map_out, series_out


SERIES_REQUEST_COLUMNS = [
    "match_id", "series_datetime", "team1_canonical", "team2_canonical",
    "team1_eligible", "team2_eligible",
]


def process_combined_stream(store, map_rows, series_requests=None,
                             emit_map_features=True, lookback_days=MAP_POOL_LOOKBACK_DAYS):
    """Same series-atomic two-phase contract as `process_map_stream`, but the
    set of series needing PRE-VETO features is supplied SEPARATELY.

    Why this exists: a series that has no map rows of its own still needs
    pre-veto map-pool features, because those features describe the two teams'
    PRIOR map history, not the maps of the target match. Driving series emission
    off the map stream alone would silently drop every such match. Conversely a
    series request never contributes state - only completed maps do.

    Both event kinds are batched on the same `series_datetime` axis, so a series
    request at time T sees exactly the state that a map at time T sees: nothing
    from T or later.
    """
    missing = set(REQUIRED_MAP_STREAM_COLUMNS) - set(map_rows.columns)
    if missing:
        raise ValueError(f"process_combined_stream: map_rows missing {sorted(missing)}")

    maps = map_rows.sort_values(["series_datetime", "match_id", "game_id"]).reset_index(drop=True)

    if series_requests is None:
        reqs = pd.DataFrame(columns=SERIES_REQUEST_COLUMNS)
    else:
        rmissing = set(SERIES_REQUEST_COLUMNS) - set(series_requests.columns)
        if rmissing:
            raise ValueError(f"process_combined_stream: series_requests missing {sorted(rmissing)}")
        reqs = series_requests.sort_values(["series_datetime", "match_id"]).reset_index(drop=True)

    timestamps = sorted(set(maps["series_datetime"]) | set(reqs["series_datetime"]))
    map_by_ts = {ts: g for ts, g in maps.groupby("series_datetime", sort=False)}
    req_by_ts = {ts: g for ts, g in reqs.groupby("series_datetime", sort=False)}

    map_out, series_out = [], []

    for ts in timestamps:
        # ---------------- PHASE A: read-only emission ----------------
        for _, q in req_by_ts.get(ts, pd.DataFrame(columns=SERIES_REQUEST_COLUMNS)).iterrows():
            if not (bool(q["team1_eligible"]) and bool(q["team2_eligible"])):
                continue
            feats = build_future_series_map_pool_features(
                store, q["team1_canonical"], q["team2_canonical"],
                q.get("bestOf"), ts, q.get("tier"), lookback_days)
            series_out.append({"match_id": q["match_id"], "series_datetime": ts, **feats})

        if emit_map_features and ts in map_by_ts:
            for _, r in map_by_ts[ts].iterrows():
                if not (bool(r["team1_eligible"]) and bool(r["team2_eligible"])):
                    continue
                feats = build_future_map_matchup_features(
                    store, r["team1_canonical"], r["team2_canonical"],
                    r.get("bestOf"), r["map_name"], ts, r.get("tier"))
                out_row = r.to_dict()
                out_row.update(feats)
                map_out.append(out_row)

        # ---------------- PHASE B: apply results (only now) ----------------
        if ts in map_by_ts:
            for _, r in map_by_ts[ts].iterrows():
                apply_map_result(store, r)

    return map_out, series_out
