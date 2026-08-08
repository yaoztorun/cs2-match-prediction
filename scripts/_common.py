"""Shared helpers for Phase 2 scripts. Read-only against data/raw/."""

import hashlib
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
REPORTS = ROOT / "reports"
INTERIM.mkdir(exist_ok=True, parents=True)
REPORTS.mkdir(exist_ok=True)

GAMES_FILES = {
    "tier1": "cs2_tier1_games.csv",
    "tier2": "cs2_tier2_games.csv",
    "tier3": "cs2_tier3_games.csv",
}

PLAYER_ID_COLS = [f"team{t}_player{i}_id" for t in (1, 2) for i in range(1, 6)]
PLAYER_NAME_COLS = [f"team{t}_player{i}" for t in (1, 2) for i in range(1, 6)]
PLAYER_STAT_SUFFIXES = ["kills", "deaths", "assists", "adr", "kast", "kddiff"]
PLAYER_STAT_COLS = [
    f"team{t}_player{i}_{s}"
    for t in (1, 2)
    for i in range(1, 6)
    for s in PLAYER_STAT_SUFFIXES
]


def load_games_tiered():
    """Load tier1/2/3 raw CSVs with an explicit `tier` column, all-string dtype."""
    frames = []
    for tier, fname in GAMES_FILES.items():
        df = pd.read_csv(RAW / fname, dtype=str, keep_default_na=False)
        df["tier"] = tier
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def to_num(s):
    import numpy as np
    return pd.to_numeric(s.replace("", pd.NA), errors="coerce")


def raw_file_hashes():
    """sha256 of every file under data/raw/, for before/after integrity checks."""
    hashes = {}
    for p in sorted(RAW.rglob("*")):
        if p.is_file():
            hashes[str(p.relative_to(RAW))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return hashes
