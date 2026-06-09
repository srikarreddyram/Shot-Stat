"""
Central configuration for NBA Shot Quality Engine.
"""
import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "nba_shots.db"
CACHE_DIR = DATA_DIR / "cache" / "bref"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── Database ───────────────────────────────────────────────────────────────
SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_PATH}"

# ── Seasons ────────────────────────────────────────────────────────────────
# Full range: 2010-11 through 2024-25 (15 seasons)
ALL_SEASONS = [
    "2010-11", "2011-12", "2012-13", "2013-14", "2014-15",
    "2015-16", "2016-17", "2017-18", "2018-19", "2019-20",
    "2020-21", "2021-22", "2022-23", "2023-24", "2024-25",
    "2025-26",
]

# For single-season testing
TEST_SEASONS = ["2023-24"]

# ── Rate Limiting (nba_api) ────────────────────────────────────────────────
REQUEST_DELAY = 0.6          # seconds between API calls
MAX_RETRIES = 5              # max retry attempts on failure
BACKOFF_MULTIPLIER = 2.0     # exponential backoff multiplier
INITIAL_BACKOFF = 1.0        # initial backoff in seconds

# ── Basketball Reference ───────────────────────────────────────────────────
BREF_BASE_URL = "https://www.basketball-reference.com"
BREF_REQUEST_DELAY = 3.0     # BR is stricter on rate limits

# ── Wingspan Imputation (inches, by position) ──────────────────────────────
# From PRD §5 — approximate position-level averages
WINGSPAN_DEFAULTS = {
    "PG": 79.0,
    "SG": 81.0,
    "SF": 83.0,
    "PF": 85.0,
    "C":  87.0,
    # Fallbacks for multi-position or unknown
    "G":  80.0,
    "F":  84.0,
    "G-F": 82.0,
    "F-G": 82.0,
    "F-C": 86.0,
    "C-F": 86.0,
}
WINGSPAN_FALLBACK = 82.0  # If position is completely unknown
