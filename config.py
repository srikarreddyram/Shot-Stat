"""
Central configuration for NBA Shot Quality Engine.
"""
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
def season_string(start_year: int) -> str:
    """2023 -> '2023-24'."""
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def current_season(today=None) -> str:
    """
    The season in progress. NBA seasons start in October, so anything from
    October onward belongs to the season named for that calendar year, and
    January through September belongs to the one named for the previous year.

    Used to keep the lists below from needing a hand edit every autumn — the
    old hardcoded ALL_SEASONS silently stopped covering the current year until
    someone remembered to extend it.
    """
    import datetime
    today = today or datetime.date.today()
    return season_string(today.year if today.month >= 10 else today.year - 1)


def seasons_in_database(engine=None) -> list[str]:
    """
    Seasons that actually have shot data, read from the database.

    Prefer this over the static list wherever the caller can accept a database
    dependency: it cannot drift from reality, whereas a literal list can (and
    did) claim seasons that were never ingested.
    """
    from sqlalchemy import text
    if engine is None:
        from src.db.database import get_engine
        engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT DISTINCT season FROM shots ORDER BY season")
        ).fetchall()
    return [r[0] for r in rows]


# Static fallback, used where a database connection is not available (tests,
# CLI defaults). Kept in sync with `current_season()` by extending the range
# rather than by appending literals.
ALL_SEASONS = [
    "2010-11", "2011-12", "2012-13", "2013-14", "2014-15",
    "2015-16", "2016-17", "2017-18", "2018-19", "2019-20",
    "2020-21", "2021-22", "2022-23", "2023-24", "2024-25",
    "2025-26",
]

# Seasons with a roster (via CommonTeamRoster) but no games played yet, so no
# shot/zone/defender stats exist. Kept separate from ALL_SEASONS, which the
# training pipeline treats as seasons with real outcome data to learn from.
CURRENT_SEASONS = [
    "2026-27",
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
