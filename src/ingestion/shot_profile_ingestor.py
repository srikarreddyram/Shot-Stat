"""
Shot Profile Ingestor — shooting splits by shot-difficulty context.

Populates `player_shot_profile` from LeagueDashPlayerPtShot, which returns
every player in the league for one (season, split) combination in a single
call. We iterate the split values we care about:

    dribbles   — 0 / 1 / 2 / 3-6 / 7+ dribbles before the shot
    def_dist   — closest defender 0-2 / 2-4 / 4-6 / 6+ feet away
    touch_time — ball held < 2s / 2-6s / 6+s
    general    — Catch and Shoot / Pull Ups / Less Than 10 ft

Why these matter
----------------
Per-shot defender distance is not exposed by any public endpoint, so contest
level — the single largest driver of whether a shot goes in — is unobservable
at the shot level. The `def_dist` split recovers it as a PLAYER-level trait:
what share of this player's attempts come wide open versus tightly contested.
That share is exactly what separates a self-creator from a spot-up shooter,
and it feeds both the shot-quality model and the attainability model that
decides whether a player could realistically generate a given look.

The `dribbles` and `general` splits serve the complementary purpose: the gap
between a player's Pull Ups FG% and their Catch and Shoot FG% measures how
much efficiency they keep when they have to create the shot themselves.

Availability: 2013-14 onward (tracking era).

Cost: 15 API calls per season (one per split value), each returning the whole
league — far cheaper than any per-player endpoint.

Usage:
    python -m src.ingestion.shot_profile_ingestor                 # test season
    python -m src.ingestion.shot_profile_ingestor --full           # all seasons
    python -m src.ingestion.shot_profile_ingestor --season 2023-24
"""
import sys
import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerptshot
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from src.db.database import get_engine, init_db
from src.db.models import PlayerShotProfile

TRACKING_START_SEASON = "2013-14"

REQUEST_DELAY = 1.2
RETRY_DELAY = 10.0
MAX_RETRIES = 3

# split_type → (endpoint keyword argument, [split values to request])
# The split values are the exact strings the NBA API expects; they are also
# what gets stored in PlayerShotProfile.split_value, so downstream feature
# code can match on them directly.
SPLIT_SPECS = {
    "dribbles": (
        "dribble_range_nullable",
        ["0 Dribbles", "1 Dribble", "2 Dribbles", "3-6 Dribbles", "7+ Dribbles"],
    ),
    "def_dist": (
        "close_def_dist_range_nullable",
        ["0-2 Feet - Very Tight", "2-4 Feet - Tight",
         "4-6 Feet - Open", "6+ Feet - Wide Open"],
    ),
    "touch_time": (
        "touch_time_range_nullable",
        ["Touch < 2 Seconds", "Touch 2-6 Seconds", "Touch 6+ Seconds"],
    ),
    "general": (
        "general_range_nullable",
        ["Catch and Shoot", "Less Than 10 ft"],
    ),
}

# "Pull Ups" is NOT a valid general_range value on LeagueDashPlayerPtShot —
# the endpoint rejects every spelling of it. Pull-up shooting is exposed
# instead as its own LeagueDashPtStats measure type, so we fetch it from
# there and store it under the same split_type="general" umbrella, giving
# feature code one consistent place to read the Catch-and-Shoot vs Pull-Up
# pair from. That pair is the whole point: the gap between them is how much
# efficiency a player retains when he has to create the shot himself.
PULLUP_MEASURE_TYPE = "PullUpShot"
PULLUP_SPLIT_VALUE = "Pull Ups"
PULLUP_COLUMN_MAP = {
    "GP": "gp",
    "PULL_UP_FGM": "fgm",
    "PULL_UP_FGA": "fga",
    "PULL_UP_FG_PCT": "fg_pct",
    "PULL_UP_EFG_PCT": "efg_pct",
    "PULL_UP_FG3M": "fg3m",
    "PULL_UP_FG3A": "fg3a",
    "PULL_UP_FG3_PCT": "fg3_pct",
}

# API column → ORM column
COLUMN_MAP = {
    "GP": "gp",
    "FGA_FREQUENCY": "fga_frequency",
    "FGM": "fgm",
    "FGA": "fga",
    "FG_PCT": "fg_pct",
    "EFG_PCT": "efg_pct",
    "FG3M": "fg3m",
    "FG3A": "fg3a",
    "FG3_PCT": "fg3_pct",
}

_INT_COLS = {"gp", "fgm", "fga", "fg3m", "fg3a"}


def _fetch_split(season: str, kwarg: str, split_value: str) -> pd.DataFrame | None:
    """Fetch one (season, split) combination. Returns None on failure."""
    for attempt in range(MAX_RETRIES):
        try:
            result = leaguedashplayerptshot.LeagueDashPlayerPtShot(
                season=season,
                timeout=45,
                **{kwarg: split_value},
            )
            frames = result.get_data_frames()
            if not frames or frames[0].empty:
                return None
            return frames[0]
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"      ⚠ {split_value} attempt {attempt+1} failed "
                      f"({type(e).__name__}) — retrying in {RETRY_DELAY}s")
                time.sleep(RETRY_DELAY)
            else:
                print(f"      ✗ {split_value} failed after {MAX_RETRIES} "
                      f"attempts: {type(e).__name__}")
                return None
    return None


def _rows_from_frame(df: pd.DataFrame, season: str,
                     split_type: str, split_value: str) -> list[dict]:
    """Convert one endpoint frame into PlayerShotProfile row dicts."""
    rows = []
    for row in df.itertuples(index=False):
        rec = {
            "player_id": str(getattr(row, "PLAYER_ID")),
            "season": season,
            "split_type": split_type,
            "split_value": split_value,
        }
        for api_col, orm_col in COLUMN_MAP.items():
            if api_col not in df.columns:
                rec[orm_col] = None
                continue
            val = getattr(row, api_col, None)
            if pd.isna(val):
                rec[orm_col] = None
            elif orm_col in _INT_COLS:
                rec[orm_col] = int(val)
            else:
                rec[orm_col] = float(val)
        rows.append(rec)
    return rows


def _fetch_pullups(season: str) -> list[dict]:
    """
    Fetch pull-up shooting from LeagueDashPtStats and shape it into
    PlayerShotProfile rows under split_type="general".

    fga_frequency is left NULL here — this endpoint reports pull-up volume in
    absolute terms with no denominator, and back-filling a frequency would
    mean dividing by a total FGA sourced from a different endpoint with
    different minimum-minutes filtering. Feature code derives the pull-up
    *share* from fga against the player's own shot totals instead.
    """
    from nba_api.stats.endpoints import leaguedashptstats

    for attempt in range(MAX_RETRIES):
        try:
            df = leaguedashptstats.LeagueDashPtStats(
                season=season,
                player_or_team="Player",
                pt_measure_type=PULLUP_MEASURE_TYPE,
                per_mode_simple="Totals",  # Totals, so fgm/fga are real counts
                timeout=45,
            ).get_data_frames()[0]
            break
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"      ⚠ pull-ups attempt {attempt+1} failed "
                      f"({type(e).__name__}) — retrying in {RETRY_DELAY}s")
                time.sleep(RETRY_DELAY)
            else:
                print(f"      ✗ pull-ups failed after {MAX_RETRIES} attempts: "
                      f"{type(e).__name__}")
                return []
    else:
        return []

    if df.empty:
        return []

    rows = []
    for row in df.itertuples(index=False):
        rec = {
            "player_id": str(getattr(row, "PLAYER_ID")),
            "season": season,
            "split_type": "general",
            "split_value": PULLUP_SPLIT_VALUE,
            "fga_frequency": None,
        }
        for api_col, orm_col in PULLUP_COLUMN_MAP.items():
            val = getattr(row, api_col, None)
            if val is None or pd.isna(val):
                rec[orm_col] = None
            elif orm_col in _INT_COLS:
                rec[orm_col] = int(val)
            else:
                rec[orm_col] = float(val)
        rows.append(rec)
    return rows


def ingest_season(season: str, engine=None) -> int:
    """Ingest every split for one season. Returns rows written."""
    if engine is None:
        engine = get_engine()

    if season < TRACKING_START_SEASON:
        print(f"  → {season}: before tracking era ({TRACKING_START_SEASON}), skipping")
        return 0

    all_rows: list[dict] = []

    for split_type, (kwarg, split_values) in SPLIT_SPECS.items():
        for split_value in split_values:
            df = _fetch_split(season, kwarg, split_value)
            time.sleep(REQUEST_DELAY)
            if df is None:
                continue
            all_rows.extend(_rows_from_frame(df, season, split_type, split_value))
        print(f"    ✓ {split_type:<11} {len(split_values)} splits")

    # Pull-ups come from a different endpoint (see PULLUP_MEASURE_TYPE).
    pullup_rows = _fetch_pullups(season)
    time.sleep(REQUEST_DELAY)
    if pullup_rows:
        all_rows.extend(pullup_rows)
        print(f"    ✓ {'pull-ups':<11} {len(pullup_rows)} players")

    if not all_rows:
        return 0

    # Upsert in chunks — SQLite caps variables per statement.
    with engine.begin() as conn:
        for i in range(0, len(all_rows), 500):
            chunk = all_rows[i:i + 500]
            stmt = sqlite_insert(PlayerShotProfile).values(chunk)
            update_cols = {
                c.name: stmt.excluded[c.name]
                for c in PlayerShotProfile.__table__.columns
                if c.name not in ("player_id", "season", "split_type", "split_value")
            }
            conn.execute(stmt.on_conflict_do_update(
                index_elements=["player_id", "season", "split_type", "split_value"],
                set_=update_cols,
            ))

    return len(all_rows)


def ingest(seasons: list[str]) -> int:
    """Ingest shot profiles for a list of seasons."""
    engine = get_engine()
    init_db(engine)

    eligible = [s for s in seasons if s >= TRACKING_START_SEASON]
    skipped = [s for s in seasons if s < TRACKING_START_SEASON]

    n_calls = sum(len(v[1]) for v in SPLIT_SPECS.values())
    print(f"\n{'='*60}")
    print("  SHOT PROFILE INGESTOR (openness + self-creation)")
    print(f"  Seasons: {len(eligible)} eligible"
          + (f", {len(skipped)} pre-tracking skipped" if skipped else ""))
    print(f"  ~{n_calls} API calls per season")
    print(f"{'='*60}")

    total = 0
    for season in tqdm(eligible, desc="  Seasons"):
        print(f"\n  → {season}")
        total += ingest_season(season, engine=engine)

    print(f"\n{'='*60}")
    print(f"  SHOT PROFILE INGEST COMPLETE — {total:,} rows written")
    print(f"{'='*60}\n")
    return total


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingest per-player shooting splits by dribbles / defender distance / touch time."
    )
    parser.add_argument("--full", action="store_true",
                        help="All seasons in config.ALL_SEASONS (2013-14 onward)")
    parser.add_argument("--season", type=str, default=None, help="Single season")
    args = parser.parse_args()

    if args.season:
        target = [args.season]
    elif args.full:
        target = config.ALL_SEASONS
    else:
        target = config.TEST_SEASONS

    ingest(target)
