"""
Player Tracking Ingestor — Creation Skill (handle + passing).

Populates `player_tracking_stats` from the NBA API's LeagueDashPtStats
endpoint, which exposes SportVU/Second Spectrum camera tracking aggregates.

Three measure types are merged into one row per (player_id, season):

    Possessions → touches, time of possession, AVG_DRIB_PER_TOUCH
                  ("how much does this player pound the ball")
    Drives      → drives, drive FG%, drive pass/assist/turnover rates
                  ("how much rim pressure does the handle actually create")
    Passing     → potential assists, assist points created, AST:pass ratio
                  ("how much does the defense have to respect the pass")

Availability: 2013-14 onward. Tracking cameras were installed league-wide for
the 2013-14 season; earlier seasons have no data and are skipped rather than
imputed, consistent with this project's "exact data or NULL" policy.

Cost: 3 API calls per season (one per measure type) — the endpoint returns
every player in the league at once, so this is cheap compared to the
per-player ingestors.

Usage:
    python -m src.ingestion.tracking_ingestor                 # single test season
    python -m src.ingestion.tracking_ingestor --full           # all tracking seasons
    python -m src.ingestion.tracking_ingestor --season 2023-24
"""
import sys
import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import leaguedashptstats
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from src.db.database import get_engine, init_db
from src.db.models import PlayerTrackingStats

# ── Configuration ──────────────────────────────────────────────────────────
# SportVU tracking begins in 2013-14. Anything earlier returns empty frames.
TRACKING_START_SEASON = "2013-14"

REQUEST_DELAY = 1.2   # seconds between API calls
RETRY_DELAY = 10.0    # seconds after a rate-limit failure
MAX_RETRIES = 3

# Map each measure type's API columns → our ORM column names. Only columns
# listed here are read; anything else the endpoint returns is ignored, so a
# new upstream column can't silently change what we store.
MEASURE_COLUMN_MAP = {
    "Possessions": {
        "TOUCHES": "touches",
        "FRONT_CT_TOUCHES": "front_ct_touches",
        "TIME_OF_POSS": "time_of_poss",
        "AVG_SEC_PER_TOUCH": "avg_sec_per_touch",
        "AVG_DRIB_PER_TOUCH": "avg_drib_per_touch",
        "PTS_PER_TOUCH": "pts_per_touch",
        "ELBOW_TOUCHES": "elbow_touches",
        "POST_TOUCHES": "post_touches",
        "PAINT_TOUCHES": "paint_touches",
    },
    "Drives": {
        "DRIVES": "drives",
        "DRIVE_FG_PCT": "drive_fg_pct",
        "DRIVE_PTS": "drive_pts",
        "DRIVE_PASSES_PCT": "drive_passes_pct",
        "DRIVE_AST_PCT": "drive_ast_pct",
        "DRIVE_TOV_PCT": "drive_tov_pct",
        "DRIVE_PF_PCT": "drive_pf_pct",
    },
    "Passing": {
        "PASSES_MADE": "passes_made",
        "PASSES_RECEIVED": "passes_received",
        "AST": "ast",
        "SECONDARY_AST": "secondary_ast",
        "POTENTIAL_AST": "potential_ast",
        "AST_POINTS_CREATED": "ast_points_created",
        "AST_TO_PASS_PCT": "ast_to_pass_pct",
        "AST_TO_PASS_PCT_ADJ": "ast_to_pass_pct_adj",
    },
}


def _fetch_measure(season: str, measure_type: str) -> pd.DataFrame | None:
    """
    Fetch one LeagueDashPtStats measure type for a season.
    Returns None on unrecoverable error (caller skips that measure).
    """
    for attempt in range(MAX_RETRIES):
        try:
            result = leaguedashptstats.LeagueDashPtStats(
                season=season,
                player_or_team="Player",
                pt_measure_type=measure_type,
                per_mode_simple="PerGame",
                timeout=45,
            )
            frames = result.get_data_frames()
            if not frames or frames[0].empty:
                return None
            return frames[0]
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"    ⚠ {measure_type} {season} attempt {attempt+1} failed "
                      f"({type(e).__name__}) — retrying in {RETRY_DELAY}s")
                time.sleep(RETRY_DELAY)
            else:
                print(f"    ✗ {measure_type} {season} failed after "
                      f"{MAX_RETRIES} attempts: {type(e).__name__}: {e}")
                return None
    return None


def ingest_season(season: str, engine=None) -> int:
    """
    Ingest all three tracking measure types for one season and upsert them as
    a single merged row per player. Returns the number of rows written.
    """
    if engine is None:
        engine = get_engine()

    if season < TRACKING_START_SEASON:
        print(f"  → {season}: before tracking era ({TRACKING_START_SEASON}), skipping")
        return 0

    # Accumulate per-player dicts across the three measure types.
    players: dict[str, dict] = {}

    for measure_type, column_map in MEASURE_COLUMN_MAP.items():
        df = _fetch_measure(season, measure_type)
        time.sleep(REQUEST_DELAY)

        if df is None:
            print(f"    ⚠ {season} {measure_type}: no data returned")
            continue

        missing = [c for c in column_map if c not in df.columns]
        if missing:
            print(f"    ⚠ {season} {measure_type}: missing columns {missing} — "
                  f"those fields stay NULL")

        for row in df.itertuples(index=False):
            pid = str(getattr(row, "PLAYER_ID"))
            rec = players.setdefault(pid, {"player_id": pid, "season": season})

            # GP / MIN appear in every measure type; take the first non-null.
            if rec.get("gp") is None:
                gp = getattr(row, "GP", None)
                rec["gp"] = int(gp) if pd.notna(gp) else None
            if rec.get("min_per_game") is None:
                mins = getattr(row, "MIN", None)
                rec["min_per_game"] = float(mins) if pd.notna(mins) else None

            for api_col, orm_col in column_map.items():
                if api_col not in df.columns:
                    continue
                val = getattr(row, api_col, None)
                rec[orm_col] = float(val) if pd.notna(val) else None

        print(f"    ✓ {measure_type:<12} {len(df):>4} players")

    if not players:
        return 0

    records = list(players.values())

    # Upsert — idempotent, so a re-run refreshes rather than duplicating.
    with engine.begin() as conn:
        stmt = sqlite_insert(PlayerTrackingStats).values(records)
        update_cols = {
            c.name: stmt.excluded[c.name]
            for c in PlayerTrackingStats.__table__.columns
            if c.name not in ("player_id", "season")
        }
        conn.execute(stmt.on_conflict_do_update(
            index_elements=["player_id", "season"],
            set_=update_cols,
        ))

    return len(records)


def ingest(seasons: list[str]) -> int:
    """Ingest tracking stats for a list of seasons."""
    engine = get_engine()
    init_db(engine)

    eligible = [s for s in seasons if s >= TRACKING_START_SEASON]
    skipped = [s for s in seasons if s < TRACKING_START_SEASON]

    print(f"\n{'='*60}")
    print("  PLAYER TRACKING INGESTOR (creation skill)")
    print(f"  Seasons: {len(eligible)} eligible"
          + (f", {len(skipped)} pre-tracking skipped" if skipped else ""))
    print(f"{'='*60}")

    total = 0
    for season in tqdm(eligible, desc="  Seasons"):
        print(f"\n  → {season}")
        total += ingest_season(season, engine=engine)

    print(f"\n{'='*60}")
    print(f"  TRACKING INGEST COMPLETE — {total:,} player-seasons written")
    print(f"{'='*60}\n")
    return total


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingest player tracking stats (handle + passing creation skill)."
    )
    parser.add_argument("--full", action="store_true",
                        help="All seasons in config.ALL_SEASONS (2013-14 onward)")
    parser.add_argument("--season", type=str, default=None,
                        help="Single season, e.g. 2023-24")
    args = parser.parse_args()

    if args.season:
        target = [args.season]
    elif args.full:
        target = config.ALL_SEASONS
    else:
        target = config.TEST_SEASONS

    ingest(target)
