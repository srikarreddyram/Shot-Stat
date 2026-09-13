"""
Defensive Activity Ingestor — blocks, steals, deflections.

Populates `player_defensive_activity` from two NBA API endpoints, merged into
one row per (player_id, season):

    LeagueDashPlayerStats (Base, PerGame)   → STL, BLK, GP, MIN
    LeagueHustleStatsPlayer (PerGame)       → DEFLECTIONS

Two separate endpoints because STL/BLK are ordinary box-score counting stats
(available every season the league has existed) while DEFLECTIONS is a
SportVU/hustle-tracking stat with no season parameter restriction of its own
but no data before the league started publishing hustle stats league-wide in
2016-17 — which happens to be exactly this project's training window, so
there is no gap to work around.

Cost: 2 API calls per season — both endpoints return the whole league at
once, same shape as tracking_ingestor.py.

Usage:
    python -m src.ingestion.defensive_activity_ingestor                # test season
    python -m src.ingestion.defensive_activity_ingestor --full          # all seasons
    python -m src.ingestion.defensive_activity_ingestor --season 2023-24
"""
import sys
import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerstats, leaguehustlestatsplayer
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from src.db.database import get_engine, init_db
from src.db.models import PlayerDefensiveActivity

# ── Configuration ──────────────────────────────────────────────────────────
# Hustle stats (deflections) are published league-wide from 2016-17, which is
# also where config.ALL_SEASONS starts — nothing to skip.
DEFENSIVE_ACTIVITY_START_SEASON = "2016-17"

REQUEST_DELAY = 1.2
RETRY_DELAY = 10.0
MAX_RETRIES = 3


def _fetch(endpoint_cls, season: str, label: str, **kwargs) -> pd.DataFrame | None:
    for attempt in range(MAX_RETRIES):
        try:
            result = endpoint_cls(season=season, timeout=45, **kwargs)
            frames = result.get_data_frames()
            if not frames or frames[0].empty:
                return None
            return frames[0]
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"    ⚠ {label} {season} attempt {attempt+1} failed "
                      f"({type(e).__name__}) — retrying in {RETRY_DELAY}s")
                time.sleep(RETRY_DELAY)
            else:
                print(f"    ✗ {label} {season} failed after "
                      f"{MAX_RETRIES} attempts: {type(e).__name__}: {e}")
                return None


def ingest_season(season: str, engine=None) -> int:
    """Ingest STL/BLK/DEFLECTIONS for one season. Returns rows written."""
    if engine is None:
        engine = get_engine()

    if season < DEFENSIVE_ACTIVITY_START_SEASON:
        print(f"  → {season}: before hustle-stats era "
              f"({DEFENSIVE_ACTIVITY_START_SEASON}), skipping")
        return 0

    players: dict[str, dict] = {}

    base = _fetch(leaguedashplayerstats.LeagueDashPlayerStats, season, "Base",
                  per_mode_detailed="PerGame")
    time.sleep(REQUEST_DELAY)
    if base is not None:
        for row in base.itertuples(index=False):
            pid = str(getattr(row, "PLAYER_ID"))
            rec = players.setdefault(pid, {"player_id": pid, "season": season})
            gp = getattr(row, "GP", None)
            mins = getattr(row, "MIN", None)
            stl = getattr(row, "STL", None)
            blk = getattr(row, "BLK", None)
            rec["gp"] = int(gp) if pd.notna(gp) else None
            rec["min_per_game"] = float(mins) if pd.notna(mins) else None
            rec["stl"] = float(stl) if pd.notna(stl) else None
            rec["blk"] = float(blk) if pd.notna(blk) else None
        print(f"    ✓ Base            {len(base):>4} players")
    else:
        print(f"    ⚠ {season} Base: no data returned")

    hustle = _fetch(leaguehustlestatsplayer.LeagueHustleStatsPlayer, season,
                     "Hustle", per_mode_time="PerGame")
    time.sleep(REQUEST_DELAY)
    if hustle is not None:
        for row in hustle.itertuples(index=False):
            pid = str(getattr(row, "PLAYER_ID"))
            rec = players.setdefault(pid, {"player_id": pid, "season": season})
            defl = getattr(row, "DEFLECTIONS", None)
            rec["deflections"] = float(defl) if pd.notna(defl) else None
        print(f"    ✓ Hustle          {len(hustle):>4} players")
    else:
        print(f"    ⚠ {season} Hustle: no data returned")

    if not players:
        return 0

    # Normalize every record to the same key set before the bulk insert. A
    # player can appear in one endpoint's frame but not the other's (Base
    # and Hustle don't guarantee identical rosters), which would otherwise
    # leave some dicts missing a key the others have — SQLAlchemy's
    # multi-row VALUES insert requires uniform columns across the whole
    # batch and raises a CompileError rather than silently NULLing the gap.
    all_cols = [c.name for c in PlayerDefensiveActivity.__table__.columns]
    records = [
        {col: rec.get(col) for col in all_cols}
        for rec in players.values()
    ]
    with engine.begin() as conn:
        stmt = sqlite_insert(PlayerDefensiveActivity).values(records)
        update_cols = {
            c.name: stmt.excluded[c.name]
            for c in PlayerDefensiveActivity.__table__.columns
            if c.name not in ("player_id", "season")
        }
        conn.execute(stmt.on_conflict_do_update(
            index_elements=["player_id", "season"],
            set_=update_cols,
        ))

    return len(records)


def ingest(seasons: list[str]) -> int:
    engine = get_engine()
    init_db(engine)

    eligible = [s for s in seasons if s >= DEFENSIVE_ACTIVITY_START_SEASON]
    skipped = [s for s in seasons if s < DEFENSIVE_ACTIVITY_START_SEASON]

    print(f"\n{'='*60}")
    print("  DEFENSIVE ACTIVITY INGESTOR (blocks / steals / deflections)")
    print(f"  Seasons: {len(eligible)} eligible"
          + (f", {len(skipped)} pre-hustle-stats skipped" if skipped else ""))
    print(f"{'='*60}")

    total = 0
    for season in tqdm(eligible, desc="  Seasons"):
        print(f"\n  → {season}")
        total += ingest_season(season, engine=engine)

    print(f"\n{'='*60}")
    print(f"  DEFENSIVE ACTIVITY INGEST COMPLETE — {total:,} player-seasons written")
    print(f"{'='*60}\n")
    return total


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingest defensive activity stats (blocks, steals, deflections)."
    )
    parser.add_argument("--full", action="store_true",
                        help="All seasons in config.ALL_SEASONS (2016-17 onward)")
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
