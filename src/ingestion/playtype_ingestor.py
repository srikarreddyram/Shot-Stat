"""
Play-Type Ingestor — Synergy offensive tendencies: isolation, both
pick-and-roll roles, post-up, spot-up, hand-off, cut, off-ball screen,
transition, and putbacks (offensive rebounds put back up).

Populates `player_play_type` from SynergyPlayTypes — one league-wide call
per (season, play_type), offensive grouping only, in Totals per-mode so
every stored column is a real count a career figure can later be
recomputed from, the same reasoning zone_stats_ingestor.py and
defender_stats_ingestor.py already apply.

Data begins 2015-16, the first season NBA.com published Synergy-sourced
play-type data; earlier seasons are skipped, the same handling
defensive_activity_ingestor.py already applies to hustle stats' 2016-17
start.

Cost: 10 play types x 1 call each = 10 calls per season, ~90 for a full
2015-16-onward backfill (config.ALL_SEASONS starts in 2010-11; the 5
pre-Synergy seasons are skipped automatically).

Usage:
    python -m src.ingestion.playtype_ingestor                # test season
    python -m src.ingestion.playtype_ingestor --full          # all eligible seasons
    python -m src.ingestion.playtype_ingestor --season 2023-24
"""
import sys
import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import synergyplaytypes
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from src.db.database import get_engine, init_db
from src.db.models import PlayerPlayType

PLAYTYPE_START_SEASON = "2015-16"

# Synergy's own play-type codes. OffRebound covers putbacks off an offensive
# rebound — included because it is a genuine offensive action type, not
# because it is a "play type" in the isolation/PnR sense.
PLAY_TYPES = [
    "Isolation", "Transition", "PRBallHandler", "PRRollman", "Postup",
    "Spotup", "Handoff", "Cut", "OffScreen", "OffRebound",
]

REQUEST_DELAY = 1.0
RETRY_DELAY = 10.0
MAX_RETRIES = 3


def _fetch(season: str, play_type: str) -> pd.DataFrame | None:
    for attempt in range(MAX_RETRIES):
        try:
            result = synergyplaytypes.SynergyPlayTypes(
                season=season,
                season_type_all_star="Regular Season",
                play_type_nullable=play_type,
                type_grouping_nullable="offensive",
                player_or_team_abbreviation="P",
                per_mode_simple="Totals",
                timeout=45,
            )
            frames = result.get_data_frames()
            if not frames or frames[0].empty:
                return None
            return frames[0]
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"    ⚠ {season} {play_type} attempt {attempt+1} failed "
                      f"({type(e).__name__}) — retrying in {RETRY_DELAY}s")
                time.sleep(RETRY_DELAY)
            else:
                print(f"    ✗ {season} {play_type} failed after "
                      f"{MAX_RETRIES} attempts: {type(e).__name__}: {e}")
                return None


def ingest_season(season: str, engine=None) -> int:
    """Ingest every play type for one season. Returns rows written."""
    if engine is None:
        engine = get_engine()

    if season < PLAYTYPE_START_SEASON:
        print(f"  → {season}: before Synergy play-type era "
              f"({PLAYTYPE_START_SEASON}), skipping")
        return 0

    records = []
    for play_type in PLAY_TYPES:
        df = _fetch(season, play_type)
        time.sleep(REQUEST_DELAY)
        if df is None:
            print(f"    ⚠ {season} {play_type}: no data returned")
            continue

        for row in df.itertuples(index=False):
            def _int(v):
                return int(v) if pd.notna(v) else None

            def _float(v):
                return float(v) if pd.notna(v) else None

            records.append({
                "player_id": str(getattr(row, "PLAYER_ID")),
                "season": season,
                "play_type": play_type,
                "gp": _int(getattr(row, "GP", None)),
                "poss": _int(getattr(row, "POSS", None)),
                "poss_pct": _float(getattr(row, "POSS_PCT", None)),
                "pts": _int(getattr(row, "PTS", None)),
                "fgm": _int(getattr(row, "FGM", None)),
                "fga": _int(getattr(row, "FGA", None)),
                "fg_pct": _float(getattr(row, "FG_PCT", None)),
                "efg_pct": _float(getattr(row, "EFG_PCT", None)),
                "ppp": _float(getattr(row, "PPP", None)),
                "percentile": _float(getattr(row, "PERCENTILE", None)),
            })
        print(f"    ✓ {play_type:14s} {len(df):>4} players")

    if not records:
        return 0

    with engine.begin() as conn:
        stmt = sqlite_insert(PlayerPlayType).values(records)
        update_cols = {
            c.name: stmt.excluded[c.name]
            for c in PlayerPlayType.__table__.columns
            if c.name not in ("player_id", "season", "play_type")
        }
        conn.execute(stmt.on_conflict_do_update(
            index_elements=["player_id", "season", "play_type"],
            set_=update_cols,
        ))

    return len(records)


def ingest(seasons: list[str]) -> int:
    engine = get_engine()
    init_db(engine)

    eligible = [s for s in seasons if s >= PLAYTYPE_START_SEASON]
    skipped = [s for s in seasons if s < PLAYTYPE_START_SEASON]

    print(f"\n{'='*60}")
    print("  PLAY-TYPE INGESTOR (Synergy offensive tendencies)")
    print(f"  Seasons: {len(eligible)} eligible"
          + (f", {len(skipped)} pre-Synergy skipped" if skipped else ""))
    print(f"{'='*60}")

    total = 0
    for season in tqdm(eligible, desc="  Seasons"):
        print(f"\n  → {season}")
        total += ingest_season(season, engine=engine)

    print(f"\n{'='*60}")
    print(f"  PLAY-TYPE INGEST COMPLETE — {total:,} player-season-playtype rows written")
    print(f"{'='*60}\n")
    return total


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingest Synergy play-type data (isolation, PnR, post-up, spot-up, etc.)."
    )
    parser.add_argument("--full", action="store_true",
                        help="All seasons in config.ALL_SEASONS (2015-16 onward)")
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
