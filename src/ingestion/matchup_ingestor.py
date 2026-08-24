"""
Matchup Ingestor — Pulls game-level who-guarded-who data.

Uses BoxScoreMatchupsV3 to get matchup minutes, FGM/FGA per offensive-defensive
player pair for each game. Available from 2016-17 onward.

~13,000 API calls. Fully resumable — tracks completed games.
Run overnight.

Usage:
    python -m src.ingestion.matchup_ingestor                  # All games 2016+
    python -m src.ingestion.matchup_ingestor --season 2023-24 # One season
"""
import sys
import time
from pathlib import Path

from nba_api.stats.endpoints import boxscorematchupsv3
from sqlalchemy import text
from sqlalchemy.dialects.sqlite import insert as sqlite_upsert
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from src.db.database import get_engine, get_session_factory
from src.db.models import Matchup

# Matchup data is only available from 2016-17 onward
MATCHUP_START_SEASON = "2016-17"


def _safe_int(val, default=None):
    if val is None:
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def _safe_float(val, default=None):
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _get_games_to_process(engine, season=None):
    """Get list of game_ids that need matchup data."""
    with engine.connect() as conn:
        # Get all games from 2016-17 onward
        if season:
            result = conn.execute(text(
                "SELECT DISTINCT game_id FROM shots "
                "WHERE season = :season "
                "ORDER BY game_id"
            ), {"season": season})
        else:
            # All seasons from 2016-17 onward
            eligible_seasons = [
                s for s in config.ALL_SEASONS
                if s >= MATCHUP_START_SEASON
            ]
            placeholders = ", ".join(f"'{s}'" for s in eligible_seasons)
            result = conn.execute(text(
                f"SELECT DISTINCT game_id FROM shots "
                f"WHERE season IN ({placeholders}) "
                f"ORDER BY game_id"
            ))
        return [row[0] for row in result]


def _get_completed_games(engine):
    """Get set of game_ids already in the matchups table."""
    with engine.connect() as conn:
        result = conn.execute(text("SELECT DISTINCT game_id FROM matchups"))
        return set(row[0] for row in result)


def ingest_matchups(season=None):
    """Pull matchup data for all games from 2016-17 onward."""
    engine = get_engine()
    Session = get_session_factory(engine)

    all_games = _get_games_to_process(engine, season)
    completed = _get_completed_games(engine)
    games_to_do = [g for g in all_games if g not in completed]

    print(f"\n{'='*60}")
    print("  MATCHUP INGESTOR")
    if season:
        print(f"  Season: {season}")
    else:
        print(f"  All seasons from {MATCHUP_START_SEASON} onward")
    print(f"{'='*60}")
    print(f"  → {len(all_games):,} total games")
    print(f"  → {len(completed):,} already completed")
    print(f"  → {len(games_to_do):,} games to process")
    print(f"  → Estimated time: ~{len(games_to_do) * 1.1 / 60:.0f} minutes")
    print()

    if not games_to_do:
        print("  ✓ Nothing to do!")
        return 0

    total_rows = 0
    total_errors = 0

    for game_id in tqdm(games_to_do, desc="  Processing games"):
        df = None
        for attempt in range(3):  # Up to 3 retries
            try:
                data = boxscorematchupsv3.BoxScoreMatchupsV3(game_id=game_id)
                df = data.get_data_frames()[0]
                break  # Success
            except Exception as e:
                if attempt < 2:
                    wait = config.REQUEST_DELAY * (2 ** (attempt + 1))  # Exponential backoff
                    time.sleep(wait)
                else:
                    total_errors += 1
                    if total_errors <= 10:
                        tqdm.write(f"  ⚠ {game_id}: {e}")
                    time.sleep(config.REQUEST_DELAY * 2)

        if df is None:
            continue

        if df.empty:
            time.sleep(config.REQUEST_DELAY)
            continue

        rows_to_upsert = []
        for _, row in df.iterrows():
            off_id = str(_safe_int(row.get("personIdOff"), 0))
            def_id = str(_safe_int(row.get("personIdDef"), 0))

            if off_id == "0" or def_id == "0":
                continue

            rows_to_upsert.append({
                "game_id": str(row.get("gameId", game_id)),
                "offense_player_id": off_id,
                "defense_player_id": def_id,
                "matchup_minutes": _safe_float(row.get("matchupMinutesSort")),
                "partial_possessions": _safe_float(row.get("partialPossessions")),
                "player_points": _safe_int(row.get("playerPoints")),
                "matchup_fgm": _safe_int(row.get("matchupFieldGoalsMade")),
                "matchup_fga": _safe_int(row.get("matchupFieldGoalsAttempted")),
                "matchup_fg_pct": _safe_float(row.get("matchupFieldGoalsPercentage")),
            })

        if rows_to_upsert:
            with Session() as session:
                for r in rows_to_upsert:
                    stmt = sqlite_upsert(Matchup.__table__).values(**r)
                    stmt = stmt.on_conflict_do_update(
                        index_elements=["game_id", "offense_player_id", "defense_player_id"],
                        set_={
                            "matchup_minutes": stmt.excluded.matchup_minutes,
                            "partial_possessions": stmt.excluded.partial_possessions,
                            "player_points": stmt.excluded.player_points,
                            "matchup_fgm": stmt.excluded.matchup_fgm,
                            "matchup_fga": stmt.excluded.matchup_fga,
                            "matchup_fg_pct": stmt.excluded.matchup_fg_pct,
                        },
                    )
                    session.execute(stmt)
                session.commit()
                total_rows += len(rows_to_upsert)

        time.sleep(config.REQUEST_DELAY)

    print(f"\n{'='*60}")
    print("  MATCHUP INGESTOR COMPLETE")
    print(f"  Total matchup rows: {total_rows:,}")
    print(f"  Games failed:       {total_errors:,}")
    print(f"{'='*60}\n")

    return total_rows


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ingest matchup data from BoxScoreMatchupsV3.")
    parser.add_argument("--season", type=str, default=None,
                        help="Process a specific season only (e.g. '2023-24')")
    args = parser.parse_args()

    ingest_matchups(season=args.season)
