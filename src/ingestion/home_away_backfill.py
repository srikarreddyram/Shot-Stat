"""
Home/Away Backfill — Fixes the `home_away` column for all existing shots.

The original shot_ingestor.py had a bug where it compared full team names
("Los Angeles Lakers") against HTM/VTM abbreviations ("LAL"/"PHX"), so
home_away was always NULL.

This script re-derives home_away by:
  1. Querying each game's home/away teams from the Games table
  2. Querying each player's team_id from the API (cached per player-season)
  3. Mapping team_id -> abbreviation using nba_api static data
  4. Setting home_away = 1 if player's team == home_team, 0 if away

This is a BULK SQL approach — no API calls needed since we can derive
team membership from the shots + games tables.

Usage:
    python -m src.ingestion.home_away_backfill
"""
import sys
import time
from pathlib import Path

from sqlalchemy import text
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from src.db.database import get_engine


def backfill_home_away():
    """
    Derive home_away from game metadata.

    Strategy: The ShotChartDetail API doesn't store team_abbr on the shot,
    BUT each game has exactly 2 teams. We can figure out which team a player
    was on by checking the game's box score or roster.

    Simpler approach: Use the nba_api to pull team_id for each player-season,
    then cross-reference with games table.

    Simplest approach: Re-pull ShotChartDetail per player-season (same as
    original ingestion) but only update home_away. However, this is 8000+
    API calls.

    FASTEST approach: We already have the shots. For each shot, we know the
    game_id. From the games table, we know home_team and away_team (abbreviations).
    We need to know which team the player was on during that game.

    We can get this from LeagueDashPlayerStats which has TEAM_ABBREVIATION.
    One call per season = 16 calls total. Then join player -> team -> game.
    """
    engine = get_engine()

    print(f"\n{'='*60}")
    print("  HOME/AWAY BACKFILL")
    print(f"{'='*60}")

    # Step 1: Build player_id -> team_abbreviation mapping per season
    # Using LeagueDashPlayerStats (one call per season, returns all players + teams)
    from nba_api.stats.endpoints import leaguedashplayerstats

    player_team_map = {}  # (player_id, season) -> team_abbreviation

    for season in tqdm(config.ALL_SEASONS, desc="  Pulling team rosters"):
        try:
            stats = leaguedashplayerstats.LeagueDashPlayerStats(
                season=season, season_type_all_star="Regular Season"
            )
            df = stats.get_data_frames()[0]
            for _, row in df.iterrows():
                pid = str(row["PLAYER_ID"])
                team_abbr = row.get("TEAM_ABBREVIATION", "")
                if team_abbr:
                    player_team_map[(pid, season)] = team_abbr
        except Exception as e:
            print(f"\n  ⚠ Failed for {season}: {e}")
        time.sleep(config.REQUEST_DELAY)

    print(f"  ✓ Built team map for {len(player_team_map):,} player-seasons")

    # Also pull playoff rosters (some players only appear in playoffs)
    for season in tqdm(config.ALL_SEASONS, desc="  Pulling playoff rosters"):
        try:
            stats = leaguedashplayerstats.LeagueDashPlayerStats(
                season=season, season_type_all_star="Playoffs"
            )
            df = stats.get_data_frames()[0]
            for _, row in df.iterrows():
                pid = str(row["PLAYER_ID"])
                team_abbr = row.get("TEAM_ABBREVIATION", "")
                key = (pid, season)
                if team_abbr and key not in player_team_map:
                    player_team_map[key] = team_abbr
        except Exception:
            pass
        time.sleep(config.REQUEST_DELAY)

    print(f"  ✓ Total player-season team mappings: {len(player_team_map):,}")

    # Step 2: Load games into memory (game_id -> home_team, away_team)
    game_map = {}  # game_id -> (home_team, away_team)
    with engine.connect() as conn:
        result = conn.execute(text("SELECT game_id, home_team, away_team FROM games"))
        for row in result:
            game_map[row[0]] = (row[1], row[2])

    print(f"  ✓ Loaded {len(game_map):,} games")

    # Step 3: Read all shots with NULL home_away, compute, and batch update
    with engine.connect() as conn:
        null_count = conn.execute(
            text("SELECT COUNT(*) FROM shots WHERE home_away IS NULL")
        ).scalar()
        print(f"  → {null_count:,} shots need home_away")

    if null_count == 0:
        print("  ✓ Nothing to backfill!")
        return

    # Process in chunks to avoid memory issues
    CHUNK_SIZE = 50_000
    total_updated = 0
    total_unresolved = 0

    with engine.begin() as conn:
        # Read all shots needing update
        result = conn.execute(text(
            "SELECT shot_id, game_id, player_id, season "
            "FROM shots WHERE home_away IS NULL"
        ))

        batch = []
        for row in tqdm(result, total=null_count, desc="  Computing home_away"):
            shot_id, game_id, player_id, season = row

            game_info = game_map.get(game_id)
            team_abbr = player_team_map.get((player_id, season))

            if game_info and team_abbr:
                home_team, away_team = game_info
                if team_abbr == home_team:
                    home_away = 1
                elif team_abbr == away_team:
                    home_away = 0
                else:
                    # Player might have been traded mid-season
                    home_away = None
                    total_unresolved += 1
            else:
                home_away = None
                total_unresolved += 1

            if home_away is not None:
                batch.append({"sid": shot_id, "ha": home_away})

            if len(batch) >= CHUNK_SIZE:
                # Batch update using a temp approach
                for item in batch:
                    conn.execute(
                        text("UPDATE shots SET home_away = :ha WHERE shot_id = :sid"),
                        item
                    )
                total_updated += len(batch)
                batch = []

        # Flush remaining
        if batch:
            for item in batch:
                conn.execute(
                    text("UPDATE shots SET home_away = :ha WHERE shot_id = :sid"),
                    item
                )
            total_updated += len(batch)

    print(f"\n{'='*60}")
    print("  HOME/AWAY BACKFILL COMPLETE")
    print(f"  Updated:    {total_updated:,}")
    print(f"  Unresolved: {total_unresolved:,} (traded mid-season or missing game)")
    print(f"{'='*60}\n")

    return total_updated


if __name__ == "__main__":
    backfill_home_away()
