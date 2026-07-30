"""
Game metadata ingestion from nba_api.

Pulls all games per season from LeagueGameLog (team-level),
deduplicates (each game appears twice — once per team),
and upserts into the Games table.
"""
import sys
import time
from pathlib import Path
from datetime import datetime

from nba_api.stats.endpoints import leaguegamelog
from sqlalchemy.dialects.sqlite import insert as sqlite_upsert
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from src.db.database import get_engine, get_session_factory
from src.db.models import Game


def _parse_matchup(matchup: str, team_abbr: str, wl: str):
    """
    Parse 'DEN vs. LAL' or 'LAL @ DEN' to determine home/away teams.
    Returns (home_team, away_team, home_team_win) or None if this row is the away team's entry.
    """
    if " vs. " in matchup:
        # This is the home team's row — "HOME vs. AWAY"
        parts = matchup.split(" vs. ")
        home_won = 1 if wl == "W" else 0
        return parts[0].strip(), parts[1].strip(), home_won
    else:
        # This is the away team's row — "AWAY @ HOME" — skip to avoid dupes
        return None


def _is_playoff_game(game_id: str) -> int:
    """
    NBA game IDs encode game type in the leading digits:
    - 002xxxxx = Regular Season
    - 004xxxxx = Playoffs
    - 005xxxxx = Play-In
    """
    if game_id.startswith("004") or game_id.startswith("005"):
        return 1
    return 0


def ingest_games(seasons: list[str], include_playoffs: bool = True):
    """
    Pull game metadata for the given seasons and upsert into the Games table.

    Args:
        seasons: List of season strings, e.g. ["2023-24"]
        include_playoffs: Whether to also pull playoff games
    """
    engine = get_engine()
    Session = get_session_factory(engine)

    total_inserted = 0
    total_updated = 0

    for season in tqdm(seasons, desc="Seasons"):
        season_types = ["Regular Season"]
        if include_playoffs:
            season_types.append("Playoffs")

        for season_type in season_types:
            print(f"  Pulling {season_type} games for {season}...")
            try:
                gl = leaguegamelog.LeagueGameLog(
                    season=season,
                    season_type_all_star=season_type,
                    player_or_team_abbreviation="T",
                )
                df = gl.get_data_frames()[0]
            except Exception as e:
                print(f"  ✗ Error fetching {season} {season_type}: {e}")
                time.sleep(config.REQUEST_DELAY * 2)
                continue

            if df.empty:
                print(f"  No data for {season} {season_type}")
                time.sleep(config.REQUEST_DELAY)
                continue

            # Process each row — only keep the home team's entry to avoid dupes
            games_to_upsert = []
            for _, row in df.iterrows():
                parsed = _parse_matchup(row["MATCHUP"], row["TEAM_ABBREVIATION"], row["WL"])
                if parsed is None:
                    continue  # Skip away-team duplicate rows

                home_team, away_team, home_team_win = parsed
                game_date = datetime.strptime(row["GAME_DATE"], "%Y-%m-%d").date()
                game_id = row["GAME_ID"]

                games_to_upsert.append({
                    "game_id": game_id,
                    "date": game_date,
                    "home_team": home_team,
                    "away_team": away_team,
                    "playoff_flag": _is_playoff_game(game_id),
                    "home_team_win": home_team_win,
                })

            # Batch upsert using SQLite INSERT OR REPLACE
            if games_to_upsert:
                with Session() as session:
                    for game_data in games_to_upsert:
                        stmt = sqlite_upsert(Game.__table__).values(**game_data)
                        stmt = stmt.on_conflict_do_update(
                            index_elements=["game_id"],
                            set_={
                                "date": stmt.excluded.date,
                                "home_team": stmt.excluded.home_team,
                                "away_team": stmt.excluded.away_team,
                                "playoff_flag": stmt.excluded.playoff_flag,
                                "home_team_win": stmt.excluded.home_team_win,
                            },
                        )
                        session.execute(stmt)
                    session.commit()
                    inserted = len(games_to_upsert)
                    total_inserted += inserted
                    print(f"  ✓ {inserted} games upserted for {season} {season_type}")

            time.sleep(config.REQUEST_DELAY)

    print(f"\n✓ Game ingestion complete: {total_inserted} total games processed")
    return total_inserted


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ingest NBA game schedules and results.")
    parser.add_argument("--seasons", nargs="+", default=None, help="Specific seasons to process, e.g. 2023-24 2024-25. Default: TEST_SEASONS. Use --full for all.")
    parser.add_argument("--full", action="store_true", help="Process all seasons defined in config.")
    args = parser.parse_args()

    if args.seasons:
        seasons = args.seasons
    elif args.full:
        seasons = config.ALL_SEASONS
    else:
        seasons = config.TEST_SEASONS

    ingest_games(seasons)
