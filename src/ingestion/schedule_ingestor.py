"""
Schedule Ingestor — Computes rest days and back-to-back flags for each team in each game.

Derives this from the existing `games` table to avoid API rate limits.
Saves to `team_schedule` table.

Usage:
    python -m src.ingestion.schedule_ingestor
"""
import sys
from pathlib import Path
import pandas as pd
from sqlalchemy.dialects.sqlite import insert as sqlite_upsert

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.db.database import get_engine, get_session_factory
from src.db.models import TeamSchedule

def compute_schedule():
    engine = get_engine()
    Session = get_session_factory(engine)

    print("Fetching games from database...")
    with engine.connect() as conn:
        games_df = pd.read_sql("SELECT game_id, date, home_team, away_team FROM games ORDER BY date", conn)
        
        # Get seasons mapping from shots table to know which season a game belongs to
        seasons_df = pd.read_sql("SELECT DISTINCT game_id, season FROM shots", conn)

    # Merge to get season
    games_df = games_df.merge(seasons_df, on="game_id", how="inner")
    
    games_df["date"] = pd.to_datetime(games_df["date"])

    # Create an unpivoted dataframe of Team -> Game -> Date
    home_games = games_df[["game_id", "date", "season", "home_team"]].rename(columns={"home_team": "team_id"})
    away_games = games_df[["game_id", "date", "season", "away_team"]].rename(columns={"away_team": "team_id"})
    
    all_team_games = pd.concat([home_games, away_games]).sort_values(["team_id", "date"])

    print("Computing rest days...")
    # Calculate days since last game per team
    all_team_games["prev_date"] = all_team_games.groupby("team_id")["date"].shift(1)
    all_team_games["rest_days"] = (all_team_games["date"] - all_team_games["prev_date"]).dt.days - 1
    
    # If first game of season or missing, assume well-rested (e.g. 5 days)
    # Better yet, only shift within the same season
    all_team_games["prev_date_season"] = all_team_games.groupby(["team_id", "season"])["date"].shift(1)
    all_team_games["rest_days"] = (all_team_games["date"] - all_team_games["prev_date_season"]).dt.days - 1
    
    all_team_games["rest_days"] = all_team_games["rest_days"].fillna(5) # Default rest for first game
    all_team_games["rest_days"] = all_team_games["rest_days"].clip(lower=0, upper=5) # Cap at 5+ days
    
    all_team_games["is_back_to_back"] = (all_team_games["rest_days"] == 0).astype(int)

    print(f"Upserting {len(all_team_games)} team-game records...")
    
    # Fast bulk insert
    all_team_games["date_str"] = all_team_games["date"].dt.strftime("%Y-%m-%d")
    
    records = []
    for _, row in all_team_games.iterrows():
        records.append({
            "team_id": row["team_id"],
            "game_id": row["game_id"],
            "date": row["date"].to_pydatetime().date(),
            "season": row["season"],
            "rest_days": int(row["rest_days"]),
            "is_back_to_back": int(row["is_back_to_back"])
        })
    
    with Session() as session:
        # Use bulk insert for speed, or one by one
        # Because we want upsert:
        for chunk in [records[i:i + 1000] for i in range(0, len(records), 1000)]:
            stmt = sqlite_upsert(TeamSchedule).values(chunk)
            stmt = stmt.on_conflict_do_update(
                index_elements=["team_id", "game_id"],
                set_={
                    "rest_days": stmt.excluded.rest_days,
                    "is_back_to_back": stmt.excluded.is_back_to_back
                }
            )
            session.execute(stmt)
        session.commit()

    print("Done!")

if __name__ == "__main__":
    compute_schedule()
