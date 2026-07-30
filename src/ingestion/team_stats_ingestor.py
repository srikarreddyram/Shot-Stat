"""
Team Stats Ingestor — Pulls team-level defensive rating per season.

Uses LeagueDashTeamStats to get each team's defensive rating (DEF_RATING) for the season.

Usage:
    python -m src.ingestion.team_stats_ingestor
"""
import sys
import time
from pathlib import Path

from nba_api.stats.endpoints import leaguedashteamstats
from nba_api.stats.static import teams as nba_teams
from sqlalchemy.dialects.sqlite import insert as sqlite_upsert

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from src.db.database import get_engine, get_session_factory
from src.db.models import TeamStats

# Build team_id → abbreviation mapping
_TEAM_ID_TO_ABBREV = {str(t["id"]): t["abbreviation"] for t in nba_teams.get_teams()}


def ingest_team_stats(seasons=None):
    if seasons is None:
        seasons = config.ALL_SEASONS

    engine = get_engine()
    Session = get_session_factory(engine)

    print(f"Ingesting team stats for {len(seasons)} seasons...")
    
    for season in seasons:
        print(f"  Fetching {season}...")
        try:
            stats = leaguedashteamstats.LeagueDashTeamStats(
                season=season,
                measure_type_detailed_defense="Advanced"
            ).get_data_frames()[0]

            with Session() as session:
                for _, row in stats.iterrows():
                    tid = str(row["TEAM_ID"])
                    abbrev = _TEAM_ID_TO_ABBREV.get(tid, row.get("TEAM_ABBREVIATION", None))
                    stmt = sqlite_upsert(TeamStats).values(
                        team_id=tid,
                        season=season,
                        team_name=row["TEAM_NAME"],
                        team_abbrev=abbrev,
                        def_rating=row["DEF_RATING"]
                    )
                    stmt = stmt.on_conflict_do_update(
                        index_elements=["team_id", "season"],
                        set_={
                            "team_name": stmt.excluded.team_name,
                            "team_abbrev": stmt.excluded.team_abbrev,
                            "def_rating": stmt.excluded.def_rating,
                        }
                    )
                    session.execute(stmt)
                session.commit()
            time.sleep(1)  # Rate limiting
        except Exception as e:
            print(f"    Failed for {season}: {e}")

    print("Done!")

if __name__ == "__main__":
    # We only really need from 2016-17 onward since matchup data starts then
    seasons_to_use = [s for s in config.ALL_SEASONS if s >= "2016-17"]
    ingest_team_stats(seasons_to_use)
