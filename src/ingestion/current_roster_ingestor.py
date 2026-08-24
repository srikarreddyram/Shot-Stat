"""
Current-Season Roster Ingestion.

roster_ingestor.py sources from LeagueDashPlayerStats, which is derived from
played games — it returns nothing for a season before that season's games
have started. This ingestor sources from CommonTeamRoster instead, which
reflects each team's currently-signed roster independent of games played, so
it works for the upcoming season during the offseason (new draftees, trades,
signings already show up).

Populates only identity fields (name, position, team). Stat columns are left
NULL — there is no real data to put there yet. A player who has never
appeared in the `shots` table has no measured shooting profile; predictions
for them fall back to src/training/position_priors.py's labeled prior,
never to a value from this ingestor.

Usage:
    python -m src.ingestion.current_roster_ingestor              # config.CURRENT_SEASONS
    python -m src.ingestion.current_roster_ingestor --seasons 2026-27
"""
import sys
import time
from pathlib import Path

from nba_api.stats.endpoints import commonteamroster
from nba_api.stats.static import teams as nba_teams_static
from sqlalchemy.dialects.sqlite import insert as sqlite_upsert
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from src.db.database import get_engine, get_session_factory
from src.db.models import Player


def _normalize_position(pos: str) -> str | None:
    if not pos or not pos.strip():
        return None
    pos = pos.strip()
    position_map = {
        "Guard": "G", "Point Guard": "PG", "Shooting Guard": "SG",
        "Forward": "F", "Small Forward": "SF", "Power Forward": "PF",
        "Center": "C", "Forward-Center": "F-C", "Center-Forward": "C-F",
        "Guard-Forward": "G-F", "Forward-Guard": "F-G",
    }
    return position_map.get(pos, pos)


def ingest_current_rosters(seasons: list[str]):
    engine = get_engine()
    Session = get_session_factory(engine)
    all_teams = nba_teams_static.get_teams()

    print(f"\n{'='*60}")
    print("  CURRENT ROSTER INGESTION")
    print(f"  {len(seasons)} season(s) x {len(all_teams)} teams")
    print(f"{'='*60}")

    total_players = 0
    failed_calls = []

    for season in seasons:
        season_players = 0
        print(f"\n  Season {season}:")

        for team in tqdm(all_teams, desc=f"  {season}"):
            team_id = team["id"]
            team_abbr = team["abbreviation"]

            try:
                roster = commonteamroster.CommonTeamRoster(team_id=team_id, season=season)
                df = roster.get_data_frames()[0]
            except Exception as e:
                failed_calls.append((season, team_abbr, str(e)))
                time.sleep(config.REQUEST_DELAY * 2)
                continue

            if df.empty:
                time.sleep(config.REQUEST_DELAY)
                continue

            players_to_upsert = []
            for _, row in df.iterrows():
                players_to_upsert.append({
                    "player_id": str(row["PLAYER_ID"]),
                    "season": season,
                    "name": row["PLAYER"],
                    "position": _normalize_position(str(row.get("POSITION", ""))),
                })

            with Session() as session:
                for p_data in players_to_upsert:
                    stmt = sqlite_upsert(Player.__table__).values(**p_data)
                    stmt = stmt.on_conflict_do_update(
                        index_elements=["player_id", "season"],
                        set_={
                            "name": stmt.excluded.name,
                            "position": stmt.excluded.position,
                        },
                    )
                    session.execute(stmt)
                session.commit()

            season_players += len(players_to_upsert)
            time.sleep(config.REQUEST_DELAY)

        total_players += season_players
        print(f"  ✓ {season}: {season_players} players rostered")

    print(f"\n{'='*60}")
    print(f"  ✓ Current roster ingestion complete: {total_players} records")
    if failed_calls:
        print(f"  ⚠ {len(failed_calls)} team calls failed:")
        for season, team, err in failed_calls[:5]:
            print(f"    • {season} {team}: {err}")
    print(f"{'='*60}\n")

    return total_players


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ingest current NBA rosters (pre-season, no games played yet).")
    parser.add_argument("--seasons", nargs="+", default=None, help="Specific seasons. Default: config.CURRENT_SEASONS.")
    args = parser.parse_args()

    seasons = args.seasons if args.seasons else config.CURRENT_SEASONS
    ingest_current_rosters(seasons)
