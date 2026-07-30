"""
Position Backfill — Populates the `position` column in the Players table.

The `LeagueDashPlayerStats` endpoint used by roster_ingestor.py does NOT
return a POSITION column. Instead, we pull position from `CommonTeamRoster`
(one call per team per season → 30 teams × 16 seasons = 480 calls).

This script:
  1. Iterates every season in config.ALL_SEASONS
  2. For each season, pulls the roster for all 30 NBA teams
  3. Updates the `position` column for every matching (player_id, season) row
  4. Only updates NULL positions — never overwrites existing data

Usage:
    python -m src.ingestion.position_backfill              # TEST_SEASONS only
    python -m src.ingestion.position_backfill --full       # All seasons
"""
import sys
import time
from pathlib import Path

from nba_api.stats.endpoints import commonteamroster
from nba_api.stats.static import teams as nba_teams_static
from sqlalchemy import update
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from src.db.database import get_engine, get_session_factory
from src.db.models import Player


def _normalize_position(pos: str) -> str | None:
    """Normalize position strings from the API to our standard abbreviations."""
    if not pos or not pos.strip():
        return None
    pos = pos.strip()
    # CommonTeamRoster already returns abbreviated forms (G, F, C, F-C, etc.)
    # but let's handle any long-form values just in case
    position_map = {
        "Guard": "G", "Point Guard": "PG", "Shooting Guard": "SG",
        "Forward": "F", "Small Forward": "SF", "Power Forward": "PF",
        "Center": "C", "Forward-Center": "F-C", "Center-Forward": "C-F",
        "Guard-Forward": "G-F", "Forward-Guard": "F-G",
    }
    return position_map.get(pos, pos)


def backfill_positions(seasons: list[str]):
    """Pull position data from CommonTeamRoster and backfill the Players table."""
    engine = get_engine()
    Session = get_session_factory(engine)

    all_teams = nba_teams_static.get_teams()
    print(f"\n{'='*60}")
    print(f"  POSITION BACKFILL")
    print(f"  {len(seasons)} seasons × {len(all_teams)} teams = {len(seasons) * len(all_teams)} API calls")
    print(f"{'='*60}")

    total_updated = 0
    total_not_found = 0
    failed_calls = []

    for season in seasons:
        season_updated = 0
        season_skipped = 0

        print(f"\n  Season {season}:")

        for team in tqdm(all_teams, desc=f"  {season}", leave=False):
            team_id = team["id"]
            team_abbr = team["abbreviation"]

            try:
                roster = commonteamroster.CommonTeamRoster(
                    team_id=team_id, season=season
                )
                df = roster.get_data_frames()[0]
            except Exception as e:
                failed_calls.append((season, team_abbr, str(e)))
                time.sleep(config.REQUEST_DELAY * 2)
                continue

            if df.empty:
                time.sleep(config.REQUEST_DELAY)
                continue

            with Session() as session:
                for _, row in df.iterrows():
                    player_id = str(row["PLAYER_ID"])
                    position = _normalize_position(str(row.get("POSITION", "")))

                    if not position:
                        continue

                    # Only update if position is currently NULL
                    result = session.execute(
                        update(Player)
                        .where(Player.player_id == player_id)
                        .where(Player.season == season)
                        .where(Player.position.is_(None))
                        .values(position=position)
                    )

                    if result.rowcount > 0:
                        season_updated += 1
                    else:
                        season_skipped += 1

                session.commit()

            time.sleep(config.REQUEST_DELAY)

        total_updated += season_updated
        print(f"  ✓ {season}: {season_updated} positions updated, {season_skipped} already had position or not in DB")

    print(f"\n{'='*60}")
    print(f"  POSITION BACKFILL COMPLETE")
    print(f"  Total updated:  {total_updated:,}")
    if failed_calls:
        print(f"  Failed calls:   {len(failed_calls)}")
        for season, team, err in failed_calls[:5]:
            print(f"    • {season} {team}: {err}")
    print(f"{'='*60}\n")

    return total_updated


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Backfill player positions from CommonTeamRoster.")
    parser.add_argument("--seasons", nargs="+", default=None,
                        help="Specific seasons. Default: TEST_SEASONS. Use --full for all.")
    parser.add_argument("--full", action="store_true",
                        help="Process all seasons defined in config.")
    args = parser.parse_args()

    if args.seasons:
        seasons = args.seasons
    elif args.full:
        seasons = config.ALL_SEASONS
    else:
        seasons = config.TEST_SEASONS

    backfill_positions(seasons)
