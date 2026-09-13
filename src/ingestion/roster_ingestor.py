"""
Phase 1: Roster Ingestion
Fetches the base roster of players for each season and initializes rows in the Players table.
Does NOT fetch physical attributes or shot data.
"""
import sys
import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerstats
from sqlalchemy.dialects.sqlite import insert as sqlite_upsert
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from src.db.database import get_engine, get_session_factory
from src.db.models import Player

def _safe_float(val, default=None):
    if val is None: return default
    try: return float(val)
    except (ValueError, TypeError): return default

def ingest_rosters(seasons: list[str]):
    engine = get_engine()
    Session = get_session_factory(engine)
    total_players = 0

    for season in tqdm(seasons, desc="Seasons (Roster)"):
        print(f"\n  Pulling roster for {season}...")
        try:
            stats = leaguedashplayerstats.LeagueDashPlayerStats(
                season=season,
                season_type_all_star="Regular Season",
            )
            df = stats.get_data_frames()[0]
        except Exception as e:
            print(f"  ✗ Error fetching roster for {season}: {e}")
            time.sleep(config.REQUEST_DELAY * 2)
            continue
            
        if df.empty:
            continue

        players_to_upsert = []
        for _, row in df.iterrows():
            players_to_upsert.append({
                "player_id": str(row["PLAYER_ID"]),
                "season": season,
                "name": row["PLAYER_NAME"],
                # No position: LeagueDashPlayerStats does not return one. This
                # used to read a "PLAYER_POSITION" column that has never
                # existed on this endpoint, so it silently wrote NULL over
                # every player in every season it touched — and because the
                # upsert below also SET position, re-running it would erase a
                # position another ingestor had correctly filled in. Position
                # is owned by current_roster_ingestor.py (CommonTeamRoster),
                # which actually has it.
                "team_id": str(row["TEAM_ID"]) if pd.notna(row.get("TEAM_ID")) else None,
                "season_fg_pct": _safe_float(row.get("FG_PCT")),
                "ast": _safe_float(row.get("AST")) / max(1, _safe_float(row.get("GP"), 1)),
                "tov": _safe_float(row.get("TOV")) / max(1, _safe_float(row.get("GP"), 1)),
                "ft_pct": _safe_float(row.get("FT_PCT")),
                # Physicals remain NULL
                "height": None,
                "weight": None,
                "wingspan": None,
                "wingspan_source": None,
            })

        if players_to_upsert:
            with Session() as session:
                for p_data in players_to_upsert:
                    stmt = sqlite_upsert(Player.__table__).values(**p_data)
                    stmt = stmt.on_conflict_do_update(
                        index_elements=["player_id", "season"],
                        set_={
                            "name": stmt.excluded.name,
                            "team_id": stmt.excluded.team_id,
                            "season_fg_pct": stmt.excluded.season_fg_pct,
                            "ast": stmt.excluded.ast,
                            "tov": stmt.excluded.tov,
                            "ft_pct": stmt.excluded.ft_pct,
                        }
                    )
                    session.execute(stmt)
                session.commit()
                total_players += len(players_to_upsert)
                print(f"  ✓ {len(players_to_upsert)} players added for {season}")
        time.sleep(config.REQUEST_DELAY)

    print(f"\n✓ Roster ingestion complete: {total_players} records")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ingest NBA rosters.")
    parser.add_argument("--seasons", nargs="+", default=None, help="Specific seasons to process. Default: TEST_SEASONS. Use --full for all.")
    parser.add_argument("--full", action="store_true", help="Process all seasons defined in config.")
    args = parser.parse_args()

    if args.seasons:
        seasons = args.seasons
    elif args.full:
        seasons = config.ALL_SEASONS
    else:
        seasons = config.TEST_SEASONS

    ingest_rosters(seasons)
