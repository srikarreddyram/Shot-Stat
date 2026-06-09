"""
Shot chart data ingestion from nba_api.

Pulls ShotChartDetail for every player who played in each season.
Uses `shots_fetched_reg` and `shots_fetched_ply` flags in the Player table
for perfectly safe resuming.
"""
import sys
import time
import math
from pathlib import Path

from nba_api.stats.endpoints import shotchartdetail
from sqlalchemy import select, update
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from src.db.database import get_engine, get_session_factory
from src.db.models import Shot, Player

def _make_shot_id(game_id: str, game_event_id: int) -> str:
    return f"{game_id}_{game_event_id}"

def _time_remaining_seconds(minutes_remaining: int, seconds_remaining: int) -> float:
    return (minutes_remaining or 0) * 60 + (seconds_remaining or 0)

def _safe_float(val, default=None):
    if val is None: return default
    try: return float(val)
    except (ValueError, TypeError): return default

def _safe_int(val, default=None):
    if val is None: return default
    try: return int(val)
    except (ValueError, TypeError): return default

def _compute_shot_angle(loc_x: float, loc_y: float) -> float | None:
    if loc_x is None or loc_y is None: return None
    try: return math.degrees(math.atan2(loc_y, loc_x))
    except (ValueError, ZeroDivisionError): return None

def _determine_home_away(htm: str, vtm: str, team_name: str) -> int | None:
    if not team_name or (not htm and not vtm): return None
    if team_name == htm: return 1
    elif team_name == vtm: return 0
    return None

def _is_playoff_game(game_id: str) -> int:
    if game_id.startswith("004") or game_id.startswith("005"): return 1
    return 0

def _api_call_with_retry(player_id: int, season: str, season_type: str = "Regular Season", max_retries: int = None):
    if max_retries is None: max_retries = config.MAX_RETRIES
    backoff = config.INITIAL_BACKOFF
    for attempt in range(max_retries):
        try:
            chart = shotchartdetail.ShotChartDetail(
                team_id=0,
                player_id=player_id,
                season_nullable=season,
                season_type_all_star=season_type,
                context_measure_simple="FGA",
            )
            return chart.get_data_frames()[0]
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(backoff)
                backoff *= config.BACKOFF_MULTIPLIER
            else:
                return None
    return None

def ingest_shots(seasons: list[str]):
    engine = get_engine()
    Session = get_session_factory(engine)
    total_shots = 0
    total_players_processed = 0

    for season in seasons:
        print(f"\n{'='*60}\n  Shot ingestion for {season}\n{'='*60}")
        
        with Session() as session:
            reg_players = session.execute(
                select(Player.player_id, Player.name).where(Player.season == season)
            ).all()

        if not reg_players:
            print(f"  No players found in DB for {season}. Run roster ingestor first.")
            continue
            
        print(f"  Found {len(reg_players)} players for {season}")
        
        try:
            from nba_api.stats.endpoints import leaguedashplayerstats
            playoff_stats = leaguedashplayerstats.LeagueDashPlayerStats(
                season=season, season_type_all_star="Playoffs"
            )
            playoff_df = playoff_stats.get_data_frames()[0]
            playoff_player_ids = set(str(pid) for pid in playoff_df["PLAYER_ID"])
            ply_players = [p for p in reg_players if str(p.player_id) in playoff_player_ids]
        except Exception:
            ply_players = reg_players

        season_types = ["Regular Season", "Playoffs"]
        failed_players = []
        season_shots = 0

        for season_type in season_types:
            type_label = "REG" if season_type == "Regular Season" else "PLY"
            current_players = ply_players if season_type == "Playoffs" else reg_players
            
            with Session() as session:
                flag_col = Player.shots_fetched_reg if season_type == "Regular Season" else Player.shots_fetched_ply
                existing = session.execute(
                    select(Player.player_id).where(Player.season == season).where(flag_col == True)
                ).scalars().all()
                existing_set = set(str(p) for p in existing)

            print(f"\n  Pulling {season_type} shots... (Skipping {len(existing_set)} already done)")

            for player_id, player_name in tqdm(current_players, desc=f"  {season} {type_label}"):
                if str(player_id) in existing_set:
                    if season_type == "Regular Season": total_players_processed += 1
                    continue
                    
                df = _api_call_with_retry(int(player_id), season, season_type)

                if df is None:
                    failed_players.append((player_id, player_name, season_type))
                    continue

                shots_to_insert = []
                if not df.empty:
                    for _, row in df.iterrows():
                        game_id = row["GAME_ID"]
                        game_event_id = row["GAME_EVENT_ID"]
                        loc_x = _safe_float(row.get("LOC_X"))
                        loc_y = _safe_float(row.get("LOC_Y"))

                        shots_to_insert.append({
                            "shot_id": _make_shot_id(game_id, game_event_id),
                            "game_id": game_id,
                            "player_id": str(player_id),
                            "season": season,
                            "defender_id": None,
                            "shot_made": _safe_int(row.get("SHOT_MADE_FLAG", 0)),
                            "loc_x": loc_x,
                            "loc_y": loc_y,
                            "shot_distance": _safe_float(row.get("SHOT_DISTANCE")),
                            "shot_type": row.get("SHOT_TYPE"),
                            "zone": row.get("SHOT_ZONE_BASIC"),
                            "shot_angle": _compute_shot_angle(loc_x, loc_y),
                            "quarter": _safe_int(row.get("PERIOD")),
                            "time_remaining": _time_remaining_seconds(
                                _safe_int(row.get("MINUTES_REMAINING")),
                                _safe_int(row.get("SECONDS_REMAINING")),
                            ),
                            "score_diff": None,
                            "home_away": _determine_home_away(
                                row.get("HTM", ""), row.get("VTM", ""), row.get("TEAM_NAME", "")
                            ),
                            "playoff_flag": _is_playoff_game(game_id),
                            "touch_time": None, "dribbles": None,
                            "catch_and_shoot": None, "closest_defender_dist": None,
                        })

                with Session() as session:
                    if shots_to_insert:
                        # Fast bulk insert since we know we won't insert duplicates for a checked player
                        session.bulk_insert_mappings(Shot, shots_to_insert)
                        season_shots += len(shots_to_insert)
                    
                    # Flip the flag so we never check this player for this season type again
                    if season_type == "Regular Season":
                        stmt = update(Player).where(Player.player_id == str(player_id)).where(Player.season == season).values(shots_fetched_reg=True)
                    else:
                        stmt = update(Player).where(Player.player_id == str(player_id)).where(Player.season == season).values(shots_fetched_ply=True)
                    session.execute(stmt)
                    session.commit()

                if season_type == "Regular Season":
                    total_players_processed += 1
                time.sleep(config.REQUEST_DELAY)

        total_shots += season_shots
        print(f"\n  ✓ {season}: {season_shots} shots from {total_players_processed} players")
        if failed_players:
            print(f"  ⚠ {len(failed_players)} player-calls failed after retries")

    print(f"\n{'='*60}\n✓ Shot ingestion complete: {total_shots} total shots across all seasons\n{'='*60}")
    return total_shots

if __name__ == "__main__":
    seasons = config.TEST_SEASONS
    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        seasons = config.ALL_SEASONS
    ingest_shots(seasons)

