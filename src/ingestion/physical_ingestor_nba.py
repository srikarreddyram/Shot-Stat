"""
Phase 2A: NBA API Physical Data Collection
Iterates through every unique player missing physicals and fetches exact measurements from the NBA API.
"""
import sys
import time
import re
from pathlib import Path

from nba_api.stats.endpoints import commonplayerinfo, draftcombinestats
from sqlalchemy import select, update
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from src.db.database import get_engine, get_session_factory
from src.db.models import Player

def _safe_float(val, default=None):
    if val is None: return default
    try: return float(val)
    except (ValueError, TypeError): return default

def _height_to_inches(height_str: str) -> float | None:
    if not height_str: return None
    match = re.match(r"(\d+)-(\d+)", str(height_str))
    if match:
        return int(match.group(1)) * 12.0 + int(match.group(2))
    return None

def _get_wingspan_lookup() -> dict:
    print("  Building wingspan lookup from Draft Combine data...")
    wingspan_map = {}
    # Draft combine season strings run "2000-01" .. the most recent season we
    # track (ALL_SEASONS + CURRENT_SEASONS), so a newly-added season/draft
    # class is picked up automatically instead of this range going stale.
    latest_tracked = max(config.ALL_SEASONS + getattr(config, "CURRENT_SEASONS", []))
    end_year = int(latest_tracked.split("-")[0]) + 1
    for year in range(2000, end_year):
        season_str = f"{year}-{str(year + 1)[-2:]}"
        try:
            dc = draftcombinestats.DraftCombineStats(season_all_time=season_str)
            df = dc.get_data_frames()[0]
        except Exception:
            time.sleep(config.REQUEST_DELAY)
            continue
        if df.empty:
            continue
        for _, row in df.iterrows():
            player_id = str(row.get("PLAYER_ID", ""))
            wingspan = _safe_float(row.get("WINGSPAN"))
            if player_id and wingspan and wingspan > 0:
                wingspan_map[player_id] = wingspan
        time.sleep(config.REQUEST_DELAY)
    return wingspan_map

def _fetch_player_info_with_retry(player_id: str, max_retries=5) -> dict | None:
    backoff = config.INITIAL_BACKOFF
    for attempt in range(max_retries):
        try:
            info = commonplayerinfo.CommonPlayerInfo(player_id=int(player_id), timeout=15)
            df = info.get_data_frames()[0]
            if not df.empty:
                return df.iloc[0].to_dict()
            return None
        except Exception:
            if attempt == max_retries - 1: return None
            time.sleep(backoff)
            backoff *= config.BACKOFF_MULTIPLIER
    return None

def ingest_physicals_nba(season_scope: str | None = None):
    """
    season_scope: if given, only check players who have a Players row for
    that season (e.g. this year's roster) instead of scanning every player
    in the database. Without it, this scans the full historical table, which
    is expensive and mostly re-confirms known gaps (wingspan is genuinely
    unmeasured for many older players — see docs/prd_checklist.md's known
    risks — not something a rerun will fix).
    """
    engine = get_engine()
    Session = get_session_factory(engine)

    wingspan_map = _get_wingspan_lookup()

    with Session() as session:
        query = select(Player.player_id).where(
            (Player.height.is_(None)) |
            (Player.weight.is_(None)) |
            (Player.wingspan.is_(None))
        ).distinct()
        if season_scope:
            query = query.where(
                Player.player_id.in_(
                    select(Player.player_id).where(Player.season == season_scope)
                )
            )
        missing_players = session.execute(query).scalars().all()

        print(f"\nFound {len(missing_players)} unique players needing physical data checks...")
        
        updated_count = 0
        for pid in tqdm(missing_players, desc="NBA API Physicals"):
            info = _fetch_player_info_with_retry(pid)
            
            height = None
            weight = None
            if info:
                height = _height_to_inches(info.get("HEIGHT"))
                weight = _safe_float(info.get("WEIGHT"))
                
            wingspan = wingspan_map.get(str(pid))
            wingspan_source = "NBA_API" if wingspan else None
            
            if height or weight or wingspan:
                stmt = update(Player).where(Player.player_id == pid).values(
                    height=height,
                    weight=weight,
                    wingspan=wingspan,
                    wingspan_source=wingspan_source
                )
                session.execute(stmt)
                updated_count += 1
                
            time.sleep(config.REQUEST_DELAY)
            
        session.commit()
        print(f"\n✓ Checked {len(missing_players)} players, applied data to {updated_count}.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Backfill height/weight/wingspan from the NBA API.")
    parser.add_argument("--season", type=str, default=None,
                         help="Only check players rostered for this season (e.g. 2026-27), instead of the full history.")
    args = parser.parse_args()
    ingest_physicals_nba(season_scope=args.season)
