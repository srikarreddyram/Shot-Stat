"""
Defender Stats Ingestor — Pulls season-level defensive metrics.

Uses LeagueDashPtDefend to get how good each player is at defending
different shot zones. 6 categories × 16 seasons × 2 season types = 192 calls.

Usage:
    python -m src.ingestion.defender_stats_ingestor --full
    python -m src.ingestion.defender_stats_ingestor --seasons 2023-24
"""
import sys
import time
from pathlib import Path

from nba_api.stats.endpoints import leaguedashptdefend
from sqlalchemy.dialects.sqlite import insert as sqlite_upsert
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from src.db.database import get_engine, get_session_factory
from src.db.models import DefenderStats

DEFENSE_CATEGORIES = [
    "Overall",
    "3 Pointers",
    "2 Pointers",
    "Less Than 6Ft",
    "Less Than 10Ft",
    "Greater Than 15Ft",
]

SEASON_TYPES = ["Regular Season", "Playoffs"]


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


def ingest_defender_stats(seasons: list[str]):
    """Pull defensive metrics for all players across all defense categories."""
    engine = get_engine()
    Session = get_session_factory(engine)

    total_calls = len(seasons) * len(SEASON_TYPES) * len(DEFENSE_CATEGORIES)
    print(f"\n{'='*60}")
    print("  DEFENDER STATS INGESTOR")
    print(f"  {len(seasons)} seasons × {len(SEASON_TYPES)} types × {len(DEFENSE_CATEGORIES)} categories = {total_calls} API calls")
    print(f"{'='*60}")

    total_upserted = 0
    total_errors = 0

    for season in seasons:
        season_count = 0

        for season_type in SEASON_TYPES:
            for category in tqdm(
                DEFENSE_CATEGORIES,
                desc=f"  {season} {season_type[:3]}",
                leave=False,
            ):
                try:
                    data = leaguedashptdefend.LeagueDashPtDefend(
                        season=season,
                        defense_category=category,
                        season_type_all_star=season_type,
                    )
                    df = data.get_data_frames()[0]
                except Exception as e:
                    total_errors += 1
                    if total_errors <= 5:
                        tqdm.write(f"  ⚠ {season} {season_type} {category}: {e}")
                    time.sleep(config.REQUEST_DELAY * 2)
                    continue

                if df.empty:
                    time.sleep(config.REQUEST_DELAY)
                    continue

                rows_to_upsert = []
                
                # Each category uses different column names in the API
                COL_MAP = {
                    "Overall":          {"fgm": "D_FGM",     "fga": "D_FGA",     "pct": "D_FG_PCT",   "normal": "NORMAL_FG_PCT", "pm": "PCT_PLUSMINUS"},
                    "3 Pointers":       {"fgm": "FG3M",      "fga": "FG3A",      "pct": "FG3_PCT",    "normal": "NS_FG3_PCT",    "pm": "PLUSMINUS"},
                    "2 Pointers":       {"fgm": "FG2M",      "fga": "FG2A",      "pct": "FG2_PCT",    "normal": "NS_FG2_PCT",    "pm": "PLUSMINUS"},
                    "Less Than 6Ft":    {"fgm": "FGM_LT_06", "fga": "FGA_LT_06", "pct": "LT_06_PCT",  "normal": "NS_LT_06_PCT",  "pm": "PLUSMINUS"},
                    "Less Than 10Ft":   {"fgm": "FGM_LT_10", "fga": "FGA_LT_10", "pct": "LT_10_PCT",  "normal": "NS_LT_10_PCT",  "pm": "PLUSMINUS"},
                    "Greater Than 15Ft":{"fgm": "FGM_GT_15", "fga": "FGA_GT_15", "pct": "GT_15_PCT",  "normal": "NS_GT_15_PCT",  "pm": "PLUSMINUS"},
                }
                cm = COL_MAP.get(category, COL_MAP["Overall"])
                
                for _, row in df.iterrows():
                    rows_to_upsert.append({
                        "player_id": str(row["CLOSE_DEF_PERSON_ID"]),
                        "season": season,
                        "defense_category": category,
                        "season_type": season_type,
                        "gp": _safe_int(row.get("GP")),
                        "freq": _safe_float(row.get("FREQ")),
                        "d_fgm": _safe_int(row.get(cm["fgm"])),
                        "d_fga": _safe_int(row.get(cm["fga"])),
                        "d_fg_pct": _safe_float(row.get(cm["pct"])),
                        "normal_fg_pct": _safe_float(row.get(cm["normal"])),
                        "pct_plusminus": _safe_float(row.get(cm["pm"])),
                    })

                if rows_to_upsert:
                    with Session() as session:
                        for r in rows_to_upsert:
                            stmt = sqlite_upsert(DefenderStats.__table__).values(**r)
                            stmt = stmt.on_conflict_do_update(
                                # season_type is part of the key: without it the
                                # playoff pass overwrites the regular-season row
                                # for every player whose team made the postseason.
                                index_elements=["player_id", "season", "defense_category", "season_type"],
                                set_={
                                    "gp": stmt.excluded.gp,
                                    "freq": stmt.excluded.freq,
                                    "d_fgm": stmt.excluded.d_fgm,
                                    "d_fga": stmt.excluded.d_fga,
                                    "d_fg_pct": stmt.excluded.d_fg_pct,
                                    "normal_fg_pct": stmt.excluded.normal_fg_pct,
                                    "pct_plusminus": stmt.excluded.pct_plusminus,
                                },
                            )
                            session.execute(stmt)
                        session.commit()
                        season_count += len(rows_to_upsert)

                time.sleep(config.REQUEST_DELAY)

        total_upserted += season_count
        print(f"  ✓ {season}: {season_count:,} defender stat rows")

    print(f"\n{'='*60}")
    print("  DEFENDER STATS INGESTOR COMPLETE")
    print(f"  Total upserted: {total_upserted:,}")
    if total_errors:
        print(f"  Errors: {total_errors}")
    print(f"{'='*60}\n")

    return total_upserted


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ingest defender stats.")
    parser.add_argument("--seasons", nargs="+", default=None)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()

    if args.seasons:
        seasons = args.seasons
    elif args.full:
        seasons = config.ALL_SEASONS
    else:
        seasons = config.TEST_SEASONS

    ingest_defender_stats(seasons)
