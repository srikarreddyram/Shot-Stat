"""
Zone-Level Shooting Efficiency Ingestor

Populates the `player_zone_stats` table with per-zone FG% for every
player-season using the NBA API's PlayerDashboardByShootingSplits endpoint.

Zone breakdown (from the NBA API "Shot Area" DataFrame):
    - Restricted Area         ← rim finishing (inside the painted circle)
    - In The Paint (Non-RA)   ← paint outside the circle
    - Mid-Range               ← all mid-range jumpers (Shaun Livingston territory)
    - Left Corner 3
    - Right Corner 3
    - Above the Break 3       ← above-the-break triples (Steph Curry territory)

Usage:
    python -m src.ingestion.zone_stats_ingestor
"""

import sys
import time
from pathlib import Path

from nba_api.stats.endpoints import playerdashboardbyshootingsplits
from sqlalchemy import select, func
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.db.database import get_engine, get_session_factory
from src.db.models import Player, PlayerZoneStats

# ── Configuration ──────────────────────────────────────────────────────────
# The NBA API "Shot Area" breakdown is DataFrame index 3 returned by the endpoint
SHOT_AREA_DF_INDEX = 3

# Zones we care about — exactly as they appear in GROUP_VALUE
VALID_ZONES = {
    "Restricted Area",
    "In The Paint (Non-RA)",
    "Mid-Range",
    "Left Corner 3",
    "Right Corner 3",
    "Above the Break 3",
}

# Minimum FGA threshold — skip rows with 0 attempts (not meaningful)
MIN_FGA = 0

# Polite rate limiting — NBA API is strict
REQUEST_DELAY = 1.2   # seconds between players
RETRY_DELAY   = 10.0  # seconds after a rate-limit 429
MAX_RETRIES   = 3


def _fetch_zone_splits(player_id: str, season: str, season_type: str = "Regular Season") -> list[dict] | None:
    """
    Call PlayerDashboardByShootingSplits and return a list of zone-row dicts.
    Returns None on unrecoverable error.
    """
    for attempt in range(MAX_RETRIES):
        try:
            result = playerdashboardbyshootingsplits.PlayerDashboardByShootingSplits(
                player_id=player_id,
                season=season,
                season_type_playoffs=season_type,
                timeout=30,
            )
            dfs = result.get_data_frames()

            # DataFrame 3 is the "Shot Area" breakdown
            if len(dfs) <= SHOT_AREA_DF_INDEX:
                return None

            df = dfs[SHOT_AREA_DF_INDEX]
            if df.empty or "GROUP_VALUE" not in df.columns:
                return None

            rows = []
            for _, row in df.iterrows():
                zone = row.get("GROUP_VALUE", "")
                if zone not in VALID_ZONES:
                    continue

                fga = int(row.get("FGA", 0) or 0)
                rows.append({
                    "player_id": str(player_id),
                    "season":    season,
                    "zone":      zone,
                    "fgm":       int(row.get("FGM", 0) or 0),
                    "fga":       fga,
                    "fg_pct":    float(row.get("FG_PCT", 0.0) or 0.0),
                    "fg3m":      int(row.get("FG3M", 0) or 0),
                    "fg3a":      int(row.get("FG3A", 0) or 0),
                    "fg3_pct":   float(row.get("FG3_PCT", 0.0) or 0.0),
                })
            return rows

        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "rate" in err_str:
                time.sleep(RETRY_DELAY * (attempt + 1))
            elif attempt < MAX_RETRIES - 1:
                time.sleep(REQUEST_DELAY * 2)
            else:
                return None

    return None


def _upsert_zone_rows(session, rows: list[dict]) -> int:
    """
    INSERT OR REPLACE zone stat rows into player_zone_stats.
    Returns the number of rows upserted.
    """
    if not rows:
        return 0

    stmt = sqlite_insert(PlayerZoneStats).values(rows)
    stmt = stmt.on_conflict_do_update(
        index_elements=["player_id", "season", "zone"],
        set_={
            "fgm":     stmt.excluded.fgm,
            "fga":     stmt.excluded.fga,
            "fg_pct":  stmt.excluded.fg_pct,
            "fg3m":    stmt.excluded.fg3m,
            "fg3a":    stmt.excluded.fg3a,
            "fg3_pct": stmt.excluded.fg3_pct,
        },
    )
    session.execute(stmt)
    return len(rows)


def ingest_zone_stats(seasons: list[str] | None = None, season_type: str = "Regular Season"):
    """
    Main entry point. Fetches zone-level shooting splits for every unique
    player-season in the Players table and upserts into PlayerZoneStats.

    Args:
        seasons:     If given, only process these seasons. Otherwise all seasons.
        season_type: "Regular Season" or "Playoffs".
    """
    engine  = get_engine()
    Session = get_session_factory(engine)

    # ── Create table if it doesn't exist yet ──────────────────────────────
    from src.db.models import Base
    Base.metadata.create_all(engine)

    print("\n" + "=" * 60)
    print("  Zone Stats Ingestion")
    print("=" * 60)

    with Session() as session:
        # Find all (player_id, season) pairs from the Players table
        q = select(Player.player_id, Player.season, Player.name).distinct()
        if seasons:
            q = q.where(Player.season.in_(seasons))
        all_players = session.execute(q).all()

        # Find which (player_id, season) pairs already have zone stats
        existing_q = select(
            PlayerZoneStats.player_id,
            PlayerZoneStats.season,
        ).distinct()
        existing = {
            (row.player_id, row.season)
            for row in session.execute(existing_q).all()
        }

    # Filter to only players missing zone stats
    todo = [(pid, season, name) for pid, season, name in all_players
            if (pid, season) not in existing]

    print(f"\n  Total player-seasons in DB:     {len(all_players):,}")
    print(f"  Already have zone stats:        {len(existing):,}")
    print(f"  To fetch:                       {len(todo):,}")

    if not todo:
        print("\n  ✓ All player-seasons already have zone stats. Nothing to do.")
        return

    # Estimate time
    est_mins = len(todo) * REQUEST_DELAY / 60
    print(f"  Estimated time:                 ~{est_mins:.0f} minutes\n")

    total_rows    = 0
    error_count   = 0
    error_samples = []

    with Session() as session:
        for pid, season, name in tqdm(todo, desc="  Zone Stats"):
            rows = _fetch_zone_splits(pid, season, season_type)

            if rows is None:
                error_count += 1
                if len(error_samples) < 10:
                    error_samples.append(f"{name} ({season})")
            else:
                n = _upsert_zone_rows(session, rows)
                total_rows += n

            time.sleep(REQUEST_DELAY)

            # Commit in batches of 100 players to avoid long open transactions
            if total_rows > 0 and total_rows % (100 * 6) == 0:
                session.commit()

        session.commit()

    print(f"\n  ✓ Inserted/updated {total_rows:,} zone-stat rows across "
          f"{len(todo) - error_count:,} player-seasons.")
    if error_count:
        print(f"  ⚠  {error_count} player-seasons failed (API error / no data).")
        if error_samples:
            print(f"     Sample: {', '.join(error_samples)}")

    # ── Quick sanity report ───────────────────────────────────────────────
    _print_zone_report()


def _print_zone_report():
    """Print a quick zone coverage summary after ingestion."""
    engine  = get_engine()
    Session = get_session_factory(engine)

    print("\n  📊 Zone Coverage Summary:")
    with Session() as session:
        result = session.execute(
            select(
                PlayerZoneStats.zone,
                func.count().label("n_rows"),
                func.avg(PlayerZoneStats.fg_pct).label("avg_fg_pct"),
                func.avg(PlayerZoneStats.fga).label("avg_fga"),
            )
            .group_by(PlayerZoneStats.zone)
            .order_by(func.count().desc())
        ).all()

        print(f"\n     {'Zone':<30} {'Rows':>7} {'Avg FGA':>8} {'Avg FG%':>8}")
        print(f"     {'-'*30} {'-'*7} {'-'*8} {'-'*8}")
        for zone, n, avg_pct, avg_fga in result:
            print(f"     {zone:<30} {n:>7,} {avg_fga:>8.1f} {avg_pct*100:>7.1f}%")

    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest zone-level shooting stats.")
    parser.add_argument(
        "--seasons", nargs="+", default=None,
        help="Specific seasons to process, e.g. 2023-24 2024-25. Default: all."
    )
    parser.add_argument(
        "--type", choices=["Regular Season", "Playoffs"], default="Regular Season",
        dest="season_type",
        help="Season type. Default: Regular Season."
    )
    args = parser.parse_args()
    ingest_zone_stats(seasons=args.seasons, season_type=args.season_type)
