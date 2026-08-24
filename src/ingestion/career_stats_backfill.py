"""
Career Stats Backfill — Computes career_fg_pct and career_3p_pct for every player.

Instead of calling an external API, this script derives career stats directly
from our own shots table. For each (player_id, season), career FG% is the
cumulative make rate across all seasons UP TO AND INCLUDING the current season.

This is more accurate than the NBA API's career endpoint because:
  1. It uses the exact same shot data our model trains on
  2. It's computed from 3.5M+ shots with zero API calls
  3. It's perfectly aligned with our season boundaries

Usage:
    python -m src.ingestion.career_stats_backfill
"""
import sys
from pathlib import Path
from collections import defaultdict

from sqlalchemy import text
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.db.database import get_engine


def backfill_career_stats():
    """Compute cumulative career FG% and 3P% from the shots table."""
    engine = get_engine()

    print(f"\n{'='*60}")
    print("  CAREER STATS BACKFILL")
    print(f"{'='*60}")

    # Step 1: Aggregate FGM/FGA and 3PM/3PA per player per season
    print("  → Aggregating shots by player-season...")

    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT 
                player_id,
                season,
                COUNT(*) as fga,
                SUM(shot_made) as fgm,
                SUM(CASE WHEN shot_type LIKE '3PT%' THEN 1 ELSE 0 END) as fg3a,
                SUM(CASE WHEN shot_type LIKE '3PT%' AND shot_made = 1 THEN 1 ELSE 0 END) as fg3m
            FROM shots
            GROUP BY player_id, season
            ORDER BY player_id, season
        """))

        # Build per-player-season aggregates
        # Structure: player_id -> [(season, fga, fgm, fg3a, fg3m), ...]
        player_data = defaultdict(list)
        for row in result:
            player_data[row[0]].append({
                "season": row[1],
                "fga": row[2],
                "fgm": row[3],
                "fg3a": row[4],
                "fg3m": row[5],
            })

    print(f"  ✓ Aggregated stats for {len(player_data):,} unique players")

    # Step 2: Compute cumulative career stats for each player-season
    print("  → Computing cumulative career FG% and 3P%...")

    updates = []  # (career_fg_pct, career_3p_pct, player_id, season)

    for player_id, seasons_data in player_data.items():
        # Seasons are already sorted by ORDER BY
        cumul_fgm = 0
        cumul_fga = 0
        cumul_fg3m = 0
        cumul_fg3a = 0

        for s in seasons_data:
            cumul_fgm += s["fgm"]
            cumul_fga += s["fga"]
            cumul_fg3m += s["fg3m"]
            cumul_fg3a += s["fg3a"]

            career_fg_pct = round(cumul_fgm / cumul_fga, 4) if cumul_fga > 0 else None
            career_3p_pct = round(cumul_fg3m / cumul_fg3a, 4) if cumul_fg3a > 0 else None

            updates.append({
                "cfg": career_fg_pct,
                "c3p": career_3p_pct,
                "pid": player_id,
                "ssn": s["season"],
            })

    print(f"  ✓ Computed {len(updates):,} career stat rows")

    # Step 3: Batch update the players table
    print("  → Updating players table...")

    total_updated = 0
    BATCH_SIZE = 500

    with engine.begin() as conn:
        for i in tqdm(range(0, len(updates), BATCH_SIZE), desc="  Updating"):
            batch = updates[i:i + BATCH_SIZE]
            for u in batch:
                result = conn.execute(
                    text("""
                        UPDATE players 
                        SET career_fg_pct = :cfg, career_3p_pct = :c3p
                        WHERE player_id = :pid AND season = :ssn
                    """),
                    u
                )
                total_updated += result.rowcount

    # Step 4: Verify
    with engine.connect() as conn:
        filled = conn.execute(text(
            "SELECT COUNT(*) FROM players WHERE career_fg_pct IS NOT NULL"
        )).scalar()
        total = conn.execute(text("SELECT COUNT(*) FROM players")).scalar()
        still_null = total - filled

    print(f"\n{'='*60}")
    print("  CAREER STATS BACKFILL COMPLETE")
    print(f"  Updated:     {total_updated:,} player-season rows")
    print(f"  Coverage:    {filled:,} / {total:,} ({filled/max(total,1)*100:.1f}%)")
    print(f"  Still NULL:  {still_null:,} (players with no shots in DB)")
    print(f"{'='*60}\n")

    return total_updated


if __name__ == "__main__":
    backfill_career_stats()
