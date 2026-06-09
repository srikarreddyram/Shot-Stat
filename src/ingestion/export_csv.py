"""
Export database tables to CSV files for easy inspection.

Usage:
    python -m src.ingestion.export_csv              # Export all tables
    python -m src.ingestion.export_csv --table shots # Export a specific table
"""
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import select, text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from src.db.database import get_engine


# Output directory for CSVs
CSV_DIR = config.DATA_DIR / "csv"
CSV_DIR.mkdir(parents=True, exist_ok=True)


def export_table(table_name: str, engine=None):
    """Export a single table to CSV."""
    if engine is None:
        engine = get_engine()

    output_path = CSV_DIR / f"{table_name}.csv"

    df = pd.read_sql_table(table_name, engine)
    df.to_csv(output_path, index=False)
    print(f"  ✓ {table_name}: {len(df):,} rows → {output_path}")
    return df


def export_all():
    """Export all tables to CSV files."""
    engine = get_engine()

    print(f"\n{'='*60}")
    print(f"  EXPORTING DATABASE TO CSV")
    print(f"  Output: {CSV_DIR}/")
    print(f"{'='*60}\n")

    tables = ["games", "players", "shots"]
    for table in tables:
        try:
            export_table(table, engine)
        except Exception as e:
            print(f"  ✗ {table}: {e}")

    # Also export a summary/sample view for quick inspection
    print(f"\n  Creating summary views...")

    # Shot summary: first 100 rows with player name joined
    try:
        query = text("""
            SELECT
                s.shot_id,
                s.game_id,
                p.name AS player_name,
                s.player_id,
                s.season,
                s.shot_made,
                s.loc_x,
                s.loc_y,
                s.shot_distance,
                s.shot_type,
                s.zone,
                s.quarter,
                s.time_remaining,
                s.home_away,
                s.playoff_flag
            FROM shots s
            LEFT JOIN players p ON s.player_id = p.player_id AND s.season = p.season
            LIMIT 500
        """)
        df_sample = pd.read_sql(query, engine)
        sample_path = CSV_DIR / "shots_sample_500.csv"
        df_sample.to_csv(sample_path, index=False)
        print(f"  ✓ shots_sample_500: {len(df_sample)} rows → {sample_path}")
    except Exception as e:
        print(f"  ✗ shots_sample: {e}")

    # Zone summary: FG% by zone
    try:
        query = text("""
            SELECT
                zone,
                COUNT(*) as total_shots,
                SUM(shot_made) as makes,
                ROUND(AVG(shot_made) * 100, 1) as fg_pct,
                ROUND(AVG(shot_distance), 1) as avg_distance
            FROM shots
            GROUP BY zone
            ORDER BY total_shots DESC
        """)
        df_zones = pd.read_sql(query, engine)
        zone_path = CSV_DIR / "zone_summary.csv"
        df_zones.to_csv(zone_path, index=False)
        print(f"  ✓ zone_summary: {len(df_zones)} rows → {zone_path}")
    except Exception as e:
        print(f"  ✗ zone_summary: {e}")

    # Player summary: top 20 by shot volume
    try:
        query = text("""
            SELECT
                p.name,
                p.player_id,
                p.season,
                p.position,
                p.height,
                p.weight,
                p.wingspan,
                p.wingspan_source,
                p.season_fg_pct,
                COUNT(s.shot_id) as total_shots,
                SUM(s.shot_made) as makes,
                ROUND(AVG(s.shot_made) * 100, 1) as fg_pct
            FROM players p
            LEFT JOIN shots s ON p.player_id = s.player_id AND p.season = s.season
            GROUP BY p.player_id, p.season
            ORDER BY total_shots DESC
            LIMIT 50
        """)
        df_top = pd.read_sql(query, engine)
        top_path = CSV_DIR / "top_players_by_volume.csv"
        df_top.to_csv(top_path, index=False)
        print(f"  ✓ top_players_by_volume: {len(df_top)} rows → {top_path}")
    except Exception as e:
        print(f"  ✗ top_players: {e}")

    print(f"\n{'='*60}")
    print(f"  ✓ Export complete! CSVs are in: {CSV_DIR}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--table":
        export_table(sys.argv[2])
    else:
        export_all()
