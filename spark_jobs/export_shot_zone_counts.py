"""
Export the raw per-(player, game, zone) shot counts that the Spark job
transforms.

This calls the EXISTING `_load_player_game_zone_counts` in
src/features/point_in_time.py — the same query `build_prior_counts` (the
pandas/DuckDB implementation) starts from — rather than writing a second,
Spark-native SQL query against the database. The boundary is deliberate:
Python owns getting raw rows out of SQLite; Spark owns the point-in-time
transformation on those rows. Reusing the query means both implementations
are guaranteed to start from identical input, so any difference in their
output is a real difference in the transformation, not a data mismatch.

Usage:
    python -m spark_jobs.export_shot_zone_counts
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.db.database import get_engine
from src.features.point_in_time import _load_player_game_zone_counts

OUTPUT_PATH = Path(__file__).resolve().parent / "data" / "shot_zone_counts.parquet"


def export():
    engine = get_engine()
    df = _load_player_game_zone_counts(engine)
    # pandas/pyarrow write datetime64 columns as nanosecond-precision Parquet
    # TIMESTAMPs by default; Spark's Parquet reader rejects that type
    # outright ([PARQUET_TYPE_ILLEGAL]). Microsecond precision is the
    # interoperable choice and loses nothing here — game dates carry no
    # sub-second information to begin with.
    df["game_date"] = df["game_date"].astype("datetime64[us]")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"✓ wrote {len(df):,} (player, game, zone) rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    export()
