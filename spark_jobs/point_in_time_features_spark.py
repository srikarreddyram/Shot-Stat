"""
PySpark reimplementation of the point-in-time shooter-rate pipeline.

Read this before anything else: this dataset (2.1M shots, 1.3M
(player, game, zone) rows) does NOT need distributed processing. It already
runs fine — in seconds — in the existing pandas/DuckDB pipeline
(`src.features.point_in_time.build_prior_counts`). This file exists to
demonstrate the Spark DataFrame API (window functions, cross joins,
partitioned aggregation) applied faithfully to a real, already-solved
problem from this project, not because Spark was load-driven. See
spark_jobs/README.md for the honest framing — the same thing belongs in any
resume bullet built from this file.

What it reproduces
-------------------
`build_prior_counts`'s core trick, translated into Spark's Window API:
for each (player, zone), a STRICTLY PRIOR running total of makes/attempts —
cumulative through the game before this one, never including the current
game's own shots. In pandas this is `cumsum() - current_row`; here it is a
window frame from unbounded-preceding to the current row, again with the
current row's own contribution subtracted off. Same trick, different engine.

It also applies empirical-Bayes shrinkage (see src/features/shrinkage.py for
the original): `p_hat = (makes + k*prior_mean) / (attempts + k)`, with a
per-zone prior mean and concentration k fit from the same data via a Spark
aggregation — a simplified method-of-moments estimate, not a byte-for-byte
port of shrinkage.py's fitting code (which additionally trims low-attempt
players before estimating k; this version does not, and says so rather than
silently claiming parity).

Usage:
    python -m spark_jobs.export_shot_zone_counts     # once, or after new data
    python spark_jobs/point_in_time_features_spark.py
"""
import time
from pathlib import Path

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

DATA_DIR = Path(__file__).resolve().parent / "data"
INPUT_PATH = DATA_DIR / "shot_zone_counts.parquet"
OUTPUT_PATH = DATA_DIR / "point_in_time_features_spark.parquet"

ZONES = [
    "Restricted Area", "In The Paint (Non-RA)", "Mid-Range",
    "Left Corner 3", "Right Corner 3", "Above the Break 3",
]


def build_point_in_time_features(spark: SparkSession):
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"{INPUT_PATH} not found — run "
            f"'python -m spark_jobs.export_shot_zone_counts' first."
        )

    counts = spark.read.parquet(str(INPUT_PATH))

    # Every (player, game) that appears at all, crossed with the fixed list
    # of zones — mirrors build_prior_counts's cross-join. A player who took
    # no mid-range shot in a game still needs his running mid-range total
    # carried forward at that game, or the cumulative sum only advances on
    # games where he happened to shoot from that specific zone.
    player_games = counts.select("player_id", "game_id", "season", "game_date").distinct()
    zones_df = spark.createDataFrame([(z,) for z in ZONES], ["zone"])
    grid = player_games.crossJoin(zones_df)

    grid = grid.join(
        counts.select("player_id", "game_id", "zone", "makes", "attempts"),
        on=["player_id", "game_id", "zone"], how="left",
    ).fillna({"makes": 0, "attempts": 0})

    # Strictly-prior cumulative sums via window functions. The frame runs to
    # CURRENT ROW (inclusive) and then the current row's own makes/attempts
    # are subtracted — the Spark-native way to express "cumsum() - self",
    # since Spark's window frames don't have pandas' shift(1)-after-cumsum
    # shorthand.
    career_window = (
        Window.partitionBy("player_id", "zone")
        .orderBy("game_date", "game_id")
        .rowsBetween(Window.unboundedPreceding, Window.currentRow)
    )
    season_window = (
        Window.partitionBy("player_id", "zone", "season")
        .orderBy("game_date", "game_id")
        .rowsBetween(Window.unboundedPreceding, Window.currentRow)
    )

    grid = (
        grid
        .withColumn("pit_car_mk", F.sum("makes").over(career_window) - F.col("makes"))
        .withColumn("pit_car_att", F.sum("attempts").over(career_window) - F.col("attempts"))
        .withColumn("pit_ssn_mk", F.sum("makes").over(season_window) - F.col("makes"))
        .withColumn("pit_ssn_att", F.sum("attempts").over(season_window) - F.col("attempts"))
    )

    # ── Empirical-Bayes shrinkage, fit per zone ────────────────────────────
    # Simplified method-of-moments: prior mean is the league-wide zone rate;
    # k (prior strength, in equivalent attempts) is derived from the
    # between-player variance of each player's OWN overall rate in that zone
    # — a noisier zone (more true talent spread) gets a smaller k and is
    # shrunk less. Real analogue of shrinkage.py's fit_priors, computed here
    # as a genuinely distributed groupBy/agg rather than pandas.
    per_player_zone = (
        counts.groupBy("player_id", "zone")
        .agg(F.sum("makes").alias("career_mk"), F.sum("attempts").alias("career_att"))
        .filter(F.col("career_att") >= 10)
        .withColumn("rate", F.col("career_mk") / F.col("career_att"))
    )
    zone_priors = (
        per_player_zone.groupBy("zone")
        .agg(F.mean("rate").alias("prior_mean"), F.variance("rate").alias("rate_var"))
        .withColumn(
            "k",
            F.when(
                F.col("rate_var") > 0,
                F.greatest(F.lit(5.0), F.least(F.lit(500.0),
                    (F.col("prior_mean") * (1 - F.col("prior_mean")) / F.col("rate_var")) - 1
                )),
            ).otherwise(F.lit(50.0)),
        )
        .select("zone", "prior_mean", "k")
    )

    grid = grid.join(zone_priors, on="zone", how="left")
    grid = grid.withColumn(
        "pit_car_rate_shrunk",
        (F.col("pit_car_mk") + F.col("k") * F.col("prior_mean"))
        / (F.col("pit_car_att") + F.col("k")),
    )

    return grid, zone_priors


def main():
    spark = (
        SparkSession.builder
        .appName("nba-point-in-time-features")
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    t0 = time.time()
    result, zone_priors = build_point_in_time_features(spark)

    n_rows = result.count()
    elapsed = time.time() - t0
    print(f"\n✓ {n_rows:,} (player, game, zone) point-in-time rows "
          f"in {elapsed:.1f}s (local[*], driver-only)")

    print("\nFitted per-zone shrinkage priors:")
    zone_priors.orderBy("zone").show(truncate=False)

    print("Sample output (one player, chronological):")
    result.filter(F.col("zone") == "Restricted Area") \
        .orderBy("player_id", "game_date") \
        .select("player_id", "game_date", "pit_car_mk", "pit_car_att", "pit_car_rate_shrunk") \
        .show(10, truncate=False)

    result.write.mode("overwrite").parquet(str(OUTPUT_PATH))
    print(f"✓ wrote result to {OUTPUT_PATH}")

    spark.stop()


if __name__ == "__main__":
    main()
