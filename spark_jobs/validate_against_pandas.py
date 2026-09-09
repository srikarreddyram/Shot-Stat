"""
Correctness check: does the Spark job's strictly-prior cumulative logic
agree with the existing pandas implementation
(`src.features.point_in_time.build_prior_counts`) on real data?

This is the check that actually justifies calling the Spark version a
faithful reimplementation rather than a superficially similar one. It reads
both outputs (already computed — this script does not re-run either
pipeline) and compares career makes/attempts for a sample of real
(player_id, game_id) rows.

Usage:
    python -m src.training.train ... (not needed)
    python -m spark_jobs.export_shot_zone_counts
    python spark_jobs/point_in_time_features_spark.py
    python -m spark_jobs.validate_against_pandas
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.db.database import get_engine
from src.features.point_in_time import ZONE_SUFFIX, build_prior_counts

SPARK_OUTPUT = Path(__file__).resolve().parent / "data" / "point_in_time_features_spark.parquet"


def validate(n_sample_players: int = 200):
    print("Loading Spark output ...")
    spark_df = pd.read_parquet(SPARK_OUTPUT)

    print("Building pandas reference (src.features.point_in_time.build_prior_counts) ...")
    engine = get_engine()
    pandas_wide = build_prior_counts(engine)

    import numpy as np
    rng = np.random.default_rng(0)
    sample_players = rng.choice(
        pandas_wide["player_id"].unique(), size=n_sample_players, replace=False
    )

    mismatches = 0
    compared = 0
    for zone, suffix in ZONE_SUFFIX.items():
        pandas_zone = pandas_wide[pandas_wide["player_id"].isin(sample_players)][
            ["player_id", "game_id", f"pit_car_mk_{suffix}", f"pit_car_att_{suffix}"]
        ].rename(columns={f"pit_car_mk_{suffix}": "pandas_mk", f"pit_car_att_{suffix}": "pandas_att"})

        spark_zone = spark_df[
            (spark_df["zone"] == zone) & (spark_df["player_id"].isin(sample_players))
        ][["player_id", "game_id", "pit_car_mk", "pit_car_att"]].rename(
            columns={"pit_car_mk": "spark_mk", "pit_car_att": "spark_att"}
        )

        merged = pandas_zone.merge(spark_zone, on=["player_id", "game_id"], how="inner")
        compared += len(merged)
        bad = merged[
            (merged["pandas_mk"] != merged["spark_mk"])
            | (merged["pandas_att"] != merged["spark_att"])
        ]
        if len(bad) > 0:
            mismatches += len(bad)
            print(f"  ✗ {zone}: {len(bad)} mismatches out of {len(merged)} compared")
            print(bad.head(3))

    print(f"\nCompared {compared:,} (player, game, zone) rows across "
          f"{n_sample_players} sampled players.")
    if mismatches == 0:
        print("✓ Spark's strictly-prior cumulative makes/attempts EXACTLY MATCH "
              "the pandas pipeline on every compared row.")
    else:
        print(f"✗ {mismatches:,} mismatches found — see above.")


if __name__ == "__main__":
    validate()
