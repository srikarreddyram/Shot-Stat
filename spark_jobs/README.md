# PySpark point-in-time feature pipeline

A PySpark reimplementation of the project's point-in-time shooter-rate
aggregation, runnable standalone in local mode (`local[*]`, no cluster).

## Why this exists — read this before the resume bullet

**This dataset does not need distributed processing.** 2.1M shots and 1.3M
(player, game, zone) rows already run through the existing pandas/DuckDB
pipeline (`src.features.point_in_time.build_prior_counts`) in well under a
minute. This directory exists to demonstrate the PySpark DataFrame API —
window functions, cross joins, partitioned aggregation — applied faithfully
to a real, already-solved problem from this project, **not because Spark was
load-driven.** If asked in an interview: the honest answer is "I built this
to show I can write the window-function/partition patterns Spark work
actually requires, not because this data needed it."

## What it does

Reproduces `build_prior_counts`'s core trick — for each (player, zone), a
STRICTLY PRIOR running total of makes and attempts, never including the
shot(s) from the game being scored — using Spark's `Window` API instead of
pandas' `cumsum().shift()`. It also fits and applies empirical-Bayes
shrinkage (`p_hat = (makes + k·prior_mean) / (attempts + k)`), with the prior
mean and concentration `k` estimated per zone via a genuine distributed
`groupBy`/`agg`, not a port of `shrinkage.py`'s exact fitting code (that
version additionally trims low-attempt players before estimating `k`; this
one doesn't, and the module docstring says so).

## Verified correct, not just similar

`validate_against_pandas.py` compares Spark's career makes/attempts against
the pandas pipeline's for a random sample of real players. Result:

```
Compared 261,750 (player, game, zone) rows across 200 sampled players.
✓ Spark's strictly-prior cumulative makes/attempts EXACTLY MATCH
  the pandas pipeline on every compared row.
```

Measured throughput: 2,464,212 output rows in 5.3s on `local[*]` (Apple
Silicon, driver-only — no worker cluster).

## Environment note

PySpark's Python worker subprocess must run the same Python that PySpark
itself is installed under, or you'll hit
`TypeError: unsupported operand type(s) for |: 'type' and 'type'` (PySpark's
own code uses `X | Y` union-type syntax requiring Python ≥3.10, and without
`PYSPARK_PYTHON` set, Spark can silently fall back to whatever `python3`
resolves to on `PATH` — on this machine that was a system Python 3.9).
Set explicitly:

```bash
export PYSPARK_PYTHON=$(which python)
export PYSPARK_DRIVER_PYTHON=$(which python)
```

This project's PySpark install also lives in its own conda env
(`conda create -n nba-spark python=3.11 && conda install -n nba-spark -c
conda-forge pyspark openjdk=17`) rather than the main project environment —
installing PySpark from PyPI directly pulled in a 450MB source tarball that
repeatedly stalled on this network; conda-forge's prebuilt package, bundled
with a matched JDK 17, installed cleanly and sidesteps any host Java version
question entirely.

## Running it

```bash
conda activate nba-spark
export PYSPARK_PYTHON=$(which python)
export PYSPARK_DRIVER_PYTHON=$(which python)

python -m spark_jobs.export_shot_zone_counts      # once, or after new data — writes data/shot_zone_counts.parquet
python spark_jobs/point_in_time_features_spark.py  # the actual Spark job
python -m spark_jobs.validate_against_pandas       # optional correctness check
```
