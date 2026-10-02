"""
Offline build of the "comparable shots" nearest-neighbor index used by
src/inference/explain/comparable_shots.py.

Run once after (re)training a shot-quality model, whenever the underlying
shot data changes materially:

    python -m src.training.build_comparable_index --model shot-quality-v20

Deliberately a separate, explicit step rather than something the API builds
lazily on first request: it scans the full shot history through DuckDB
(the same scan `build_matrix` does for training) and is too slow to hide
behind a request. The result is a single joblib artifact,
models/comparable_shots_<model-name>.joblib, loaded once at API startup.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import config
from src.inference.explain.comparable_shots import build_comparable_shots_index

MODEL_DIR = Path(config.PROJECT_ROOT) / "models"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="shot-quality-v20",
                        help="Model whose feature_cols/feature_bounds/seasons to match")
    parser.add_argument("--max-rows", type=int, default=250_000,
                        help="Cap on indexed shots (uniform sample) — keeps "
                             "k-NN query latency bounded as history grows")
    args = parser.parse_args()

    metadata_path = MODEL_DIR / f"metadata_{args.model}.json"
    metadata = json.loads(metadata_path.read_text())

    seasons = metadata["seasons"]
    feature_cols = metadata["feature_cols"]
    feature_bounds = metadata["feature_bounds"]
    # train.py fits build_matrix's priors through `val_season` (see
    # train.py's `_split_seasons`/`train`) — matching that exactly here means
    # the index's point-in-time features were shrunk toward the SAME priors
    # the model itself trained under, not a refit that would silently drift.
    prior_through_season = metadata["val_season"]

    print(f"Building comparable-shots index for {args.model} "
          f"({len(feature_cols)} features, seasons {seasons[0]}–{seasons[-1]}, "
          f"cap {args.max_rows:,} rows)...")
    start = time.time()
    index = build_comparable_shots_index(
        seasons=seasons,
        feature_cols=feature_cols,
        feature_bounds=feature_bounds,
        prior_through_season=prior_through_season,
        max_rows=args.max_rows,
    )
    out_path = MODEL_DIR / f"comparable_shots_{args.model}.joblib"
    index.save(out_path)
    elapsed = time.time() - start
    print(f"Indexed {len(index.meta):,} shots -> {out_path} ({elapsed:.0f}s)")


if __name__ == "__main__":
    main()
