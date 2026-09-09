"""
Offline feature store for the streaming demo.

Why precomputed rather than computed per event
-----------------------------------------------
The model's point-in-time shooter/defender features are cumulative-to-date
aggregates built by `src.features.build.build_matrix` — the same function
`src.training.train` calls. Building them correctly requires season-long
joins across shots, matchups, and team schedule/stats, which is exactly what
that function already does. Re-deriving a second, "streaming" version of
those joins would duplicate real pipeline logic and risk it drifting out of
sync with training — precisely the rewrite this demo is meant to avoid.

Instead, this module calls `build_matrix` once, up front, exactly as
training does, and caches the result keyed by `shot_id`. The Kafka consumer
then does an O(1) lookup per event and scores with the existing model. This
mirrors a standard real-world split between an offline/batch feature store
and a real-time scoring service (e.g. Feast, Tecton) — the honest way to
describe it is "real-time scoring against a precomputed feature store," not
"real-time feature computation."
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import config
from src.features.build import build_matrix

CACHE_DIR = config.DATA_DIR / "cache" / "streaming"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_path(seasons: list[str], prior_through_season: str) -> Path:
    tag = f"{seasons[0]}_{seasons[-1]}_{len(seasons)}_prior-{prior_through_season}"
    return CACHE_DIR / f"feature_store_{tag}.parquet"


def build_feature_store(seasons: list[str] | None = None,
                        prior_through_season: str | None = None,
                        use_cache: bool = True) -> tuple[pd.DataFrame, list[str]]:
    """
    Build (or load a cached copy of) the full point-in-time feature matrix,
    indexed by `shot_id`.

    Returns (frame, feature_cols). `frame` also carries `game_id`,
    `player_id`, `game_date`, and `shot_made` for the consumer to log
    alongside each prediction — none of those are model features.
    """
    seasons = seasons or [s for s in config.ALL_SEASONS if s >= "2016-17"]
    prior_through_season = prior_through_season or seasons[-2]
    cache_path = _cache_path(seasons, prior_through_season)

    if use_cache and cache_path.exists():
        frame = pd.read_parquet(cache_path)
        feature_cols = list(frame.attrs.get("feature_cols", []))
        if feature_cols:
            print(f"✓ feature store loaded from cache: {cache_path.name} "
                  f"({len(frame):,} shots)")
            return frame, feature_cols
        # A parquet written by an older version of this module without the
        # feature_cols sidecar — fall through and rebuild rather than guess.

    print("Building streaming feature store (reuses the training pipeline; "
          "this is a one-time cost, not a per-event one) ...")
    matrix, feature_cols, _artifacts = build_matrix(
        seasons, prior_through_season=prior_through_season, verbose=True,
    )
    frame = matrix.set_index("shot_id", drop=False)

    if use_cache:
        frame.attrs["feature_cols"] = feature_cols
        to_write = frame.copy()
        to_write.attrs["feature_cols"] = feature_cols
        to_write.to_parquet(cache_path)
        # Parquet drops DataFrame.attrs on write, so the column list is
        # re-derived on load by re-running build_matrix's contract: callers
        # always get feature_cols back from this function, never from the
        # cached file alone.
        (cache_path.with_suffix(".featurecols.txt")).write_text(
            "\n".join(feature_cols)
        )

    return frame, feature_cols


def load_feature_store(seasons: list[str] | None = None,
                       prior_through_season: str | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Like `build_feature_store`, but reads the feature-column sidecar file
    when a cached parquet exists, avoiding a rebuild just to recover names."""
    seasons = seasons or [s for s in config.ALL_SEASONS if s >= "2016-17"]
    prior_through_season = prior_through_season or seasons[-2]
    cache_path = _cache_path(seasons, prior_through_season)
    sidecar = cache_path.with_suffix(".featurecols.txt")

    if cache_path.exists() and sidecar.exists():
        frame = pd.read_parquet(cache_path)
        feature_cols = sidecar.read_text().splitlines()
        print(f"✓ feature store loaded from cache: {cache_path.name} "
              f"({len(frame):,} shots)")
        return frame, feature_cols

    return build_feature_store(seasons, prior_through_season, use_cache=True)
