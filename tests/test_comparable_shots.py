"""
Tests for src/inference/explain/comparable_shots.py.

`build_comparable_shots_index` needs the full DB (it calls `build_matrix`,
the same heavy DuckDB scan training uses), so these tests exercise
`ComparableShotsIndex.query` directly against a small, hand-built index —
the part with actual retrieval logic to get wrong (self-exclusion, historical
make-rate averaging, behaving sanely with fewer shots than k) — rather than
re-testing `build_matrix` itself, which already has its own test coverage.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.neighbors import NearestNeighbors

from src.inference.explain.comparable_shots import ComparableShotsIndex

FEATURE_COLS = ["zone_rate", "def_fg_pct_zone"]


def _make_index(rows: list[dict]) -> ComparableShotsIndex:
    columns = ["shot_id", "player_id", "player_name", "game_date", "zone",
               "def_team", "shot_made", *FEATURE_COLS]
    meta = pd.DataFrame(rows, columns=columns)
    X = meta[FEATURE_COLS].to_numpy(dtype="float32")
    scale = np.ones(len(FEATURE_COLS), dtype="float32")
    nn = NearestNeighbors(metric="euclidean").fit(X / scale)
    return ComparableShotsIndex(feature_cols=FEATURE_COLS, scale=scale, meta=meta, nn=nn)


def _row(shot_id, zone_rate, def_fg, made, name="Player", zone="Mid-Range", game_date="2023-01-01", def_team="BOS"):
    return {
        "shot_id": shot_id, "player_id": "p1", "player_name": name,
        "game_date": pd.Timestamp(game_date), "zone": zone, "def_team": def_team,
        "shot_made": made, "zone_rate": zone_rate, "def_fg_pct_zone": def_fg,
    }


def _query_row(zone_rate, def_fg):
    return pd.DataFrame([{"zone_rate": zone_rate, "def_fg_pct_zone": def_fg}])


def test_query_finds_the_closest_neighbors_by_feature_distance():
    index = _make_index([
        _row("s1", 0.50, 0.45, made=1, name="Close A"),
        _row("s2", 0.51, 0.46, made=1, name="Close B"),
        _row("s3", 0.90, 0.90, made=0, name="Far"),
    ])
    result = index.query(_query_row(0.50, 0.45), k=2)
    names = {ex["player_name"] for ex in result["examples"]}
    assert names == {"Close A", "Close B"}
    assert "Far" not in names


def test_historical_make_rate_is_the_mean_of_the_returned_neighbors_not_the_whole_index():
    index = _make_index([
        _row("s1", 0.50, 0.45, made=1),
        _row("s2", 0.50, 0.45, made=1),
        _row("s3", 0.50, 0.45, made=0),
        _row("s4", 0.99, 0.99, made=0),  # far away — must not affect the rate for k=3
    ])
    result = index.query(_query_row(0.50, 0.45), k=3)
    assert result["count"] == 3
    assert result["historical_make_rate"] == pytest.approx(2 / 3, abs=1e-3)


def test_exclude_shot_id_removes_the_shot_being_explained_from_its_own_comparables():
    # All three share an identical feature vector, so "target" is its own
    # nearest possible match — if exclusion didn't work, a shot would
    # trivially report itself back as its own best comparable, which is
    # meaningless as a sanity check.
    index = _make_index([
        _row("target", 0.50, 0.45, made=1, name="Target"),
        _row("s2", 0.50, 0.45, made=0, name="Neighbor 2"),
        _row("s3", 0.50, 0.45, made=0, name="Neighbor 3"),
    ])
    result = index.query(_query_row(0.50, 0.45), k=2, exclude_shot_id="target")
    names = {ex["player_name"] for ex in result["examples"]}
    assert result["count"] == 2
    assert names == {"Neighbor 2", "Neighbor 3"}


def test_query_does_not_crash_when_fewer_shots_exist_than_k():
    index = _make_index([_row("s1", 0.5, 0.5, made=1)])
    result = index.query(_query_row(0.5, 0.5), k=8)
    assert result["count"] == 1
    assert result["historical_make_rate"] == 1.0


def test_save_and_load_round_trip_preserves_query_behavior(tmp_path: Path):
    index = _make_index([
        _row("s1", 0.50, 0.45, made=1, name="A"),
        _row("s2", 0.90, 0.90, made=0, name="B"),
    ])
    path = tmp_path / "index.joblib"
    index.save(path)
    loaded = ComparableShotsIndex.load(path)

    result = loaded.query(_query_row(0.50, 0.45), k=1)
    assert result["examples"][0]["player_name"] == "A"
