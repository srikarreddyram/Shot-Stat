"""
Tests for src/training/feature_engineering.py.

build_training_matrix() calls get_engine() internally rather than accepting an
engine parameter, so these tests monkeypatch the module-level get_engine to
point at the seeded in-memory DB instead of the real config.SQLALCHEMY_DATABASE_URL.
"""
import numpy as np
import pytest

import src.training.feature_engineering as fe


@pytest.fixture()
def matrix(seeded_db, monkeypatch):
    monkeypatch.setattr(fe, "get_engine", lambda: seeded_db)
    return fe.build_training_matrix(["2023-24"], use_defender=True)


def _row_for(df, shot_id):
    match = df[df["shot_id"] == shot_id]
    assert len(match) == 1, f"expected exactly one row for {shot_id}"
    return match.iloc[0]


def test_matrix_has_expected_row_count(matrix):
    # 5 shots seeded, all pass the WHERE clause (score_diff/home_away/zone set).
    assert len(matrix) == 5


def test_zone_efficiency_matches_shooters_own_zone(matrix):
    # P_TALL shooting from Restricted Area -> should pick up fg_pct_restricted_area (0.667),
    # not the mid-range number, even though both exist for this player.
    row = _row_for(matrix, "S1")
    assert row["zone_efficiency"] == pytest.approx(0.667)

    row_mid = _row_for(matrix, "S2")
    assert row_mid["zone_efficiency"] == pytest.approx(0.333)


def test_zone_efficiency_falls_back_to_nan_when_zone_stats_missing(matrix):
    # P_TALL shoots from Left Corner 3 (S5) but has no player_zone_stats row for
    # that zone -> zone_efficiency must be NaN, not 0 or an unrelated zone's value.
    row = _row_for(matrix, "S5")
    assert np.isnan(row["zone_efficiency"])


def test_def_fg_pct_zone_uses_zone_level_stats_not_overall(matrix):
    """
    Regression test for the COL_MAP bug: an elite rim protector (P_DEF) has a
    mediocre "Overall" FG% allowed (0.50) but a much better zone-level FG%
    allowed at the rim (0.357). def_fg_pct_zone must reflect the zone-level
    number, not silently collapse to Overall.
    """
    row = _row_for(matrix, "S1")  # Restricted Area shot defended by P_DEF
    assert row["def_fg_pct_zone"] == pytest.approx(0.357)
    assert row["def_fg_pct_overall"] == pytest.approx(0.50)
    assert row["def_fg_pct_zone"] != row["def_fg_pct_overall"]


def test_def_pct_plusminus_zone_uses_zone_level_stats(matrix):
    row = _row_for(matrix, "S1")
    assert row["def_pct_plusminus_zone"] == pytest.approx(-0.243)
    assert row["def_pct_plusminus"] == pytest.approx(0.03)


def test_def_fg_pct_zone_falls_back_to_overall_when_zone_category_missing(matrix):
    # S2 is a Mid-Range shot -> maps to "Greater Than 15Ft", which was
    # deliberately NOT seeded for P_DEF. matchup_advantage must fall back to
    # the Overall FG% allowed rather than becoming NaN.
    row = _row_for(matrix, "S2")
    assert np.isnan(row["def_fg_pct_zone"])
    assert row["matchup_advantage"] == pytest.approx(row["zone_efficiency"] - row["def_fg_pct_overall"])


def test_matchup_advantage_computed_correctly_for_rim_shot(matrix):
    row = _row_for(matrix, "S1")
    expected = 0.667 - 0.357  # attacker zone efficiency minus defender zone FG% allowed
    assert row["matchup_advantage"] == pytest.approx(expected, abs=1e-6)


def test_matchup_advantage_for_three_point_shot(matrix):
    row = _row_for(matrix, "S3")
    expected = 0.391 - 0.4286
    assert row["matchup_advantage"] == pytest.approx(expected, abs=1e-4)


def test_get_feature_columns_includes_zone_and_defender_features(matrix):
    cols = fe.get_feature_columns(matrix)
    for expected_col in (
        "zone_efficiency", "def_fg_pct_zone", "def_pct_plusminus_zone",
        "matchup_advantage", "height_diff",
    ):
        assert expected_col in cols
