"""
Tests for per-shot mechanic handling at serving time.

Mechanics are the most valuable features in the model, and they are also the
only ones that do not exist for the thing the recommender is asked about — a
shot that has not been taken has no mechanic. The serving path therefore
marginalises over the player's mix, and these tests pin the properties that
makes correct.
"""
import numpy as np
import pandas as pd
import pytest

from src.features.mechanics import (
    MIN_MECHANIC_SHARE,
    expand_grid_over_mechanics,
    marginalize,
)


@pytest.fixture()
def grid():
    return pd.DataFrame([
        {"loc_x": 0.0, "loc_y": 240.0, "zone": "Above the Break 3", "shot_distance": 24.0},
        {"loc_x": 80.0, "loc_y": 230.0, "zone": "Above the Break 3", "shot_distance": 24.5},
        {"loc_x": 0.0, "loc_y": 10.0, "zone": "Restricted Area", "shot_distance": 1.0},
    ])


@pytest.fixture()
def mix():
    return {
        "Above the Break 3": {"jumper": 0.5, "pullup": 0.3, "stepback": 0.2},
        "Restricted Area": {"driving": 0.6, "cutting": 0.4},
    }


def test_expansion_covers_every_location_and_mechanic(grid, mix):
    expanded = expand_grid_over_mechanics(grid, mix)
    # 2 above-the-break locations x 3 mechanics + 1 rim location x 2 mechanics
    assert len(expanded) == 2 * 3 + 1 * 2
    assert set(expanded["_mechanic"]) == {"jumper", "pullup", "stepback",
                                          "driving", "cutting"}


def test_weights_form_a_distribution_per_location(grid, mix):
    expanded = expand_grid_over_mechanics(grid, mix)
    totals = expanded.groupby(["loc_x", "loc_y"])["_mech_weight"].sum()
    assert np.allclose(totals.values, 1.0)


def test_marginal_is_the_weighted_expectation(grid, mix):
    expanded = expand_grid_over_mechanics(grid, mix)
    # Deterministic per-mechanic probabilities so the expectation is checkable.
    probs = {"jumper": 0.40, "pullup": 0.30, "stepback": 0.20,
             "driving": 0.60, "cutting": 0.80}
    expanded["make_probability"] = expanded["_mechanic"].map(probs)

    out = marginalize(expanded, key_cols=("loc_x", "loc_y", "zone", "shot_distance"))
    assert len(out) == 3

    atb = out[out["zone"] == "Above the Break 3"].iloc[0]
    expected = 0.5 * 0.40 + 0.3 * 0.30 + 0.2 * 0.20
    assert atb["make_probability"] == pytest.approx(expected)

    rim = out[out["zone"] == "Restricted Area"].iloc[0]
    assert rim["make_probability"] == pytest.approx(0.6 * 0.60 + 0.4 * 0.80)


def test_marginal_lies_between_the_best_and_worst_mechanic(grid, mix):
    """
    A weighted average of the per-mechanic predictions cannot exceed the best
    of them or fall below the worst. If it does, the weights are wrong.
    """
    expanded = expand_grid_over_mechanics(grid, mix)
    probs = {"jumper": 0.40, "pullup": 0.30, "stepback": 0.20,
             "driving": 0.60, "cutting": 0.80}
    expanded["make_probability"] = expanded["_mechanic"].map(probs)
    out = marginalize(expanded, key_cols=("loc_x", "loc_y", "zone", "shot_distance"))

    for _, row in out.iterrows():
        same = expanded[
            (expanded["loc_x"] == row["loc_x"]) & (expanded["loc_y"] == row["loc_y"])
        ]["make_probability"]
        assert same.min() - 1e-9 <= row["make_probability"] <= same.max() + 1e-9


def test_best_mechanic_is_the_highest_scoring_one(grid, mix):
    expanded = expand_grid_over_mechanics(grid, mix)
    probs = {"jumper": 0.40, "pullup": 0.30, "stepback": 0.20,
             "driving": 0.60, "cutting": 0.80}
    expanded["make_probability"] = expanded["_mechanic"].map(probs)
    out = marginalize(expanded, key_cols=("loc_x", "loc_y", "zone", "shot_distance"))

    atb = out[out["zone"] == "Above the Break 3"].iloc[0]
    assert atb["best_mechanic"] == "jumper"
    assert atb["best_mechanic_prob"] == pytest.approx(0.40)

    rim = out[out["zone"] == "Restricted Area"].iloc[0]
    assert rim["best_mechanic"] == "cutting"


def test_empty_mix_falls_back_without_crashing(grid):
    """A player with no play-by-play history must still get a prediction."""
    expanded = expand_grid_over_mechanics(grid, {})
    assert len(expanded) == len(grid)
    assert set(expanded["_mechanic"]) == {"other"}
    assert np.allclose(expanded["_mech_weight"].values, 1.0)


def test_min_share_threshold_is_a_sane_fraction():
    assert 0.0 < MIN_MECHANIC_SHARE < 0.2
