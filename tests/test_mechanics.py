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
        "Above the Break 3": {"spot_up jumper": 0.5, "pullup jumper": 0.3,
                              "stepback jumper": 0.2},
        "Restricted Area": {"driving layup": 0.6, "driving dunk": 0.4},
    }


def test_expansion_covers_every_location_and_mechanic(grid, mix):
    expanded = expand_grid_over_mechanics(grid, mix)
    # 2 above-the-break locations x 3 mechanics + 1 rim location x 2 mechanics
    assert len(expanded) == 2 * 3 + 1 * 2
    assert set(expanded["_mechanic"]) == {
        "spot_up jumper", "pullup jumper", "stepback jumper",
        "driving layup", "driving dunk"}


def test_weights_form_a_distribution_per_location(grid, mix):
    expanded = expand_grid_over_mechanics(grid, mix)
    totals = expanded.groupby(["loc_x", "loc_y"])["_mech_weight"].sum()
    assert np.allclose(totals.values, 1.0)


def test_marginal_is_the_weighted_expectation(grid, mix):
    expanded = expand_grid_over_mechanics(grid, mix)
    # Deterministic per-mechanic probabilities so the expectation is checkable.
    probs = {"spot_up jumper": 0.40, "pullup jumper": 0.30,
             "stepback jumper": 0.20, "driving layup": 0.60,
             "driving dunk": 0.80}
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
    probs = {"spot_up jumper": 0.40, "pullup jumper": 0.30,
             "stepback jumper": 0.20, "driving layup": 0.60,
             "driving dunk": 0.80}
    expanded["make_probability"] = expanded["_mechanic"].map(probs)
    out = marginalize(expanded, key_cols=("loc_x", "loc_y", "zone", "shot_distance"))

    for _, row in out.iterrows():
        same = expanded[
            (expanded["loc_x"] == row["loc_x"]) & (expanded["loc_y"] == row["loc_y"])
        ]["make_probability"]
        assert same.min() - 1e-9 <= row["make_probability"] <= same.max() + 1e-9


def test_best_mechanic_is_the_highest_scoring_one(grid, mix):
    expanded = expand_grid_over_mechanics(grid, mix)
    probs = {"spot_up jumper": 0.40, "pullup jumper": 0.30,
             "stepback jumper": 0.20, "driving layup": 0.60,
             "driving dunk": 0.80}
    expanded["make_probability"] = expanded["_mechanic"].map(probs)
    out = marginalize(expanded, key_cols=("loc_x", "loc_y", "zone", "shot_distance"))

    atb = out[out["zone"] == "Above the Break 3"].iloc[0]
    assert atb["best_mechanic"] == "spot_up jumper"
    assert atb["best_mechanic_prob"] == pytest.approx(0.40)

    rim = out[out["zone"] == "Restricted Area"].iloc[0]
    assert rim["best_mechanic"] == "driving dunk"


def test_empty_mix_falls_back_without_crashing(grid):
    """A player with no play-by-play history must still get a prediction."""
    expanded = expand_grid_over_mechanics(grid, {})
    assert len(expanded) == len(grid)
    assert set(expanded["_mechanic"]) == {"spot_up jumper"}
    assert np.allclose(expanded["_mech_weight"].values, 1.0)


def test_min_share_threshold_is_a_sane_fraction():
    assert 0.0 < MIN_MECHANIC_SHARE < 0.2


def test_finish_is_independent_of_creation():
    """
    Creation and finish are orthogonal and must not compete for one bucket.
    "Driving Dunk Shot" is a driving CREATION and a dunk FINISH; the ordered
    creation scan gives `driving` the win, which left `dunk` catching only the
    bare "Dunk Shot" label.

    That is not cosmetic. Driving dunks convert at 88.8% and driving layups at
    60.5% over 2024-25 restricted-area shots, and both encoded identically.
    """
    from src.features.spec import classify_finish, classify_mechanic

    cases = [
        ("Driving Dunk Shot", "driving", "dunk"),
        ("Running Dunk Shot", "transition", "dunk"),
        ("Cutting Dunk Shot", "cutting", "dunk"),
        ("Alley Oop Dunk Shot", "alley_oop", "dunk"),
        ("Driving Layup Shot", "driving", "layup"),
        ("Putback Layup Shot", "putback", "layup"),
        ("Driving Finger Roll Layup Shot", "driving", "layup"),
        ("Turnaround Hook Shot", "post_up", "hook"),
        ("Step Back Jump shot", "stepback", "jumper"),
    ]
    for subtype, mech, finish in cases:
        assert classify_mechanic(subtype) == mech, subtype
        assert classify_finish(subtype) == finish, subtype


def test_every_shot_type_round_trips_through_the_classifiers():
    """
    The serving path does not set `mech_*`/`finish_*` by hand. It writes the
    shot-type name into `shot_subtype` and lets `derive_features` classify it,
    so that training and serving share one encoding. That only works if every
    name in the vocabulary classifies to itself.

    Without this, a served row silently lands in the wrong bucket: before the
    finish split, the grid named a bare creation bucket like "driving", which
    `classify_finish` did not recognise, so EVERY served row carried
    `finish_other=1` — a combination holding 1.5% of training shots.
    """
    from src.features.mechanics import SHOT_TYPES, _shot_type

    assert len(SHOT_TYPES) == 54
    for shot_type in SHOT_TYPES:
        assert _shot_type(shot_type) == shot_type


def test_fallback_shot_type_is_a_real_one():
    from src.features.mechanics import FALLBACK_SHOT_TYPE, SHOT_TYPES

    assert FALLBACK_SHOT_TYPE in SHOT_TYPES


def test_every_dunk_label_reaches_the_dunk_finish():
    """
    The regression this fixes: Antetokounmpo was offered no dunk at the rim,
    because 98% of real dunks were filed under a creation bucket and `dunk`
    fell below the mechanic-share floor.
    """
    from src.features.spec import classify_finish

    for label in ["Dunk Shot", "Driving Dunk Shot", "Running Dunk Shot",
                  "Cutting Dunk Shot", "Alley Oop Dunk Shot",
                  "Reverse Dunk Shot", "Putback Dunk Shot"]:
        assert classify_finish(label) == "dunk", label
