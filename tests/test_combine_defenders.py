"""
Tests for ShotRecommender._combine_defenders — the double-team heuristic.

This is a pure function over two defender dicts (no DB/model dependency),
so it's tested standalone rather than via the model-artifact-gated
test_recommender.py suite.

Regression context: an earlier version of this function filled a missing
physical attribute (e.g. a defender with no wingspan on file) with the
OTHER defender's real measurement, which silently borrowed a smaller
player's wingspan to stand in for a bigger player's unmeasured one —
understating the primary defender's true reach and (in one real case)
making a double-team look *easier* to score against than facing the
tougher defender alone. These tests pin the corrected behavior.
"""
from src.inference.recommender import ShotRecommender


def _defender(height=None, weight=None, wingspan=None, def_stats=None):
    return {
        "name": "Test Defender",
        "height": height, "weight": weight, "wingspan": wingspan,
        "def_stats": def_stats or {},
    }


def test_combine_defenders_physicals_take_max_when_both_known():
    primary = _defender(height=80.0, wingspan=83.0)
    secondary = _defender(height=88.0, wingspan=96.0)
    combined = ShotRecommender._combine_defenders(primary, secondary)
    assert combined["height"] == 88.0
    assert combined["wingspan"] == 96.0


def test_combine_defenders_does_not_borrow_secondary_physical_when_primary_missing():
    """The regression case: primary (the tougher/taller defender in
    practice) has no wingspan on file. The combined defender's wingspan
    must stay None, not silently become the secondary's smaller real
    measurement."""
    primary = _defender(height=88.0, wingspan=None)   # e.g. Wemby: no wingspan on file
    secondary = _defender(height=75.0, wingspan=79.0)  # a much smaller defender

    combined = ShotRecommender._combine_defenders(primary, secondary)

    assert combined["height"] == 88.0   # both known -> max
    assert combined["wingspan"] is None  # primary unknown -> stays unknown, not 79.0


def test_combine_defenders_keeps_primary_physical_when_secondary_missing():
    primary = _defender(height=80.0, wingspan=83.0)
    secondary = _defender(height=75.0, wingspan=None)
    combined = ShotRecommender._combine_defenders(primary, secondary)
    assert combined["wingspan"] == 83.0  # primary's own value carries through


def test_combine_defenders_def_stats_prefers_tougher_category_by_category():
    primary = _defender(def_stats={
        "Overall": {"d_fg_pct": 0.40, "pct_plusminus": -0.10, "freq": 1.0},
        "Less Than 6Ft": {"d_fg_pct": 0.47, "pct_plusminus": -0.18, "freq": 0.32},
    })
    secondary = _defender(def_stats={
        "Overall": {"d_fg_pct": 0.56, "pct_plusminus": 0.09, "freq": 1.0},
        "Less Than 6Ft": {"d_fg_pct": 0.72, "pct_plusminus": 0.08, "freq": 0.40},
    })
    combined = ShotRecommender._combine_defenders(primary, secondary)
    # Primary is tougher in both categories -> primary's numbers win throughout.
    assert combined["def_stats"]["Overall"]["d_fg_pct"] == 0.40
    assert combined["def_stats"]["Less Than 6Ft"]["d_fg_pct"] == 0.47


def test_combine_defenders_def_stats_takes_tougher_side_per_category_when_secondary_is_better():
    """When the SECONDARY defender is the tougher one for a given
    category, the combined result should reflect that — a double-team is
    at least as tough as its best individual defender in each category,
    regardless of which slot (primary/secondary) that defender is in."""
    primary = _defender(def_stats={
        "Less Than 6Ft": {"d_fg_pct": 0.72, "pct_plusminus": 0.08, "freq": 0.40},
    })
    secondary = _defender(def_stats={
        "Less Than 6Ft": {"d_fg_pct": 0.47, "pct_plusminus": -0.18, "freq": 0.32},
    })
    combined = ShotRecommender._combine_defenders(primary, secondary)
    assert combined["def_stats"]["Less Than 6Ft"]["d_fg_pct"] == 0.47


def test_combine_defenders_fills_category_missing_from_primary():
    primary = _defender(def_stats={"Overall": {"d_fg_pct": 0.40, "pct_plusminus": -0.10, "freq": 1.0}})
    secondary = _defender(def_stats={
        "Overall": {"d_fg_pct": 0.56, "pct_plusminus": 0.09, "freq": 1.0},
        "Less Than 6Ft": {"d_fg_pct": 0.72, "pct_plusminus": 0.08, "freq": 0.40},
    })
    combined = ShotRecommender._combine_defenders(primary, secondary)
    # Primary never defended at the rim (no category at all) -> use secondary's real number.
    assert combined["def_stats"]["Less Than 6Ft"]["d_fg_pct"] == 0.72
