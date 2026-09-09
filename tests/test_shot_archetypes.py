"""
Tests for src/analysis/shot_archetypes.py's pure logic — labeling and column
selection. No DB, no fitted model: those are exercised by actually running
`--fit`, which is documented in docs/streaming.md's sibling doc rather than
re-run here (it rebuilds the full training feature matrix, a few minutes of
work, and writes 2M+ DB rows — too heavy for routine test runs).
"""
import json

import pandas as pd
import pytest

from src.analysis.shot_archetypes import (
    ARCHETYPE_FEATURE_COLS,
    ARCHETYPE_METADATA,
    CONTEST_COLS,
    FINISH_COLS,
    MECH_COLS,
    SPATIAL_COLS,
    _label_cluster,
)


def test_feature_cols_have_no_duplicates_and_cover_every_family():
    assert len(ARCHETYPE_FEATURE_COLS) == len(set(ARCHETYPE_FEATURE_COLS))
    assert set(ARCHETYPE_FEATURE_COLS) == set(
        SPATIAL_COLS + MECH_COLS + FINISH_COLS + CONTEST_COLS
    )


def test_feature_cols_exclude_player_season_aggregates():
    """
    The module's own docstring commits to clustering the SHOT, not the
    player who took it — shooter_skill and creation-profile aggregates
    (season-long tendencies) must not sneak into the feature set, since
    that would cluster by player type instead.
    """
    player_level_features = {
        "zone_rate", "overall_rate", "three_rate", "recent_10_fg",
        "self_creation_index", "avg_drib_per_touch", "touches_per_min",
    }
    assert not player_level_features & set(ARCHETYPE_FEATURE_COLS)


def _centroid(**overrides) -> pd.Series:
    base = {c: 0.0 for c in MECH_COLS + FINISH_COLS}
    base["expected_contest"] = 3.0
    base.update(overrides)
    return pd.Series(base)


def test_label_cluster_names_dominant_mechanic_and_finish():
    centroid = _centroid(mech_driving=0.9, finish_layup=0.8, expected_contest=2.0)
    label = _label_cluster(centroid, zone_mode="Restricted Area", population_mean_contest=3.0)
    assert "driving" in label
    assert "layup" in label
    assert "rim" in label


def test_label_cluster_reflects_contest_level_relative_to_population():
    """
    "open"/"contested" is relative to the fitted population's own mean
    expected_contest, not a fixed number of feet. Two fixed cutoffs were
    tried and both failed identically: the technical clip floor (0.5 ft)
    called every real cluster "contested"; the NBA's own Tight/Open boundary
    (4 ft) then called every real cluster "open", because this jump-shot-
    heavy dataset's population average already exceeds 4 ft. A label that
    never varies across clusters carries no information regardless of which
    absolute number produced it.
    """
    population_mean = 5.0
    open_centroid = _centroid(mech_spot_up=0.9, finish_jumper=0.8, expected_contest=8.0)
    tight_centroid = _centroid(mech_spot_up=0.9, finish_jumper=0.8, expected_contest=2.0)
    assert "open" in _label_cluster(open_centroid, "Above the Break 3", population_mean)
    assert "contested" in _label_cluster(tight_centroid, "Above the Break 3", population_mean)


@pytest.mark.skipif(not ARCHETYPE_METADATA.exists(), reason="archetype model not fit")
def test_real_fitted_labels_are_not_all_the_same():
    """
    The actual regression this guards: two absolute contest thresholds were
    each tried and each produced every real cluster label saying the same
    thing ("contested" for all seven, then "open" for all seven) — a label
    that never varies carries zero information regardless of which specific
    number produced it. This checks the real, currently-fitted output
    rather than a synthetic centroid, since that is exactly what a synthetic
    test failed to catch the first two times.
    """
    metadata = json.loads(ARCHETYPE_METADATA.read_text())
    labels = [c["label"] for c in metadata["clusters"]]
    assert len(labels) >= 2

    contest_words = {label.split()[0] for label in labels}
    assert contest_words == {"open", "contested"}, (
        f"expected a mix of open/contested archetypes, got only {contest_words} "
        f"across all {len(labels)} clusters — labels: {labels}"
    )
