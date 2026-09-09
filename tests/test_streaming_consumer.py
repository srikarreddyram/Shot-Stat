"""
Tests for src/streaming/consumer.py's scoring logic. No Kafka or real model
involved — these pin the one behavior that broke in manual end-to-end
testing (see below) using a fake store and a fake recommender.
"""
import pandas as pd
import pytest

from src.streaming.consumer import score_event


class _FakeRecommender:
    """Stands in for ShotRecommender: records what `_predict` was called
    with, so a test can assert on exactly which columns were selected."""

    def __init__(self, feature_cols):
        self.feature_cols = feature_cols
        self.last_X = None

    def _predict(self, X):
        self.last_X = X
        return [0.42]


@pytest.fixture()
def store():
    # A feature store row deliberately carries MORE columns than any one
    # model uses — build_matrix returns every feature it knows how to build,
    # and a specific trained model only consumes an ablation-filtered subset
    # of them (see the docstring on score_event).
    return pd.DataFrame(
        {"shot_id": ["s1"], "wanted_a": [1.0], "wanted_b": [2.0],
         "unwanted_extra": [999.0]}
    ).set_index("shot_id", drop=False)


def test_score_event_selects_columns_by_recommender_feature_cols(store):
    """
    Regression test: the consumer originally selected columns using the
    feature STORE's own column list (everything build_matrix computes, 158
    at last count) instead of the loaded MODEL's column list (a smaller,
    ablation-filtered subset, e.g. 104 for shot-quality-v13). The two lists
    silently diverge whenever a model is trained with a feature group
    ablated out, and XGBoost raises a feature-mismatch error the first time
    a real event is scored — which is what happened running this live
    against shot-quality-v13.
    """
    recommender = _FakeRecommender(feature_cols=["wanted_a", "wanted_b"])
    event = {"shot_id": "s1"}

    prob = score_event(event, store, recommender)

    assert prob == 0.42
    assert list(recommender.last_X.columns) == ["wanted_a", "wanted_b"]
    assert "unwanted_extra" not in recommender.last_X.columns


def test_score_event_returns_none_for_unknown_shot_id(store):
    recommender = _FakeRecommender(feature_cols=["wanted_a", "wanted_b"])
    assert score_event({"shot_id": "not-in-store"}, store, recommender) is None
