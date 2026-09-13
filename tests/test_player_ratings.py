"""
Tests for src/inference/player_ratings.py's NBA 2K fallback — the gap-filler
built after src/inference/compare_ratings_2k.py's comparison (431 players,
off ρ=+0.57, def ρ=+0.42 against our own measured ratings) showed real
agreement between the two, but also showed our measured ratings reading some
obscure defensive role players BETTER than 2K does. The one property that
matters here: a 2K number may only ever fill a gap `compute_ratings` left
open, never override a measured one, however thin that measured sample is.
"""
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.db.database import register_unaccent
from src.db.models import Base, PlayerTwoKRating
from src.inference import player_ratings as pr


@pytest.fixture(autouse=True)
def _reset_module_caches(monkeypatch):
    # Both caches are single global slots with no per-engine keying (an
    # existing pattern this module already had for `_CACHE`) — fine in
    # production's single-engine-per-process world, but tests need a clean
    # slate each time or one test's fixture data leaks into the next.
    monkeypatch.setattr(pr, "_CACHE", {})
    monkeypatch.setattr(pr, "_TWO_K_FALLBACK_CACHE", None)


@pytest.fixture()
def engine():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    register_unaccent(engine)
    Base.metadata.create_all(engine)
    return engine


def _seed_two_k(engine, rows):
    Session = sessionmaker(bind=engine)
    with Session() as session:
        for row in rows:
            session.add(PlayerTwoKRating(**row))
        session.commit()


def test_two_k_fallback_ratings_rebands_onto_our_scale_and_preserves_rank(engine):
    _seed_two_k(engine, [
        {"player_id": "LOW", "overall": 65, "offense_avg": 50.0, "defense_avg": 55.0},
        {"player_id": "MID", "overall": 78, "offense_avg": 70.0, "defense_avg": 65.0},
        {"player_id": "HIGH", "overall": 92, "offense_avg": 88.0, "defense_avg": 80.0},
    ])
    out = pr.two_k_fallback_ratings(engine)

    assert set(out.keys()) == {"LOW", "MID", "HIGH"}
    for row in out.values():
        assert row["rating_source"] == "2k_fallback"
        assert pr.RATING_FLOOR <= row["off_rating"] <= pr.RATING_CEIL
        assert pr.RATING_FLOOR <= row["def_rating"] <= pr.RATING_CEIL

    # Rank order preserved: higher 2K offense_avg/defense_avg -> higher band.
    assert out["LOW"]["off_rating"] < out["MID"]["off_rating"] < out["HIGH"]["off_rating"]
    assert out["LOW"]["def_rating"] < out["MID"]["def_rating"] < out["HIGH"]["def_rating"]


def test_two_k_fallback_returns_empty_dict_for_an_empty_table(engine):
    assert pr.two_k_fallback_ratings(engine) == {}


def test_rating_lookup_never_overrides_a_measured_rating(engine, monkeypatch):
    # MEASURED has a real (if thin) measured rating; 2K disagrees sharply —
    # exactly the case the comparison flagged as our data sometimes being
    # the better read. GAP_ONLY has no measured rating at all.
    measured_table = pd.DataFrame([{
        "player_id": "MEASURED", "off_rating": 88.0, "def_rating": 91.0,
        "component_scoring": 80.0, "component_playmaking": 70.0,
        "component_volume": 60.0, "component_creation": 50.0,
        "component_defence": 95.0,
    }])
    monkeypatch.setattr(pr, "ratings_for_season", lambda engine, season: measured_table)

    _seed_two_k(engine, [
        {"player_id": "MEASURED", "overall": 65, "offense_avg": 40.0, "defense_avg": 40.0},
        {"player_id": "GAP_ONLY", "overall": 75, "offense_avg": 70.0, "defense_avg": 60.0},
    ])

    out = pr.rating_lookup(engine, "2025-26")

    assert out["MEASURED"]["off_rating"] == 88
    assert out["MEASURED"]["def_rating"] == 91
    assert out["MEASURED"]["rating_source"] == "measured"

    assert out["GAP_ONLY"]["rating_source"] == "2k_fallback"
    assert out["GAP_ONLY"]["off_rating"] is not None


def test_rating_lookup_with_no_measured_table_and_no_two_k_data_is_empty(engine, monkeypatch):
    monkeypatch.setattr(pr, "ratings_for_season", lambda engine, season: pd.DataFrame())
    assert pr.rating_lookup(engine, "2025-26") == {}
