"""
Tests for src/features/defensive_activity.py — per-minute rate normalization,
the volume guard, and the defensive_gravity composite.
"""
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from src.db.database import register_unaccent
from src.db.models import Base, PlayerDefensiveActivity
from src.features import defensive_activity as da


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


def _seed(engine, rows):
    from sqlalchemy.orm import sessionmaker
    Session = sessionmaker(bind=engine)
    with Session() as session:
        for row in rows:
            session.add(PlayerDefensiveActivity(**row))
        session.commit()


def test_low_volume_player_is_blanked_not_measured(engine):
    # WEMBY plays a full season at heavy minutes; SMALL_SAMPLE has 3 games at
    # 4 minutes each — enough attempts to accidentally lead the league in a
    # raw per-minute rate, exactly what the volume guard exists to catch.
    _seed(engine, [
        {"player_id": "WEMBY", "season": "2023-24", "gp": 71, "min_per_game": 29.7,
         "stl": 1.2, "blk": 3.6, "deflections": 1.5},
        {"player_id": "SMALL_SAMPLE", "season": "2023-24", "gp": 3, "min_per_game": 4.0,
         "stl": 0.0, "blk": 2.0, "deflections": 0.0},
        {"player_id": "ROLE_PLAYER", "season": "2023-24", "gp": 60, "min_per_game": 20.0,
         "stl": 0.8, "blk": 0.3, "deflections": 1.0},
    ])
    out = da.load_defensive_activity_profiles(engine)
    row = out[out["player_id"] == "SMALL_SAMPLE"].iloc[0]
    assert pd.isna(row["blk_per_min"])
    assert pd.isna(row["defensive_gravity"])

    wemby = out[out["player_id"] == "WEMBY"].iloc[0]
    assert wemby["blk_per_min"] == pytest.approx(3.6 / 29.7)


def test_defensive_gravity_ranks_the_shot_blocker_highest(engine):
    # WEMBY leads on blocks (the heaviest-weighted component, 0.50) despite
    # ROLE_PLAYER matching him on deflections and steals — the composite
    # should still favor the elite shot-blocker, matching the motivating
    # case (opponents avoid the rim against a dominant shot-blocker).
    _seed(engine, [
        {"player_id": "WEMBY", "season": "2023-24", "gp": 71, "min_per_game": 29.7,
         "stl": 0.5, "blk": 3.6, "deflections": 1.0},
        {"player_id": "ROLE_PLAYER", "season": "2023-24", "gp": 71, "min_per_game": 29.7,
         "stl": 0.5, "blk": 0.3, "deflections": 1.0},
    ])
    out = da.load_defensive_activity_profiles(engine)
    wemby = out[out["player_id"] == "WEMBY"].iloc[0]["defensive_gravity"]
    role = out[out["player_id"] == "ROLE_PLAYER"].iloc[0]["defensive_gravity"]
    assert wemby > role


def test_no_components_gives_nan_not_zero(engine):
    # A player with a real row but all-NaN rate columns (e.g. min_per_game
    # missing) must come out NaN on the composite, not a false "exactly
    # average" 0.0 — same reasoning as creation.py's _COMPOSITES.
    _seed(engine, [
        {"player_id": "ONLY_ONE", "season": "2023-24", "gp": 71, "min_per_game": 29.7,
         "stl": 0.5, "blk": 1.0, "deflections": 1.0},
        {"player_id": "NO_MINUTES", "season": "2023-24", "gp": 71, "min_per_game": None,
         "stl": 0.5, "blk": 1.0, "deflections": 1.0},
    ])
    out = da.load_defensive_activity_profiles(engine)
    row = out[out["player_id"] == "NO_MINUTES"].iloc[0]
    assert pd.isna(row["defensive_gravity"])


def test_empty_table_returns_empty_frame_with_expected_columns(engine):
    out = da.load_defensive_activity_profiles(engine)
    assert out.empty
    assert list(out.columns) == ["player_id", "season"] + da.DEFENSIVE_ACTIVITY_FEATURE_COLS
