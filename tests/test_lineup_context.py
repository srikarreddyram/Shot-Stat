"""
Tests for src/features/point_in_time.py's build_lineup_context — the join
that turns `shot_on_court` rows into per-shot "rest of the lineup" features.

The upstream dependencies (load_creation_profiles' z-scored composites,
build_defender_category_rates' shrinkage) are exercised elsewhere and are
expensive to seed realistically here, so they're replaced with small,
controlled stand-ins via monkeypatch — this test is about build_lineup_
context's OWN logic: does it correctly exclude the shooter and the primary
defender, and does it aggregate the remaining four on each side correctly.
"""
from datetime import date

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.db.database import register_unaccent
from src.db.models import Base, Game, Shot, ShotOnCourt
from src.features import point_in_time


@pytest.fixture()
def lineup_engine():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    register_unaccent(engine)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)

    with Session() as session:
        session.add(Game(game_id="G1", date=date(2024, 11, 1),
                         home_team="DEN", away_team="LAL"))
        session.add(Shot(
            shot_id="G1_10", game_id="G1", player_id="SHOOTER", season="2023-24",
            defender_id="PRIMARY_DEF", shot_made=1, zone="Restricted Area",
        ))
        # Offense: shooter plus 4 real teammates.
        for pid in ["SHOOTER", "MATE1", "MATE2", "MATE3", "MATE4"]:
            session.add(ShotOnCourt(shot_id="G1_10", player_id=pid,
                                    team_id="T1", role="offense"))
        # Defense: primary defender plus 4 help defenders.
        for pid in ["PRIMARY_DEF", "HELP1", "HELP2", "HELP3", "HELP4"]:
            session.add(ShotOnCourt(shot_id="G1_10", player_id=pid,
                                    team_id="T2", role="defense"))
        session.commit()

    return engine


def test_excludes_shooter_and_primary_defender_from_the_aggregate(
    lineup_engine, monkeypatch
):
    # Each teammate gets a distinct, known creation score so the mean is
    # hand-checkable; the shooter ALSO gets one, to prove it is excluded
    # rather than accidentally averaged in.
    fake_profiles = pd.DataFrame({
        "player_id": ["SHOOTER", "MATE1", "MATE2", "MATE3", "MATE4"],
        "season": ["2022-23"] * 5,
        "self_creation_index": [999.0, 1.0, 2.0, 3.0, 4.0],
        "playmaking_gravity": [999.0, 0.1, 0.2, 0.3, 0.4],
        "rim_pressure": [999.0, 10.0, 20.0, 30.0, 40.0],
    })
    monkeypatch.setattr(
        "src.features.creation.load_creation_profiles",
        lambda engine: fake_profiles,
    )

    # Each help defender gets a distinct, known FG%-allowed; the primary
    # defender also gets one, to prove it is excluded.
    fake_rates = pd.DataFrame({
        "defense_player_id": ["PRIMARY_DEF", "HELP1", "HELP2", "HELP3", "HELP4"],
        "game_id": ["G1"] * 5,
        "defense_category": ["Overall"] * 5,
        "d_fg_pct": [0.01, 0.40, 0.42, 0.44, 0.46],
    })
    monkeypatch.setattr(point_in_time, "build_defender_category_rates",
                        lambda engine, through_season: fake_rates)

    out = point_in_time.build_lineup_context(lineup_engine, through_season="2023-24")
    assert len(out) == 1
    row = out.iloc[0]

    assert row["shot_id"] == "G1_10"
    assert row["oncourt_off_n"] == 4
    assert row["oncourt_off_creation"] == pytest.approx((1.0 + 2.0 + 3.0 + 4.0) / 4)
    assert row["oncourt_off_gravity"] == pytest.approx((0.1 + 0.2 + 0.3 + 0.4) / 4)
    assert row["oncourt_off_rim_pressure"] == pytest.approx((10.0 + 20.0 + 30.0 + 40.0) / 4)

    assert row["oncourt_def_n"] == 4
    assert row["oncourt_def_fg_pct"] == pytest.approx((0.40 + 0.42 + 0.44 + 0.46) / 4)


def test_empty_shot_on_court_returns_empty_frame(monkeypatch):
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    register_unaccent(engine)
    Base.metadata.create_all(engine)

    out = point_in_time.build_lineup_context(engine, through_season="2023-24")
    assert out.empty
    assert list(out.columns) == ["shot_id"] + point_in_time.LINEUP_FEATURE_COLS
