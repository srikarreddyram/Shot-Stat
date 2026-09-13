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
    # Each teammate gets a distinct, known creation score so the mean AND
    # max are both hand-checkable; the shooter ALSO gets one, to prove it is
    # excluded rather than accidentally averaged/maxed in.
    fake_profiles = pd.DataFrame({
        "player_id": ["SHOOTER", "MATE1", "MATE2", "MATE3", "MATE4"],
        "season": ["2022-23"] * 5,
        "self_creation_index": [999.0, 1.0, 2.0, 3.0, 4.0],
        "playmaking_gravity": [999.0, 0.1, 0.2, 0.3, 0.4],
        "rim_pressure": [999.0, 10.0, 20.0, 30.0, 40.0],
        "drive_pf_pct": [999.0, 0.05, 0.10, 0.15, 0.20],
    })
    monkeypatch.setattr(
        "src.features.creation.load_creation_profiles",
        lambda engine: fake_profiles,
    )

    # Each help defender gets a distinct, known FG%-allowed; the primary
    # defender also gets one, to prove it is excluded. Category is
    # "Less Than 6Ft" (not "Overall") because the fixture shot is a
    # Restricted Area attempt and build_lineup_context now matches the
    # defender's rate to the SHOT'S zone (ZONE_TO_DEF_CATEGORY), not a
    # zone-agnostic pooled number.
    fake_rates = pd.DataFrame({
        "defense_player_id": ["PRIMARY_DEF", "HELP1", "HELP2", "HELP3", "HELP4"],
        "game_id": ["G1"] * 5,
        "defense_category": ["Less Than 6Ft"] * 5,
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

    # Peak-threat (max) versions: MATE4 is the best on every offensive trait,
    # and the shooter's own 999.0 must NOT win — proves exclusion holds for
    # max the same way it holds for mean.
    assert row["oncourt_off_creation_max"] == pytest.approx(4.0)
    assert row["oncourt_off_gravity_max"] == pytest.approx(0.4)
    assert row["oncourt_off_rim_pressure_max"] == pytest.approx(40.0)
    assert row["oncourt_off_foul_rate_max"] == pytest.approx(0.20)

    assert row["oncourt_def_n"] == 4
    assert row["oncourt_def_fg_pct"] == pytest.approx((0.40 + 0.42 + 0.44 + 0.46) / 4)
    # Toughest single help defender (lowest FG% allowed) — PRIMARY_DEF's
    # 0.01 must NOT win, proving exclusion holds for min too.
    assert row["oncourt_def_fg_pct_min"] == pytest.approx(0.40)


def test_help_defense_rate_is_matched_to_the_shots_zone_not_pooled_overall(
    lineup_engine, monkeypatch
):
    # HELP1 is a lockdown rim defender (0.30 at the rim) but a sieve on 3s
    # (0.60) — the opposite of his "Overall" pooled number, which sits
    # between the two. The fixture shot is a Restricted Area attempt, so
    # build_lineup_context must pick the "Less Than 6Ft" row for HELP1, not
    # a pooled "Overall" figure that would misrepresent him either way.
    monkeypatch.setattr(
        "src.features.creation.load_creation_profiles",
        lambda engine: pd.DataFrame(columns=[
            "player_id", "season", "self_creation_index",
            "playmaking_gravity", "rim_pressure", "drive_pf_pct",
        ]),
    )
    fake_rates = pd.DataFrame({
        "defense_player_id": ["PRIMARY_DEF"] + ["HELP1"] * 3 + ["HELP2", "HELP3", "HELP4"],
        "game_id": ["G1"] * 7,
        "defense_category": (
            ["Less Than 6Ft"]
            + ["Less Than 6Ft", "3 Pointers", "Overall"]
            + ["Less Than 6Ft"] * 3
        ),
        "d_fg_pct": [0.01, 0.30, 0.60, 0.45, 0.42, 0.44, 0.46],
    })
    monkeypatch.setattr(point_in_time, "build_defender_category_rates",
                        lambda engine, through_season: fake_rates)

    out = point_in_time.build_lineup_context(lineup_engine, through_season="2023-24")
    row = out.iloc[0]

    # (0.30 + 0.42 + 0.44 + 0.46) / 4 — HELP1 contributes his rim rate (0.30),
    # never his 3PT rate (0.60) or his pooled Overall rate (0.45).
    assert row["oncourt_def_fg_pct"] == pytest.approx((0.30 + 0.42 + 0.44 + 0.46) / 4)


def test_defensive_activity_uses_peak_threat_and_excludes_primary_defender(
    lineup_engine, monkeypatch
):
    # HELP3 is a Wembanyama-caliber shot-blocker; the other three help
    # defenders and PRIMARY_DEF are ordinary. build_lineup_context must
    # report HELP3's peak numbers (the whole point of the deterrence
    # effect — one dominant rim protector changes what the offense
    # attempts, regardless of who's the primary defender) and must NOT let
    # PRIMARY_DEF's 999.0 win despite it being the largest raw number.
    monkeypatch.setattr(
        "src.features.creation.load_creation_profiles",
        lambda engine: pd.DataFrame(columns=[
            "player_id", "season", "self_creation_index",
            "playmaking_gravity", "rim_pressure", "drive_pf_pct",
        ]),
    )
    monkeypatch.setattr(point_in_time, "build_defender_category_rates",
                        lambda engine, through_season: pd.DataFrame(columns=[
                            "defense_player_id", "game_id", "defense_category", "d_fg_pct",
                        ]))
    fake_activity = pd.DataFrame({
        "player_id": ["PRIMARY_DEF", "HELP1", "HELP2", "HELP3", "HELP4"],
        "season": ["2022-23"] * 5,
        "stl_per_min": [999.0, 0.01, 0.02, 0.03, 0.015],
        "blk_per_min": [999.0, 0.02, 0.03, 0.12, 0.01],
        "deflections_per_min": [999.0, 0.05, 0.06, 0.07, 0.04],
        "defensive_gravity": [999.0, -0.2, 0.1, 2.5, -0.5],
    })
    monkeypatch.setattr(
        "src.features.defensive_activity.load_defensive_activity_profiles",
        lambda engine: fake_activity,
    )

    out = point_in_time.build_lineup_context(lineup_engine, through_season="2023-24")
    row = out.iloc[0]

    assert row["oncourt_def_blk_max"] == pytest.approx(0.12)
    assert row["oncourt_def_stl_max"] == pytest.approx(0.03)
    assert row["oncourt_def_deflections_max"] == pytest.approx(0.07)
    assert row["oncourt_def_gravity_max"] == pytest.approx(2.5)


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
