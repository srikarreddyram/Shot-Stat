"""
Tests for the serving-path lineup resolution in src/features/point_in_time.py:
`_resolve_recent_lineup_ids`, `_resolve_roster_fallback_ids`, and
`lookup_lineup_context` — the stand-in "rest of the lineup" used for a live
recommend()/explain_matchup() query, where there is no real on-court group
the way there is for a shot that already happened.
"""
from datetime import date

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.db.database import register_unaccent
from src.db.models import (
    Base, Game, Player, PlayerTrackingStats, Shot, ShotOnCourt,
)
from src.features import point_in_time as pit


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


def test_resolve_recent_lineup_ids_picks_the_latest_game_and_excludes_anchor(engine):
    Session = sessionmaker(bind=engine)
    with Session() as session:
        session.add(Game(game_id="G1", date=date(2024, 11, 1),
                         home_team="DEN", away_team="LAL"))
        session.add(Game(game_id="G2", date=date(2024, 11, 5),
                         home_team="DEN", away_team="BOS"))
        session.add(Shot(shot_id="G1_1", game_id="G1", player_id="SHOOTER",
                         season="2024-25", shot_made=1, zone="Restricted Area"))
        session.add(Shot(shot_id="G2_1", game_id="G2", player_id="TEAMMATE1",
                         season="2024-25", shot_made=1, zone="Restricted Area"))
        # Older game: SHOOTER on court with OLD1-4.
        for pid in ["SHOOTER", "OLD1", "OLD2", "OLD3", "OLD4"]:
            session.add(ShotOnCourt(shot_id="G1_1", player_id=pid,
                                    team_id="T1", role="offense"))
        # Newer game: SHOOTER on court (via a teammate's shot) with NEW1-4.
        for pid in ["SHOOTER", "NEW1", "NEW2", "NEW3", "NEW4"]:
            session.add(ShotOnCourt(shot_id="G2_1", player_id=pid,
                                    team_id="T1", role="offense"))
        session.commit()

    with engine.connect() as conn:
        ids = pit._resolve_recent_lineup_ids(conn, "SHOOTER", "offense")

    assert "SHOOTER" not in ids
    assert set(ids) == {"NEW1", "NEW2", "NEW3", "NEW4"}


def test_resolve_recent_lineup_ids_returns_empty_when_anchor_has_no_data(engine):
    with engine.connect() as conn:
        ids = pit._resolve_recent_lineup_ids(conn, "NOBODY", "offense")
    assert ids == []


def test_resolve_roster_fallback_ranks_by_last_seasons_minutes(engine):
    Session = sessionmaker(bind=engine)
    with Session() as session:
        # Current-season roster: all four on the same team as ANCHOR.
        session.add(Player(player_id="ANCHOR", season="2025-26", name="Anchor",
                           team_id="T1"))
        for pid, mins in [("HIGH_MIN", 34.0), ("MID_MIN", 20.0), ("LOW_MIN", 8.0)]:
            session.add(Player(player_id=pid, season="2025-26", name=pid, team_id="T1"))
            session.add(PlayerTrackingStats(player_id=pid, season="2024-25",
                                            min_per_game=mins))
        # OTHER_TEAM plays elsewhere this season — must never be selected.
        session.add(Player(player_id="OTHER_TEAM", season="2025-26", name="Other",
                           team_id="T2"))
        session.add(PlayerTrackingStats(player_id="OTHER_TEAM", season="2024-25",
                                        min_per_game=40.0))
        session.commit()

    with engine.connect() as conn:
        ids = pit._resolve_roster_fallback_ids(conn, "ANCHOR", "2025-26")

    assert ids == ["HIGH_MIN", "MID_MIN", "LOW_MIN"]


def test_resolve_roster_fallback_returns_empty_when_anchor_has_no_team(engine):
    Session = sessionmaker(bind=engine)
    with Session() as session:
        session.add(Player(player_id="NO_TEAM", season="2025-26", name="X",
                           team_id=None))
        session.commit()
    with engine.connect() as conn:
        ids = pit._resolve_roster_fallback_ids(conn, "NO_TEAM", "2025-26")
    assert ids == []


def test_lookup_lineup_context_falls_back_to_roster_when_no_recent_shots(
    engine, monkeypatch
):
    # SHOOTER and DEFENDER have zero shot_on_court presence (e.g. hasn't
    # played this season yet) — lookup_lineup_context must fall back to
    # each one's current-roster teammates.
    Session = sessionmaker(bind=engine)
    with Session() as session:
        session.add(Player(player_id="SHOOTER", season="2025-26", name="S",
                           team_id="T1"))
        session.add(Player(player_id="MATE", season="2025-26", name="M", team_id="T1"))
        session.add(PlayerTrackingStats(player_id="MATE", season="2024-25",
                                        min_per_game=30.0))
        session.add(Player(player_id="DEFENDER", season="2025-26", name="D",
                           team_id="T2"))
        session.add(Player(player_id="HELPER", season="2025-26", name="H", team_id="T2"))
        session.add(PlayerTrackingStats(player_id="HELPER", season="2024-25",
                                        min_per_game=28.0))
        session.commit()

    creation_profiles = pd.DataFrame({
        "player_id": ["MATE"],
        "season": ["2024-25"],
        "self_creation_index": [1.5],
        "playmaking_gravity": [2.0],
        "rim_pressure": [0.5],
        "drive_pf_pct": [0.1],
    })
    defensive_profiles = pd.DataFrame(columns=[
        "player_id", "season", "stl_per_min", "blk_per_min",
        "deflections_per_min", "defensive_gravity",
    ])
    monkeypatch.setattr(pit, "lookup_defender_category_rates",
                        lambda conn, did, priors, as_of_date=None: {"by_category": {}})

    with engine.connect() as conn:
        out = pit.lookup_lineup_context(
            conn, creation_profiles, defensive_profiles, {},
            "SHOOTER", "DEFENDER", "2025-26",
        )

    assert out["oncourt_off_n"] == 1
    assert out["oncourt_off_gravity"] == pytest.approx(2.0)
    assert out["oncourt_off_gravity_max"] == pytest.approx(2.0)
    assert out["oncourt_def_n"] == 1
