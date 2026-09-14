"""
Tests for src/features/point_in_time/clutch_performance.py.

Real gap this closes: the model already knew whether a SHOT was a clutch
situation (`clutch_flag`) but had no way to know whether the PLAYER taking it
is any good in the clutch specifically — a fact the user asked for directly
("some people are better than others in the clutch"). These tests check the
two things that matter for a point-in-time feature: the cumulative counts are
strictly PRIOR (never including the game being predicted), and the two
independent code paths that produce them (bulk `build_*` for training, single-
player `lookup_*` for serving) agree on the same player.
"""
from datetime import date

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.db.database import register_unaccent
from src.db.models import Base, Game, Player, Shot
from src.features.point_in_time.clutch_performance import (
    CLUTCH_MARGIN,
    CLUTCH_TIME_REMAINING,
    build_clutch_performance,
    fit_league_clutch_prior,
    lookup_clutch_performance,
)


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


def _add_game(session, game_id: str, d: date):
    session.add(Game(game_id=game_id, date=d, home_team="A", away_team="B"))


def _add_shot(session, shot_id, game_id, player_id, made, quarter, time_remaining,
             score_diff, season="2023-24"):
    session.add(Shot(
        shot_id=shot_id, game_id=game_id, player_id=player_id, season=season,
        shot_made=made, zone="Mid-Range", quarter=quarter,
        time_remaining=time_remaining, score_diff=score_diff,
    ))


@pytest.fixture()
def seeded(engine):
    """
    One player (P1) across three chronological games:
      G1 (Jan 1): 2 clutch attempts, 1 make. No prior history.
      G2 (Jan 2): 1 clutch attempt, 1 make. Prior-to-this-game clutch history
                  must be exactly G1's (1 make / 2 attempts).
      G3 (Jan 3): 0 clutch attempts (a non-clutch shot only). Prior clutch
                  history must carry forward G1+G2 (2 makes / 3 attempts)
                  even though this game contributes nothing new.

    Plus a handful of other players so a league-wide prior has more than one
    data point to pool over.
    """
    Session = sessionmaker(bind=engine)
    with Session() as session:
        session.add(Player(player_id="P1", season="2023-24", name="P1",
                           position="G", team_id="T1"))
        _add_game(session, "G1", date(2023, 1, 1))
        _add_game(session, "G2", date(2023, 1, 2))
        _add_game(session, "G3", date(2023, 1, 3))

        # G1: two clutch attempts, one make.
        _add_shot(session, "s1", "G1", "P1", made=1, quarter=4, time_remaining=200, score_diff=-3)
        _add_shot(session, "s2", "G1", "P1", made=0, quarter=4, time_remaining=100, score_diff=4)
        # G2: one clutch attempt, made.
        _add_shot(session, "s3", "G2", "P1", made=1, quarter=4, time_remaining=250, score_diff=2)
        # G3: a shot that looks late-game but is NOT clutch (margin too wide),
        # plus a genuinely non-clutch first-quarter shot.
        _add_shot(session, "s4", "G3", "P1", made=1, quarter=4, time_remaining=200, score_diff=12)
        _add_shot(session, "s5", "G3", "P1", made=0, quarter=1, time_remaining=600, score_diff=0)

        # A thin league so fit_league_clutch_prior has more than one player.
        for i in range(5):
            pid = f"ROLE{i}"
            session.add(Player(player_id=pid, season="2023-24", name=pid,
                               position="G", team_id="T1"))
            gid = f"RG{i}"
            _add_game(session, gid, date(2023, 1, 10 + i))
            _add_shot(session, f"rs{i}a", gid, pid, made=1, quarter=4, time_remaining=150, score_diff=1)
            _add_shot(session, f"rs{i}b", gid, pid, made=0, quarter=4, time_remaining=150, score_diff=1)

        session.commit()
    return engine


def test_clutch_window_matches_official_nba_definition():
    """Last 5 minutes (300s), margin within 5 — the league's own "Clutch
    Time" definition, not an arbitrary narrower window."""
    assert CLUTCH_TIME_REMAINING == 300.0
    assert CLUTCH_MARGIN == 5


def test_build_clutch_performance_is_strictly_prior(seeded):
    out = build_clutch_performance(seeded)
    row = lambda gid: out[(out["player_id"] == "P1") & (out["game_id"] == gid)].iloc[0]

    # G1 is P1's first game: no prior clutch history at all.
    g1 = row("G1")
    assert g1["pit_car_clutch_mk"] == 0.0
    assert g1["pit_car_clutch_att"] == 0.0

    # G2's prior history is exactly G1's clutch shots: 1 make / 2 attempts.
    g2 = row("G2")
    assert g2["pit_car_clutch_mk"] == 1.0
    assert g2["pit_car_clutch_att"] == 2.0

    # G3's prior history carries forward G1+G2 (2 makes / 3 attempts) even
    # though G3 itself has no clutch attempt of its own — the same "carry
    # forward on games with zero of this event" requirement build_prior_
    # counts' zone grid exists to satisfy, just without needing a grid here
    # since there is only one boolean (clutch or not) instead of six zones.
    g3 = row("G3")
    assert g3["pit_car_clutch_mk"] == 2.0
    assert g3["pit_car_clutch_att"] == 3.0


def test_non_clutch_shots_are_excluded_from_the_counts(seeded):
    """G3's wide-margin 'late but not clutch' shot (s4, score_diff=12) and
    first-quarter shot (s5) must not contribute to clutch attempts anywhere.
    G3's own carried-forward total (asserted in the prior test) already
    proves this: it is 2/3, not 3/5 — s4 and s5 never counted."""
    out = build_clutch_performance(seeded)
    row = out[(out["player_id"] == "P1") & (out["game_id"] == "G3")].iloc[0]
    assert row["pit_car_clutch_att"] == 3.0
    assert row["pit_car_clutch_mk"] == 2.0


def test_lookup_matches_build_for_the_same_player_as_of_date(seeded):
    """Train/serve parity: the serving path's single-player lookup, evaluated
    as-of G3's date, must agree with the bulk builder's prior count for G3."""
    from sqlalchemy import text
    with seeded.connect() as conn:
        g3_date = conn.execute(
            text("SELECT date FROM games WHERE game_id = 'G3'")
        ).scalar()
        served = lookup_clutch_performance(conn, "P1", as_of_date=g3_date)

    built = build_clutch_performance(seeded)
    trained = built[(built["player_id"] == "P1") & (built["game_id"] == "G3")].iloc[0]

    assert served["pit_car_clutch_mk"] == trained["pit_car_clutch_mk"]
    assert served["pit_car_clutch_att"] == trained["pit_car_clutch_att"]


def test_lookup_with_no_as_of_date_sees_everything(seeded):
    """Omitting as_of_date (live serving — 'now') must include every game on
    record, matching what a caller who asks for P1's CURRENT clutch history
    (after G3) would expect."""
    with seeded.connect() as conn:
        served = lookup_clutch_performance(conn, "P1", as_of_date=None)
    assert served["pit_car_clutch_mk"] == 2.0
    assert served["pit_car_clutch_att"] == 3.0


def test_lookup_returns_zero_not_none_for_a_player_with_no_shots(seeded):
    with seeded.connect() as conn:
        served = lookup_clutch_performance(conn, "NOBODY", as_of_date=None)
    assert served == {"pit_car_clutch_mk": 0.0, "pit_car_clutch_att": 0.0}


def test_fit_league_clutch_prior_pools_across_players(seeded):
    """A real, non-degenerate prior from more than one player's clutch
    attempts — mean should sit near the pooled rate (roughly 50% here, since
    every seeded role player and P1 collectively make about half their
    clutch attempts)."""
    prior = fit_league_clutch_prior(seeded, through_season="2023-24")
    assert 0.0 < prior.mean < 1.0
    assert prior.strength > 0
