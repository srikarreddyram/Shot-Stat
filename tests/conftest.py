"""
Shared pytest fixtures.

Every test runs against an isolated, in-process SQLite database built straight
from the ORM metadata in src/db/models.py — never against the real
data/nba_shots.db (that file is 1.1GB, not committed, and its exact contents
should never be a test dependency).

The seeded fixture ("seeded_db") intentionally recreates the scenario behind
a real bug that shipped in defender_stats_ingestor.py: an elite rim-protecting
defender ("P_DEF") who looks mediocre by *overall* FG% allowed (0.50) but is
excellent specifically at the rim (zone FG% allowed 0.357) and only average on
threes (0.4286). Any code that collapses zone-level defense back down to the
"Overall" category will silently give the wrong answer for this player — which
is exactly what the COL_MAP bug did.
"""
import sys
from datetime import date
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db.database import register_unaccent
from src.db.models import (
    Base, Game, Player, PlayerZoneStats, Shot, DefenderStats, Matchup,
)

SEASON = "2023-24"


@pytest.fixture()
def db_engine():
    """
    A fresh in-memory SQLite engine with all tables created, per test.

    Plain `sqlite:///:memory:` hands out a *new*, empty in-memory database on
    every new connection the pool opens — StaticPool pins the engine to a
    single shared connection so data written via one connection (e.g. the
    seeding session) is visible to another (e.g. code under test calling
    engine.connect()).
    """
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    register_unaccent(engine)
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture()
def db_session(db_engine):
    Session = sessionmaker(bind=db_engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture()
def seeded_db(db_engine, db_session):
    """
    Seeds db_engine with a small, realistic fixture set:

      - P_TALL  — tall interior scorer (C, 84in height), efficient at the rim,
                   poor mid-range.
      - P_SHORT — short guard (PG, 72in height), efficient from three,
                   below-average at the rim.
      - P_DEF   — elite rim-protecting defender: bad "Overall" FG% allowed
                   (0.50) but excellent zone-level FG% allowed at the rim
                   (0.357) and merely average on 3s (0.4286). "Greater Than
                   15Ft" (mid-range) is deliberately left unpopulated so
                   fallback-to-overall behavior can be tested too.

    Returns db_engine (already populated) for convenience.
    """
    game = Game(
        game_id="G1", date=date(2024, 1, 15),
        home_team="LAL", away_team="BOS", playoff_flag=0,
    )

    tall = Player(
        player_id="P_TALL", season=SEASON, name="Tall Center",
        height=84.0, weight=250.0, wingspan=88.0, position="C",
        career_fg_pct=0.55, career_3p_pct=0.10, season_fg_pct=0.56,
    )
    short = Player(
        player_id="P_SHORT", season=SEASON, name="Short Guard",
        height=72.0, weight=180.0, wingspan=74.0, position="PG",
        career_fg_pct=0.42, career_3p_pct=0.38, season_fg_pct=0.43,
    )
    defender = Player(
        player_id="P_DEF", season=SEASON, name="Rim Protector",
        height=83.0, weight=245.0, wingspan=90.0, position="C",
    )

    db_session.add_all([game, tall, short, defender])
    db_session.flush()

    zone_rows = [
        PlayerZoneStats(player_id="P_TALL", season=SEASON, zone="Restricted Area",
                         fgm=200, fga=300, fg_pct=0.667),
        PlayerZoneStats(player_id="P_TALL", season=SEASON, zone="Mid-Range",
                         fgm=20, fga=60, fg_pct=0.333),
        PlayerZoneStats(player_id="P_SHORT", season=SEASON, zone="Restricted Area",
                         fgm=40, fga=80, fg_pct=0.50),
        PlayerZoneStats(player_id="P_SHORT", season=SEASON, zone="Above the Break 3",
                         fgm=90, fga=230, fg_pct=0.391, fg3m=90, fg3a=230, fg3_pct=0.391),
    ]

    def_rows = [
        DefenderStats(player_id="P_DEF", season=SEASON, defense_category="Overall",
                       gp=70, freq=1.0, d_fgm=400, d_fga=800,
                       d_fg_pct=0.50, normal_fg_pct=0.47, pct_plusminus=0.03),
        DefenderStats(player_id="P_DEF", season=SEASON, defense_category="Less Than 6Ft",
                       gp=70, freq=0.35, d_fgm=100, d_fga=280,
                       d_fg_pct=0.357, normal_fg_pct=0.60, pct_plusminus=-0.243),
        DefenderStats(player_id="P_DEF", season=SEASON, defense_category="3 Pointers",
                       gp=70, freq=0.25, d_fgm=150, d_fga=350,
                       d_fg_pct=0.4286, normal_fg_pct=0.36, pct_plusminus=0.0686),
        # NOTE: no "Greater Than 15Ft" row — used to test fallback-to-overall.
    ]

    shot_rows = [
        Shot(shot_id="S1", game_id="G1", player_id="P_TALL", season=SEASON,
             defender_id="P_DEF", shot_made=1, loc_x=0, loc_y=10, shot_distance=1.0,
             shot_type="2PT Field Goal", zone="Restricted Area", shot_angle=90.0,
             quarter=1, time_remaining=500, score_diff=0, home_away=1, playoff_flag=0),
        Shot(shot_id="S2", game_id="G1", player_id="P_TALL", season=SEASON,
             defender_id="P_DEF", shot_made=0, loc_x=10, loc_y=140, shot_distance=14.0,
             shot_type="2PT Field Goal", zone="Mid-Range", shot_angle=85.0,
             quarter=1, time_remaining=300, score_diff=2, home_away=1, playoff_flag=0),
        Shot(shot_id="S3", game_id="G1", player_id="P_SHORT", season=SEASON,
             defender_id="P_DEF", shot_made=0, loc_x=0, loc_y=250, shot_distance=25.0,
             shot_type="3PT Field Goal", zone="Above the Break 3", shot_angle=90.0,
             quarter=1, time_remaining=400, score_diff=0, home_away=1, playoff_flag=0),
        # No defender_id for this shot — exercises the "no defender" path.
        Shot(shot_id="S4", game_id="G1", player_id="P_SHORT", season=SEASON,
             defender_id=None, shot_made=1, loc_x=-30, loc_y=8, shot_distance=3.0,
             shot_type="2PT Field Goal", zone="Restricted Area", shot_angle=75.0,
             quarter=2, time_remaining=600, score_diff=-1, home_away=0, playoff_flag=0),
        # P_TALL has no player_zone_stats row for Left Corner 3 — exercises the
        # zone_efficiency NaN-fallback path in feature_engineering.py.
        Shot(shot_id="S5", game_id="G1", player_id="P_TALL", season=SEASON,
             defender_id="P_DEF", shot_made=0, loc_x=-225, loc_y=20, shot_distance=22.5,
             shot_type="3PT Field Goal", zone="Left Corner 3", shot_angle=95.0,
             quarter=3, time_remaining=200, score_diff=5, home_away=1, playoff_flag=0),
    ]

    # Point-in-time defender quality (point_in_time.build_defender_category_rates
    # / lookup_defender_category_rates) is rebuilt from matchups + shots rather
    # than read off `defender_stats`, so P_DEF needs real matchup rows too — the
    # possession weight used to split a shot's outcome across whoever guarded
    # the shooter that game. One row per (game, shooter) is enough here since
    # P_DEF is each shooter's only defender in this fixture.
    matchup_rows = [
        Matchup(game_id="G1", offense_player_id="P_TALL", defense_player_id="P_DEF",
                matchup_minutes=10.0, partial_possessions=3.0,
                matchup_fgm=1, matchup_fga=2, matchup_fg_pct=0.5),
        Matchup(game_id="G1", offense_player_id="P_SHORT", defense_player_id="P_DEF",
                matchup_minutes=8.0, partial_possessions=2.0,
                matchup_fgm=0, matchup_fga=1, matchup_fg_pct=0.0),
    ]

    db_session.add_all(zone_rows + def_rows + shot_rows + matchup_rows)
    db_session.commit()

    return db_engine
