"""
Tests for src/inference/seasons.py: the Stat Engine stays on last season
until teams have played ESTABLISHED_GAMES games of the new one.
"""
from datetime import date

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import config
from src.db.database import register_unaccent
from src.db.models import Base, Game, Player, Shot
from src.inference import stat_engine as se
from src.inference.seasons import ESTABLISHED_GAMES, NBA_TEAMS, regular_season_games, season_status


def _engine(shot_seasons=(), games=()):
    """Shots (one per season named) and games (by id), on a fresh engine —
    season_status is cached per engine, as it is per API process."""
    eng = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    register_unaccent(eng)
    Base.metadata.create_all(eng)
    Session = sessionmaker(bind=eng)
    with Session() as s:
        for gid in games:
            s.add(Game(game_id=gid, date=date(2026, 11, 1), home_team="AAA", away_team="BBB", playoff_flag=0))
        for season in shot_seasons:
            s.add(Player(player_id="P1", season=season, name="Pat", team_id="T1"))
            s.add(Shot(shot_id=f"s-{season}", game_id="G1", player_id="P1", season=season, shot_made=1))
        s.commit()
    return eng


def _new_season_games(n):
    return [f"00226{i:05d}" for i in range(1, n + 1)]


def test_games_are_counted_by_the_season_in_their_id():
    eng = _engine(games=[*_new_season_games(3), "0022500001", "0022500002", "0042600001", "0052600001"])
    assert regular_season_games(eng, "2026-27") == 3  # playoff (004) and play-in (005) ids are not
    assert regular_season_games(eng, "2025-26") == 2
    assert regular_season_games(eng, "2027-28") == 0


def test_a_new_season_waits_until_teams_have_played_ten_games():
    threshold = ESTABLISHED_GAMES * NBA_TEAMS // 2  # 150 league games
    early = season_status(_engine(["2025-26", "2026-27"], _new_season_games(threshold - 1)))
    assert early["default"] == "2025-26" and early["newest"] == "2026-27"
    assert early["newest_games"] == threshold - 1 and early["newest_games_per_team"] == pytest.approx(9.9)
    under_way = season_status(_engine(["2025-26", "2026-27"], _new_season_games(threshold)))
    assert under_way["default"] == "2026-27" and under_way["newest_games_per_team"] == 10.0


def test_with_nothing_earlier_the_newest_season_is_the_default():
    assert season_status(_engine(["2026-27"], _new_season_games(1)))["default"] == "2026-27"
    assert season_status(_engine())["default"] == config.current_season()


def test_the_stat_engine_opens_on_the_established_season():
    early = _engine(["2025-26", "2026-27"], _new_season_games(20))
    assert set(se.build_player_stat_table(early)["season"]) == {"2025-26"}
    # Asking for the new season by name still gets it.
    assert set(se.build_player_stat_table(early, "2026-27")["season"]) == {"2026-27"}
    under_way = _engine(["2025-26", "2026-27"], _new_season_games(400))
    assert set(se.build_player_stat_table(under_way)["season"]) == {"2026-27"}
