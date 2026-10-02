"""
Tests for src/inference/season_averages.py on a tiny in-memory league where
every right answer is known by hand.
"""
from datetime import date

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.db.models import Base, PlayerGameLog, TeamGameLog
from src.inference.season_averages import player_seasons, team_seasons

BOX = dict(reb=5, oreb=1, dreb=4, ast=4, stl=1, blk=0, tov=2, pf=2, fg3m=0, fg3a=0, ftm=0, fta=0)


@pytest.fixture()
def engine():
    eng = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(eng, tables=[PlayerGameLog.__table__, TeamGameLog.__table__])
    Session = sessionmaker(bind=eng)
    with Session() as s:
        # Player P: season A, two games — 1-for-2 then 12-for-20 (FG% must be 13/22, not the mean of 50% and 60%).
        for gid, fgm, fga, pts, team in (("G1", 1, 2, 2, "AAA"), ("G2", 12, 20, 30, "BBB")):
            s.add(PlayerGameLog(player_id="P", game_id=gid, season="2024-25", season_type="Regular Season",
                                game_date=date(2025, 1, int(gid[1])), player_name="Pat", team_id="1",
                                team_abbrev=team, opp_abbrev="ZZZ", home=1, win=1, minutes=30.0,
                                pts=pts, fgm=fgm, fga=fga, plus_minus=5.0, **BOX))
        s.add(PlayerGameLog(player_id="P", game_id="G9", season="2024-25", season_type="Playoffs",
                            game_date=date(2025, 5, 1), player_name="Pat", team_id="1", team_abbrev="BBB",
                            opp_abbrev="ZZZ", home=0, win=0, minutes=40.0, pts=40, fgm=15, fga=25,
                            plus_minus=-3.0, **BOX))
        # Teams T1 (wins both) vs T2, one season.
        for gid, t1pts, t2pts in (("H1", 110, 100), ("H2", 120, 104)):
            for tid, abbr, opp, pts, win, pm in (("T1", "ONE", "TWO", t1pts, 1, t1pts - t2pts),
                                                 ("T2", "TWO", "ONE", t2pts, 0, t2pts - t1pts)):
                s.add(TeamGameLog(team_id=tid, game_id=gid, season="2024-25", season_type="Regular Season",
                                  game_date=date(2025, 2, int(gid[1])), team_abbrev=abbr, opp_abbrev=opp,
                                  home=1, win=win, pts=pts, fgm=40, fga=90 if tid == "T1" else 80,
                                  fg3m=10, fg3a=30, plus_minus=float(pm), **{k: v for k, v in BOX.items()
                                                                           if k not in ("fg3m", "fg3a")}))
        s.commit()
    return eng


def test_player_per_game_averages_and_shooting_from_totals(engine):
    out = player_seasons(engine, "P")
    row = out["regular"]["seasons"][0]
    assert row["gp"] == 2 and row["pts"] == 16.0
    assert row["fg_pct"] == pytest.approx(13 / 22, abs=1e-4)
    assert row["team"] == "AAA · BBB"  # traded mid-season: both teams, in order


def test_playoffs_are_kept_separate_and_career_spans_seasons(engine):
    out = player_seasons(engine, "P")
    assert out["playoffs"]["seasons"][0]["pts"] == 40.0
    assert out["regular"]["career"]["gp"] == 2 and out["regular"]["career"]["seasons"] == 1


def test_team_record_margin_opponent_line_and_rank(engine):
    out = team_seasons(engine, "T1")
    row = out["regular"]["seasons"][0]
    assert (row["w"], row["l"]) == (2, 0)
    assert row["pts"] == 115.0 and row["opp_pts"] == 102.0 and row["margin"] == 13.0
    assert row["opp_fg_pct"] == pytest.approx(40 / 80)
    assert row["ranks"]["pts"] == 1 and row["ranks"]["opp_pts"] == 1 and row["of"] == 2
    t2 = team_seasons(engine, "T2")["regular"]["seasons"][0]
    assert t2["ranks"]["pts"] == 2 and t2["ranks"]["opp_pts"] == 2  # allowing more ranks worse


def test_unknown_ids_return_none(engine):
    assert player_seasons(engine, "nobody") is None
    assert team_seasons(engine, "nobody") is None
