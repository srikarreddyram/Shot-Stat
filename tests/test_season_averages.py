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


def test_efg_and_ts_from_season_totals(engine):
    # Season A: 32 pts on 13/22, no threes or free throws -> both reduce to
    # makes over attempts weighted by points.
    row = player_seasons(engine, "P")["regular"]["seasons"][0]
    assert row["efg_pct"] == pytest.approx(13 / 22, abs=1e-4)
    assert row["ts_pct"] == pytest.approx(32 / (2 * 22), abs=1e-4)
    # T1: 40/90 with 10 threes, 230 pts over two games of 90 FGA and 0 FTA.
    team = team_seasons(engine, "T1")["regular"]["seasons"][0]
    assert team["efg_pct"] == pytest.approx((40 + 5) / 90, abs=1e-4)
    assert team["ts_pct"] == pytest.approx(230 / (2 * 180), abs=1e-4)
    # Opponent eFG% ranks lower-is-better: T1 allowed (40+5)/80, T2 (40+5)/90.
    assert team["opp_efg_pct"] == pytest.approx(45 / 80, abs=1e-4)
    assert team["ranks"]["opp_efg_pct"] == 2


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


# ── Advanced (official NBA) ─────────────────────────────────────────────────
from src.db.models import PlayerAdvancedStats, TeamAdvancedStats  # noqa: E402
from src.inference.season_averages import player_advanced_seasons, team_advanced_seasons  # noqa: E402
from src.ingestion.advanced_stats_ingestor import _tov_scale, _value  # noqa: E402


@pytest.fixture()
def adv_engine():
    eng = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(eng, tables=[PlayerAdvancedStats.__table__, TeamAdvancedStats.__table__])
    with eng.begin() as conn:
        conn.execute(PlayerAdvancedStats.__table__.insert(), [
            dict(player_id="P", season="2024-25", season_type="Regular Season", player_name="Pat",
                 team_abbrev="AAA", gp=70, usg_pct=0.30, ts_pct=0.62, def_rating=110.0, pace=99.0),
            dict(player_id="P", season="2025-26", season_type="Regular Season", player_name="Pat",
                 team_abbrev="BBB", gp=60, usg_pct=0.28, ts_pct=0.65, def_rating=108.0, pace=101.0),
        ])
        conn.execute(TeamAdvancedStats.__table__.insert(), [
            dict(team_id="T1", season="2025-26", season_type="Regular Season", gp=82, net_rating=8.0,
                 def_rating=106.0, pace=97.0, tov_pct=0.12),
            dict(team_id="T2", season="2025-26", season_type="Regular Season", gp=82, net_rating=-3.0,
                 def_rating=114.0, pace=103.0, tov_pct=0.15),
        ])
    return eng


def test_player_advanced_seasons_newest_first_without_a_career_line(adv_engine):
    out = player_advanced_seasons(adv_engine, "P")
    rows = out["regular"]["seasons"]
    assert [r["season"] for r in rows] == ["2025-26", "2024-25"]
    assert rows[0]["team"] == "BBB" and rows[0]["ts_pct"] == pytest.approx(0.65)
    assert out["regular"]["career"] is None and out["playoffs"]["seasons"] == []
    assert "def_rating" in out["lower_is_better"] and "pace" in out["no_best"]
    assert player_advanced_seasons(adv_engine, "nobody") is None


def test_team_advanced_ranks_respect_direction_and_skip_pace(adv_engine):
    t1 = team_advanced_seasons(adv_engine, "T1")["regular"]["seasons"][0]
    assert t1["ranks"]["net_rating"] == 1
    assert t1["ranks"]["def_rating"] == 1   # allowing fewer points ranks better
    assert t1["ranks"]["tov_pct"] == 1      # fewer turnovers ranks better
    assert "pace" not in t1["ranks"] and t1["of"] == 2
    assert team_advanced_seasons(adv_engine, "nobody") is None


def test_turnover_rate_is_stored_as_a_fraction_whatever_the_endpoint_sends():
    import pandas as pd
    # Players' endpoint: per 100 plays. Teams' endpoint: already a fraction.
    players, teams = pd.DataFrame({"TM_TOV_PCT": [10.0, 12.5]}), pd.DataFrame({"TM_TOV_PCT": [0.13, 0.15]})
    assert _tov_scale(players) == 100.0 and _tov_scale(teams) == 1.0

    class Row:
        TM_TOV_PCT, TS_PCT, GP = 12.5, 0.61, 70
    assert _value(Row, "TM_TOV_PCT", "tov_pct", 100.0) == pytest.approx(0.125)
    assert _value(Row, "TS_PCT", "ts_pct", 100.0) == pytest.approx(0.61)  # only TOV is rescaled
    assert _value(Row, "GP", "gp") == 70
