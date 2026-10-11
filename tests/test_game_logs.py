"""
Tests for src/inference/game_logs.py on a two-team league: ONE beats TWO at
home, loses the return game, and meets them again in the playoffs. Player P
(ONE) has every shot of the first game in the shots table, planted out of
order; R (TWO) has three shots in that game.
"""
from datetime import date

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.db.models import Base, Game, PlayerGameLog, Shot, TeamGameLog
from src.inference.game_logs import player_game_detail, player_game_log, team_game_detail, team_game_log

SEASON = "2024-25"
BOX = dict(oreb=1, dreb=4, stl=1, blk=0, tov=2, pf=2, ftm=2, fta=2)


@pytest.fixture()
def engine():
    eng = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(eng, tables=[Game.__table__, Shot.__table__,
                                          PlayerGameLog.__table__, TeamGameLog.__table__])
    Session = sessionmaker(bind=eng)
    games = [  # game, date, home, away, ONE pts, TWO pts, playoffs
        ("G1", date(2025, 1, 1), "ONE", "TWO", 110, 100, 0),
        ("G2", date(2025, 1, 3), "TWO", "ONE", 95, 101, 0),
        ("G3", date(2025, 4, 20), "ONE", "TWO", 120, 90, 1),
    ]
    with Session() as s:
        for gid, day, home, away, one, two, po in games:
            s.add(Game(game_id=gid, date=day, home_team=home, away_team=away, playoff_flag=po))
            st = "Playoffs" if po else "Regular Season"
            for tid, abbr, opp, pts, opp_pts in (("T1", "ONE", "TWO", one, two), ("T2", "TWO", "ONE", two, one)):
                s.add(TeamGameLog(team_id=tid, game_id=gid, season=SEASON, season_type=st, game_date=day,
                                  team_abbrev=abbr, opp_abbrev=opp, home=int(home == abbr),
                                  win=int(pts > opp_pts), pts=pts, reb=40, ast=25, fgm=40,
                                  fga=88 if tid == "T1" else 90, fg3m=12, fg3a=35,
                                  plus_minus=float(pts - opp_pts), **BOX))
        # P: 10 pts on 4-8 in G1, 30 on 12-20 in G2, 20 on 8-15 in G3 (playoffs).
        for gid, day, pts, fgm, fga, ast, st in (
            ("G1", date(2025, 1, 1), 10, 4, 8, 3, "Regular Season"),
            ("G2", date(2025, 1, 3), 30, 12, 20, 9, "Regular Season"),
            ("G3", date(2025, 4, 20), 20, 8, 15, 5, "Playoffs"),
        ):
            s.add(PlayerGameLog(player_id="P", game_id=gid, season=SEASON, season_type=st, game_date=day,
                                player_name="Pat", team_id="T1", team_abbrev="ONE", opp_abbrev="TWO",
                                home=int(gid != "G2"), win=int(gid != "G2"), minutes=30.0, pts=pts, reb=5, ast=ast,
                                fgm=fgm, fga=fga, fg3m=1, fg3a=3, plus_minus=4.0, **BOX))
        s.add(PlayerGameLog(player_id="Q", game_id="G1", season=SEASON, season_type="Regular Season",
                            game_date=date(2025, 1, 1), player_name="Quinn", team_id="T1", team_abbrev="ONE",
                            opp_abbrev="TWO", home=1, win=1, minutes=20.0, pts=14, reb=2, ast=1,
                            fgm=6, fga=9, fg3m=0, fg3a=1, plus_minus=8.0, **BOX))
        s.add(PlayerGameLog(player_id="R", game_id="G1", season=SEASON, season_type="Regular Season",
                            game_date=date(2025, 1, 1), player_name="Ray", team_id="T2", team_abbrev="TWO",
                            opp_abbrev="ONE", home=0, win=0, minutes=34.0, pts=25, reb=7, ast=2,
                            fgm=9, fga=18, fg3m=3, fg3a=7, plus_minus=-10.0, **BOX))
        s.commit()
    # P's eight shots in G1, inserted out of order: (quarter, seconds left, made, x, y, zone).
    p_shots = [
        (3, 300, 1, 0, 10, "Restricted Area"), (1, 700, 0, 0, 255, "Above the Break 3"),
        (1, 200, 1, 0, 90, "In The Paint (Non-RA)"), (5, 100, 1, -230, 20, "Left Corner 3"),
        (2, 500, 0, 0, 10, "Restricted Area"), (4, 60, 0, 150, 150, "Mid-Range"),
        (2, 650, 1, 0, 10, "Restricted Area"), (4, 400, 0, 0, 255, "Above the Break 3"),
    ]
    rows = [dict(shot_id=f"P_{i}", game_id="G1", player_id="P", season=SEASON, shot_made=m,
                 loc_x=float(x), loc_y=float(y), zone=z, quarter=q, time_remaining=float(t),
                 home_away=1, playoff_flag=0) for i, (q, t, m, x, y, z) in enumerate(p_shots)]
    rows += [dict(shot_id=f"R_{i}", game_id="G1", player_id="R", season=SEASON, shot_made=int(i == 0),
                  loc_x=0.0, loc_y=255.0, zone="Above the Break 3", quarter=1, time_remaining=600.0 - i,
                  home_away=0, playoff_flag=0) for i in range(3)]
    with eng.begin() as conn:
        conn.execute(Shot.__table__.insert(), rows)
    return eng


def test_player_log_is_newest_first_with_the_final_score(engine):
    log = player_game_log(engine, "P", SEASON)
    assert [g["game_id"] for g in log["games"]] == ["G2", "G1"]  # the playoff game is its own list
    g2 = log["games"][0]
    assert (g2["team_score"], g2["opponent_score"], g2["win"]) == (95, 101, False)
    assert g2["home"] is False and g2["opponent"] == "TWO"
    assert g2["fg_pct"] == pytest.approx(12 / 20)
    assert [g["game_id"] for g in player_game_log(engine, "P", SEASON, "playoffs")["games"]] == ["G3"]
    assert player_game_log(engine, "P", "2019-20") is None


def test_player_averages_shoot_from_totals_and_highs_name_the_game(engine):
    log = player_game_log(engine, "P", SEASON)
    avg = log["averages"]
    assert avg["gp"] == 2 and avg["pts"] == 20.0 and avg["min"] == 30.0
    assert avg["fg_pct"] == pytest.approx(16 / 28, abs=1e-4)  # not the mean of 50% and 60%
    assert log["highs"]["pts"] == {"value": 30, "games": ["G2"]}
    assert log["highs"]["reb"] == {"value": 5, "games": ["G2", "G1"]}  # a tie names every game


def test_player_game_detail_replays_the_shots_in_the_order_taken(engine):
    d = player_game_detail(engine, "P", "G1")
    # Quarter by quarter, and within a quarter by the clock running down.
    assert [s[3] for s in d["shots"]] == [1, 1, 2, 2, 3, 4, 4, 5]
    assert d["shots"][0] == [0, 255, 0, 1] and d["shots"][1] == [0, 90, 1, 1]
    assert d["shots"][2][2] == 1 and d["shots"][3][2] == 0  # 650 s left came before 500 s left
    # The shots are exactly the box score's field goals, R's are not among them.
    assert (d["attempts"], d["makes"]) == (d["game"]["fga"], d["game"]["fgm"]) == (8, 4)
    assert d["periods"] == 5  # a shot in overtime adds a period
    zones = {z["zone"]: z for z in d["zones"]}
    assert (zones["Restricted Area"]["fga"], zones["Restricted Area"]["fgm"]) == (3, 2)
    assert zones["Right Corner 3"]["fga"] == 0 and zones["Right Corner 3"]["fg_pct"] is None
    # The season line per zone: every shot P has is in this game, R's are not P's.
    assert (zones["Restricted Area"]["season_fga"], zones["Restricted Area"]["season_fgm"]) == (3, 2)
    assert zones["Above the Break 3"]["season_fg_pct"] == 0.0
    assert d["season_averages"]["pts"] == 20.0 and d["season"] == SEASON


def test_player_game_detail_without_shots_or_for_a_game_not_played(engine):
    d = player_game_detail(engine, "P", "G2")
    assert d["game"]["pts"] == 30 and d["shots"] == [] and d["attempts"] == 0 and d["periods"] == 4
    assert player_game_detail(engine, "Q", "G2") is None


def test_team_log_scores_from_plus_minus(engine):
    log = team_game_log(engine, "T2", SEASON)
    assert [g["game_id"] for g in log["games"]] == ["G2", "G1"]
    g1 = log["games"][1]
    assert (g1["team_score"], g1["opponent_score"], g1["win"]) == (100, 110, False)
    assert log["averages"]["pts"] == 100.5 and "min" not in log["averages"]


def test_team_game_detail_offense_and_defense_are_each_sides_shots(engine):
    off = team_game_detail(engine, "T1", "G1", "offense")
    assert (off["attempts"], off["makes"]) == (8, 4)  # P's shots: Q's are not in the shots table
    assert off["opponent_box"]["team"] == "TWO" and off["opponent_box"]["fga"] == 90
    assert [p["name"] for p in off["players"]] == ["Quinn", "Pat"]  # by points
    dfn = team_game_detail(engine, "T1", "G1", "defense")
    assert (dfn["attempts"], dfn["makes"]) == (3, 1)  # R's shots
    assert team_game_detail(engine, "T2", "G1", "offense")["attempts"] == 3
    assert team_game_detail(engine, "T1", "G9") is None
    with pytest.raises(ValueError):
        team_game_detail(engine, "T1", "G1", "both")
