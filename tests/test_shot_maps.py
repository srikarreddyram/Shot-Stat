"""
Tests for src/inference/shot_maps.py on a synthetic league: everyone shoots
50% at the rim, 40% in the paint and 35% from the top of the arc, and the
players under test are planted to be clearly better or worse somewhere.
"""
import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from src.db.models import Base, Shot
from src.inference.shot_maps import NX, NY, X0, Y0, ZONE_PRIOR, player_shot_map, season_window

SEASON = "2025-26"


def _shots(player, n, made, loc_x, loc_y, zone, start, season=SEASON, playoff=0):
    return [dict(shot_id=f"{player}_{season}_{start + i}", game_id="0022500001", player_id=player,
                 season=season, shot_made=int(i < made), loc_x=float(loc_x), loc_y=float(loc_y),
                 zone=zone, playoff_flag=playoff) for i in range(n)]


@pytest.fixture()
def engine():
    eng = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(eng, tables=[Shot.__table__])
    rows = []
    for season in ("2023-24", "2024-25", SEASON):
        rows += _shots("LEAGUE", 2000, 1000, 0, 10, "Restricted Area", 0, season)
        rows += _shots("LEAGUE", 2000, 700, 0, 255, "Above the Break 3", 10_000, season)
        rows += _shots("LEAGUE", 1000, 400, 0, 90, "In The Paint (Non-RA)", 20_000, season)
    # Rim runner: 300 rim shots at 80%, 200 paint shots at 10%, 2 made threes.
    rows += _shots("BIG", 300, 240, 0, 10, "Restricted Area", 0)
    rows += _shots("BIG", 200, 20, 0, 90, "In The Paint (Non-RA)", 1000)
    rows += _shots("BIG", 2, 2, 0, 255, "Above the Break 3", 2000)
    # Shooter: 400 threes at 45% this season, 400 more the season before;
    # one playoff shot that must not count toward the regular season.
    rows += _shots("SHOOTER", 400, 180, 0, 255, "Above the Break 3", 0)
    rows += _shots("SHOOTER", 400, 180, 0, 255, "Above the Break 3", 0, "2024-25")
    rows += _shots("SHOOTER", 1, 1, 0, 10, "Restricted Area", 5000, playoff=1)
    with eng.begin() as conn:
        conn.execute(Shot.__table__.insert(), rows)
    return eng


def _grid(m, key):
    return np.array([np.nan if v is None else v for v in m[key]]).reshape(NY, NX)


def _at(arr, x_ft, y_ft):
    return arr[int(y_ft - Y0), int(x_ft - X0)]


def test_zone_table_is_exact(engine):
    m = player_shot_map(engine, "BIG", SEASON)
    z = {r["zone"]: r for r in m["zones"]}
    assert (z["Restricted Area"]["fga"], z["Restricted Area"]["fgm"]) == (300, 240)
    assert z["Restricted Area"]["fg_pct"] == pytest.approx(0.8)
    league = (1000 + 240) / 2300
    assert z["Restricted Area"]["league_fg_pct"] == pytest.approx(league, abs=1e-4)
    assert z["Restricted Area"]["vs_league"] == pytest.approx(0.8 - league, abs=1e-4)
    assert z["Right Corner 3"]["fga"] == 0 and z["Right Corner 3"]["fg_pct"] is None
    assert sum(r["share"] for r in m["zones"]) == pytest.approx(1.0, abs=1e-3)


def test_efficiency_is_by_zone_with_shrinkage(engine):
    m = player_shot_map(engine, "BIG", SEASON)
    z = {r["zone"]: r for r in m["zones"]}
    # Good at the rim, bad in the paint next to it: each zone stands alone.
    assert z["Restricted Area"]["vs_league_adjusted"] > 0.1
    assert z["In The Paint (Non-RA)"]["vs_league_adjusted"] < -0.1
    # Adjusted = (makes + K * league) / (attempts + K) - league.
    league = (1000 + 240) / 2300
    k = ZONE_PRIOR["Restricted Area"]
    assert z["Restricted Area"]["vs_league_adjusted"] == pytest.approx(
        (240 + k * league) / (300 + k) - league, abs=1e-4)


def test_thin_samples_are_shrunk_and_faded(engine):
    m = player_shot_map(engine, "BIG", SEASON)
    top = {r["zone"]: r for r in m["zones"]}["Above the Break 3"]
    # 2-for-2 is +65 points raw; shrunk with K=275 it is barely above zero.
    assert top["vs_league"] > 0.6
    assert 0 < top["vs_league_adjusted"] < 0.01
    assert top["support"] == pytest.approx(2 / 20)


def test_frequency_shows_where_he_shoots(engine):
    m = player_shot_map(engine, "SHOOTER", SEASON)
    freq = _grid(m, "frequency")
    assert np.nansum(freq) == pytest.approx(1.0, abs=0.03)
    assert _at(freq, 0, 25) > 50 * max(_at(freq, 0, 1), 1e-9)


def test_three_season_span_pools_shots_and_league(engine):
    one = player_shot_map(engine, "SHOOTER", SEASON)
    three = player_shot_map(engine, "SHOOTER", SEASON, span=3)
    assert one["attempts"] == 400 and three["attempts"] == 800
    assert three["seasons"] == ["2023-24", "2024-25", "2025-26"]
    # More evidence at the same 45%: shrunk less.
    adj = lambda m: {r["zone"]: r for r in m["zones"]}["Above the Break 3"]["vs_league_adjusted"]  # noqa: E731
    assert adj(three) > adj(one) > 0


def test_regular_season_and_playoffs_are_separate(engine):
    assert player_shot_map(engine, "SHOOTER", SEASON)["attempts"] == 400
    assert player_shot_map(engine, "SHOOTER", SEASON, "playoffs")["attempts"] == 1


def test_unknown_player_is_none_and_season_window(engine):
    assert player_shot_map(engine, "NOBODY", SEASON) is None
    assert season_window("2025-26", 3) == ["2023-24", "2024-25", "2025-26"]
    assert season_window("2009-10", 1) == ["2009-10"]


# ── More maps: shot types, clutch, defense, change, teams ───────────────────
from datetime import date  # noqa: E402

from src.db.models import Game, ShotContext, TeamStats  # noqa: E402
from src.inference.shot_maps import (  # noqa: E402
    player_shot_change, player_shot_maps, shot_family, team_shot_map,
)


@pytest.fixture()
def league():
    eng = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(eng, tables=[Shot.__table__, ShotContext.__table__, Game.__table__,
                                          TeamStats.__table__])
    shots, ctx = [], []
    n = 0

    def add(player, k, made, loc_y, zone, subtype, game="G1", home=1, season=SEASON,
            quarter=1, clock=600.0, diff=0, defender=None):
        nonlocal n
        for i in range(k):
            sid = f"{game}_{n}"
            n += 1
            shots.append(dict(shot_id=sid, game_id=game, player_id=player, season=season,
                              shot_made=int(i < made), loc_x=0.0, loc_y=float(loc_y), zone=zone,
                              playoff_flag=0, home_away=home, quarter=quarter, time_remaining=clock,
                              score_diff=diff, defender_id=defender))
            ctx.append(dict(shot_id=sid, game_id=game, shot_subtype=subtype))

    # League: pull-ups from the top go in 30%, set jumpers 40%, rim layups 60%.
    add("LG", 1000, 300, 255, "Above the Break 3", "Pullup Jump shot")
    add("LG", 1000, 400, 255, "Above the Break 3", "Jump Shot")
    add("LG", 1000, 600, 10, "Restricted Area", "Driving Layup Shot")
    # A pull-up shooter: 100 at 40% — above league pull-ups, below set jumpers.
    add("PULLER", 100, 40, 255, "Above the Break 3", "Pullup Jump shot")
    # Clutch: league 40-for-100 at the rim in clutch time; CLOSER 30-for-40.
    add("LG", 100, 40, 10, "Restricted Area", "Driving Layup Shot", quarter=4, clock=120, diff=-3)
    add("CLOSER", 40, 30, 10, "Restricted Area", "Driving Layup Shot", quarter=4, clock=120, diff=2)
    add("CLOSER", 60, 30, 10, "Restricted Area", "Driving Layup Shot")
    # Defense: STOPPER's assignments go 20-for-80 at the rim.
    add("LG", 80, 20, 10, "Restricted Area", "Driving Layup Shot", defender="STOPPER")
    # Teams: BOS hosts NYK in G1, visits in G2. Rows above are all G1/home=1.
    add("BOSP", 30, 15, 10, "Restricted Area", "Layup Shot", game="G2", home=0)
    add("NYKP", 20, 5, 255, "Above the Break 3", "Jump Shot", game="G2", home=1)
    # Change: MOVER went from the rim last season to the arc this season.
    add("MOVER", 200, 100, 10, "Restricted Area", "Layup Shot", season="2024-25", game="G0")
    add("MOVER", 200, 70, 255, "Above the Break 3", "Jump Shot")
    with eng.begin() as conn:
        conn.execute(Shot.__table__.insert(), shots)
        conn.execute(ShotContext.__table__.insert(), ctx)
        conn.execute(Game.__table__.insert(), [
            dict(game_id="G0", date=date(2024, 11, 1), home_team="BOS", away_team="NYK", playoff_flag=0),
            dict(game_id="G1", date=date(2025, 11, 1), home_team="BOS", away_team="NYK", playoff_flag=0),
            dict(game_id="G2", date=date(2025, 11, 3), home_team="NYK", away_team="BOS", playoff_flag=0),
        ])
        conn.execute(TeamStats.__table__.insert(), [
            dict(team_id="T_BOS", season=SEASON, team_abbrev="BOS"),
        ])
    return eng


def test_shot_family_taxonomy():
    assert shot_family("Pullup Jump shot") == "off_dribble"
    assert shot_family("Step Back Bank Jump Shot") == "off_dribble"
    assert shot_family("Jump Shot") == "set_jumpers"
    assert shot_family("Driving Floating Jump Shot") == "floaters_hooks"
    assert shot_family("Turnaround Hook Shot") == "floaters_hooks"
    assert shot_family("Turnaround Fadeaway shot") == "fadeaways"
    assert shot_family("Tip Layup Shot") == "dunks"
    assert shot_family("Cutting Finger Roll Layup Shot") == "layups"
    assert shot_family(None) is None


def test_subsets_are_compared_with_the_same_subset_league_wide(league):
    out = player_shot_maps(league, "PULLER", SEASON, "shot_types")
    pull = next(m for m in out["maps"] if m["key"] == "off_dribble")
    assert pull["attempts"] == 100 and pull["share"] == 1.0
    # Expected is the league's pull-up rate from the top (340/1100), not the
    # all-jumper rate there (740/2100): a 40% pull-up shooter is above par.
    assert pull["expected_fg_pct"] == pytest.approx(340 / 1100, abs=1e-3)
    assert pull["vs_expected"] > 0
    assert next(m for m in out["maps"] if m["key"] == "layups")["attempts"] == 0


def test_clutch_split(league):
    out = player_shot_maps(league, "CLOSER", SEASON, "clutch")
    clutch, rest = out["maps"]
    assert (clutch["key"], clutch["attempts"], rest["attempts"]) == ("clutch", 40, 60)
    # League clutch rim rate: 40 + 30 made of 140.
    assert clutch["expected_fg_pct"] == pytest.approx(70 / 140, abs=1e-3)


def test_defense_map_uses_his_assignments(league):
    out = player_shot_maps(league, "STOPPER", SEASON, "defense")
    m = out["maps"][0]
    assert m["attempts"] == 80 and m["fg_pct"] == pytest.approx(0.25)
    assert m["vs_expected"] < -0.2


def test_change_since_last_season(league):
    out = player_shot_change(league, "MOVER", SEASON)
    assert out["previous_season"] == "2024-25"
    z = {r["zone"]: r for r in out["zones"]}
    assert z["Above the Break 3"]["share_change"] == pytest.approx(1.0)
    assert z["Restricted Area"]["share_change"] == pytest.approx(-1.0)
    change = _grid({"x": out["change"]}, "x")
    assert _at(change, 0, 25) > 0 > _at(change, 0, 1)
    # A wholesale move on 200 shots each season is far beyond noise.
    confidence = _grid({"x": out["confidence"]}, "x")
    assert _at(confidence, 0, 25) == 1.0 and _at(confidence, 0, 1) == 1.0
    assert player_shot_change(league, "PULLER", SEASON) is None  # no previous season


def test_team_offense_and_defense(league):
    off = team_shot_map(league, "T_BOS", SEASON, "offense")
    dfn = team_shot_map(league, "T_BOS", SEASON, "defense")
    # BOS shots: everything at home in G1 plus BOSP's 30 away in G2.
    g1_home = 3000 + 100 + 100 + 80 + 200 + 100
    assert off["team_abbrev"] == "BOS" and off["attempts"] == g1_home + 30
    assert dfn["attempts"] == 20  # NYK's shots in G2
    assert team_shot_map(league, "T_NONE", SEASON) is None


def test_player_shots_come_back_in_game_order_and_thin_evenly(league):
    from src.inference.shot_maps import player_shots
    out = player_shots(league, "BOSP", SEASON)
    assert out["attempts"] == 30 and out["makes"] == 15 and out["games"] == 1
    assert all(len(s) == 4 for s in out["shots"])
    # Two games: MOVER's season has one; LG spans G1 only — use the cap.
    thinned = player_shots(league, "LG", SEASON, limit=100)
    assert len(thinned["shots"]) == 100 and thinned["attempts"] > 100
    assert player_shots(league, "NOBODY", SEASON) is None
