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
