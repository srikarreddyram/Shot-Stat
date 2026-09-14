"""
Tests for src/inference/career_stats.py.

The whole risk in this module is that a wrong aggregation still produces a
plausible-looking number. A career FG% computed as the mean of per-season
percentages, or a career blocks-per-game computed the same way, is off by
exactly the amount the player's workload varied — which is invisible unless
you check the arithmetic against a hand-worked example. Every test here does
that rather than asserting "is not None".
"""
import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.db.database import register_unaccent
from src.db.models import (
    Base, DefenderStats, PlayerDefensiveActivity, PlayerPlayType,
    PlayerShotProfile, PlayerTrackingStats, PlayerZoneStats, REGULAR_SEASON,
)
from src.inference import career_stats as cs


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


def _stat(panel, key):
    for section in panel["sections"]:
        for s in section["stats"]:
            if s["key"] == key:
                return s
    return None


def test_career_per_game_average_is_workload_weighted_not_a_mean_of_seasons(engine):
    """Two seasons: 80 games at 1.0 blocks, then 10 games at 5.0.

    The mean of the season averages is 3.0. The truth is
    (80x1 + 10x5) / 90 = 1.44. A player who had one short, hot stretch must
    not come out looking like a career shot-blocker.
    """
    Session = sessionmaker(bind=engine)
    with Session() as s:
        s.add(PlayerDefensiveActivity(player_id="P1", season="2023-24", gp=80, min_per_game=30.0, stl=1.0, blk=1.0, deflections=2.0))
        s.add(PlayerDefensiveActivity(player_id="P1", season="2024-25", gp=10, min_per_game=30.0, stl=1.0, blk=5.0, deflections=2.0))
        s.commit()

    panel = cs.career_panel(engine, "P1", "2024-25")
    blocks = _stat(panel, "blocks")

    assert blocks["career_avg"] == pytest.approx(130 / 90)   # 1.444, NOT 3.0
    assert blocks["career_avg"] != pytest.approx(3.0)
    assert blocks["career_total"] == pytest.approx(130)      # 80x1 + 10x5
    assert blocks["current"] == pytest.approx(5.0)           # 2024-25
    assert blocks["previous"] == pytest.approx(1.0)          # 2023-24
    assert blocks["seasons"] == 2
    assert blocks["first_season"] == "2023-24"
    assert blocks["last_season"] == "2024-25"


def test_career_fg_pct_allowed_is_recomputed_from_totals_not_averaged(engine):
    """Season A: 10-for-100 (10%). Season B: 8-for-10 (80%).

    Mean of the percentages is 45%. The truth is 18/110 = 16.4%.
    """
    Session = sessionmaker(bind=engine)
    with Session() as s:
        s.add(DefenderStats(player_id="P1", season="2023-24", defense_category="Less Than 6Ft",
                            season_type=REGULAR_SEASON, gp=70, d_fgm=10, d_fga=100,
                            d_fg_pct=0.10, normal_fg_pct=0.60, pct_plusminus=-0.50))
        s.add(DefenderStats(player_id="P1", season="2024-25", defense_category="Less Than 6Ft",
                            season_type=REGULAR_SEASON, gp=5, d_fgm=8, d_fga=10,
                            d_fg_pct=0.80, normal_fg_pct=0.60, pct_plusminus=0.20))
        s.commit()

    panel = cs.career_panel(engine, "P1", "2024-25")
    rate = _stat(panel, "def_rim_fg_pct")

    assert rate["career_avg"] == pytest.approx(18 / 110)  # 0.1636
    assert rate["career_avg"] != pytest.approx(0.45)
    assert rate["current"] == pytest.approx(0.80)
    assert rate["previous"] == pytest.approx(0.10)
    # A rate has no meaningful sum — that would be "88%" of nothing.
    assert rate["career_total"] is None
    # ...its volume is carried as its own counting stat instead.
    assert _stat(panel, "def_rim_fga")["career_total"] == pytest.approx(110)
    assert _stat(panel, "def_rim_fgm")["career_total"] == pytest.approx(18)


def test_versus_normal_is_weighted_by_attempts(engine):
    """-0.50 over 100 attempts and +0.20 over 10 is not -0.15."""
    Session = sessionmaker(bind=engine)
    with Session() as s:
        s.add(DefenderStats(player_id="P1", season="2023-24", defense_category="Less Than 6Ft",
                            season_type=REGULAR_SEASON, gp=70, d_fgm=10, d_fga=100,
                            d_fg_pct=0.10, normal_fg_pct=0.60, pct_plusminus=-0.50))
        s.add(DefenderStats(player_id="P1", season="2024-25", defense_category="Less Than 6Ft",
                            season_type=REGULAR_SEASON, gp=5, d_fgm=8, d_fga=10,
                            d_fg_pct=0.80, normal_fg_pct=0.60, pct_plusminus=0.20))
        s.commit()

    pm = _stat(cs.career_panel(engine, "P1", "2024-25"), "def_rim_pm")
    expected = ((-0.50 * 100) + (0.20 * 10)) / 110
    assert pm["career_avg"] == pytest.approx(expected)
    assert pm["career_avg"] != pytest.approx(-0.15)


def test_playoff_rows_are_excluded_from_career_figures(engine):
    """The table holds both season types. A career regular-season number must
    not quietly absorb a postseason sample — that conflation is exactly the
    bug that made this column untrustworthy in the first place."""
    Session = sessionmaker(bind=engine)
    with Session() as s:
        s.add(DefenderStats(player_id="P1", season="2024-25", defense_category="Overall",
                            season_type=REGULAR_SEASON, gp=70, d_fgm=300, d_fga=700,
                            d_fg_pct=0.4286, normal_fg_pct=0.45, pct_plusminus=-0.02))
        s.add(DefenderStats(player_id="P1", season="2024-25", defense_category="Overall",
                            season_type="Playoffs", gp=15, d_fgm=90, d_fga=150,
                            d_fg_pct=0.60, normal_fg_pct=0.45, pct_plusminus=0.15))
        s.commit()

    rate = _stat(cs.career_panel(engine, "P1", "2024-25"), "def_overall_fg_pct")
    assert rate["career_avg"] == pytest.approx(300 / 700)
    assert _stat(cs.career_panel(engine, "P1", "2024-25"), "def_overall_fga")["career_total"] == pytest.approx(700)


def test_zone_shooting_career_rate_ignores_zero_attempt_seasons(engine):
    """A season with no attempts in a zone contributes no attempts and no
    makes, and must not drag the career rate toward zero."""
    Session = sessionmaker(bind=engine)
    with Session() as s:
        s.add(PlayerZoneStats(player_id="P1", season="2023-24", zone="Left Corner 3", fgm=0, fga=0, fg_pct=0.0))
        s.add(PlayerZoneStats(player_id="P1", season="2024-25", zone="Left Corner 3", fgm=6, fga=10, fg_pct=0.60))
        s.commit()

    rate = _stat(cs.career_panel(engine, "P1", "2024-25"), "zone_fg_pct_left_corner3")
    assert rate["career_avg"] == pytest.approx(0.60)
    assert rate["previous"] is None  # never shot one, rather than "0%"
    assert _stat(cs.career_panel(engine, "P1", "2024-25"), "zone_fga_left_corner3")["career_total"] == pytest.approx(10)


def test_stats_with_no_data_are_omitted_rather_than_shown_as_zero(engine):
    Session = sessionmaker(bind=engine)
    with Session() as s:
        s.add(PlayerDefensiveActivity(player_id="P1", season="2024-25", gp=70, min_per_game=30.0, stl=1.0, blk=1.0, deflections=2.0))
        s.commit()

    panel = cs.career_panel(engine, "P1", "2024-25")
    assert _stat(panel, "blocks") is not None
    assert _stat(panel, "def_rim_fg_pct") is None  # no defender rows seeded
    assert all(section["stats"] for section in panel["sections"])


def test_panel_is_json_safe_and_reports_its_own_coverage(engine):
    Session = sessionmaker(bind=engine)
    with Session() as s:
        s.add(PlayerDefensiveActivity(player_id="P1", season="2022-23", gp=70, min_per_game=30.0, stl=1.0, blk=1.0, deflections=2.0))
        s.add(PlayerDefensiveActivity(player_id="P1", season="2024-25", gp=70, min_per_game=30.0, stl=1.0, blk=1.0, deflections=2.0))
        s.commit()

    panel = cs.career_panel(engine, "P1", "2024-25")
    json.loads(json.dumps(panel))  # no NaN, no numpy scalars

    assert panel["previous_season"] == "2023-24"
    assert panel["first_season"] == "2022-23"
    assert panel["last_season"] == "2024-25"
    # Coverage is reported so a client can say "2 seasons tracked" rather than
    # implying this is the player's whole career.
    assert panel["seasons_covered"] == 2
    assert _stat(panel, "blocks")["previous"] is None  # did not play 2023-24


def test_unknown_player_returns_an_empty_panel_not_an_error(engine):
    panel = cs.career_panel(engine, "NOBODY", "2024-25")
    assert panel["sections"] == []
    assert panel["seasons_covered"] == 0


def test_counting_stats_show_the_season_total_not_a_per_game_rate(engine):
    """"Shots defended" in the season column must be the season's total. A
    per-game figure there reads as nonsense beside the career total in the
    next column — 17 next to 14,174."""
    Session = sessionmaker(bind=engine)
    with Session() as s:
        s.add(DefenderStats(player_id="P1", season="2023-24", defense_category="Overall",
                            season_type=REGULAR_SEASON, gp=70, d_fgm=400, d_fga=900,
                            d_fg_pct=0.444, normal_fg_pct=0.46, pct_plusminus=-0.016))
        s.add(DefenderStats(player_id="P1", season="2024-25", defense_category="Overall",
                            season_type=REGULAR_SEASON, gp=72, d_fgm=500, d_fga=1100,
                            d_fg_pct=0.4545, normal_fg_pct=0.46, pct_plusminus=-0.005))
        s.commit()

    fga = _stat(cs.career_panel(engine, "P1", "2024-25"), "def_overall_fga")
    assert fga["current"] == pytest.approx(1100)          # the season, not 1100/72
    assert fga["previous"] == pytest.approx(900)
    assert fga["career_total"] == pytest.approx(2000)
    assert fga["career_avg"] == pytest.approx(1000)       # per season

    # A per-game stat keeps per-game semantics in the same columns.
    minutes_panel = cs.career_panel(engine, "P1", "2024-25")
    assert _stat(minutes_panel, "def_overall_fgm")["current"] == pytest.approx(500)


def test_shot_context_career_rate_recomputed_from_totals(engine):
    """Season A: 5-for-50 wide open (10%). Season B: 40-for-50 (80%).

    Mean of the percentages is 45%. The truth is 45/100 = 45% here too by
    coincidence of these numbers — use unequal volumes so the two diverge.
    """
    Session = sessionmaker(bind=engine)
    with Session() as s:
        s.add(PlayerShotProfile(player_id="P1", season="2023-24", split_type="def_dist",
                                split_value="6+ Feet - Wide Open", gp=70,
                                fga_frequency=0.2, fgm=10, fga=100, fg_pct=0.10))
        s.add(PlayerShotProfile(player_id="P1", season="2024-25", split_type="def_dist",
                                split_value="6+ Feet - Wide Open", gp=70,
                                fga_frequency=0.2, fgm=8, fga=10, fg_pct=0.80))
        s.commit()

    panel = cs.career_panel(engine, "P1", "2024-25")
    rate = _stat(panel, "shotctx_wide_open_fg_pct")
    assert rate["career_avg"] == pytest.approx(18 / 110)  # NOT the 45% mean
    assert rate["career_avg"] != pytest.approx(0.45)
    assert rate["current"] == pytest.approx(0.80)
    assert rate["previous"] == pytest.approx(0.10)
    assert rate["career_total"] is None  # a rate has no meaningful sum
    assert _stat(panel, "shotctx_wide_open_fga")["career_total"] == pytest.approx(110)


def test_potential_assists_use_workload_weighted_career_average(engine):
    """Same shape as the blocks test: a short hot stretch must not dominate
    the workload-weighted career figure."""
    Session = sessionmaker(bind=engine)
    with Session() as s:
        s.add(PlayerTrackingStats(player_id="P1", season="2023-24", gp=80, min_per_game=30.0,
                                  ast=5.0, potential_ast=8.0, secondary_ast=1.0,
                                  ast_points_created=12.0, passes_made=50.0))
        s.add(PlayerTrackingStats(player_id="P1", season="2024-25", gp=10, min_per_game=30.0,
                                  ast=5.0, potential_ast=20.0, secondary_ast=1.0,
                                  ast_points_created=12.0, passes_made=50.0))
        s.commit()

    panel = cs.career_panel(engine, "P1", "2024-25")
    pot = _stat(panel, "potential_ast")
    assert pot["career_avg"] == pytest.approx((80 * 8.0 + 10 * 20.0) / 90)
    assert pot["career_avg"] != pytest.approx(14.0)  # not the plain mean of 8 and 20
    assert pot["current"] == pytest.approx(20.0)
    assert pot["career_total"] == pytest.approx(80 * 8.0 + 10 * 20.0)


def test_assist_to_pass_rate_is_recomputed_from_totals_not_averaged(engine):
    """5 ast / 50 passes (10%) one season, 8 ast / 10 passes (80%) another —
    same trap as FG% allowed: the true rate is not the mean of the two."""
    Session = sessionmaker(bind=engine)
    with Session() as s:
        s.add(PlayerTrackingStats(player_id="P1", season="2023-24", gp=1,
                                  ast=5.0, passes_made=50.0, potential_ast=None,
                                  secondary_ast=None, ast_points_created=None))
        s.add(PlayerTrackingStats(player_id="P1", season="2024-25", gp=1,
                                  ast=8.0, passes_made=10.0, potential_ast=None,
                                  secondary_ast=None, ast_points_created=None))
        s.commit()

    rate = _stat(cs.career_panel(engine, "P1", "2024-25"), "ast_to_pass_pct")
    assert rate["career_avg"] == pytest.approx((5 + 8) / (50 + 10))
    assert rate["career_avg"] != pytest.approx(0.45)  # not the plain mean of 10% and 80%


def test_shot_context_and_passing_stats_are_grouped_and_omit_no_data(engine):
    Session = sessionmaker(bind=engine)
    with Session() as s:
        s.add(PlayerDefensiveActivity(player_id="P1", season="2024-25", gp=70, min_per_game=30.0, stl=1.0, blk=1.0, deflections=2.0))
        s.commit()

    panel = cs.career_panel(engine, "P1", "2024-25")
    groups = {s["group"] for s in panel["sections"]}
    # Nothing was seeded for shot_difficulty/passing, so neither section
    # should appear at all — an empty section is worse than no section,
    # since it implies data was checked and came back zero.
    assert "shot_difficulty" not in groups
    assert "passing" not in groups
    assert "defense_activity" in groups


def test_hustle_stats_career_average_is_workload_weighted(engine):
    Session = sessionmaker(bind=engine)
    with Session() as s:
        s.add(PlayerDefensiveActivity(player_id="P1", season="2023-24", gp=80,
                                      min_per_game=30.0, screen_ast=2.0, charges_drawn=0.1))
        s.add(PlayerDefensiveActivity(player_id="P1", season="2024-25", gp=10,
                                      min_per_game=30.0, screen_ast=6.0, charges_drawn=0.5))
        s.commit()

    panel = cs.career_panel(engine, "P1", "2024-25")
    screen = _stat(panel, "screen_ast")
    assert screen["career_avg"] == pytest.approx((80 * 2.0 + 10 * 6.0) / 90)
    assert screen["career_avg"] != pytest.approx(4.0)  # not the plain mean
    assert screen["current"] == pytest.approx(6.0)
    groups = {sec["group"] for sec in panel["sections"]}
    assert "hustle_offense" in groups  # screen_ast is offense-side hustle


def test_play_type_career_ppp_recomputed_from_totals(engine):
    """Season A: 90 pts on 100 isolation possessions (0.90 PPP). Season B:
    12 pts on 10 possessions (1.20 PPP). The mean of the two PPP figures is
    1.05; the truth, from totals, is 102/110 = 0.927."""
    Session = sessionmaker(bind=engine)
    with Session() as s:
        s.add(PlayerPlayType(player_id="P1", season="2023-24", play_type="Isolation",
                             gp=70, poss=100, poss_pct=0.15, pts=90, fgm=35, fga=80,
                             fg_pct=0.4375, efg_pct=0.45, ppp=0.90, percentile=0.4))
        s.add(PlayerPlayType(player_id="P1", season="2024-25", play_type="Isolation",
                             gp=10, poss=10, poss_pct=0.15, pts=12, fgm=5, fga=9,
                             fg_pct=0.556, efg_pct=0.56, ppp=1.20, percentile=0.7))
        s.commit()

    ppp = _stat(cs.career_panel(engine, "P1", "2024-25"), "playtype_isolation_ppp")
    assert ppp["career_avg"] == pytest.approx(102 / 110)
    assert ppp["career_avg"] != pytest.approx(1.05)  # not the plain mean
    assert ppp["current"] == pytest.approx(1.20)
    assert ppp["previous"] == pytest.approx(0.90)
    assert ppp["career_total"] is None  # a rate has no meaningful sum

    poss = _stat(cs.career_panel(engine, "P1", "2024-25"), "playtype_isolation_poss")
    assert poss["career_total"] == pytest.approx(110)

    fg = _stat(cs.career_panel(engine, "P1", "2024-25"), "playtype_isolation_fg_pct")
    assert fg["career_avg"] == pytest.approx(40 / 89)

    groups = {sec["group"] for sec in cs.career_panel(engine, "P1", "2024-25")["sections"]}
    assert "play_type" in groups


def test_series_reports_each_seasons_own_value_not_a_running_average(engine):
    """A trend chart must be able to show a dip: three seasons of blocks
    that go up, down, up. A cumulative running average would smooth the
    middle season's dip away entirely."""
    Session = sessionmaker(bind=engine)
    with Session() as s:
        s.add(PlayerDefensiveActivity(player_id="P1", season="2022-23", gp=70, min_per_game=30.0, blk=2.0))
        s.add(PlayerDefensiveActivity(player_id="P1", season="2023-24", gp=70, min_per_game=30.0, blk=0.5))
        s.add(PlayerDefensiveActivity(player_id="P1", season="2024-25", gp=70, min_per_game=30.0, blk=3.0))
        s.commit()

    blocks = _stat(cs.career_panel(engine, "P1", "2024-25"), "blocks")
    assert blocks["series"] == [
        {"season": "2022-23", "value": 2.0},
        {"season": "2023-24", "value": 0.5},
        {"season": "2024-25", "value": 3.0},
    ]


def test_series_omits_seasons_with_no_value_rather_than_a_zero(engine):
    Session = sessionmaker(bind=engine)
    with Session() as s:
        s.add(PlayerDefensiveActivity(player_id="P1", season="2022-23", gp=70, min_per_game=30.0, blk=2.0))
        s.add(PlayerDefensiveActivity(player_id="P1", season="2023-24", gp=70, min_per_game=30.0, blk=None))
        s.commit()

    blocks = _stat(cs.career_panel(engine, "P1", "2023-24"), "blocks")
    assert blocks["series"] == [{"season": "2022-23", "value": 2.0}]


def test_series_for_a_count_stat_uses_the_season_total(engine):
    Session = sessionmaker(bind=engine)
    with Session() as s:
        s.add(DefenderStats(player_id="P1", season="2024-25", defense_category="Overall",
                            season_type=REGULAR_SEASON, gp=70, d_fgm=300, d_fga=700,
                            d_fg_pct=0.4286, normal_fg_pct=0.45, pct_plusminus=-0.02))
        s.commit()

    fga = _stat(cs.career_panel(engine, "P1", "2024-25"), "def_overall_fga")
    assert fga["series"] == [{"season": "2024-25", "value": 700}]


def test_sections_carry_a_side_for_splitting_the_career_page(engine):
    Session = sessionmaker(bind=engine)
    with Session() as s:
        s.add(PlayerDefensiveActivity(player_id="P1", season="2024-25", gp=70, min_per_game=30.0, blk=1.0))
        s.commit()

    panel = cs.career_panel(engine, "P1", "2024-25")
    sides = {sec["group"]: sec["side"] for sec in panel["sections"]}
    assert sides["defense_activity"] == "defense"


def test_double_doubles_are_a_season_total_not_multiplied_by_games(engine):
    """dd2 is already a season count on the source row. Treating it like a
    per-game rate (value * gp) would turn 42 double-doubles across 70 games
    into a nonsense 2,940."""
    Session = sessionmaker(bind=engine)
    with Session() as s:
        s.add(PlayerDefensiveActivity(player_id="P1", season="2023-24", gp=80, dd2=68))
        s.add(PlayerDefensiveActivity(player_id="P1", season="2024-25", gp=70, dd2=42))
        s.commit()

    dd2 = _stat(cs.career_panel(engine, "P1", "2024-25"), "dd2")
    assert dd2["current"] == 42
    assert dd2["previous"] == 68
    assert dd2["career_total"] == 110  # 68 + 42, not (68*80)+(42*70)


def test_box_score_stats_are_workload_weighted_and_grouped(engine):
    Session = sessionmaker(bind=engine)
    with Session() as s:
        s.add(PlayerDefensiveActivity(player_id="P1", season="2023-24", gp=80, pts=20.0, reb=5.0))
        s.add(PlayerDefensiveActivity(player_id="P1", season="2024-25", gp=10, pts=30.0, reb=5.0))
        s.commit()

    panel = cs.career_panel(engine, "P1", "2024-25")
    pts = _stat(panel, "pts")
    assert pts["career_avg"] == pytest.approx((80 * 20.0 + 10 * 30.0) / 90)
    assert pts["current"] == pytest.approx(30.0)

    sides = {sec["group"]: sec["side"] for sec in panel["sections"]}
    assert sides["shooting"] == "offense"
    assert sides["defense_activity"] == "defense"
    reb_section = next(sec for sec in panel["sections"] if sec["group"] == "defense_activity")
    assert any(st["key"] == "reb" for st in reb_section["stats"])
