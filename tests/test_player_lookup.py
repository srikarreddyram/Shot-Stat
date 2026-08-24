"""
Tests for src/inference/player_lookup.py — the measured -> prior-season ->
position-prior fallback chain that lets the app show "who's active right
now" (a season with a roster but zero games played) while still resolving
real stats wherever they exist.
"""
from datetime import date

import pytest

from src.common.position_bucket import position_bucket, height_band, fine_bucket
from src.db.models import DefenderStats, Game, Player, PlayerZoneStats, PositionPrior, Shot
from src.inference.player_lookup import resolve_defender_stats, resolve_player_stats

EARLY_SEASON = "2023-24"
LATE_SEASON = "2024-25"
FUTURE_SEASON = "2026-27"  # rostered, no games played yet


def test_position_bucket_mapping():
    assert position_bucket("PG") == "G"
    assert position_bucket("SG") == "G"
    assert position_bucket("SF") == "F"
    assert position_bucket("PF") == "F"
    assert position_bucket("C") == "C"
    assert position_bucket("F-C") == "F"   # primary (first-listed) position wins
    assert position_bucket("C-F") == "C"
    assert position_bucket(None) is None
    assert position_bucket("") is None
    assert position_bucket("Coach") is None  # unrecognized string, not a position


def test_height_band_and_fine_bucket():
    assert height_band(90.0, "C") == "tall"    # above the 83in C median
    assert height_band(78.0, "C") == "short"   # at/below the median
    assert height_band(None, "C") is None
    assert height_band(80.0, None) is None     # unknown bucket, no threshold to compare against

    assert fine_bucket(90.0, "C") == "C-tall"
    assert fine_bucket(78.0, "PF") == "F-short"
    assert fine_bucket(None, "C") is None   # no height -> no finer bucket, caller falls back to "C"
    assert fine_bucket(90.0, None) is None


@pytest.fixture()
def resolver_db(db_engine, db_session):
    """
    Three players sharing one season history, exercising all three
    resolver tiers:

      P_EXACT  — has real zone/defender data in EARLY_SEASON. Queried at
                 EARLY_SEASON directly -> tier 1 (exact-season measured).

      P_STALE  — has real zone/defender data only in EARLY_SEASON, plus a
                 roster-only row (no stats) in LATE_SEASON. Queried at
                 LATE_SEASON -> tier 2 (falls back to their own prior
                 season, EARLY_SEASON).

      P_ROOKIE — has a roster-only row in FUTURE_SEASON and has NEVER
                 appeared in `shots`/`player_zone_stats`/`defender_stats`
                 in any season. Queried at FUTURE_SEASON -> tier 3
                 (position prior).

      P_UNKNOWN_POS — same as P_ROOKIE but with position=None, to verify
                 the resolver degrades gracefully (no crash, empty/None
                 fields) when no bucket can be determined.

    A PositionPrior row for bucket "C" is seeded so P_ROOKIE (position="C")
    has somewhere to fall back to.
    """
    game = Game(game_id="G1", date=date(2023, 11, 1), home_team="LAL", away_team="BOS", playoff_flag=0)

    p_exact = Player(
        player_id="P_EXACT", season=EARLY_SEASON, name="Exact Match", position="C",
        height=82.0, weight=240.0, wingspan=86.0,
        career_fg_pct=0.55, career_3p_pct=0.15, season_fg_pct=0.56, ast=1.0, tov=1.2, ft_pct=0.7,
    )
    p_stale_early = Player(
        player_id="P_STALE", season=EARLY_SEASON, name="Stale Vet", position="F",
        height=80.0, weight=220.0, wingspan=83.0,
        career_fg_pct=0.48, career_3p_pct=0.35, season_fg_pct=0.47, ast=2.0, tov=1.5, ft_pct=0.8,
    )
    # Roster-only row for the same player in the season with no games played yet.
    p_stale_late = Player(
        player_id="P_STALE", season=LATE_SEASON, name="Stale Vet", position="F",
    )
    p_rookie = Player(
        player_id="P_ROOKIE", season=FUTURE_SEASON, name="Brand New Rookie", position="C",
    )
    # Tall enough to land in the "C-tall" fine bucket (threshold is 83in).
    p_tall_rookie = Player(
        player_id="P_TALL_ROOKIE", season=FUTURE_SEASON, name="Towering Rookie", position="C", height=89.0,
    )
    p_unknown_pos = Player(
        player_id="P_UNKNOWN_POS", season=FUTURE_SEASON, name="No Position Listed", position=None,
    )

    db_session.add_all([game, p_exact, p_stale_early, p_stale_late, p_rookie, p_tall_rookie, p_unknown_pos])
    db_session.flush()

    zone_rows = [
        PlayerZoneStats(player_id="P_EXACT", season=EARLY_SEASON, zone="Restricted Area", fgm=100, fga=150, fg_pct=0.667),
        PlayerZoneStats(player_id="P_STALE", season=EARLY_SEASON, zone="Restricted Area", fgm=80, fga=140, fg_pct=0.571),
    ]
    def_rows = [
        DefenderStats(player_id="P_EXACT", season=EARLY_SEASON, defense_category="Overall",
                       gp=70, freq=1.0, d_fgm=300, d_fga=650, d_fg_pct=0.46, normal_fg_pct=0.47, pct_plusminus=-0.01),
        DefenderStats(player_id="P_STALE", season=EARLY_SEASON, defense_category="Overall",
                       gp=65, freq=1.0, d_fgm=320, d_fga=680, d_fg_pct=0.47, normal_fg_pct=0.47, pct_plusminus=0.00),
    ]
    # A shot is what actually defines "has real data for this season" for
    # the players resolver in some code paths — include one for completeness,
    # though resolve_player_stats keys off player_zone_stats directly.
    shot_rows = [
        Shot(shot_id="S1", game_id="G1", player_id="P_EXACT", season=EARLY_SEASON,
             shot_made=1, loc_x=0, loc_y=10, shot_distance=1.0, shot_type="2PT Field Goal",
             zone="Restricted Area", shot_angle=90.0, quarter=1, time_remaining=500,
             score_diff=0, home_away=1, playoff_flag=0),
    ]

    position_prior_rows = [
        PositionPrior(position_bucket="C", stat_type="zone", stat_key="Restricted Area",
                       fg_pct=0.64, sample_size=1000, computed_through_season=EARLY_SEASON),
        PositionPrior(position_bucket="C", stat_type="defense", stat_key="Overall",
                       d_fg_pct=0.47, pct_plusminus=0.00, sample_size=100, computed_through_season=EARLY_SEASON),
        PositionPrior(position_bucket="C", stat_type="overall", stat_key="career",
                       fg_pct=0.50, fg3_pct=0.19, ast=1.1, tov=0.95, ft_pct=0.67,
                       sample_size=200, computed_through_season=EARLY_SEASON),
        # Fine (position+height) rows — deliberately different from the
        # coarse "C" rows above, so tests can assert the fine value wins
        # for a rookie tall enough to land in this bucket. Left Corner 3
        # has NO fine row seeded, to exercise per-zone fallback to coarse.
        PositionPrior(position_bucket="C-tall", stat_type="zone", stat_key="Restricted Area",
                       fg_pct=0.70, sample_size=600, computed_through_season=EARLY_SEASON),
        PositionPrior(position_bucket="C-tall", stat_type="defense", stat_key="Overall",
                       d_fg_pct=0.44, pct_plusminus=-0.03, sample_size=20, computed_through_season=EARLY_SEASON),
    ]

    db_session.add_all(zone_rows + def_rows + shot_rows + position_prior_rows)
    db_session.commit()

    return db_engine


def test_resolve_player_stats_exact_season_is_measured(resolver_db):
    with resolver_db.connect() as conn:
        result = resolve_player_stats(conn, "P_EXACT", EARLY_SEASON)

    assert result["stats_source"] == "measured"
    assert result["resolved_season"] == EARLY_SEASON
    assert result["zone_stats"]["Restricted Area"] == pytest.approx(0.667)
    assert result["season_fg_pct"] == pytest.approx(0.56)


def test_resolve_player_stats_falls_back_to_prior_season(resolver_db):
    """Queried at LATE_SEASON (roster-only, no games played), P_STALE should
    resolve to their own real data from EARLY_SEASON — not a position prior."""
    with resolver_db.connect() as conn:
        result = resolve_player_stats(conn, "P_STALE", LATE_SEASON)

    assert result["stats_source"] == "measured"
    assert result["resolved_season"] == EARLY_SEASON
    assert result["zone_stats"]["Restricted Area"] == pytest.approx(0.571)
    assert result["position_bucket"] is None  # only set when falling back to a prior


def test_resolve_player_stats_zero_history_uses_position_prior(resolver_db):
    """P_ROOKIE has never recorded a real shot in any season -> tier 3."""
    with resolver_db.connect() as conn:
        result = resolve_player_stats(conn, "P_ROOKIE", FUTURE_SEASON)

    assert result["stats_source"] == "prior"
    assert result["resolved_season"] is None
    assert result["position_bucket"] == "C"
    assert result["zone_stats"]["Restricted Area"] == pytest.approx(0.64)
    assert result["season_fg_pct"] == pytest.approx(0.50)
    assert result["career_3p_pct"] == pytest.approx(0.19)


def test_resolve_player_stats_unknown_position_degrades_gracefully(resolver_db):
    """No position on record -> no bucket -> empty prior data, not a crash."""
    with resolver_db.connect() as conn:
        result = resolve_player_stats(conn, "P_UNKNOWN_POS", FUTURE_SEASON)

    assert result["stats_source"] == "prior"
    assert result["position_bucket"] is None
    assert result["zone_stats"] == {}
    assert result["season_fg_pct"] is None


def test_resolve_player_stats_prefers_fine_bucket_when_available(resolver_db):
    """A tall rookie center should get the C-tall fine-bucket value where
    it exists (Restricted Area), and fall back to the coarse C value for a
    zone with no fine row (Left Corner 3 wasn't seeded for C-tall)."""
    with resolver_db.connect() as conn:
        result = resolve_player_stats(conn, "P_TALL_ROOKIE", FUTURE_SEASON)

    assert result["stats_source"] == "prior"
    assert result["position_bucket"] == "C-tall"
    assert result["zone_stats"]["Restricted Area"] == pytest.approx(0.70)  # fine value, not coarse 0.64


def test_resolve_defender_stats_prefers_fine_bucket_when_available(resolver_db):
    with resolver_db.connect() as conn:
        result = resolve_defender_stats(conn, "P_TALL_ROOKIE", FUTURE_SEASON)

    assert result["stats_source"] == "prior"
    assert result["position_bucket"] == "C-tall"
    assert result["def_stats"]["Overall"]["d_fg_pct"] == pytest.approx(0.44)  # fine value, not coarse 0.47


def test_resolve_player_stats_unknown_player_returns_none(resolver_db):
    with resolver_db.connect() as conn:
        result = resolve_player_stats(conn, "NOBODY", EARLY_SEASON)
    assert result is None


def test_resolve_defender_stats_exact_season_is_measured(resolver_db):
    with resolver_db.connect() as conn:
        result = resolve_defender_stats(conn, "P_EXACT", EARLY_SEASON)

    assert result["stats_source"] == "measured"
    assert result["resolved_season"] == EARLY_SEASON
    assert result["def_stats"]["Overall"]["d_fg_pct"] == pytest.approx(0.46)


def test_resolve_defender_stats_falls_back_to_prior_season(resolver_db):
    with resolver_db.connect() as conn:
        result = resolve_defender_stats(conn, "P_STALE", LATE_SEASON)

    assert result["stats_source"] == "measured"
    assert result["resolved_season"] == EARLY_SEASON
    assert result["def_stats"]["Overall"]["d_fg_pct"] == pytest.approx(0.47)


def test_resolve_defender_stats_zero_history_uses_position_prior(resolver_db):
    with resolver_db.connect() as conn:
        result = resolve_defender_stats(conn, "P_ROOKIE", FUTURE_SEASON)

    assert result["stats_source"] == "prior"
    assert result["resolved_season"] is None
    assert result["position_bucket"] == "C"
    assert result["def_stats"]["Overall"]["d_fg_pct"] == pytest.approx(0.47)


def test_resolve_defender_stats_unknown_player_returns_none(resolver_db):
    with resolver_db.connect() as conn:
        result = resolve_defender_stats(conn, "NOBODY", EARLY_SEASON)
    assert result is None
