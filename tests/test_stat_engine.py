"""
Tests for src/inference/stat_engine.py — the player/team stat browser.

The property that matters most here is JSON-safety: this module merges
several sources that legitimately have missing data for any given player
(no zone shooting yet, no 2K rating, too few minutes for our own ratings),
and a bare NaN reaching the wire is not a null to a browser's JSON.parse —
it's a syntax error. Every test that touches a player with a real gap
checks the output is actually JSON-round-trippable, not just "doesn't
raise" while an XGBoost stats package quietly hands back nan floats.
"""
import json
from datetime import date

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.db.database import register_unaccent
from src.db.models import (
    Base, DefenderStats, Player, PlayerTwoKRating, PlayerZoneStats, Shot,
    TeamStats,
)
from src.inference import stat_engine as se


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


def _seed_basics(engine):
    """A minimal but real player: identity, one zone stat, a defender
    row, a 2K rating, and a team — enough for every section of the
    profile to have at least one real value, and one deliberate gap
    (no player_tracking_stats row at all) to exercise the NaN path."""
    Session = sessionmaker(bind=engine)
    with Session() as session:
        session.add(Player(
            player_id="P1", season="2024-25", name="Test Player",
            position="G", team_id="T1", height=78.0, weight=210.0,
            wingspan=80.0, career_fg_pct=0.48, career_3p_pct=0.36,
            season_fg_pct=0.50, ast=5.0, tov=2.0, ft_pct=0.85,
        ))
        session.add(PlayerZoneStats(
            player_id="P1", season="2024-25", zone="Restricted Area",
            fgm=50, fga=80, fg_pct=0.625,
        ))
        session.add(DefenderStats(
            player_id="P1", season="2024-25", defense_category="Overall",
            gp=60, freq=1.0, d_fgm=100, d_fga=220, d_fg_pct=0.4545,
            normal_fg_pct=0.46, pct_plusminus=-0.0055,
        ))
        session.add(PlayerTwoKRating(
            player_id="P1", edition="NBA 2K27", overall=82,
            offense_avg=75.0, defense_avg=70.0,
        ))
        session.add(TeamStats(
            team_id="T1", season="2024-25", team_name="Test Team",
            team_abbrev="TST", def_rating=110.5,
        ))
        # A shot, so resolve_rating_season finds a real season to anchor to.
        session.add(Shot(
            shot_id="s1", game_id="g1", player_id="P1", season="2024-25",
            shot_made=1, zone="Restricted Area",
        ))
        session.commit()


def test_clean_converts_nan_and_numpy_scalars():
    assert se._clean(np.nan) is None
    assert se._clean(float("nan")) is None
    assert se._clean(None) is None
    assert se._clean(np.float64(1.5)) == 1.5
    assert isinstance(se._clean(np.float64(1.5)), float)
    assert se._clean(np.int64(7)) == 7
    assert isinstance(se._clean(np.int64(7)), int)
    assert se._clean("text") == "text"
    assert se._clean(3.2) == 3.2


def test_build_player_stat_table_merges_every_source(engine):
    _seed_basics(engine)
    table = se.build_player_stat_table(engine, "2024-25")

    assert len(table) == 1
    row = table.iloc[0]
    assert row["name"] == "Test Player"
    assert row["zone_fg_pct_rim"] == pytest.approx(0.625)
    assert row["def_fg_pct_allowed"] == pytest.approx(0.4545)
    assert row["two_k_overall"] == 82
    # No player_tracking_stats or player_defensive_activity row was seeded —
    # these columns must come back as NaN (pandas' missing marker), not
    # raise, and not silently default to 0 (which would read as "measured
    # zero", a confident wrong answer for "we have no data").
    assert pd.isna(row["self_creation_index"])
    assert pd.isna(row["blk_per_min"])
    assert row["zone_fga_rim"] == 80


def test_player_full_profile_sections_are_json_safe(engine):
    _seed_basics(engine)
    profile = se.player_full_profile(engine, "P1", "2024-25")

    assert profile is not None
    assert profile["name"] == "Test Player"
    group_names = [s["group"] for s in profile["sections"]]
    assert group_names == se.GROUP_ORDER

    # The whole point: this must round-trip through real JSON, the same
    # path the API layer uses, with no NaN surviving to break a client's
    # JSON.parse.
    encoded = json.dumps(profile)
    decoded = json.loads(encoded)
    assert decoded["player_id"] == "P1"

    shooting = next(s for s in profile["sections"] if s["group"] == "shooting")
    rim = next(s for s in shooting["stats"] if s["key"] == "zone_fg_pct_rim")
    assert rim["value"] == pytest.approx(0.625)

    creation = next(s for s in profile["sections"] if s["group"] == "creation")
    missing_stat = next(s for s in creation["stats"] if s["key"] == "self_creation_index")
    assert missing_stat["value"] is None  # NaN cleaned to None, not 0 or NaN


def test_player_full_profile_returns_none_for_unknown_player(engine):
    _seed_basics(engine)
    assert se.player_full_profile(engine, "NOT_A_REAL_PLAYER", "2024-25") is None


def test_build_team_stat_table_labels_roster_derived_columns(engine):
    _seed_basics(engine)
    teams = se.build_team_stat_table(engine, "2024-25")

    assert len(teams) == 1
    row = teams.iloc[0]
    assert row["def_rating"] == pytest.approx(110.5)  # the one real team stat
    # Roster-derived aggregates are explicitly prefixed, never presented as
    # if they were separately-measured team numbers we don't actually have.
    assert "roster_avg_season_fg_pct" in teams.columns
    assert row["roster_avg_season_fg_pct"] == pytest.approx(0.50)
    assert row["roster_size"] == 1


def test_team_full_profile_roster_entries_are_json_safe(engine):
    _seed_basics(engine)
    profile = se.team_full_profile(engine, "T1", "2024-25")

    assert profile is not None
    assert profile["team_name"] == "Test Team"
    assert len(profile["roster"]) == 1
    assert profile["roster"][0]["player_id"] == "P1"

    encoded = json.dumps(profile)
    json.loads(encoded)  # must not raise on a bare NaN anywhere in the roster


def test_team_full_profile_returns_none_for_unknown_team(engine):
    _seed_basics(engine)
    assert se.team_full_profile(engine, "NOT_A_REAL_TEAM", "2024-25") is None


def test_player_column_metadata_is_ordered_basic_to_advanced(engine):
    _seed_basics(engine)
    table = se.build_player_stat_table(engine, "2024-25")
    columns = se.player_column_metadata(table.columns)

    keys = [c["key"] for c in columns]
    # Only columns that actually exist in the table are described — an entry
    # for a column that isn't there renders as a permanently-empty column.
    assert set(keys) <= set(table.columns)
    assert "height" in keys and "two_k_overall" in keys

    groups = [c["group"] for c in columns]
    # Sections appear in GROUP_ORDER and never interleave, so a client can
    # render them in sequence without sorting.
    assert groups == sorted(groups, key=se.GROUP_ORDER.index)
    assert all(c["label"] and c["fmt"] for c in columns)


def test_team_column_metadata_marks_roster_derived_columns(engine):
    _seed_basics(engine)
    table = se.build_team_stat_table(engine, "2024-25")
    columns = se.team_column_metadata(table.columns)

    by_key = {c["key"]: c for c in columns}
    assert by_key["def_rating"]["derived"] is False  # genuinely measured
    assert by_key["roster_avg_season_fg_pct"]["derived"] is True  # a roster mean
    assert set(by_key) <= set(table.columns)


def test_position_is_carried_forward_from_the_last_season_that_had_one(engine):
    """roster_ingestor.py left position NULL for the seasons it owns. A
    listed position is per-player identity, so the player's own most recent
    one fills the gap — but only the MOST recent, and only for players who
    actually have one somewhere."""
    _seed_basics(engine)
    Session = sessionmaker(bind=engine)
    with Session() as session:
        # The current season's row has no position; two older ones do, and
        # the player changed position between them.
        session.add(Player(player_id="P2", season="2024-25", name="Gap Player", position=None))
        session.add(Player(player_id="P2", season="2022-23", name="Gap Player", position="G"))
        session.add(Player(player_id="P2", season="2023-24", name="Gap Player", position="F"))
        session.add(Player(player_id="P3", season="2024-25", name="Never Listed", position=None))
        session.commit()

    table = se.build_player_stat_table(engine, "2024-25").set_index("player_id")
    assert table.loc["P2", "position"] == "F"  # 2023-24, not the older 2022-23 "G"
    assert pd.isna(table.loc["P3", "position"])  # no position anywhere: stays missing
    assert table.loc["P1", "position"] == "G"  # already had one; untouched


def test_zone_percentage_with_no_attempts_is_missing_not_zero(engine):
    """The ingestor writes fg_pct = 0.0 for a zone a player never shot
    from. Reported as-is that says "0% from the corner", which is a claim
    about their shooting rather than an absence of one, and it sinks them
    to the bottom of that zone's leaderboard."""
    _seed_basics(engine)
    Session = sessionmaker(bind=engine)
    with Session() as session:
        session.add(PlayerZoneStats(
            player_id="P1", season="2024-25", zone="Left Corner 3",
            fgm=0, fga=0, fg_pct=0.0,
        ))
        # A genuine 0-for-5 IS a measurement and must survive.
        session.add(PlayerZoneStats(
            player_id="P1", season="2024-25", zone="Mid-Range",
            fgm=0, fga=5, fg_pct=0.0,
        ))
        session.commit()

    row = se.build_player_stat_table(engine, "2024-25").iloc[0]
    assert pd.isna(row["zone_fg_pct_left_corner3"])
    assert row["zone_fga_left_corner3"] == 0
    assert row["zone_fg_pct_midrange"] == 0.0
    assert row["zone_fga_midrange"] == 5
