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
    Base, DefenderStats, Player, PlayerDefensiveActivity, PlayerPlayType,
    PlayerShotProfile, PlayerTrackingStats, PlayerTwoKRating, PlayerZoneStats,
    REGULAR_SEASON, Shot, TeamStats,
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
            season_type=REGULAR_SEASON,
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


def test_defensive_categories_are_surfaced_and_playoffs_are_excluded(engine):
    """The five non-Overall categories ARE rim/paint/perimeter defence, and
    they were ingested all along. Playoff rows now sit beside the regular
    season ones in the same table, and must never be read as the season:
    before season_type was part of the key, the playoff pass overwrote the
    regular-season row and a 72-game season silently became a 15-game one.
    """
    _seed_basics(engine)
    Session = sessionmaker(bind=engine)
    with Session() as session:
        session.add(DefenderStats(
            player_id="P1", season="2024-25", defense_category="Less Than 6Ft",
            season_type=REGULAR_SEASON,
            gp=60, freq=0.3, d_fgm=120, d_fga=200, d_fg_pct=0.60,
            normal_fg_pct=0.66, pct_plusminus=-0.06,
        ))
        # A small, much worse postseason sample for the same category.
        session.add(DefenderStats(
            player_id="P1", season="2024-25", defense_category="Less Than 6Ft",
            season_type="Playoffs",
            gp=5, freq=0.3, d_fgm=18, d_fga=20, d_fg_pct=0.90,
            normal_fg_pct=0.66, pct_plusminus=0.24,
        ))
        session.add(DefenderStats(
            player_id="P1", season="2024-25", defense_category="3 Pointers",
            season_type=REGULAR_SEASON,
            gp=60, freq=0.25, d_fgm=60, d_fga=180, d_fg_pct=0.333,
            normal_fg_pct=0.36, pct_plusminus=-0.027,
        ))
        session.commit()

    row = se.build_player_stat_table(engine, "2024-25").iloc[0]
    assert row["def_rim_fg_pct"] == pytest.approx(0.60)   # regular season
    assert row["def_rim_fg_pct"] != pytest.approx(0.90)   # not the playoff row
    assert row["def_rim_fga"] == 200
    assert row["def_perimeter_fg_pct"] == pytest.approx(0.333)
    assert row["def_fg_pct_allowed"] == pytest.approx(0.4545)  # Overall, unchanged


def test_every_defensive_category_column_is_described_for_the_client(engine):
    _seed_basics(engine)
    table = se.build_player_stat_table(engine, "2024-25")
    described = {c["key"] for c in se.player_column_metadata(table.columns)}
    for slug in se.DEF_CATEGORY_SLUG.values():
        for suffix in ("fg_pct", "pm", "fga"):
            assert f"def_{slug}_{suffix}" in described


def test_rate_columns_carry_a_volume_qualifier(engine):
    """A rate stat must tell the client how many attempts sit behind it and
    how many are needed to be ranked. Without this, a "best rim defender"
    leaderboard is topped by whoever defended a single shot and got a stop."""
    _seed_basics(engine)
    Session = sessionmaker(bind=engine)
    with Session() as session:
        session.add(DefenderStats(
            player_id="P1", season="2024-25", defense_category="Less Than 6Ft",
            season_type=REGULAR_SEASON, gp=60, freq=0.3, d_fgm=120, d_fga=200,
            d_fg_pct=0.60, normal_fg_pct=0.66, pct_plusminus=-0.06,
        ))
        session.commit()

    columns = {c["key"]: c for c in se.player_column_metadata(
        se.build_player_stat_table(engine, "2024-25").columns)}

    rim = columns["def_rim_fg_pct"]
    assert rim["qualify_key"] == "def_rim_fga"
    assert rim["qualify_min"] > 0
    # The column it points at must actually exist, or the client silently
    # treats every player as unqualified and the leaderboard empties.
    assert rim["qualify_key"] in columns

    # A counting stat is its own evidence and needs no qualifier.
    assert "qualify_key" not in columns["def_rim_fga"]
    assert "qualify_key" not in columns["height"]


def test_every_qualifier_points_at_a_column_that_exists(engine):
    """A typo in QUALIFIERS would not raise — it would quietly rank nobody."""
    _seed_basics(engine)
    table = se.build_player_stat_table(engine, "2024-25")
    for stat_key, (volume_key, minimum) in se.QUALIFIERS.items():
        assert stat_key in se.STAT_GROUPS, f"{stat_key} is not a known stat"
        assert volume_key in se.STAT_GROUPS, f"{stat_key} points at unknown {volume_key}"
        assert minimum > 0


def test_passing_quality_columns_are_surfaced_from_tracking(engine):
    """potential_ast/secondary_ast/ast_points_created/ast_to_pass_pct sat in
    player_tracking_stats fully populated and unused — this is pure wiring,
    no new computation, so the values must come through unchanged."""
    _seed_basics(engine)
    Session = sessionmaker(bind=engine)
    with Session() as session:
        session.add(PlayerTrackingStats(
            player_id="P1", season="2024-25", gp=70, min_per_game=32.0,
            potential_ast=15.2, secondary_ast=1.1, ast_points_created=24.6,
            ast_to_pass_pct=0.14,
        ))
        session.commit()

    row = se.build_player_stat_table(engine, "2024-25").iloc[0]
    assert row["potential_ast"] == pytest.approx(15.2)
    assert row["secondary_ast"] == pytest.approx(1.1)
    assert row["ast_points_created"] == pytest.approx(24.6)
    assert row["ast_to_pass_pct"] == pytest.approx(0.14)


def test_shot_context_pivots_every_bucket_and_ignores_unrelated_split_types(engine):
    _seed_basics(engine)
    Session = sessionmaker(bind=engine)
    with Session() as session:
        session.add(PlayerShotProfile(
            player_id="P1", season="2024-25", split_type="def_dist",
            split_value="6+ Feet - Wide Open", gp=70, fga_frequency=0.4,
            fgm=90, fga=180, fg_pct=0.50,
        ))
        session.add(PlayerShotProfile(
            player_id="P1", season="2024-25", split_type="dribbles",
            split_value="7+ Dribbles", gp=70, fga_frequency=0.1,
            fgm=20, fga=50, fg_pct=0.40,
        ))
        # Never attempted — the ingestor leaves fg_pct NULL for this (verified
        # against the live DB), unlike player_zone_stats which needed a patch.
        session.add(PlayerShotProfile(
            player_id="P1", season="2024-25", split_type="touch_time",
            split_value="Touch 6+ Seconds", gp=70, fga_frequency=0.0,
            fgm=0, fga=0, fg_pct=None,
        ))
        session.commit()

    row = se.build_player_stat_table(engine, "2024-25").iloc[0]
    assert row["shotctx_wide_open_fg_pct"] == pytest.approx(0.50)
    assert row["shotctx_wide_open_fga"] == 180
    assert row["shotctx_seven_plus_dribbles_fg_pct"] == pytest.approx(0.40)
    assert pd.isna(row["shotctx_long_touch_fg_pct"])
    assert row["shotctx_long_touch_fga"] == 0
    # A bucket with no row at all for this player must still exist as a
    # column (reindexed onto every known slug), reading NaN rather than
    # vanishing from the table.
    assert "shotctx_tight_fg_pct" in row.index
    assert pd.isna(row["shotctx_tight_fg_pct"])


def test_shot_context_rate_columns_carry_qualifiers(engine):
    _seed_basics(engine)
    Session = sessionmaker(bind=engine)
    with Session() as session:
        # Consistent with the zone-shooting pivot: if NO player_shot_profile
        # row exists anywhere in the season, the whole column family is
        # absent rather than present-and-all-NaN — at least one real row is
        # needed to exercise the pivot at all.
        session.add(PlayerShotProfile(
            player_id="P1", season="2024-25", split_type="def_dist",
            split_value="6+ Feet - Wide Open", gp=70, fga_frequency=0.4,
            fgm=90, fga=180, fg_pct=0.50,
        ))
        session.commit()

    table = se.build_player_stat_table(engine, "2024-25")
    columns = {c["key"]: c for c in se.player_column_metadata(table.columns)}
    wide_open = columns["shotctx_wide_open_fg_pct"]
    assert wide_open["qualify_key"] == "shotctx_wide_open_fga"
    assert wide_open["qualify_min"] > 0
    assert "qualify_key" not in columns["shotctx_wide_open_fga"]


def test_every_shot_context_bucket_is_registered_and_grouped(engine):
    for slug in se.SHOT_CONTEXT_SLUG.values():
        assert f"shotctx_{slug}_fg_pct" in se.STAT_GROUPS
        assert f"shotctx_{slug}_fga" in se.STAT_GROUPS
        assert se.STAT_GROUPS[f"shotctx_{slug}_fg_pct"]["group"] == "shot_difficulty"
    assert "shot_difficulty" in se.GROUP_ORDER


def test_hustle_stats_are_surfaced_from_the_same_activity_table(engine):
    """Screen assists through contested shots come from the same table and
    endpoint as deflections — added later, so this checks the wiring, not
    new computation."""
    _seed_basics(engine)
    Session = sessionmaker(bind=engine)
    with Session() as session:
        session.add(PlayerDefensiveActivity(
            player_id="P1", season="2024-25", gp=70, min_per_game=30.0,
            stl=1.0, blk=1.0, deflections=2.0,
            screen_ast=3.5, screen_ast_pts=7.9, box_outs=2.1,
            off_boxouts=0.8, def_boxouts=1.3, loose_balls_recovered=0.6,
            off_loose_balls_recovered=0.3, def_loose_balls_recovered=0.3,
            charges_drawn=0.1, contested_shots=4.2,
            contested_shots_2pt=3.0, contested_shots_3pt=1.2,
        ))
        session.commit()

    row = se.build_player_stat_table(engine, "2024-25").iloc[0]
    assert row["screen_ast"] == pytest.approx(3.5)
    assert row["screen_ast_pts"] == pytest.approx(7.9)
    assert row["box_outs"] == pytest.approx(2.1)
    assert row["charges_drawn"] == pytest.approx(0.1)
    assert row["contested_shots_3pt"] == pytest.approx(1.2)

    columns = {c["key"] for c in se.player_column_metadata(se.build_player_stat_table(engine, "2024-25").columns)}
    assert "screen_ast" in columns
    col = next(c for c in se.player_column_metadata(se.build_player_stat_table(engine, "2024-25").columns) if c["key"]=="screen_ast")
    assert col["group"] == "hustle_offense"
    assert col["side"] == "offense"


def test_play_type_columns_are_pivoted_and_qualified(engine):
    """poss_pct/poss/ppp/fg_pct per play type, pivoted the same way as shot
    context. PPP for a play type with almost no possessions must not be
    ranked — a "best isolation scorer" leaderboard topped by one possession
    of garbage time is the same failure class as the thin-sample rim
    defenders this project already fixed once."""
    _seed_basics(engine)
    Session = sessionmaker(bind=engine)
    with Session() as session:
        session.add(PlayerPlayType(
            player_id="P1", season="2024-25", play_type="Isolation",
            gp=70, poss=180, poss_pct=0.12, pts=190, fgm=70, fga=150,
            fg_pct=0.467, efg_pct=0.48, ppp=1.056, percentile=0.82,
        ))
        session.commit()

    row = se.build_player_stat_table(engine, "2024-25").iloc[0]
    assert row["playtype_isolation_ppp"] == pytest.approx(1.056)
    assert row["playtype_isolation_poss"] == 180
    assert row["playtype_isolation_poss_pct"] == pytest.approx(0.12)
    assert row["playtype_isolation_fg_pct"] == pytest.approx(0.467)
    # A play type never run at all this season must exist as a column
    # (reindexed onto every known slug) reading NaN, not vanish.
    assert "playtype_postup_ppp" in row.index
    assert pd.isna(row["playtype_postup_ppp"])

    columns = {c["key"]: c for c in se.player_column_metadata(
        se.build_player_stat_table(engine, "2024-25").columns)}
    assert columns["playtype_isolation_ppp"]["qualify_key"] == "playtype_isolation_poss"
    assert columns["playtype_isolation_ppp"]["group"] == "play_type"
    assert "qualify_key" not in columns["playtype_isolation_poss"]


def test_every_play_type_is_registered_in_the_right_group(engine):
    for slug in se.PLAY_TYPE_SLUG.values():
        for suffix in ("poss_pct", "poss", "ppp", "fg_pct"):
            key = f"playtype_{slug}_{suffix}"
            assert key in se.STAT_GROUPS, key
            assert se.STAT_GROUPS[key]["group"] == "play_type"
    assert "play_type" in se.GROUP_ORDER


def test_every_group_has_a_side_and_hustle_is_split(engine):
    """Every group used anywhere in STAT_GROUPS must resolve to a side, or a
    client asking "which tab does this stat belong on" gets a KeyError deep
    in a render. Hustle specifically must be split rather than a single
    mixed group, since it genuinely contains both ends of the floor."""
    groups_in_use = {meta["group"] for meta in se.STAT_GROUPS.values()}
    assert groups_in_use <= set(se.GROUP_SIDE)
    assert se.GROUP_SIDE["hustle_offense"] == "offense"
    assert se.GROUP_SIDE["hustle_defense"] == "defense"
    assert se.GROUP_SIDE["overview"] == "info"
    assert se.GROUP_SIDE["ratings"] == "info"
    assert se.GROUP_SIDE["defense"] == "defense"
    assert se.GROUP_SIDE["shooting"] == "offense"


def test_team_columns_carry_a_side(engine):
    _seed_basics(engine)
    teams = se.build_team_stat_table(engine, "2024-25")
    cols = {c["key"]: c for c in se.team_column_metadata(teams.columns)}
    assert cols["def_rating"]["side"] == "defense"
    assert cols["roster_avg_off_rating"]["side"] == "offense"
    assert cols["roster_size"]["side"] == "info"


def test_team_id_uses_the_most_recent_roster_even_from_a_future_season(engine):
    """resolve_rating_season falls back to the last season with real games,
    so the stats shown can be a year stale by the time a trade happens. The
    team badge next to them must not be — it should reflect the most recent
    roster we have, even one from a season with no games played yet, not
    whichever season's box score stats happen to be on screen."""
    _seed_basics(engine)
    Session = sessionmaker(bind=engine)
    with Session() as session:
        # P1 was traded after 2024-25: a newer, roster-only season row
        # exists with a different team and no stats behind it at all.
        session.add(Player(
            player_id="P1", season="2025-26", name="Test Player",
            position="G", team_id="T2", height=78.0, weight=210.0,
        ))
        session.commit()

    row = se.build_player_stat_table(engine, "2024-25").iloc[0]
    assert row["team_id"] == "T2"  # not "T1", the team from the displayed season


def test_box_score_stats_are_surfaced_from_the_same_activity_table(engine):
    """Points, rebounds, and fouls sat in the same already-fetched Base
    response as blocks/steals the whole time — same wiring check as the
    hustle-stat extension, not new computation."""
    _seed_basics(engine)
    Session = sessionmaker(bind=engine)
    with Session() as session:
        session.add(PlayerDefensiveActivity(
            player_id="P1", season="2024-25", gp=70, min_per_game=30.0,
            pts=24.6, fgm=8.5, fga=17.2, fg3m=1.8, fg3a=4.9,
            ftm=5.8, fta=6.7, oreb=1.9, dreb=6.1, reb=8.0,
            pf=2.4, pfd=5.1, dd2=42, td3=3,
        ))
        session.commit()

    row = se.build_player_stat_table(engine, "2024-25").iloc[0]
    assert row["pts"] == pytest.approx(24.6)
    assert row["reb"] == pytest.approx(8.0)
    assert row["oreb"] == pytest.approx(1.9)
    assert row["dreb"] == pytest.approx(6.1)
    assert row["dd2"] == 42

    columns = se.player_column_metadata(se.build_player_stat_table(engine, "2024-25").columns)
    by_key = {c["key"]: c for c in columns}
    assert by_key["pts"]["side"] == "offense"
    assert by_key["dreb"]["side"] == "defense"
    assert by_key["oreb"]["side"] == "offense"
    assert by_key["dd2"]["side"] == "info"
