"""
Stat Engine — every stat this project tracks, browsable per player and per
team, basic through advanced. Distinct from the earlier `/stats/features`
page (still live at a separate route): that page answers "what does the
MODEL use and how much"; this one answers "what does this PLAYER (or team)
actually do", independent of whether any of it ends up in a model at all.

Design
------
One wide table per player (`build_player_stat_table`) merging every source
this project has already built for other reasons — box-score/physical
identity (`players`), zone-level shooting (`player_zone_stats`), handle/
passing/rim-pressure tracking (`creation.load_creation_profiles`), defensive
activity (`defensive_activity.load_defensive_activity_profiles`), FG%-allowed
by category (`defender_stats`), our own computed ratings
(`player_ratings.rating_lookup`), and NBA 2K's (`player_two_k_ratings`).
None of these are recomputed here — this module is a read-only join over
data other modules already own, on purpose: two sources of truth for the
same number is exactly the class of bug this project has hit and fixed
several times already this session (def_freq_zone, the bare-zone bugs).

Unlike the creation/defensive-activity profiles' USE elsewhere in this
project (always lagged one season for leak-free training features — see
creation.py's module docstring), this reads them for the season they
describe with NO lag: a stats browser is reporting what happened, not
predicting a future shot, so there is nothing to leak.

STAT_GROUPS documents every column's section and label once, so the API
and a client can agree on how to organize the page without either one
hand-maintaining a duplicate list.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.creation import load_creation_profiles
from src.features.defensive_activity import load_defensive_activity_profiles
from src.inference.player_ratings import rating_lookup, resolve_rating_season

ZONE_SLUG = {
    "Restricted Area": "rim",
    "In The Paint (Non-RA)": "paint",
    "Mid-Range": "midrange",
    "Left Corner 3": "left_corner3",
    "Right Corner 3": "right_corner3",
    "Above the Break 3": "above_break3",
}

# One entry per stat column this module can emit, keyed by the column name
# in build_player_stat_table's output. "group" orders the profile page
# basic-to-advanced; "label" is the reader-facing name; "fmt" matches
# explain.py's _format_value vocabulary, plus "count" for whole-number
# tallies (attempts, roster size) so a table column does not mix "14" and
# "14.0" down its length.
STAT_GROUPS: dict[str, dict] = {
    # ── Overview ────────────────────────────────────────────────────────
    "height": {"group": "overview", "label": "Height", "fmt": "in"},
    "weight": {"group": "overview", "label": "Weight", "fmt": "count"},
    "wingspan": {"group": "overview", "label": "Wingspan", "fmt": "in"},
    "position": {"group": "overview", "label": "Position", "fmt": "text"},
    "off_rating": {"group": "overview", "label": "Offensive rating", "fmt": "num"},
    "def_rating_ours": {"group": "overview", "label": "Defensive rating", "fmt": "num"},
    # ── Shooting (basic) ────────────────────────────────────────────────
    "career_fg_pct": {"group": "shooting", "label": "Career FG%", "fmt": "pct"},
    "season_fg_pct": {"group": "shooting", "label": "Season FG%", "fmt": "pct"},
    "career_3p_pct": {"group": "shooting", "label": "Career 3P%", "fmt": "pct"},
    "ft_pct": {"group": "shooting", "label": "FT%", "fmt": "pct"},
    # Each zone's percentage is paired with its attempt count on purpose: a
    # bare 60% is unreadable without knowing whether it is 3-for-5 or
    # 300-for-500, and a zone leaderboard sorted on percentage alone is
    # topped entirely by players with a handful of attempts.
    "zone_fg_pct_rim": {"group": "shooting", "label": "Restricted area FG%", "fmt": "pct"},
    "zone_fga_rim": {"group": "shooting", "label": "Restricted area attempts", "fmt": "count"},
    "zone_fg_pct_paint": {"group": "shooting", "label": "Paint (non-RA) FG%", "fmt": "pct"},
    "zone_fga_paint": {"group": "shooting", "label": "Paint (non-RA) attempts", "fmt": "count"},
    "zone_fg_pct_midrange": {"group": "shooting", "label": "Mid-range FG%", "fmt": "pct"},
    "zone_fga_midrange": {"group": "shooting", "label": "Mid-range attempts", "fmt": "count"},
    "zone_fg_pct_left_corner3": {"group": "shooting", "label": "Left corner 3 FG%", "fmt": "pct"},
    "zone_fga_left_corner3": {"group": "shooting", "label": "Left corner 3 attempts", "fmt": "count"},
    "zone_fg_pct_right_corner3": {"group": "shooting", "label": "Right corner 3 FG%", "fmt": "pct"},
    "zone_fga_right_corner3": {"group": "shooting", "label": "Right corner 3 attempts", "fmt": "count"},
    "zone_fg_pct_above_break3": {"group": "shooting", "label": "Above the break 3 FG%", "fmt": "pct"},
    "zone_fga_above_break3": {"group": "shooting", "label": "Above the break 3 attempts", "fmt": "count"},
    # ── Creation & playmaking (advanced) ───────────────────────────────
    "self_creation_index": {"group": "creation", "label": "Self-creation index", "fmt": "num"},
    "playmaking_gravity": {"group": "creation", "label": "Playmaking gravity", "fmt": "num"},
    "rim_pressure": {"group": "creation", "label": "Rim pressure", "fmt": "num"},
    "avg_drib_per_touch": {"group": "creation", "label": "Dribbles per touch", "fmt": "num"},
    "drives_per_min": {"group": "creation", "label": "Drives per minute", "fmt": "num"},
    "pullup_share": {"group": "creation", "label": "Pull-up share", "fmt": "pct"},
    "catch_shoot_share": {"group": "creation", "label": "Catch-and-shoot share", "fmt": "pct"},
    "open_share": {"group": "creation", "label": "Open-shot share", "fmt": "pct"},
    "tight_share": {"group": "creation", "label": "Tightly-contested share", "fmt": "pct"},
    "drive_pf_pct": {"group": "creation", "label": "Foul-drawn rate on drives", "fmt": "pct"},
    "ast": {"group": "creation", "label": "Assists per game", "fmt": "num"},
    "tov": {"group": "creation", "label": "Turnovers per game", "fmt": "num"},
    # ── Defense (advanced) ─────────────────────────────────────────────
    "def_fg_pct_allowed": {"group": "defense", "label": "FG% allowed (overall)", "fmt": "pct"},
    "def_plus_minus": {"group": "defense", "label": "FG% allowed vs. league normal", "fmt": "pct"},
    "blk_per_min": {"group": "defense", "label": "Blocks per minute", "fmt": "num"},
    "stl_per_min": {"group": "defense", "label": "Steals per minute", "fmt": "num"},
    "deflections_per_min": {"group": "defense", "label": "Deflections per minute", "fmt": "num"},
    "defensive_gravity": {"group": "defense", "label": "Defensive gravity", "fmt": "num"},
    # ── Ratings & the outside opinion ───────────────────────────────────
    "rating_source": {"group": "ratings", "label": "Rating source", "fmt": "text"},
    "two_k_overall": {"group": "ratings", "label": "NBA 2K overall", "fmt": "num"},
    "two_k_offense_avg": {"group": "ratings", "label": "NBA 2K offense (our rollup)", "fmt": "num"},
    "two_k_defense_avg": {"group": "ratings", "label": "NBA 2K defense (our rollup)", "fmt": "num"},
}

GROUP_ORDER = ["overview", "shooting", "creation", "defense", "ratings"]
GROUP_LABELS = {
    "overview": "Overview",
    "shooting": "Shooting",
    "creation": "Creation & Playmaking",
    "defense": "Defense",
    "ratings": "Ratings",
}

# Team columns are few enough and flat enough not to need sections. The
# roster_avg_ prefix is load-bearing, not cosmetic — see
# build_team_stat_table's docstring on why these must read as derived.
TEAM_STAT_LABELS: dict[str, dict] = {
    "def_rating": {"label": "Defensive rating", "fmt": "num", "derived": False},
    "roster_size": {"label": "Roster size", "fmt": "count", "derived": False},
    "roster_avg_season_fg_pct": {"label": "Avg FG%", "fmt": "pct", "derived": True},
    "roster_avg_off_rating": {"label": "Avg offensive rating", "fmt": "num", "derived": True},
    "roster_avg_def_rating_ours": {"label": "Avg defensive rating", "fmt": "num", "derived": True},
    "roster_avg_self_creation_index": {"label": "Avg self-creation", "fmt": "num", "derived": True},
    "roster_avg_playmaking_gravity": {"label": "Avg playmaking gravity", "fmt": "num", "derived": True},
    "roster_avg_rim_pressure": {"label": "Avg rim pressure", "fmt": "num", "derived": True},
    "roster_avg_def_fg_pct_allowed": {"label": "Avg FG% allowed", "fmt": "pct", "derived": True},
    "roster_avg_blk_per_min": {"label": "Avg blocks/min", "fmt": "num", "derived": True},
    "roster_avg_stl_per_min": {"label": "Avg steals/min", "fmt": "num", "derived": True},
    "roster_avg_deflections_per_min": {"label": "Avg deflections/min", "fmt": "num", "derived": True},
}


def player_column_metadata(columns) -> list[dict]:
    """Describe the stat columns actually present in a built table, in
    basic-to-advanced order. The leaderboard's column list is derived from
    STAT_GROUPS here rather than retyped client-side — a duplicated column
    list is how a renamed stat quietly becomes an empty column."""
    present = set(columns)
    out = []
    for group in GROUP_ORDER:
        for col, meta in STAT_GROUPS.items():
            if meta["group"] == group and col in present:
                out.append({"key": col, "label": meta["label"], "fmt": meta["fmt"],
                            "group": group, "group_label": GROUP_LABELS[group]})
    return out


def team_column_metadata(columns) -> list[dict]:
    """Same idea as player_column_metadata, flat (teams have ~10 columns)."""
    present = set(columns)
    return [
        {"key": col, "label": meta["label"], "fmt": meta["fmt"], "derived": meta["derived"]}
        for col, meta in TEAM_STAT_LABELS.items() if col in present
    ]


def build_player_stat_table(engine, season: str | None = None) -> pd.DataFrame:
    """
    One row per player rostered in `season` (default: the most recent season
    with real box-score data — see resolve_rating_season), every stat this
    project tracks. Basic identity/shooting on the left, advanced tracking-
    derived and defensive-activity stats on the right, our ratings and NBA
    2K's last.
    """
    season = resolve_rating_season(engine, season or "2026-27")

    players = pd.read_sql(
        "SELECT player_id, name, position, team_id, height, weight, wingspan, "
        "career_fg_pct, career_3p_pct, season_fg_pct, ast, tov, ft_pct "
        "FROM players WHERE season = ?",
        engine, params=(season,),
    )
    if players.empty:
        return players

    # ── Position, carried forward when this season's row lacks one ─────
    # roster_ingestor.py wrote NULL position for every player in the seasons
    # it owns (its source endpoint has no position column — see its own
    # comment), and position_backfill.py has not been run for them. A listed
    # position is per-player identity, not a per-season measurement, so the
    # player's own most recent known position is the same fact, not a guess.
    missing_position = players["position"].isna()
    if missing_position.any():
        # SQLite's documented bare-column rule: alongside MAX(), the
        # non-aggregated columns come from the row that held the maximum,
        # so `position` here is the most recent one on record.
        known = pd.read_sql(
            "SELECT player_id, position, MAX(season) FROM players "
            "WHERE position IS NOT NULL AND season <= ? "
            "GROUP BY player_id",
            engine, params=(season,),
        ).set_index("player_id")["position"]
        players.loc[missing_position, "position"] = (
            players.loc[missing_position, "player_id"].map(known)
        )

    # ── Zone shooting, pivoted wide ────────────────────────────────────
    zones = pd.read_sql(
        "SELECT player_id, zone, fga, fg_pct FROM player_zone_stats WHERE season = ?",
        engine, params=(season,),
    )
    if not zones.empty:
        zones["slug"] = zones["zone"].map(ZONE_SLUG)
        # A zone the player never shot from is stored as fg_pct = 0.0, not
        # NULL, by the ingestor. Reported as-is that reads "0% from the
        # corner" — a confident wrong answer for "never took one" — and it
        # drags the player to the bottom of that zone's leaderboard. Only a
        # real attempt earns a percentage.
        zones.loc[zones["fga"] == 0, "fg_pct"] = np.nan
        # Reindexed onto every zone rather than only the ones present, so the
        # column set is fixed by ZONE_SLUG and not by which zones happen to
        # have data — a pivot alone silently drops an all-missing zone, and
        # the column would then vanish from the page instead of reading "—".
        slugs = list(ZONE_SLUG.values())
        for measure in ("fg_pct", "fga"):
            wide = zones.pivot_table(index="player_id", columns="slug", values=measure, aggfunc="first")
            wide = wide.reindex(columns=slugs)
            wide.columns = [f"zone_{measure}_{c}" for c in wide.columns]
            players = players.merge(wide, on="player_id", how="left")

    # ── Creation/tracking — THIS season's own numbers, not lagged (a stats
    # browser reports what happened; lagging is a training-leakage concern
    # that doesn't apply here) ─────────────────────────────────────────
    creation = load_creation_profiles(engine)
    creation = creation[creation["season"] == season].drop(columns=["season"])
    players = players.merge(creation, on="player_id", how="left")

    # ── Defensive activity, same season, no lag ────────────────────────
    defense_activity = load_defensive_activity_profiles(engine)
    defense_activity = defense_activity[defense_activity["season"] == season].drop(columns=["season"])
    players = players.merge(defense_activity, on="player_id", how="left")

    # ── FG%-allowed overall, from the same category rates every other
    # defender feature in this project already uses ──────────────────
    defense = pd.read_sql(
        "SELECT player_id, d_fg_pct AS def_fg_pct_allowed, pct_plusminus AS def_plus_minus "
        "FROM defender_stats WHERE season = ? AND defense_category = 'Overall'",
        engine, params=(season,),
    )
    players = players.merge(defense, on="player_id", how="left")

    # ── Our own computed ratings ────────────────────────────────────────
    ratings = rating_lookup(engine, season)
    ratings_df = pd.DataFrame([
        {"player_id": pid, "off_rating": r.get("off_rating"),
         "def_rating_ours": r.get("def_rating"), "rating_source": r.get("rating_source")}
        for pid, r in ratings.items()
    ])
    if not ratings_df.empty:
        players = players.merge(ratings_df, on="player_id", how="left")
    else:
        players["off_rating"] = players["def_rating_ours"] = players["rating_source"] = None

    # ── NBA 2K, for the side-by-side comparison ────────────────────────
    two_k = pd.read_sql(
        "SELECT player_id, overall AS two_k_overall, offense_avg AS two_k_offense_avg, "
        "defense_avg AS two_k_defense_avg FROM player_two_k_ratings",
        engine,
    )
    players = players.merge(two_k, on="player_id", how="left")

    players["season"] = season
    return players


def _clean(value):
    """
    NaN -> None (bare NaN is not valid JSON — some clients tolerate it,
    browsers' own JSON.parse does not, so a NaN reaching the wire is a
    silent frontend crash waiting for whichever row hits it first) and
    numpy scalar -> native Python (FastAPI's default encoder chokes on
    numpy int64/float64 the same way it would on a Decimal).
    """
    if value is None:
        return None
    if isinstance(value, (np.floating, float)) and np.isnan(value):
        return None
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def player_full_profile(engine, player_id: str, season: str | None = None) -> dict | None:
    """One player's full stat table row, reshaped into STAT_GROUPS sections
    for a profile page. Returns None if the player has no row for the
    resolved season at all (never rostered, or a bad id)."""
    table = build_player_stat_table(engine, season)
    if table.empty:
        return None
    row = table[table["player_id"] == str(player_id)]
    if row.empty:
        return None
    row = row.iloc[0]

    sections = []
    for group in GROUP_ORDER:
        stats = []
        for col, meta in STAT_GROUPS.items():
            if meta["group"] != group or col not in row.index:
                continue
            stats.append({"key": col, "label": meta["label"], "fmt": meta["fmt"],
                          "value": _clean(row[col])})
        sections.append({"group": group, "label": GROUP_LABELS[group], "stats": stats})

    return {
        "player_id": str(player_id),
        "name": row.get("name"),
        "team_id": row.get("team_id"),
        "season": row.get("season"),
        "sections": sections,
    }


def build_team_stat_table(engine, season: str | None = None) -> pd.DataFrame:
    """
    One row per team: the one real team-level stat this project tracks
    (def_rating), plus honest ROSTER-DERIVED aggregates — the mean of each
    roster's own player-level stats, not a separately-collected team number,
    since team_stats.def_rating is genuinely the only team-specific column
    this project ingests (see TeamStats' own docstring). Fabricating more
    "team stats" that are secretly just player averages under a different
    name would be the same mistake the old frontend rating formula made —
    presenting an invented number as if it were measured.
    """
    season = resolve_rating_season(engine, season or "2026-27")

    teams = pd.read_sql(
        "SELECT team_id, team_name, team_abbrev, def_rating FROM team_stats WHERE season = ?",
        engine, params=(season,),
    )

    players = build_player_stat_table(engine, season)
    if players.empty or teams.empty:
        return teams

    roster_avg_cols = [
        "season_fg_pct", "off_rating", "def_rating_ours",
        "self_creation_index", "playmaking_gravity", "rim_pressure",
        "def_fg_pct_allowed", "blk_per_min", "stl_per_min", "deflections_per_min",
    ]
    present = [c for c in roster_avg_cols if c in players.columns]
    agg = players.groupby("team_id")[present].mean().add_prefix("roster_avg_")
    agg["roster_size"] = players.groupby("team_id").size()

    teams = teams.merge(agg, left_on="team_id", right_index=True, how="left")
    teams["season"] = season
    return teams


def team_full_profile(engine, team_id: str, season: str | None = None) -> dict | None:
    """A team's real stat plus its roster, each player carrying their own
    full stat row — a team profile IS its roster here, not a separate
    invented team-level number set. See build_team_stat_table's docstring."""
    season = resolve_rating_season(engine, season or "2026-27")

    team_row = pd.read_sql(
        "SELECT team_id, team_name, team_abbrev, def_rating FROM team_stats "
        "WHERE season = ? AND team_id = ?",
        engine, params=(season, str(team_id)),
    )

    players = build_player_stat_table(engine, season)
    roster = players[players["team_id"] == str(team_id)] if not players.empty else players

    if team_row.empty and roster.empty:
        return None

    return {
        "team_id": str(team_id),
        "team_name": team_row.iloc[0]["team_name"] if not team_row.empty else None,
        "team_abbrev": team_row.iloc[0]["team_abbrev"] if not team_row.empty else None,
        "def_rating": _clean(team_row.iloc[0]["def_rating"]) if not team_row.empty else None,
        "season": season,
        "roster": [
            {"player_id": r["player_id"], "name": r["name"], "position": _clean(r.get("position")),
             "off_rating": _clean(r.get("off_rating")), "def_rating_ours": _clean(r.get("def_rating_ours"))}
            for _, r in roster.sort_values("name").iterrows()
        ] if not roster.empty else [],
    }
