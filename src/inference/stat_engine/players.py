"""Builds the one-row-per-player stat table and reshapes it into a single
player's profile. See this package's __init__.py for the overall design
note (why this is a read-only join over data other modules already own).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.creation import load_creation_profiles
from src.features.defensive_activity import load_defensive_activity_profiles
from src.inference.player_ratings import rating_lookup, resolve_rating_season
from .formatting import _clean
from .metadata import (
    DEF_CATEGORY_SLUG,
    GROUP_LABELS,
    GROUP_ORDER,
    GROUP_SIDE,
    PLAY_TYPE_SLUG,
    SHOT_CONTEXT_SLUG,
    STAT_GROUPS,
    ZONE_SLUG,
)


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

    # ── Team, overridden to the MOST RECENT roster we have — even one from
    # AFTER the season whose box score stats are being shown ─────────────
    # resolve_rating_season deliberately falls back to the last season with
    # real games (a brand-new season has rosters but no shots yet), so the
    # stats on this page can be a year stale by the time a trade happens.
    # The team shown next to them should not be: 115 players currently carry
    # a different team_id in 2026-27's roster-only data than in 2025-26's
    # played season, and showing the older one reads as "this app doesn't
    # know Team X traded for this player" the moment anyone who follows the
    # league looks at it. Unlike `position` above (bounded to <= season,
    # because a HISTORICAL season's position should stay that season's),
    # team has no such bound — "what team are they on" means today, no
    # matter which season's numbers are on screen.
    current_team = pd.read_sql(
        "SELECT player_id, team_id, MAX(season) FROM players "
        "WHERE team_id IS NOT NULL GROUP BY player_id",
        engine,
    ).set_index("player_id")["team_id"]
    players["team_id"] = players["player_id"].map(current_team).fillna(players["team_id"])

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

    # ── Passing quality, straight from SportVU tracking ─────────────────
    passing = pd.read_sql(
        "SELECT player_id, potential_ast, secondary_ast, ast_points_created, "
        "ast_to_pass_pct FROM player_tracking_stats WHERE season = ?",
        engine, params=(season,),
    )
    players = players.merge(passing, on="player_id", how="left")

    # ── Shot difficulty & diet, pivoted the same way as zone shooting ───
    # A shot type a player never attempted comes back as fg_pct = NULL from
    # the ingestor already (unlike player_zone_stats, which needed the
    # fga==0 patch) — confirmed against a live pull, so no equivalent patch
    # is needed here.
    shot_ctx = pd.read_sql(
        "SELECT player_id, split_type, split_value, fg_pct, fga FROM player_shot_profile "
        "WHERE season = ? AND split_type IN ('def_dist', 'dribbles', 'general', 'touch_time')",
        engine, params=(season,),
    )
    if not shot_ctx.empty:
        shot_ctx["slug"] = list(zip(shot_ctx["split_type"], shot_ctx["split_value"]))
        shot_ctx["slug"] = shot_ctx["slug"].map(SHOT_CONTEXT_SLUG)
        shot_ctx = shot_ctx.dropna(subset=["slug"])
        all_slugs = list(dict.fromkeys(SHOT_CONTEXT_SLUG.values()))
        for measure in ("fg_pct", "fga"):
            wide = shot_ctx.pivot_table(index="player_id", columns="slug", values=measure, aggfunc="first")
            wide = wide.reindex(columns=all_slugs)
            wide.columns = [f"shotctx_{c}_{measure}" for c in wide.columns]
            players = players.merge(wide, on="player_id", how="left")

    # ── Play type (Synergy), pivoted the same way ───────────────────────
    # poss_pct/poss/ppp/fg_pct are already the ONLY columns we read out of
    # player_play_type — see PLAY_TYPE_SLUG's docstring on why efg_pct and
    # NBA's own percentile are deliberately left out for now.
    play_type = pd.read_sql(
        "SELECT player_id, play_type, poss_pct, poss, ppp, fg_pct "
        "FROM player_play_type WHERE season = ?",
        engine, params=(season,),
    )
    if not play_type.empty:
        play_type["slug"] = play_type["play_type"].map(PLAY_TYPE_SLUG)
        play_type = play_type.dropna(subset=["slug"])
        all_playtype_slugs = list(PLAY_TYPE_SLUG.values())
        for measure in ("poss_pct", "poss", "ppp", "fg_pct"):
            wide = play_type.pivot_table(index="player_id", columns="slug", values=measure, aggfunc="first")
            wide = wide.reindex(columns=all_playtype_slugs)
            wide.columns = [f"playtype_{c}_{measure}" for c in wide.columns]
            players = players.merge(wide, on="player_id", how="left")

    # ── Defensive activity, same season, no lag ────────────────────────
    defense_activity = load_defensive_activity_profiles(engine)
    defense_activity = defense_activity[defense_activity["season"] == season].drop(columns=["season"])
    players = players.merge(defense_activity, on="player_id", how="left")

    # ── Raw defensive counting stats, per game ──────────────────────────
    # load_defensive_activity_profiles gives per-MINUTE rates, which are the
    # right shape for a model feature and the wrong one for a reader: "2.6
    # blocks a game" is the number people know. Both are kept.
    activity_counts = pd.read_sql(
        "SELECT player_id, gp AS games_played, min_per_game, blk AS blk_per_game, "
        "stl AS stl_per_game, deflections AS deflections_per_game, "
        "screen_ast, screen_ast_pts, off_boxouts, def_boxouts, box_outs, "
        "off_loose_balls_recovered, def_loose_balls_recovered, loose_balls_recovered, "
        "charges_drawn, contested_shots, contested_shots_2pt, contested_shots_3pt, "
        "pts, fgm, fga, fg3m, fg3a, ftm, fta, oreb, dreb, reb, pf, pfd, dd2, td3 "
        "FROM player_defensive_activity WHERE season = ?",
        engine, params=(season,),
    )
    players = players.merge(activity_counts, on="player_id", how="left")

    # ── FG%-allowed, per defensive category ─────────────────────────────
    # Overall keeps its original column names, which other consumers already
    # read (api.py's player card, the frontend's mapBackendPlayer). The other
    # five categories are what "rim / paint / perimeter defence" actually
    # means in this data — they were ingested all along and simply never
    # surfaced. Regular season only: the playoff rows live alongside them now
    # (see DefenderStats' docstring) and must not be mixed in.
    defense_rows = pd.read_sql(
        "SELECT player_id, defense_category, d_fg_pct, pct_plusminus, d_fga "
        "FROM defender_stats WHERE season = ? AND season_type = 'Regular Season'",
        engine, params=(season,),
    )
    if not defense_rows.empty:
        overall = defense_rows[defense_rows["defense_category"] == "Overall"][
            ["player_id", "d_fg_pct", "pct_plusminus", "d_fga"]
        ].rename(columns={
            "d_fg_pct": "def_fg_pct_allowed",
            "pct_plusminus": "def_plus_minus",
            "d_fga": "def_fga_overall",
        })
        players = players.merge(overall, on="player_id", how="left")

        for category, slug in DEF_CATEGORY_SLUG.items():
            part = defense_rows[defense_rows["defense_category"] == category][
                ["player_id", "d_fg_pct", "pct_plusminus", "d_fga"]
            ].rename(columns={
                "d_fg_pct": f"def_{slug}_fg_pct",
                "pct_plusminus": f"def_{slug}_pm",
                "d_fga": f"def_{slug}_fga",
            })
            players = players.merge(part, on="player_id", how="left")

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
        sections.append({"group": group, "label": GROUP_LABELS[group],
                         "side": GROUP_SIDE[group], "stats": stats})

    return {
        "player_id": str(player_id),
        "name": row.get("name"),
        "team_id": row.get("team_id"),
        "season": row.get("season"),
        "sections": sections,
    }
