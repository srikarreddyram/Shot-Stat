"""One "long" (season, key, per_game, total, games, num, den) builder per
data source this project already owns — defensive activity, defender
categories, zone shooting, shot difficulty context, passing quality, and
Synergy play type. `career_panel` in __init__.py concatenates all six and
aggregates them together; the column-name/label maps here are also read
by catalogue.py to build the flat list of every reportable stat.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy import text

from src.db.models import REGULAR_SEASON

# ── What the defensive categories are actually called ───────────────────────
# LeagueDashPtDefend's own category names are shooting-distance buckets. These
# are the basketball words for them, which is what the user asked to see.
DEF_CATEGORIES: dict[str, tuple[str, str]] = {
    "Overall": ("overall", "Overall"),
    "Less Than 6Ft": ("rim", "Rim (< 6 ft)"),
    "Less Than 10Ft": ("paint", "Paint (< 10 ft)"),
    "2 Pointers": ("two_pt", "Two-point"),
    "Greater Than 15Ft": ("mid_long", "Long two (15+ ft)"),
    "3 Pointers": ("perimeter", "Perimeter (3PT)"),
}

# Order the defensive section reads in: overall first, then inside-out.
DEF_CATEGORY_ORDER = ["overall", "rim", "paint", "two_pt", "mid_long", "perimeter"]

# Column on player_defensive_activity -> (career-panel key, reader label,
# side). Screen assists through contested shots are the "hustle stats"
# extension — same table, same per-game shape as blocks/steals/deflections,
# added later. `side` matches stat_engine.STAT_GROUPS' hustle_offense /
# hustle_defense split exactly, so a player's career page can be divided the
# same way its season page is.
HUSTLE_STAT_COLUMNS: dict[str, tuple[str, str, str]] = {
    "screen_ast": ("screen_ast", "Screen assists", "offense"),
    "screen_ast_pts": ("screen_ast_pts", "Points off screen assists", "offense"),
    "off_boxouts": ("off_boxouts", "Offensive box-outs", "offense"),
    "off_loose_balls_recovered": ("off_loose_balls_recovered", "Offensive loose balls recovered", "offense"),
    "box_outs": ("box_outs", "Box-outs", "defense"),
    "def_boxouts": ("def_boxouts", "Defensive box-outs", "defense"),
    "loose_balls_recovered": ("loose_balls_recovered", "Loose balls recovered", "defense"),
    "def_loose_balls_recovered": ("def_loose_balls_recovered", "Defensive loose balls recovered", "defense"),
    "charges_drawn": ("charges_drawn", "Charges drawn", "defense"),
    "contested_shots": ("contested_shots", "Shots contested", "defense"),
    "contested_shots_2pt": ("contested_shots_2pt", "2PT shots contested", "defense"),
    "contested_shots_3pt": ("contested_shots_3pt", "3PT shots contested", "defense"),
}

# Ordinary box-score counting stats, same table, same per-game shape as
# blocks/steals — added even later than the hustle-stat extension (see
# defensive_activity_ingestor.py's own note on why points/rebounds sat in an
# already-fetched response unused). oreb/pfd read as offensive contributions
# (an offensive rebound extends your own possession; drawing a foul is an
# offensive skill); dreb/reb/pf read as the defensive side, matching this
# module's existing box_outs/loose_balls_recovered convention for totals.
BOX_SCORE_COLUMNS: dict[str, tuple[str, str, str]] = {
    "pts": ("pts", "Points", "offense"),
    "fgm": ("fgm", "Field goals made", "offense"),
    "fga": ("fga", "Field goals attempted", "offense"),
    "fg3m": ("fg3m", "Three-pointers made", "offense"),
    "fg3a": ("fg3a", "Three-pointers attempted", "offense"),
    "ftm": ("ftm", "Free throws made", "offense"),
    "fta": ("fta", "Free throws attempted", "offense"),
    "oreb": ("oreb", "Offensive rebounds", "offense"),
    "pfd": ("pfd", "Fouls drawn", "offense"),
    "dreb": ("dreb", "Defensive rebounds", "defense"),
    "reb": ("reb", "Total rebounds", "defense"),
    "pf": ("pf", "Personal fouls", "defense"),
}
# dd2/td3 are already SEASON TOTALS on the source row (not a per-game rate to
# multiply up) — "double-doubles per game" is not a real quantity, so these
# need the COUNT treatment `_defensive_activity_long` gives games_played,
# not the PER_GAME treatment everything else in this table gets.
SEASON_TOTAL_COLUMNS: dict[str, tuple[str, str, str]] = {
    "dd2": ("dd2", "Double-doubles", "offense"),
    "td3": ("td3", "Triple-doubles", "offense"),
}


def _defensive_activity_long(engine, player_id: str) -> pd.DataFrame:
    """Blocks, steals, deflections, hustle stats, box score and minutes —
    all stored per game (dd2/td3 as season totals) in
    player_defensive_activity."""
    per_game_source_cols = {**HUSTLE_STAT_COLUMNS, **BOX_SCORE_COLUMNS}
    select_cols = ", ".join({**per_game_source_cols, **SEASON_TOTAL_COLUMNS})
    df = pd.read_sql(
        text(f"""
            SELECT season, gp, min_per_game, stl, blk, deflections, {select_cols}
            FROM player_defensive_activity
            WHERE player_id = :pid
        """),
        engine, params={"pid": str(player_id)},
    )
    if df.empty:
        return pd.DataFrame()

    rows = []
    for per_game_col, key in [
        ("blk", "blocks"), ("stl", "steals"),
        ("deflections", "deflections"), ("min_per_game", "minutes"),
        *[(col, col) for col in per_game_source_cols],
    ]:
        part = df[["season", "gp", per_game_col]].dropna(subset=[per_game_col]).copy()
        part["key"] = key
        part["per_game"] = part[per_game_col]
        part["total"] = part[per_game_col] * part["gp"]
        part["games"] = part["gp"]
        rows.append(part[["season", "key", "per_game", "total", "games"]])

    # dd2/td3: already a season TOTAL on the row — total is the value as-is,
    # not value*gp (which would double-count games played into a count that
    # has nothing to do with a rate).
    for col in SEASON_TOTAL_COLUMNS:
        part = df[["season", "gp", col]].dropna(subset=[col]).copy()
        part["key"] = col
        part["total"] = part[col]
        part["per_game"] = part[col] / part["gp"].replace(0, np.nan)
        part["games"] = part["gp"]
        rows.append(part[["season", "key", "per_game", "total", "games"]])

    games = df[["season", "gp"]].dropna().copy()
    games["key"] = "games_played"
    games["per_game"] = np.nan
    games["total"] = games["gp"]
    games["games"] = games["gp"]
    rows.append(games[["season", "key", "per_game", "total", "games"]])

    rows = [r for r in rows if not r.empty]
    out = pd.concat(rows, ignore_index=True)
    out["num"] = np.nan
    out["den"] = np.nan
    return out


def _defender_category_long(engine, player_id: str) -> pd.DataFrame:
    """FG% allowed, versus-normal, and defended volume, per category."""
    df = pd.read_sql(
        text("""
            SELECT season, defense_category, gp, d_fgm, d_fga, d_fg_pct,
                   normal_fg_pct, pct_plusminus
            FROM defender_stats
            WHERE player_id = :pid AND season_type = :stype AND d_fga IS NOT NULL
        """),
        engine, params={"pid": str(player_id), "stype": REGULAR_SEASON},
    )
    if df.empty:
        return pd.DataFrame()

    rows = []
    for category, (slug, _label) in DEF_CATEGORIES.items():
        part = df[df["defense_category"] == category]
        if part.empty:
            continue

        # FG% allowed: a rate, carried as makes/attempts so the career figure
        # can be recomputed rather than averaged.
        rate = part[["season"]].copy()
        rate["key"] = f"def_{slug}_fg_pct"
        rate["num"] = part["d_fgm"].values
        rate["den"] = part["d_fga"].values
        rate["per_game"] = part["d_fg_pct"].values
        rate["total"] = np.nan
        rate["games"] = part["gp"].values
        rows.append(rate)

        # Versus league normal: a difference of two rates, weighted by the
        # attempts behind it.
        diff = part[["season"]].copy()
        diff["key"] = f"def_{slug}_pm"
        diff["num"] = (part["pct_plusminus"] * part["d_fga"]).values
        diff["den"] = part["d_fga"].values
        diff["per_game"] = part["pct_plusminus"].values
        diff["total"] = np.nan
        diff["games"] = part["gp"].values
        rows.append(diff)

        # Volume defended: a genuine counting stat, and the context that makes
        # the two above readable.
        for col, suffix in [("d_fga", "fga"), ("d_fgm", "fgm")]:
            vol = part[["season"]].copy()
            vol["key"] = f"def_{slug}_{suffix}"
            vol["num"] = np.nan
            vol["den"] = np.nan
            vol["total"] = part[col].values
            vol["games"] = part["gp"].values
            vol["per_game"] = (part[col] / part["gp"].replace(0, np.nan)).values
            rows.append(vol)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)[
        ["season", "key", "per_game", "total", "games", "num", "den"]
    ]


def _shooting_long(engine, player_id: str) -> pd.DataFrame:
    """Zone shooting — the offensive side, same treatment."""
    from src.inference.stat_engine import ZONE_SLUG

    df = pd.read_sql(
        text("""
            SELECT season, zone, fgm, fga FROM player_zone_stats
            WHERE player_id = :pid AND fga IS NOT NULL
        """),
        engine, params={"pid": str(player_id)},
    )
    if df.empty:
        return pd.DataFrame()

    df = df[df["zone"].isin(ZONE_SLUG)]
    if df.empty:
        return pd.DataFrame()
    df["slug"] = df["zone"].map(ZONE_SLUG)

    rows = []
    rate = df[["season"]].copy()
    rate["key"] = "zone_fg_pct_" + df["slug"].values
    rate["num"] = df["fgm"].values
    rate["den"] = df["fga"].values
    # A zero-attempt zone has no percentage; see stat_engine's note on why a
    # stored 0.0 there is "never shot from", not "0%".
    rate["per_game"] = np.where(df["fga"].values > 0, df["fgm"].values / df["fga"].replace(0, np.nan).values, np.nan)
    rate["total"] = np.nan
    rate["games"] = np.nan
    rows.append(rate)

    vol = df[["season"]].copy()
    vol["key"] = "zone_fga_" + df["slug"].values
    vol["num"] = np.nan
    vol["den"] = np.nan
    vol["per_game"] = np.nan
    vol["total"] = df["fga"].values
    vol["games"] = np.nan
    rows.append(vol)

    return pd.concat(rows, ignore_index=True)[
        ["season", "key", "per_game", "total", "games", "num", "den"]
    ]


def _shot_context_long(engine, player_id: str) -> pd.DataFrame:
    """Shot difficulty & diet — same FG%/volume treatment as zone shooting
    (`_shooting_long`), just split by contest distance, dribbles and touch
    time instead of court location. See stat_engine.SHOT_CONTEXT_SLUG for
    the (split_type, split_value) -> slug mapping, reused here rather than
    duplicated."""
    from src.inference.stat_engine import SHOT_CONTEXT_SLUG

    df = pd.read_sql(
        text("""
            SELECT season, split_type, split_value, fgm, fga
            FROM player_shot_profile
            WHERE player_id = :pid AND fga IS NOT NULL
        """),
        engine, params={"pid": str(player_id)},
    )
    if df.empty:
        return pd.DataFrame()

    df["slug"] = list(zip(df["split_type"], df["split_value"]))
    df["slug"] = df["slug"].map(SHOT_CONTEXT_SLUG)
    df = df.dropna(subset=["slug"])
    if df.empty:
        return pd.DataFrame()

    rows = []
    rate = df[["season"]].copy()
    rate["key"] = "shotctx_" + df["slug"].values + "_fg_pct"
    rate["num"] = df["fgm"].values
    rate["den"] = df["fga"].values
    # A shot type never attempted in a season has fga == 0 and fg_pct NULL
    # already (unlike player_zone_stats, verified live — see stat_engine's
    # note), so no zero-attempt patch is needed here either.
    rate["per_game"] = np.where(df["fga"].values > 0, df["fgm"].values / df["fga"].replace(0, np.nan).values, np.nan)
    rate["total"] = np.nan
    rate["games"] = np.nan
    rows.append(rate)

    vol = df[["season"]].copy()
    vol["key"] = "shotctx_" + df["slug"].values + "_fga"
    vol["num"] = np.nan
    vol["den"] = np.nan
    vol["per_game"] = np.nan
    vol["total"] = df["fga"].values
    vol["games"] = np.nan
    rows.append(vol)

    return pd.concat(rows, ignore_index=True)[
        ["season", "key", "per_game", "total", "games", "num", "den"]
    ]


def _passing_long(engine, player_id: str) -> pd.DataFrame:
    """Passing quality, from SportVU tracking. Potential/secondary assists
    and points created are stored per game, same treatment as blocks/steals.
    Assist-to-pass rate is a true rate — recomputed from total assists over
    total passes, not averaged as a percentage — using the same `ast` and
    `passes_made` columns the season-level table already reads."""
    df = pd.read_sql(
        text("""
            SELECT season, gp, ast, potential_ast, secondary_ast,
                   ast_points_created, passes_made
            FROM player_tracking_stats
            WHERE player_id = :pid
        """),
        engine, params={"pid": str(player_id)},
    )
    if df.empty:
        return pd.DataFrame()

    rows = []
    for per_game_col, key in [
        ("potential_ast", "potential_ast"),
        ("secondary_ast", "secondary_ast"),
        ("ast_points_created", "ast_points_created"),
    ]:
        part = df[["season", "gp", per_game_col]].dropna(subset=[per_game_col]).copy()
        part["key"] = key
        part["per_game"] = part[per_game_col]
        part["total"] = part[per_game_col] * part["gp"]
        part["games"] = part["gp"]
        part["num"] = np.nan
        part["den"] = np.nan
        rows.append(part[["season", "key", "per_game", "total", "games", "num", "den"]])

    both = df.dropna(subset=["ast", "passes_made", "gp"])
    if not both.empty:
        rate = both[["season"]].copy()
        rate["key"] = "ast_to_pass_pct"
        rate["num"] = (both["ast"] * both["gp"]).values
        rate["den"] = (both["passes_made"] * both["gp"]).values
        rate["per_game"] = (both["ast"] / both["passes_made"].replace(0, np.nan)).values
        rate["total"] = np.nan
        rate["games"] = np.nan
        rows.append(rate[["season", "key", "per_game", "total", "games", "num", "den"]])

    rows = [r for r in rows if not r.empty]
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _play_type_long(engine, player_id: str) -> pd.DataFrame:
    """Synergy play-type tendency and efficiency — poss/pts/fga are true
    season TOTALS (the ingestor pulls per_mode_simple='Totals'), so PPP and
    FG% can be recomputed from summed points/possessions and makes/attempts
    the same way every other rate in this module is, rather than averaged."""
    from src.inference.stat_engine import PLAY_TYPE_SLUG

    df = pd.read_sql(
        text("""
            SELECT season, play_type, poss, pts, fgm, fga
            FROM player_play_type
            WHERE player_id = :pid AND poss IS NOT NULL
        """),
        engine, params={"pid": str(player_id)},
    )
    if df.empty:
        return pd.DataFrame()

    df["slug"] = df["play_type"].map(PLAY_TYPE_SLUG)
    df = df.dropna(subset=["slug"])
    if df.empty:
        return pd.DataFrame()

    rows = []
    # PPP: points per possession, recomputed from summed totals.
    ppp = df[["season"]].copy()
    ppp["key"] = "playtype_" + df["slug"].values + "_ppp"
    ppp["num"] = df["pts"].values
    ppp["den"] = df["poss"].values
    ppp["per_game"] = np.where(df["poss"].values > 0, df["pts"].values / df["poss"].replace(0, np.nan).values, np.nan)
    ppp["total"] = np.nan
    ppp["games"] = np.nan
    rows.append(ppp)

    # FG%, same treatment.
    fg = df.dropna(subset=["fga"])
    if not fg.empty:
        rate = fg[["season"]].copy()
        rate["key"] = "playtype_" + fg["slug"].values + "_fg_pct"
        rate["num"] = fg["fgm"].values
        rate["den"] = fg["fga"].values
        rate["per_game"] = np.where(fg["fga"].values > 0, fg["fgm"].values / fg["fga"].replace(0, np.nan).values, np.nan)
        rate["total"] = np.nan
        rate["games"] = np.nan
        rows.append(rate)

    # Possessions run: the volume that makes PPP/FG% readable.
    vol = df[["season"]].copy()
    vol["key"] = "playtype_" + df["slug"].values + "_poss"
    vol["num"] = np.nan
    vol["den"] = np.nan
    vol["per_game"] = np.nan
    vol["total"] = df["poss"].values
    vol["games"] = np.nan
    rows.append(vol)

    return pd.concat(rows, ignore_index=True)[
        ["season", "key", "per_game", "total", "games", "num", "den"]
    ]
