"""What the shooter's OWN team gives him — leave-one-out, point-in-time."""
from __future__ import annotations

import numpy as np
import pandas as pd

# Teammate attempts required this season before a supporting-cast rate is
# reported at all. Roughly two games' worth of team shooting — enough that
# the rate describes a cast rather than a night.
MIN_CAST_ATTEMPTS = 150


def build_supporting_cast(engine) -> pd.DataFrame:
    """
    What the shooter's OWN team gives him, computed leave-one-out and
    point-in-time.

    Why this exists
    ---------------
    The model knows a great deal about who is defending a shot and nothing at
    all about who is playing alongside it. `team_stats` carries exactly one
    column, `def_rating`, and it is consumed as `opp_def_rating` — the
    opponent's. The shooter's supporting cast has never been a feature.

    Why leave-one-out is not optional
    ---------------------------------
    A star's team stats are mostly the star. Denver's assisted-FG rate ranked
    6th of 30 in 2024-25 and Oklahoma City's 27th, which reads as "Jokic plays
    with better passers than Shai" — but Denver's number IS Jokic. Feeding the
    raw team rate into a Jokic shot re-encodes Jokic, and the model would
    happily learn a circular relationship.

    Every quantity here therefore EXCLUDES the shooter's own contribution:
    these are his teammates' numbers, not his team's. That is the quantity
    that answers "does this player get help", which is the question worth
    asking.

    Strictly prior, same as everything else in this package: a shot never
    contributes to the features describing it.

    Returns one row per (player_id, game_id) with:
        cast_ast_rate   teammates' assisted share of made field goals —
                        ball movement the shooter is not himself producing
        cast_3p_rate    teammates' three-point percentage — the spacing that
                        determines how much help defence he draws
        cast_efg        teammates' effective field-goal percentage — how much
                        attention the rest of the lineup commands
        cast_att        teammate attempts behind those rates, the model's
                        confidence cue
    """
    rows = pd.read_sql("""
        SELECT s.player_id,
               s.game_id,
               s.season,
               g.date AS game_date,
               CASE WHEN s.home_away = 1 THEN g.home_team ELSE g.away_team END
                   AS off_team,
               s.shot_made,
               CASE WHEN s.zone IN ('Left Corner 3', 'Right Corner 3',
                                    'Above the Break 3') THEN 1 ELSE 0 END AS is_three,
               CASE WHEN sc.shot_id IS NULL THEN 0 ELSE 1 END AS has_context,
               COALESCE(sc.is_assisted, 0) AS is_assisted
        FROM shots s
        JOIN games g ON g.game_id = s.game_id
        LEFT JOIN shot_context sc ON sc.shot_id = s.shot_id
        WHERE s.zone IS NOT NULL
          AND s.zone != 'Backcourt'
          AND s.home_away IS NOT NULL
    """, engine)

    if rows.empty:
        return pd.DataFrame(columns=["player_id", "game_id"])

    rows["game_date"] = pd.to_datetime(rows["game_date"])
    rows["made_three"] = rows["shot_made"] * rows["is_three"]
    # `is_assisted` is a per-shot label leak (assists are credited only on
    # makes) and is banned as a shot feature by tests/test_no_leaky_features.py.
    # Aggregated over a team's PRIOR games it leaks nothing about tonight's
    # shot, which is what makes the ball-movement signal reachable at all.
    #
    # The assisted rate is computed over makes WITH play-by-play context only.
    # Coverage starts in 2016-17, and counting an uncovered make as unassisted
    # would not read as missing data — it would read as a team that never
    # passes, which is a confident wrong answer rather than an absent one.
    rows["assisted_make"] = rows["shot_made"] * rows["is_assisted"]
    rows["make_with_context"] = rows["shot_made"] * rows["has_context"]
    # Effective FG% weights a made three at 1.5, the standard adjustment for
    # its extra point.
    rows["efg_num"] = rows["shot_made"] + 0.5 * rows["made_three"]

    stat_cols = ["shot_made", "made_three", "is_three", "assisted_make",
                 "make_with_context", "efg_num"]

    # Team totals per game, then the same totals attributed to each player, so
    # subtracting one from the other leaves the teammates' contribution alone.
    agg = {c: "sum" for c in stat_cols}
    team_game = rows.groupby(
        ["off_team", "season", "game_id", "game_date"], as_index=False
    ).agg({**agg, "player_id": "size"}).rename(columns={"player_id": "fga"})

    player_game = rows.groupby(
        ["player_id", "off_team", "season", "game_id", "game_date"], as_index=False
    ).agg({**agg, "shot_made": "sum"})
    player_game["fga"] = rows.groupby(
        ["player_id", "off_team", "season", "game_id", "game_date"]
    ).size().values

    # Cumulative totals reset EACH SEASON. A roster turns over completely; a
    # team's ball movement three years ago describes a different set of
    # players and is not evidence about the cast around this shooter tonight.
    team_game = team_game.sort_values(
        ["off_team", "season", "game_date", "game_id"]
    ).reset_index(drop=True)
    g_team = team_game.groupby(["off_team", "season"], sort=False)
    for col in stat_cols + ["fga"]:
        team_game[f"team_{col}"] = g_team[col].cumsum() - team_game[col]

    # Same for the player, within (team, season) so a midseason trade does not
    # carry his old team's contribution into the subtraction.
    player_game = player_game.sort_values(
        ["player_id", "off_team", "season", "game_date", "game_id"]
    ).reset_index(drop=True)
    g_player = player_game.groupby(["player_id", "off_team", "season"], sort=False)
    for col in stat_cols + ["fga"]:
        player_game[f"self_{col}"] = g_player[col].cumsum() - player_game[col]

    merged = player_game.merge(
        team_game[["off_team", "season", "game_id"]
                  + [f"team_{c}" for c in stat_cols + ["fga"]]],
        on=["off_team", "season", "game_id"], how="left",
    )

    # The leave-one-out step.
    for col in stat_cols + ["fga"]:
        merged[f"cast_{col}"] = merged[f"team_{col}"] - merged[f"self_{col}"]

    def _rate(num: str, den: str) -> pd.Series:
        """
        A rate, or NaN when the teammates have not yet accumulated enough
        evidence this season to support one.

        Below the gate the honest answer is "not known yet", which XGBoost
        handles natively as a missing value. Emitting 0.667 off three attempts
        would instead assert an elite shooting cast on no evidence — the same
        low-sample trap `shrink()` exists to avoid elsewhere, handled here by
        abstention because these rates have no fitted prior to regress toward.
        """
        denom = merged[den]
        out = merged[num] / denom.replace(0, np.nan)
        return out.where(denom >= MIN_CAST_ATTEMPTS)

    merged["cast_ast_rate"] = _rate("cast_assisted_make", "cast_make_with_context")
    merged["cast_3p_rate"] = _rate("cast_made_three", "cast_is_three")
    merged["cast_efg"] = _rate("cast_efg_num", "cast_fga")
    merged["cast_att"] = merged["cast_fga"]

    return merged[[
        "player_id", "game_id",
        "cast_ast_rate", "cast_3p_rate", "cast_efg", "cast_att",
    ]]


def lookup_supporting_cast(conn, player_id: str, season: str,
                           as_of_date=None) -> dict:
    """
    Serving-path equivalent of `build_supporting_cast` for one player.

    Returns the same `cast_*` keys the training builder produces. Without this
    the attainability model would receive NaN for every cast feature at
    serving time while seeing them populated for virtually every training row
    — the precise skew that made `recent_10_fg` look like a career debut on
    every served shot, and which tests/test_train_serve_parity.py exists to
    catch.

    The player's team is taken as the one he took the most shots for this
    season, so a midseason trade resolves to where he actually plays now.
    """
    from sqlalchemy import text

    date_clause = "AND g.date < :as_of" if as_of_date is not None else ""
    params = {"pid": str(player_id), "season": season}
    if as_of_date is not None:
        params["as_of"] = str(as_of_date)

    team_row = conn.execute(text(f"""
        SELECT CASE WHEN s.home_away = 1 THEN g.home_team ELSE g.away_team END AS team,
               COUNT(*) AS n
        FROM shots s
        JOIN games g ON g.game_id = s.game_id
        WHERE s.player_id = :pid AND s.season = :season
          AND s.home_away IS NOT NULL
          {date_clause}
        GROUP BY team ORDER BY n DESC LIMIT 1
    """), params).fetchone()

    empty = {c: None for c in
             ("cast_ast_rate", "cast_3p_rate", "cast_efg", "cast_att")}
    if team_row is None:
        return empty

    params["team"] = team_row[0]
    row = conn.execute(text(f"""
        SELECT SUM(s.shot_made)                                        AS makes,
               COUNT(*)                                                AS fga,
               SUM(CASE WHEN s.zone IN ('Left Corner 3','Right Corner 3',
                                        'Above the Break 3')
                        THEN 1 ELSE 0 END)                             AS threes,
               SUM(CASE WHEN s.zone IN ('Left Corner 3','Right Corner 3',
                                        'Above the Break 3')
                        THEN s.shot_made ELSE 0 END)                   AS made_threes,
               SUM(CASE WHEN sc.shot_id IS NOT NULL AND s.shot_made = 1
                        THEN 1 ELSE 0 END)                             AS makes_with_ctx,
               SUM(CASE WHEN sc.is_assisted = 1 AND s.shot_made = 1
                        THEN 1 ELSE 0 END)                             AS assisted
        FROM shots s
        JOIN games g ON g.game_id = s.game_id
        LEFT JOIN shot_context sc ON sc.shot_id = s.shot_id
        WHERE s.season = :season
          AND s.player_id != :pid
          AND s.home_away IS NOT NULL
          AND s.zone IS NOT NULL AND s.zone != 'Backcourt'
          AND CASE WHEN s.home_away = 1 THEN g.home_team ELSE g.away_team END = :team
          {date_clause}
    """), params).fetchone()

    if row is None or not row[1]:
        return empty

    makes, fga, threes, made_threes, makes_ctx, assisted = (
        float(v or 0) for v in row
    )
    if fga < MIN_CAST_ATTEMPTS:
        return empty

    return {
        "cast_ast_rate": (assisted / makes_ctx) if makes_ctx else None,
        "cast_3p_rate": (made_threes / threes) if threes else None,
        "cast_efg": ((makes + 0.5 * made_threes) / fga) if fga else None,
        "cast_att": fga,
    }
