"""
Point-in-time clutch performance — is this player a clutch outlier?

Why this exists
----------------
The model already knows whether THIS SHOT is a clutch situation (`clutch_flag`
in spec/__init__.py: 4th quarter or overtime, under 5 minutes left, score
within 5 — the NBA's own official "Clutch Time" definition). What it never
knew is whether THIS PLAYER is any good in that situation specifically, as
opposed to his normal game. Some players are — some visibly are not — and a
flag that only says "this moment is clutch" cannot express that difference on
its own; it needs a per-player fact to interact with.

`clutch_fg_delta` is that fact: a player's career-to-date clutch FG% (shrunk
toward a league-wide clutch prior — clutch attempts are a small fraction of a
season, so raw ratios are mostly noise) minus his career-to-date OVERALL FG%
(the same `overall_rate` shrinkage.apply_hierarchy already computes). Positive
means he shoots better in the clutch than his normal baseline; negative means
worse. `clutch_edge` multiplies that delta by `clutch_flag`, so the effect is
zero on every non-clutch shot and only engages when the shot itself is one —
XGBoost can in principle learn this interaction from the two raw columns
alone, but every other genuine interaction in this codebase (matchup_
advantage, creation_edge, openness_vs_defender — see spec/groups.py's
"interaction" group) is handed to the trees pre-multiplied rather than left
for them to rediscover, and there is no reason for this one to be the
exception.

Why not ingest the NBA's own LeagueDashPlayerClutch endpoint instead
----------------------------------------------------------------------
That endpoint is real and official, but it is a SEASON aggregate per player —
it cannot say what a specific shot's own game state was, only a player's
whole-season clutch split. Everything this feature needs (which shots were
clutch, who took them, whether they went in) is already sitting in the `shots`
table via `quarter`/`time_remaining`/`score_diff` — the same three columns
`clutch_flag` is built from — so building this point-in-time, shot-by-shot,
costs zero new ingestion and zero new API calls.

Single pooled prior, not per-zone
----------------------------------
Clutch shots are roughly 5% of all shots league-wide. Splitting a league prior
by zone on top of that (the way shooting_rates.py's zone hierarchy does for
the much larger overall-shooting population) would starve most zones of
usable evidence. The question this exists to answer is coarse by nature — "is
this player, overall, a clutch outperformer or underperformer" — not
zone-specific, so one pooled Beta prior over all clutch attempts is the right
grain.
"""
from __future__ import annotations

import pandas as pd

from src.features.shrinkage import BetaPrior, fit_beta_prior

# The NBA's own official "Clutch Time" definition (see LeagueDashPlayerClutch's
# "Last 5 Minutes" / "Ahead or Behind" / point_diff=5 filter): 4th quarter or
# overtime, 5 minutes or less remaining, score within 5 points. `clutch_flag`
# in spec/__init__.py uses these same two constants so a shot's own
# "is this clutch" flag and a player's "how does he do in the clutch" history
# describe the identical window rather than two different notions of clutch.
CLUTCH_TIME_REMAINING = 300.0
CLUTCH_MARGIN = 5


def fit_league_clutch_prior(engine, through_season: str) -> BetaPrior:
    """
    One league-wide Beta prior over clutch-shot FG%, fit from every clutch
    shot through `through_season` (inclusive) — never the test season; see
    fit_league_zone_priors's identical warning.
    """
    df = pd.read_sql(f"""
        SELECT s.player_id, SUM(s.shot_made) AS makes, COUNT(*) AS attempts
        FROM shots s
        WHERE s.quarter >= 4 AND s.time_remaining <= {CLUTCH_TIME_REMAINING}
          AND ABS(s.score_diff) <= {CLUTCH_MARGIN}
          AND s.season <= '{through_season}'
        GROUP BY s.player_id
    """, engine)
    if df.empty or df["attempts"].sum() == 0:
        return BetaPrior(mean=0.45, strength=50.0)
    return fit_beta_prior(df["makes"].to_numpy(), df["attempts"].to_numpy())


def build_clutch_performance(engine) -> pd.DataFrame:
    """
    One row per (player_id, game_id): career-to-date clutch-shot makes and
    attempts, strictly prior to that game (same shift-after-cumsum
    construction every other point-in-time builder in this package uses).

    Deliberately does NOT also rebuild "overall" makes/attempts — build.py
    already merges shooting_rates.build_prior_counts's `pit_car_mk_all`/
    `pit_car_att_all` onto the same frame, and spec/__init__.py combines
    THOSE with this table's clutch counts into `clutch_fg_delta`. Recomputing
    a second copy of the overall counts here would risk the two drifting —
    the two-sources-of-truth bug this project has hit before.
    """
    rows = pd.read_sql(f"""
        SELECT s.player_id, s.game_id, g.date AS game_date,
               SUM(CASE WHEN s.quarter >= 4 AND s.time_remaining <= {CLUTCH_TIME_REMAINING}
                             AND ABS(s.score_diff) <= {CLUTCH_MARGIN}
                        THEN s.shot_made ELSE 0 END) AS clutch_makes,
               SUM(CASE WHEN s.quarter >= 4 AND s.time_remaining <= {CLUTCH_TIME_REMAINING}
                             AND ABS(s.score_diff) <= {CLUTCH_MARGIN}
                        THEN 1 ELSE 0 END) AS clutch_attempts
        FROM shots s
        JOIN games g ON g.game_id = s.game_id
        GROUP BY s.player_id, s.game_id, g.date
    """, engine)
    if rows.empty:
        return pd.DataFrame(columns=["player_id", "game_id", "pit_car_clutch_mk", "pit_car_clutch_att"])

    rows["game_date"] = pd.to_datetime(rows["game_date"])
    rows = rows.sort_values(["player_id", "game_date", "game_id"]).reset_index(drop=True)

    g = rows.groupby("player_id", sort=False)
    rows["pit_car_clutch_mk"] = g["clutch_makes"].cumsum() - rows["clutch_makes"]
    rows["pit_car_clutch_att"] = g["clutch_attempts"].cumsum() - rows["clutch_attempts"]

    return rows[["player_id", "game_id", "pit_car_clutch_mk", "pit_car_clutch_att"]]


def lookup_clutch_performance(conn, player_id: str, as_of_date=None) -> dict:
    """
    Serving-path equivalent of `build_clutch_performance` for one player.

    Same `as_of_date` semantics as `lookup_prior_counts`: omitted, it means
    "everything in the database so far" (live serving); passed, it reproduces
    what the model would have seen on that date (the backtest harness).
    """
    from sqlalchemy import text

    date_clause = "AND g.date < :as_of" if as_of_date is not None else ""
    params = {"pid": str(player_id)}
    if as_of_date is not None:
        params["as_of"] = str(as_of_date)

    row = conn.execute(text(f"""
        SELECT SUM(CASE WHEN s.quarter >= 4 AND s.time_remaining <= {CLUTCH_TIME_REMAINING}
                             AND ABS(s.score_diff) <= {CLUTCH_MARGIN}
                        THEN s.shot_made ELSE 0 END) AS clutch_makes,
               SUM(CASE WHEN s.quarter >= 4 AND s.time_remaining <= {CLUTCH_TIME_REMAINING}
                             AND ABS(s.score_diff) <= {CLUTCH_MARGIN}
                        THEN 1 ELSE 0 END) AS clutch_attempts
        FROM shots s
        JOIN games g ON g.game_id = s.game_id
        WHERE s.player_id = :pid
          {date_clause}
    """), params).fetchone()

    if row is None or row[1] is None:
        return {"pit_car_clutch_mk": 0.0, "pit_car_clutch_att": 0.0}
    return {"pit_car_clutch_mk": float(row[0] or 0.0), "pit_car_clutch_att": float(row[1] or 0.0)}
