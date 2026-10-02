"""
Season-by-season averages for the major box-score stats, for one player or
one team, regular season and playoffs separately — the table at the top of a
basketball-reference page. Built from the per-game box scores in
player_game_logs / team_game_logs (src/ingestion/game_log_ingestor.py).

Two rules that keep these matching the NBA's own numbers:

  * Per-game stats are season totals divided by games played, not an average
    of anything else.
  * Shooting percentages are total makes over total attempts. Averaging each
    game's FG% would weight a 1-for-2 night the same as a 12-for-20 one.

Teams additionally get their opponents' line (from the other team's row in
the same game) and a league rank per stat per season, so "112.4 points" can
be read as "7th in the league".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy import text

SEASON_TYPES = {"regular": "Regular Season", "playoffs": "Playoffs"}
COUNTING = ["pts", "reb", "oreb", "dreb", "ast", "stl", "blk", "tov", "pf", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta"]

# (key, label, fmt) in display order. fmt matches the frontend's formatStat.
PLAYER_COLUMNS = [
    ("gp", "GP", "count"), ("min", "MIN", "num"), ("pts", "PTS", "num"), ("reb", "REB", "num"),
    ("ast", "AST", "num"), ("stl", "STL", "num"), ("blk", "BLK", "num"), ("tov", "TOV", "num"),
    ("fg_pct", "FG%", "pct"), ("fg3_pct", "3P%", "pct"), ("ft_pct", "FT%", "pct"),
    ("fg3m", "3PM", "num"), ("oreb", "OREB", "num"), ("dreb", "DREB", "num"), ("pf", "PF", "num"),
    ("plus_minus", "+/−", "num"),
]
TEAM_COLUMNS = [
    ("gp", "GP", "count"), ("w", "W", "count"), ("l", "L", "count"), ("win_pct", "WIN%", "pct"),
    ("pts", "PTS", "num"), ("opp_pts", "OPP PTS", "num"), ("margin", "MARGIN", "num"),
    ("reb", "REB", "num"), ("ast", "AST", "num"), ("stl", "STL", "num"), ("blk", "BLK", "num"),
    ("tov", "TOV", "num"), ("fg_pct", "FG%", "pct"), ("fg3_pct", "3P%", "pct"), ("ft_pct", "FT%", "pct"),
    ("fg3m", "3PM", "num"), ("opp_fg_pct", "OPP FG%", "pct"), ("opp_fg3_pct", "OPP 3P%", "pct"),
]
# For team ranks: stats where fewer is better (1st = the fewest).
LOWER_IS_BETTER = {"opp_pts", "tov", "opp_fg_pct", "opp_fg3_pct", "l", "pf"}
# Neither better nor worse — no rank.
UNRANKED = {"gp"}


def _div(a, b):
    return np.where(b > 0, a / np.where(b > 0, b, 1), np.nan)


def _summarise(g: pd.DataFrame, extra: dict | None = None) -> pd.DataFrame:
    """Per-season averages from grouped per-game rows. `g` is a groupby
    result carrying summed counting stats plus `games`."""
    out = pd.DataFrame(index=g.index)
    out["gp"] = g["games"]
    for c in COUNTING:
        if c in g:
            out[c] = g[c] / g["games"]
    out["fg_pct"] = _div(g["fgm"], g["fga"])
    out["fg3_pct"] = _div(g["fg3m"], g["fg3a"])
    out["ft_pct"] = _div(g["ftm"], g["fta"])
    for k, v in (extra or {}).items():
        out[k] = v
    return out


def _clean(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if np.isnan(v) else round(float(v), 4)
    return round(v, 4) if isinstance(v, float) else v


def _rows(df: pd.DataFrame, keys: list[str], extra_cols: tuple = ()) -> list[dict]:
    return [{"season": idx, **{c: _clean(r.get(c)) for c in (*extra_cols, *keys)}}
            for idx, r in df.sort_index(ascending=False).iterrows()]


# ── Players ────────────────────────────────────────────────────────────────
def player_seasons(engine, player_id: str) -> dict | None:
    logs = pd.read_sql(text("SELECT * FROM player_game_logs WHERE player_id = :pid"), engine,
                       params={"pid": str(player_id)})
    if logs.empty:
        return None
    logs = logs.sort_values("game_date")
    keys = [k for k, _, _ in PLAYER_COLUMNS]
    out = {"player_id": str(player_id), "name": str(logs["player_name"].iloc[-1]),
           "columns": [{"key": k, "label": lab, "fmt": f} for k, lab, f in PLAYER_COLUMNS]}
    for key, stype in SEASON_TYPES.items():
        part = logs[logs["season_type"] == stype]
        if part.empty:
            out[key] = {"seasons": [], "career": None}
            continue
        agg = {c: "sum" for c in COUNTING}
        agg.update({"minutes": "sum", "plus_minus": "sum"})
        g = part.groupby("season").agg(agg)
        g["games"] = part.groupby("season").size()
        # A player traded mid-season appears under every team he played for,
        # in the order he played for them.
        teams = part.groupby("season")["team_abbrev"].agg(lambda s: " · ".join(dict.fromkeys(s)))
        season_df = _summarise(g, {"min": g["minutes"] / g["games"], "plus_minus": g["plus_minus"] / g["games"],
                                   "team": teams})
        tot = g.sum(numeric_only=True).to_frame().T
        career = _summarise(tot, {"min": tot["minutes"] / tot["games"],
                                  "plus_minus": tot["plus_minus"] / tot["games"]}).iloc[0]
        out[key] = {
            "seasons": _rows(season_df, keys, ("team",)),
            "career": {"season": "Career", "seasons": int(len(g)), **{k: _clean(career.get(k)) for k in keys}},
        }
    out["first_season"] = str(logs["season"].min())
    out["last_season"] = str(logs["season"].max())
    return out


# ── Teams ──────────────────────────────────────────────────────────────────
def _team_frame(engine, season_type: str) -> pd.DataFrame:
    """Every team's per-season averages for one season type, with its
    opponents' line — league-wide, so ranks can be computed."""
    logs = pd.read_sql(text("SELECT * FROM team_game_logs WHERE season_type = :st"), engine,
                       params={"st": season_type})
    if logs.empty:
        return pd.DataFrame()
    opp = logs[["game_id", "team_id", "pts", "fgm", "fga", "fg3m", "fg3a"]].rename(
        columns={"team_id": "opp_team_id", "pts": "o_pts", "fgm": "o_fgm", "fga": "o_fga",
                 "fg3m": "o_fg3m", "fg3a": "o_fg3a"})
    m = logs.merge(opp, on="game_id")
    m = m[m["team_id"] != m["opp_team_id"]]
    agg = {c: "sum" for c in COUNTING + ["o_pts", "o_fgm", "o_fga", "o_fg3m", "o_fg3a", "plus_minus", "win"]}
    g = m.groupby(["team_id", "season"]).agg(agg)
    g["games"] = m.groupby(["team_id", "season"]).size()
    abbrev = m.groupby(["team_id", "season"])["team_abbrev"].last()
    return _summarise(g, {
        "team": abbrev, "w": g["win"], "l": g["games"] - g["win"], "win_pct": g["win"] / g["games"],
        "opp_pts": g["o_pts"] / g["games"], "margin": g["plus_minus"] / g["games"],
        "opp_fg_pct": _div(g["o_fgm"], g["o_fga"]), "opp_fg3_pct": _div(g["o_fg3m"], g["o_fg3a"]),
    })


def _ranks(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    ranks = pd.DataFrame(index=frame.index)
    for k in keys:
        if k in UNRANKED:
            continue
        ranks[k] = frame.groupby(level="season")[k].rank(ascending=k in LOWER_IS_BETTER, method="min")
    return ranks


def team_seasons(engine, team_id: str) -> dict | None:
    keys = [k for k, _, _ in TEAM_COLUMNS]
    out = {"team_id": str(team_id), "columns": [{"key": k, "label": lab, "fmt": f} for k, lab, f in TEAM_COLUMNS],
           "lower_is_better": sorted(LOWER_IS_BETTER)}
    found = False
    for key, stype in SEASON_TYPES.items():
        league = _team_frame(engine, stype)
        if league.empty or str(team_id) not in league.index.get_level_values("team_id"):
            out[key] = {"seasons": [], "career": None}
            continue
        found = True
        ranks = _ranks(league, keys)
        teams_per_season = league.groupby(level="season").size()
        mine = league.xs(str(team_id), level="team_id")
        my_ranks = ranks.xs(str(team_id), level="team_id")
        rows = _rows(mine, keys, ("team",))
        for r in rows:
            rk = my_ranks.loc[r["season"]]
            r["ranks"] = {k: int(rk[k]) for k in rk.index if pd.notna(rk[k])}
            r["of"] = int(teams_per_season.loc[r["season"]])
        out[key] = {"seasons": rows, "career": None}
        out["team"] = rows[0]["team"]
    return out if found else None
