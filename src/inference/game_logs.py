"""
Game by game: a player's or team's season as a list of games with the box
score for each, and any one game opened up — its box score against the
season average, and every shot of it for the shot replay.

Box scores come from player_game_logs / team_game_logs (LeagueGameLog,
2016-17 onward — src/ingestion/game_log_ingestor.py) and the shots from the
shots table; the two share the NBA's game_id, and a player's FGA in a game's
box score is exactly the number of shots they took in it.
"""
from __future__ import annotations

import pandas as pd
from sqlalchemy import text

SEASON_TYPES = {"regular": "Regular Season", "playoffs": "Playoffs"}
BOX = ["pts", "reb", "oreb", "dreb", "ast", "stl", "blk", "tov", "pf",
       "fgm", "fga", "fg3m", "fg3a", "ftm", "fta"]
ZONE_ORDER = [
    "Restricted Area", "In The Paint (Non-RA)", "Mid-Range",
    "Left Corner 3", "Right Corner 3", "Above the Break 3",
]


def _num(v, digits: int | None = None):
    if v is None or pd.isna(v):
        return None
    f = float(v)
    if digits is None:
        return int(f) if f.is_integer() else f
    return round(f, digits)


def _pct(made, att):
    return None if not att else round(float(made) / float(att), 4)


def _averages(games: pd.DataFrame, minutes: bool) -> dict:
    """Per-game averages over these games; shooting as total makes over total
    attempts, the same rule as the season averages table."""
    n = len(games)
    out = {"gp": n}
    for col in ["pts", "reb", "ast", "stl", "blk", "tov", "pf", "fg3m"]:
        out[col] = round(float(games[col].sum()) / n, 2) if n else None
    if minutes:
        out["min"] = round(float(games["minutes"].sum()) / n, 1) if n else None
    out["fg_pct"] = _pct(games["fgm"].sum(), games["fga"].sum())
    out["fg3_pct"] = _pct(games["fg3m"].sum(), games["fg3a"].sum())
    out["ft_pct"] = _pct(games["ftm"].sum(), games["fta"].sum())
    out["plus_minus"] = round(float(games["plus_minus"].sum()) / n, 2) if n else None
    return out


def _game_row(r, team_pts=None, team_margin=None) -> dict:
    out = {
        "game_id": r["game_id"],
        "date": str(pd.Timestamp(r["game_date"]).date()),
        "team": r["team_abbrev"],
        "opponent": r["opp_abbrev"],
        "home": bool(r["home"]),
        "win": None if pd.isna(r["win"]) else bool(r["win"]),
        **{c: _num(r[c]) for c in BOX},
        "plus_minus": _num(r["plus_minus"]),
        "fg_pct": _pct(r["fgm"], r["fga"]),
        "fg3_pct": _pct(r["fg3m"], r["fg3a"]),
        "ft_pct": _pct(r["ftm"], r["fta"]),
    }
    if "minutes" in r:
        out["min"] = _num(r["minutes"], 1)
    pts = r["pts"] if team_pts is None else team_pts
    margin = r["plus_minus"] if team_margin is None else team_margin
    if pts is not None and not pd.isna(pts) and margin is not None and not pd.isna(margin):
        out["team_score"] = int(pts)
        out["opponent_score"] = int(round(float(pts) - float(margin)))
    return out


def _highs(games: pd.DataFrame, cols: list[str]) -> dict:
    """The season high in each column, and which game(s) reached it."""
    out = {}
    for col in cols:
        if games[col].notna().any():
            top = games[col].max()
            out[col] = {"value": _num(top), "games": games.loc[games[col] == top, "game_id"].tolist()}
    return out


def _shots(engine, where: str, params: dict) -> pd.DataFrame:
    return pd.read_sql(text(f"""
        SELECT s.loc_x, s.loc_y, s.shot_made, s.quarter, s.time_remaining, s.zone, s.player_id
        FROM shots s JOIN games g ON g.game_id = s.game_id
        WHERE {where}
        ORDER BY s.quarter, s.time_remaining DESC, s.shot_id
    """), engine, params=params)


def _shot_payload(shots: pd.DataFrame) -> dict:
    """[loc_x, loc_y, made, quarter] per shot, in order, plus a zone table."""
    shots = shots[shots["loc_x"].notna() & shots["loc_y"].notna()]
    by_zone = shots[shots["zone"].isin(ZONE_ORDER)].groupby("zone")["shot_made"].agg(fga="size", fgm="sum")
    zones = []
    for z in ZONE_ORDER:
        fga = int(by_zone.loc[z, "fga"]) if z in by_zone.index else 0
        fgm = int(by_zone.loc[z, "fgm"]) if fga else 0
        zones.append({"zone": z, "fga": fga, "fgm": fgm, "fg_pct": _pct(fgm, fga)})
    quarters = shots["quarter"].fillna(1).astype(int)
    return {
        "shots": [[int(x), int(y), int(m), int(q)] for x, y, m, q in
                  zip(shots["loc_x"], shots["loc_y"], shots["shot_made"], quarters)],
        "attempts": len(shots),
        "makes": int(shots["shot_made"].sum()) if len(shots) else 0,
        "periods": int(max(4, quarters.max())) if len(shots) else 4,
        "zones": zones,
    }


# ── Players ────────────────────────────────────────────────────────────────
_PLAYER_LOG = """
    SELECT p.game_id, p.game_date, p.team_id, p.team_abbrev, p.opp_abbrev, p.home, p.win, p.minutes,
           p.pts, p.reb, p.oreb, p.dreb, p.ast, p.stl, p.blk, p.tov, p.pf,
           p.fgm, p.fga, p.fg3m, p.fg3a, p.ftm, p.fta, p.plus_minus,
           t.pts AS team_pts, t.plus_minus AS team_margin
    FROM player_game_logs p
    LEFT JOIN team_game_logs t ON t.game_id = p.game_id AND t.team_id = p.team_id
    WHERE {where}
    ORDER BY p.game_date DESC
"""


def player_game_log(engine, player_id: str, season: str, season_type: str = "regular") -> dict | None:
    """Every game of a player's season, newest first, with the player's box
    score and the final score, plus season averages and highs over those
    games."""
    games = pd.read_sql(text(_PLAYER_LOG.format(
        where="p.player_id = :pid AND p.season = :season AND p.season_type = :st")),
        engine, params={"pid": str(player_id), "season": season, "st": SEASON_TYPES[season_type]})
    if games.empty:
        return None
    return {
        "player_id": str(player_id), "season": season, "season_type": season_type,
        "games": [_game_row(r, r["team_pts"], r["team_margin"]) for _, r in games.iterrows()],
        "averages": _averages(games, minutes=True),
        "highs": _highs(games, ["pts", "reb", "ast", "stl", "blk", "fg3m"]),
    }


def player_game_detail(engine, player_id: str, game_id: str) -> dict | None:
    """One game: the player's box score, the season average to set it
    against, and every shot the player took in it, in order."""
    row = pd.read_sql(text(_PLAYER_LOG.format(where="p.player_id = :pid AND p.game_id = :gid")),
                      engine, params={"pid": str(player_id), "gid": game_id})
    if row.empty:
        return None
    r = row.iloc[0]
    season_row = pd.read_sql(text(
        "SELECT season, season_type FROM player_game_logs WHERE player_id = :pid AND game_id = :gid"),
        engine, params={"pid": str(player_id), "gid": game_id}).iloc[0]
    season_games = pd.read_sql(text(_PLAYER_LOG.format(
        where="p.player_id = :pid AND p.season = :season AND p.season_type = :st")),
        engine, params={"pid": str(player_id), "season": season_row["season"], "st": season_row["season_type"]})
    shots = _shots(engine, "s.player_id = :pid AND s.game_id = :gid", {"pid": str(player_id), "gid": game_id})
    payload = _shot_payload(shots)
    # Each zone's season FG% (same season type, this game included), to set
    # the game's shooting zone by zone against the player's norm.
    season_zones = pd.read_sql(text("""
        SELECT zone, COUNT(*) AS fga, SUM(shot_made) AS fgm FROM shots
        WHERE player_id = :pid AND season = :season AND playoff_flag = :po
        GROUP BY zone
    """), engine, params={"pid": str(player_id), "season": season_row["season"],
                          "po": int(season_row["season_type"] == SEASON_TYPES["playoffs"])}).set_index("zone")
    for z in payload["zones"]:
        fga = int(season_zones.loc[z["zone"], "fga"]) if z["zone"] in season_zones.index else 0
        fgm = int(season_zones.loc[z["zone"], "fgm"]) if fga else 0
        z.update(season_fga=fga, season_fgm=fgm, season_fg_pct=_pct(fgm, fga))
    return {
        "player_id": str(player_id),
        "season": season_row["season"],
        "game": _game_row(r, r["team_pts"], r["team_margin"]),
        "season_averages": _averages(season_games, minutes=True),
        **payload,
    }


# ── Teams ──────────────────────────────────────────────────────────────────
_TEAM_LOG = """
    SELECT game_id, game_date, team_id, team_abbrev, opp_abbrev, home, win,
           pts, reb, oreb, dreb, ast, stl, blk, tov, pf, fgm, fga, fg3m, fg3a, ftm, fta, plus_minus
    FROM team_game_logs WHERE {where} ORDER BY game_date DESC
"""


def team_game_log(engine, team_id: str, season: str, season_type: str = "regular") -> dict | None:
    """Every game of a team's season, newest first, with its box score."""
    games = pd.read_sql(text(_TEAM_LOG.format(where="team_id = :tid AND season = :season AND season_type = :st")),
                        engine, params={"tid": str(team_id), "season": season, "st": SEASON_TYPES[season_type]})
    if games.empty:
        return None
    return {
        "team_id": str(team_id), "season": season, "season_type": season_type,
        "games": [_game_row(r) for _, r in games.iterrows()],
        "averages": _averages(games, minutes=False),
        "highs": _highs(games, ["pts", "reb", "ast", "fg3m"]),
    }


def team_game_detail(engine, team_id: str, game_id: str, side: str = "offense") -> dict | None:
    """One game: the team's box score and its opponent's, the team's players'
    lines, and every shot the team (offense) or its opponent (defense) took."""
    if side not in ("offense", "defense"):
        raise ValueError("side must be offense or defense")
    row = pd.read_sql(text(_TEAM_LOG.format(where="team_id = :tid AND game_id = :gid")),
                      engine, params={"tid": str(team_id), "gid": game_id})
    if row.empty:
        return None
    r = row.iloc[0]
    opponent = pd.read_sql(text(_TEAM_LOG.format(where="team_id != :tid AND game_id = :gid")),
                           engine, params={"tid": str(team_id), "gid": game_id})
    season_row = pd.read_sql(text(
        "SELECT season, season_type FROM team_game_logs WHERE team_id = :tid AND game_id = :gid"),
        engine, params={"tid": str(team_id), "gid": game_id}).iloc[0]
    season_games = pd.read_sql(text(_TEAM_LOG.format(
        where="team_id = :tid AND season = :season AND season_type = :st")),
        engine, params={"tid": str(team_id), "season": season_row["season"], "st": season_row["season_type"]})
    players = pd.read_sql(text("""
        SELECT player_id, player_name, minutes, pts, reb, ast, stl, blk, tov, fgm, fga, fg3m, fg3a,
               ftm, fta, plus_minus
        FROM player_game_logs WHERE game_id = :gid AND team_id = :tid
        ORDER BY pts DESC, minutes DESC
    """), engine, params={"gid": game_id, "tid": str(team_id)})
    op = "=" if side == "offense" else "!="
    shots = _shots(engine, f"""s.game_id = :gid AND s.home_away IS NOT NULL
        AND (CASE WHEN s.home_away = 1 THEN g.home_team ELSE g.away_team END) {op} :abbr""",
                   {"gid": game_id, "abbr": r["team_abbrev"]})
    return {
        "team_id": str(team_id),
        "season": season_row["season"],
        "side": side,
        "game": _game_row(r),
        "opponent_box": _game_row(opponent.iloc[0]) if not opponent.empty else None,
        "season_averages": _averages(season_games, minutes=False),
        "players": [{
            "player_id": p["player_id"], "name": p["player_name"], "min": _num(p["minutes"], 1),
            **{c: _num(p[c]) for c in ["pts", "reb", "ast", "stl", "blk", "tov", "fgm", "fga",
                                       "fg3m", "fg3a", "ftm", "fta"]},
            "plus_minus": _num(p["plus_minus"]),
        } for _, p in players.iterrows()],
        **_shot_payload(shots),
    }
