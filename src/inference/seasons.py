"""
Which season the Stat Engine opens on, and how far into the newest one the
league is.

A new season's rosters exist months before its first game, and its first
games arrive weeks before its numbers mean anything. One game is not a
season: a leaderboard ranked on it — or a percentile tier, a rating, a
radar — would be noise presented as fact. So the Stat Engine stays on the
last season until the new one is established, meaning teams have played
ESTABLISHED_GAMES games on average (about three weeks in). Until then the
new season is one click away (?season=, the page's season switch), just not
the default.

Ten games is where the leaderboards' own volume minimums start to be met by
regulars (1,000 possessions is about ten games of starter's minutes; see
QUALIFIERS in stat_engine/metadata.py), and where a per-game average has
stopped swinging by whole points from one night to the next.
"""
from __future__ import annotations

import weakref

from sqlalchemy import text

import config

ESTABLISHED_GAMES = 10
NBA_TEAMS = 30

# Like every Stat Engine cache: the answer only changes when data is
# ingested, after which the API is restarted.
_CACHE: "weakref.WeakKeyDictionary[object, dict]" = weakref.WeakKeyDictionary()


def regular_season_games(engine, season: str) -> int:
    """Regular-season games played in `season`, from the games table (which
    holds games once they are played). An NBA game id carries its season —
    002 for the regular season, then the last two digits of the year it
    starts in ("0022600001" is 2026-27's first game) — so this is a range
    scan of the primary key: every id with that prefix sorts at or after the
    prefix and before the prefix followed by ':', the character after '9'."""
    prefix = f"002{season[2:4]}"
    with engine.connect() as conn:
        return conn.execute(text("SELECT COUNT(*) FROM games WHERE game_id >= :lo AND game_id < :hi"),
                            {"lo": prefix, "hi": prefix + ":"}).scalar() or 0


def season_status(engine) -> dict:
    """The season the Stat Engine opens on (`default`), the newest season
    with any shots (`newest`), and how far into the newest one the league
    is. `default` is `newest` once that season is established, or when
    there is no earlier season to fall back to."""
    cached = _CACHE.get(engine)
    if cached is None:
        with engine.connect() as conn:
            newest = conn.execute(text("SELECT MAX(season) FROM shots")).scalar()
            previous = conn.execute(text("SELECT MAX(season) FROM shots WHERE season < :s"),
                                    {"s": newest}).scalar() if newest else None
        if newest is None:
            cached = {"default": config.current_season(), "newest": None,
                      "newest_games": 0, "newest_games_per_team": 0.0}
        else:
            played = regular_season_games(engine, newest)
            per_team = 2 * played / NBA_TEAMS
            established = per_team >= ESTABLISHED_GAMES or previous is None
            cached = {"default": newest if established else previous, "newest": newest,
                      "newest_games": played, "newest_games_per_team": round(per_team, 1)}
        cached["established_games"] = ESTABLISHED_GAMES
        _CACHE[engine] = cached
    return dict(cached)


def stat_engine_season(engine) -> str:
    """The season the Stat Engine shows when none is asked for."""
    return season_status(engine)["default"]
