"""
Career stat panels — for one player, every stat this project tracks seen over
time rather than for a single season: the current season, the season before it
as a reference point, a career average, and a career total.

"Career" means career WITHIN OUR DATA, and the window differs per stat — the
NBA only began publishing tracking data partway through the period this
project covers. Every panel therefore reports first_season, last_season and
the number of seasons behind the number, so a client can say "career (8
seasons tracked)" instead of implying we have a player's whole life.

This used to be one 649-line career_stats.py. It is now:
    sources.py   — one "long" builder per data source (defensive activity,
                   defender categories, zone shooting, shot context,
                   passing, play type)
    aggregate.py — collapsing one stat's per-season rows into
                   current/previous/career_avg/career_total (see its
                   docstring for why this genuinely isn't a one-liner)
    catalogue.py — the flat list of every reportable stat, plus section
                   metadata
    __init__.py  — this file: career_panel(), the actual orchestration

This __init__.py re-exports everything so every existing
`from src.inference.career_stats import X` (or `career_stats.X` on the
module) keeps working unchanged.
"""
from __future__ import annotations

from dataclasses import asdict

import pandas as pd

from src.features.creation import _previous_season
from .aggregate import COUNT, DIFF, PER_GAME, RATE, CareerStat, _f, aggregate as _aggregate
from .catalogue import GROUP_LABELS, GROUP_ORDER, GROUP_SIDE, catalogue as _catalogue
from .sources import (
    BOX_SCORE_COLUMNS,
    DEF_CATEGORIES,
    DEF_CATEGORY_ORDER,
    HUSTLE_STAT_COLUMNS,
    SEASON_TOTAL_COLUMNS,
    _defender_category_long,
    _defensive_activity_long,
    _passing_long,
    _play_type_long,
    _shooting_long,
    _shot_context_long,
)


def career_panel(engine, player_id: str, season: str) -> dict:
    """
    Every tracked stat for one player across every season we hold, as
    current / previous / career average / career total.
    """
    parts = [
        _defensive_activity_long(engine, player_id),
        _defender_category_long(engine, player_id),
        _shooting_long(engine, player_id),
        _shot_context_long(engine, player_id),
        _passing_long(engine, player_id),
        _play_type_long(engine, player_id),
    ]
    parts = [p for p in parts if not p.empty]
    long = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["season", "key", "per_game", "total", "games", "num", "den"]
    )

    stats: list[CareerStat] = []
    for key, label, group, fmt, kind in _catalogue():
        agg = _aggregate(long, key, kind, season)
        if agg["seasons"] == 0:
            continue  # nothing measured for this player, ever — omit the row
        stats.append(CareerStat(key=key, label=label, group=group, fmt=fmt, kind=kind, **agg))

    sections = []
    for group in GROUP_ORDER:
        in_group = [asdict(s) for s in stats if s.group == group]
        if in_group:
            sections.append({
                "group": group, "label": GROUP_LABELS[group],
                "side": GROUP_SIDE[group], "stats": in_group,
            })

    covered = sorted({s for s in long["season"].dropna().unique()}) if not long.empty else []
    return {
        "player_id": str(player_id),
        "season": season,
        "previous_season": _previous_season(season),
        "first_season": covered[0] if covered else None,
        "last_season": covered[-1] if covered else None,
        "seasons_covered": len(covered),
        "sections": sections,
    }


__all__ = [
    "career_panel",
    "CareerStat", "RATE", "PER_GAME", "DIFF", "COUNT",
    "DEF_CATEGORIES", "DEF_CATEGORY_ORDER",
    "HUSTLE_STAT_COLUMNS", "BOX_SCORE_COLUMNS", "SEASON_TOTAL_COLUMNS",
    "GROUP_LABELS", "GROUP_ORDER", "GROUP_SIDE",
]
