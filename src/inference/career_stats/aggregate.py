"""
Collapsing one stat's per-season long-format rows into the four temporal
numbers a career panel reports: current, previous, career average, career
total.

Why this is not a one-liner
---------------------------
The three quantities aggregate differently, and getting it wrong produces
numbers that look plausible and are simply false.

  RATE   (FG% allowed, FG%) — a career rate MUST be recomputed from summed
         makes and attempts. Averaging per-season percentages weights a
         nine-game season identically to an eighty-two-game one, which is how
         a player with one hot cameo ends up "career 60% from three".

  PER_GAME (blocks, steals, deflections) — stored per game, so a season total
         is value x games, and the career average is total games divided into
         total events. Again, not the mean of the per-season averages.

  DIFF   (FG% allowed versus league normal) — a difference of two rates, so it
         is aggregated as an attempt-weighted mean of the per-season
         differences, not as a plain mean.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.features.creation import _previous_season

RATE, PER_GAME, DIFF, COUNT = "rate", "per_game", "diff", "count"


@dataclass
class CareerStat:
    key: str
    label: str
    group: str
    fmt: str
    kind: str
    current: float | None
    previous: float | None
    career_avg: float | None
    career_total: float | None
    first_season: str | None
    last_season: str | None
    seasons: int
    # Every season this player has a value for, in order — what each season
    # ACTUALLY was, not a running cumulative average up to that point (that
    # would make an early-career dip invisible, smoothed away by everything
    # that came after it). This is what a trend-line chart draws from; a
    # missing season is simply absent from the list rather than a zero.
    series: list[dict]


def _f(value) -> float | None:
    """Anything numpy/pandas can hand back -> JSON-safe float, NaN -> None."""
    if value is None:
        return None
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    return value


def aggregate(long: pd.DataFrame, key: str, kind: str, season: str) -> dict:
    """Collapse one stat's per-season rows into the four temporal numbers."""
    part = long[long["key"] == key]
    if part.empty:
        return {
            "current": None, "previous": None, "career_avg": None,
            "career_total": None, "first_season": None, "last_season": None,
            "seasons": 0, "series": [],
        }

    prev_season = _previous_season(season)
    by_season = part.set_index("season")

    # Which per-season column the "this season"/"last season" columns show.
    # A counting stat's season figure is its TOTAL — "1,211 shots defended"
    # is the season; "17" is a per-game rate nobody asked for and which does
    # not square with the career total sitting next to it.
    season_column = "total" if kind == COUNT else "per_game"

    def at(s):
        return _f(by_season[season_column].get(s)) if s in by_season.index else None

    if kind in (RATE, DIFF):
        den = part["den"].sum(skipna=True)
        career_avg = _f(part["num"].sum(skipna=True) / den) if den else None
        # A rate has no meaningful sum; its volume is a separate stat.
        career_total = None
    else:
        total = part["total"].sum(skipna=True)
        career_total = _f(total)
        if kind == PER_GAME:
            games = part["games"].sum(skipna=True)
            career_avg = _f(total / games) if games else None
        else:
            career_avg = _f(part["total"].mean(skipna=True))

    seasons = sorted(part["season"].dropna().unique())
    # The chartable history: what this stat WAS in each season, not a
    # cumulative average up to that point — a trend line has to be able to
    # show a dip or a peak, which a running average would smooth away.
    series = [
        {"season": s, "value": _f(by_season[season_column].get(s))}
        for s in seasons
    ]
    series = [pt for pt in series if pt["value"] is not None]

    return {
        "current": at(season),
        "previous": at(prev_season),
        "career_avg": career_avg,
        "career_total": career_total,
        "first_season": seasons[0] if seasons else None,
        "last_season": seasons[-1] if seasons else None,
        "seasons": len(seasons),
        "series": series,
    }
