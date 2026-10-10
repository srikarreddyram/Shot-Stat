"""
Shot maps for the Stat Engine profiles: where a player shoots from, and how
well he shoots in each zone compared with the league.

Both come from the shots table's exact court coordinates. Frequency is a
field on a 1-foot grid covering the half court the engine draws (sideline to
sideline, baseline to 38 ft out); efficiency is one adjusted value per zone,
which the client fills into that zone's exact shape on the court. The client
paints, and does no statistics of its own.

Where he shoots: a smooth density
---------------------------------
Each shot is blurred with a 2 ft Gaussian and the result is divided by his
attempts, giving his share of shots per square foot. Shot location is very
stable from one half of a season to the other, so this can be drawn as a
continuous heat map without overstating anything.

How well: by zone, not by spot
------------------------------
Spot-by-spot "hot zone" maps are the obvious thing to draw and they are
mostly noise. Splitting each of the 150 highest-volume players' 2025-26
shots into random halves and drawing the map from each half, the two halves
agreed with a correlation of only 0.16-0.23 at the spot level however the
smoothing was tuned (full-season reliability 0.28-0.38, Spearman-Brown); a
single season simply does not have enough shots per square foot. The same
test on the six court zones gives 0.63 for one season and 0.79 for three,
so efficiency is coloured by zone, and a three-season view is offered for
the steadier picture.

Each zone's rate is shrunk toward the league's rate in that zone before it
is drawn — empirical Bayes, the same idea as the model's features
(src/features/shrinkage.py):

    shown = (makes + K * league) / (attempts + K) - league

K is the zone's prior strength in attempts, estimated by the method of
moments over 2023-26: the true spread between players is only 3-5.5 points
of FG% in any zone, so K = p(1-p) / spread^2 comes out between ~75 at the
rim and ~275 on above-the-break threes. A 20-for-36 corner-three season is
not evidence of a +16-point corner shooter, and the map does not paint it as
one; the raw numbers stay in the zone table next to it.

Each zone's `support` (0-1) fades out zones he barely shoots from.
"""
from __future__ import annotations

import weakref

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from sqlalchemy import bindparam, text

# Grid: feet from the hoop, the engine court's own frame (geometry.ts).
X0, Y0 = -25.0, -5.25          # left sideline, baseline
CELL = 1.0
NX, NY = 50, 38                # 50 ft wide, 38 ft from the baseline
SIGMA_FREQ_FT = 2.0            # smoothing radius for where he shoots
FULL_SUPPORT_SHOTS = 20        # zone attempts for a fully opaque colour

ZONE_ORDER = [
    "Restricted Area", "In The Paint (Non-RA)", "Mid-Range",
    "Left Corner 3", "Right Corner 3", "Above the Break 3",
]
# Prior strength per zone, in attempts (see the module docstring).
ZONE_PRIOR = {
    "Restricted Area": 75,
    "In The Paint (Non-RA)": 90,
    "Mid-Range": 150,
    "Left Corner 3": 200,
    "Right Corner 3": 200,
    "Above the Break 3": 275,
}
SEASON_TYPES = {"regular": 0, "playoffs": 1}

# League zone rates only change when shots are ingested (after which the API
# is restarted), and computing them reads every shot in the window — about
# 230k a season — so they are cached per engine.
_league_cache: "weakref.WeakKeyDictionary[object, dict]" = weakref.WeakKeyDictionary()


def season_window(season: str, n: int) -> list[str]:
    """`season` and the n-1 before it, oldest first: ("2025-26", 3) ->
    ["2023-24", "2024-25", "2025-26"]."""
    start = int(season[:4])
    return [f"{y}-{str(y + 1)[-2:]}" for y in range(start - n + 1, start + 1)]


def _load(engine, where: str, params: dict) -> pd.DataFrame:
    stmt = text(f"SELECT s.loc_x, s.loc_y, s.shot_made, s.zone FROM shots s WHERE {where}")
    if "seasons" in params:
        stmt = stmt.bindparams(bindparam("seasons", expanding=True))
    return pd.read_sql(stmt, engine, params=params)


def league_zone_rates(engine, seasons: list[str], season_type: str = "regular") -> dict[str, float]:
    """The league's FG% in each zone over these seasons (cached)."""
    cache = _league_cache.setdefault(engine, {})
    key = (tuple(seasons), season_type)
    if key not in cache:
        cache[key] = _query_league_zone_rates(engine, seasons, season_type)
    return cache[key]


def _query_league_zone_rates(engine, seasons: list[str], season_type: str) -> dict[str, float]:
    stmt = text("""
        SELECT zone, AVG(shot_made) FROM shots
        WHERE season IN :seasons AND playoff_flag = :po GROUP BY zone
    """).bindparams(bindparam("seasons", expanding=True))
    with engine.connect() as conn:
        rows = conn.execute(stmt, {"seasons": seasons, "po": SEASON_TYPES[season_type]}).fetchall()
    return {zone: float(rate) for zone, rate in rows if zone in ZONE_PRIOR and rate is not None}


def _field(values: np.ndarray, digits: int) -> list:
    """Row-major list, baseline row first; NaN becomes null for JSON."""
    flat = np.round(values.astype(float).ravel(), digits)
    return [None if np.isnan(v) else float(v) for v in flat]


def _round(v, digits=4):
    return None if v is None or pd.isna(v) else round(float(v), digits)


def build_shot_map(shots: pd.DataFrame, league: dict[str, float]) -> dict:
    """Frequency field and the zone table (with each zone's adjusted
    FG%-vs-league and support) for one set of shots."""
    x = shots["loc_x"].astype(float).to_numpy() / 10.0
    y = shots["loc_y"].astype(float).to_numpy() / 10.0
    ix = np.floor((x - X0) / CELL).astype(int)
    iy = np.floor((y - Y0) / CELL).astype(int)
    on = (ix >= 0) & (ix < NX) & (iy >= 0) & (iy < NY)
    att = np.zeros((NY, NX))
    np.add.at(att, (iy[on], ix[on]), 1.0)
    drawn = int(on.sum())
    frequency = gaussian_filter(att, sigma=SIGMA_FREQ_FT / CELL, mode="constant") / max(drawn, 1)

    by_zone = shots[shots["zone"].isin(ZONE_ORDER)].groupby("zone")["shot_made"].agg(fga="size", fgm="sum")
    zone_rows = []
    for zone in ZONE_ORDER:
        fga = int(by_zone.loc[zone, "fga"]) if zone in by_zone.index else 0
        fgm = int(by_zone.loc[zone, "fgm"]) if fga else 0
        lg = league.get(zone)
        fg_pct = fgm / fga if fga else None
        k = ZONE_PRIOR[zone]
        shrunk = None if lg is None else (fgm + k * lg) / (fga + k) - lg
        zone_rows.append({
            "zone": zone,
            "fga": fga,
            "fgm": fgm,
            "fg_pct": _round(fg_pct),
            "league_fg_pct": _round(lg),
            "vs_league": _round(None if fg_pct is None or lg is None else fg_pct - lg),
            "vs_league_adjusted": _round(shrunk),
            "support": round(min(1.0, fga / FULL_SUPPORT_SHOTS), 3),
            "share": round(fga / max(len(shots), 1), 4),
        })

    return {
        "grid": {"x0": X0, "y0": Y0, "cell": CELL, "nx": NX, "ny": NY},
        "attempts": len(shots),
        "makes": int(shots["shot_made"].sum()) if len(shots) else 0,
        "drawn": drawn,
        "frequency": _field(frequency, 5),
        "zones": zone_rows,
    }


def player_shot_map(engine, player_id: str, season: str, season_type: str = "regular",
                    span: int = 1) -> dict | None:
    """Where one player shoots from, and how well vs the league by zone, over
    `season` (span=1) or it and the seasons before it (span=3)."""
    if season_type not in SEASON_TYPES:
        raise ValueError(f"season_type must be one of {sorted(SEASON_TYPES)}")
    seasons = season_window(season, span)
    params = {"pid": player_id, "seasons": seasons, "po": SEASON_TYPES[season_type]}
    shots = _load(engine, "s.player_id = :pid AND s.season IN :seasons AND s.playoff_flag = :po", params)
    if shots.empty:
        return None
    out = build_shot_map(shots, league_zone_rates(engine, seasons, season_type))
    return {"player_id": player_id, "season": season, "seasons": seasons,
            "season_type": season_type, **out}
