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

More maps (player_shot_maps, player_shot_change, team_shot_map)
---------------------------------------------------------------
The advanced view splits the same picture by shot type (from the league's
own play-by-play taxonomy), by clutch time, and by the shots his defensive
assignments took, and shows how his shot diet moved from last season; the
team pages draw the team's shots and its opponents'. Every subset is
compared with the league's rate for the SAME subset in the same zone: a
pull-up is harder than the average jumper, so measuring a player's
pull-ups against all jumpers would call every pull-up shooter cold. Each
map also carries `expected_fg_pct`, the league rate for his own zone mix,
so its headline gap is adjusted for where the shots came from.
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
SIGMA_CHANGE_FT = 3.0          # smoothing radius for the season-to-season change map
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


def league_zone_rates(engine, seasons: list[str], season_type: str = "regular",
                      subset: str | None = None) -> dict[str, float]:
    """The league's FG% in each zone over these seasons — for all shots, or
    for one subset: a shot family (SHOT_FAMILIES) or "clutch" / "not_clutch".
    Cached: each breakdown reads every shot in the window once."""
    if subset is None:
        by = "all"
    elif subset in ("clutch", "not_clutch"):
        by = "clutch"
    else:
        by = "family"
    cache = _league_cache.setdefault(engine, {})
    key = (tuple(seasons), season_type, by)
    if key not in cache:
        cache[key] = _query_league_breakdown(engine, seasons, season_type, by)
    table = cache[key]
    rows = table[table["subset"] == (subset or "all")]
    return {r.zone: r.fgm / r.fga for r in rows.itertuples() if r.zone in ZONE_PRIOR and r.fga > 0}


def _query_league_breakdown(engine, seasons: list[str], season_type: str, by: str) -> pd.DataFrame:
    """[subset, zone, fga, fgm] for the league over these seasons."""
    params = {"seasons": seasons, "po": SEASON_TYPES[season_type]}
    if by == "family":
        sql = """
            SELECT s.zone, sc.shot_subtype AS subtype, COUNT(*) AS fga, SUM(s.shot_made) AS fgm
            FROM shots s JOIN shot_context sc ON sc.shot_id = s.shot_id
            WHERE s.season IN :seasons AND s.playoff_flag = :po
            GROUP BY s.zone, sc.shot_subtype
        """
    elif by == "clutch":
        sql = f"""
            SELECT zone, {CLUTCH_SQL} AS clutch, COUNT(*) AS fga, SUM(shot_made) AS fgm
            FROM shots s
            WHERE season IN :seasons AND playoff_flag = :po AND score_diff IS NOT NULL
            GROUP BY zone, clutch
        """
    else:
        sql = """
            SELECT zone, COUNT(*) AS fga, SUM(shot_made) AS fgm FROM shots
            WHERE season IN :seasons AND playoff_flag = :po GROUP BY zone
        """
    df = pd.read_sql(text(sql).bindparams(bindparam("seasons", expanding=True)), engine, params=params)
    if by == "family":
        df["subset"] = df["subtype"].map(shot_family)
        df = df.dropna(subset=["subset"])
    elif by == "clutch":
        df["subset"] = np.where(df["clutch"] == 1, "clutch", "not_clutch")
    else:
        df["subset"] = "all"
    return df.groupby(["subset", "zone"], as_index=False)[["fga", "fgm"]].sum()


# ── Subsets ────────────────────────────────────────────────────────────────
# The league's play-by-play shot taxonomy, grouped into six families. Order
# matters: "Driving Floating Jump Shot" is a floater, not a jumper, and
# "Tip Layup Shot" belongs with the tips.
SHOT_FAMILIES = [
    ("off_dribble", "Off the dribble", "Pull-ups and step-backs"),
    ("set_jumpers", "Set jumpers", "Jumpers that are not pull-ups, step-backs or fadeaways: mostly catch-and-shoot"),
    ("fadeaways", "Fadeaways & turnarounds", "Fadeaway and turnaround jumpers"),
    ("floaters_hooks", "Floaters & hooks", "Floaters, runners and hook shots"),
    ("layups", "Layups", "Driving, cutting and putback layups and finger rolls"),
    ("dunks", "Dunks & tips", "Dunks, alley-oops and tip-ins"),
]


def shot_family(subtype: str | None) -> str | None:
    if not subtype:
        return None
    t = subtype.lower()
    if "dunk" in t or t.startswith("tip"):
        return "dunks"
    if "layup" in t or "finger roll" in t:
        return "layups"
    if "hook" in t or "float" in t:
        return "floaters_hooks"
    if "fadeaway" in t or "turnaround" in t:
        return "fadeaways"
    if "pull" in t or "step back" in t:
        return "off_dribble"
    if "jump" in t or "bank" in t:
        return "set_jumpers"
    return None


# Clutch time: the last five minutes of the fourth quarter or overtime with
# the score within five — the NBA's own definition, and the model's.
CLUTCH_SQL = "(s.quarter >= 4 AND s.time_remaining <= 300 AND ABS(s.score_diff) <= 5)"


def _is_clutch(shots: pd.DataFrame) -> pd.Series:
    return (shots["quarter"] >= 4) & (shots["time_remaining"] <= 300) & (shots["score_diff"].abs() <= 5)


def _field(values: np.ndarray, digits: int) -> list:
    """Row-major list, baseline row first; NaN becomes null for JSON."""
    flat = np.round(values.astype(float).ravel(), digits)
    return [None if np.isnan(v) else float(v) for v in flat]


def _round(v, digits=4):
    return None if v is None or pd.isna(v) else round(float(v), digits)


def _counts(shots: pd.DataFrame) -> tuple[np.ndarray, int]:
    """Shots per grid square, and how many landed on the grid (heaves from
    beyond 33 ft are counted in the totals, not drawn)."""
    x = shots["loc_x"].astype(float).to_numpy() / 10.0
    y = shots["loc_y"].astype(float).to_numpy() / 10.0
    ix = np.floor((x - X0) / CELL).astype(int)
    iy = np.floor((y - Y0) / CELL).astype(int)
    on = (ix >= 0) & (ix < NX) & (iy >= 0) & (iy < NY)
    att = np.zeros((NY, NX))
    np.add.at(att, (iy[on], ix[on]), 1.0)
    return att, int(on.sum())


def _frequency(shots: pd.DataFrame, sigma_ft: float = SIGMA_FREQ_FT) -> tuple[np.ndarray, int]:
    """Share of these shots per square foot, blurred with a sigma_ft kernel."""
    att, drawn = _counts(shots)
    return gaussian_filter(att, sigma=sigma_ft / CELL, mode="constant") / max(drawn, 1), drawn


def _share_and_variance(shots: pd.DataFrame, sigma_ft: float) -> tuple[np.ndarray, np.ndarray]:
    """Blurred share of shots per square foot and its sampling variance,
    treating the shot counts as Poisson: with a normalised Gaussian kernel
    w of width s, var(sum w * count) = sum w^2 * count, and sum w^2 is the
    count blurred at s/sqrt 2 divided by 4 pi s^2."""
    att, drawn = _counts(shots)
    n = max(drawn, 1)
    s = sigma_ft / CELL
    share = gaussian_filter(att, sigma=s, mode="constant") / n
    var = gaussian_filter(att, sigma=s / np.sqrt(2.0), mode="constant") / (4.0 * np.pi * s * s) / (n * n)
    return share, var


def build_shot_map(shots: pd.DataFrame, league: dict[str, float]) -> dict:
    """Frequency field and the zone table (with each zone's adjusted
    FG%-vs-league and support) for one set of shots."""
    frequency, drawn = _frequency(shots)

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

    # The league's FG% on this exact zone mix: what an average shooter would
    # have made taking the same shots from the same places.
    rated = [z for z in zone_rows if z["fga"] and z["league_fg_pct"] is not None]
    rated_fga = sum(z["fga"] for z in rated)
    expected = sum(z["fga"] * z["league_fg_pct"] for z in rated) / rated_fga if rated_fga else None
    makes = int(shots["shot_made"].sum()) if len(shots) else 0
    fg_pct = makes / len(shots) if len(shots) else None

    return {
        "grid": {"x0": X0, "y0": Y0, "cell": CELL, "nx": NX, "ny": NY},
        "attempts": len(shots),
        "makes": makes,
        "fg_pct": _round(fg_pct),
        "expected_fg_pct": _round(expected),
        "vs_expected": _round(None if fg_pct is None or expected is None else fg_pct - expected),
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


# ── More maps ──────────────────────────────────────────────────────────────
MAP_GROUPS = ("shot_types", "clutch", "defense")
_DETAIL_COLUMNS = ("s.loc_x, s.loc_y, s.shot_made, s.zone, s.quarter, s.time_remaining, "
                   "s.score_diff, sc.shot_subtype")


def _load_detail(engine, where: str, params: dict) -> pd.DataFrame:
    stmt = text(f"""
        SELECT {_DETAIL_COLUMNS}
        FROM shots s LEFT JOIN shot_context sc ON sc.shot_id = s.shot_id
        WHERE {where}
    """).bindparams(bindparam("seasons", expanding=True))
    return pd.read_sql(stmt, engine, params=params)


def player_shot_maps(engine, player_id: str, season: str, group: str,
                     season_type: str = "regular", span: int = 1) -> dict | None:
    """A set of maps for the advanced view: his shots split by shot type
    ("shot_types"), clutch time vs the rest ("clutch"), or the shots taken
    by the players he was assigned to guard ("defense")."""
    if group not in MAP_GROUPS:
        raise ValueError(f"group must be one of {MAP_GROUPS}")
    seasons = season_window(season, span)
    params = {"pid": player_id, "seasons": seasons, "po": SEASON_TYPES[season_type]}
    who = "s.defender_id" if group == "defense" else "s.player_id"
    shots = _load_detail(engine, f"{who} = :pid AND s.season IN :seasons AND s.playoff_flag = :po", params)
    if shots.empty:
        return None

    total = len(shots)
    maps = []

    def add(key, label, note, subset_shots, subset):
        out = build_shot_map(subset_shots, league_zone_rates(engine, seasons, season_type, subset))
        maps.append({"key": key, "label": label, "note": note,
                     "share": round(len(subset_shots) / total, 4), **out})

    if group == "shot_types":
        family = shots["shot_subtype"].map(shot_family)
        for key, label, note in SHOT_FAMILIES:
            add(key, label, note, shots[family == key], key)
    elif group == "clutch":
        known = shots[shots["score_diff"].notna()]
        clutch = _is_clutch(known)
        add("clutch", "Clutch time", "Last 5 minutes of the 4th or overtime, score within 5",
            known[clutch], "clutch")
        add("not_clutch", "Rest of the game", "Every other shot", known[~clutch], "not_clutch")
    else:
        add("defense", "Shots by his assignments",
            "Shots by the player he guarded most in each game. The matchup is assigned per "
            "game, not per shot, so this is where his assignments shot, not who contested each one.",
            shots, None)
    return {"player_id": player_id, "season": season, "seasons": seasons,
            "season_type": season_type, "group": group, "maps": maps}


def player_shot_change(engine, player_id: str, season: str, season_type: str = "regular") -> dict | None:
    """How his shot diet moved from last season to this one: the change in
    his share of shots per square foot, and per zone."""
    previous = season_window(season, 2)[0]
    load = lambda ssn: _load(engine, "s.player_id = :pid AND s.season IN :seasons AND s.playoff_flag = :po",  # noqa: E731
                             {"pid": player_id, "seasons": [ssn], "po": SEASON_TYPES[season_type]})
    now, before = load(season), load(previous)
    if now.empty or before.empty:
        return None
    # The difference of two seasons' densities is mostly noise wherever he
    # shoots little, so each spot's change is also expressed in standard
    # errors, and `confidence` fades a change in from 2 SE to full at 3.5.
    # Checked against a null: one season split into random halves (no real
    # change) colours 3% of the floor at 2 SE, where real season-to-season
    # changes colour 18% (median of 60 high-volume players).
    f_now, v_now = _share_and_variance(now, SIGMA_CHANGE_FT)
    f_before, v_before = _share_and_variance(before, SIGMA_CHANGE_FT)
    diff = f_now - f_before
    se = np.sqrt(v_now + v_before)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(se > 0, np.abs(diff) / se, 0.0)
    confidence = np.clip((z - 2.0) / 1.5, 0.0, 1.0)

    def zone_split(df):
        g = df[df["zone"].isin(ZONE_ORDER)].groupby("zone")["shot_made"].agg(fga="size", fgm="sum")
        return {z: (int(g.loc[z, "fga"]), int(g.loc[z, "fgm"])) if z in g.index else (0, 0) for z in ZONE_ORDER}

    zn, zb = zone_split(now), zone_split(before)
    zones = []
    for z in ZONE_ORDER:
        (fga, fgm), (pfga, pfgm) = zn[z], zb[z]
        share, pshare = fga / len(now), pfga / len(before)
        zones.append({
            "zone": z, "fga": fga, "previous_fga": pfga,
            "share": round(share, 4), "previous_share": round(pshare, 4),
            "share_change": round(share - pshare, 4),
            "fg_pct": _round(fgm / fga if fga else None),
            "previous_fg_pct": _round(pfgm / pfga if pfga else None),
        })
    return {
        "player_id": player_id, "season": season, "previous_season": previous,
        "season_type": season_type,
        "grid": {"x0": X0, "y0": Y0, "cell": CELL, "nx": NX, "ny": NY},
        "attempts": len(now), "previous_attempts": len(before),
        "change": _field(diff, 5),
        "confidence": _field(confidence, 3),
        # The larger of the two seasons' peak densities: the change map's
        # colour scale, so "twice as often" reads the same on every player.
        "scale": round(float(max(f_now.max(), f_before.max())), 5),
        "zones": zones,
    }


def team_abbrev(engine, team_id: str, season: str) -> str | None:
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT team_abbrev FROM team_stats WHERE team_id = :tid
            ORDER BY (season = :season) DESC, season DESC LIMIT 1
        """), {"tid": team_id, "season": season}).fetchone()
    return row[0] if row else None


def _team_shots(engine, abbrev: str, seasons: list[str], side: str, season_type: str) -> pd.DataFrame:
    """A team's shots (offense) or its opponents' (defense), in game order."""
    # The date window is a pre-filter that lets SQLite start from the team's
    # games (as in supporting_cast.py); s.season is the exact test.
    params = {"t": abbrev, "seasons": seasons, "po": SEASON_TYPES[season_type],
              "start": f"{seasons[0][:4]}-09-01", "end": f"{int(seasons[-1][:4]) + 1}-11-01"}
    op = "=" if side == "offense" else "!="
    stmt = text(f"""
        SELECT s.loc_x, s.loc_y, s.shot_made, s.zone, g.date
        FROM games g CROSS JOIN shots s ON s.game_id = g.game_id
        WHERE (g.home_team = :t OR g.away_team = :t)
          AND g.date >= :start AND g.date < :end
          AND s.season IN :seasons AND s.playoff_flag = :po AND s.home_away IS NOT NULL
          AND (CASE WHEN s.home_away = 1 THEN g.home_team ELSE g.away_team END) {op} :t
        ORDER BY g.date, s.game_id, s.quarter, s.time_remaining DESC
    """).bindparams(bindparam("seasons", expanding=True))
    return pd.read_sql(stmt, engine, params=params)


def team_shot_map(engine, team_id: str, season: str, side: str = "offense",
                  season_type: str = "regular", span: int = 1) -> dict | None:
    """Where a team shoots from and how well by zone ("offense"), or the
    same for the shots its opponents took against it ("defense")."""
    if side not in ("offense", "defense"):
        raise ValueError("side must be offense or defense")
    abbrev = team_abbrev(engine, team_id, season)
    if abbrev is None:
        return None
    seasons = season_window(season, span)
    shots = _team_shots(engine, abbrev, seasons, side, season_type)
    if shots.empty:
        return None
    out = build_shot_map(shots, league_zone_rates(engine, seasons, season_type))
    return {"team_id": team_id, "team_abbrev": abbrev, "side": side, "season": season,
            "seasons": seasons, "season_type": season_type, **out}


def warm_league_cache(engine) -> None:
    """Compute the league breakdowns for the latest season with shots, so the
    first shot map anyone opens does not pay for them (~1-2 s each). Run in a
    background thread at API startup."""
    try:
        with engine.connect() as conn:
            season = conn.execute(text("SELECT MAX(season) FROM shots")).scalar()
        if not season:
            return
        for span in (1, 3):
            seasons = season_window(season, span)
            for subset in (None, SHOT_FAMILIES[0][0], "clutch"):
                league_zone_rates(engine, seasons, "regular", subset)
    except Exception as e:  # an optimisation only: requests compute on demand
        print(f"  ⚠ Shot-map cache warm-up skipped: {e}")


# ── Individual shots, for the animated shot chart ─────────────────────────
MAX_SHOTS = 2500


def player_shots(engine, player_id: str, season: str, season_type: str = "regular",
                 span: int = 1, limit: int = MAX_SHOTS) -> dict | None:
    """Every shot a player took in the window, in the order he took them —
    for the profile's shot replay, which plays them back one by one. Each
    shot is [loc_x, loc_y, made, game number]. Past `limit` the shots are
    thinned evenly, which keeps both the order and the make rate."""
    if season_type not in SEASON_TYPES:
        raise ValueError(f"season_type must be one of {sorted(SEASON_TYPES)}")
    seasons = season_window(season, span)
    stmt = text("""
        SELECT s.loc_x, s.loc_y, s.shot_made, g.date
        FROM shots s JOIN games g ON g.game_id = s.game_id
        WHERE s.player_id = :pid AND s.season IN :seasons AND s.playoff_flag = :po
          AND s.loc_x IS NOT NULL AND s.loc_y IS NOT NULL
        ORDER BY g.date, s.game_id, s.quarter, s.time_remaining DESC
    """).bindparams(bindparam("seasons", expanding=True))
    df = pd.read_sql(stmt, engine, params={"pid": player_id, "seasons": seasons,
                                           "po": SEASON_TYPES[season_type]})
    if df.empty:
        return None
    return {"player_id": player_id, "season": season, "seasons": seasons, "season_type": season_type,
            **_replay_payload(df, limit)}


def _replay_payload(df: pd.DataFrame, limit: int) -> dict:
    """[loc_x, loc_y, made, game number] per shot, in order, thinned evenly
    past `limit` (which keeps both the order and the make rate)."""
    df = df[df["loc_x"].notna() & df["loc_y"].notna()]
    game_number = (df["date"] != df["date"].shift()).cumsum()
    total, makes = len(df), int(df["shot_made"].sum())
    if total > limit:
        keep = np.linspace(0, total - 1, limit).round().astype(int)
        df, game_number = df.iloc[keep], game_number.iloc[keep]
    shots = [[int(x), int(y), int(m), int(g)] for x, y, m, g in
             zip(df["loc_x"], df["loc_y"], df["shot_made"], game_number)]
    return {"attempts": total, "makes": makes, "games": int(game_number.max()), "shots": shots}


def team_shots(engine, team_id: str, season: str, side: str = "offense", season_type: str = "regular",
               span: int = 1, limit: int = MAX_SHOTS) -> dict | None:
    """A team's shots (or its opponents'), in order, for the team replay."""
    if side not in ("offense", "defense"):
        raise ValueError("side must be offense or defense")
    abbrev = team_abbrev(engine, team_id, season)
    if abbrev is None:
        return None
    seasons = season_window(season, span)
    df = _team_shots(engine, abbrev, seasons, side, season_type)
    if df.empty:
        return None
    return {"team_id": team_id, "side": side, "season": season, "seasons": seasons,
            "season_type": season_type, **_replay_payload(df, limit)}
