"""
Player ratings — offensive and defensive, computed from real data.

What was wrong
--------------
Ratings were invented in the frontend (`src/lib/api.ts`) from a stack of magic
numbers: `score = 65; score += (fg - 43) * 0.8; ...` with hard clamps at each
term and at the end. Three things followed from that, all of them visible:

  * `ast`, `tov` and `ft_pct` are NULL for every player in the database — they
    were never ingested. The formula read them anyway, so its entire playmaking
    contribution evaluated to zero for everyone. The best passer alive scored
    the same on playmaking as a player who has never recorded an assist.

  * The remaining terms were dominated by raw FG%, which rewards low-volume rim
    finishers. The formula's top-rated offensive players were end-of-bench
    centres who only dunk, while primary creators sat in the low 70s.

  * Both scales saturated. 41% of the league was pinned at the offensive floor
    of 68, and 69 players were tied at the defensive cap of 87 — so "87" never
    meant a rating, it meant "hit the ceiling".

What this does instead
----------------------
Ratings are percentile ranks within the season, computed from measured data,
so a number is defined rather than asserted: 99 is the best in the league, 50
is exactly average, and the scale cannot saturate because ranks are uniform by
construction.

Three deliberate choices:

  Volume-shrunk efficiency. Scoring is points per shot attempt, regressed
  toward the league mean by attempts (the same empirical-Bayes idea the model's
  features use). This is what stops a 40-attempt dunker from outranking a
  primary option — his raw efficiency is high, his evidence is thin, and the
  prior pulls him back to the middle where he belongs.

  Real playmaking. `player_tracking_stats` carries actual assist, potential-
  assist and assist-points-created figures, which the old formula's NULL
  columns did not. Playmaking is now a real term with real weight.

  Volume counts. An efficient low-usage role player is not a better offensive
  player than a high-usage creator at similar efficiency, and a rating that
  says otherwise is the reason the old one looked absurd.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# How many shot attempts of evidence it takes to move a player off the league
# mean scoring rate. Deliberately substantial: points-per-attempt is noisy, and
# the failure mode being corrected is exactly small samples reading as elite.
SCORING_PRIOR_ATTEMPTS = 250.0

# Minimum activity to be rated at all. Below this a rating is noise wearing a
# number, and — worse — including these players compresses the percentile scale
# for everyone who does play.
MIN_SHOTS_FOR_RATING = 100
MIN_MINUTES_FOR_RATING = 8.0

# Ratings are mapped onto this band rather than 0-100. A replacement-level NBA
# player is not a "2 out of 100" — everyone here is among the best few hundred
# basketball players alive, and a scale that starts at zero implies a range the
# population does not have.
RATING_FLOOR, RATING_CEIL = 40, 99

# Offensive composite. Weights are a judgement call and are stated here rather
# than buried in arithmetic, so they can be argued with directly.
OFFENSIVE_WEIGHTS = {
    "scoring": 0.40,      # shrunk points per shot attempt
    "playmaking": 0.28,   # assist points created, potential assists
    "volume": 0.20,       # shot and touch load carried
    "creation": 0.12,     # can he generate his own shot
}


def _percentile(series: pd.Series) -> pd.Series:
    """
    Rank-to-percentile in [0, 1]. NaN stays NaN so a missing input never
    silently reads as "average" — it drops out of the composite instead.
    """
    return series.rank(pct=True, na_option="keep")


def _to_rating(percentile: pd.Series) -> pd.Series:
    return (RATING_FLOOR + percentile * (RATING_CEIL - RATING_FLOOR)).round()


def compute_ratings(engine, season: str) -> pd.DataFrame:
    """
    Offensive and defensive ratings for every sufficiently active player in a
    season, plus the components they were built from.
    """
    # ── Scoring: points per shot attempt, shrunk by volume ────────────────
    scoring = pd.read_sql(f"""
        SELECT s.player_id,
               COUNT(*) AS fga,
               SUM(s.shot_made * CASE WHEN s.shot_type = '3PT Field Goal'
                                      THEN 3 ELSE 2 END) AS points
        FROM shots s
        WHERE s.season = '{season}'
          AND s.zone IS NOT NULL AND s.zone != 'Backcourt'
        GROUP BY s.player_id
    """, engine)

    if scoring.empty:
        return pd.DataFrame()

    league_ppa = scoring["points"].sum() / scoring["fga"].sum()
    scoring["ppa"] = (
        (scoring["points"] + SCORING_PRIOR_ATTEMPTS * league_ppa)
        / (scoring["fga"] + SCORING_PRIOR_ATTEMPTS)
    )

    # ── Tracking: playmaking, usage, creation ────────────────────────────
    tracking = pd.read_sql(f"""
        SELECT player_id, gp, min_per_game,
               ast, potential_ast, ast_points_created, ast_to_pass_pct_adj,
               touches, time_of_poss, avg_drib_per_touch, drives
        FROM player_tracking_stats
        WHERE season = '{season}'
    """, engine)

    df = scoring.merge(tracking, on="player_id", how="left")

    # Per-36 normalization, so a bench creator is not penalised for minutes.
    minutes = df["min_per_game"].replace(0, np.nan)
    for col in ("ast_points_created", "potential_ast", "ast", "touches",
                "time_of_poss", "drives"):
        if col in df.columns:
            df[f"{col}_per36"] = df[col] / minutes * 36.0

    games = df["gp"].replace(0, np.nan)
    df["fga_per36"] = df["fga"] / games / minutes * 36.0

    # ── Defence: FG% allowed vs league normal, volume-weighted ───────────
    defence = pd.read_sql(f"""
        SELECT player_id,
               SUM(d_fga) AS d_fga,
               SUM(pct_plusminus * d_fga) / NULLIF(SUM(d_fga), 0) AS pm
        FROM defender_stats
        WHERE season = '{season}' AND defense_category != 'Overall'
          AND d_fga IS NOT NULL
        GROUP BY player_id
    """, engine)

    # Shrink toward zero (league-normal) by how many shots the player actually
    # defended — the same small-sample problem the offensive side has, and a
    # worse one here: the median player has only 250 defended attempts, and the
    # extreme values all come from thin samples (one wing posts -0.194 on 129
    # attempts, better than any high-volume defender in the league).
    #
    # A caveat worth stating plainly, because the UI should not imply more than
    # this number supports: FG%-allowed-versus-normal is a weak defensive
    # metric. It credits whoever was nearest the shooter, ignores who was
    # actually responsible, and is heavily confounded by scheme and by the
    # difficulty of the assignment a player draws. It identifies elite rim
    # protection reasonably well; it does not reliably rank perimeter
    # defenders against each other.
    if not defence.empty:
        prior_fga = 400.0
        defence["pm_shrunk"] = (
            defence["pm"] * defence["d_fga"] / (defence["d_fga"] + prior_fga)
        )
        # Below this, decline to publish a number at all rather than publish a
        # confident-looking one built on noise.
        defence.loc[defence["d_fga"] < 150, "pm_shrunk"] = np.nan
        df = df.merge(
            defence[["player_id", "pm_shrunk", "d_fga"]], on="player_id", how="left"
        )
    else:
        df["pm_shrunk"] = np.nan
        df["d_fga"] = np.nan

    # ── Eligibility ──────────────────────────────────────────────────────
    minutes_played = pd.to_numeric(df["min_per_game"], errors="coerce").fillna(0.0)
    eligible = (
        (df["fga"] >= MIN_SHOTS_FOR_RATING)
        & (minutes_played >= MIN_MINUTES_FOR_RATING)
    )
    df = df[eligible].copy()
    if df.empty:
        return df

    # ── Composites ───────────────────────────────────────────────────────
    parts = {
        "scoring": _percentile(df["ppa"]),
        "playmaking": (
            _percentile(df.get("ast_points_created_per36"))
            * 0.6
            + _percentile(df.get("potential_ast_per36")) * 0.4
        ),
        "volume": (
            _percentile(df["fga_per36"]) * 0.6
            + _percentile(df.get("time_of_poss_per36")) * 0.4
        ),
        "creation": (
            _percentile(df.get("avg_drib_per_touch")) * 0.5
            + _percentile(df.get("drives_per36")) * 0.5
        ),
    }

    # Renormalize over whichever components a player actually has, so a missing
    # tracking profile shrinks the composite's basis rather than dragging the
    # player toward zero.
    total = pd.Series(0.0, index=df.index)
    weight = pd.Series(0.0, index=df.index)
    for name, values in parts.items():
        w = OFFENSIVE_WEIGHTS[name]
        filled = values.fillna(0.0)
        present = values.notna().astype(float)
        total += w * filled
        weight += w * present
        df[f"component_{name}"] = (values * 100).round()

    df["off_percentile"] = total / weight.replace(0, np.nan)
    df["off_rating"] = _to_rating(_percentile(df["off_percentile"]))

    # Defence: lower FG% allowed than league-normal is better, so invert.
    df["def_rating"] = _to_rating(_percentile(-df["pm_shrunk"]))
    df["component_defence"] = (_percentile(-df["pm_shrunk"]) * 100).round()

    return df[[
        "player_id", "off_rating", "def_rating",
        "component_scoring", "component_playmaking", "component_volume",
        "component_creation", "component_defence",
        "fga", "ppa", "pm_shrunk",
    ]]


_CACHE: dict[str, pd.DataFrame] = {}


def resolve_rating_season(engine, season: str) -> str:
    """
    The most recent season at or before `season` that actually has shots.

    The API defaults to the latest season present in `players`, which is a
    roster-only season for most of the year — 2026-27 rosters exist long before
    a single game is played. Rating that season produces nothing, and the UI
    then renders a dash next to every player. Falling back to the most recent
    season with real shot data means a rating is shown from the moment rosters
    are known, based on the last season the player actually played.
    """
    from sqlalchemy import text

    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT MAX(season) FROM shots WHERE season <= :season
        """), {"season": season}).fetchone()
    return (row[0] if row and row[0] else season)


def ratings_for_season(engine, season: str) -> pd.DataFrame:
    """
    Cached per season. Ratings are percentile ranks over the whole league, so
    they cannot be computed one player at a time — the population is part of
    the definition — and recomputing the league on every search keystroke would
    be wasteful.
    """
    season = resolve_rating_season(engine, season)
    if season not in _CACHE:
        _CACHE[season] = compute_ratings(engine, season)
    return _CACHE[season]


def rating_lookup(engine, season: str) -> dict[str, dict]:
    """`{player_id: {off_rating, def_rating, ...}}` for the API layer."""
    table = ratings_for_season(engine, season)
    if table.empty:
        return {}
    return {
        row["player_id"]: {
            "off_rating": None if pd.isna(row["off_rating"]) else int(row["off_rating"]),
            "def_rating": None if pd.isna(row["def_rating"]) else int(row["def_rating"]),
            "scoring": None if pd.isna(row["component_scoring"]) else int(row["component_scoring"]),
            "playmaking": None if pd.isna(row["component_playmaking"]) else int(row["component_playmaking"]),
            "volume": None if pd.isna(row["component_volume"]) else int(row["component_volume"]),
            "creation": None if pd.isna(row["component_creation"]) else int(row["component_creation"]),
            "defence": None if pd.isna(row["component_defence"]) else int(row["component_defence"]),
        }
        for _, row in table.iterrows()
    }


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.db.database import get_engine

    engine = get_engine()
    season = sys.argv[1] if len(sys.argv) > 1 else "2025-26"
    table = compute_ratings(engine, season)

    names = pd.read_sql(
        f"SELECT DISTINCT player_id, name FROM players WHERE season = '{season}'", engine
    )
    table = table.merge(names, on="player_id", how="left")

    print(f"\n{len(table)} players rated in {season}\n")
    print("── Top 12 offensive ──")
    for _, r in table.nlargest(12, "off_rating").iterrows():
        print(f"  {str(r['name'])[:24]:<24} OFF {int(r.off_rating):>2}  "
              f"(score {int(r.component_scoring):>2} / pass {int(r.component_playmaking):>2} / "
              f"vol {int(r.component_volume):>2} / create {int(r.component_creation):>2})")
    print("\n── Top 12 defensive ──")
    for _, r in table.nlargest(12, "def_rating").iterrows():
        print(f"  {str(r['name'])[:24]:<24} DEF {int(r.def_rating):>2}")
