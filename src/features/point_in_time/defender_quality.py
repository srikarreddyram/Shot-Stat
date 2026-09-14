"""
Point-in-time defender quality.

`defender_stats` (LeagueDashPtDefend) is a season aggregate — the NBA API
never published a per-game or as-of-date version of it. Every defender
feature that reads it (`def_fg_pct_zone`, `def_pct_plusminus_zone`,
`def_fg_pct_overall`, `def_pct_plusminus`, `def_freq_zone`) was therefore
joined by season alone, exactly the leak this package exists to remove for
shooters: a shot contested by a defender contributes to that defender's own
season FG%-allowed, which is then used as a feature describing that same
shot. What follows rebuilds the same (defender, category) -> FG%-allowed
figures from `matchups` + `shots` instead, strictly prior-games-only, using
the same possession-weighted mixture `build.build_defender_mixture` already
uses for physicals — so the fix only touches where the QUALITY numbers come
from, not how they get blended into a shot's feature vector on either path.

Also carries the plain league-average-defender and opponent-zone-defence
lookups, which are simpler point-in-time defender quantities that don't need
the full possession-weighted-exposure machinery below.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.shrinkage import BetaPrior, fit_priors, shrink
from .zones import DEFENSE_CATEGORIES, ZONE_SUFFIX, ZONE_TO_DEF_CATEGORY, ZONES


def league_average_defender(conn, season: str) -> dict:
    """
    The average defender, per defense category.

    Used when the caller names no defender. Leaving those features NaN was
    wrong in a specific way: in the training matrix NaN defender columns mean
    "this shot has no matchup data at all" — an entire class of older or
    unlinked games with its own scoring characteristics — whereas an API caller
    who omits a defender means "against a typical defender". Filling with league
    means says the second thing, which is what the user asked.
    """
    from sqlalchemy import text

    rows = conn.execute(text("""
        SELECT defense_category,
               AVG(d_fg_pct), AVG(pct_plusminus), AVG(freq)
        FROM defender_stats
        WHERE season <= :season AND season_type = 'Regular Season'
        GROUP BY defense_category
    """), {"season": season}).fetchall()

    physicals = conn.execute(text("""
        SELECT AVG(height), AVG(weight), AVG(wingspan)
        FROM players WHERE season <= :season
    """), {"season": season}).fetchone()

    return {
        "by_category": {
            r[0]: {"d_fg_pct": r[1], "pct_plusminus": r[2], "freq": r[3]}
            for r in rows
        },
        "height": physicals[0] if physicals else None,
        "weight": physicals[1] if physicals else None,
        "wingspan": physicals[2] if physicals else None,
    }


def build_opponent_zone_defence(engine) -> pd.DataFrame:
    """
    Per (game, defending team) shooting allowed per zone, from strictly prior
    games.

    Why this exists
    ---------------
    The model's entire knowledge of opponent defence was one season-level
    number, `team_stats.def_rating` — no zone split, not point-in-time. Scoring
    v8 on its held-out season and aggregating residuals by defending team shows
    how much that misses: the standard deviation of team z-scores is 2.22
    overall and **2.82 in the restricted area**, against 1.0 for a correctly
    specified model. Eighteen of thirty teams sit beyond |z| > 2 at the rim.

    Actual rim FG% allowed ranges from 0.622 to 0.709 across teams, and the
    model cannot see any of it.

    The rate is worth carrying because it persists: team rim defence correlates
    +0.43 to +0.79 season over season, so prior games genuinely predict the next
    one rather than describing noise that has already passed.

    Defending team needs no new data — it is `games.home_team`/`away_team`
    selected by `shots.home_away`.

    Returns one row per (game_id, def_team) with, per zone suffix S:
        opp_def_mk_S, opp_def_att_S   makes and attempts allowed before this game
    """
    counts = pd.read_sql("""
        SELECT g.game_id,
               g.date AS game_date,
               CASE WHEN s.home_away = 1 THEN g.away_team ELSE g.home_team END
                   AS def_team,
               s.zone,
               SUM(s.shot_made) AS makes,
               COUNT(*)         AS attempts
        FROM shots s
        JOIN games g ON s.game_id = g.game_id
        WHERE s.zone IS NOT NULL
          AND s.zone != 'Backcourt'
          AND s.home_away IS NOT NULL
        GROUP BY g.game_id, g.date, def_team, s.zone
    """, engine)

    if counts.empty:
        return pd.DataFrame(columns=["game_id", "def_team"])

    counts["game_date"] = pd.to_datetime(counts["game_date"])

    team_games = (
        counts[["game_id", "def_team", "game_date"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    grid = team_games.merge(pd.DataFrame({"zone": ZONES}), how="cross")
    grid = grid.merge(
        counts[["game_id", "def_team", "zone", "makes", "attempts"]],
        on=["game_id", "def_team", "zone"], how="left",
    )
    grid["makes"] = grid["makes"].fillna(0.0)
    grid["attempts"] = grid["attempts"].fillna(0.0)

    grid = grid.sort_values(
        ["def_team", "zone", "game_date", "game_id"]
    ).reset_index(drop=True)

    grouped = grid.groupby(["def_team", "zone"], sort=False)
    # Same strictly-prior discipline as build_prior_counts: the current game
    # never contributes to the rate used to predict it.
    grid["opp_def_mk"] = grouped["makes"].cumsum() - grid["makes"]
    grid["opp_def_att"] = grouped["attempts"].cumsum() - grid["attempts"]

    wide = grid.pivot_table(
        index=["game_id", "def_team"],
        columns="zone",
        values=["opp_def_mk", "opp_def_att"],
        aggfunc="first",
    )
    wide.columns = [f"{stat}_{ZONE_SUFFIX[zone]}" for stat, zone in wide.columns]
    return wide.reset_index()


def apply_opponent_defence(df: pd.DataFrame,
                           zone_priors: dict[str, BetaPrior]) -> pd.DataFrame:
    """
    Turn opponent prior counts into shrunk rates, and select the one matching
    each shot's own zone.

    Shrinks toward the same league zone priors the shooter rates use, so a team
    twelve games into a season is not credited with an elite rim defence on
    forty possessions.
    """
    out = df.copy()
    for zone in ZONES:
        suffix = ZONE_SUFFIX[zone]
        prior = zone_priors.get(zone)
        mk = out.get(f"opp_def_mk_{suffix}", 0.0)
        att = out.get(f"opp_def_att_{suffix}", 0.0)
        if prior is None:
            out[f"opp_def_rate_{suffix}"] = np.nan
        else:
            out[f"opp_def_rate_{suffix}"] = shrink(mk, att, prior)
    return out


def _load_defender_exposure(engine) -> pd.DataFrame:
    """
    Per (shot, defender who guarded that shooter that game) fractional
    credit for the shot's outcome, weighted by possession share within that
    game.

    A shot is not linked to its own contesting defender in this data — only
    "who guarded this shooter how much, this game" is known (`matchups`).
    Every shot a shooter took in a game therefore inherits the SAME defender
    weight distribution, which is the same simplification
    `build_defender_mixture` already makes and accepts; this does not make it
    worse, only reuses it for a second purpose.
    """
    weights = pd.read_sql("""
        SELECT game_id, offense_player_id, defense_player_id,
               COALESCE(NULLIF(partial_possessions, 0), matchup_minutes, 0) AS raw_weight
        FROM matchups
        WHERE COALESCE(NULLIF(partial_possessions, 0), matchup_minutes, 0) > 0
    """, engine)
    total = weights.groupby(["game_id", "offense_player_id"])["raw_weight"].transform("sum")
    weights["w"] = weights["raw_weight"] / total.replace(0, np.nan)
    weights = weights.dropna(subset=["w"])

    shots = pd.read_sql("""
        SELECT s.game_id, s.player_id AS offense_player_id, s.zone,
               s.shot_made, g.date AS game_date
        FROM shots s
        JOIN games g ON g.game_id = s.game_id
        WHERE s.zone IS NOT NULL AND s.zone != 'Backcourt'
    """, engine)
    shots["game_date"] = pd.to_datetime(shots["game_date"])
    shots["category"] = shots["zone"].map(ZONE_TO_DEF_CATEGORY)

    exposure = shots.merge(
        weights[["game_id", "offense_player_id", "defense_player_id", "w"]],
        on=["game_id", "offense_player_id"], how="inner",
    )
    return exposure


def fit_league_category_priors(engine, through_season: str) -> dict[str, BetaPrior]:
    """
    One Beta prior per defense category, pooling the zones that map to it —
    the defender-side equivalent of `shooting_rates.fit_league_zone_priors`.
    A category like "3 Pointers" spans zones with different true league rates
    (corners run cooler than above-the-break), so this refits from scratch by
    category rather than averaging the zone priors after the fact.
    """
    df = pd.read_sql(f"""
        SELECT s.zone, s.player_id, s.season,
               SUM(s.shot_made) AS makes, COUNT(*) AS attempts
        FROM shots s
        WHERE s.zone IS NOT NULL
          AND s.zone != 'Backcourt'
          AND s.season <= '{through_season}'
        GROUP BY s.zone, s.player_id, s.season
    """, engine)
    df["category"] = df["zone"].map(ZONE_TO_DEF_CATEGORY)
    priors = fit_priors(df, ["category"], makes_col="makes", attempts_col="attempts")
    return {cat: priors[(cat,)] for cat in df["category"].unique() if (cat,) in priors}


def build_defender_category_rates(engine, through_season: str) -> pd.DataFrame:
    """
    Point-in-time FG%-allowed per (defender, game, defense category) — the
    leak-free replacement for reading `defender_stats` by season.

    Same strictly-prior discipline as `shooting_rates.build_prior_counts`:
    cumulative sum then shift, so a game's own shots never contribute to that
    game's own defender features.

    Returns one row per (defense_player_id, game_id, defense_category) with
    d_fg_pct / pct_plusminus / freq, plus a pooled "Overall" row per
    (defense_player_id, game_id) — the same shape `defender_stats` provided,
    so `build.build_defender_mixture` only needs its join key changed to use
    this instead, not its downstream blending logic.
    """
    exposure = _load_defender_exposure(engine)
    exposure["w_mk"] = exposure["w"] * exposure["shot_made"]

    per_game = exposure.groupby(
        ["defense_player_id", "game_id", "game_date", "category"], as_index=False
    ).agg(w_att=("w", "sum"), w_mk=("w_mk", "sum"))

    defender_games = per_game[["defense_player_id", "game_id", "game_date"]].drop_duplicates()
    grid = defender_games.merge(pd.DataFrame({"category": DEFENSE_CATEGORIES}), how="cross")
    grid = grid.merge(
        per_game, on=["defense_player_id", "game_id", "game_date", "category"], how="left"
    )
    grid["w_att"] = grid["w_att"].fillna(0.0)
    grid["w_mk"] = grid["w_mk"].fillna(0.0)

    grid = grid.sort_values(
        ["defense_player_id", "category", "game_date", "game_id"]
    ).reset_index(drop=True)
    g = grid.groupby(["defense_player_id", "category"], sort=False)
    # Strictly prior: row i holds the total through row i-1.
    grid["pit_mk"] = g["w_mk"].cumsum() - grid["w_mk"]
    grid["pit_att"] = g["w_att"].cumsum() - grid["w_att"]

    priors = fit_league_category_priors(engine, through_season=through_season)
    pooled_mean = float(np.mean([p.mean for p in priors.values()])) if priors else 0.45
    pooled_strength = float(np.mean([p.strength for p in priors.values()])) if priors else 100.0

    # Single-level shrinkage (career-to-date only, toward the league category
    # rate) rather than the two-level career->season hierarchy shooting rates
    # use: a defender's possession-weighted evidence in one category is far
    # sparser per season than a shooter's own attempts in a zone, so a second,
    # noisier season-only layer looked more likely to overfit than to track a
    # real hot/cold defensive stretch. Worth re-measuring with `--ablate` if
    # the point-in-time feature earns its place at all.
    rate = pd.Series(np.nan, index=grid.index)
    category_mean = pd.Series(np.nan, index=grid.index)
    for cat in DEFENSE_CATEGORIES:
        mask = (grid["category"] == cat).to_numpy()
        prior = priors.get(cat, BetaPrior(mean=pooled_mean, strength=pooled_strength))
        rate.loc[mask] = shrink(grid.loc[mask, "pit_mk"], grid.loc[mask, "pit_att"], prior)
        category_mean.loc[mask] = prior.mean
    grid["d_fg_pct"] = rate
    grid["pct_plusminus"] = grid["d_fg_pct"] - category_mean

    total_att = grid.groupby(["defense_player_id", "game_id"])["pit_att"].transform("sum")
    grid["freq"] = grid["pit_att"] / total_att.replace(0, np.nan)

    long_table = grid[
        ["defense_player_id", "game_id", "category", "d_fg_pct", "pct_plusminus", "freq"]
    ].rename(columns={"category": "defense_category"})

    # Pooled "Overall" row, matching the shape `defender_stats` already had —
    # `build_defender_mixture` filters `defense_category == "Overall"` for it.
    overall_prior = BetaPrior(mean=pooled_mean, strength=pooled_strength)
    overall = grid.groupby(["defense_player_id", "game_id"], as_index=False).agg(
        pit_mk=("pit_mk", "sum"), pit_att=("pit_att", "sum")
    )
    overall["d_fg_pct"] = shrink(overall["pit_mk"], overall["pit_att"], overall_prior)
    overall["pct_plusminus"] = overall["d_fg_pct"] - pooled_mean
    overall["freq"] = 1.0
    overall["defense_category"] = "Overall"
    overall = overall[
        ["defense_player_id", "game_id", "defense_category", "d_fg_pct", "pct_plusminus", "freq"]
    ]

    return pd.concat([long_table, overall], ignore_index=True)


def lookup_defender_category_rates(
    conn, defender_id: str, category_priors: dict[str, BetaPrior], as_of_date=None
) -> dict:
    """
    Serving-path equivalent of `build_defender_category_rates` for a single
    named defender — the parity guarantee that keeps the recommender from
    reading the leaky `defender_stats` table the way it used to.

    `category_priors` must be the SAME fitted priors the model was trained
    against (loaded from run metadata, same as `zone_priors` is), not
    refitted from whatever is in the database at request time — the same
    discipline `apply_hierarchy`'s `zone_priors` argument follows.

    Returns `{"by_category": {category: {"d_fg_pct", "pct_plusminus", "freq"}}}`,
    matching the shape `_defender_row` in `recommender.py` already expects
    from `defender_stats`, so only the data source changes there, not the
    blending logic downstream of it.
    """
    from sqlalchemy import text

    date_clause = "AND g.date < :as_of" if as_of_date is not None else ""
    params = {"pid": str(defender_id)}
    if as_of_date is not None:
        params["as_of"] = str(as_of_date)

    rows = conn.execute(text(f"""
        SELECT s.zone, SUM(s.shot_made) AS makes, COUNT(*) AS attempts
        FROM matchups m
        JOIN shots s ON s.game_id = m.game_id AND s.player_id = m.offense_player_id
        JOIN games g ON g.game_id = m.game_id
        WHERE m.defense_player_id = :pid
          AND COALESCE(NULLIF(m.partial_possessions, 0), m.matchup_minutes, 0) > 0
          AND s.zone IS NOT NULL AND s.zone != 'Backcourt'
          {date_clause}
        GROUP BY s.zone
    """), params).fetchall()

    by_cat: dict[str, dict] = {cat: {"mk": 0.0, "att": 0.0} for cat in DEFENSE_CATEGORIES}
    for zone, makes, attempts in rows:
        cat = ZONE_TO_DEF_CATEGORY.get(zone)
        if cat is None:
            continue
        by_cat[cat]["mk"] += float(makes or 0)
        by_cat[cat]["att"] += float(attempts or 0)

    priors = category_priors
    if not priors:
        return {"by_category": {}}

    pooled_mean = float(np.mean([p.mean for p in priors.values()]))
    pooled_strength = float(np.mean([p.strength for p in priors.values()]))

    def _to_native(value) -> float | None:
        # `shrink()` returns a numpy scalar even for plain-float inputs, which
        # FastAPI's encoder cannot always serialize on its own. A 0/0 divide
        # (no exposure and, in tests, a strength=0 prior) yields NaN, which
        # means the same thing None does here — no basis for a rate — so both
        # collapse to the one JSON-safe representation of "unknown".
        f = float(value)
        return None if np.isnan(f) else f

    result = {}
    total_mk = total_att = 0.0
    for cat, counts in by_cat.items():
        prior = priors.get(cat, BetaPrior(mean=pooled_mean, strength=pooled_strength))
        rate = _to_native(shrink(counts["mk"], counts["att"], prior))
        result[cat] = {
            "d_fg_pct": rate,
            "pct_plusminus": (rate - prior.mean) if rate is not None else None,
            "freq": None,  # filled in below once the total is known
        }
        total_mk += counts["mk"]
        total_att += counts["att"]

    for cat, counts in by_cat.items():
        result[cat]["freq"] = (counts["att"] / total_att) if total_att > 0 else None

    overall_prior = BetaPrior(mean=pooled_mean, strength=pooled_strength)
    overall_rate = _to_native(shrink(total_mk, total_att, overall_prior))
    result["Overall"] = {
        "d_fg_pct": overall_rate,
        "pct_plusminus": (overall_rate - pooled_mean) if overall_rate is not None else None,
        "freq": 1.0,
    }
    return {"by_category": result}
