"""
Training matrix assembly.

Replaces the single 150-line SQL string in the old
`src/training/feature_engineering.py`, where ten LEFT JOINs and the feature
list were interleaved in one unreadable, untestable block. Here each input is
loaded by a small named function that can be exercised on its own, and the
only place features are actually computed is `spec.derive_features` — the
same function the recommender calls.

Two substantive changes to what the matrix contains:

  1. Shooting rates are point-in-time and shrunk (see `point_in_time`), not
     whole-season aggregates. This removes the label leak and makes the
     features reproducible at prediction time.

  2. Defender features come from a possession-weighted MIXTURE over everyone
     who guarded the shooter, not from one hard-assigned defender per game.
     See `build_defender_mixture` for why the old approach was wrong.

DuckDB is used for the heavy scans. The workload — a handful of joins and
group-bys over three and a half million rows, read once — is exactly what a
columnar engine is for, and it reads the existing SQLite file in place, so
nothing about the storage layout has to change.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src.features import creation as creation_mod
from src.features.point_in_time import (
    ZONES,
    apply_hierarchy,
    apply_opponent_defence,
    build_contest_history,
    build_defender_category_rates,
    build_opponent_zone_defence,
    build_prior_counts,
    build_rolling_form,
    fit_league_category_priors,
    fit_league_zone_priors,
)
from src.features.spec import ZONE_TO_DEF_CATEGORY, derive_features

TARGET_COL = "shot_made"

# Play-by-play context is only used once it covers most of the training window.
# See the note where it is joined.
MIN_CONTEXT_COVERAGE = 0.90


def _duckdb_connection():
    """
    A DuckDB connection with the project's SQLite file attached read-only.

    Read-only matters: the API process and the ingestors may hold the same
    file open, and a training run has no business taking a write lock on it.
    """
    import duckdb

    con = duckdb.connect()
    con.execute("INSTALL sqlite; LOAD sqlite;")
    con.execute(f"ATTACH '{config.DB_PATH}' AS nba (TYPE sqlite, READ_ONLY);")
    return con


def load_base_shots(seasons: list[str], con=None) -> pd.DataFrame:
    """
    Shots joined to game context, shooter physicals, schedule rest, and
    opponent defensive rating.

    Notably absent: any season-aggregate shooting statistic. Those were the
    leak, and they are now supplied by `point_in_time` from strictly prior
    games instead.
    """
    close = con is None
    con = con or _duckdb_connection()
    season_list = ", ".join(f"'{s}'" for s in seasons)

    try:
        df = con.execute(f"""
            SELECT
                s.shot_id, s.season, s.player_id, s.game_id, s.shot_made,
                s.loc_x, s.loc_y, s.shot_distance, s.shot_type, s.zone,
                s.quarter, s.time_remaining, s.score_diff,
                s.home_away, s.playoff_flag,
                g.date AS game_date,
                p.height, p.weight, p.wingspan, p.position,
                CASE WHEN s.home_away = 1 THEN ts_home.rest_days
                     ELSE ts_away.rest_days END AS rest_days,
                CASE WHEN s.home_away = 1 THEN ts_home.is_back_to_back
                     ELSE ts_away.is_back_to_back END AS is_back_to_back,
                CASE WHEN s.home_away = 1 THEN opp_away.def_rating
                     ELSE opp_home.def_rating END AS opp_def_rating,
                -- The defending side, for point-in-time opponent zone defence.
                -- Derivable with no new data from the game's two teams and
                -- which side the shooter was on.
                CASE WHEN s.home_away = 1 THEN g.away_team ELSE g.home_team END
                    AS def_team
            FROM nba.shots s
            JOIN nba.games g   ON s.game_id = g.game_id
            JOIN nba.players p ON s.player_id = p.player_id AND s.season = p.season
            LEFT JOIN nba.team_schedule ts_home
                   ON g.game_id = ts_home.game_id AND g.home_team = ts_home.team_id
            LEFT JOIN nba.team_schedule ts_away
                   ON g.game_id = ts_away.game_id AND g.away_team = ts_away.team_id
            LEFT JOIN nba.team_stats opp_away
                   ON opp_away.team_abbrev = g.away_team AND opp_away.season = s.season
            LEFT JOIN nba.team_stats opp_home
                   ON opp_home.team_abbrev = g.home_team AND opp_home.season = s.season
            WHERE s.score_diff IS NOT NULL
              AND s.home_away IS NOT NULL
              AND s.zone IS NOT NULL
              AND s.zone != 'Backcourt'
              AND s.season IN ({season_list})
            -- Chronological order is load-bearing: the hyperparameter search
            -- slices this frame by row position to build temporal folds, and
            -- without an explicit ORDER BY those "temporal" folds were
            -- whatever order the engine happened to emit.
            ORDER BY g.date, s.game_id, s.shot_id
        """).df()
    finally:
        if close:
            con.close()

    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


def build_defender_mixture(
    seasons: list[str], con=None, defender_category_rates: pd.DataFrame | None = None
) -> pd.DataFrame:
    """
    Possession-weighted defender aggregates per (game, shooter, defense category).

    Why the old approach was wrong
    ------------------------------
    `defender_linker.py` assigned each shot the single defender with the most
    matchup minutes across the ENTIRE game, so all eighteen of a player's
    shots on a given night carried the same defender — including transition
    layups contested by somebody else entirely. It also discarded the other
    two million rows of matchup detail by taking an argmax over them.

    What this does instead
    ----------------------
    Every defender who guarded the shooter contributes in proportion to the
    possessions he spent doing it. The result is an expected defender rather
    than a guessed one, which is both more accurate on average and honest
    about the uncertainty: `def_matchup_dispersion` reports how spread out the
    assignment was, letting the model discount defender features exactly when
    the shooter was guarded by a committee.

    What this still does not fix
    ----------------------------
    On a shot at the rim the relevant defender is frequently the help-side
    big, not whoever was nominally guarding the ball. Correcting that needs
    the five defenders on the floor, which needs lineup reconstruction from
    play-by-play. This mixture is a strict improvement over an argmax, not a
    solution to that problem.

    Where the quality numbers come from
    ------------------------------------
    `defender_category_rates` supplies `d_fg_pct` / `pct_plusminus` / `freq`
    per (defender, game, category) — pass the output of
    `point_in_time.build_defender_category_rates`, computed point-in-time
    from `matchups` + `shots`. This used to read `defender_stats`
    (LeagueDashPtDefend) directly, joined by season alone; that table is a
    whole-season aggregate with no as-of-date version, so a shot's own
    outcome was inside the feature describing it — the same leak
    `point_in_time.py` was built to remove for shooters, just never applied
    to defenders. `defender_category_rates=None` falls back to all-NaN
    quality columns (physicals and matchup concentration still populate).
    """
    close = con is None
    con = con or _duckdb_connection()
    season_list = ", ".join(f"'{s}'" for s in seasons)

    try:
        # Weight defenders by partial possessions within each (game, shooter).
        # Falling back to matchup_minutes keeps rows where the possession
        # count is missing but the time is recorded.
        weights = con.execute(f"""
            SELECT m.game_id,
                   m.offense_player_id,
                   m.defense_player_id,
                   COALESCE(NULLIF(m.partial_possessions, 0),
                            m.matchup_minutes, 0) AS raw_weight
            FROM nba.matchups m
            WHERE COALESCE(NULLIF(m.partial_possessions, 0),
                           m.matchup_minutes, 0) > 0
              AND m.game_id IN (
                    SELECT DISTINCT game_id FROM nba.games
                    WHERE game_id IN (SELECT DISTINCT game_id FROM nba.shots
                                      WHERE season IN ({season_list}))
              )
        """).df()

        defenders = con.execute("""
            SELECT player_id, season, height, weight, wingspan
            FROM nba.players
        """).df()

        game_seasons = con.execute(f"""
            SELECT DISTINCT game_id, season FROM nba.shots
            WHERE season IN ({season_list})
        """).df()
    finally:
        if close:
            con.close()

    if weights.empty:
        return pd.DataFrame(columns=["game_id", "player_id", "defense_category"])

    weights = weights.merge(game_seasons, on="game_id", how="inner")

    total = weights.groupby(["game_id", "offense_player_id"])["raw_weight"].transform("sum")
    weights["w"] = weights["raw_weight"] / total.replace(0, np.nan)

    # Assignment concentration, computed before the category fan-out so it is
    # one number per (game, shooter) rather than six copies of it.
    concentration = weights.groupby(["game_id", "offense_player_id"]).agg(
        def_matchup_share=("w", "max"),
        _hhi=("w", lambda s: float(np.sum(s.values ** 2))),
    ).reset_index()
    # 1 - HHI: 0 when one defender took every possession, approaching 1 as the
    # assignment splits across more defenders.
    concentration["def_matchup_dispersion"] = 1.0 - concentration["_hhi"]
    concentration = concentration.drop(columns=["_hhi"])

    # ── Weighted physicals ───────────────────────────────────────────────
    phys = weights.merge(
        defenders,
        left_on=["defense_player_id", "season"],
        right_on=["player_id", "season"],
        how="left",
    )
    for col in ("height", "weight", "wingspan"):
        phys[f"_w_{col}"] = phys[col] * phys["w"]
        # Renormalize against the weight that actually had a measurement, so a
        # defender with no recorded wingspan does not drag the average toward
        # zero — he simply does not vote.
        phys[f"_wv_{col}"] = phys["w"] * phys[col].notna()

    phys_agg = phys.groupby(["game_id", "offense_player_id"]).agg(
        **{f"_s_{c}": (f"_w_{c}", "sum") for c in ("height", "weight", "wingspan")},
        **{f"_v_{c}": (f"_wv_{c}", "sum") for c in ("height", "weight", "wingspan")},
    ).reset_index()

    for col in ("height", "weight", "wingspan"):
        phys_agg[f"def_{col}"] = (
            phys_agg[f"_s_{col}"] / phys_agg[f"_v_{col}"].replace(0, np.nan)
        )
    phys_agg = phys_agg[["game_id", "offense_player_id",
                         "def_height", "def_weight", "def_wingspan"]]

    # ── Weighted defensive quality, per category (point-in-time) ─────────
    if defender_category_rates is None or defender_category_rates.empty:
        def_stats = pd.DataFrame(columns=[
            "defense_player_id", "game_id", "defense_category",
            "d_fg_pct", "pct_plusminus", "freq",
        ])
    else:
        def_stats = defender_category_rates

    # Keyed on (defender, game) rather than (defender, season) — the whole
    # point of the point-in-time rebuild is that quality no longer needs a
    # season bucket to be looked up, it is already specific to this game.
    qual = weights.merge(
        def_stats,
        on=["defense_player_id", "game_id"],
        how="inner",
    )
    for col in ("d_fg_pct", "pct_plusminus", "freq"):
        qual[f"_w_{col}"] = qual[col] * qual["w"]
        qual[f"_wv_{col}"] = qual["w"] * qual[col].notna()

    qual_agg = qual.groupby(["game_id", "offense_player_id", "defense_category"]).agg(
        **{f"_s_{c}": (f"_w_{c}", "sum") for c in ("d_fg_pct", "pct_plusminus", "freq")},
        **{f"_v_{c}": (f"_wv_{c}", "sum") for c in ("d_fg_pct", "pct_plusminus", "freq")},
    ).reset_index()

    rename = {"d_fg_pct": "def_fg_pct", "pct_plusminus": "def_pct_plusminus",
              "freq": "def_freq"}
    for col, out_name in rename.items():
        qual_agg[out_name] = (
            qual_agg[f"_s_{col}"] / qual_agg[f"_v_{col}"].replace(0, np.nan)
        )
    qual_agg = qual_agg[["game_id", "offense_player_id", "defense_category",
                         "def_fg_pct", "def_pct_plusminus", "def_freq"]]

    out = qual_agg.merge(phys_agg, on=["game_id", "offense_player_id"], how="left")
    out = out.merge(concentration, on=["game_id", "offense_player_id"], how="left")
    return out.rename(columns={"offense_player_id": "player_id"})


def _attach_defender_features(df: pd.DataFrame, mixture: pd.DataFrame) -> pd.DataFrame:
    """
    Join the defender mixture onto shots twice: once for the zone-appropriate
    defense category, once for 'Overall'.
    """
    if mixture.empty:
        for col in ("def_height", "def_weight", "def_wingspan",
                    "def_fg_pct_zone", "def_pct_plusminus_zone", "def_freq_zone",
                    "def_fg_pct_overall", "def_pct_plusminus",
                    "def_matchup_share", "def_matchup_dispersion"):
            df[col] = np.nan
        return df

    out = df.copy()
    out["_def_category"] = out["zone"].map(ZONE_TO_DEF_CATEGORY)

    zone_side = mixture.rename(columns={
        "defense_category": "_def_category",
        "def_fg_pct": "def_fg_pct_zone",
        "def_pct_plusminus": "def_pct_plusminus_zone",
        "def_freq": "def_freq_zone",
    })
    out = out.merge(
        zone_side[["game_id", "player_id", "_def_category",
                   "def_fg_pct_zone", "def_pct_plusminus_zone", "def_freq_zone",
                   "def_height", "def_weight", "def_wingspan",
                   "def_matchup_share", "def_matchup_dispersion"]],
        on=["game_id", "player_id", "_def_category"],
        how="left",
    )

    overall = mixture[mixture["defense_category"] == "Overall"][
        ["game_id", "player_id", "def_fg_pct", "def_pct_plusminus"]
    ].rename(columns={
        "def_fg_pct": "def_fg_pct_overall",
        "def_pct_plusminus": "def_pct_plusminus",
    })
    out = out.merge(overall, on=["game_id", "player_id"], how="left")

    return out.drop(columns=["_def_category"])


def build_matrix(
    seasons: list[str],
    prior_through_season: str | None = None,
    use_defender: bool = True,
    verbose: bool = True,
) -> tuple[pd.DataFrame, list[str], dict]:
    """
    Build the full training matrix.

    `prior_through_season` bounds the seasons used to fit the league Beta
    priors and the league zone baselines. It must be the last TRAINING season,
    never the test season — a prior fit on the data the model is about to be
    scored against is the same leak this rewrite removed, laundered through an
    aggregate. Defaults to the second-to-last season in `seasons`, which is the
    correct choice for the standard split.

    Returns (matrix, feature_columns, artifacts). `artifacts` carries the
    fitted priors and league baselines so the serving path can reuse the exact
    objects the model was trained against, rather than refitting them from
    whatever data happens to be in the database at inference time.
    """
    from src.db.database import get_engine

    engine = get_engine()
    con = _duckdb_connection()

    if prior_through_season is None:
        prior_through_season = seasons[-2] if len(seasons) > 1 else seasons[-1]

    def log(msg):
        if verbose:
            print(msg)

    log(f"\n{'='*62}")
    log("  FEATURE BUILD")
    log(f"  Seasons: {seasons[0]} → {seasons[-1]} ({len(seasons)})")
    log(f"  Priors fit through: {prior_through_season}")
    log(f"{'='*62}")

    defender_category_rates = None
    category_priors: dict = {}
    if use_defender:
        log("  → defender category rates (point-in-time) ...")
        defender_category_rates = build_defender_category_rates(
            engine, through_season=prior_through_season
        )
        # Threaded through `artifacts` below, same as `zone_priors` — the
        # recommender must shrink toward the SAME fitted priors the model was
        # trained against, not ones refit from whatever is in the database at
        # request time.
        category_priors = fit_league_category_priors(
            engine, through_season=prior_through_season
        )
        log(f"    {len(defender_category_rates):,} (defender, game, category) rows")

    try:
        log("  → shots + context ...")
        df = load_base_shots(seasons, con=con)
        log(f"    {len(df):,} shots")

        if use_defender:
            log("  → defender mixture (possession-weighted) ...")
            mixture = build_defender_mixture(
                seasons, con=con, defender_category_rates=defender_category_rates
            )
            log(f"    {len(mixture):,} (game, shooter, category) aggregates")
            df = _attach_defender_features(df, mixture)
    finally:
        con.close()

    log("  → point-in-time shooting counts ...")
    prior_counts = build_prior_counts(engine)
    df = df.merge(prior_counts, on=["player_id", "game_id"], how="left")

    log("  → league zone priors (empirical Bayes) ...")
    zone_priors = fit_league_zone_priors(engine, through_season=prior_through_season)
    if verbose:
        for zone in ZONES:
            p = zone_priors.get(zone)
            if p:
                log(f"    {zone:<24} mean={p.mean:.3f}  k={p.strength:>7.1f}  "
                    f"(n={p.n_players} players)")

    df = apply_hierarchy(df, zone_priors)

    log("  → opponent zone defence (point-in-time) ...")
    opp = build_opponent_zone_defence(engine)
    if not opp.empty:
        df = df.merge(opp, on=["game_id", "def_team"], how="left")
        df = apply_opponent_defence(df, zone_priors)
        log(f"    {len(opp):,} (game, defending team) rows")

    log("  → per-game contest history (point-in-time) ...")
    contest = build_contest_history(engine)
    if not contest.empty:
        df = df.merge(contest, on=["player_id", "game_id"], how="left")
        log(f"    {len(contest):,} (player, game) contest rows")
    else:
        log("    none ingested yet — skipping")

    log("  → rolling form ...")
    df = df.merge(build_rolling_form(engine), on=["player_id", "game_id"], how="left")

    log("  → play-by-play shot context ...")
    # Escape hatch for the A/B: with PBP_DISABLE set, the context block is
    # skipped entirely, so an otherwise-identical run measures exactly what the
    # play-by-play features are worth.
    import os
    if os.environ.get("PBP_DISABLE"):
        log("    disabled via PBP_DISABLE")
        context = pd.DataFrame()
    else:
        context = pd.read_sql("""
            SELECT shot_id, shot_subtype, is_putback,
                   seconds_since_prev_event, prev_event_type
            FROM shot_context
        """, engine)
    if not context.empty:
        df = df.merge(context, on="shot_id", how="left")
        coverage = df["shot_subtype"].notna().mean()
        log(f"    {len(context):,} rows; covers {coverage:.1%} of shots")
        if coverage < MIN_CONTEXT_COVERAGE:
            # Partial coverage is worse than none. The ingest walks games in id
            # order, so an incomplete run covers the OLDEST seasons only —
            # "context is missing" would then be a near-perfect proxy for
            # "recent season", and the model would learn the proxy rather than
            # the mechanics. Features are dropped until the backfill is far
            # enough along for absence to be uninformative.
            log(f"    ⚠ below {MIN_CONTEXT_COVERAGE:.0%} — dropping context "
                f"features (partial coverage correlates with season)")
            df = df.drop(columns=["shot_subtype", "is_putback",
                                  "seconds_since_prev_event",
                                  "prev_event_type"])
    else:
        log("    none ingested yet — skipping")

    log("  → creation profiles (lagged one season) ...")
    profiles = creation_mod.load_creation_profiles(engine)
    positions = pd.read_sql("SELECT player_id, season, position FROM players", engine)
    df = creation_mod.attach_creation_features(df, profiles, positions=positions)
    if verbose and "creation_is_prior" in df.columns:
        share = df["creation_is_prior"].mean()
        log(f"    {len(profiles):,} profiles; {share*100:.1f}% of shots on a "
            f"position-bucket fallback")

    log("  → deriving features ...")
    league_zone_rates = {z: p.mean for z, p in zone_priors.items()}
    df = derive_features(df, league_zone_rates=league_zone_rates)

    from src.features.spec import all_feature_columns
    feature_cols = all_feature_columns(df)

    artifacts = {
        "zone_priors": zone_priors,
        "league_zone_rates": league_zone_rates,
        "category_priors": category_priors,
        "prior_through_season": prior_through_season,
        "feature_cols": feature_cols,
    }

    if verbose:
        nulls = {c: int(df[c].isna().sum()) for c in feature_cols
                 if df[c].isna().any()}
        log(f"\n  ✓ {len(df):,} rows × {len(feature_cols)} features")
        log(f"  ✓ target mean: {df[TARGET_COL].mean():.4f}")
        if nulls:
            log("\n  Features with missing values (XGBoost handles natively):")
            for col, n in sorted(nulls.items(), key=lambda kv: -kv[1])[:15]:
                log(f"    {col:<34} {n:>9,} ({n/len(df)*100:5.1f}%)")
        log(f"{'='*62}\n")

    return df, feature_cols, artifacts


if __name__ == "__main__":
    seasons = [s for s in config.ALL_SEASONS if s >= "2016-17"]
    matrix, cols, artifacts = build_matrix(seasons)
    print(f"matrix={matrix.shape} features={len(cols)}")
