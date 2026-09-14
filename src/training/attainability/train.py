"""
Attainability — can this player actually GET this shot?

The problem
-----------
The shot-quality model estimates P(make | a shot was taken from here). The
recommender was reading that as P(make | you go take a shot here). Those are
different quantities, and the gap between them is selection: the shots that
exist at a location are the ones somebody chose to take, usually because they
were open enough to be worth taking. A player who only shoots from the elbow
when he is wide open posts a fine elbow percentage, and "shoot more from the
elbow" does not inherit the wide-open part.

The practical symptom is that ranking purely by expected points recommends the
restricted area to everyone, always. That is true, useless, and exactly what a
model that ignores attainability must conclude — the rim is the highest-value
spot on the floor for every player alive, and the entire difficulty is getting
there.

What this model estimates
-------------------------
Given a player and a location, what share of that player's shot diet would
plausibly come from there — i.e. how readily can they generate this look? It
is fit on observed shot-frequency distributions: for each (player, season,
zone), the fraction of their attempts taken in that zone, with the same
empirical-Bayes shrinkage the shooting rates get, so a player with 40 attempts
does not register as a 100%-corner-three specialist.

Why creation skill is the backbone
----------------------------------
Attainability is mostly a handle question, which is where the creation
features earn their place. Whether a player can generate an above-the-break
three off the dribble is not really about his shooting — it is about whether
he can create the separation. `self_creation_index`, `pullup_share`,
`avg_drib_per_touch` and `drives_per_min` are the features that carry it, and
they are the reason an elite creator and a standstill shooter with identical
shooting percentages receive different advice.

How the recommender uses it
---------------------------
Not as a hard filter. A rim-running center's attainability for a corner three
is low but not zero, and zeroing it out would hide a real, if rare, option.
The recommender ranks by expected points weighted by attainability and reports
both numbers, so the interface can distinguish "this would be a great shot for
you" from "this is a shot you can actually get".

Usage:
    python -m src.training.attainability --name attainability-v1
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import config
from src.common import runs
from src.features.point_in_time import ZONES, fit_league_creation_priors
from .build import (
    _league_sub_zone_priors,
    build_attainability_matrix,
    build_season_cast,
)
from .features import (
    ATTAINABILITY_FEATURE_COLS,
    CAST_FEATURE_COLS,
    DIET_SNAPSHOTS,
    POSITION_BUCKETS,
    PRIOR_DIET_COLS,
    SUB_ZONES,
    attach_sub_zone,
    with_bare_zone_entries,
)

MODEL_DIR = Path(config.PROJECT_ROOT) / "models"


def train(seasons: list[str] | None = None, name: str = "attainability",
          use_cast: bool = True, use_prior_diet: bool = True) -> dict:
    """
    Fit the attainability model.

    Regression on the shrunk share with a squared-error objective rather than
    a classifier: the target is a proportion in [0, 1], not an event. Trained
    per (player, season, zone) row — roughly three thousand rows a season, so
    the model is deliberately small and heavily regularized.
    """
    from src.db.database import get_engine

    seasons = seasons or [s for s in config.ALL_SEASONS if s >= "2014-15"]
    engine = get_engine()

    run = runs.start_run(f"{name}", seasons=seasons)

    print(f"\n{'='*62}")
    print("  ATTAINABILITY MODEL")
    print("  Estimating: what share of a player's shots come from each zone")
    print(f"{'='*62}")

    df = build_attainability_matrix(engine, seasons)
    print(f"  {len(df):,} (player, season, zone) rows")

    test_season = seasons[-1]
    val_season = seasons[-2]
    fit_df = df[~df["season"].isin([val_season, test_season])]
    val_df = df[df["season"] == val_season]
    test_df = df[df["season"] == test_season]

    candidate_cols = list(ATTAINABILITY_FEATURE_COLS)
    if use_cast:
        candidate_cols += CAST_FEATURE_COLS
    if use_prior_diet:
        candidate_cols += PRIOR_DIET_COLS
    feature_cols = [c for c in candidate_cols if c in df.columns]
    print(f"  fit {len(fit_df):,} / val {len(val_df):,} / test {len(test_df):,}")
    ablated = [lbl for lbl, on in
               (("supporting cast", use_cast), ("prior diet", use_prior_diet))
               if not on]
    print(f"  features: {len(feature_cols)}"
          f"{'  (ablated: ' + ', '.join(ablated) + ')' if ablated else ''}")

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=600,
        max_depth=5,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=20,
        reg_lambda=2.0,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=40,
    )
    model.fit(
        fit_df[feature_cols], fit_df["zone_share"],
        eval_set=[(val_df[feature_cols], val_df["zone_share"])],
        verbose=False,
    )

    preds = np.clip(model.predict(test_df[feature_cols]), 0.0, 1.0)
    actual = test_df["zone_share"].values

    mae = float(np.mean(np.abs(preds - actual)))
    rmse = float(np.sqrt(np.mean((preds - actual) ** 2)))

    # The baseline that matters: predict each zone's league-average share and
    # ignore the player entirely. If the model cannot beat that, it has learned
    # nothing about individual players' shot diets and the whole exercise is
    # just a lookup table of zone frequencies.
    zone_means = fit_df.groupby("sub_zone")["zone_share"].mean()
    zone_means = with_bare_zone_entries(zone_means)
    base_preds = test_df["sub_zone"].map(zone_means).values
    base_mae = float(np.mean(np.abs(base_preds - actual)))

    print(f"\n  MAE           {mae:.4f}")
    print(f"  RMSE          {rmse:.4f}")
    print(f"  league-avg MAE {base_mae:.4f}")
    print(f"  improvement   {(base_mae - mae) / base_mae:+.1%}")

    importances = sorted(
        zip(feature_cols, model.feature_importances_),
        key=lambda kv: -kv[1],
    )
    print("\n  Top features (what determines which shots you can get):")
    for feature, importance in importances[:12]:
        print(f"    {feature:<32} {importance:.4f}")

    # League distribution of every model input, for explanations. A SHAP
    # contribution says a feature pushed the estimate down; it cannot say
    # whether the underlying value was unusual. Percentiles turn "pullup_share
    # = 0.05 lowered this" into "he pulls up on 5% of touches, 12th percentile
    # league-wide" — which is the half a reader can actually act on. Computed
    # on the fit window only, so the reference distribution never includes the
    # held-out season.
    # Quantiles are stored as a grid rather than a few landmarks so the
    # explainer can interpolate an approximate percentile for any value
    # without shipping the full training distribution alongside the model.
    quantile_grid = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    reference = {}
    for col in feature_cols:
        values = fit_df[col].dropna()
        if values.empty:
            continue
        reference[col] = {
            "mean": float(values.mean()),
            "quantiles": {str(q): float(values.quantile(q)) for q in quantile_grid},
        }

    model.save_model(str(MODEL_DIR / f"xgb_{name}.json"))
    metadata = {
        "name": name,
        "feature_cols": feature_cols,
        "zones": ZONES,
        "sub_zones": SUB_ZONES,
        "position_buckets": POSITION_BUCKETS,
        "league_zone_shares": {k: float(v) for k, v in zone_means.items()},
        "feature_reference": reference,
        # Per-zone priors over "did the player create this himself". Not a
        # model input — attainability's target is frequency, and this is the
        # separate creation question the explanation reports alongside it.
        # Travels with the model so serving shrinks toward the same priors,
        # fit on the fit window only.
        # League sub-zone priors, so the serving path shrinks a partial season
        # toward exactly what training shrank toward.
        "sub_zone_priors": {
            z: {"mean": p.mean, "strength": p.strength}
            for z, p in _league_sub_zone_priors(
                attach_sub_zone(pd.read_sql(
                    "SELECT player_id, season, zone, loc_x, loc_y FROM shots "
                    "WHERE zone IS NOT NULL AND zone != 'Backcourt'", engine))
            ).items()
        },
        "max_season_progress": max(DIET_SNAPSHOTS),
        "creation_priors": {
            zone: {"mean": p.mean, "strength": p.strength,
                   "n_players": p.n_players, "n_attempts": p.n_attempts}
            for zone, p in fit_league_creation_priors(
                engine, through_season=fit_df["season"].max()
            ).items()
        },
        "test_season": test_season,
        "metrics": {"mae": mae, "rmse": rmse, "baseline_mae": base_mae},
    }
    (MODEL_DIR / f"metadata_{name}.json").write_text(
        json.dumps(metadata, indent=2, default=str)
    )

    run.log_metrics(mae=mae, rmse=rmse, baseline_mae=base_mae,
                    improvement=(base_mae - mae) / base_mae)
    run.log_data(n_rows=len(df), n_features=len(feature_cols))
    run.finish()

    print(f"\n  ✓ saved models/xgb_{name}.json")
    print(f"{'='*62}\n")
    return metadata
