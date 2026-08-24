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

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from src.common import runs
from src.features.creation import CREATION_FEATURE_COLS
from src.features.point_in_time import ZONES
from src.features.shrinkage import fit_beta_prior

MODEL_DIR = Path(config.PROJECT_ROOT) / "models"

# Features the attainability model is allowed to see. Deliberately excludes
# everything about whether the shot GOES IN — this model answers a question
# about shot generation, and letting it peek at shooting skill would blur the
# two quantities the recommender needs kept separate.
ATTAINABILITY_FEATURE_COLS = [
    # Who the player is, physically
    "height", "weight", "wingspan",
    # How they get their shots — the substance of the model
    *CREATION_FEATURE_COLS,
    # Where the shot is
    "zone_index", "is_three",
]


def build_zone_frequency_targets(engine, min_attempts: int = 50) -> pd.DataFrame:
    """
    Observed shot-diet shares per (player, season, zone), shrunk.

    Raw shares are noisy for low-volume players in exactly the way raw
    shooting percentages are, and for the same reason: a player with 60 total
    attempts who happened to take four corner threes reads as a 6.7% corner-
    three shooter with no evidence behind it. Each zone's share is shrunk
    toward the league distribution using a Beta prior fit per zone.

    Returns one row per (player_id, season, zone) with `zone_share`.
    """
    counts = pd.read_sql("""
        SELECT s.player_id, s.season, s.zone, COUNT(*) AS attempts
        FROM shots s
        WHERE s.zone IS NOT NULL AND s.zone != 'Backcourt'
        GROUP BY s.player_id, s.season, s.zone
    """, engine)

    totals = counts.groupby(["player_id", "season"])["attempts"].sum().rename("total")
    counts = counts.merge(totals, on=["player_id", "season"])
    counts = counts[counts["total"] >= min_attempts]

    # Complete the grid: a player who took zero mid-range shots has a
    # meaningful zero, and dropping the row would leave the model to infer
    # absence from missingness.
    grid = (
        counts[["player_id", "season", "total"]].drop_duplicates()
        .merge(pd.DataFrame({"zone": ZONES}), how="cross")
    )
    grid = grid.merge(
        counts[["player_id", "season", "zone", "attempts"]],
        on=["player_id", "season", "zone"], how="left",
    )
    grid["attempts"] = grid["attempts"].fillna(0.0)

    out = []
    for zone, group in grid.groupby("zone"):
        prior = fit_beta_prior(group["attempts"].values, group["total"].values)
        g = group.copy()
        g["zone_share"] = (
            (g["attempts"] + prior.alpha) / (g["total"] + prior.strength)
        )
        out.append(g)

    result = pd.concat(out, ignore_index=True)

    # Renormalize so a player's six shrunk shares sum to one. Shrinking each
    # zone independently does not preserve the simplex, and a "share" vector
    # summing to 1.04 would quietly inflate every expected-points figure the
    # recommender derives from it.
    share_sum = result.groupby(["player_id", "season"])["zone_share"].transform("sum")
    result["zone_share"] = result["zone_share"] / share_sum

    return result[["player_id", "season", "zone", "attempts", "total", "zone_share"]]


def build_attainability_matrix(engine, seasons: list[str]) -> pd.DataFrame:
    """
    Join shrunk zone shares to lagged creation profiles and physicals.

    Creation features are lagged exactly as they are for the shot-quality
    model — the profile from season S-1 predicts the shot diet in season S.
    That is what makes this usable prospectively: it answers "given how this
    player creates, what shots will he be able to get", not "given the shots
    he took, what shots did he take".
    """
    from src.features.creation import attach_creation_features, load_creation_profiles

    targets = build_zone_frequency_targets(engine)
    targets = targets[targets["season"].isin(seasons)]

    players = pd.read_sql(
        "SELECT player_id, season, height, weight, wingspan, position FROM players",
        engine,
    )
    df = targets.merge(players, on=["player_id", "season"], how="left")

    profiles = load_creation_profiles(engine)
    df = attach_creation_features(df, profiles, positions=players)

    df["zone_index"] = df["zone"].map({z: i for i, z in enumerate(ZONES)})
    df["is_three"] = df["zone"].isin(
        ["Left Corner 3", "Right Corner 3", "Above the Break 3"]
    ).astype(int)

    return df


def train(seasons: list[str] | None = None, name: str = "attainability") -> dict:
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

    feature_cols = [c for c in ATTAINABILITY_FEATURE_COLS if c in df.columns]
    print(f"  fit {len(fit_df):,} / val {len(val_df):,} / test {len(test_df):,}")
    print(f"  features: {len(feature_cols)}")

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
    zone_means = fit_df.groupby("zone")["zone_share"].mean()
    base_preds = test_df["zone"].map(zone_means).values
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

    model.save_model(str(MODEL_DIR / f"xgb_{name}.json"))
    metadata = {
        "name": name,
        "feature_cols": feature_cols,
        "zones": ZONES,
        "zone_index": {z: i for i, z in enumerate(ZONES)},
        "league_zone_shares": {k: float(v) for k, v in zone_means.items()},
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the attainability model.")
    parser.add_argument("--name", default="attainability")
    args = parser.parse_args()
    train(name=args.name)
