"""
Hyperparameter search over the current feature set.

Rewritten. The previous version imported `build_training_matrix` from the
deprecated `feature_engineering` module, so it tuned against the old leaky
features — and the parameters it produced (`models/best_params_v3.json`) were
never used by anything afterwards. Every model in the repo trained on
hand-picked defaults instead.

Two things the old search also got wrong:

  Folds were not temporal. It sliced the matrix by row position, on the stated
  assumption that rows were time-ordered, but the query had no ORDER BY — so
  the "temporal" folds were whatever order the engine happened to emit. Folds
  here are built from SEASONS, which cannot silently stop being chronological.

  Early stopping was scored on the same fold it selected against, so each
  trial's reported loss was mildly optimistic. Ranking survives that, but the
  numbers were not comparable to anything else in the pipeline. Each fold now
  early-stops on the tail of its own training window and scores the held-out
  season untouched.

Usage:
    python -m src.training.tune_hyperparams --trials 40
    python -m src.training.tune_hyperparams --trials 60 --sample-frac 0.5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import optuna
import xgboost as xgb
from sklearn.metrics import log_loss

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from src.common import runs
from src.features.build import TARGET_COL, build_matrix
from src.features.spec import FEATURE_GROUPS, as_model_matrix

MODEL_DIR = Path(config.PROJECT_ROOT) / "models"


def _search_space(trial) -> dict:
    return {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1,
        "max_depth": trial.suggest_int("max_depth", 4, 9),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 300, 1400, step=100),
        # The floor is high on purpose. Shot outcomes are noisy, and small leaves
        # are exactly how the model came to fit 25 rows of extreme height
        # mismatch and produce an 18-point spurious defender effect.
        "min_child_weight": trial.suggest_int("min_child_weight", 20, 200, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 0.5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
    }


def _objective(trial, df, feature_cols, folds):
    params = _search_space(trial)
    scores = []

    for fit_seasons, val_season in folds:
        fit_df = df[df["season"].isin(fit_seasons)]
        val_df = df[df["season"] == val_season]
        if len(fit_df) < 50_000 or len(val_df) < 10_000:
            continue

        # Early-stop on the tail of the FIT window, never on the fold's own
        # scoring season — otherwise the trial selects its tree count against
        # the number it is about to report.
        stop_season = fit_seasons[-1]
        inner_fit = fit_df[fit_df["season"] != stop_season]
        inner_stop = fit_df[fit_df["season"] == stop_season]
        if len(inner_fit) < 20_000 or len(inner_stop) < 5_000:
            inner_fit, inner_stop = fit_df, val_df

        model = xgb.XGBClassifier(**params, early_stopping_rounds=40)
        model.fit(
            as_model_matrix(inner_fit, feature_cols), inner_fit[TARGET_COL],
            eval_set=[(as_model_matrix(inner_stop, feature_cols), inner_stop[TARGET_COL])],
            verbose=False,
        )
        preds = model.predict_proba(as_model_matrix(val_df, feature_cols))[:, 1]
        scores.append(log_loss(val_df[TARGET_COL], preds))

        trial.report(float(np.mean(scores)), len(scores))
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(scores)) if scores else float("inf")


def tune(n_trials: int = 40, name: str = "tuning", sample_frac: float = 1.0) -> dict:
    seasons = [s for s in config.ALL_SEASONS if s >= "2016-17"]

    # The final season is the project's test set and is excluded entirely — a
    # hyperparameter chosen by looking at it is a hyperparameter fitted to it.
    tune_seasons = seasons[:-1]

    df, feature_cols, _ = build_matrix(
        seasons, prior_through_season=seasons[-2], verbose=True
    )
    physical = set(FEATURE_GROUPS["defender_physical"])
    feature_cols = [c for c in feature_cols if c not in physical]
    df = df[df["season"].isin(tune_seasons)]

    if sample_frac < 1.0:
        # Sampled within season so every fold keeps its shape; hyperparameter
        # rankings are far more robust to subsampling than final quality is.
        df = df.groupby("season", group_keys=False).sample(
            frac=sample_frac, random_state=42
        )

    # Expanding-window folds: train on everything before, score the next season.
    folds = [
        (tune_seasons[:i], tune_seasons[i])
        for i in range(len(tune_seasons) - 3, len(tune_seasons))
        if i >= 2
    ]

    print(f"\n{'='*62}")
    print("  HYPERPARAMETER SEARCH")
    print(f"  {n_trials} trials, {len(folds)} expanding-window folds")
    print(f"  scoring seasons: {[v for _, v in folds]}")
    print(f"  test season {seasons[-1]} excluded entirely")
    print(f"  {len(df):,} rows x {len(feature_cols)} features")
    print(f"{'='*62}\n")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8),
    )

    def progress(study, trial):
        if trial.number % 5 == 0 or trial.number == n_trials - 1:
            print(f"  trial {trial.number + 1}/{n_trials}: best {study.best_value:.5f}")

    study.optimize(
        lambda t: _objective(t, df, feature_cols, folds),
        n_trials=n_trials,
        callbacks=[progress],
    )

    print(f"\n  best log-loss {study.best_value:.5f}")
    for key, value in sorted(study.best_params.items()):
        print(f"    {key:<20} {value}")

    path = MODEL_DIR / "best_params.json"
    path.write_text(json.dumps(study.best_params, indent=2))
    print(f"\n  ✓ {path}")

    run = runs.start_run(name, n_trials=n_trials, folds=[v for _, v in folds],
                         sample_frac=sample_frac)
    run.log_metrics(best_cv_log_loss=study.best_value)
    run.log_params(**study.best_params)
    run.finish()

    return study.best_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune XGBoost on the current features.")
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--sample-frac", type=float, default=1.0)
    parser.add_argument("--name", default="tuning")
    args = parser.parse_args()
    tune(n_trials=args.trials, name=args.name, sample_frac=args.sample_frac)
