"""
Hyperparameter Tuning — Uses Optuna (Bayesian optimization) to find the best XGBoost parameters.

Runs N trials with 3-fold temporal cross-validation on the training data.
Saves the best parameters to models/best_params_{version}.json.

Usage:
    python -m src.training.tune_hyperparams --trials 50 --version v3
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
import xgboost as xgb
from sklearn.metrics import log_loss

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from src.training.feature_engineering import build_training_matrix, get_feature_columns, TARGET_COL


def objective(trial, X_train, y_train, feature_cols):
    """Optuna objective: train XGBoost with sampled hyperparameters, return CV log-loss."""

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "enable_categorical": False,
        "random_state": 42,
        "n_jobs": -1,

        # Tunable hyperparameters
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
        "min_child_weight": trial.suggest_int("min_child_weight", 10, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 0.5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 0.5),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 3.0),
    }

    # 3-fold temporal cross-validation
    # Split training data into 3 temporal chunks (respects time ordering)
    n = len(X_train)
    fold_size = n // 4  # Use last 25% as validation for each fold

    scores = []
    for fold in range(3):
        # Each fold shifts the split point forward
        val_start = n - fold_size * (fold + 1)
        val_end = n - fold_size * fold if fold > 0 else n

        X_tr = X_train.iloc[:val_start]
        y_tr = y_train.iloc[:val_start]
        X_val = X_train.iloc[val_start:val_end]
        y_val = y_train.iloc[val_start:val_end]

        if len(X_tr) < 10000 or len(X_val) < 5000:
            continue

        model = xgb.XGBClassifier(**params, early_stopping_rounds=20)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=0,
        )

        preds = model.predict_proba(X_val)[:, 1]
        ll = log_loss(y_val, preds)
        scores.append(ll)

    return np.mean(scores) if scores else float("inf")


def tune(n_trials: int = 50, version: str = "v3"):
    """Run Optuna hyperparameter search."""

    print(f"\n{'='*60}")
    print(f"  HYPERPARAMETER TUNING (Optuna)")
    print(f"  {n_trials} trials with 3-fold temporal CV")
    print(f"{'='*60}\n")

    # Build training matrix (defender features, 2016+ only)
    seasons = [s for s in config.ALL_SEASONS if s >= "2016-17"]
    df = build_training_matrix(seasons, use_defender=True)
    feature_cols = get_feature_columns(df)

    # Temporal split: train on all but last season
    TRAIN_SEASONS = seasons[:-1]
    train_df = df[df["season"].isin(TRAIN_SEASONS)].copy()

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL]

    print(f"  Training data: {len(X_train):,} shots")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Starting {n_trials} trials...\n")

    # Create Optuna study
    study = optuna.create_study(
        direction="minimize",
        study_name=f"xgb_{version}_tuning",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
    )

    # Suppress Optuna's verbose logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Run optimization with a callback for progress
    def callback(study, trial):
        if trial.number % 5 == 0 or trial.number == n_trials - 1:
            print(f"  Trial {trial.number + 1}/{n_trials}: "
                  f"log-loss={trial.value:.5f} "
                  f"(best={study.best_value:.5f})")

    study.optimize(
        lambda trial: objective(trial, X_train, y_train, feature_cols),
        n_trials=n_trials,
        callbacks=[callback],
    )

    # Print results
    print(f"\n{'='*60}")
    print(f"  TUNING COMPLETE")
    print(f"{'='*60}")
    print(f"  Best log-loss:  {study.best_value:.5f}")
    print(f"  Best trial:     #{study.best_trial.number}")
    print(f"\n  Best Parameters:")
    for k, v in sorted(study.best_params.items()):
        print(f"    {k:<25} {v}")

    # Save best params
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    params_path = model_dir / f"best_params_{version}.json"
    with open(params_path, "w") as f:
        json.dump(study.best_params, f, indent=2)
    print(f"\n  ✓ Saved best params to {params_path}")

    # Compare with v2 defaults
    print(f"\n  v2 defaults for reference:")
    v2_defaults = {
        "max_depth": 6, "learning_rate": 0.05, "n_estimators": 500,
        "min_child_weight": 50, "subsample": 0.8, "colsample_bytree": 0.8,
        "gamma": 0, "reg_alpha": 0.1, "reg_lambda": 1.0,
    }
    for k, v in sorted(v2_defaults.items()):
        best_v = study.best_params.get(k, "N/A")
        changed = "  ←" if best_v != v else ""
        print(f"    {k:<25} {v} → {best_v}{changed}")

    return study.best_params


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Tune XGBoost hyperparameters with Optuna.")
    parser.add_argument("--trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--version", type=str, default="v3", help="Version tag for saved params")
    args = parser.parse_args()

    tune(n_trials=args.trials, version=args.version)
