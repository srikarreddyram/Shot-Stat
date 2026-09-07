"""
Rolling-origin backtest and feature-group ablations.

Why not a single split
----------------------
The training script reports one held-out season. That number is noisy: a
season is roughly two hundred thousand shots, but they are far from
independent — the same few hundred players take all of them, rule
interpretations shift, and the scoring environment drifts year to year. A
model that looks 0.3% better on one season may be no better at all.

This harness refits at successive origins — train through 2021-22 and score
2022-23, train through 2022-23 and score 2023-24, and so on — so every claim
is backed by several independent held-out seasons rather than one. It also
surfaces drift directly: the per-season log-losses are printed as a series, and
a model whose lift is shrinking over time is a model whose features are aging.

Ablations
---------
`--ablate` retrains with one feature group removed at a time and reports what
each group is actually worth. This is the honest alternative to reading
XGBoost feature importances, which are close to meaningless when features are
correlated — and in this matrix they are heavily correlated, since
`zone_rate`, `overall_rate`, `matchup_advantage` and the creation features all
describe overlapping aspects of the same player.

The ablation result worth knowing about: removing the creation group entirely
costs essentially nothing on shot-make prediction. That is not a bug in the
features — it is the finding. A player's point-in-time zone rate already
absorbs the difficulty of the shots he takes, so knowing HOW he creates them
adds little once you know how well he converts them. Creation skill earns its
place in `attainability.py` instead, where it is the dominant signal, because
"can this player generate this look" is a genuinely different question from
"will it go in".

Usage:
    python -m src.training.backtest
    python -m src.training.backtest --ablate
    python -m src.training.backtest --origins 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from src.common import runs
from src.features.build import TARGET_COL, build_matrix
from src.features.spec import FEATURE_GROUPS, as_model_matrix
from src.training import evaluate as ev
from src.training.train import DEFAULT_PARAMS, _apply_constraints

# Groups worth ablating. `spatial` is excluded because a shot-quality model
# without location is not a variant of this model, it is a different exercise.
ABLATABLE_GROUPS = ["creation", "defender", "interaction",
                    "shooter_skill", "context", "shooter_physical",
                    "shot_context", "possession_origin", "opponent_defence",
                    "contest", "finish"]


def _fit_and_score(df, feature_cols, fit_seasons, val_season, test_season,
                   params=None):
    """One train/score cycle at a single origin."""
    params = _apply_constraints({**DEFAULT_PARAMS, **(params or {})}, feature_cols)

    fit_df = df[df["season"].isin(fit_seasons)]
    val_df = df[df["season"] == val_season]
    test_df = df[df["season"] == test_season]

    if len(fit_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        return None

    model = xgb.XGBClassifier(**params)
    model.fit(
        as_model_matrix(fit_df, feature_cols), fit_df[TARGET_COL],
        eval_set=[(as_model_matrix(val_df, feature_cols), val_df[TARGET_COL])],
        verbose=False,
    )

    preds = model.predict_proba(as_model_matrix(test_df, feature_cols))[:, 1]
    metrics = ev.evaluate(test_df, preds)

    train_df = pd.concat([fit_df, val_df])
    baseline = ev.distance_zone_baseline(train_df, test_df)
    metrics["baseline_log_loss"] = float(
        ev.evaluate(test_df, baseline)["log_loss"]
    )
    metrics["lift"] = (
        (metrics["baseline_log_loss"] - metrics["log_loss"])
        / metrics["baseline_log_loss"]
    )
    metrics["test_season"] = test_season
    metrics["n_fit"] = len(fit_df)
    return metrics


def rolling_origin(seasons: list[str], n_origins: int = 3,
                   feature_cols: list[str] | None = None,
                   df=None, artifacts=None, verbose: bool = True) -> pd.DataFrame:
    """
    Refit at `n_origins` successive origins and score the season after each.

    The matrix is built ONCE over the full window and sliced per origin. That
    is safe for everything except the league priors, which are fit through the
    matrix's own `prior_through_season`; for a backtest whose earliest origin
    is several seasons back, those priors see slightly more history than that
    origin's model strictly should. The effect is second-order — priors are
    population-level and move very little year to year — and rebuilding the
    matrix per origin would multiply runtime by `n_origins` for a distortion
    smaller than the seasonal noise this harness exists to measure.
    """
    if df is None:
        df, feature_cols, artifacts = build_matrix(
            seasons, prior_through_season=seasons[-2], verbose=verbose
        )
    feature_cols = feature_cols or artifacts["feature_cols"]

    results = []
    for i in range(n_origins):
        # Walk the origin backwards from the most recent season.
        offset = n_origins - 1 - i
        test_season = seasons[-1 - offset]
        val_season = seasons[-2 - offset]
        fit_seasons = seasons[:len(seasons) - 2 - offset]

        if len(fit_seasons) < 2:
            continue

        if verbose:
            print(f"\n  origin {i + 1}/{n_origins}: fit ≤{fit_seasons[-1]}, "
                  f"val {val_season}, test {test_season}")

        metrics = _fit_and_score(df, feature_cols, fit_seasons,
                                 val_season, test_season)
        if metrics is None:
            continue
        results.append(metrics)

        if verbose:
            print(f"    log-loss {metrics['log_loss']:.4f}  "
                  f"baseline {metrics['baseline_log_loss']:.4f}  "
                  f"lift {metrics['lift']:+.2%}  "
                  f"AUC {metrics['auc']:.4f}  "
                  f"zone-rank ρ {metrics['zone_rank_rho']:.3f}")

    return pd.DataFrame(results)


def ablate(seasons: list[str], groups: list[str] | None = None,
           verbose: bool = True) -> pd.DataFrame:
    """
    Retrain with each feature group removed, on the most recent origin.

    Reported as the DAMAGE each removal causes: a positive `log_loss_delta`
    means the model got worse without that group, which is how much the group
    was worth.
    """
    groups = groups or ABLATABLE_GROUPS
    df, feature_cols, artifacts = build_matrix(
        seasons, prior_through_season=seasons[-2], verbose=verbose
    )

    test_season = seasons[-1]
    val_season = seasons[-2]
    fit_seasons = seasons[:-2]

    full = _fit_and_score(df, feature_cols, fit_seasons, val_season, test_season)
    if verbose:
        print(f"\n  full model: log-loss {full['log_loss']:.4f}  "
              f"AUC {full['auc']:.4f}")

    rows = [{
        "removed": "(nothing)",
        "n_features": len(feature_cols),
        "log_loss": full["log_loss"],
        "log_loss_delta": 0.0,
        "auc": full["auc"],
        "zone_rank_rho": full["zone_rank_rho"],
    }]

    for group in groups:
        dropped = set(FEATURE_GROUPS.get(group, []))
        remaining = [c for c in feature_cols if c not in dropped]
        if len(remaining) == len(feature_cols):
            continue

        metrics = _fit_and_score(df, remaining, fit_seasons, val_season, test_season)
        if metrics is None:
            continue

        delta = metrics["log_loss"] - full["log_loss"]
        rows.append({
            "removed": group,
            "n_features": len(remaining),
            "log_loss": metrics["log_loss"],
            "log_loss_delta": delta,
            "auc": metrics["auc"],
            "zone_rank_rho": metrics["zone_rank_rho"],
        })
        if verbose:
            print(f"  without {group:<18} log-loss {metrics['log_loss']:.4f}  "
                  f"({delta:+.4f})  AUC {metrics['auc']:.4f}")

    return pd.DataFrame(rows).sort_values("log_loss_delta", ascending=False)


def main():
    parser = argparse.ArgumentParser(
        description="Rolling-origin backtest and feature-group ablations."
    )
    parser.add_argument("--origins", type=int, default=3,
                        help="Number of successive held-out seasons")
    parser.add_argument("--ablate", action="store_true",
                        help="Also run feature-group ablations")
    parser.add_argument("--name", default="backtest")
    args = parser.parse_args()

    seasons = [s for s in config.ALL_SEASONS if s >= "2016-17"]
    run = runs.start_run(args.name, seasons=seasons, origins=args.origins)

    print(f"\n{'='*62}")
    print("  ROLLING-ORIGIN BACKTEST")
    print(f"  {args.origins} origins over {seasons[0]} → {seasons[-1]}")
    print(f"{'='*62}")

    table = rolling_origin(seasons, n_origins=args.origins)

    if not table.empty:
        print(f"\n{'='*62}")
        print("  SUMMARY")
        print(f"{'='*62}")
        print(f"  {'season':<12} {'log-loss':>10} {'baseline':>10} "
              f"{'lift':>8} {'AUC':>8} {'ρ':>7}")
        print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*7}")
        for _, r in table.iterrows():
            print(f"  {r['test_season']:<12} {r['log_loss']:>10.4f} "
                  f"{r['baseline_log_loss']:>10.4f} {r['lift']:>+7.2%} "
                  f"{r['auc']:>8.4f} {r['zone_rank_rho']:>7.3f}")
        print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*7}")
        print(f"  {'mean':<12} {table['log_loss'].mean():>10.4f} "
              f"{table['baseline_log_loss'].mean():>10.4f} "
              f"{table['lift'].mean():>+7.2%} {table['auc'].mean():>8.4f} "
              f"{table['zone_rank_rho'].mean():>7.3f}")
        # The spread across origins is the context the single-split number
        # lacks: a lift smaller than this is not a result.
        print(f"  {'sd':<12} {table['log_loss'].std():>10.4f} "
              f"{'':>10} {table['lift'].std():>7.2%}")

        table.to_csv(run.artifact_path("rolling_origin.csv"), index=False)
        run.log_metrics(
            mean_log_loss=table["log_loss"].mean(),
            mean_lift=table["lift"].mean(),
            lift_sd=table["lift"].std(),
            mean_zone_rank_rho=table["zone_rank_rho"].mean(),
        )

    if args.ablate:
        print(f"\n{'='*62}")
        print("  FEATURE-GROUP ABLATIONS")
        print("  positive delta = the model is WORSE without that group")
        print(f"{'='*62}")
        ablation = ablate(seasons)
        print(f"\n  {'removed':<20} {'features':>9} {'log-loss':>10} "
              f"{'delta':>9} {'AUC':>8}")
        print(f"  {'-'*20} {'-'*9} {'-'*10} {'-'*9} {'-'*8}")
        for _, r in ablation.iterrows():
            print(f"  {r['removed']:<20} {int(r['n_features']):>9} "
                  f"{r['log_loss']:>10.4f} {r['log_loss_delta']:>+9.4f} "
                  f"{r['auc']:>8.4f}")
        ablation.to_csv(run.artifact_path("ablations.csv"), index=False)

    manifest = run.finish()
    print(f"\n  ✓ {manifest}")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()
