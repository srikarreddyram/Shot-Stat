"""
Shot-quality model training.

Replaces `train_baseline.py`. What changed and why:

  features       Point-in-time and shrunk, built by `src.features.build`.
                 The old whole-season aggregates put each shot's own outcome
                 inside its own features and could not be reproduced at
                 prediction time.

  defender       A possession-weighted mixture over everyone who guarded the
                 shooter, not the single defender with the most minutes that
                 night.

  creation       Lagged handle/passing profile, so the model can tell a
                 contested pull-up from an open catch-and-shoot at the same
                 spot on the floor.

  one model      The interior/perimeter split is gone. It hard-coded an
                 interaction the trees can learn from distance, halved the
                 data available to each half, and left two independently
                 calibrated heads meeting at the paint boundary. `--split`
                 still trains the old way for comparison; `backtest.py`
                 reports which wins.

  calibration    No longer fit on the model's own training rows (see
                 `calibration.py` for why that was wrong). It is also no
                 longer applied unconditionally: the mapping is fit on the
                 most recent unseen data and adopted only if it improves
                 held-out log-loss. On this dataset it does not — the raw
                 model is already well calibrated, and a mapping learned from
                 an earlier scoring environment makes it worse.

  baseline       Compared against zone + distance-spline + season logistic,
                 not just a zone average. Beating a zone average mostly
                 demonstrates that shots get harder with distance.

  bookkeeping    Every run writes runs/<stamp>__<name>/run.json with params,
                 metrics, a data hash, and the git SHA.

Usage:
    python -m src.training.train
    python -m src.training.train --name with-creation
    python -m src.training.train --no-creation      # ablation
    python -m src.training.train --split            # old interior/perimeter
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from src.common import runs
from src.features.build import TARGET_COL, build_matrix
from src.features.spec import FEATURE_GROUPS, as_model_matrix, interior_mask
from src.training import evaluate as ev
from src.training.calibration import (
    fit_calibrator_holdout,
    fit_calibrator_oof,
    save_calibration_plot,
)

MODEL_DIR = Path(config.PROJECT_ROOT) / "models"

# Thread cap for XGBoost. Defaults to every core, but a full-throttle training
# run on a laptop is a real thermal event — the hyperparameter search at
# n_jobs=-1 was enough to force a stop mid-run. Set XGB_N_JOBS to keep some
# headroom (e.g. XGB_N_JOBS=4) at the cost of a longer wall clock.
_N_JOBS = int(os.environ.get("XGB_N_JOBS", "-1"))

DEFAULT_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "n_estimators": 900,
    "max_depth": 6,
    "learning_rate": 0.04,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 60,
    "reg_alpha": 0.1,
    "reg_lambda": 1.5,
    "random_state": 42,
    "n_jobs": _N_JOBS,
    "early_stopping_rounds": 40,
}

# Monotone constraints encode basketball facts the model should never be
# allowed to contradict, however the training data happens to wiggle. Each is
# a direction we are confident about a priori; anything we are not confident
# about is left unconstrained.
#
# Constraints also buy honesty in the recommender, which sweeps these features
# across a grid of hypotheticals. Without them the model can learn locally
# non-monotone quirks that make the UI say a shooter improves against a better
# defender, which is never a modelling insight and always an artifact.
MONOTONE_CONSTRAINTS = {
    "zone_rate": 1,             # better shooter from here → likelier make
    "zone_rate_vs_league": 1,
    "overall_rate": 1,
    "def_fg_pct_zone": 1,       # defender allows more → likelier make
    "def_fg_pct_overall": 1,
    "def_pct_plusminus": 1,     # opponents shoot better vs him → likelier make
    "def_pct_plusminus_zone": 1,
    "matchup_advantage": 1,
    "expected_contest": 1,      # more daylight → likelier make
}


def _apply_constraints(params: dict, feature_cols: list[str]) -> dict:
    """
    Attach monotone constraints as a positional tuple.

    XGBoost's dict form silently ignores names it does not recognize, so a
    renamed feature would drop its constraint without a word. Building the
    tuple by position against the actual feature list makes the mapping
    explicit and lets us log exactly how many constraints bound.
    """
    out = dict(params)
    constraints = tuple(
        MONOTONE_CONSTRAINTS.get(col, 0) for col in feature_cols
    )
    out["monotone_constraints"] = constraints
    return out


def _split_seasons(seasons: list[str]) -> tuple[list[str], str, str]:
    """
    Chronological three-way split.

    fit    → the model learns here
    val    → early stopping only, never scored
    test   → held out entirely, the only source of reported metrics

    The old pipeline briefly used the test season as its early-stopping
    eval_set, which let the model choose its tree count by watching the score
    it was about to report. That is fixed upstream of here; the split is kept
    explicit so it stays fixed.
    """
    test_season = seasons[-1]
    val_season = seasons[-2]
    fit_seasons = seasons[:-2]
    return fit_seasons, val_season, test_season


def train(
    seasons: list[str] | None = None,
    name: str = "shot-quality",
    use_creation: bool = True,
    use_defender: bool = True,
    hierarchical_split: bool = False,
    use_defender_physicals: bool = False,
    use_spatial_basis: bool = False,
    params: dict | None = None,
    calibration_folds: int = 3,
    calibration_mode: str = "recent",
) -> dict:
    """Train, evaluate, and persist one model. Returns the metrics dict."""
    seasons = seasons or [s for s in config.ALL_SEASONS if s >= "2016-17"]
    params = {**DEFAULT_PARAMS, **(params or {})}

    fit_seasons, val_season, test_season = _split_seasons(seasons)

    run = runs.start_run(
        name,
        seasons=seasons, fit_seasons=fit_seasons,
        val_season=val_season, test_season=test_season,
        use_creation=use_creation, use_defender=use_defender,
        use_defender_physicals=use_defender_physicals,
        use_spatial_basis=use_spatial_basis,
        hierarchical_split=hierarchical_split,
        xgb_params={k: v for k, v in params.items() if k != "monotone_constraints"},
        calibration_folds=calibration_folds,
        calibration_mode=calibration_mode,
    )

    # ── Build ────────────────────────────────────────────────────────────
    df, feature_cols, artifacts = build_matrix(
        seasons,
        prior_through_season=val_season,
        use_defender=use_defender,
    )

    if not use_spatial_basis:
        # See FEATURE_GROUPS["spatial_basis"] — measured slightly harmful.
        basis_cols = set(FEATURE_GROUPS["spatial_basis"])
        feature_cols = [c for c in feature_cols if c not in basis_cols]

    if not use_defender_physicals:
        # See FEATURE_GROUPS["defender_physical"] for why these are out by
        # default: no measurable effect, and a large spurious one at serve time.
        physical_cols = set(FEATURE_GROUPS["defender_physical"])
        feature_cols = [c for c in feature_cols if c not in physical_cols]

    if not use_creation:
        # Ablation: drop the creation group entirely so its contribution can be
        # read off the metric difference rather than guessed from importances,
        # which are unreliable when features are correlated.
        creation_cols = set(FEATURE_GROUPS["creation"])
        feature_cols = [c for c in feature_cols if c not in creation_cols]

    df = df.sort_values(["game_date", "game_id", "shot_id"]).reset_index(drop=True)

    fit_df = df[df["season"].isin(fit_seasons)]
    val_df = df[df["season"] == val_season]
    test_df = df[df["season"] == test_season]

    X_fit, y_fit = as_model_matrix(fit_df, feature_cols), fit_df[TARGET_COL]
    X_val, y_val = as_model_matrix(val_df, feature_cols), val_df[TARGET_COL]
    X_test, y_test = as_model_matrix(test_df, feature_cols), test_df[TARGET_COL]

    print(f"\n{'='*62}")
    print("  SPLIT")
    print(f"{'='*62}")
    print(f"  fit   {len(X_fit):>10,}  ({fit_seasons[0]} → {fit_seasons[-1]})")
    print(f"  val   {len(X_val):>10,}  ({val_season}) — early stopping only")
    print(f"  test  {len(X_test):>10,}  ({test_season}) — held out")
    print(f"  features: {len(feature_cols)}")

    run.log_data(
        n_fit=len(X_fit), n_val=len(X_val), n_test=len(X_test),
        n_features=len(feature_cols),
        data_hash=runs.frame_hash(df, feature_cols + [TARGET_COL]),
        test_base_rate=float(y_test.mean()),
    )

    # ── Baselines ────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print("  BASELINES")
    print(f"{'='*62}")

    train_df = pd.concat([fit_df, val_df])
    train_source = train_df
    zone_avg = train_df.groupby("zone")[TARGET_COL].mean()
    zone_preds = test_df["zone"].map(zone_avg).fillna(y_fit.mean()).clip(0.01, 0.99)
    zone_metrics = ev.evaluate(test_df, zone_preds, label="zone-average")
    print(f"  zone average          log-loss {zone_metrics['log_loss']:.4f}")

    strong_preds = ev.distance_zone_baseline(train_df, test_df)
    strong_metrics = ev.evaluate(test_df, strong_preds, label="zone+distance+season")
    print(f"  zone+distance+season  log-loss {strong_metrics['log_loss']:.4f}")

    run.log_metrics(prefix="baseline_zone", **{
        k: v for k, v in zone_metrics.items() if not k.startswith("_")})
    run.log_metrics(prefix="baseline_strong", **{
        k: v for k, v in strong_metrics.items() if not k.startswith("_")})

    # ── Model ────────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  XGBOOST ({'interior/perimeter split' if hierarchical_split else 'single model'})")
    print(f"{'='*62}")

    fit_params = _apply_constraints(params, feature_cols)
    n_bound = sum(1 for c in fit_params["monotone_constraints"] if c != 0)
    print(f"  monotone constraints bound: {n_bound}")

    def make_model():
        return xgb.XGBClassifier(**fit_params)

    def make_calibration_model(n_rounds: int):
        """
        A fold model for out-of-fold calibration.

        Early stopping is removed: each fold predicts forward with no
        validation slice to watch, and XGBoost refuses to fit at all if
        `early_stopping_rounds` is set without an eval_set. The tree count is
        pinned to what the production model actually settled on, so the fold
        models have the same capacity — and therefore the same overconfidence
        profile — as the model whose probabilities the calibrator will correct.
        Letting them run to a different depth would calibrate the wrong curve.
        """
        cal_params = {k: v for k, v in fit_params.items()
                      if k != "early_stopping_rounds"}
        cal_params["n_estimators"] = max(int(n_rounds), 50)
        return xgb.XGBClassifier(**cal_params)

    if hierarchical_split:
        models, raw_test = {}, np.zeros(len(X_test))
        for label, mask_fn in (
            ("interior", lambda d: interior_mask(d)),
            ("perimeter", lambda d: ~interior_mask(d)),
        ):
            m_fit, m_val, m_test = mask_fn(fit_df), mask_fn(val_df), mask_fn(test_df)
            model = make_model()
            model.fit(X_fit[m_fit.values], y_fit[m_fit.values],
                      eval_set=[(X_val[m_val.values], y_val[m_val.values])],
                      verbose=False)
            models[label] = model
            raw_test[m_test.values] = model.predict_proba(X_test[m_test.values])[:, 1]
            print(f"  {label:<10} trained on {int(m_fit.sum()):,} shots, "
                  f"best_iteration={model.best_iteration}")
        model = models
    else:
        model = make_model()
        model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
        print(f"  best_iteration={model.best_iteration}")
        raw_test = model.predict_proba(X_test)[:, 1]

    raw_metrics = ev.evaluate(test_df, raw_test, label="xgb-raw")
    print(f"\n  raw   log-loss {raw_metrics['log_loss']:.4f}  "
          f"AUC {raw_metrics['auc']:.4f}  ECE {raw_metrics['ece']:.4f}")

    # ── Calibration ──────────────────────────────────────────────────────
    #
    # Calibration is now a decision, not a reflex. The first corrected run
    # showed out-of-fold calibration making the model WORSE on the test season
    # (ECE 0.0071 → 0.0142): the raw model was already close to calibrated, and
    # a mapping fit across 2016-17 → 2023-24 encodes that era's scoring
    # environment, which it then imposes on a later season that scores
    # differently. A stale correction is worse than no correction.
    #
    # So: fit the mapping on the most RECENT out-of-sample data available (the
    # first half of the validation season), judge it on data used for neither
    # fitting nor early stopping (the second half), and adopt it only if it
    # actually helps. The test season is untouched by any of this.
    print(f"\n{'='*62}")
    print(f"  CALIBRATION (mode: {calibration_mode})")
    print(f"{'='*62}")

    def _fit_calibrators(predict_fn):
        """
        Fit calibrators under the selected mode and return
        (calibrators, adopted, decision_note).

        `predict_fn(X)` returns raw probabilities from the trained model(s).
        """
        if calibration_mode == "none":
            return {}, False, "disabled"

        if calibration_mode == "oof":
            cals = {}
            if hierarchical_split:
                for label, mask_fn in (
                    ("interior", interior_mask),
                    ("perimeter", lambda d: ~interior_mask(d)),
                ):
                    m_fit = mask_fn(fit_df)
                    n_rounds = models[label].best_iteration or params["n_estimators"]
                    cals[label] = fit_calibrator_oof(
                        lambda n=n_rounds: make_calibration_model(n),
                        X_fit[m_fit.values], y_fit[m_fit.values],
                        order=fit_df.loc[m_fit.values, "game_date"],
                        n_folds=calibration_folds,
                    )
            else:
                n_rounds = model.best_iteration or params["n_estimators"]
                cals["all"] = fit_calibrator_oof(
                    lambda n=n_rounds: make_calibration_model(n),
                    X_fit, y_fit, order=fit_df["game_date"],
                    n_folds=calibration_folds,
                )
            return cals, True, f"oof-{calibration_folds}fold (adopted unconditionally)"

        # ── recent-holdout mode (default) ────────────────────────────────
        # The validation season, split chronologically. The model never
        # trained on either half; the first half fits the mapping, the second
        # half decides whether the mapping is worth keeping.
        val_sorted = val_df.sort_values(["game_date", "game_id", "shot_id"])
        midpoint = len(val_sorted) // 2
        cal_fit_df, cal_eval_df = val_sorted.iloc[:midpoint], val_sorted.iloc[midpoint:]

        raw_cal_fit = predict_fn(as_model_matrix(cal_fit_df, feature_cols))
        raw_cal_eval = predict_fn(as_model_matrix(cal_eval_df, feature_cols))

        cals = {}
        if hierarchical_split:
            for label, mask_fn in (
                ("interior", interior_mask),
                ("perimeter", lambda d: ~interior_mask(d)),
            ):
                m = mask_fn(cal_fit_df).values
                cals[label] = fit_calibrator_holdout(
                    raw_cal_fit[m], cal_fit_df[TARGET_COL].values[m]
                )
            adjusted_eval = np.zeros(len(cal_eval_df))
            for label, mask_fn in (
                ("interior", interior_mask),
                ("perimeter", lambda d: ~interior_mask(d)),
            ):
                m = mask_fn(cal_eval_df).values
                adjusted_eval[m] = cals[label].predict(raw_cal_eval[m])
        else:
            cals["all"] = fit_calibrator_holdout(
                raw_cal_fit, cal_fit_df[TARGET_COL].values
            )
            adjusted_eval = cals["all"].predict(raw_cal_eval)

        y_cal_eval = cal_eval_df[TARGET_COL].values
        raw_ll = ev.evaluate(cal_eval_df, raw_cal_eval)["log_loss"]
        cal_ll = ev.evaluate(cal_eval_df, adjusted_eval)["log_loss"]
        raw_ece = ev.expected_calibration_error(y_cal_eval, raw_cal_eval)
        cal_ece = ev.expected_calibration_error(y_cal_eval, adjusted_eval)

        print(f"    holdout decision on {len(cal_eval_df):,} unseen shots:")
        print(f"      raw         log-loss {raw_ll:.4f}  ECE {raw_ece:.4f}")
        print(f"      calibrated  log-loss {cal_ll:.4f}  ECE {cal_ece:.4f}")

        adopted = cal_ll < raw_ll
        note = ("adopted: improves held-out log-loss"
                if adopted else
                "REJECTED: raw model already better calibrated than the mapping")
        print(f"      → {note}")
        run.log_metrics(prefix="calibration_decision",
                        raw_log_loss=raw_ll, calibrated_log_loss=cal_ll,
                        raw_ece=raw_ece, calibrated_ece=cal_ece,
                        adopted=int(adopted))
        return cals, adopted, note

    def _predict_raw(X_frame):
        if hierarchical_split:
            # Reconstruct the interior mask from the encoded zone indicators,
            # so this works on any frame carrying the model's feature columns.
            out = np.zeros(len(X_frame))
            int_cols = ["zone_is_restricted_area", "zone_is_paint"]
            present = [c for c in int_cols if c in X_frame.columns]
            m = X_frame[present].sum(axis=1).values > 0
            if m.any():
                out[m] = models["interior"].predict_proba(X_frame[m])[:, 1]
            if (~m).any():
                out[~m] = models["perimeter"].predict_proba(X_frame[~m])[:, 1]
            return out
        return model.predict_proba(X_frame)[:, 1]

    calibrators, calibration_adopted, calibration_note = _fit_calibrators(_predict_raw)

    if not calibration_adopted:
        calibrated_test = raw_test
    elif hierarchical_split:
        calibrated_test = np.zeros(len(X_test))
        for label, mask_fn in (
            ("interior", interior_mask),
            ("perimeter", lambda d: ~interior_mask(d)),
        ):
            m = mask_fn(test_df).values
            calibrated_test[m] = calibrators[label].predict(raw_test[m])
    else:
        calibrated_test = calibrators["all"].predict(raw_test)

    cal_metrics = ev.evaluate(test_df, calibrated_test, label="xgb-calibrated")
    print(f"\n  raw        log-loss {raw_metrics['log_loss']:.4f}  "
          f"ECE {raw_metrics['ece']:.4f}")
    print(f"  calibrated log-loss {cal_metrics['log_loss']:.4f}  "
          f"ECE {cal_metrics['ece']:.4f}")

    # ── Report ───────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  TEST SEASON {test_season}")
    print(f"{'='*62}")

    zone_table = ev.per_zone_metrics(test_df, calibrated_test)
    residuals = ev.per_player_residuals(test_df, calibrated_test)
    ev.print_report(cal_metrics, zone_table, residuals)

    lift_zone = (zone_metrics["log_loss"] - cal_metrics["log_loss"]) / zone_metrics["log_loss"]
    lift_strong = (strong_metrics["log_loss"] - cal_metrics["log_loss"]) / strong_metrics["log_loss"]
    print(f"\n  lift over zone average           {lift_zone:+.2%}")
    print(f"  lift over zone+distance+season   {lift_strong:+.2%}   ← the honest number")

    run.log_metrics(prefix="test", **{
        k: v for k, v in cal_metrics.items() if not k.startswith("_")})
    run.log_metrics(prefix="test_raw", **{
        k: v for k, v in raw_metrics.items() if not k.startswith("_")})
    run.log_metrics(
        lift_over_zone_average=lift_zone,
        lift_over_strong_baseline=lift_strong,
        residual_z_mean=float(residuals["z"].mean()) if not residuals.empty else float("nan"),
        residual_z_sd=float(residuals["z"].std()) if not residuals.empty else float("nan"),
    )

    # ── Persist ──────────────────────────────────────────────────────────
    MODEL_DIR.mkdir(exist_ok=True)
    if hierarchical_split:
        for label, m in model.items():
            m.save_model(str(MODEL_DIR / f"xgb_{label}_{name}.json"))
    else:
        model.save_model(str(MODEL_DIR / f"xgb_{name}.json"))

    # Per-feature training range, stored so the serving path can clamp to it.
    # The recommender invents feature vectors that never occurred (this player,
    # this spot, that defender) and a boosted tree does not degrade gracefully
    # outside the region it was fitted on — it applies the outermost leaf's
    # value with full confidence. Clipping bounds the damage to "the most
    # extreme matchup the data actually contains".
    feature_bounds = {}
    for col in feature_cols:
        if col not in train_source.columns:
            continue
        values = train_source[col].dropna()
        if values.empty:
            continue
        lo, hi = float(values.quantile(0.01)), float(values.quantile(0.99))
        if hi > lo:
            feature_bounds[col] = [lo, hi]

    metadata = {
        "name": name,
        "feature_bounds": feature_bounds,
        "feature_cols": feature_cols,
        "hierarchical_split": hierarchical_split,
        "use_creation": use_creation,
        "use_defender": use_defender,
        "seasons": seasons,
        "fit_seasons": fit_seasons,
        "val_season": val_season,
        "test_season": test_season,
        "zone_avg": {k: float(v) for k, v in zone_avg.items()},
        "league_zone_rates": artifacts["league_zone_rates"],
        # Priors travel with the model. Refitting them at serving time from
        # whatever is in the database would mean the API silently applying a
        # different shrinkage than the model was trained under.
        "zone_priors": {
            zone: {"mean": p.mean, "strength": p.strength,
                   "n_players": p.n_players, "n_attempts": p.n_attempts}
            for zone, p in artifacts["zone_priors"].items()
        },
        # Point-in-time defender quality (see `build_defender_category_rates`)
        # shrinks toward these — travel with the model for the same reason
        # `zone_priors` does.
        "category_priors": {
            cat: {"mean": p.mean, "strength": p.strength,
                  "n_players": p.n_players, "n_attempts": p.n_attempts}
            for cat, p in artifacts.get("category_priors", {}).items()
        },
        "calibrators": {k: c.to_dict() for k, c in calibrators.items()},
        "calibration_adopted": calibration_adopted,
        "calibration_note": calibration_note,
        "run_directory": str(run.directory),
        "test_metrics": {k: v for k, v in cal_metrics.items() if not k.startswith("_")},
    }
    metadata_path = MODEL_DIR / f"metadata_{name}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str))

    save_calibration_plot(
        y_test, raw_test, calibrated_test,
        run.artifact_path("calibration.png"),
        title=f"{name} — test {test_season}",
    )
    zone_table.to_csv(run.artifact_path("per_zone.csv"), index=False)
    residuals.to_csv(run.artifact_path("player_residuals.csv"), index=False)

    manifest = run.finish()
    print(f"\n  ✓ model     {MODEL_DIR / f'xgb_{name}.json'}")
    print(f"  ✓ metadata  {metadata_path}")
    print(f"  ✓ run       {manifest}")
    print(f"{'='*62}\n")

    return cal_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the shot-quality model.")
    parser.add_argument("--name", default="shot-quality")
    parser.add_argument("--from-season", default="2016-17",
                        help="Earliest season to train on. A shorter window "
                             "trades data for the ability to use features whose "
                             "backfill is not finished league-wide yet — the "
                             "play-by-play context gate in build_matrix scores "
                             "coverage over exactly the seasons requested.")
    parser.add_argument("--no-creation", action="store_true",
                        help="Ablate creation features")
    parser.add_argument("--no-defender", action="store_true",
                        help="Ablate defender features")
    parser.add_argument("--defender-physicals", action="store_true",
                        help="Include defender size features (off by default — "
                             "they carry no measurable signal and produce a "
                             "large spurious effect at serving time)")
    parser.add_argument("--spatial-basis", action="store_true",
                        help="Include the radial spatial basis (off by default "
                             "— measured slightly harmful)")
    parser.add_argument("--split", action="store_true",
                        help="Train separate interior/perimeter models (old behaviour)")
    parser.add_argument("--calibration", default="recent",
                        choices=["recent", "oof", "none"],
                        help="recent: fit on the latest unseen data and adopt "
                             "only if it helps (default); oof: chronological "
                             "out-of-fold over the fit window; none: skip")
    parser.add_argument("--folds", type=int, default=3,
                        help="Chronological folds for out-of-fold calibration")
    parser.add_argument("--params", type=str, default=None,
                        help="Path to a tuned-parameter JSON")
    args = parser.parse_args()

    tuned = json.loads(Path(args.params).read_text()) if args.params else None

    train(
        seasons=[s for s in config.ALL_SEASONS if s >= args.from_season],
        name=args.name,
        use_creation=not args.no_creation,
        use_defender=not args.no_defender,
        hierarchical_split=args.split,
        use_defender_physicals=args.defender_physicals,
        use_spatial_basis=args.spatial_basis,
        params=tuned,
        calibration_folds=args.folds,
        calibration_mode=args.calibration,
    )
