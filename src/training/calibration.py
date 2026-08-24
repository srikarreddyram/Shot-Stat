"""
Probability calibration — fit on out-of-fold predictions.

The bug this replaces
---------------------
The previous implementation fit the isotonic calibrator on predictions the
model made about its OWN TRAINING ROWS:

    xgb_train_preds_int = xgb_interior.predict_proba(X_train_int)[:, 1]
    calibrator_int, _, _ = fit_calibrator(xgb_train_preds_int, y_train_int)

Holding out 15% of those rows inside `fit_calibrator` did not help: the model
had already seen all of them during fitting. A boosted tree ensemble is
sharply overconfident on rows it has memorized — its training-set predictions
are much closer to the truth than its test-set predictions at the same nominal
probability. So the calibrator learned "when the model says 0.72, reality is
0.72", which is true in-sample and false out-of-sample, and it then applied
that near-identity mapping to genuinely new shots that needed real correction.

The fix
-------
Calibrate on predictions made about data the model did not train on. Two
supported modes:

  out-of-fold (default) — the fit window is split into K chronological folds;
      a model is trained on the folds before each one and predicts it. Every
      fit row ends up with a prediction from a model that never saw it, so the
      calibrator sees the full data range with honest probabilities. Costs K
      extra fits.

  holdout — a dedicated season, excluded from fitting, is predicted once and
      used to fit the calibrator. Cheaper, but the calibrator only sees one
      season's worth of the probability range.

Isotonic regression is kept over Platt scaling. The miscalibration here is not
a monotone sigmoid distortion — it is zone-dependent and lumpy, especially at
the rim where the true rate saturates near 0.65 — and isotonic can follow that
where a two-parameter sigmoid cannot.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

# Predictions are clipped away from 0 and 1 so downstream log-loss stays finite
# even if a calibration bin happens to be pure.
_EPS = 1e-4


@dataclass
class Calibrator:
    """
    A fitted isotonic mapping plus a record of how it was fit.

    `method` and `n_calibration_rows` are stored because a calibrator fit
    in-sample and one fit out-of-fold are not interchangeable objects, and the
    difference is invisible once the mapping is serialized. Persisting the
    provenance means a loaded model can say how its probabilities were made.
    """
    iso: IsotonicRegression
    method: str
    n_calibration_rows: int

    def predict(self, raw_preds) -> np.ndarray:
        raw = np.asarray(raw_preds, dtype=float)
        return np.clip(self.iso.predict(raw), _EPS, 1 - _EPS)

    def to_dict(self) -> dict:
        """
        Serialize as plain knots rather than pickling the sklearn object.

        A joblib of an IsotonicRegression pins the sklearn version that wrote
        it; a pair of float arrays does not. Model artifacts outlive the
        environment that produced them, and a calibrator that fails to load
        after a routine dependency bump silently degrades every probability
        the API serves.
        """
        return {
            "x": np.asarray(self.iso.X_thresholds_, dtype=float).tolist(),
            "y": np.asarray(self.iso.y_thresholds_, dtype=float).tolist(),
            "method": self.method,
            "n_calibration_rows": self.n_calibration_rows,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "Calibrator":
        iso = IsotonicRegression(y_min=_EPS, y_max=1 - _EPS, out_of_bounds="clip")
        x = np.asarray(payload["x"], dtype=float)
        y = np.asarray(payload["y"], dtype=float)
        iso.fit(x, y)
        return cls(iso=iso, method=payload.get("method", "unknown"),
                   n_calibration_rows=int(payload.get("n_calibration_rows", 0)))


def _fit_isotonic(raw_preds, y_true, method: str) -> Calibrator:
    iso = IsotonicRegression(y_min=_EPS, y_max=1 - _EPS, out_of_bounds="clip")
    iso.fit(np.asarray(raw_preds, dtype=float), np.asarray(y_true, dtype=float))
    return Calibrator(iso=iso, method=method, n_calibration_rows=int(len(y_true)))


def out_of_fold_predictions(
    model_factory,
    X: pd.DataFrame,
    y: pd.Series,
    order: pd.Series | None = None,
    n_folds: int = 3,
    verbose: bool = True,
) -> np.ndarray:
    """
    Chronological out-of-fold predictions over the fit window.

    Folds are expanding-window, not shuffled K-fold: fold i trains on
    everything before it and predicts only forward. Shuffled folds would let a
    model calibrate using March to predict November of the same season, which
    is the same time-travel the point-in-time feature work removed.

    The first fold has no earlier data to train on and is left unpredicted;
    those rows are dropped from calibration by the NaN mask.

    `model_factory` must return a fresh, unfitted estimator each call.
    """
    n = len(X)
    if order is not None:
        sort_idx = np.argsort(np.asarray(order), kind="stable")
    else:
        sort_idx = np.arange(n)

    X_sorted = X.iloc[sort_idx]
    y_sorted = np.asarray(y)[sort_idx]

    preds_sorted = np.full(n, np.nan)
    bounds = np.linspace(0, n, n_folds + 1).astype(int)

    for i in range(1, n_folds):
        train_end = bounds[i]
        pred_start, pred_end = bounds[i], bounds[i + 1]
        if pred_end - pred_start < 1 or train_end < 1000:
            continue

        if verbose:
            print(f"      fold {i}/{n_folds - 1}: fit {train_end:,} → "
                  f"predict {pred_end - pred_start:,}")

        model = model_factory()
        model.fit(X_sorted.iloc[:train_end], y_sorted[:train_end], verbose=False)
        preds_sorted[pred_start:pred_end] = model.predict_proba(
            X_sorted.iloc[pred_start:pred_end]
        )[:, 1]

    # Undo the sort so the caller gets predictions aligned to its own rows.
    preds = np.full(n, np.nan)
    preds[sort_idx] = preds_sorted
    return preds


def fit_calibrator_oof(
    model_factory,
    X: pd.DataFrame,
    y: pd.Series,
    order: pd.Series | None = None,
    n_folds: int = 3,
    verbose: bool = True,
) -> Calibrator:
    """Fit a calibrator on chronological out-of-fold predictions."""
    if verbose:
        print(f"    → out-of-fold calibration ({n_folds} chronological folds)")
    oof = out_of_fold_predictions(model_factory, X, y, order=order,
                                  n_folds=n_folds, verbose=verbose)
    mask = ~np.isnan(oof)
    if mask.sum() < 1000:
        raise ValueError(
            f"only {mask.sum()} out-of-fold predictions available; "
            "too few to calibrate on"
        )
    return _fit_isotonic(oof[mask], np.asarray(y)[mask], method=f"oof-{n_folds}fold")


def fit_calibrator_holdout(raw_preds, y_true) -> Calibrator:
    """
    Fit on predictions over a holdout the model never trained on.

    The caller is responsible for that guarantee — this function cannot verify
    it, and getting it wrong silently reproduces the original bug.
    """
    return _fit_isotonic(raw_preds, y_true, method="holdout")


def calibration_report(y_true, raw_preds, calibrated_preds, n_bins: int = 10) -> pd.DataFrame:
    """Side-by-side reliability table for raw vs calibrated predictions."""
    from src.training.evaluate import reliability_table

    raw_table = reliability_table(y_true, raw_preds, n_bins=n_bins)
    cal_table = reliability_table(y_true, calibrated_preds, n_bins=n_bins)
    return raw_table.merge(cal_table, on="bin", suffixes=("_raw", "_cal"))


def save_calibration_plot(y_true, raw_preds, calibrated_preds, path,
                          title: str = "Calibration") -> bool:
    """Reliability diagram. Returns False if matplotlib is unavailable."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    from src.training.evaluate import reliability_table

    raw_table = reliability_table(y_true, raw_preds, n_bins=12)
    cal_table = reliability_table(y_true, calibrated_preds, n_bins=12)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.plot([0, 1], [0, 1], "--", color="#94a3b8", lw=1, label="perfect")
    ax.plot(raw_table["predicted"], raw_table["actual"], "o-",
            color="#ef4444", label="raw")
    ax.plot(cal_table["predicted"], cal_table["actual"], "o-",
            color="#22c55e", label="calibrated")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(path), dpi=150)
    plt.close(fig)
    return True
