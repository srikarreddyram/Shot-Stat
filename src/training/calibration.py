"""
Probability Calibration — Makes model predictions match real-world FG percentages.

XGBoost's raw probabilities are often poorly calibrated (e.g., predicts 88% when the
real FG% is 73%). Isotonic Regression learns a monotonic mapping from raw predictions
to calibrated probabilities using a held-out calibration set.

Usage:
    # During training (called by train_baseline.py):
    from src.training.calibration import fit_calibrator, apply_calibration

    calibrator = fit_calibrator(raw_preds, y_true)
    calibrated = apply_calibration(calibrator, raw_preds)

    # Verification:
    python -m src.training.calibration --verify
"""
import sys
from pathlib import Path

import numpy as np
import joblib
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def fit_calibrator(
    raw_preds: np.ndarray,
    y_true: np.ndarray,
    calibration_fraction: float = 0.15,
    random_state: int = 42,
) -> tuple:
    """
    Fit an isotonic regression calibrator on a held-out fraction of the data.

    Args:
        raw_preds: Raw model probabilities (from XGBoost predict_proba)
        y_true: Actual outcomes (0/1)
        calibration_fraction: Fraction of data to hold out for calibration
        random_state: Random seed for reproducibility

    Returns:
        (calibrator, cal_indices, eval_indices):
            - calibrator: fitted IsotonicRegression object
            - cal_indices: indices used for calibration fitting
            - eval_indices: remaining indices for evaluation
    """
    n = len(raw_preds)
    indices = np.arange(n)

    # Split into calibration set and evaluation set
    eval_idx, cal_idx = train_test_split(
        indices, test_size=calibration_fraction, random_state=random_state
    )

    cal_preds = raw_preds[cal_idx]
    cal_true = y_true[cal_idx]

    # Fit isotonic regression: learns the monotonic mapping raw → calibrated
    calibrator = IsotonicRegression(
        y_min=0.01,  # avoid exact 0
        y_max=0.99,  # avoid exact 1
        out_of_bounds="clip",
    )
    calibrator.fit(cal_preds, cal_true)

    return calibrator, cal_idx, eval_idx


def apply_calibration(calibrator: IsotonicRegression, raw_preds: np.ndarray) -> np.ndarray:
    """Apply the fitted calibrator to raw model predictions."""
    return calibrator.predict(raw_preds)


def save_calibrator(calibrator: IsotonicRegression, model_dir: str, version: str):
    """Save the calibrator alongside the model."""
    path = Path(model_dir) / f"calibrator_{version}.joblib"
    joblib.dump(calibrator, path)
    print(f"  ✓ Saved calibrator to {path}")


def load_calibrator(model_dir: str, version: str) -> IsotonicRegression:
    """Load a previously saved calibrator."""
    path = Path(model_dir) / f"calibrator_{version}.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


def print_calibration_report(
    raw_preds: np.ndarray,
    calibrated_preds: np.ndarray,
    y_true: np.ndarray,
    zones: np.ndarray = None,
):
    """
    Print a calibration report comparing raw vs calibrated predictions.

    Shows how well the predicted probabilities match actual make rates
    across decile bins and (optionally) per zone.
    """
    from sklearn.metrics import log_loss, brier_score_loss

    print(f"\n{'='*60}")
    print(f"  CALIBRATION REPORT")
    print(f"{'='*60}")

    # Overall metrics
    raw_ll = log_loss(y_true, raw_preds)
    cal_ll = log_loss(y_true, calibrated_preds)
    raw_brier = brier_score_loss(y_true, raw_preds)
    cal_brier = brier_score_loss(y_true, calibrated_preds)

    print(f"\n  {'Metric':<25} {'Raw':>10} {'Calibrated':>12} {'Change':>10}")
    print(f"  {'-'*25} {'-'*10} {'-'*12} {'-'*10}")
    print(f"  {'Log-Loss':<25} {raw_ll:>10.4f} {cal_ll:>12.4f} {(cal_ll-raw_ll):>+10.4f}")
    print(f"  {'Brier Score':<25} {raw_brier:>10.4f} {cal_brier:>12.4f} {(cal_brier-raw_brier):>+10.4f}")

    # Decile calibration: bin predictions into 10 groups and compare predicted vs actual
    print(f"\n  Decile Calibration (predicted → actual):")
    print(f"  {'Bin':>6} {'Pred Avg':>10} {'Actual FG%':>12} {'Gap':>8} {'Count':>8}")
    print(f"  {'-'*6} {'-'*10} {'-'*12} {'-'*8} {'-'*8}")

    # Use calibrated predictions for binning
    bin_edges = np.linspace(0, 1, 11)
    for i in range(10):
        mask = (calibrated_preds >= bin_edges[i]) & (calibrated_preds < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        pred_avg = calibrated_preds[mask].mean()
        actual_avg = y_true[mask].mean()
        gap = pred_avg - actual_avg
        count = mask.sum()
        icon = "✅" if abs(gap) < 0.03 else "⚠️" if abs(gap) < 0.05 else "❌"
        bin_label = f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}"
        print(f"  {icon} {bin_label:>5} {pred_avg:>10.3f} {actual_avg:>12.3f} {gap:>+8.3f} {count:>8,}")

    # Per-zone calibration (if zones provided)
    if zones is not None:
        print(f"\n  Per-Zone Calibration:")
        print(f"  {'Zone':<30} {'Pred Avg':>10} {'Actual FG%':>12} {'Gap':>8}")
        print(f"  {'-'*30} {'-'*10} {'-'*12} {'-'*8}")

        for zone in sorted(np.unique(zones)):
            mask = zones == zone
            if mask.sum() < 100:
                continue
            pred_avg = calibrated_preds[mask].mean()
            actual_avg = y_true[mask].mean()
            gap = pred_avg - actual_avg
            icon = "✅" if abs(gap) < 0.02 else "⚠️"
            print(f"  {icon} {zone:<28} {pred_avg:>10.3f} {actual_avg:>12.3f} {gap:>+8.3f}")

    print()


def save_calibration_plot(
    raw_preds: np.ndarray,
    calibrated_preds: np.ndarray,
    y_true: np.ndarray,
    model_dir: str,
    version: str,
):
    """Save a reliability diagram (calibration curve) as a PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.calibration import calibration_curve

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: calibration curves
        ax = axes[0]
        for preds, label, color in [
            (raw_preds, "Raw XGBoost", "#f97316"),
            (calibrated_preds, "Calibrated", "#22c55e"),
        ]:
            prob_true, prob_pred = calibration_curve(y_true, preds, n_bins=20, strategy="uniform")
            ax.plot(prob_pred, prob_true, "s-", label=label, color=color, linewidth=2)

        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Actual FG%")
        ax.set_title(f"Calibration Curve ({version})")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Right: prediction distributions
        ax = axes[1]
        ax.hist(raw_preds, bins=50, alpha=0.5, label="Raw", color="#f97316", density=True)
        ax.hist(calibrated_preds, bins=50, alpha=0.5, label="Calibrated", color="#22c55e", density=True)
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Density")
        ax.set_title("Prediction Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = Path(model_dir) / f"calibration_curve_{version}.png"
        fig.savefig(str(path), dpi=150)
        plt.close(fig)
        print(f"  ✓ Saved calibration plot to {path}")
    except Exception as e:
        print(f"  ⚠ Could not save calibration plot: {e}")


# ── CLI entry point ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true", help="Verify calibration quality")
    parser.add_argument("--version", type=str, default="v2", help="Model version to verify")
    args = parser.parse_args()

    if args.verify:
        import xgboost as xgb
        import pandas as pd

        from src.training.feature_engineering import build_training_matrix, get_feature_columns, TARGET_COL
        import config

        print("Loading model and data for calibration verification...")
        seasons = [s for s in config.ALL_SEASONS if s >= "2016-17"]
        df = build_training_matrix(seasons, use_defender=True)
        feature_cols = get_feature_columns(df)

        test_df = df[df["season"] == seasons[-1]].copy()
        X_test = test_df[feature_cols]
        y_test = test_df[TARGET_COL].values

        model = xgb.XGBClassifier()
        model.load_model(f"models/xgb_{args.version}.json")
        raw_preds = model.predict_proba(X_test)[:, 1]

        calibrator = load_calibrator("models", args.version)
        if calibrator is None:
            print("No calibrator found — fitting one now...")
            calibrator, _, _ = fit_calibrator(raw_preds, y_test)

        calibrated = apply_calibration(calibrator, raw_preds)
        print_calibration_report(raw_preds, calibrated, y_test, zones=test_df["zone"].values)
        save_calibration_plot(raw_preds, calibrated, y_test, "models", args.version)
    else:
        print("Use --verify to check calibration quality")
        print("Calibration is automatically applied during training (train_baseline.py)")
