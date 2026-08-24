"""
Evaluation — metrics that correspond to what the product actually does.

The old report gave log-loss, accuracy, and AUC against a zone-average
baseline. Three problems with that as the whole picture:

  - The zone-average baseline is very weak. Beating it by 6% sounds like
    skill and is mostly just "distance matters". `distance_zone_baseline`
    here is the honest comparison: zone plus a distance spline plus a season
    effect, which is roughly what any competent analyst would build in an
    afternoon. Lift over THAT is the real claim.

  - Accuracy on a base rate near 0.46 is close to meaningless, and thresholding
    a probability at 0.5 throws away the calibration the model works hard to
    achieve. It is reported here only because it is conventional.

  - Nothing measured the recommender. The product ranks court locations for a
    player; whether that ranking is any good is a separate empirical question
    from whether shot-level probabilities are sharp, and it went unasked.
    `zone_ranking_correlation` asks it.

Per-player residuals are the diagnostic that pays for itself: if the model is
well specified, each player's summed (actual − predicted) over a held-out
season should be noise around zero. Players who are consistently under- or
over-predicted tell you exactly which feature is missing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

# Points awarded, by zone. Used for expected-points calculations.
ZONE_POINTS = {
    "Restricted Area": 2, "In The Paint (Non-RA)": 2, "Mid-Range": 2,
    "Left Corner 3": 3, "Right Corner 3": 3, "Above the Break 3": 3,
}


def expected_calibration_error(y_true, y_pred, n_bins: int = 20) -> float:
    """
    Weighted mean gap between predicted probability and observed frequency.

    Equal-count bins, not equal-width. Equal-width bins put most of the mass
    in two or three buckets (shot probabilities cluster hard around the zone
    means) and let nearly-empty tail bins dominate the average.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0:
        return float("nan")

    quantiles = np.quantile(y_pred, np.linspace(0, 1, n_bins + 1))
    quantiles[0], quantiles[-1] = -np.inf, np.inf
    bins = np.digitize(y_pred, quantiles[1:-1])

    total = 0.0
    for b in np.unique(bins):
        mask = bins == b
        weight = mask.sum() / len(y_true)
        total += weight * abs(y_pred[mask].mean() - y_true[mask].mean())
    return float(total)


def reliability_table(y_true, y_pred, n_bins: int = 10) -> pd.DataFrame:
    """Per-bin predicted vs observed, for the calibration report and plots."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    quantiles = np.quantile(y_pred, np.linspace(0, 1, n_bins + 1))
    quantiles[0], quantiles[-1] = -np.inf, np.inf
    bins = np.digitize(y_pred, quantiles[1:-1])

    rows = []
    for b in np.unique(bins):
        mask = bins == b
        rows.append({
            "bin": int(b),
            "n": int(mask.sum()),
            "predicted": float(y_pred[mask].mean()),
            "actual": float(y_true[mask].mean()),
            "gap": float(y_pred[mask].mean() - y_true[mask].mean()),
        })
    return pd.DataFrame(rows)


def distance_zone_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame,
                           target_col: str = "shot_made") -> np.ndarray:
    """
    A serious baseline: logistic regression on zone indicators, a natural
    cubic spline in shot distance, and a season effect.

    This is what the model has to beat to have earned its complexity. Lift
    over a plain zone average flatters any model that knows shots get harder
    as you back up.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import SplineTransformer, StandardScaler

    def design(df):
        X = pd.DataFrame(index=df.index)
        X["shot_distance"] = df["shot_distance"].fillna(
            train_df["shot_distance"].median()
        )
        for zone in sorted(train_df["zone"].dropna().unique()):
            X[f"z_{zone}"] = (df["zone"] == zone).astype(float)
        # Season as an ordinal trend, capturing league-wide efficiency drift
        # without spending a coefficient per season.
        X["season_idx"] = df["season"].str.slice(0, 4).astype(float)
        return X

    pipeline = Pipeline([
        ("spline", SplineTransformer(n_knots=6, degree=3, include_bias=False)),
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000, C=1.0)),
    ])
    pipeline.fit(design(train_df), train_df[target_col])
    return pipeline.predict_proba(design(test_df))[:, 1]


def per_player_residuals(df: pd.DataFrame, y_pred, min_shots: int = 100,
                         target_col: str = "shot_made") -> pd.DataFrame:
    """
    Summed (actual − predicted) per player over the evaluation set, with a
    z-score against the binomial standard error of that sum.

    A well-specified model produces z-scores that look like draws from a
    standard normal. Systematic outliers are the model telling you which
    players it cannot explain — and, usually, which feature is missing.
    """
    work = pd.DataFrame({
        "player_id": df["player_id"].values,
        "actual": np.asarray(df[target_col], dtype=float),
        "pred": np.asarray(y_pred, dtype=float),
    })
    grouped = work.groupby("player_id").agg(
        shots=("actual", "size"),
        actual_makes=("actual", "sum"),
        expected_makes=("pred", "sum"),
        variance=("pred", lambda p: float(np.sum(p * (1.0 - p)))),
    ).reset_index()

    grouped = grouped[grouped["shots"] >= min_shots].copy()
    grouped["residual"] = grouped["actual_makes"] - grouped["expected_makes"]
    grouped["z"] = grouped["residual"] / np.sqrt(grouped["variance"].clip(lower=1e-9))
    return grouped.sort_values("z")


def zone_ranking_correlation(df: pd.DataFrame, y_pred, min_shots_per_zone: int = 15,
                             min_zones: int = 4, target_col: str = "shot_made") -> dict:
    """
    The recommender's actual KPI.

    For each player in the held-out season, rank the six zones by the model's
    mean predicted expected points, and rank them again by what the player
    actually produced. The Spearman correlation between those two rankings is
    what "does this engine give good advice" means, and nothing in the old
    pipeline measured it.

    Reported as the mean across players plus the share of players with a
    positive correlation — the latter is the more honest headline, since a
    few well-ranked stars can carry a mean.
    """
    work = pd.DataFrame({
        "player_id": df["player_id"].values,
        "zone": df["zone"].values,
        "actual": np.asarray(df[target_col], dtype=float),
        "pred": np.asarray(y_pred, dtype=float),
    })
    work["points"] = work["zone"].map(ZONE_POINTS).astype(float)
    work["actual_ep"] = work["actual"] * work["points"]
    work["pred_ep"] = work["pred"] * work["points"]

    grouped = work.groupby(["player_id", "zone"]).agg(
        n=("actual", "size"),
        actual_ep=("actual_ep", "mean"),
        pred_ep=("pred_ep", "mean"),
    ).reset_index()
    grouped = grouped[grouped["n"] >= min_shots_per_zone]

    correlations = []
    for _, grp in grouped.groupby("player_id"):
        if len(grp) < min_zones:
            continue
        # Constant predictions or outcomes make the correlation undefined
        # rather than zero; skip instead of poisoning the mean with NaN.
        if grp["pred_ep"].nunique() < 2 or grp["actual_ep"].nunique() < 2:
            continue
        rho, _ = stats.spearmanr(grp["pred_ep"], grp["actual_ep"])
        if not np.isnan(rho):
            correlations.append(rho)

    if not correlations:
        return {"mean_rho": float("nan"), "share_positive": float("nan"), "n_players": 0}

    arr = np.array(correlations)
    return {
        "mean_rho": float(arr.mean()),
        "median_rho": float(np.median(arr)),
        "share_positive": float((arr > 0).mean()),
        "n_players": int(len(arr)),
    }


def evaluate(df: pd.DataFrame, y_pred, y_true=None, target_col: str = "shot_made",
             label: str = "model") -> dict:
    """Headline metrics for one prediction vector."""
    y_true = np.asarray(df[target_col] if y_true is None else y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 1e-6, 1 - 1e-6)

    metrics = {
        "log_loss": float(log_loss(y_true, y_pred)),
        "brier": float(brier_score_loss(y_true, y_pred)),
        "auc": float(roc_auc_score(y_true, y_pred)),
        "ece": expected_calibration_error(y_true, y_pred),
        "accuracy": float(((y_pred >= 0.5).astype(int) == y_true).mean()),
        "n": int(len(y_true)),
    }

    ranking = zone_ranking_correlation(df, y_pred, target_col=target_col)
    metrics["zone_rank_rho"] = ranking["mean_rho"]
    metrics["zone_rank_share_positive"] = ranking["share_positive"]
    metrics["zone_rank_n_players"] = ranking["n_players"]

    metrics["_label"] = label
    return metrics


def per_zone_metrics(df: pd.DataFrame, y_pred, target_col: str = "shot_made",
                     min_shots: int = 100) -> pd.DataFrame:
    """Log-loss, ECE, and calibration gap broken out by zone."""
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 1e-6, 1 - 1e-6)
    y_true = np.asarray(df[target_col], dtype=float)

    rows = []
    for zone in sorted(df["zone"].dropna().unique()):
        mask = (df["zone"] == zone).values
        if mask.sum() < min_shots:
            continue
        rows.append({
            "zone": zone,
            "n": int(mask.sum()),
            "log_loss": float(log_loss(y_true[mask], y_pred[mask], labels=[0, 1])),
            "ece": expected_calibration_error(y_true[mask], y_pred[mask], n_bins=10),
            "predicted": float(y_pred[mask].mean()),
            "actual": float(y_true[mask].mean()),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["gap"] = out["predicted"] - out["actual"]
    return out


def print_report(metrics: dict, zone_table: pd.DataFrame | None = None,
                 residuals: pd.DataFrame | None = None) -> None:
    """Human-readable summary for the training console."""
    print(f"\n  {'Metric':<26} {'Value':>12}")
    print(f"  {'-'*26} {'-'*12}")
    for key in ("log_loss", "brier", "auc", "ece", "accuracy"):
        if key in metrics:
            print(f"  {key:<26} {metrics[key]:>12.4f}")
    if not np.isnan(metrics.get("zone_rank_rho", float("nan"))):
        print(f"  {'zone_rank_rho':<26} {metrics['zone_rank_rho']:>12.4f}")
        print(f"  {'  share positive':<26} {metrics['zone_rank_share_positive']:>12.1%}"
              f"   (n={metrics['zone_rank_n_players']} players)")

    if zone_table is not None and not zone_table.empty:
        print(f"\n  {'Zone':<24} {'n':>8} {'log-loss':>10} {'ECE':>8} {'gap':>8}")
        print(f"  {'-'*24} {'-'*8} {'-'*10} {'-'*8} {'-'*8}")
        for _, r in zone_table.iterrows():
            print(f"  {r['zone']:<24} {int(r['n']):>8,} {r['log_loss']:>10.4f} "
                  f"{r['ece']:>8.4f} {r['gap']:>+8.4f}")

    if residuals is not None and not residuals.empty:
        print(f"\n  Per-player residuals: mean z = {residuals['z'].mean():+.3f}, "
              f"sd = {residuals['z'].std():.3f}  (target: 0.0, 1.0)")
        extreme = (residuals["z"].abs() > 3).mean()
        print(f"  |z| > 3: {extreme:.1%} of players  (expect ~0.3% if well specified)")
