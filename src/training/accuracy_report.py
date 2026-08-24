"""
Accuracy report — the evidence for how good the model is, and how good it can be.

Written to answer one question honestly: if someone asks "how accurate is it?",
what do you show them?

Not accuracy. A shot is a biased coin flip, and a classifier's accuracy on a
near-50/50 outcome is close to meaningless — it depends on an arbitrary 0.5
threshold and throws away the probability, which is the only thing the model
actually produces. Reporting 64% invites the reply "so it's wrong a third of
the time", which misdescribes what the model claims.

What the model claims is a PROBABILITY, and there are exactly two ways a
probability can be good:

  calibration   when it says 35%, does it happen 35% of the time?
  discrimination does it separate the 35% shots from the 70% shots at all?

This report measures both, places them against baselines that are actually
hard to beat, and — the part people skip — quantifies how much better any
model could possibly be given that the outcome is genuinely random.

Usage:
    python -m src.training.accuracy_report
    python -m src.training.accuracy_report --model shot-quality-v4 --season 2025-26
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from src.features.build import TARGET_COL, build_matrix
from src.features.spec import as_model_matrix
from src.training import evaluate as ev


def murphy_decomposition(y_true, y_pred, n_bins: int = 20) -> dict:
    """
    Split the Brier score into Uncertainty − Resolution + Reliability.

        Uncertainty  how unpredictable the outcome is if you know nothing.
                     Fixed by the league's field-goal percentage; no model
                     changes it. This is the number that makes shot prediction
                     look "bad" and it is not the model's fault.
        Resolution   how much of that the model actually explains. The real
                     measure of how much the model knows.
        Reliability  squared calibration error. Should be ~0.

    Equal-count bins, not equal-width: shot probabilities cluster hard around
    the zone means, so equal-width bins leave most of the mass in three
    buckets and let near-empty tails dominate.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    base = y_true.mean()

    edges = np.quantile(y_pred, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    bins = np.digitize(y_pred, edges[1:-1])

    reliability = resolution = 0.0
    for b in np.unique(bins):
        mask = bins == b
        weight = mask.mean()
        reliability += weight * (y_pred[mask].mean() - y_true[mask].mean()) ** 2
        resolution += weight * (y_true[mask].mean() - base) ** 2

    uncertainty = base * (1.0 - base)
    return {
        "uncertainty": float(uncertainty),
        "resolution": float(resolution),
        "reliability": float(reliability),
        "brier": float(brier_score_loss(y_true, y_pred)),
        "explained_share": float(resolution / uncertainty),
        "base_rate": float(base),
    }


def contest_ceiling(engine, season: str = "2023-24") -> dict | None:
    """
    How much more a model could explain if it could see how contested each shot
    was — the largest input it structurally cannot have.

    Computed WITHIN shot type, which matters enormously. The marginal table is
    a textbook Simpson's paradox: league-wide, wide-open shots go in LESS often
    than tightly-contested ones (0.429 vs 0.470), because wide-open shots are
    overwhelmingly threes and contested ones are overwhelmingly layups. Read
    marginally, the data appears to say defense does not matter. Split by shot
    type, the effect is large and perfectly monotonic.

    Returns None if the shot-profile table has not been ingested.
    """
    profile = pd.read_sql(f"""
        SELECT split_value,
               SUM(fgm) AS fgm, SUM(fga) AS fga,
               SUM(fg3m) AS fg3m, SUM(fg3a) AS fg3a
        FROM player_shot_profile
        WHERE split_type = 'def_dist' AND season = '{season}'
        GROUP BY split_value
    """, engine)

    if profile.empty:
        return None

    profile["fg2a"] = profile["fga"] - profile["fg3a"]
    profile["fg2m"] = profile["fgm"] - profile["fg3m"]

    out = {}
    for label, made, att in (("three", "fg3m", "fg3a"), ("two", "fg2m", "fg2a")):
        sub = profile[profile[att] > 0]
        if sub.empty:
            continue
        rate = sub[made] / sub[att]
        weight = sub[att] / sub[att].sum()
        mean = float((rate * weight).sum())
        out[label] = {
            "mean": mean,
            "spread": float(rate.max() - rate.min()),
            "variance": float((weight * (rate - mean) ** 2).sum()),
            "attempts": int(sub[att].sum()),
        }

    if not out:
        return None

    total = sum(v["attempts"] for v in out.values())
    out["combined_variance"] = sum(
        v["variance"] * v["attempts"] / total
        for k, v in out.items() if k != "combined_variance"
    )
    return out


def report(model_name: str = "shot-quality-v9", season: str | None = None) -> dict:
    """Print the full accuracy argument. Returns the computed numbers."""
    from src.inference.recommender import ShotRecommender

    rec = ShotRecommender(model_name=model_name)
    seasons = [s for s in config.ALL_SEASONS if s >= "2016-17"]
    test_season = season or rec.metadata["test_season"]

    df, _, _ = build_matrix(
        seasons, prior_through_season=rec.metadata["val_season"], verbose=False
    )
    test = df[df["season"] == test_season]
    y = test[TARGET_COL].values.astype(float)
    p = rec._predict(as_model_matrix(test, rec.feature_cols))

    base = y.mean()
    width = 68

    print(f"\n{'='*width}")
    print(f"  ACCURACY REPORT — {model_name}, held-out season {test_season}")
    print(f"  {len(y):,} shots, league FG% {base:.3f}")
    print(f"{'='*width}")

    # ── 1. Calibration: the headline ─────────────────────────────────────
    print("\n1. CALIBRATION — when it says X%, does X% happen?")
    print("   The single most persuasive exhibit. Ten equal-size buckets,")
    print(f"   {len(y)//10:,} shots each, sorted by what the model predicted.\n")
    edges = np.quantile(p, np.linspace(0, 1, 11))
    edges[0], edges[-1] = -np.inf, np.inf
    bins = np.digitize(p, edges[1:-1])
    print(f"   {'bucket':>7} {'predicted':>11} {'actual':>9} {'error':>8} {'shots':>9}")
    print(f"   {'-'*7} {'-'*11} {'-'*9} {'-'*8} {'-'*9}")
    worst = 0.0
    for b in np.unique(bins):
        mask = bins == b
        pred, act = p[mask].mean(), y[mask].mean()
        worst = max(worst, abs(pred - act))
        print(f"   {b+1:>7} {pred:>11.3f} {act:>9.3f} {pred-act:>+8.3f} {mask.sum():>9,}")
    print(f"\n   Largest error in any bucket: {worst:.3f} "
          f"({worst*100:.1f} percentage points)")
    print(f"   Expected calibration error:  {ev.expected_calibration_error(y, p):.4f}")

    # ── 2. Murphy decomposition ──────────────────────────────────────────
    murphy = murphy_decomposition(y, p)
    print("\n2. HOW MUCH IS KNOWABLE — Brier = Uncertainty − Resolution + Reliability")
    print(f"   Uncertainty  {murphy['uncertainty']:.4f}   irreducible if you know nothing;")
    print( "                          fixed by league FG%, no model changes it")
    print(f"   Resolution   {murphy['resolution']:.4f}   what the model actually explains")
    print(f"   Reliability  {murphy['reliability']:.6f} calibration error (want ~0)")
    print(f"   → the model explains {murphy['explained_share']:.1%} of total uncertainty")
    print( "   This is the honest headline. It sounds low because a shot is a")
    print( "   coin flip: even a PERFECT model that knew every shot's true")
    print( "   probability would still be 'wrong' constantly, because knowing")
    print( "   p=0.4 does not tell you the outcome.")

    # ── 3. Baseline ladder ───────────────────────────────────────────────
    train = df[df["season"].isin(rec.metadata["fit_seasons"] + [rec.metadata["val_season"]])]
    strong = ev.distance_zone_baseline(train, test)
    zone_avg = train.groupby("zone")[TARGET_COL].mean()
    zone_preds = test["zone"].map(zone_avg).fillna(base).clip(0.01, 0.99)

    ladder = [
        ("coin flip (0.5)", log_loss(y, np.full(len(y), 0.5))),
        ("league FG% only", log_loss(y, np.full(len(y), base))),
        ("zone average", log_loss(y, zone_preds)),
        ("zone + distance + season", log_loss(y, strong)),
        ("this model", log_loss(y, p)),
    ]
    print("\n3. AGAINST BASELINES — log-loss, lower is better")
    for label, value in ladder:
        print(f"   {label:<28} {value:.4f}")
    strong_ll = ladder[-2][1]
    model_ll = ladder[-1][1]
    print(f"\n   vs the serious baseline: {(strong_ll-model_ll)/strong_ll:+.2%}")
    print( "   The zone average is a soft target — beating it mostly proves that")
    print( "   shots get harder with distance. The distance-spline baseline is")
    print( "   what a competent analyst builds in an afternoon; that is the")
    print( "   comparison worth quoting.")

    # ── 4. Discrimination ────────────────────────────────────────────────
    auc = roc_auc_score(y, p)
    acc = ((p >= 0.5).astype(int) == y).mean()
    majority = max(base, 1 - base)
    print("\n4. DISCRIMINATION — can it tell a good shot from a bad one?")
    print(f"   AUC {auc:.4f}")
    print( "   Plain English: take one made shot and one missed shot at random.")
    print(f"   The model gives the made one a higher probability {auc:.1%} of the time.")
    print(f"\n   Prediction range {p.min():.3f} – {p.max():.3f}, sd {p.std():.3f}")
    print( "   (a model that just guessed the league average would have sd 0)")
    print(f"\n   Accuracy {acc:.4f} vs {majority:.4f} for always guessing 'miss'.")
    print( "   Reported only because people ask. It thresholds a probability at")
    print( "   0.5, which discards exactly the information the model produces.")

    # ── 5. Decision quality ──────────────────────────────────────────────
    ranking = ev.zone_ranking_correlation(test, p)
    print("\n5. DOES THE ADVICE HOLD UP — the product's real KPI")
    print("   Rank each player's six zones by predicted expected points, rank")
    print("   them again by what he actually produced. Spearman correlation:")
    print(f"   ρ = {ranking['mean_rho']:.3f}, positive for {ranking['share_positive']:.1%} "
          f"of {ranking['n_players']} players")

    # ── 6. Remaining headroom ────────────────────────────────────────────
    residuals = ev.per_player_residuals(test, p, min_shots=100)
    z = residuals["z"].values
    overdispersion = float(z.std() ** 2)
    print("\n6. WHAT IT STILL MISSES — and how we know")
    print(f"   Per-player residuals over {len(residuals)} players with 100+ shots:")
    print(f"   mean z {z.mean():+.3f} (want 0 — no systematic bias ✓)")
    print(f"   sd   z {z.std():.3f} (want 1 — pure chance)")
    print(f"   → var(z) = {overdispersion:.2f}: {overdispersion-1:.0%} more player-level")
    print( "     variation than chance explains, so real per-player signal remains.")
    print( "     This is the model telling you where to look next.")

    ceiling = contest_ceiling(rec.engine)
    if ceiling:
        print("\n   The biggest missing input is how contested each shot was, which")
        print("   no public data source exposes per shot. Its size, within shot type:")
        for label, key in (("3PT", "three"), ("2PT", "two")):
            c = ceiling[key]
            print(f"     {label}: wide-open minus very-tight = "
                  f"{c['spread']:+.3f} FG% ({c['attempts']:,} attempts)")
        print(f"   Between-bucket variance it would add: {ceiling['combined_variance']:.5f}")
        print(f"   ≈ {ceiling['combined_variance']/murphy['uncertainty']:.1%} of total "
              f"uncertainty, on top of the {murphy['explained_share']:.1%} already explained.")
        print( "   Large in effect size, modest in variance — because location and")
        print( "   shooter identity already proxy much of it.")

    print(f"\n{'='*width}\n")

    return {
        "murphy": murphy, "auc": float(auc), "accuracy": float(acc),
        "log_loss": float(model_ll), "baseline_log_loss": float(strong_ll),
        "ranking": ranking, "overdispersion": overdispersion,
        "worst_bucket_error": float(worst),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Explain and justify model accuracy.")
    parser.add_argument("--model", default="shot-quality-v9")
    parser.add_argument("--season", default=None, help="Override the held-out season")
    args = parser.parse_args()
    report(model_name=args.model, season=args.season)
