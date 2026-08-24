"""
Hierarchical player and defender effects.

The problem this addresses
--------------------------
Per-player residuals over a held-out season have var(z) = 1.83, where a
correctly-specified model gives 1.0. That is 83% more player-level variation
than chance explains, with 2.2% of players beyond |z| > 3 against 0.3%
expected. The model is systematically wrong about specific players, and it says
so in a directly measurable way.

The features cannot fix this on their own. `zone_rate` and friends summarise a
player as a handful of shrunk rates, which is a lossy description of a shooter —
it cannot express "converts better than his percentages imply against set
defenses", or any other stable idiosyncrasy.

The approach
------------
A two-stage model, which is the standard structure for exactly this situation:

  stage 1  the gradient-boosted model, which already knows about location,
           context, defender quality and the shooter's own rates
  stage 2  a ridge-penalised linear logistic model on PLAYER and DEFENDER
           indicators, fit with stage 1's log-odds as a fixed OFFSET

The offset is what makes stage 2 a residual model rather than a second opinion.
Without it the indicators would learn each player's overall conversion rate,
which stage 1 already knows; with it they can only learn what stage 1 got
wrong. scikit-learn's LogisticRegression has no offset parameter, and XGBoost's linear
booster (which does, via `base_margin`) is both deprecated and badly behaved
here — on a synthetic check where one player carried a real +0.8 logit edge, it
recovered weights of order 1e-5. So the penalised logistic is solved directly
with L-BFGS. It is a small convex problem with a closed-form gradient, the
design matrix is sparse, and solving it outright removes any question about
whether the optimiser converged.

Stage 2 asks one question: after everything stage 1 knows, is this player
still systematically over- or under-predicted? The ridge penalty is what makes
the answer trustworthy — it is exactly the shrinkage the rest of this pipeline
applies to rates, in the form a linear model takes. A player with 40 shots gets
an effect near zero; one with 1,400 earns a real one, and the penalty strength
decides how much evidence "real" requires.

Why this is a separate stage rather than a feature
--------------------------------------------------
Handing player identity to the tree as a categorical would let it memorise
individuals with no shrinkage at all, which is how the defender-size features
came to encode 25 rows of noise as an 18-point effect. A penalised linear term
cannot do that: every effect is pulled toward zero by a single explicit,
tunable parameter, and the amount of pull is selected on held-out data rather
than assumed.

The effects are also worth reading directly — `shooter skill above expectation`
is a quantity the UI can show, and the extremes are a standing diagnostic of
what the feature set still misses.

Usage:
    python -m src.training.player_effects --model shot-quality-v5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from src.features.build import TARGET_COL, build_matrix
from src.features.spec import as_model_matrix
from src.training import evaluate as ev

MODEL_DIR = Path(config.PROJECT_ROOT) / "models"

# Below this many shots in the fit window a player gets no individual effect and
# falls into a shared "other" bucket. Not a statistical necessity — the ridge
# penalty would shrink them to nearly zero anyway — but it keeps the design
# matrix from carrying thousands of columns that all resolve to zero.
MIN_SHOTS_FOR_EFFECT = 200

# L2 penalties to try. LARGER lambda = stronger penalty = more shrinkage, so
# this grid runs the opposite direction from scikit-learn's C.
LAMBDA_GRID = [3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _build_design(df: pd.DataFrame, players: list[str], defenders: list[str]):
    """
    Sparse indicator matrix for player and defender identity.

    Two million rows by a few thousand columns, almost entirely zero — dense
    would be hundreds of gigabytes, sparse is a few tens of megabytes.
    """
    player_index = {p: i for i, p in enumerate(players)}
    defender_index = {d: len(players) + i for i, d in enumerate(defenders)}
    n_cols = len(players) + len(defenders)

    rows, cols = [], []
    for r, (pid, did) in enumerate(zip(df["player_id"].values,
                                       df.get("defender_id", pd.Series(index=df.index)).values)):
        j = player_index.get(pid)
        if j is not None:
            rows.append(r)
            cols.append(j)
        k = defender_index.get(did)
        if k is not None:
            rows.append(r)
            cols.append(k)

    data = np.ones(len(rows), dtype=np.float32)
    return sparse.csr_matrix(
        (data, (rows, cols)), shape=(len(df), n_cols), dtype=np.float32
    )


def fit_effects(model_name: str = "shot-quality-v9") -> dict:
    from src.inference.recommender import ShotRecommender

    rec = ShotRecommender(model_name=model_name)
    seasons = [s for s in config.ALL_SEASONS if s >= "2016-17"]
    meta = rec.metadata
    fit_seasons, val_season, test_season = (
        meta["fit_seasons"], meta["val_season"], meta["test_season"]
    )

    df, _, _ = build_matrix(seasons, prior_through_season=val_season, verbose=False)

    # The defender mixture has no single id, so defender effects use the
    # possession-dominant defender where one is recorded on the shot.
    if "defender_id" not in df.columns:
        from src.db.database import get_engine
        link = pd.read_sql(
            "SELECT shot_id, defender_id FROM shots WHERE defender_id IS NOT NULL",
            get_engine(),
        )
        df = df.merge(link, on="shot_id", how="left")

    fit_df = df[df["season"].isin(fit_seasons)].copy()
    val_df = df[df["season"] == val_season].copy()
    test_df = df[df["season"] == test_season].copy()

    print(f"\n{'='*62}")
    print("  HIERARCHICAL PLAYER / DEFENDER EFFECTS")
    print(f"  stage 1: {model_name}")
    print(f"  fit {len(fit_df):,} | val {len(val_df):,} | test {len(test_df):,}")
    print(f"{'='*62}")

    # Stage 1 predictions, used as a fixed offset.
    for frame in (fit_df, val_df, test_df):
        frame["_stage1"] = rec._predict(as_model_matrix(frame, rec.feature_cols))

    counts = fit_df["player_id"].value_counts()
    players = sorted(counts[counts >= MIN_SHOTS_FOR_EFFECT].index)
    dcounts = fit_df["defender_id"].value_counts(dropna=True)
    defenders = sorted(dcounts[dcounts >= MIN_SHOTS_FOR_EFFECT].index)
    print(f"  {len(players):,} players and {len(defenders):,} defenders "
          f"with >= {MIN_SHOTS_FOR_EFFECT} shots")

    X_fit = _build_design(fit_df, players, defenders)
    X_val = _build_design(val_df, players, defenders)
    X_test = _build_design(test_df, players, defenders)

    base_val = ev.evaluate(val_df, val_df["_stage1"])["log_loss"]
    print(f"\n  stage 1 alone, val log-loss {base_val:.5f}")
    print("  selecting the L2 penalty on the validation season:")

    y_fit = fit_df[TARGET_COL].values.astype(np.float64)
    offset_fit = _logit(fit_df["_stage1"].values)

    def _fit(lam: float) -> np.ndarray:
        """
        Minimise L2-penalised logistic loss with a fixed offset.

            loss(w) = sum_i softplus(o_i + x_i.w) - y_i (o_i + x_i.w)
                      + (lam / 2) ||w||^2

        The gradient is X^T (sigmoid(o + Xw) - y) + lam*w, so L-BFGS converges
        in a few dozen iterations even at two million rows.
        """
        n_features = X_fit.shape[1]

        def objective(w):
            margin = offset_fit + X_fit @ w
            # softplus, computed stably for large |margin|
            loss = np.sum(np.logaddexp(0.0, margin) - y_fit * margin)
            prob = 1.0 / (1.0 + np.exp(-margin))
            grad = X_fit.T @ (prob - y_fit) + lam * w
            return loss + 0.5 * lam * float(w @ w), grad

        result = minimize(
            objective, np.zeros(n_features), jac=True, method="L-BFGS-B",
            options={"maxiter": 200, "ftol": 1e-10},
        )
        return result.x

    def _apply(weights, frame, X):
        margin = _logit(frame["_stage1"].values) + X @ weights
        return 1.0 / (1.0 + np.exp(-margin))

    best = (None, base_val, None)
    for lam in LAMBDA_GRID:
        weights = _fit(lam)
        val_pred = _apply(weights, val_df, X_val)
        ll = ev.evaluate(val_df, val_pred)["log_loss"]
        flag = ""
        if ll < best[1]:
            best = (lam, ll, weights)
            flag = "  <-- best"
        print(f"    lambda={lam:<7} val log-loss {ll:.5f}  "
              f"|w|max {np.abs(weights).max():.4f}{flag}")

    lam, val_ll, weights = best
    if weights is None:
        print("\n  No penalty strength improved on stage 1. Effects not adopted.")
        return {"adopted": False, "val_log_loss": base_val}

    test_pred = _apply(weights, test_df, X_test)

    before = ev.evaluate(test_df, test_df["_stage1"])
    after = ev.evaluate(test_df, test_pred)

    res_before = ev.per_player_residuals(test_df, test_df["_stage1"])
    res_after = ev.per_player_residuals(test_df, test_pred)

    print(f"\n{'='*62}")
    print(f"  TEST SEASON {test_season}")
    print(f"{'='*62}")
    print(f"  {'metric':<22} {'stage 1':>10} {'+ effects':>11}")
    print(f"  {'-'*22} {'-'*10} {'-'*11}")
    for key in ("log_loss", "auc", "ece", "zone_rank_rho"):
        print(f"  {key:<22} {before[key]:>10.4f} {after[key]:>11.4f}")
    print(f"  {'residual var(z)':<22} {res_before['z'].var():>10.2f} "
          f"{res_after['z'].var():>11.2f}    (target 1.00)")

    coefs = weights
    effects = pd.DataFrame({
        "player_id": players + defenders,
        "kind": ["shooter"] * len(players) + ["defender"] * len(defenders),
        "effect": coefs[: len(players) + len(defenders)],
    })

    payload = {
        "adopted": True,
        "lambda": lam,
        "stage1_model": model_name,
        "min_shots": MIN_SHOTS_FOR_EFFECT,
        "val_log_loss": val_ll,
        "test_before": {k: v for k, v in before.items() if not k.startswith("_")},
        "test_after": {k: v for k, v in after.items() if not k.startswith("_")},
        "effects": {
            row["player_id"]: {"kind": row["kind"], "effect": float(row["effect"])}
            for _, row in effects.iterrows()
        },
    }
    out = MODEL_DIR / f"player_effects_{model_name}.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"\n  ✓ {out}")

    names = pd.read_sql(
        f"SELECT DISTINCT player_id, name FROM players WHERE season='{test_season}'",
        rec.engine,
    )
    shooters = effects[effects.kind == "shooter"].merge(names, on="player_id", how="left")
    print("\n  Largest positive shooter effects (better than the features imply):")
    for _, r in shooters.nlargest(6, "effect").iterrows():
        print(f"    {str(r['name'])[:26]:<26} {r.effect:+.4f}")
    print("  Largest negative:")
    for _, r in shooters.nsmallest(6, "effect").iterrows():
        print(f"    {str(r['name'])[:26]:<26} {r.effect:+.4f}")

    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit hierarchical player/defender effects.")
    parser.add_argument("--model", default="shot-quality-v9")
    args = parser.parse_args()
    fit_effects(model_name=args.model)
