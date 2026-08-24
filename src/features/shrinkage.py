"""
Empirical-Bayes shrinkage for small-sample shooting rates.

The problem this solves
-----------------------
A player who has taken 9 corner threes and made 5 does not shoot 55.6% from
the corner. Feeding that raw ratio to a model as `zone_efficiency` teaches it
to trust noise, and — because the old pipeline computed those rates over the
player's FULL season, including the very shot being predicted — it also leaked
the label into the feature.

The fix is to treat each rate as a posterior, not a ratio:

    p̂ = (makes + k·p_prior) / (attempts + k)

where `p_prior` is what we believed before seeing this player's attempts and
`k` is how many attempts' worth of evidence that belief is worth. With zero
attempts p̂ collapses to the prior; with a thousand it converges to the raw
rate. Nothing is ever undefined, and nothing has to be imputed.

Why `k` is estimated, not chosen
--------------------------------
`k` is the concentration (α+β) of a Beta prior fit to the actual spread of
true talent in the population. Estimating it by method of moments means the
data decides how much regression each zone needs: restricted-area rates are
tightly clustered across players and regress hard, above-the-break-3 rates
vary more between players and regress less. Hard-coding one constant for both
would over-shrink real shooters and under-shrink noisy ones.

This also makes `PositionPrior` unnecessary. That table existed to give a
player with zero NBA history *some* starting estimate, and had to be a special
case flagged through the whole stack. Shrinkage is that same idea applied
continuously: a rookie's first shot uses a pure prior, his 400th barely uses
it at all, and no consumer needs to branch on which case it is.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

# A prior fit from fewer than this many players is too unstable to trust;
# callers fall back to a coarser grouping (see fit_priors' `fallback`).
MIN_PLAYERS_FOR_PRIOR = 20

# Players below this many attempts are excluded when ESTIMATING the prior.
# Their observed rates are almost pure binomial noise, which inflates the
# apparent variance of true talent and drives the estimated k downward —
# i.e. low-volume players would make the model shrink everyone less, exactly
# backwards. They still RECEIVE shrinkage; they just don't get a vote in
# setting it.
MIN_ATTEMPTS_FOR_PRIOR = 25

# Bounds on the fitted concentration. Beyond these the method-of-moments
# estimate is being driven by degenerate variance (all players identical, or
# a single outlier) rather than by real spread in talent.
K_MIN, K_MAX = 5.0, 2000.0


@dataclass(frozen=True)
class BetaPrior:
    """
    A fitted Beta(α, β) prior over a population's true shooting rates.

    mean     — the population's talent-weighted average rate (α / (α+β))
    strength — α+β, the number of attempts this prior is "worth"
    n_players / n_attempts — the sample the prior was fit from, kept so a
               consumer can tell a well-grounded prior from a thin one
    """
    mean: float
    strength: float
    n_players: int = 0
    n_attempts: int = 0

    @property
    def alpha(self) -> float:
        return self.mean * self.strength

    @property
    def beta(self) -> float:
        return (1.0 - self.mean) * self.strength


def fit_beta_prior(makes: np.ndarray, attempts: np.ndarray) -> BetaPrior:
    """
    Fit a Beta prior to a population of (makes, attempts) pairs by method of
    moments, correcting for binomial sampling noise.

    The correction is the whole trick. The observed spread of per-player rates
    is inflated by sampling error:

        Var(p_observed) = Var(p_true) + E[ p(1-p) / n ]

    so subtracting the expected binomial variance recovers the spread of TRUE
    talent, which is what the prior is supposed to describe. Skipping the
    correction would attribute all the noise to real differences between
    players and produce a prior far too weak to shrink anything.
    """
    makes = np.asarray(makes, dtype=float)
    attempts = np.asarray(attempts, dtype=float)

    eligible = attempts >= MIN_ATTEMPTS_FOR_PRIOR
    m, n = makes[eligible], attempts[eligible]

    if len(n) < MIN_PLAYERS_FOR_PRIOR or n.sum() == 0:
        # Not enough population to fit a spread. Fall back to the pooled rate
        # with a deliberately weak strength — better than inventing variance.
        pooled = float(makes.sum() / attempts.sum()) if attempts.sum() > 0 else 0.5
        return BetaPrior(mean=pooled, strength=K_MIN,
                         n_players=int(eligible.sum()), n_attempts=int(attempts.sum()))

    p_obs = m / n

    # Attempt-weighted mean: a 900-attempt player should inform the population
    # mean more than a 30-attempt one.
    mu = float(np.average(p_obs, weights=n))

    var_obs = float(np.average((p_obs - mu) ** 2, weights=n))
    var_binomial = float(np.average(p_obs * (1.0 - p_obs) / n, weights=n))
    var_true = var_obs - var_binomial

    max_var = mu * (1.0 - mu)
    if var_true <= 0 or var_true >= max_var:
        # Observed spread is fully explained by sampling noise (or the
        # correction overshot). Either way there is no measurable talent
        # spread here, so shrink hard toward the pooled mean.
        k = K_MAX
    else:
        k = max_var / var_true - 1.0

    k = float(np.clip(k, K_MIN, K_MAX))
    return BetaPrior(mean=mu, strength=k,
                     n_players=int(len(n)), n_attempts=int(n.sum()))


def shrink(makes, attempts, prior: BetaPrior):
    """
    Apply a fitted prior to observed counts. Vectorized; accepts scalars,
    numpy arrays, or pandas Series and returns the matching type.

    This is the single function both the training path and the serving path
    call. They differ in how they obtain `makes`/`attempts` — a cumulative
    sum over prior games versus a live lookup — but never in how counts
    become a feature.
    """
    makes = np.nan_to_num(np.asarray(makes, dtype=float), nan=0.0)
    attempts = np.nan_to_num(np.asarray(attempts, dtype=float), nan=0.0)
    return (makes + prior.alpha) / (attempts + prior.strength)


def shrink_toward(makes, attempts, prior_mean, strength):
    """
    Shrink toward a per-row prior mean rather than one population constant.

    Used for the second level of the hierarchy, where each player's
    season-to-date rate regresses toward *their own* shrunk career rate
    instead of toward the league. `prior_mean` is an array aligned with
    `makes`/`attempts`; `strength` may be scalar or array.
    """
    makes = np.nan_to_num(np.asarray(makes, dtype=float), nan=0.0)
    attempts = np.nan_to_num(np.asarray(attempts, dtype=float), nan=0.0)
    prior_mean = np.asarray(prior_mean, dtype=float)
    strength = np.asarray(strength, dtype=float)
    return (makes + strength * prior_mean) / (attempts + strength)


def posterior_interval(makes, attempts, prior: BetaPrior, level: float = 0.90):
    """
    Credible interval on the shrunk rate, from the Beta posterior
    Beta(makes + α, attempts - makes + β).

    The recommender surfaces this so a projection built on 9 corner threes
    reads as the wide guess it actually is, instead of rendering with the same
    visual confidence as one built on 900 above-the-break attempts.
    """
    makes = np.nan_to_num(np.asarray(makes, dtype=float), nan=0.0)
    attempts = np.nan_to_num(np.asarray(attempts, dtype=float), nan=0.0)
    misses = np.maximum(attempts - makes, 0.0)

    a = makes + prior.alpha
    b = misses + prior.beta
    tail = (1.0 - level) / 2.0
    return stats.beta.ppf(tail, a, b), stats.beta.ppf(1.0 - tail, a, b)


def fit_priors(
    df: pd.DataFrame,
    group_cols: list[str],
    makes_col: str = "makes",
    attempts_col: str = "attempts",
    fallback: BetaPrior | None = None,
) -> dict[tuple, BetaPrior]:
    """
    Fit one Beta prior per group (e.g. per zone, or per zone × season).

    Groups too thin to fit on their own inherit `fallback`, so a caller can
    build a coarse league-wide prior first and pass it in as the backstop for
    fine-grained slices.

    Returns a dict keyed by the group tuple. Single-column groupings are still
    keyed by a 1-tuple, so lookup code never has to branch on key shape.
    """
    priors: dict[tuple, BetaPrior] = {}
    for key, grp in df.groupby(group_cols, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        prior = fit_beta_prior(grp[makes_col].values, grp[attempts_col].values)
        if prior.n_players < MIN_PLAYERS_FOR_PRIOR and fallback is not None:
            prior = fallback
        priors[key] = prior
    return priors
