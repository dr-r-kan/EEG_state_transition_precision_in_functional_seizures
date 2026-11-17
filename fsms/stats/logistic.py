"""Numerical engine for the logistic-regression analysis: maximum likelihood fit, AUC, ROC, pseudo-R2.

The logistic model is fitted by Newton-Raphson with step halving, which is fast enough to be repeated thousands of
times in bootstrap and leave-one-out loops.  ``converged`` is False when the iteration limit is reached and
``separated`` flags (quasi-)complete separation, in which case maximum-likelihood estimates and Wald standard
errors are not meaningful (the likelihood-ratio statistics and the AUC remain usable).

The AUC is the Mann-Whitney U statistic divided by ``n_positive * n_negative``, using mid-ranks for ties.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit
from scipy.stats import rankdata

MAX_ITERATIONS = 100
LOG_LIKELIHOOD_TOLERANCE = 1e-10
SEPARATION_COEFFICIENT = 30.0


@dataclass(frozen=True)
class LogitFit:
    """Maximum-likelihood logistic fit; ``covariance`` is the inverse observed information."""

    beta: np.ndarray
    covariance: np.ndarray
    log_likelihood: float
    converged: bool
    separated: bool

    @property
    def se(self) -> np.ndarray:
        return np.sqrt(np.clip(np.diag(self.covariance), 0, None))


def log_likelihood(design: np.ndarray, y: np.ndarray, beta: np.ndarray) -> float:
    eta = design @ beta
    return float(np.sum(y * eta - np.logaddexp(0.0, eta)))


def null_log_likelihood(y: np.ndarray) -> float:
    """Log-likelihood of the intercept-only model."""
    n, positives = y.size, float(y.sum())
    if positives in (0.0, float(n)):
        return 0.0
    rate = positives / n
    return float(positives * np.log(rate) + (n - positives) * np.log1p(-rate))


def fit_logit(design: np.ndarray, y: np.ndarray) -> LogitFit:
    """Fit ``P(y = 1) = expit(design @ beta)``; ``design`` must contain the intercept column."""
    n_obs, k = design.shape
    nan_fit = LogitFit(np.full(k, np.nan), np.full((k, k), np.nan), np.nan, False, False)
    if not np.all(np.isfinite(design)) or y.min() == y.max():
        return nan_fit
    beta = np.zeros(k)
    current = log_likelihood(design, y, beta)
    converged = False
    for _ in range(MAX_ITERATIONS):
        p = expit(design @ beta)
        weights = p * (1.0 - p)
        information = (design * weights[:, None]).T @ design
        try:
            step = np.linalg.solve(information + 1e-12 * np.eye(k), design.T @ (y - p))
        except np.linalg.LinAlgError:
            break
        scale = 1.0
        while scale > 1e-10:
            candidate = log_likelihood(design, y, beta + scale * step)
            if candidate >= current - 1e-12:
                break
            scale /= 2.0
        else:
            break
        beta = beta + scale * step
        improvement, current = candidate - current, candidate
        if abs(improvement) < LOG_LIKELIHOOD_TOLERANCE and np.max(np.abs(scale * step)) < 1e-6:
            converged = True
            break
    p = expit(design @ beta)
    information = (design * (p * (1.0 - p))[:, None]).T @ design
    try:
        covariance = np.linalg.inv(information)
    except np.linalg.LinAlgError:
        covariance = np.full((k, k), np.nan)
    separated = bool(np.max(np.abs(beta)) > SEPARATION_COEFFICIENT or current > -1e-6 * n_obs)
    return LogitFit(beta, covariance, float(current), converged, separated)


def auc(score: np.ndarray, y: np.ndarray) -> float:
    """Area under the ROC curve of ``score`` for ``y`` in {0, 1} (Mann-Whitney U / (n1 n0))."""
    positive = y == 1
    n1, n0 = int(positive.sum()), int((~positive).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = rankdata(score)
    return float((ranks[positive].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def roc_curve(score: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """False- and true-positive rates for every distinct threshold (rule: predict 1 when score >= threshold)."""
    order = np.argsort(-score, kind="stable")
    sorted_score, sorted_y = score[order], y[order]
    true_positives = np.cumsum(sorted_y == 1)
    false_positives = np.cumsum(sorted_y == 0)
    last_of_ties = np.r_[np.diff(sorted_score) != 0, True]
    tpr = np.r_[0.0, true_positives[last_of_ties] / true_positives[-1]]
    fpr = np.r_[0.0, false_positives[last_of_ties] / false_positives[-1]]
    thresholds = np.r_[np.inf, sorted_score[last_of_ties]]
    return fpr, tpr, thresholds


def mcfadden_r2(log_lik: float, null_log_lik: float) -> float:
    return float(1.0 - log_lik / null_log_lik) if null_log_lik != 0 else float("nan")


def nagelkerke_r2(log_lik: float, null_log_lik: float, n_obs: int) -> float:
    cox_snell = 1.0 - np.exp(2.0 * (null_log_lik - log_lik) / n_obs)
    maximum = 1.0 - np.exp(2.0 * null_log_lik / n_obs)
    return float(cox_snell / maximum) if maximum > 0 else float("nan")
