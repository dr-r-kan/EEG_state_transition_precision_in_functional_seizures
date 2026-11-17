"""Shared inference primitives: Holm adjustment, Welch / paired / one-sample t-tests, Hedges' g, Mann-Whitney.

All functions return flat ``dict`` records of plain floats so that they can be stacked into tables.  Tests that
cannot be computed (too few observations, zero variance) return NaN for every statistic rather than raising.

Effect sizes
------------
* Between groups: Hedges' g = J(df) * (mean1 - mean2) / s_pooled, with the exact small-sample correction
  J(df) = Gamma(df/2) / (sqrt(df/2) * Gamma((df-1)/2)) and df = n1 + n2 - 2.
* Paired / one-sample: dz = mean(d) / sd(d); ``hedges_g`` is dz multiplied by J(n - 1).
* Confidence intervals of g use the large-sample standard error
  sqrt((n1 + n2) / (n1 n2) + g^2 / (2 (n1 + n2 - 2))) (paired: sqrt(1/n + g^2 / (2n))); they are approximate.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import stats
from scipy.special import gammaln

CONFIDENCE = 0.95
_NAN = float("nan")


def holm(p_values: Sequence[float]) -> np.ndarray:
    """Holm step-down adjusted p-values.

    Non-finite entries are left as NaN and are not counted in the family size.  Adjusted values are monotone
    non-decreasing in the order of the raw p-values and capped at 1.
    """
    raw = np.asarray(p_values, dtype=float)
    adjusted = np.full(raw.shape, np.nan)
    valid = np.flatnonzero(np.isfinite(raw))
    if valid.size == 0:
        return adjusted
    order = valid[np.argsort(raw[valid], kind="stable")]
    scaled = (valid.size - np.arange(valid.size)) * raw[order]
    adjusted[order] = np.minimum(1.0, np.maximum.accumulate(scaled))
    return adjusted


def hedges_j(df: float) -> float:
    """Exact small-sample bias correction factor for Cohen's d with ``df`` degrees of freedom."""
    if not df > 1:
        return _NAN
    return float(np.exp(gammaln(df / 2.0) - gammaln((df - 1.0) / 2.0)) / np.sqrt(df / 2.0))


def finite(values: Sequence[float]) -> np.ndarray:
    """The finite entries of ``values`` as a float array."""
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def _empty_two_sample(n1: int, n2: int) -> dict[str, float]:
    keys = ("mean_1", "mean_2", "sd_1", "sd_2", "estimate", "se", "ci_low", "ci_high", "t", "df", "p",
            "hedges_g", "hedges_g_ci_low", "hedges_g_ci_high")
    return {"n_1": n1, "n_2": n2, **{key: _NAN for key in keys}}


def welch_test(first: Sequence[float], second: Sequence[float]) -> dict[str, float]:
    """Welch two-sample t-test of ``first`` minus ``second`` with confidence interval and Hedges' g."""
    x, y = finite(first), finite(second)
    n1, n2 = x.size, y.size
    result = _empty_two_sample(n1, n2)
    if n1 < 1 or n2 < 1:
        return result
    result["mean_1"], result["mean_2"] = float(x.mean()), float(y.mean())
    if n1 < 2 or n2 < 2:
        return result
    v1, v2 = x.var(ddof=1), y.var(ddof=1)
    result["sd_1"], result["sd_2"] = float(np.sqrt(v1)), float(np.sqrt(v2))
    difference = result["mean_1"] - result["mean_2"]
    squared_se = v1 / n1 + v2 / n2
    result["estimate"] = difference
    if squared_se <= 0:
        return result
    se = float(np.sqrt(squared_se))
    df = squared_se ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
    t_value = difference / se
    margin = stats.t.ppf(0.5 + CONFIDENCE / 2, df) * se
    result.update(se=se, df=float(df), t=float(t_value), p=float(2 * stats.t.sf(abs(t_value), df)),
                  ci_low=difference - margin, ci_high=difference + margin)
    pooled_df = n1 + n2 - 2
    pooled_sd = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / pooled_df)
    if pooled_sd > 0:
        g = hedges_j(pooled_df) * difference / pooled_sd
        se_g = np.sqrt((n1 + n2) / (n1 * n2) + g ** 2 / (2 * pooled_df))
        z = stats.norm.ppf(0.5 + CONFIDENCE / 2)
        result.update(hedges_g=float(g), hedges_g_ci_low=float(g - z * se_g), hedges_g_ci_high=float(g + z * se_g))
    return result


def one_sample_test(values: Sequence[float], mu: float = 0.0) -> dict[str, float]:
    """One-sample (or paired-difference) t-test against ``mu`` with confidence interval, dz and Hedges-corrected dz."""
    d = finite(values)
    n = d.size
    keys = ("mean", "sd", "estimate", "se", "ci_low", "ci_high", "t", "df", "p", "dz", "hedges_g",
            "hedges_g_ci_low", "hedges_g_ci_high")
    result = {"n": n, **{key: _NAN for key in keys}}
    if n < 1:
        return result
    result["mean"] = result["estimate"] = float(d.mean())
    if n < 2:
        return result
    sd = float(d.std(ddof=1))
    result["sd"] = sd
    if sd <= 0:
        return result
    se = sd / np.sqrt(n)
    t_value = (d.mean() - mu) / se
    margin = stats.t.ppf(0.5 + CONFIDENCE / 2, n - 1) * se
    dz = float((d.mean() - mu) / sd)
    g = hedges_j(n - 1) * dz
    se_g = np.sqrt(1.0 / n + g ** 2 / (2.0 * n))
    z = stats.norm.ppf(0.5 + CONFIDENCE / 2)
    result.update(se=float(se), t=float(t_value), df=float(n - 1), p=float(2 * stats.t.sf(abs(t_value), n - 1)),
                  ci_low=float(d.mean() - margin), ci_high=float(d.mean() + margin), dz=dz, hedges_g=float(g),
                  hedges_g_ci_low=float(g - z * se_g), hedges_g_ci_high=float(g + z * se_g))
    return result


def mann_whitney(first: Sequence[float], second: Sequence[float]) -> dict[str, float]:
    """Two-sided Mann-Whitney U test (statistic for ``first``); exact when feasible, otherwise asymptotic."""
    x, y = finite(first), finite(second)
    if x.size < 1 or y.size < 1:
        return {"u": _NAN, "p": _NAN}
    outcome = stats.mannwhitneyu(x, y, alternative="two-sided", method="auto")
    return {"u": float(outcome.statistic), "p": float(outcome.pvalue)}


def wilcoxon_signed_rank(differences: Sequence[float]) -> dict[str, float]:
    """Two-sided Wilcoxon signed-rank test of paired differences (zero differences are dropped)."""
    d = finite(differences)
    d = d[d != 0]
    if d.size < 1:
        return {"statistic": _NAN, "p": _NAN}
    outcome = stats.wilcoxon(d, alternative="two-sided", method="auto")
    return {"statistic": float(outcome.statistic), "p": float(outcome.pvalue)}
