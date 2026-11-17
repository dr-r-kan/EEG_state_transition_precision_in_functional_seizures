"""A3: linear mixed model of the change in JS and kappa over the ten minutes before onset.

Data.  For every eligible seizure and minute window (``window_end_min`` 9..0) the change is the window value
minus that seizure's baseline-block mean.  Changes are averaged over a participant's seizures per minute,
giving one row per participant and minute.

Model (REML, ``statsmodels`` MixedLM)::

    change = b0 + b1 * t + b2 * FS + b3 * t * FS + u0_p + u1_p * t + e,     t = -window_end_min  (-9 ... 0)
    (u0_p, u1_p) ~ N(0, G) with G an unstructured 2 x 2 covariance,         e ~ N(0, s^2)

``FS`` is 1 for FS-only and 0 for ES-only participants.  Because t = 0 is the final minute, ``b0`` is the ES
change at the final minute, ``b2`` the FS minus ES difference at the final minute and ``b3`` the difference in
gradient toward onset (change per minute).  The random intercept *and* the random slope are always kept.

Fitting.  Optimisers are tried in the order ``OPTIMISERS`` (each from the default starting values) until one
reports convergence.  ``fit_status`` is ``converged`` (first optimiser), ``converged_after_retry`` (a later
one), ``singular_fit`` (converged but the random-effects covariance is singular) or ``not_converged`` (the
attempt with the highest log-likelihood is reported).  The covariance ``G`` is called singular when it is not
positive definite, when a variance is below ``VARIANCE_TOLERANCE * s^2`` or when the intercept-slope correlation
is at least ``CORRELATION_TOLERANCE`` in absolute value (the boundary solutions the optimiser typically reaches).

Inference.  Fixed-effect tests are Wald z-tests with model-based standard errors.  The interaction ``b3`` is
additionally tested with an ML likelihood-ratio test (full model against the model without ``t * FS``, both with
the random intercept and slope).  Simple slopes and final-minute levels per group are linear contrasts of the
REML fixed effects.  Multiplicity: Holm across JS and kappa for ``t:group_FS`` (headline) and, separately, for
``group_FS``.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.regression.mixed_linear_model import MixedLM

from .inference import CONFIDENCE, holm
from .release import GROUPS, RepresentationData

A3_OUTCOMES = ("JS", "kappa")
OPTIMISERS = ("lbfgs", "powell", "nm")
MAX_ITERATIONS = 500
VARIANCE_TOLERANCE = 1e-6
CORRELATION_TOLERANCE = 0.999
FIXED_TERMS = ("intercept", "t", "group_FS", "t:group_FS")
TERM_MEANING = {
    "intercept": "ES change at the final minute",
    "t": "ES gradient toward onset (change per minute)",
    "group_FS": "FS minus ES change at the final minute",
    "t:group_FS": "FS minus ES difference in gradient toward onset (change per minute)",
}


@dataclass(frozen=True)
class ModelFit:
    """A fitted MixedLM together with how it was obtained."""

    result: object | None
    optimiser: str
    converged: bool
    attempts: str
    messages: str


@dataclass(frozen=True)
class TrajectoryFit:
    """Full output of the A3 model for one outcome."""

    summary: dict
    fixed: pd.DataFrame
    variance: pd.DataFrame
    simple: pd.DataFrame


@dataclass(frozen=True)
class TrajectoryResult:
    """Tables of analysis A3 (all outcomes of one representation)."""

    data: pd.DataFrame
    models: pd.DataFrame
    fixed: pd.DataFrame
    variance: pd.DataFrame
    simple: pd.DataFrame


def trajectory_table(rep: RepresentationData, outcome: str) -> pd.DataFrame:
    """Participant x minute mean change for FS-only and ES-only participants (``t`` = -window_end_min)."""
    minutes = rep.minutes[(rep.minutes["outcome"] == outcome) & rep.minutes["group"].isin(GROUPS)]
    table = minutes.dropna(subset=["change"]).groupby(["participant_id", "group", "window_end_min"])["change"].agg(
        change="mean", n_events="count").reset_index()
    table.insert(3, "t", -table["window_end_min"].astype(float))
    return table.drop(columns="window_end_min")


def _design(table: pd.DataFrame) -> pd.DataFrame:
    t = table["t"].to_numpy(float)
    fs = (table["group"] == "FS").to_numpy(float)
    return pd.DataFrame({"intercept": 1.0, "t": t, "group_FS": fs, "t:group_FS": t * fs}, index=table.index)


def fit_mixedlm(table: pd.DataFrame, *, reml: bool, interaction: bool = True) -> ModelFit:
    """Fit ``change ~ t * group`` with a random intercept and slope per participant, trying ``OPTIMISERS`` in turn."""
    exog = _design(table)
    if not interaction:
        exog = exog.drop(columns="t:group_FS")
    endog = table["change"].to_numpy(float)
    groups = table["participant_id"].to_numpy()
    exog_re = exog[["intercept", "t"]].to_numpy()
    attempts: list[str] = []
    messages: list[str] = []
    best: tuple[float, object, str] | None = None
    for optimiser in OPTIMISERS:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                result = MixedLM(endog, exog, groups=groups, exog_re=exog_re).fit(
                    reml=reml, method=optimiser, maxiter=MAX_ITERATIONS)
            except (np.linalg.LinAlgError, ValueError) as error:
                attempts.append(f"{optimiser}:error")
                messages.append(f"{optimiser}: {error}")
                continue
        messages.extend(f"{optimiser}: {w.message}" for w in caught)
        converged = bool(result.converged) and bool(np.isfinite(result.llf))
        attempts.append(f"{optimiser}:{'converged' if converged else 'not_converged'}")
        if converged:
            return ModelFit(result, optimiser, True, ";".join(attempts), " | ".join(dict.fromkeys(messages)))
        if np.isfinite(result.llf) and (best is None or result.llf > best[0]):
            best = (float(result.llf), result, optimiser)
    if best is None:
        return ModelFit(None, "", False, ";".join(attempts), " | ".join(dict.fromkeys(messages)))
    return ModelFit(best[1], best[2], False, ";".join(attempts), " | ".join(dict.fromkeys(messages)))


def is_singular(cov_re: np.ndarray, scale: float) -> bool:
    """True when the 2 x 2 random-effects covariance is singular (criteria in the module docstring)."""
    cov_re = np.asarray(cov_re, float)
    if not np.all(np.isfinite(cov_re)) or not scale > 0:
        return True
    variances = np.diag(cov_re)
    if np.any(variances <= VARIANCE_TOLERANCE * scale):
        return True
    correlation = cov_re[0, 1] / np.sqrt(variances[0] * variances[1])
    return bool(abs(correlation) >= CORRELATION_TOLERANCE or np.linalg.eigvalsh(cov_re).min() <= 0)


def _wald_rows(names: list[str], estimate: np.ndarray, cov: np.ndarray) -> pd.DataFrame:
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = estimate / se
    margin = stats.norm.ppf(0.5 + CONFIDENCE / 2) * se
    return pd.DataFrame({"term": names, "estimate": estimate, "se": se, "z": z, "p": 2 * stats.norm.sf(np.abs(z)),
                         "ci_low": estimate - margin, "ci_high": estimate + margin})


def _empty_fit(status: str, n_obs: int, n_fs: int, n_es: int, messages: str = "") -> TrajectoryFit:
    summary = {"fit_status": status, "converged": False, "singular": np.nan, "optimiser": "", "attempts": "",
               "n_obs": n_obs, "n_participants": n_fs + n_es, "n_fs": n_fs, "n_es": n_es,
               "random_effects_dim": 0, "lrt_chi2": np.nan, "lrt_df": 1, "lrt_p": np.nan, "lrt_status": status,
               "messages": messages}
    fixed = pd.DataFrame({"term": list(FIXED_TERMS), "estimate": np.nan, "se": np.nan, "z": np.nan, "p": np.nan,
                          "ci_low": np.nan, "ci_high": np.nan})
    fixed["meaning"] = fixed["term"].map(TERM_MEANING)
    return TrajectoryFit(summary, fixed, pd.DataFrame(columns=["component", "estimate"]), pd.DataFrame(
        columns=["quantity", "estimate", "se", "z", "p", "ci_low", "ci_high"]))


def fit_trajectory_model(table: pd.DataFrame, *, with_lrt: bool = True) -> TrajectoryFit:
    """Fit the A3 model to a participant x minute table (columns ``participant_id``, ``group``, ``t``, ``change``)."""
    table = table.dropna(subset=["change"]).reset_index(drop=True)
    counts = table.drop_duplicates("participant_id")["group"].value_counts()
    n_fs, n_es = int(counts.get("FS", 0)), int(counts.get("ES", 0))
    if min(n_fs, n_es) < 2 or table["t"].nunique() < 3:
        return _empty_fit("insufficient_data", len(table), n_fs, n_es)

    full = fit_mixedlm(table, reml=True)
    if full.result is None:
        return _empty_fit("not_converged", len(table), n_fs, n_es, full.messages)
    result = full.result
    cov_re = np.asarray(result.cov_re, float)
    singular = is_singular(cov_re, float(result.scale))
    if not full.converged:
        status = "not_converged"
    elif singular:
        status = "singular_fit"
    elif full.optimiser != OPTIMISERS[0]:
        status = "converged_after_retry"
    else:
        status = "converged"

    k_fe = len(FIXED_TERMS)
    estimate = np.asarray(result.fe_params, float)
    cov_fixed = np.asarray(result.cov_params(), float)[:k_fe, :k_fe]
    fixed = _wald_rows(list(FIXED_TERMS), estimate, cov_fixed)
    fixed["meaning"] = fixed["term"].map(TERM_MEANING)

    sd = np.sqrt(np.clip(np.diag(cov_re), 0, None))
    correlation = cov_re[0, 1] / (sd[0] * sd[1]) if sd.min() > 0 else np.nan
    variance = pd.DataFrame({"component": [
        "var_intercept", "var_slope", "cov_intercept_slope", "corr_intercept_slope", "var_residual"],
        "estimate": [cov_re[0, 0], cov_re[1, 1], cov_re[0, 1], correlation, float(result.scale)]})

    contrasts = {"slope_ES": [0, 1, 0, 0], "slope_FS": [0, 1, 0, 1],
                 "level_ES_final_minute": [1, 0, 0, 0], "level_FS_final_minute": [1, 0, 1, 0]}
    matrix = np.array(list(contrasts.values()), float)
    simple = _wald_rows(list(contrasts), matrix @ estimate, matrix @ cov_fixed @ matrix.T).rename(
        columns={"term": "quantity"})

    lrt = {"lrt_chi2": np.nan, "lrt_df": 1, "lrt_p": np.nan, "lrt_status": "not_requested"}
    if with_lrt:
        lrt["lrt_status"] = "not_converged"
        ml_full, ml_reduced = fit_mixedlm(table, reml=False), fit_mixedlm(table, reml=False, interaction=False)
        if ml_full.converged and ml_reduced.converged:
            chi2 = max(0.0, 2.0 * (float(ml_full.result.llf) - float(ml_reduced.result.llf)))
            lrt.update(lrt_chi2=chi2, lrt_p=float(stats.chi2.sf(chi2, 1)), lrt_status="converged")
    summary = {"fit_status": status, "converged": full.converged, "singular": singular, "optimiser": full.optimiser,
               "attempts": full.attempts, "n_obs": len(table), "n_participants": n_fs + n_es, "n_fs": n_fs,
               "n_es": n_es, "random_effects_dim": int(result.k_re), **lrt, "messages": full.messages}
    return TrajectoryFit(summary, fixed, variance, simple)


def run_trajectory_model(rep: RepresentationData) -> TrajectoryResult:
    """Run A3 for JS and kappa and apply the Holm families described in the module docstring."""
    data, models, fixed, variance, simple = [], [], [], [], []
    for outcome in A3_OUTCOMES:
        table = trajectory_table(rep, outcome)
        fit = fit_trajectory_model(table)
        table.insert(1, "outcome", outcome)
        data.append(table)
        for frame, sink in ((pd.DataFrame([fit.summary]), models), (fit.fixed, fixed), (fit.variance, variance),
                            (fit.simple, simple)):
            frame = frame.copy()
            frame.insert(0, "outcome", outcome)
            sink.append(frame)
    fixed_table = pd.concat(fixed, ignore_index=True)
    fixed_table["p_holm_outcomes"] = np.nan
    for term in ("t:group_FS", "group_FS"):
        mask = fixed_table["term"] == term
        fixed_table.loc[mask, "p_holm_outcomes"] = holm(fixed_table.loc[mask, "p"])
    models_table = pd.concat(models, ignore_index=True)
    models_table["lrt_p_holm_outcomes"] = holm(models_table["lrt_p"])
    return TrajectoryResult(data=pd.concat(data, ignore_index=True), models=models_table, fixed=fixed_table,
                            variance=pd.concat(variance, ignore_index=True), simple=pd.concat(simple, ignore_index=True))
