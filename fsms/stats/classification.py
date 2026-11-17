"""A4: logistic regression of seizure type (FS = 1, ES = 0) on the change toward onset.

Predictors, per participant, are built from the A2 change quantities (final minute minus baseline block):
``JS_change``, ``CSI_change`` and ``CPI_change``.  CSI and CPI are already z-scored to the participant's
interictal recordings.  With ``a = (CSI_change - CPI_change) / 2``,

    z_JS = standardise(JS_change),  z_a = standardise(a),  L = z_JS + z_a

where ``standardise`` subtracts the sample mean and divides by the sample standard deviation (ddof = 1) of the
analysis sample.  The index L has equal weights and nothing is fitted; the constants are stored so that it can be
applied to new data.

Models: ``M_JS`` (z_JS), ``M_a`` (z_a), ``M_L`` (L) and the joint ``M_JS+a`` (z_JS and z_a).

Primary analysis: FS-only vs ES-only participants.  Each model reports odds ratios (per unit and per sample SD
of the predictor) with Wald confidence intervals, Wald and likelihood-ratio p-values, McFadden and Nagelkerke
R2, AIC, the apparent AUC, a bootstrap optimism-corrected AUC (Harrell; the standardisation constants are
re-estimated in every resample) and a leave-one-out cross-validated AUC (pooled held-out linear predictors;
constants re-estimated in every training fold).  Multiplicity: Holm across the three single-predictor models; the
joint model reports a likelihood-ratio test per term and forms no family.

Secondary analyses (no multiplicity adjustment):

* sensitivity: the same models with change measured against the participant's interictal mean;
* event level: all eligible seizures, including those of participants with both seizure types (DD), seizure type
  of the event as outcome, standardisation over events, participant-clustered robust standard errors;
* DD participants: for each DD participant with seizures of both types, the mean of ``JS_change``, ``a`` and
  ``L`` over their FS seizures is compared with that over their ES seizures (paired t-test and Wilcoxon
  signed-rank test), with ``L`` computed from the primary standardisation constants.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from . import logistic
from .change_test import participant_changes
from .inference import holm, one_sample_test, wilcoxon_signed_rank
from .release import RepresentationData

MODELS: dict[str, tuple[str, ...]] = {"M_JS": ("z_JS",), "M_a": ("z_a",), "M_L": ("L",), "M_JS+a": ("z_JS", "z_a")}
SINGLE_PREDICTOR_MODELS = ("M_JS", "M_a", "M_L")
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20260101
_Z = stats.norm.ppf(0.975)


@dataclass(frozen=True)
class ModelSet:
    """Summary, coefficient and ROC tables of the four models fitted to one sample."""

    models: pd.DataFrame
    coefficients: pd.DataFrame
    roc: pd.DataFrame


@dataclass(frozen=True)
class ClassificationResult:
    """Tables of analysis A4."""

    predictors: pd.DataFrame
    primary: ModelSet
    sensitivity: ModelSet
    event_models: pd.DataFrame
    event_coefficients: pd.DataFrame
    dd_pairs: pd.DataFrame
    l_definition: dict


def _exp(value: float) -> float:
    """``exp`` that returns ``inf`` instead of warning when an odds ratio is unbounded (e.g. separation)."""
    with np.errstate(over="ignore"):
        return float(np.exp(value))


def standardisation(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Column means and sample SDs of the raw predictors ``[JS_change, a]``."""
    return raw.mean(axis=0), raw.std(axis=0, ddof=1)


def features(model: str, train: np.ndarray, apply: np.ndarray) -> np.ndarray:
    """Predictor columns of ``model`` for the rows ``apply``, standardised with constants from ``train``.

    ``train`` and ``apply`` have columns ``[JS_change, a]``.  Columns are NaN when a constant is not finite.
    """
    mean, sd = standardisation(train)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (apply - mean) / sd
    columns = {"M_JS": z[:, [0]], "M_a": z[:, [1]], "M_L": z.sum(axis=1, keepdims=True), "M_JS+a": z}[model]
    return columns if np.all(sd > 0) else np.full_like(columns, np.nan)


def _design(columns: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(columns)), columns])


def bootstrap_optimism(model: str, raw: np.ndarray, y: np.ndarray, n_boot: int, seed: int) -> tuple[float, int]:
    """Mean optimism of the AUC (bootstrap-sample AUC minus original-sample AUC of the bootstrap fit)."""
    rng = np.random.default_rng(seed)
    n_obs = len(y)
    optimism = []
    for _ in range(n_boot):
        index = rng.integers(0, n_obs, n_obs)
        y_boot = y[index]
        if y_boot.min() == y_boot.max():
            continue
        x_boot = _design(features(model, raw[index], raw[index]))
        fit = logistic.fit_logit(x_boot, y_boot)
        if not np.all(np.isfinite(fit.beta)):
            continue
        x_original = _design(features(model, raw[index], raw))
        optimism.append(logistic.auc(x_boot @ fit.beta, y_boot) - logistic.auc(x_original @ fit.beta, y))
    return (float(np.mean(optimism)) if optimism else float("nan")), len(optimism)


def leave_one_out_auc(model: str, raw: np.ndarray, y: np.ndarray) -> float:
    """AUC of the pooled leave-one-out linear predictors."""
    n_obs = len(y)
    predicted = np.full(n_obs, np.nan)
    for held_out in range(n_obs):
        train = np.arange(n_obs) != held_out
        fit = logistic.fit_logit(_design(features(model, raw[train], raw[train])), y[train])
        predicted[held_out] = (_design(features(model, raw[train], raw[[held_out]])) @ fit.beta)[0]
    return logistic.auc(predicted, y) if np.all(np.isfinite(predicted)) else float("nan")


def fit_model_set(raw: np.ndarray, y: np.ndarray, *, n_boot: int = N_BOOTSTRAP, seed: int = BOOTSTRAP_SEED) -> ModelSet:
    """Fit the four models to ``raw`` (columns ``[JS_change, a]``) and ``y`` (FS = 1) and summarise them."""
    n_obs = len(y)
    if n_obs < 3 or y.min() == y.max():
        raise ValueError(f"logistic analysis needs both seizure types among at least 3 participants (n = {n_obs})")
    null_ll = logistic.null_log_likelihood(y)
    model_rows, coefficient_rows, roc_rows = [], [], []
    for model, terms in MODELS.items():
        columns = features(model, raw, raw)
        design = _design(columns)
        fit = logistic.fit_logit(design, y)
        lr_chi2 = max(0.0, 2.0 * (fit.log_likelihood - null_ll))
        score = design @ fit.beta
        optimism, n_effective = bootstrap_optimism(model, raw, y, n_boot, seed) if n_boot > 0 else (np.nan, 0)
        apparent = logistic.auc(score, y)
        row = {
            "model": model, "n": n_obs, "n_fs": int(y.sum()), "n_es": int(n_obs - y.sum()),
            "converged": fit.converged, "separated": fit.separated,
            "log_likelihood": fit.log_likelihood, "log_likelihood_null": null_ll,
            "lr_chi2": lr_chi2, "lr_df": len(terms), "p_lr": float(stats.chi2.sf(lr_chi2, len(terms))),
            "mcfadden_r2": logistic.mcfadden_r2(fit.log_likelihood, null_ll),
            "nagelkerke_r2": logistic.nagelkerke_r2(fit.log_likelihood, null_ll, n_obs),
            "aic": 2.0 * (len(terms) + 1) - 2.0 * fit.log_likelihood,
            "auc_apparent": apparent, "auc_optimism": optimism, "auc_corrected": apparent - optimism,
            "auc_loo": leave_one_out_auc(model, raw, y), "n_bootstrap": n_boot, "n_bootstrap_used": n_effective,
        }
        model_rows.append(row)
        for position, term in enumerate(terms, start=1):
            sd_term = float(columns[:, position - 1].std(ddof=1))
            estimate, se = float(fit.beta[position]), float(fit.se[position])
            wald_z = estimate / se if se > 0 else np.nan
            if len(terms) == 1:
                p_term_lr = row["p_lr"]
            else:
                kept = [i for i in range(len(terms)) if i != position - 1]
                reduced = logistic.fit_logit(_design(columns[:, kept]), y)
                p_term_lr = float(stats.chi2.sf(max(0.0, 2.0 * (fit.log_likelihood - reduced.log_likelihood)), 1))
            coefficient_rows.append({
                "model": model, "term": term, "estimate": estimate, "se": se, "z": wald_z,
                "p_wald": float(2 * stats.norm.sf(abs(wald_z))) if np.isfinite(wald_z) else np.nan,
                "p_lr": p_term_lr, "or_per_unit": _exp(estimate),
                "or_per_unit_ci_low": _exp(estimate - _Z * se),
                "or_per_unit_ci_high": _exp(estimate + _Z * se),
                "sd_predictor": sd_term, "or_per_sd": _exp(estimate * sd_term),
                "or_per_sd_ci_low": _exp((estimate - _Z * se) * sd_term),
                "or_per_sd_ci_high": _exp((estimate + _Z * se) * sd_term),
            })
        if np.all(np.isfinite(score)):
            fpr, tpr, thresholds = logistic.roc_curve(score, y)
            roc_rows.append(pd.DataFrame({"model": model, "fpr": fpr, "tpr": tpr, "threshold_linear_predictor": thresholds}))
    models = pd.DataFrame(model_rows)
    coefficients = pd.DataFrame(coefficient_rows)
    models["p_wald"] = np.nan
    models["p_holm_lr"] = models["p_holm_wald"] = np.nan
    single = models["model"].isin(SINGLE_PREDICTOR_MODELS)
    for model in SINGLE_PREDICTOR_MODELS:
        models.loc[models["model"] == model, "p_wald"] = coefficients.loc[coefficients["model"] == model, "p_wald"].iloc[0]
    models.loc[single, "p_holm_lr"] = holm(models.loc[single, "p_lr"])
    models.loc[single, "p_holm_wald"] = holm(models.loc[single, "p_wald"])
    return ModelSet(models, coefficients, pd.concat(roc_rows, ignore_index=True) if roc_rows else pd.DataFrame(
        columns=["model", "fpr", "tpr", "threshold_linear_predictor"]))


def predictor_table(changes: pd.DataFrame) -> pd.DataFrame:
    """Complete-case participants with ``JS_change``, ``CSI_change``, ``CPI_change`` and ``a``."""
    wide = changes.pivot(index=["participant_id", "group"], columns="outcome", values="change").reset_index()
    wide = wide.rename(columns={"JS": "JS_change", "CSI": "CSI_change", "CPI": "CPI_change"})
    wide = wide.dropna(subset=["JS_change", "CSI_change", "CPI_change"]).reset_index(drop=True)
    wide["a"] = (wide["CSI_change"] - wide["CPI_change"]) / 2.0
    return wide[["participant_id", "group", "JS_change", "CSI_change", "CPI_change", "a"]]


def _raw_and_outcome(table: pd.DataFrame, positive: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    return table[["JS_change", "a"]].to_numpy(float), positive.to_numpy(float)


def event_level_models(rep: RepresentationData) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Event-level logistic models with participant-clustered robust standard errors (all eligible seizures)."""
    wide = rep.events.pivot_table(index=["recording_id", "participant_id", "group", "event_type"], columns="outcome",
                                  values="change", dropna=False).reset_index()
    wide = wide.dropna(subset=["JS", "CSI", "CPI"]).reset_index(drop=True)
    wide["a"] = (wide["CSI"] - wide["CPI"]) / 2.0
    raw = np.column_stack([wide["JS"], wide["a"]]).astype(float)
    y = (wide["event_type"] == "FS").to_numpy(float)
    clusters = pd.factorize(wide["participant_id"])[0]
    model_rows, coefficient_rows = [], []
    for model, terms in MODELS.items():
        columns = features(model, raw, raw)
        status, fit = "ok", None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                fit = sm.Logit(y, _design(columns)).fit(disp=0, cov_type="cluster", cov_kwds={"groups": clusters})
            except (np.linalg.LinAlgError, ValueError, sm.tools.sm_exceptions.PerfectSeparationError):
                status = "failed"
        model_rows.append({
            "model": model, "n_events": len(wide), "n_participants": int(clusters.max() + 1) if len(wide) else 0,
            "n_fs_events": int(y.sum()), "n_es_events": int(len(y) - y.sum()),
            "n_events_dd_participants": int((wide["group"] == "DD").sum()), "fit_status": status})
        if fit is None:
            continue
        for position, term in enumerate(terms, start=1):
            estimate, se = float(fit.params[position]), float(fit.bse[position])
            sd_term = float(columns[:, position - 1].std(ddof=1))
            z = estimate / se if se > 0 else np.nan
            coefficient_rows.append({
                "model": model, "term": term, "estimate": estimate, "robust_se": se, "z": z,
                "p": float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else np.nan, "sd_predictor": sd_term,
                "or_per_sd": _exp(estimate * sd_term),
                "or_per_sd_ci_low": _exp((estimate - _Z * se) * sd_term),
                "or_per_sd_ci_high": _exp((estimate + _Z * se) * sd_term)})
    return pd.DataFrame(model_rows), pd.DataFrame(coefficient_rows)


def dd_paired_comparison(rep: RepresentationData, constants: dict) -> pd.DataFrame:
    """Paired comparison, within DD participants, of FS-seizure vs ES-seizure ``JS_change``, ``a`` and ``L``."""
    events = rep.events[rep.events["group"] == "DD"]
    wide = events.pivot_table(index=["recording_id", "participant_id", "event_type"], columns="outcome",
                              values="change", dropna=False).reset_index()
    wide = wide.dropna(subset=["JS", "CSI", "CPI"])
    wide = wide.assign(a=(wide["CSI"] - wide["CPI"]) / 2.0)
    per_type = wide.groupby(["participant_id", "event_type"])[["JS", "a"]].mean().unstack("event_type").dropna()
    rows = []
    for measure in ("JS_change", "a", "L"):
        if per_type.empty:
            fs_values = es_values = pd.Series(dtype=float)
        else:
            js, a = per_type["JS"], per_type["a"]
            z_js = (js - constants["JS_change"]["mean"]) / constants["JS_change"]["sd"]
            z_a = (a - constants["a"]["mean"]) / constants["a"]["sd"]
            value = {"JS_change": js, "a": a, "L": z_js + z_a}[measure]
            fs_values, es_values = value["FS"], value["ES"]
        difference = (fs_values - es_values).to_numpy(float)
        one, rank = one_sample_test(difference), wilcoxon_signed_rank(difference)
        rows.append({
            "measure": measure, "n_pairs": len(difference),
            "mean_fs_events": float(fs_values.mean()) if len(difference) else np.nan,
            "mean_es_events": float(es_values.mean()) if len(difference) else np.nan,
            "estimate_fs_minus_es": one["estimate"], "se": one["se"], "ci_low": one["ci_low"],
            "ci_high": one["ci_high"], "t": one["t"], "df": one["df"], "p_paired_t": one["p"], "dz": one["dz"],
            "hedges_dz": one["hedges_g"], "wilcoxon_statistic": rank["statistic"], "p_wilcoxon": rank["p"]})
    return pd.DataFrame(rows)


def run_classification(rep: RepresentationData, *, n_boot: int = N_BOOTSTRAP, seed: int = BOOTSTRAP_SEED) -> ClassificationResult:
    """Run A4: primary, sensitivity, event-level and DD analyses for one representation."""
    predictors = predictor_table(participant_changes(rep, "baseline_block"))
    raw, y = _raw_and_outcome(predictors, predictors["group"] == "FS")
    primary = fit_model_set(raw, y, n_boot=n_boot, seed=seed)

    mean, sd = standardisation(raw)
    constants = {
        "JS_change": {"mean": float(mean[0]), "sd": float(sd[0])},
        "a": {"mean": float(mean[1]), "sd": float(sd[1])},
        "ddof": 1,
    }
    z = (raw - mean) / sd
    predictors = predictors.assign(z_JS=z[:, 0], z_a=z[:, 1], L=z.sum(axis=1))

    sensitivity_table = predictor_table(participant_changes(rep, "interictal_mean"))
    raw_s, y_s = _raw_and_outcome(sensitivity_table, sensitivity_table["group"] == "FS")
    sensitivity = fit_model_set(raw_s, y_s, n_boot=n_boot, seed=seed)

    event_models, event_coefficients = event_level_models(rep)
    definition = {
        "index": "L = z(JS_change) + z(a)",
        "a": "(CSI_change - CPI_change) / 2",
        "change": "final-minute value minus mean of the baseline block (window_end_min 10..19)",
        "standardisation": constants,
        "n_participants": len(predictors),
        "n_fs": int(y.sum()),
        "n_es": int(len(y) - y.sum()),
        "outcome_columns": dict(rep.columns),
        "seizure_type_coding": "FS = 1, ES = 0",
    }
    return ClassificationResult(
        predictors=predictors, primary=primary, sensitivity=sensitivity, event_models=event_models,
        event_coefficients=event_coefficients, dd_pairs=dd_paired_comparison(rep, constants), l_definition=definition)
