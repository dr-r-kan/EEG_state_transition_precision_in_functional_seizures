# Statistical analysis

## Purpose

The analysis asks whether the changes in EEG microstate transition organisation and in cardiac autonomic state
that precede seizure onset differ between participants with functional seizures only (FS) and participants with
epileptic seizures only (ES). Inference is at the level of the participant. The analysis reads only the
de-identified release bundle (see `DATA_DICTIONARY.md`), so it can be reproduced from the published files alone.

```
python -m fsms.stats --release <release_dir> --out <out_dir> --prereg-id <id> [--representations matrix tensor] [--n-bootstrap 2000]
```

```python
from fsms.stats import run_statistics
results = run_statistics(release_dir, out_dir, prereg_id, representations=("matrix", "tensor"))
```

`run_statistics` raises `StatisticsGateError` unless `<release_dir>/inclusion_checkpoint.json` exists and its
`prereg_id` equals the identifier passed in. The identifier is written into every output (a `prereg_id` column
in every CSV, the PNG metadata and footer, `L_definition.json`, `results.json` and `results_summary.md`).
Results are deterministic: the bootstrap seed is fixed and outputs contain no timestamps.

## Data

Tables read: `windows.csv.gz`, `events.csv`, `participants.csv`. The `semiology` column of `events.csv` is used
only by the exploratory analysis E1.

* **Representations.** `matrix` uses `js_excess_ordinary` (JS) and `log_kappa_ordinary` (kappa); `tensor` uses
  `js_excess_tensor` and `log_kappa_tensor`. Both use `csi_z` (CSI) and `cpi_z` (CPI). Each representation is a
  complete, separate analysis with its own eligible sample and its own multiplicity families; results are written
  to `<out_dir>/<representation>/`.
* **Eligibility.** A representation uses only participants with `eligible_<representation> = true` and seizures
  (preictal recordings) with `eligible_<representation> = true` in `participants.csv` / `events.csv`.
* **Groups.** Between-group tests compare participants with `group == 'FS'` (FS-only) and `group == 'ES'`
  (ES-only). Participants with `group == 'DD'` (both seizure types) are never part of a between-group test; they
  contribute only to the event-level and paired analyses of A4 (secondary).
* **Time axis.** Windows are 60 s, aligned to onset: baseline block `window_end_min` 19..10 (the 20-10 minute
  block), minutes 9..0 (0 = the final minute), and interictal windows.
* **Outcomes.** JS, kappa, CSI, CPI.

Quantities derived per eligible seizure and outcome:

* `baseline_mean`: mean over the valid windows with `window_end_min` 10..19; missing unless at least 5 windows are valid;
* `final`: value at `window_end_min` 0;
* `change` = `final` - `baseline_mean`;
* per-minute `change` (`window_end_min` 9..0) = window value - `baseline_mean`.

Per eligible participant and outcome: `interictal` = mean over the participant's interictal windows. A
participant's value of any seizure-derived quantity is the unweighted mean over their eligible seizures.

## A1: group x condition mixed ANOVA

Per outcome, each FS-only/ES-only participant contributes `interictal` and the participant mean of `final`
(final-minute window). With two within-participant levels the mixed ANOVA is computed exactly as three F-tests on
one number per participant, valid for unequal group sizes (N = participants in the outcome's sample):

| effect | score tested | test |
|---|---|---|
| group | `(interictal + final) / 2` | FS vs ES, F(1, N - 2) (unweighted marginal means) |
| condition | `final - interictal` | intercept of the effect-coded model `difference ~ 1 + group`, F(1, N - 2) (Type III) |
| interaction | `final - interictal` | FS vs ES, F(1, N - 2) |

Partial eta squared = `F * df1 / (F * df1 + df2)`.

Post-hoc simple effects, per outcome: FS vs ES at interictal and at the final minute (Welch t-test); final minute
vs interictal within FS and within ES (paired t-test). Each reports the estimate (FS minus ES, or final minus
interictal), 95% CI, t, df, p and an effect size: Hedges' g (between) or Hedges-corrected dz (paired), with an
approximate large-sample CI.

Multiplicity:

* omnibus effects: Holm across the 4 outcomes, separately for group, condition and interaction (`p_holm`);
* post-hoc tests: raw p; Holm across the 4 outcomes within each simple-effect type (`p_holm_outcomes`); Holm
  across all 16 post-hoc tests (`p_holm_all`, the headline).

## A2: change test

`change` (final minute minus baseline-block mean) is computed per eligible seizure and averaged per participant.
FS vs ES: Welch t-test (estimate = FS minus ES, 95% CI, Hedges' g with approximate CI) per outcome, plus a
two-sided Mann-Whitney U test as a sensitivity analysis. Within each group (descriptive): one-sample t-test of the
participant changes against 0. Holm across the 4 outcomes for the Welch test, for the Mann-Whitney test, and
within each group for the one-sample tests.

## A3: linear mixed model of the change over the last ten minutes (JS and kappa)

Data: for every eligible seizure and minute window (`window_end_min` 9..0) the change from that seizure's
baseline-block mean; averaged over a participant's seizures per minute, giving one row per participant and
minute. Model (REML, `statsmodels` MixedLM):

```
change = b0 + b1 * t + b2 * FS + b3 * t * FS + u0_p + u1_p * t + e        t = -window_end_min  (-9 ... 0)
(u0_p, u1_p) ~ N(0, G), G an unstructured 2 x 2 covariance;  e ~ N(0, s^2)
```

`FS` = 1 for FS-only and 0 for ES-only participants. With t = 0 the final minute, `b0` = ES change at the final
minute, `b2` = FS minus ES difference at the final minute, `b3` = difference in gradient toward onset (change per
minute; the headline). A random intercept and a random slope are always included.

Fitting: optimisers `lbfgs`, `powell`, `nm` are tried in that order, each from the default starting values,
until one reports convergence; if none does, the attempt with the highest log-likelihood is reported with
`fit_status = not_converged`. The random slope is never dropped. The random-effects covariance is flagged
singular when it is not positive definite, a variance is below `1e-6 * s^2`, or the intercept-slope correlation
is at least 0.999 in absolute value. Fixed-effect tests are Wald z-tests with model-based standard errors. The
interaction is also tested by an ML likelihood-ratio test (1 df) of the full model against the model without
`t * FS`. Simple slopes (ES, FS) and final-minute levels (ES, FS) are linear contrasts of the fixed effects.

Multiplicity: Holm across the 2 outcomes (JS, kappa) for `t:group_FS` (headline) and, separately, for
`group_FS`; the likelihood-ratio p-values are Holm-adjusted across the 2 outcomes as well.

## A4: logistic regression of seizure type (FS = 1, ES = 0)

Predictors are computed per participant from the A2 change quantities (final minute minus baseline-block mean):
`JS_change`, `CSI_change`, `CPI_change`; `a = (CSI_change - CPI_change) / 2` (CSI and CPI are z-scored to the
participant's interictal recordings). Participants need all three changes (complete cases). `z_JS` and `z_a` are
`JS_change` and `a` standardised with the analysis sample's mean and SD (ddof = 1); `L = z_JS + z_a` (equal
weights, nothing fitted). Models: `M_JS` (z_JS), `M_a` (z_a), `M_L` (L), `M_JS+a` (z_JS and z_a). The constants and
the definition of L are saved in `L_definition.json` so the index can be applied to new data.

Per model: coefficients with odds ratios per unit and per sample SD of the predictor (Wald 95% CI), Wald and
likelihood-ratio p-values, McFadden and Nagelkerke R2, AIC, apparent AUC (Mann-Whitney U / (n1 n0)), a bootstrap
optimism-corrected AUC (Harrell; 2000 resamples of participants, fixed seed; the standardisation constants are
re-estimated in each resample) and a leave-one-out cross-validated AUC (pooled held-out linear predictors;
constants re-estimated in each training fold). The joint model reports a likelihood-ratio test of each term.

Multiplicity: Holm across the 3 single-predictor models (`p_holm_lr`, headline, and `p_holm_wald`); the joint
model forms no family.

Secondary analyses (no multiplicity adjustment):

1. **Sensitivity: interictal reference.** The same models with change measured against the participant's
   interictal mean (`change_reference = interictal_mean`).
2. **Event level.** All eligible seizures, including those of DD participants; outcome = seizure type of the
   event; predictors standardised over events; standard errors robust to clustering by participant.
3. **DD participants.** For each DD participant with eligible seizures of both types: mean `JS_change`, `a` and `L`
   over FS seizures versus over ES seizures; paired t-test and Wilcoxon signed-rank test; `n_pairs` pairs. `L` uses
   the primary standardisation constants.

## E1 (exploratory): hypomotor versus hypermotor functional seizures

Not confirmatory. Seizures: eligible functional seizures (`event_type == 'FS'`) of FS-only and DD participants whose
`semiology` in `events.csv` is `hypo` or `hyper` (`unclassified` is left out; see `docs/DATA_DICTIONARY.md` for
how the category is derived from the clinical workbooks). The quantity is the A2
`change` (final minute minus baseline-block mean) per outcome. A participant whose classified functional seizures
all have one semiology is assigned to it and contributes the mean change over those seizures; a participant with
seizures of both semiologies is `mixed` and is left out of the comparison (`n_mixed_excluded`). Hypo versus hyper
participants: Welch t-test (estimate = hypo minus hyper, 95% CI, Hedges' g) and a two-sided Mann-Whitney U test,
per outcome. p-values are not adjusted, form no multiplicity family and are not used for inference.

## Multiplicity families (per representation)

| analysis | family | m |
|---|---|---|
| A1 | omnibus group effect over outcomes | 4 |
| A1 | omnibus condition effect over outcomes | 4 |
| A1 | omnibus interaction over outcomes | 4 |
| A1 | post-hoc, per simple-effect type over outcomes (`p_holm_outcomes`) | 4 |
| A1 | post-hoc, all simple effects (`p_holm_all`, headline) | 16 |
| A2 | Welch FS vs ES over outcomes (and separately Mann-Whitney) | 4 |
| A2 | within-group one-sample tests over outcomes, per group | 4 |
| A3 | `t:group_FS` over JS, kappa (headline); `group_FS` separately | 2 |
| A4 | single-predictor models M_JS, M_a, M_L (LR and, separately, Wald) | 3 |

Holm adjustment is the step-down procedure: adjusted p-values are monotone and capped at 1; tests that cannot be
computed (NaN) are not counted in the family size.

## Outputs

Under `<out_dir>/`: `results.json` (headline numbers per representation, Holm families, sample sizes, input file
hashes, output table hashes, software versions), `results_summary.md` (tables of the numbers, no interpretation).
Under `<out_dir>/<representation>/`: `tables/*.csv`, `figures/*.png` (300 dpi, colour-blind-safe palette),
`L_definition.json`. Every CSV begins with `prereg_id` and `representation`; `group` is `FS`/`ES`;
`n_fs`/`n_es` are participant counts; all p-values are two-sided; confidence intervals are 95%.

### Tables

**`a1_participant_values.csv`**: `participant_id`; `outcome`; `group`; `interictal` (mean over interictal windows);
`final` (mean over eligible seizures of the final-minute value); `n_events` (seizures contributing);
`n_interictal_windows`.

**`a1_cell_means.csv`**: `outcome`, `group`, `condition` (`interictal`, `final_minute`); `n`, `mean`, `sd`,
`ci_low`, `ci_high` (t interval over participants).

**`a1_omnibus.csv`**: `outcome`; `effect` (`group`, `condition`, `interaction`); `estimate` (group: FS minus ES on
the condition-averaged score; condition: unweighted mean of the two groups' final-minus-interictal means;
interaction: FS minus ES in final-minus-interictal); `se`; `f`, `df1`, `df2`, `p`; `partial_eta_sq`; `n_fs`,
`n_es`; `p_holm` (Holm across the 4 outcomes within the effect).

**`a1_posthoc.csv`**: `outcome`; `effect` (`group_at_interictal`, `group_at_final_minute`, `condition_in_FS`,
`condition_in_ES`); `comparison` (text); `n_fs`, `n_es`; `estimate`, `se`, `ci_low`, `ci_high`, `t`, `df`, `p`
(raw); `effect_size`, `effect_size_type` (`hedges_g` or `hedges_dz`), `effect_size_ci_low`, `effect_size_ci_high`;
`p_holm_outcomes` (Holm across the 4 outcomes within the effect type); `p_holm_all` (Holm across all 16 tests;
headline).

**`a2_participant_change.csv`**: `participant_id`, `group`, `outcome`, `change` (participant mean of seizure
changes), `n_events`.

**`a2_between_groups.csv`**: `outcome`; `n_fs`, `n_es`; `mean_fs`, `mean_es`, `sd_fs`, `sd_es`; Welch
`estimate` (FS minus ES), `se`, `ci_low`, `ci_high`, `t`, `df`, `p`; `hedges_g` with `hedges_g_ci_low`,
`hedges_g_ci_high`; `mannwhitney_u` (statistic for FS), `p_mannwhitney`; `p_holm`, `p_mannwhitney_holm` (Holm
across the 4 outcomes).

**`a2_within_group.csv`**: `group`, `outcome`, `n`, `mean`, `sd`, `se`, `ci_low`, `ci_high`, `t`, `df`, `p`, `dz`,
`hedges_g` (Hedges-corrected dz), `p_holm` (Holm across the 4 outcomes within the group).

**`a3_participant_minute_changes.csv`**: the model data: `participant_id`, `outcome`, `group`, `t`
(`-window_end_min`), `change` (participant mean over seizures), `n_events`.

**`a3_models.csv`** (one row per outcome): `fit_status` (`converged`, `converged_after_retry`, `singular_fit`,
`not_converged`, `insufficient_data`); `converged`; `singular` (random-effects covariance singular); `optimiser`
(the optimiser of the reported fit); `attempts` (optimisers tried and their outcome); `n_obs`, `n_participants`,
`n_fs`, `n_es`; `random_effects_dim` (2 = intercept and slope); `lrt_chi2`, `lrt_df`, `lrt_p`, `lrt_status` (ML
likelihood-ratio test of `t:group_FS`); `messages` (optimiser warnings); `lrt_p_holm_outcomes`.

**`a3_fixed_effects.csv`**: `outcome`; `term` (`intercept`, `t`, `group_FS`, `t:group_FS`); `estimate`, `se`, `z`,
`p`, `ci_low`, `ci_high`; `meaning`; `p_holm_outcomes` (Holm across JS and kappa; filled for `t:group_FS` and
`group_FS` only).

**`a3_variance_components.csv`**: `outcome`; `component` (`var_intercept`, `var_slope`, `cov_intercept_slope`,
`corr_intercept_slope`, `var_residual`); `estimate`.

**`a3_simple_effects.csv`**: `outcome`; `quantity` (`slope_ES`, `slope_FS`, `level_ES_final_minute`,
`level_FS_final_minute`); `estimate`, `se`, `z`, `p`, `ci_low`, `ci_high`.

**`a4_participant_predictors.csv`**: `change_reference`; `participant_id`, `group`; `JS_change`, `CSI_change`,
`CPI_change`; `a`; `z_JS`, `z_a`, `L`.

**`a4_models.csv`**, **`a4_sensitivity_models.csv`** (one row per model; `analysis`, `change_reference`):
`model`; `n`, `n_fs`, `n_es`; `converged`; `separated` (quasi-complete separation: Wald results unreliable);
`log_likelihood`, `log_likelihood_null`; `lr_chi2`, `lr_df`, `p_lr`; `mcfadden_r2`, `nagelkerke_r2`; `aic`;
`auc_apparent`; `auc_optimism`, `auc_corrected` (= apparent - optimism); `auc_loo`; `n_bootstrap`,
`n_bootstrap_used` (resamples with both seizure types); `p_wald` (Wald p of the single predictor; single-predictor
models only); `p_holm_lr`, `p_holm_wald` (Holm across M_JS, M_a, M_L).

**`a4_coefficients.csv`**, **`a4_sensitivity_coefficients.csv`**: `model`, `term` (`z_JS`, `z_a`, `L`); `estimate`
(log odds per unit), `se`, `z`, `p_wald`; `p_lr` (likelihood-ratio test of the term); `or_per_unit` with
`or_per_unit_ci_low/high`; `sd_predictor` (sample SD of the predictor); `or_per_sd` with `or_per_sd_ci_low/high`.

**`a4_roc_curve_points.csv`**: `model`, `fpr`, `tpr`, `threshold_linear_predictor` (apparent ROC of the fitted
model; predict FS when the linear predictor is at least the threshold).

**`a4_event_level_models.csv`**: `model`; `n_events`, `n_participants`; `n_fs_events`, `n_es_events`;
`n_events_dd_participants`; `fit_status`.

**`a4_event_level_coefficients.csv`**: `model`, `term`; `estimate`; `robust_se` (clustered by participant); `z`;
`p`; `sd_predictor`; `or_per_sd` with CI columns.

**`a4_dd_paired.csv`**: `measure` (`JS_change`, `a`, `L`); `n_pairs`; `mean_fs_events`, `mean_es_events`;
`estimate_fs_minus_es`, `se`, `ci_low`, `ci_high`, `t`, `df`, `p_paired_t`; `dz`, `hedges_dz`;
`wilcoxon_statistic`, `p_wilcoxon`.

**`e1_semiology_participant_change.csv`** (`analysis` = `exploratory`): `participant_id`, `group` (`FS` or `DD`),
`semiology` (`hypo`, `hyper`, `mixed`), `outcome`, `change` (mean over the participant's classified functional
seizures), `n_events`.

**`e1_semiology_between.csv`** (`analysis` = `exploratory`): `outcome`; `n_hypo`, `n_hyper` (participants);
`n_mixed_excluded`; `n_events_hypo`, `n_events_hyper`; `mean_hypo`, `mean_hyper`, `sd_hypo`, `sd_hyper`; Welch
`estimate` (hypo minus hyper), `se`, `ci_low`, `ci_high`, `t`, `df`, `p` (unadjusted); `hedges_g` with
`hedges_g_ci_low`, `hedges_g_ci_high`; `mannwhitney_u` (statistic for hypo), `p_mannwhitney`.

### Figures

`a1_interaction.png` (2 x 2: group x condition means with 95% CI, individual participants and their paired lines);
`a2_change_distributions.png` (participant change by group, per outcome); `a3_trajectories.png` (mean change per
minute with SE, and the model's fixed-effect lines, for JS and kappa); `a4_roc_curves.png` (ROC curves of M_JS,
M_a, M_L).

## Assumptions and limitations

* Participants are the units. Seizure-level quantities are averaged per participant without weighting, so
  participants with more seizures are not given more weight; the precision of a participant's mean differs with
  the number of seizures.
* The samples can differ between outcomes and analyses (a participant needs the relevant values); the sizes are
  reported in `results.json` and in the `n_*` columns.
* The Welch and F tests assume approximately normal participant-level quantities; the Mann-Whitney test (A2) and
  Wilcoxon test (A4 DD) are provided as rank-based checks.
* A3 Wald tests use the normal distribution and model-based standard errors without a small-sample degrees-of-freedom
  correction; they can be liberal with few participants. Random-effects covariances with few participants are
  often at or near the boundary; `fit_status` and `singular` report this and no random term is removed.
* Logistic regression with few participants is subject to separation and wide intervals; `separated` flags
  affected fits. The bootstrap optimism correction and the leave-one-out AUC are estimates of out-of-sample
  discrimination in the same population; the pooled leave-one-out AUC can be biased downward in small samples.
* In simulations without any group difference and about 24 participants, the likelihood-ratio tests of the
  logistic models rejected at about 7% (nominal 5%) and the Wald tests at about 3%; the other headline tests were
  within simulation error of 5%. Both p-values are reported for every model.
* The index L is applied with equal weights by definition; its AUC is therefore not optimised on the data, but
  the standardisation constants are estimated from the analysis sample.
* The event-level analysis treats seizures within a participant as correlated (cluster-robust standard errors,
  with a small-sample correction of the statistics library); with few participants these standard errors are
  approximate.
* CSI and CPI are standardised to the participant's own interictal recordings in the bundle; interictal means of
  CSI and CPI are therefore near zero by construction, which affects the interictal level of A1 for these outcomes.
* The analysis is associational; no covariates (age, medication, seizure duration) are modelled.
