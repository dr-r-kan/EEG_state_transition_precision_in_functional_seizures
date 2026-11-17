# Preregistration - updated

## Overview
### Research questions or hypotheses
Background: under an Active Inference account of dissociative symptoms, autonomic state should constrain how EEG microstates transition, and this coupling may change differently before functional than before epileptic seizures.

Primary hypotheses (two-sided; FS-only versus ES-only participants). Change is the final minute before seizure onset minus the same seizure's baseline block 20 to 10 minutes before onset; every window is scored against the participant's interictal reference.
H1. Deviation of the autonomic-conditioned microstate transition tensor from the participant's interictal baseline (noise-corrected, "excess" Jensen-Shannon divergence). We expect a larger deviation before functional seizures than before epileptic seizures.
H2. Precision of that tensor (log kappa). We expect a larger loss of precision before functional seizures than before epileptic seizures.
H3. We expect bigger changes in time-resolved measures of sympathetic and parasympathetic autonomic activity before functional seizures than epileptic seizures.

Secondary hypotheses: the same contrasts for the ordinary (non-conditioned) transition matrix (own multiplicity families), and FS versus ES within participants who have both (DD; paired, unadjusted, not used for the main inference).
Exploratory (no hypothesis tested): FS hypomotor versus hypermotor semiology.
In addition, a group × condition analysis contrasts the final minute with the participant's interictal mean.

### Foreknowledge of data or evidence
Authors have observed the data, but have not performed the proposed analyses. At least some of the data that will be used for this analysis plan has been accessed and observed by the authors. The authors have sufficiently observed relevant evidence to influence their analysis decisions or conclusions. However, the authors have not yet performed any of the proposed analyses in this plan and will not do so until after this plan is registered.

### Explanation of foreknowledge and managing unintended influences
Earlier work on these data (group-level microstate fitting and inspection, and exploratory global, autonomic and transition analyses) predates this plan. For this plan, the study data have been processed by the pipeline up to the de-identified release bundle: preprocessing, microstate fitting (including selection of the number of microstates, K = 4), autonomic indices, window metrics and eligibility. This was used to identify the number of available participants and seizures reported under Sample size. The inclusion checkpoint and the statistical analyses have not been run, and no group comparison of any outcome has been computed or inspected. The pipeline, its tests and the statistical code were developed on synthetic data and on public dataset recordings with synthetic labels. The pipeline code is in the repository. The statistics stage refuses to run unless an inclusion checkpoint bound to this registration's identifier exists and the release bundle is unchanged since that checkpoint; the identifier is written into every output.

## Research Design
### Study type
Non-randomized study: Does not include random assignment of subjects to treatments or conditions. This usually includes correlational studies, observational studies, surveys, natural experiments, real-world evidence studies, regression discontinuity designs, and descriptive studies.

### Intention for causal interpretation
No causal relationship inferred: This study is not intended to inform a causal relationship. 

### Blinding of experimental treatments
No blinding is involved.

### Additional blinding during research or analysis
Upstream preprocessing, exclusions, selection of the number of microstates and microstate labelling are rule-based, fixed in code, and never read diagnostic labels or outcomes. Participant-level microstate maps are fitted with the maps of the participant's diagnostic group (FS, ES or both) as the prior; this step uses the diagnostic group but no outcome.

### Study design
Retrospective, observational, repeated-measures comparison of functional (FS) and epileptic (ES) seizures, including participants with both (DD), with an exploratory comparison of FS hypo- versus hypermotor semiology. Between participants: FS-only versus ES-only. Within participants: interictal reference versus preictal windows, and FS versus ES events in DD participants (paired).

Pipeline (full detail in the repository): HEPPy EEG/ECG preprocessing; microstate cluster count (variational Bayes GMM) optimised based on fit quality, and labels performed using template; microstate transitions recovered at EEG-sample resolution, each classified by soft cardiac parasympathetic index/cardiac sympathetic index (CPI/CSI) gradient quadrants (rising/falling CPI crossed with rising/falling CSI); a participant-specific interictal reference transition grammar; 60-s preictal windows scored against it.

Primary outcomes: autonomic-conditioned (tensor) excess JS divergence and log kappa, and autonomic changes over the same period. The estimand is the change from the 20-10 minute baseline block to the final minute before onset (change test, mixed model and logistic regression); the group × condition analysis contrasts the final minute with the interictal mean.

## Sampling
### Data collection procedures
Existing data; no new collection. EEG from patients undergoing routine video telemetry was reviewed. Events categorised by an expert rater as functional or epileptic seizures were clipped and semiologically categorised, then pseudonymised and stored as EDF+. Eligibility for the analysis is set by pre-specified, group-blind quality rules fixed in the repository (see Analysis Plan).

### Sample size
We will analyse participants from the clinical video-EEG/ECG archive who have recorded seizure clips (N = 204 participants with clips at the start of preprocessing). Sample size is determined by data availability and by rules fixed before any group comparison was computed; no power-based target was set. Participants are removed if their recordings fail the pre-specified, group-blind preprocessing quality rule (6), leaving 198 participants (96 with functional seizures only, 62 with epileptic seizures only, 40 with both) in the dataset. Within recordings, a 60-second interictal or preictal segment is excluded if a preprocessing chunk overlapping it had more than half of its EEG channels interpolated (or could not be fitted); this limit, 0.50, is fixed in the repository. It was raised from an initial 0.30 after reviewing how many segments and participants, by group, each limit retained; no outcome value and no group comparison of any outcome was examined.

For the analyses, a participant needs an interictal reference of at least 300 microstate transitions and at least 5 valid interictal 60-s windows, and at least one eligible seizure. A seizure is eligible if its final-minute window and at least 5 of its 10 baseline windows (20 to 10 minutes before onset) have valid values for all four outcomes. Applying these rules gave 160 participants for the ordinary transition-matrix analysis (76 FS, 53 ES, 31 DD) with 465 seizures (148 FS, 185 ES, 132 DD), and 159 participants for the autonomic tensor analysis (75 FS, 53 ES, 31 DD) with 464 seizures. 

### Sample size rationale
Fixed existing dataset; no power analysis. Precision is reported through 95% confidence intervals, and a limited-feasibility conclusion is drawn if numbers or measurement quality are inadequate.

### Starting and stopping rules
N/A - a set amount of data has been collected.

## Variables
### Manipulated variables
None.


### Measured variables
EEG: microstate sequences and transitions. ECG: R-peaks and time-resolved autonomic indices (cardiac sympathetic index CSI and cardiac parasympathetic index CPI). Clinical: participant, diagnosis (functional seizures, epileptic seizures or both), event type, FS semiology (hypo- or hypermotor; exploratory analysis only), event onset time and segment type (interictal or preictal).

### Indices
Per 60-s window, relative to the participant's interictal reference transition grammar: (1) excess Jensen-Shannon divergence, the observed divergence minus its expected value if the window were drawn from the reference (estimated by simulation); (2) log kappa, the log precision (concentration) of the transition distribution relative to the reference. Each is computed for the autonomic-gradient tensor (primary) and the ordinary transition matrix (secondary). Windows are averaged within event and interval, then the early-to-final-minute change is averaged within participant.

## Analysis Plan
### Statistical models
Window outcomes are summarised to one value per participant, then compared with simple tests (full specification: docs/STATISTICAL_ANALYSIS.md and the repository code).
 - Mixed ANOVA (group × condition, condition = interictal mean versus final minute) per outcome (JS, kappa, CSI, CPI), computed as exact F-tests on one score per participant. Post-hoc: Welch for FS vs ES, paired t for final vs interictal, with Hedges’ g.
 - Welch t-test of the change (final minute minus the 20-10 minute baseline block) per outcome, with a Mann–Whitney sensitivity test. One-sample tests per group, descriptive only.
 - Linear mixed model (REML) of JS and kappa change over minutes 9 to 0. Fixed effects: time, group and time × group. Random intercept and slope per participant. The headline is the time × group term, with a likelihood-ratio test as backup.
 - Logistic regression of seizure type on z-scored JS change, z-scored a change (a = (CSI change - CPI change) / 2), and L = z_JS + z_a, in four models: JS, a, L and JS+a. Reports odds ratios, apparent, bootstrap-corrected and leave-one-out AUCs.

Multiplicity: Holm correction within each family; 4 outcomes in A1 and A2, 16 post-hoc tests in A1, 2 outcomes in A3, and 3 single-predictor models in A4.

### Transformations
Divergence is corrected by subtracting its expected value under no change (excess JS); kappa is analysed on the log scale. CPI/CSI slopes are standardised within recording to define the soft gradient quadrants. Group (FS/ES) and semiology (hypo/hyper, exploratory analysis only) are categorical; no other transformations or covariates.

### Inference criteria
Criteria for inference. All inference is frequentist and at the level of the participant. A result counts as statistically significant if its Holm-adjusted two-sided p-value is below .05 (α = .05). All tests are two-sided, with no one-tailed tests. No Bayes factors are used. Each effect is reported with a 95% confidence interval and an effect size (Hedges’ g, Hedges-corrected dz, partial η² or an odds ratio). Fit indices (AIC, McFadden and Nagelkerke R², AUC) are descriptive only, with no cut-off. A non-significant result is not read as evidence of no effect, and no equivalence tests are planned.

Headline test for each analysis. These are the tests that decide whether the hypothesis is supported:

Mixed ANOVA: the group × condition interaction.
Change test: the FS vs ES Welch test of change.
Mixed model: the time × group term, as a Wald z-test. The likelihood-ratio test is a sensitivity check.
Logistic regression: the likelihood-ratio p-value of each single-predictor model (JS, a, L).

Other tests are reported but not used as the main inference. These are the ANOVA main effects, the Mann–Whitney check, the within-group one-sample tests, the joint logistic model, the AUCs, and the interictal-reference, event-level and DD-paired analyses.

Multiplicity. Holm correction controls the family-wise error rate within each family:

4 outcomes for each ANOVA effect.
All 16 post-hoc tests, which are the only post-hoc results interpreted.
4 outcomes for the change test.
2 outcomes (JS, kappa) for the mixed-model time × group term.
3 single-predictor models for the logistic regression.

There is no further correction across the four analyses, because each addresses a separate question. The tensor representation is the primary analysis. The matrix representation is a secondary analysis with its own families and no correction across the two. Exploratory and sensitivity analyses are not adjusted and are not used for inference.

### Data inclusion and exclusion
Rule-based, group-blind and outcome-blind. Recordings and interictal/preictal segments failing pre-specified EEG/ECG quality checks are excluded; windows, events and participants need minimum amounts of usable transitions and autonomic data (thresholds fixed in the repository). No outcome-based exclusion, outlier removal or optional stopping. Eligibility is reported by group at every step. The majority of exclusions are expected to have been identified at the point of the analysis preregistration.

### Missing data
No imputation. Ineligible windows, events or participants are dropped from the affected test only, and the amount lost is reported by group.

### Other planned analysis
Secondary: the same analyses applied to ordinary transition matrices instead of autonomically-conditioned tensors (own multiplicity families; see Inference criteria). Reported without adjustment and not used for the main inference: the ANOVA main effects, the Mann-Whitney check, the within-group one-sample tests, the joint logistic model, the AUCs, and the interictal-reference, event-level and DD-paired analyses.

Exploratory: FS hypomotor versus hypermotor semiology. Among participants with eligible functional seizures (FS-only or DD), the change from the 20-10 minute block to the final minute is averaged per participant and compared between participants whose functional seizures are all hypomotor and those whose are all hypermotor (Welch t-test with Hedges' g, and a Mann-Whitney test), per outcome and representation. Participants with seizures of both semiologies are left out of the comparison and counted. Semiology is taken from the expert rater's entries in the clinical records: the recorded hypo/hyper category where there is one, otherwise the recorded seizure classification, mapped by a fixed rule (classifications naming a motor manifestation, such as generalised motor, focal motor, tonic or clonic, are hypermotor; akinetic, subjective, sensory, aura and subclinical events are hypomotor). The mapping was fixed after reviewing the distribution of recorded classifications and motor scores, not any EEG outcome. All eligible functional seizures have a category: 233 in the matrix analysis (98 hypomotor, 135 hypermotor) and 232 in the tensor analysis (98, 134). By participant, the matrix analysis has 42 hypomotor, 44 hypermotor and 19 with both (tensor: 42, 43 and 19). These p-values are unadjusted and are not used for inference.

## Other
### Context and additional information
The exact commit hash will be added as a comment on this registration. The statistics cannot run without an inclusion checkpoint bound to this registration's identifier, and any change to the release bundle after the checkpoint blocks them. The pipeline was developed and tested on synthetic data and on public TUEG recordings with synthetic labels; the study data were then processed up to the release bundle, as described under foreknowledge.
