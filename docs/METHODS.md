# Methods

## Preprocessing (HEPPy)

EEG and ECG clips are preprocessed with the vendored HEPPy stage: band-pass and mains filtering, PyPREP bad-channel
handling, ICA with ICLabel for cardiac-field and other artefacts, and R-peak detection with automatic correction
(corrected peaks are written as sidecars). Recordings are excluded by a group-blind, outcome-blind quality rule
(`qc_rule` in `config/heppy_options.json`); every exclusion carries a reason code.

## Microstates (VG-microstates, `fsms/matlab`)

1. GFP-peak maps of all included recordings are pooled; a global variational Bayesian Gaussian mixture
   (SPM `spm_mix`) is fitted for **every K from 2 to 7**. Because `spm_mix` starts from random k-means and can land
   in a poor local optimum, each K is fitted from 20 seeded random initialisations and the fit with the highest free
   energy is kept (the free energy of every start is saved in `microstate_solution/free_energy_starts.csv`).
2. **K is chosen by the free-energy elbow**: free energy and K are scaled to [0, 1]; the score of a candidate is its
   distance above the chord joining the first and the last candidate; the largest score wins. If no candidate lies
   above the chord, or fewer than three fits are valid, the candidate with the largest free energy is chosen. The whole
   curve is saved (`microstate_solution/k_selection.csv`).
3. **Template matching does not force any class to be present.** The K fitted maps are matched to the MetaMaps
   template classes A-G by the exact assignment that maximises the summed absolute spatial correlation (polarity
   invariant). Each state receives its best available distinct class; a match with |r| < 0.5 is flagged
   (`below_threshold`) but kept. Class names are nominal and no analysis depends on them.
4. Group, condition and participant solutions are fitted with the global solution as prior at the selected K. Each
   participant's solution is backfitted to the GFP peaks of every recording; the genuine ordered state changes
   (sample, from, to) are exported exactly, and the export is validated against a full sample-wise backfit and against
   an independent re-extraction of the GFP peaks.

## Autonomic indices

CSI and CPI are computed from corrected R-peaks in 15-s windows (robust Poincare features with shrinkage
covariance, Candia-Rivera et al.) and standardised to the participant's interictal recordings. CSI and CPI are linearly
interpolated to each transition within contiguous runs (gaps over 5 s are never bridged); gradients are the
corresponding slopes. The autonomic arousal axis used in the statistics is `a = (CSI - CPI) / 2`.

## Windows

Preictal recordings end at seizure onset. Non-overlapping 60-s windows are aligned to onset: the ten windows from 20 to
10 minutes before onset form the *baseline* block, the ten windows from 10 minutes to onset the *minute* windows (the
last is the final minute). Interictal recordings are cut into consecutive 60-s windows. A window needs at least 30
transitions.

## Transition grammar

For each window, the K x K matrix of destination counts per source state (no self transitions) is compared with the
participant's **interictal reference grammar** (all interictal windows pooled; an interictal window is evaluated
against the reference without itself; the reference needs at least 300 transitions).

* **JS (excess)**: weighted Jensen-Shannon divergence between the window's destination distributions and the reference
  rows, minus its expectation under the reference given the window's own source sequence (400 hash-seeded parametric
  bootstrap draws), which removes the small-sample bias.
* **kappa**: the inverse temperature `kappa` that maximises the likelihood of the window under
  `softmax(kappa * log reference)`; analysed as `log kappa` (0 = as precise as the reference).
* **Tensor**: each transition is split over four soft quadrants of the CPI and CSI gradients (increasing/decreasing);
  the tensor reference is the interictal tensor shrunk towards the ordinary matrix with 10 pseudo-counts per row. JS
  and kappa are computed on the tensor in the same way and need at least 30 transitions with a valid gradient.

## Event labels

Diagnostic group, seizure type and semiology are clinical labels recorded in two workbooks from which the analysis
database was compiled. The `labels` stage reads the workbooks again and, for every seizure row of the database,
confirms that the workbook row it came from is the same patient with the same group and seizure type; any
disagreement stops the run. Semiology (hypomotor / hypermotor) is taken from the rater's hypo/hyper entry in one
workbook and from the rater's seizure classification in the other, by a fixed rule. A classification that names a motor manifestation (motor, tonic, clonic, myoclonic, hypermotor, automotor, gelastic, secondary or bilateral generalisation) is `hyper`; otherwise one that names an akinetic, atonic, absence, subjective, sensory, aura, psychic, subclinical or electrographic event is `hypo`; anything else is `unclassified`. Labels are joined to the
bundle at the release stage only: preprocessing, microstate fitting and the window metrics never read them.

## Eligibility, release and checkpoint

See `fsms/eligibility.py`, `docs/DATA_DICTIONARY.md` and `docs/RUNBOOK.md`. A seizure is eligible for a representation
when its final minute and at least five of its ten baseline windows have JS, kappa, CSI and CPI. The number of
participants and seizures that satisfy the rules is fixed in the inclusion checkpoint before any statistic is computed.
