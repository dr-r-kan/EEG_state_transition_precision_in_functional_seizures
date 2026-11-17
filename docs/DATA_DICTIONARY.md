# Release bundle: data dictionary

The pipeline writes one self-contained, de-identified bundle to `<run_root>/release/`. Every statistical
analysis reads **only** this bundle, so the statistics can be reproduced by anyone from the published files
without access to the recordings.

De-identification rules (enforced by `fsms/release.py` and audited by `leakage_audit.json`):

* participants and recordings carry random pseudonyms (`P001`, `P001_R03`); the key linking them to source
  identifiers is written to `<run_root>/private/` and is never part of the bundle;
* no file names, paths, clinical identifiers, dates, clock times, free-text semiology or raw signals;
* all times are relative to the recording or to seizure onset;
* the audit fails the release if any source identifier, file stem, drive path or absolute date string occurs in
  any bundle file.

## Time axis

Preictal recordings end at seizure onset. Windows are non-overlapping 60-s blocks aligned to onset. A window
is identified by `window_end_min`, the number of whole minutes between the end of the window and onset:

| `window_role` | `window_end_min` | span before onset |
|---|---|---|
| `baseline` | 19 ... 10 | 20 to 10 minutes (ten windows) |
| `minute` | 9 ... 0 | 10 minutes to onset, one window per minute (`0` = the final minute) |
| `interictal` | empty | non-overlapping 60-s windows of interictal recordings |

## `windows.csv.gz` (one row per window)

| column | meaning |
|---|---|
| `participant_id` | pseudonym |
| `recording_id` | pseudonym of the recording (one preictal recording = one seizure) |
| `group` | `FS`, `ES` or `DD` (participant diagnosis) |
| `event_type` | `FS` or `ES`: seizure type of a preictal recording (empty for interictal rows) |
| `period` | `interictal` or `preictal` |
| `window_role`, `window_end_min` | see above |
| `n_transitions` | genuine state transitions in the window |
| `n_action_transitions` | those with a valid autonomic gradient (used by the tensor) |
| `js_ordinary`, `js_null_ordinary`, `js_excess_ordinary` | Jensen-Shannon divergence of the window's K x K transition matrix from the participant's interictal grammar; its expectation under that grammar; the difference (the analysed outcome, "JS") |
| `kappa_ordinary`, `log_kappa_ordinary` | precision of the window's transition distribution relative to the interictal grammar (inverse temperature) and its logarithm ("kappa") |
| `js_excess_tensor`, `log_kappa_tensor` | the same quantities for the autonomic-conditioned transition tensor |
| `csi_z`, `cpi_z` | window means of the cardiac sympathetic and parasympathetic indices, standardised to the participant's interictal recordings |
| `csi_gradient`, `cpi_gradient` | window means of the index gradients (z per second) |

Rows exist only for windows with at least 30 transitions and an interictal reference of at least 300
transitions (the evaluated window is left out of its own reference). Missing metrics are empty.

## `events.csv` (one row per preictal recording)

`recording_id`, `participant_id`, `group`, `event_type`, `semiology`, `n_baseline_windows_matrix`, `n_baseline_windows_tensor`,
`has_final_minute_matrix`, `has_final_minute_tensor`, `eligible_matrix`, `eligible_tensor`, `exclusion_reason_matrix`,
`exclusion_reason_tensor`.

`semiology` is the semiology of the seizure as a category: `hypo` (hypomotor), `hyper` (hypermotor) or
`unclassified`. It is read from the originating clinical workbooks by the `labels` stage (`fsms/labels.py`), not from
the compiled database: the rater's hypo/hyper entry where the workbook has one, otherwise the rater's seizure
classification mapped by a fixed rule. A classification that names a motor manifestation (motor, tonic, clonic, myoclonic, hypermotor, automotor, gelastic, secondary or bilateral generalisation) is `hyper`; otherwise one that names an akinetic, atonic, absence, subjective, sensory, aura, psychic, subclinical or electrographic event is `hypo`; anything else is `unclassified`. Every seizure in the bundle has a label row; the release stage stops
if one does not. Only the category is released, never the source text. It is used only by the exploratory analysis
E1 and plays no part in eligibility.

A seizure is eligible for a representation (matrix or tensor) when its final-minute window has all four outcomes
(JS, kappa, CSI, CPI) for that representation and at least 5 of its ten baseline windows do too. The analysis of
each representation uses only its own eligible events, so the four outcomes always rest on the same sample.

## `participants.csv` (one row per participant)

`participant_id`, `group`, `n_interictal_recordings`, `n_interictal_windows`, `n_interictal_transitions`,
`n_preictal_recordings`, `n_eligible_events_matrix`, `n_eligible_events_tensor`, `eligible_matrix`,
`eligible_tensor`, `exclusion_reason_matrix`, `exclusion_reason_tensor`.

A participant is eligible for a representation when they have an interictal reference grammar (at least 300
transitions), at least 5 valid interictal windows for that representation, and at least one eligible seizure.

## `reference_grammar.csv`

Per participant, the K x K interictal transition counts that define the reference grammar:
`participant_id`, `from_state`, `to_state`, `count`.

## `window_transition_counts.csv.gz`

Non-zero transition counts per window, so every metric can be recomputed:
`participant_id`, `recording_id`, `window_role`, `window_end_min`, `from_state`, `to_state`, `count`,
`weight_a1` ... `weight_a4` (soft autonomic-gradient quadrant mass; the tensor counts).

## `microstate_solution/`

* `k_selection.csv`: for every candidate K (2 to 7) the variational free energy of the best random start, the
  elbow score, a `selected` flag, the number of starts and the median free energy over the starts;
  `k_selection.json` gives the rule and the chosen K.
* `free_energy_starts.csv`: `K`, `start`, `seed`, `free_energy`, `best` for every random initialisation of every
  candidate K (`best` marks the start that was kept).
* `state_labels.csv`: per fitted state, the best-matching template class, its spatial correlation and a
  `below_threshold` flag (matching never forces a class to be present; unmatched states keep their best match
  and are flagged).
* `global_centres.csv`: the global microstate maps (channel labels x states), no individual data.

## `inclusion_checkpoint.json`

Written by the `checkpoint` stage: the eligibility rules, the numbers of eligible participants and seizures per
group (and every exclusion reason with its count), the eligible functional seizures per semiology category, and the preregistration identifier supplied by the analyst.
The statistics stage refuses to run without a checkpoint for the same identifier.

## `MANIFEST.json`

SHA-256 and size of every bundle file, the code version, the software versions and the preregistration
identifier.
