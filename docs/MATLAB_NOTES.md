# MATLAB stage: microstate hierarchy and exact transitions

The MATLAB code lives in `fsms/matlab/` (pipeline functions) and `external/VG-microstates/` (the vendored VB-GMM
microstate fitter, see its README). The Python orchestrator calls one entry point headlessly.

## Entry point

```matlab
result = run_study_hierarchy(manifest_csv, out_dir, Name, Value, ...)
```

```
matlab -batch "addpath('<repo>/fsms/matlab'); run_study_hierarchy('<manifest.csv>', '<out_dir>')"
```

SPM (with `toolbox/mixture`) must be on the MATLAB path; `fsms_paths` adds the repository code itself.

| Option | Default | Meaning |
| --- | --- | --- |
| `K_candidates` | `2:7` | candidate numbers of microstates, each 2..7 |
| `criterion` | `'free_energy_elbow'` | K-selection rule (`'free_energy'` = largest free energy) |
| `n_starts` | `20` | random initialisations of the global fit per K; the start with the highest free energy is kept |
| `validation_windows_csv` | `<out_dir>/validation/directional_peak_backfit_windows.csv` | windows export |
| `conditions` | `{'II','preict'}` | conditions exported as transition caches |
| `min_peaks_per_record` | `20` | GFP-peak minimum for a record |
| `min_peaks_per_participant` | `[]` (= 10 x max K candidate) | GFP-peak minimum for a participant |
| `validation_records` | `'all'` | records re-extracted for the window gate (`'first_per_participant'`) |
| `full_backfit_validation` | `'first_per_participant'` | records whose compact export is compared with the full backfit (`'first'`, `'all'`) |
| `max_records` | `Inf` | debugging only |
| `hierarchy_args` | `{}` | extra name-value pairs for the vendored hierarchy (diagnostics; part of the fingerprint) |

`manifest_csv` needs `participant, condition, group, file_path, crop_start_seconds, crop_end_seconds`.
`out_dir` must be empty or have been written by the same call (same manifest, options and sources);
otherwise the call errors (`FSMS:StaleOutputDir`, `FSMS:StageInputsChanged`, `FSMS:UnstampedOutputDir`).
A repeated call on a finished folder verifies the stored hashes and returns immediately (`result.skipped = true`).
If only derived files (windows CSV, transition caches, index, audit) are missing or changed, the call heals them;
a changed frozen file (hierarchy, labels, K selection, manifests) is an error.

The returned struct has `out_dir, hierarchy_dir, backfit_dir, transition_index, recovery_audit, fit_labels,
selected_K, labelling_quality, n_records_analysed, n_excluded_records, inputs_fingerprint, done_file, skipped`.

## K selection

The global VB-GMM is fitted for every K in `K_candidates` (candidates that are not smaller than the number of maps
are dropped with a warning). SPM's `spm_mix` starts from random k-means, so each K is fitted from `n_starts`
seeded initialisations (start `s` uses seed `random_seed + K + 1000*(s-1)`) and the fit with the highest free
energy (the first start on a tie) is kept; its free energy is the K's value on the curve. Starts that fail or give a
non-finite free energy do not compete. The same K therefore always gives the same solution, and the free energy of
every start is written to `free_energy_starts.csv` (`K, start, seed, free_energy, best`). Only the global fit is
restarted: group, condition and participant fits are seeded by their parent's maps.
`free_energy_elbow` (`_internal/matlab/select_spm_vb_k_by_criterion.m`):

1. candidates with a non-finite or zero free energy (failed fits) are ignored;
2. K and free energy of the rest are min-max scaled to [0, 1];
3. the elbow score of a candidate is the perpendicular distance of its scaled point *above* the chord joining the
   first and the last candidate (a point on or below the chord scores 0, so a dip in free energy is never an
   elbow); the end points score 0, so the smallest and largest candidate are never chosen by the elbow itself;
4. the candidate with the largest score is selected; ties go to the smallest K;
5. if fewer than three candidates are valid, or all scores are <= eps, the candidate with the largest free energy is
   selected.

Every group, condition and participant fit and all transition exports use the selected K.

## Labelling

`global_state_labels` matches the K fitted global states to the MetaMaps K = 7 template (classes A-G) with the exact
optimal one-to-one assignment maximising the summed polarity-invariant |spatial correlation|
(`match_template_labels`). All seven classes are available; no class is required or excluded. Ties within 1e-9 take the
lexicographically smallest assignment vector (and are reported as a technical failure, since they only occur for
identical maps). A state whose assigned |r| is below 0.5 is flagged (`below_threshold`) and keeps its label; the run
continues. Technical failures (channel matching other than by labels or coordinate interpolation, fewer than 8
channels, non-finite maps, K outside 2..7) write `global/LABELLING_REJECTED.json` and stop the stage.

## Output layout (`<out_dir>`)

```
STAGE_C_INPUTS.json  STAGE_C_DONE.json  STAGE_C_FAILED.json (only after an error)
manifest_full.csv  analysis_manifest.csv  exclusions.csv  screening.json  screening_peak_counts.csv
screening_peaks/                          peak extraction used for the peak-count screening
hierarchy/                                <hierarchy_root>/hierarchy
  global/ groups/ conditions/ participants/ gfp_peaks/ ...
validation/directional_peak_backfit_windows.csv
timestamped_backfits/
  transition_index.csv
  recovery_audit.json                     contains K and fit_labels (class label per fitted-state index)
  transition_cache/<NNNN>_<participant>_<condition>_transitions.mat
```

`transition_cache/*.mat` holds one `transition` struct per record (`sfreq, n_samples, transition_sample, from_state,
to_state, ...`); `from_state`/`to_state` are 1-based fitted-state indices, `fit_labels{i}` is the class of state i.
`STAGE_C_DONE.json` repeats `K`/`selected_K`, `fit_labels` and `labelling_quality`.

### Files in `hierarchy/global/` (the release bundle copies the first four)

| File | Content |
| --- | --- |
| `k_selection.csv` | `K, free_energy, elbow_score, selected, n_starts, free_energy_median`; one row per fitted candidate; `free_energy` is the best start, `selected` is 1 for the chosen K |
| `free_energy_starts.csv` | `K, start, seed, free_energy, best`; every random start of every candidate K |
| `k_selection.json` | `rule` (text), `criterion`, `selected_K`, `n_starts`, `start_rule`, `labels` (class label per fitted state, state 1..K), `K_requested`, `K_fitted`, `degenerate_curve`, `n_maps`, `curve` |
| `state_labels.csv` | `fitted_state, label, abs_r, below_threshold, runner_up_label` (runner-up = best other class by \|r\|) |
| `global_centres.csv` | long format, columns `channel, state_label, fitted_state, value`: the selected global maps, one row per state and channel. Only standard 10-20/10-10 electrode labels are written. Each map is zero-mean and unit-norm with polarity fixed to the matched template. Contains no participant data. |
| `labelling_audit.json` | rule, template and script SHA-256, full correlation matrix (K x 7), assignment, quality, channel-matching mode, MATLAB/SPM versions |
| `global_manual_labels.csv/.mat` | labels in the format the vendored hierarchy reads back |
| `global_solution.mat` | `Fit` (all candidate solutions in `Fit.centers_by_K`), `common_labels`, `common_chanlocs`, `cfg` |

`global_centers.csv` (US spelling) is the vendored raw wide matrix without channel names; it is not part of the release.

## Tests

```matlab
addpath('fsms/matlab/tests'); test_match_template_labels
```

builds synthetic states from the template (e.g. K = 4 from classes B, D, F, G; K = 7 with all classes), and checks that
no class is forced, that the assignment equals the optimum found by brute force and `matchpairs`, that weak matches are
flagged without relabelling, and that technical failures are reported.
