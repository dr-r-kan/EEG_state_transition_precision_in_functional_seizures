# VG-microstates (vendored subset)

This folder holds the part of [dr-r-kan/VG-microstates](https://github.com/dr-r-kan/VG-microstates) that the
FS/ES transition pipeline needs. It is a **modified derivative** of that repository (MIT licence, see
`LICENSE`; citation metadata in `CITATION.cff`). The upstream project provides the SPM variational-Bayes
Gaussian-mixture (VB-GMM) microstate fitter, the MetaMaps template alignment and a hierarchical
(global, group, condition, participant) fitting scheme.

## What is vendored

| Path | Role |
| --- | --- |
| `fit_microstate_hierarchical_csv.m` | entry point: manifest CSV, name-value options |
| `fit_microstate_hierarchical_dataset.m` | GFP-peak extraction, global fit for every candidate K, K selection, child fits, narrow commands `extract_peak_record` and `timestamped_transitions` |
| `setup_microstate_paths.m` | adds the helper folders to the MATLAB path |
| `_internal/matlab/align_microstates_to_template.m`, `load_metamaps_templates.m` | polarity-invariant template correlation, MetaMaps loader |
| `_internal/matlab/backfit_microstate_timecourse.m` | full-record hard backfit (used to validate the exact transition export) |
| `_internal/matlab/microstate_partial_alignment.m` | one-to-one alignment of child solutions to the global solution |
| `_internal/matlab/select_spm_vb_k_by_criterion.m` | K selection from the free energy (elbow rule, documented in the file) |
| `_internal/matlab/spm_mix_prior.m` | SPM mixture fit with component-mean priors (needs SPM12/SPM25+ with `toolbox/mixture`) |
| `_internal/matlab/microstate_utilities.m` | preprocessing, channel and configuration helpers |
| `_internal/matlab/compat/spm_bsd/` | compatibility `spm_logdet` |
| `config/microstate_config.json` | preprocessing and hierarchy defaults |
| `MetaMaps_2023_06.set` | MetaMaps microstate templates, classes A-G (K = 7 solution used for labelling) |

SPM and EEGLAB are not vendored; SPM must be on the MATLAB path (EEGLAB is only needed to read `.set` files).

## Functional changes relative to upstream

1. **K is selected, not fixed.** The global VB-GMM is fitted for every K in `K_candidates` (default 2:7) and K
   is chosen by the `free_energy_elbow` criterion: the perpendicular distance of the min-max scaled (K, free
   energy) points to the chord between the first and last candidate, largest distance wins, ties to the
   smallest K, free-energy maximum if the curve is degenerate. All group, condition and participant fits use
   the selected K. The per-K table is stored in `Fit.model_comparison` (columns K, free_energy,
   selection_score, selected, ...). Candidates K that are not smaller than the number of maps are dropped
   with a warning.
   **Multiple starts.** SPM's `spm_mix` starts from random k-means, so a single fit can land in a poor local
   optimum. The global fit (the only one without a parent solution) is run from `n_starts` seeded
   initialisations per K (start `s` uses seed `random_seed + K + 1000*(s-1)`); the start with the highest free
   energy is kept and its free energy enters the K-selection curve. The free energy of every start is stored
   in `Fit.free_energy_starts` (K candidates x starts). Group, condition and participant fits are seeded by
   their parent's maps and are fitted once. The default is `n_starts = 1`.
2. **No class is required.** Global labels are any K unique letters from A-G (K = 2..7). The requirement that
   the label set contains A, C and E was removed, as was the assumption of exactly six states.
3. **Narrower scope.** Removed: simulation benchmark and analysis scripts, K-means baseline, single-file
   analysis, TANOVA/topoplot reports, cardiac-phase analyses, pre-ictal window refitting, morphology and
   axis-dynamics tables, backfit metric tables, plotting, interactive labelling, folder discovery of EEG
   files and digitised-localizer lookup. The entry point reads a CSV manifest only.
4. **Hard-coded provenance strings** for the original development environment (SPM commit hashes) were
   removed; the SPM version in use is recorded by the calling pipeline.
5. Prior-source labels in the hierarchy summary are `global`, `group`, `condition`, `participant`.
6. `config/microstate_config.json` and the built-in defaults were reduced to the options the hierarchy uses.
