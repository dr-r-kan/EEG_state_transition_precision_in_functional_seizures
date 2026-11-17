# FS_MS_Dynamics

Microstate transition dynamics and cardiac autonomic state before seizures in functional (FS) and epileptic (ES)
seizures.

The analysis asks how the organisation of EEG microstate transitions, and the cardiac sympathetic (CSI) and
parasympathetic (CPI) indices, change in the minutes before a seizure, and whether the change differs between
participants who have only functional seizures and participants who have only epileptic seizures. The work rests
on an active-inference account of dissociative symptoms: the transition grammar of a window is compared with the
participant's own interictal grammar, and described by two quantities.

* **JS** - how far the window's grammar has drifted from the interictal reference (null-calibrated Jensen-Shannon
  divergence, "excess JS");
* **kappa** - how precise (concentrated) the window's transition distributions are relative to that reference
  (inverse temperature, analysed as log kappa).

Both are computed for the ordinary K x K transition matrix and for a tensor in which every transition is split over
four CPI/CSI-gradient quadrants.

```
clips (EDF) -> HEPPy EEG/ECG preprocessing -> manifest
            -> microstates (K = 2..7, free-energy elbow, template matching)
            -> CSI/CPI native table
            -> window metrics (matrix, tensor, CSI, CPI)
            -> de-identified release bundle
            -> inclusion checkpoint   (needs the preregistration serial)
            -> statistics             (reads only the release bundle)
```

## Quick start

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v      # synthetic tests, no study data
.\run_pipeline.ps1 preflight --config config\study.json --deep      # environment, MATLAB, SPM, EEGLAB
```

Copy `config/study.example.json` to `config/study.json` and set the two paths. Then:

```powershell
# 1. everything up to the inclusion checkpoint (stops there and prints the final numbers)
.\run_pipeline.ps1 run --config config\study.json --prereg-id abc12

# 2. after reviewing the numbers: the statistics
.\run_pipeline.ps1 run --config config\study.json --prereg-id abc12 --through stats
```

`--prereg-id` is the serial of the preregistration (the five characters of the registration URL). It is checked,
written into the checkpoint and stamped on every statistical output; the statistics refuse to run without a
checkpoint for the same serial, and refuse to run if the release bundle changed after the checkpoint.

## What the statistics do

All analyses are at participant level, FS-only against ES-only participants, with the four outcomes JS, kappa, CSI
and CPI, run separately for the matrix and the tensor representation (see
[`docs/STATISTICAL_ANALYSIS.md`](docs/STATISTICAL_ANALYSIS.md)):

1. mixed ANOVA, group x condition (interictal vs the final minute before onset), with post-hoc tests;
2. change from the 20-10 minute block to the final minute, FS vs ES;
3. linear mixed model of the change in JS and kappa over the last ten minutes, one window per minute, random
   intercept and slope per participant;
4. logistic regression of seizure type on z-scored JS change, the autonomic arousal axis a = (CSI - CPI)/2, and
   their sum L.

Multiplicity is controlled with Holm's step-down procedure over the four (or two) outcomes.

An exploratory comparison of hypomotor and hypermotor functional seizures (E1) is reported without adjustment.

## Reproducibility and data release

* Every stage is idempotent and records a fingerprint of its parameters, code and upstream outputs.
* `run_root/release/` is a self-contained bundle ([`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md)) with
  pseudonymous identifiers only. A leakage audit scans it for source identifiers, file names, paths, e-mail
  addresses and dates, and fails the release if it finds any. The pseudonym key stays in `run_root/private/`.
* The statistics need only that bundle, so they can be rerun from the published files.
* Vendored third-party code is hashed in `external/PROVENANCE.json`; the Python environment is pinned in
  `requirements-lock.txt`.

## Layout

| Path | Purpose |
|---|---|
| `run_pipeline.py`, `run_pipeline.ps1` | single entry point |
| `fsms/pipeline.py`, `config.py`, `ledger.py`, `atomic.py` | stage runner, configuration, idempotency ledger |
| `fsms/heppy/`, `clips/`, `manifest/`, `autonomic.py` | preprocessing, manifest, CSI/CPI native table |
| `fsms/matlab/` | microstate stages (K selection, template matching, hierarchy, exact backfit) |
| `fsms/metrics.py`, `windows.py` | transition-grammar measures and onset-aligned windows |
| `fsms/eligibility.py`, `release.py`, `checkpoint.py` | inclusion rules, de-identified bundle, checkpoint |
| `fsms/stats/` | the four statistical analyses |
| `external/` | vendored HEPPy and VG-microstates with provenance |
| `docs/` | method, data dictionary, statistical analysis, runbook |
| `tests/` | synthetic unit and integration tests; opt-in slow fixture test |

Requires Windows, Python 3.12, MATLAB R2025a with [SPM](https://github.com/spm/spm) and EEGLAB.
External code: [HEPPy](https://github.com/dr-r-kan/HEPPy) and VG-microstates
(`dr-r-kan/VG-microstates`; the vendored tree is a modified derivative, see `external/VG-microstates/README.md`).
