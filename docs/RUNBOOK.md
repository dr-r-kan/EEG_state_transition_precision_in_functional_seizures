# Runbook

## 1. Environment

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\run_pipeline.ps1 preflight --config config\study.json --deep
```

`preflight` checks the Python version, the pinned packages, the hashes of the vendored code, MATLAB on `PATH`,
free disk space and (with `--deep`) that SPM (`spm_mix`) and EEGLAB (`pop_loadset`) are on the MATLAB path.
It reads no study data.

## 2. Configuration

Copy `config/study.example.json` to `config/study.json` (ignored by git) and set:

* `paths.data_root`: folder with `clips/*.edf`, `clips/snip_keyfile.csv` and `analysis_database.csv`;
* `paths.run_root`: where everything is written (never inside `data_root`);
* `sources.workbooks`: the originating clinical workbooks, one per `source` value of the database. The `labels`
  stage checks every seizure row of the database against them (same patient, group and seizure type) and takes the
  semiology from them. Without this section the labels come from the database and nothing is verified;
* `paths.existing_heppy_dir` with `heppy.use_existing_output: true` to reuse an existing HEPPy output tree in
  place (read-only) instead of preprocessing again.

The `analysis` section holds the eligibility constants; the defaults are listed in `fsms/config.py` and recorded in
the checkpoint.

## 3. Run to the checkpoint

```powershell
.\run_pipeline.ps1 run --config config\study.json --prereg-id abc12
```

Stages run in order and each is skipped when its fingerprint is current:
`clips_check, heppy, manifest, microstates, autonomic, metrics, labels, release, checkpoint`.
`labels` depends only on the database and the workbooks, and `release` on the content of the label table, so a
corrected label reruns those two stages (about a minute) and nothing upstream.
The run stops after `checkpoint` and prints, per representation (matrix, tensor) and group, the participants and
seizures that enter the statistics, with every exclusion reason. The same information is in
`run_root/release/inclusion_checkpoint.json`.

`--prereg-id` is the five-character serial of the preregistration (or its URL). It is required from the checkpoint
onwards.

## 4. Run the statistics

```powershell
.\run_pipeline.ps1 run --config config\study.json --prereg-id abc12 --through stats
```

The statistics refuse to run unless the checkpoint exists for the same serial and the release bundle is unchanged
since. Results and figures are written to `run_root/results/<representation>/`; the leakage audit is repeated on
them.

## 5. Publishing

Publish `run_root/release/` (data) and `run_root/results/` (outputs). Never publish `run_root/private/`
(pseudonym key), `analysis/` (source identifiers), `heppy_output/`, `vg_microstates/` or `logs/`.

## 6. Reruns and changes

* A rerun with nothing changed executes nothing.
* Changing a config section, code file, vendored code or an upstream output reruns the affected stage and
  everything downstream. `--force STAGE` reruns one stage.
* `python run_pipeline.py status --config config\study.json` lists the completion time of each stage.
* A change to the release bundle after the checkpoint invalidates the checkpoint; rerun the pipeline through
  `checkpoint` and review the numbers again.

## 7. Tests

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

The synthetic tests need no study data. The opt-in end-to-end test uses Temple University EEG recordings with
synthetic labels: set `FSMS_RUN_SLOW=1` and `FSMS_TUEG_ROOT` to the TUEG folder, then
`.\run_pipeline.ps1 fixture --work-root <dir>`.
