# Running the checkpoint and the statistics from the terminal

Everything up to and including the release bundle (`release` stage) is already built. This note covers the last two
stages, `checkpoint` and `stats`, which are bound to the preregistration serial. Run all commands in PowerShell from
the repository root (`FS_MS_Dynamics`). Nothing here needs MATLAB, SPM or the raw recordings.

Replace `abc12` below with the serial of the new registration (five characters, or its full OSF URL).

## 0. Before you start

Check that the earlier stages are current:

```powershell
.\.venv\Scripts\python.exe -B run_pipeline.py status --config config\study.json
```

`clips_check` ... `release` should all show a completion time. `checkpoint` and `stats` show `never run` until you
run them. Do not use the serial of the earlier registration.

If you changed any code or config since the last run, the command in step 1 re-runs only the affected stages. A
change to a MATLAB or microstate setting re-runs the microstates (about two hours), so do not change those between
registering and running.

## 1. Run to the checkpoint

```powershell
.\run_pipeline.ps1 run --config config\study.json --prereg-id abc12
```

Expected output: `[SKIP]` for every earlier stage, then `[RUN ] checkpoint` and a table. For each representation
(`matrix`, `tensor`) it lists the participants and seizures that will enter the statistics, per group (FS, ES, DD),
the dual-diagnosis participants who have both seizure types, and every exclusion reason with its count. The selected
K is printed at the top.

The same information is saved in `F:\FS_MS_analysis\dynamics_run\release\inclusion_checkpoint.json`. 

## 2. Run the statistics

Only after you have reviewed the checkpoint:

```powershell
.\run_pipeline.ps1 run --config config\study.json --prereg-id abc12 --through stats
```

The statistics refuse to run unless a checkpoint exists for the same serial and the release bundle is unchanged
since it was written. They then:

* A1: mixed ANOVA of group x condition (interictal vs final minute) with post-hoc tests;
* A2: FS vs ES comparison of the change from baseline to the final minute (t-tests);
* A3: linear mixed model of the change in JS and kappa over minute windows 9 to 0, with random intercepts and slopes;
* A4: logistic regression of event type on z-scored JS, the autonomic axis `a` and L.

Each runs for the matrix and the tensor representation. The bootstrap for the AUC takes the longest.

Results are written to `F:\FS_MS_analysis\dynamics_run\results\`:

* `results.json` and `results_summary.md`: headline numbers, Holm families, sample sizes, input hashes;
* `matrix\` and `tensor\`: `tables\*.csv` and `figures\*.png` (every file carries the serial);
* a repeat of the leakage audit over the results.

Column definitions for each table are in `docs\STATISTICAL_ANALYSIS.md`.

## 3. Running the statistics module directly (optional)

This bypasses the pipeline ledger and the leakage audit of the results, but keeps the checkpoint gate. It is useful
for a separate output folder, or for fewer bootstrap replicates when testing:

```powershell
.\.venv\Scripts\python.exe -B -m fsms.stats --release F:\FS_MS_analysis\dynamics_run\release --out F:\FS_MS_analysis\dynamics_run\results_test --prereg-id abc12 --representations matrix --n-bootstrap 200
```

Use the pipeline command in step 2 for the registered analysis.

## 4. Re-running and errors

* A repeated command with nothing changed executes nothing (`[SKIP] ... (current)`).
* `--force stats` re-runs the statistics only: `.\run_pipeline.ps1 run --config config\study.json --prereg-id abc12 --through stats --force stats`.
* `ERROR: Reaching the checkpoint needs --prereg-id` means the serial was omitted.
* `The checkpoint belongs to preregistration ...` means the serial differs from the one the checkpoint was written
  with. Re-run step 1 with the intended serial.
* Only one pipeline process can run at a time; `Another pipeline process is already running` means close the other
  window first.
* Exit codes: 0 ok, 1 failed check, 2 refused or failed. The log of each stage is in `F:\FS_MS_analysis\dynamics_run\logs\`.

## 5. Publishing

Publish `release\` and `results\`. Never publish `private\` (pseudonym key), `analysis\`, `heppy_output\`,
`vg_microstates\` or `logs\`.
