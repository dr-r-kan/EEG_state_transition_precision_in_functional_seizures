"""Stage DAG and runner.

    clips_check -> heppy -> manifest -> microstates -> autonomic -> metrics -> labels -> release -> checkpoint -> stats

Every stage is idempotent: it records a fingerprint (config section + code hashes + upstream fingerprints +
input listing) and its outputs in ``<run_root>/state/<stage>.done.json``; a rerun with nothing changed executes
nothing.  A changed stage reruns from scratch and invalidates everything downstream.

The ``checkpoint`` stage fixes the final number of participants and seizures and is bound to the preregistration
identifier; ``stats`` only runs after it, for the same identifier, and is requested explicitly with
``--through stats``.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd
import psutil

from . import checkpoint as cp
from . import provenance
from .atomic import atomic_write_json, fingerprint, read_json, sha256_file
from .config import DEFAULT_N_STARTS, Config
from .ledger import Ledger, Timer

STAGE_ORDER = ("clips_check", "heppy", "manifest", "microstates", "autonomic", "metrics", "labels", "release",
               "checkpoint", "stats")
PYTHON = sys.executable


class PipelineError(RuntimeError):
    pass


# ------------------------------------------------------------------ helpers
def code_hash(config: Config, *patterns: str) -> str:
    items = {}
    for pattern in patterns:
        for path in sorted(config.repo.glob(pattern)):
            if path.is_file() and "__pycache__" not in path.parts:
                items[path.relative_to(config.repo).as_posix()] = sha256_file(path)
    if not items:
        raise PipelineError(f"No code files match {patterns}")
    return fingerprint(items)


def file_listing(directory: Path, suffix: str) -> list[list]:
    """Name and size only: clips can be tens of GB; per-clip sentinels hash what matters."""
    return sorted([p.name, p.stat().st_size] for p in directory.glob(f"*{suffix}") if p.is_file())


def csv_column(path: Path, column: str) -> list[str]:
    import csv
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [row[column] for row in csv.DictReader(handle) if row.get(column)]


def existing(paths) -> list[Path]:
    return [Path(p) for p in paths if p and Path(p).is_file()]


def run_logged(command: list[str], log_path: Path, cwd: Path, env: dict | None = None) -> int:
    """Run a subprocess, mirroring combined output to the console and a log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = {**os.environ, "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1", **(env or {})}
    with log_path.open("a", encoding="utf-8", errors="replace") as log:
        log.write(f"\n===== {datetime.now(timezone.utc).isoformat()} :: {' '.join(command)}\n")
        log.flush()
        process = subprocess.Popen(command, cwd=cwd, env=environment, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")

        def pump():
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                log.flush()
                print("   | " + line.rstrip(), flush=True)

        thread = threading.Thread(target=pump, daemon=True)
        thread.start()
        code = process.wait()
        thread.join()
        log.write(f"===== exit {code}\n")
    return code


class RunLock:
    """One pipeline run per run_root; stale locks from dead processes are reclaimed."""

    def __init__(self, directory: Path):
        self.path = directory / ".run.lock"
        self.directory = directory

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                handle = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(handle, json.dumps({"pid": os.getpid(), "created": time.time()}).encode())
                os.close(handle)
                return self
            except FileExistsError:
                try:
                    pid = json.loads(self.path.read_text()).get("pid")
                except (OSError, ValueError):
                    pid = None
                if pid and psutil.pid_exists(pid):
                    raise PipelineError(f"Another pipeline run (pid {pid}) is using this run_root")
                self.path.unlink(missing_ok=True)
        raise PipelineError("Could not acquire the run lock")

    def __exit__(self, *exc):
        self.path.unlink(missing_ok=True)
        return False


@dataclass
class Stage:
    name: str
    fingerprint: Callable[["Context"], str]
    run: Callable[["Context"], None]
    outputs: Callable[["Context"], list[Path]]


@dataclass
class Context:
    config: Config
    ledger: Ledger
    prereg_id: str | None = None
    fingerprints: dict = field(default_factory=dict)

    def python_module(self, module: str, *args: str, stage: str) -> int:
        return run_logged([PYTHON, "-B", "-m", module, *args], self.config.log_dir / f"{stage}.log", self.config.repo)


# ------------------------------------------------------------------ stage implementations
def _clips_fp(ctx: Context) -> str:
    c = ctx.config
    return fingerprint({"clips": file_listing(c.clips_dir, ".edf"), "database": sha256_file(c.database_csv),
                        "keyfile": sha256_file(c.snip_keyfile)})


def _clips_run(ctx: Context) -> None:
    c = ctx.config
    missing = [str(p) for p in (c.clips_dir, c.database_csv, c.snip_keyfile) if not p.exists()]
    if missing:
        raise PipelineError("Missing inputs: " + ", ".join(missing))
    if not list(c.clips_dir.glob("*.edf")):
        raise PipelineError(f"No clip EDFs in {c.clips_dir}")
    atomic_write_json(c.state_dir / "clips_check.json", {"edf_clips": len(list(c.clips_dir.glob('*.edf')))})


def _heppy_options(c: Config) -> Path:
    return c.repo_file(c.get("heppy.options"))


def _heppy_fp(ctx: Context) -> str:
    c = ctx.config
    return fingerprint({"upstream": ctx.fingerprints["clips_check"], "options": sha256_file(_heppy_options(c)),
                        "code": code_hash(c, "fsms/heppy/*.py", "fsms/atomic.py"),
                        "vendored": provenance.tree_hash(c.repo / "external" / "HEPPy")[0],
                        "existing": bool(c.get("heppy.use_existing_output", False))})


def _heppy_run(ctx: Context) -> None:
    c = ctx.config
    args = ["--clips-dir", str(c.clips_dir), "--heppy-out", str(c.heppy_out), "--options", str(_heppy_options(c)),
            "--workers", str(c.get("heppy.workers", 1))]
    if c.get("heppy.use_existing_output", False) and c.existing_heppy_dir:
        args += ["--adopt-dir", str(c.existing_heppy_dir)]
    if ctx.python_module("fsms.heppy.run_heppy_stage", *args, stage="heppy") != 0:
        raise PipelineError("Preprocessing stage failed; see logs/heppy.log")


def _heppy_outputs(ctx: Context) -> list[Path]:
    """Summary files plus every per-recording product, so deleting one FIF or sidecar makes the stage rerun."""
    from .autonomic import sidecar_for
    review = ctx.config.heppy_out / "ecg_qc_review.csv"
    products: list[str] = []
    for column in ("raw_fif", "raw_fif_keep"):
        products += csv_column(review, column)
    products += [str(sidecar_for(p)) for p in csv_column(review, "raw_fif")]
    return [ctx.config.heppy_out / "heppy_summary.json", review, *existing(products)]


def _manifest_fp(ctx: Context) -> str:
    c = ctx.config
    return fingerprint({"upstream": ctx.fingerprints["heppy"], "manifest": c.get("manifest"),
                        "code": code_hash(c, "fsms/manifest/*.py"),
                        "database": sha256_file(c.database_csv), "keyfile": sha256_file(c.snip_keyfile),
                        "heppy_summary": sha256_file(c.heppy_out / "heppy_summary.json")})


def _manifest_run(ctx: Context) -> None:
    c = ctx.config
    args = ["--snip-keyfile", str(c.snip_keyfile), "--heppy-out", str(c.heppy_out), "--database", str(c.database_csv),
            "--out-dir", str(c.vg_dir), "--format", c.get("manifest.format", "mat"),
            "--montage", c.get("manifest.montage", "standard_1020"),
            "--min-duration-s", str(c.get("manifest.min_duration_s", 10))]
    if ctx.python_module("fsms.manifest.build_vg_manifest", *args, stage="manifest") != 0:
        raise PipelineError("Manifest stage failed; see logs/manifest.log")


def _manifest_outputs(ctx: Context) -> list[Path]:
    return [ctx.config.manifest_mat, *existing(csv_column(ctx.config.manifest_mat, "file_path"))]


def hierarchy_dir(c: Config) -> Path:
    params = fingerprint({"manifest": sha256_file(c.manifest_mat), "microstates": c.get("microstates", {}),
                          "n_starts": c.get("microstates.n_starts", DEFAULT_N_STARTS),
                          "vg": provenance.tree_hash(c.repo / "external" / "VG-microstates")[0],
                          "code": code_hash(c, "fsms/matlab/*.m")})
    return c.vg_dir / f"hierarchy_{params[:12]}"


def _micro_fp(ctx: Context) -> str:
    return fingerprint({"upstream": ctx.fingerprints["manifest"], "dir": hierarchy_dir(ctx.config).name})


def _micro_run(ctx: Context) -> None:
    c = ctx.config
    out = hierarchy_dir(c)
    quote = lambda value: str(value).replace("'", "''")
    candidates = c.get("microstates.K_candidates", [2, 3, 4, 5, 6, 7])
    n_starts = c.get("microstates.n_starts", DEFAULT_N_STARTS)
    if not isinstance(n_starts, int) or isinstance(n_starts, bool) or n_starts < 1:
        raise PipelineError(f"microstates.n_starts must be a positive integer, got {n_starts!r}")
    command = (f"addpath('{quote(c.repo)}'); addpath(genpath(fullfile('{quote(c.repo)}','fsms','matlab'))); "
               f"run_study_hierarchy('{quote(c.manifest_mat)}','{quote(out)}',"
               f"'K_candidates',{json.dumps(list(candidates))},'criterion','{c.get('microstates.criterion', 'free_energy_elbow')}',"
               f"'n_starts',{n_starts});")
    matlab = shutil.which("matlab")
    if not matlab:
        raise PipelineError("matlab executable not found on PATH")
    if run_logged([matlab, "-batch", command], c.log_dir / "microstates.log", c.repo) != 0:
        raise PipelineError("Microstate stage failed; see logs/microstates.log")


def _backfit_dir(c: Config) -> Path:
    return hierarchy_dir(c) / "timestamped_backfits"


def _micro_outputs(ctx: Context) -> list[Path]:
    root = hierarchy_dir(ctx.config)
    out = _backfit_dir(ctx.config)
    glob = root / "hierarchy" / "global"
    return [out / "transition_index.csv", out / "recovery_audit.json", glob / "global_solution.mat",
            glob / "k_selection.csv", glob / "k_selection.json", glob / "free_energy_starts.csv",
            glob / "state_labels.csv",
            *existing(csv_column(out / "transition_index.csv", "cache_file"))]


def _auto_fp(ctx: Context) -> str:
    c = ctx.config
    return fingerprint({"upstream": ctx.fingerprints["manifest"], "code": code_hash(c, "fsms/autonomic.py"),
                        "heppy": ctx.fingerprints["heppy"]})


def _auto_run(ctx: Context) -> None:
    c = ctx.config
    if ctx.python_module("fsms.autonomic", "--manifest", str(c.manifest_mat), "--out-dir", str(c.autonomic_dir),
                         stage="autonomic") != 0:
        raise PipelineError("Autonomic stage failed; see logs/autonomic.log")


def _auto_outputs(ctx: Context) -> list[Path]:
    return [ctx.config.autonomic_native, ctx.config.autonomic_dir / "autonomic_audit.json"]


def _metrics_fp(ctx: Context) -> str:
    c = ctx.config
    out = _backfit_dir(c)
    return fingerprint({"upstream": [ctx.fingerprints["microstates"], ctx.fingerprints["autonomic"]],
                        "code": code_hash(c, "fsms/windows.py", "fsms/metrics.py"), "analysis": c.analysis(),
                        "index": sha256_file(out / "transition_index.csv"),
                        "autonomic": sha256_file(c.autonomic_native), "manifest": sha256_file(c.manifest_mat)})


def _metrics_run(ctx: Context) -> None:
    from . import windows
    c = ctx.config
    out = _backfit_dir(c)
    audit = read_json(out / "recovery_audit.json")
    if not (audit.get("compact_export_validated_against_full_backfit")
            and audit.get("focused_window_fit_recovery_validation", {}).get("transition_matrix_exact_match")):
        raise PipelineError("The transition export did not pass both validation gates; rerun the microstate stage")
    windows.run(out / "transition_index.csv", c.autonomic_native, c.manifest_mat, out / "recovery_audit.json",
                c.analysis_dir, windows.Settings.from_mapping(c.analysis()), log=lambda m: print("   | " + m, flush=True))


def _metrics_outputs(ctx: Context) -> list[Path]:
    a = ctx.config.analysis_dir
    return [a / "window_metrics.csv.gz", a / "window_counts.csv.gz", a / "recordings.csv",
            a / "reference_counts.csv.gz", a / "window_summary.json"]


def _labels_fp(ctx: Context) -> str:
    """Labels depend on the database, the originating workbooks and their own code; on no processing stage."""
    c = ctx.config
    missing = [f"{name}: {path}" for name, path in c.source_workbooks.items() if not path.is_file()]
    if missing:
        raise PipelineError("Originating workbook not found (sources.workbooks): " + "; ".join(missing))
    return fingerprint({"code": code_hash(c, "fsms/labels.py"), "database": sha256_file(c.database_csv),
                        "workbooks": {name: sha256_file(path) for name, path in c.source_workbooks.items()}})


def _labels_run(ctx: Context) -> None:
    from . import labels
    c = ctx.config
    try:
        audit = labels.build(c.database_csv, c.source_workbooks or None, c.labels_dir)
    except labels.LabelError as error:
        raise PipelineError(str(error)) from error
    how = ("checked against the workbooks" if audit["verified_against_workbooks"]
           else "taken from the database (no workbooks configured)")
    print(f"   | {audit['seizure_rows']} seizure rows {how}; functional seizures without hypo/hyper: "
          f"{audit['functional_seizures_without_hypo_or_hyper']}; labels {'changed' if audit['changed'] else 'unchanged'}")


def _labels_outputs(ctx: Context) -> list[Path]:
    return [ctx.config.labels_csv, ctx.config.labels_dir / "labels_audit.json"]


def _release_fp(ctx: Context) -> str:
    """The label *content* is the dependency: a workbook edit that changes no label does not rebuild the bundle."""
    c = ctx.config
    return fingerprint({"upstream": ctx.fingerprints["metrics"],
                        "code": code_hash(c, "fsms/release.py", "fsms/eligibility.py", "fsms/labels.py"),
                        "labels": sha256_file(c.labels_csv) if c.labels_csv.is_file() else None,
                        "analysis": c.analysis()})


def _release_run(ctx: Context) -> None:
    from . import release
    c = ctx.config
    index = pd.read_csv(c.manifest_mat)
    labels = pd.read_csv(c.labels_csv, dtype=str, keep_default_na=False)
    release.build_release(c.analysis_dir, c.release_dir, c.private_dir, hierarchy_dir(c) / "hierarchy" / "global", index,
                          c.analysis(), labels=labels)


def _release_outputs(ctx: Context) -> list[Path]:
    r = ctx.config.release_dir
    return [r / "windows.csv.gz", r / "events.csv", r / "participants.csv", r / "reference_grammar.csv",
            r / "window_transition_counts.csv.gz", r / "leakage_audit.json", r / "MANIFEST.json"]


def _require_id(ctx: Context) -> str:
    if not ctx.prereg_id:
        raise PipelineError("The checkpoint and statistics stages need --prereg-id (the preregistration serial)")
    return cp.parse_prereg_id(ctx.prereg_id)


def _checkpoint_fp(ctx: Context) -> str:
    return fingerprint({"upstream": ctx.fingerprints["release"], "prereg_id": _require_id(ctx),
                        "code": code_hash(ctx.config, "fsms/checkpoint.py"), "rules": ctx.config.analysis()})


def _checkpoint_run(ctx: Context) -> None:
    checkpoint = cp.write(ctx.config.release_dir, ctx.config.analysis(), _require_id(ctx))
    print(cp.render(checkpoint))


def _checkpoint_outputs(ctx: Context) -> list[Path]:
    return [ctx.config.release_dir / cp.CHECKPOINT_NAME]


def audit_tokens(config: Config, index: pd.DataFrame) -> set[str]:
    """Identifiers the statistics outputs must not contain.

    The manifest's file paths contribute every long word in them, which includes the names of the pipeline's own
    folders (``run_root``, ``data_root``).  Those words are not identifiers, and a folder called ``..._analysis``
    would otherwise flag every table that has an ``analysis`` column.  They are dropped unless they are also a
    participant or patient identifier.
    """
    from . import release
    tokens = release.forbidden_tokens(index)
    folders = {t.lower() for path in (config.run_root, config.data_root) for t in re.findall(r"[A-Za-z0-9]+", str(path))}
    identifiers = {str(v).strip().lower() for col in ("participant", "patient_id") if col in index
                   for v in index[col].dropna().unique()}
    return tokens - (folders - identifiers)


def _stats_fp(ctx: Context) -> str:
    c = ctx.config
    return fingerprint({"checkpoint": ctx.fingerprints["checkpoint"], "prereg_id": _require_id(ctx),
                        "code": code_hash(c, "fsms/stats/*.py")})


def _stats_run(ctx: Context) -> None:
    from . import release
    c = ctx.config
    prereg_id = _require_id(ctx)
    cp.require(c.release_dir, prereg_id)
    from .stats import run_statistics
    run_statistics(c.release_dir, c.results_dir, prereg_id)
    release.audit_directory(c.results_dir, audit_tokens(c, pd.read_csv(c.manifest_mat)),
                            set(pd.read_csv(c.release_dir / "participants.csv").participant_id))


def _stats_outputs(ctx: Context) -> list[Path]:
    return [ctx.config.results_dir / "results.json"]


STAGES: dict[str, Stage] = {
    "clips_check": Stage("clips_check", _clips_fp, _clips_run, lambda ctx: [ctx.config.state_dir / "clips_check.json"]),
    "heppy": Stage("heppy", _heppy_fp, _heppy_run, _heppy_outputs),
    "manifest": Stage("manifest", _manifest_fp, _manifest_run, _manifest_outputs),
    "microstates": Stage("microstates", _micro_fp, _micro_run, _micro_outputs),
    "autonomic": Stage("autonomic", _auto_fp, _auto_run, _auto_outputs),
    "metrics": Stage("metrics", _metrics_fp, _metrics_run, _metrics_outputs),
    "labels": Stage("labels", _labels_fp, _labels_run, _labels_outputs),
    "release": Stage("release", _release_fp, _release_run, _release_outputs),
    "checkpoint": Stage("checkpoint", _checkpoint_fp, _checkpoint_run, _checkpoint_outputs),
    "stats": Stage("stats", _stats_fp, _stats_run, _stats_outputs),
}


# ------------------------------------------------------------------ driver
def run(config: Config, through: str = "checkpoint", prereg_id: str | None = None, force: str | None = None) -> dict:
    if through not in STAGE_ORDER:
        raise PipelineError(f"Unknown stage {through!r}; choose from {STAGE_ORDER}")
    if STAGE_ORDER.index(through) >= STAGE_ORDER.index("checkpoint") and not prereg_id:
        raise PipelineError("Reaching the checkpoint needs --prereg-id (the preregistration serial)")
    ledger = Ledger(config.state_dir)
    ctx = Context(config, ledger, prereg_id)
    report: dict[str, str] = {}
    with RunLock(config.state_dir):
        for name in STAGE_ORDER[: STAGE_ORDER.index(through) + 1]:
            stage = STAGES[name]
            ctx.fingerprints[name] = stage.fingerprint(ctx)
            current, reason = ledger.verify(name, ctx.fingerprints[name])
            if current and force != name:
                print(f"[SKIP] {name} (current)")
                report[name] = "skipped"
                continue
            print(f"[RUN ] {name}  ({reason})")
            with Timer() as timer:
                stage.run(ctx)
            ctx.fingerprints[name] = stage.fingerprint(ctx)    # outputs now exist; the fingerprint may include them
            ledger.mark_done(name, ctx.fingerprints[name], stage.outputs(ctx), seconds=timer.seconds)
            print(f"[DONE] {name}  ({timer.seconds:.0f}s)")
            report[name] = "ran"
    return report


def status(config: Config) -> dict[str, str]:
    ledger = Ledger(config.state_dir)
    result = {}
    for name in STAGE_ORDER:
        record = ledger.read(name)
        result[name] = "never run" if record is None else f"done {record['completed_utc']}"
    return result


def runtime_mismatches() -> list[str]:
    """Installed package versions that differ from ``requirements-lock.txt``."""
    lock = Path(__file__).resolve().parents[1] / "requirements-lock.txt"
    problems = []
    for line in lock.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"\s*([A-Za-z0-9_.\-]+)==(\S+)\s*", line)
        if not match:
            continue
        name, wanted = match.groups()
        try:
            have = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            problems.append(f"{name} not installed")
            continue
        if have != wanted:
            problems.append(f"{name} {have} != {wanted}")
    return problems


def preflight(config: Config | None, deep: bool = False) -> list[tuple[str, bool, str]]:
    """Static environment checks; never reads study data."""
    checks: list[tuple[str, bool, str]] = []

    def add(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, bool(ok), detail))

    add("python", platform.python_version().startswith("3.12"), platform.python_version())
    mismatches = runtime_mismatches()
    add("pinned packages", not mismatches, "; ".join(mismatches[:5]) or "match requirements-lock.txt")
    problems = provenance.verify()
    add("vendored externals", not problems, "; ".join(problems) or "match external/PROVENANCE.json")
    matlab = shutil.which("matlab")
    add("matlab on PATH", matlab, matlab or "not found")
    if config is not None:
        add("config", True, str(config.path))
        anchor = config.run_root
        while not anchor.exists() and anchor.parent != anchor:
            anchor = anchor.parent
        add("free disk (GB)", shutil.disk_usage(anchor).free / 1e9 > 100, f"{shutil.disk_usage(anchor).free / 1e9:.0f}")
    if deep and matlab:
        code = subprocess.run([matlab, "-batch", "assert(~isempty(which('spm_mix')) && ~isempty(which('pop_loadset')));"],
                              capture_output=True, text=True, timeout=600).returncode
        add("MATLAB SPM + EEGLAB on path", code == 0)
    return checks
