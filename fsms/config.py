"""Study configuration: one JSON file defines a run.

Everything the pipeline *writes* lives under ``run_root``; ``data_root`` is only read::

    <data_root>/clips/*.edf, clips/snip_keyfile.csv, analysis_database.csv     inputs
    <run_root>/state/<stage>.done.json                                         idempotency ledger
    <run_root>/heppy_output/                                                   EEG/ECG preprocessing
    <run_root>/vg_microstates/                                                 manifests, microstate hierarchy
    <run_root>/autonomic/                                                      CSI/CPI native table
    <run_root>/analysis/                                                       window metrics (contains source identifiers)
    <run_root>/labels/                                                         event labels checked against the workbooks
    <run_root>/private/                                                        pseudonym key (never published)
    <run_root>/release/                                                        de-identified bundle read by the statistics
    <run_root>/results/                                                        statistical results and figures
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .atomic import fingerprint

REPO = Path(__file__).resolve().parents[1]
SCHEMA = "fsms-config/1"
_MISSING = object()

#: random initialisations of the global microstate fit per K (the start with the highest free energy is kept)
DEFAULT_N_STARTS = 20

#: analysis constants that define eligibility and the window metrics (recorded in every checkpoint)
ANALYSIS_DEFAULTS = {
    "prior_strength": 10.0,            # pseudo-counts shrinking the tensor reference towards the ordinary matrix
    "max_autonomic_gap_s": 5.0,        # longest gap bridged when interpolating CSI/CPI between heartbeats
    "min_window_transitions": 30,      # transitions required in a 60-s window
    "min_reference_transitions": 300,  # interictal transitions required for a participant's reference grammar
    "min_baseline_windows": 5,         # valid 20-10 min windows required per seizure
    "min_interictal_windows": 5,       # valid interictal windows required per participant
    "null_draws": 400,                 # parametric-bootstrap draws per window for the expected JS under the reference
}


class ConfigError(ValueError):
    pass


class Config:
    def __init__(self, raw: dict[str, Any], path: Path):
        self.raw = raw
        self.path = path
        self.repo = REPO
        self.name: str = raw.get("name", "")
        if raw.get("schema") != SCHEMA:
            raise ConfigError(f"Unsupported config schema {raw.get('schema')!r}; expected {SCHEMA!r}")
        for key in ("paths.data_root", "paths.run_root"):
            self.get(key)

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "Config":
        target = Path(path)
        if not target.is_absolute():
            target = (REPO / target) if (REPO / target).exists() else target.resolve()
        try:
            raw = json.loads(target.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError) as error:
            raise ConfigError(f"Cannot read config {target}: {error}") from error
        return cls(raw, target)

    def get(self, dotted: str, default: Any = _MISSING) -> Any:
        node: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                if default is _MISSING:
                    raise ConfigError(f"Missing required config key {dotted!r} in {self.path}")
                return default
            node = node[part]
        return node

    def analysis(self) -> dict[str, Any]:
        """Analysis constants: defaults overridden by the config's ``analysis`` section."""
        merged = dict(ANALYSIS_DEFAULTS)
        merged.update(self.raw.get("analysis", {}))
        return merged

    def _path(self, value: str, base: Path) -> Path:
        candidate = Path(os.path.expandvars(str(value)))
        return candidate if candidate.is_absolute() else (base / candidate)

    # ---- inputs ---------------------------------------------------------------------------
    @property
    def data_root(self) -> Path:
        return self._path(self.get("paths.data_root"), self.repo)

    @property
    def run_root(self) -> Path:
        return self._path(self.get("paths.run_root"), self.repo)

    @property
    def existing_heppy_dir(self) -> Path | None:
        value = self.get("paths.existing_heppy_dir", None)
        return self._path(value, self.repo) if value else None

    def repo_file(self, relative: str) -> Path:
        return self._path(relative, self.repo)

    @property
    def clips_dir(self) -> Path:
        return self._path(self.get("clips.dir", "clips"), self.data_root)

    @property
    def database_csv(self) -> Path:
        return self._path(self.get("clips.database", "analysis_database.csv"), self.data_root)

    @property
    def snip_keyfile(self) -> Path:
        return self._path(self.get("clips.snip_keyfile", "clips/snip_keyfile.csv"), self.data_root)

    @property
    def source_workbooks(self) -> dict[str, Path]:
        """Originating workbooks by database ``source`` name (``sources.workbooks``); empty when not configured."""
        return {name: self._path(value, self.data_root)
                for name, value in sorted(self.get("sources.workbooks", {}).items())}

    # ---- run layout -----------------------------------------------------------------------
    @property
    def state_dir(self) -> Path:
        return self.run_root / "state"

    @property
    def log_dir(self) -> Path:
        return self.run_root / "logs"

    @property
    def heppy_out(self) -> Path:
        return self.run_root / "heppy_output"

    @property
    def vg_dir(self) -> Path:
        return self.run_root / "vg_microstates"

    @property
    def manifest_mat(self) -> Path:
        return self.vg_dir / "vg_microstate_manifest_mat.csv"

    @property
    def autonomic_dir(self) -> Path:
        return self.run_root / "autonomic"

    @property
    def autonomic_native(self) -> Path:
        return self.autonomic_dir / "autonomic_native_ii_standardised.csv.gz"

    @property
    def labels_dir(self) -> Path:
        return self.run_root / "labels"

    @property
    def labels_csv(self) -> Path:
        return self.labels_dir / "event_labels.csv"

    @property
    def analysis_dir(self) -> Path:
        return self.run_root / "analysis"

    @property
    def private_dir(self) -> Path:
        return self.run_root / "private"

    @property
    def release_dir(self) -> Path:
        return self.run_root / "release"

    @property
    def results_dir(self) -> Path:
        return self.run_root / "results"

    def section_fingerprint(self, *sections: str) -> str:
        return fingerprint({name: self.raw.get(name) for name in sections})
