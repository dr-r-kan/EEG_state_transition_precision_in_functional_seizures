"""Window metrics: transition matrix, autonomic tensor and CSI/CPI in onset-aligned 60-s windows.

Inputs (all produced by earlier stages)

* the exact ordered state changes of every recording (``transition_index.csv`` plus one cache per recording),
* the CSI/CPI native table (15-s cardiac estimates standardised to each participant's interictal recordings),
* the manifest (diagnostic group and seizure type per recording).

Windows are non-overlapping 60-s blocks.  A preictal recording ends at seizure onset, so window ``k`` (1-based)
covers ``[D - 60 k, D - 60 (k - 1)]`` seconds for a recording of length ``D``; its ``window_end_min`` is
``k - 1`` (0 = the final minute).  Windows 1-10 are the ``minute`` windows, windows 11-20 the ``baseline`` block
(20 to 10 minutes before onset).  Interictal recordings are cut into consecutive 60-s windows from the start.

For each window the matrix and tensor are compared with the participant's interictal reference grammar
(see :mod:`fsms.metrics`); an interictal window is evaluated against the reference with that window left out.
CSI and CPI are the means of the heartbeat-interpolated, interictal-standardised indices at the window's
transitions.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.io import loadmat

from . import metrics as M

WINDOW_S = 60.0
N_BASELINE, N_MINUTE = 10, 10
ROLE_BASELINE, ROLE_MINUTE, ROLE_INTERICTAL = "baseline", "minute", "interictal"


def normalise_path(value):
    """Case-insensitive, slash-insensitive key for matching file paths across tables."""
    if isinstance(value, pd.Series):
        return value.astype(str).str.replace("\\", "/", regex=False).str.lower()
    return str(value).replace("\\", "/").lower()


@dataclass(frozen=True)
class Settings:
    prior_strength: float
    max_autonomic_gap_s: float
    min_window_transitions: int
    min_reference_transitions: int
    null_draws: int

    @classmethod
    def from_mapping(cls, mapping: dict) -> "Settings":
        return cls(**{name: mapping[name] for name in cls.__dataclass_fields__})


# ------------------------------------------------------------------------------ inputs
def _hdf_array(group: h5py.Group, name: str) -> np.ndarray:
    return np.asarray(group[name]).ravel()


def load_transition_cache(path: Path) -> dict:
    """Ordered genuine state changes of one recording (state indices are 1-based fitted-state indices)."""
    try:
        value = loadmat(path, squeeze_me=True, struct_as_record=False)["transition"]
        return {"sfreq": float(value.sfreq), "n_samples": int(value.n_samples),
                "sample": np.atleast_1d(value.transition_sample).astype(np.int64),
                "from": np.atleast_1d(value.from_state).astype(np.int16),
                "to": np.atleast_1d(value.to_state).astype(np.int16)}
    except NotImplementedError:                                    # MATLAB v7.3 (HDF5) file
        with h5py.File(path, "r") as handle:
            value = handle["transition"]
            return {"sfreq": float(_hdf_array(value, "sfreq")[0]), "n_samples": int(_hdf_array(value, "n_samples")[0]),
                    "sample": _hdf_array(value, "transition_sample").astype(np.int64),
                    "from": _hdf_array(value, "from_state").astype(np.int16),
                    "to": _hdf_array(value, "to_state").astype(np.int16)}


def validate_cache(cache: dict, n_states: int) -> None:
    samples = cache["sample"]
    if (not np.isfinite(cache["sfreq"]) or cache["sfreq"] <= 0 or cache["n_samples"] < 1
            or len(samples) != len(cache["from"]) or len(samples) != len(cache["to"])
            or np.any(np.diff(samples) <= 0) or np.any(samples < 1) or np.any(samples > cache["n_samples"])
            or not np.isin(cache["from"], np.arange(1, n_states + 1)).all()
            or not np.isin(cache["to"], np.arange(1, n_states + 1)).all()
            or np.any(cache["from"] == cache["to"])):
        raise ValueError("Invalid ordered genuine-transition cache")


def interpolate_autonomic(native: pd.DataFrame, times: np.ndarray, max_gap_s: float):
    """CSI/CPI levels and gradients at ``times``: linear interpolation inside contiguous runs, never across gaps."""
    levels, gradients = [], []
    for column in ("csi_z", "cpi_z"):
        level = np.full(len(times), np.nan)
        gradient = level.copy()
        if not native.empty:
            subset = native[["event_time_s", column]].apply(pd.to_numeric, errors="coerce")
            subset = subset.dropna(subset=["event_time_s"]).groupby("event_time_s", as_index=False)[column].mean()
            x, y = subset.event_time_s.to_numpy(), subset[column].to_numpy()
            good = np.isfinite(x) & np.isfinite(y)
            starts = np.flatnonzero(good & np.r_[True, ~good[:-1] | (np.diff(x) > max_gap_s)])
            for start in starts:
                stop = start + 1
                while stop < len(x) and good[stop] and x[stop] - x[stop - 1] <= max_gap_s:
                    stop += 1
                if stop - start < 2:
                    continue
                xx, yy = x[start:stop], y[start:stop]
                use = (times >= xx[0]) & (times <= xx[-1])
                level[use] = np.interp(times[use], xx, yy)
                gradient[use] = np.interp(times[use], xx, np.gradient(yy, xx))
        levels.append(level)
        gradients.append(gradient)
    return levels[0], levels[1], gradients[0], gradients[1]


def canonical_states(fit_labels) -> tuple[list[str], np.ndarray]:
    """Alphabetical state names and the map from 0-based fitted-state index to canonical index."""
    labels = [str(label).strip().upper() for label in fit_labels]
    if len(labels) < 2 or len(set(labels)) != len(labels):
        raise ValueError(f"fit_labels must be unique state labels, got {labels}")
    names = sorted(labels)
    return names, np.array([names.index(label) for label in labels], dtype=np.int64)


def window_specs(condition: str, duration_s: float) -> list[tuple[str, int | None, float]]:
    """(role, window_end_min, start_s) for every complete 60-s window of a recording."""
    if duration_s < WINDOW_S:
        return []
    if condition == "preict":
        specs = []
        for k in range(1, N_BASELINE + N_MINUTE + 1):
            start = duration_s - WINDOW_S * k
            if start < -1e-9:
                break
            specs.append((ROLE_MINUTE if k <= N_MINUTE else ROLE_BASELINE, k - 1, max(start, 0.0)))
        return specs
    return [(ROLE_INTERICTAL, None, start) for start in np.arange(0, duration_s - WINDOW_S + 1e-9, WINDOW_S)]


# ------------------------------------------------------------------------------ per participant
@dataclass
class _Window:
    record: int
    role: str
    end_min: int | None
    start_s: float
    source: np.ndarray
    weights: np.ndarray
    counts: np.ndarray
    tensor: np.ndarray
    n_action: int
    csi: float
    cpi: float
    csi_gradient: float
    cpi_gradient: float


def _mean(values: np.ndarray) -> float:
    return float(np.nanmean(values)) if np.isfinite(values).any() else np.nan


def _participant_windows(records: pd.DataFrame, native: dict[str, pd.DataFrame], to_canonical: np.ndarray,
                         settings: Settings) -> tuple[list[_Window], list[dict]]:
    windows: list[_Window] = []
    recording_rows: list[dict] = []
    n_states = len(to_canonical)
    for number, row in enumerate(records.itertuples(index=False)):
        cache = load_transition_cache(Path(row.cache_file))
        validate_cache(cache, n_states)
        times = (cache["sample"] - 1) / cache["sfreq"]
        source = to_canonical[cache["from"] - 1]
        destination = to_canonical[cache["to"] - 1]
        csi, cpi, csi_gradient, cpi_gradient = interpolate_autonomic(native.get(row.file_key, pd.DataFrame()), times,
                                                                     settings.max_autonomic_gap_s)
        weights = M.gradient_weights(cpi_gradient, csi_gradient)
        duration = cache["n_samples"] / cache["sfreq"]
        kept = 0
        specs = window_specs(row.condition, duration)
        for role, end_min, start in specs:
            left, right = np.searchsorted(times, (start, start + WINDOW_S), side="left")
            if right - left < settings.min_window_transitions:
                continue
            window = slice(left, right)
            windows.append(_Window(
                record=number, role=role, end_min=end_min, start_s=float(start), source=source[window],
                weights=weights[window], counts=M.transition_counts(source[window], destination[window], n_states),
                tensor=M.tensor_counts(source[window], destination[window], weights[window], n_states),
                n_action=int(np.isfinite(weights[window]).all(axis=1).sum()),
                csi=_mean(csi[window]), cpi=_mean(cpi[window]),
                csi_gradient=_mean(csi_gradient[window]), cpi_gradient=_mean(cpi_gradient[window])))
            kept += 1
        recording_rows.append({
            "participant": row.participant, "file_path": row.file_path, "file_key": row.file_key,
            "condition": row.condition, "group": row.group, "event_type": row.event_type,
            "duration_s": duration, "n_transitions": int(len(times)),
            "autonomic_fraction": float(np.isfinite(weights).all(axis=1).mean()) if len(times) else np.nan,
            "n_windows_possible": len(specs), "n_windows_enough_transitions": kept})
    return windows, recording_rows


def _evaluate(window: _Window, record_key: str, reference: np.ndarray, tensor_reference: np.ndarray,
              settings: Settings) -> dict | None:
    """Metrics of one window against a reference grammar (already without the window itself if interictal)."""
    if reference.sum() < settings.min_reference_transitions:
        return None
    ordinary = M.row_probabilities(reference)
    tensor = M.row_probabilities(tensor_reference, np.broadcast_to(ordinary, tensor_reference.shape),
                                 settings.prior_strength)
    key = f"{record_key}|{window.start_s:.6f}"
    tensor_ok = window.n_action >= settings.min_window_transitions
    valid = np.isfinite(window.weights).all(axis=1)
    js = M.weighted_js(window.counts, ordinary)
    js_null = M.expected_js_under_reference(window.source, np.ones((len(window.source), 1)), ordinary[None],
                                            key + "|ordinary", settings.null_draws)
    kappa = M.fit_kappa(window.counts, ordinary)
    result = {"js_ordinary": js, "js_null_ordinary": js_null, "js_excess_ordinary": js - js_null,
              "kappa_ordinary": kappa, "log_kappa_ordinary": math.log(kappa) if kappa > 0 else np.nan,
              "js_tensor": np.nan, "js_null_tensor": np.nan, "js_excess_tensor": np.nan,
              "kappa_tensor": np.nan, "log_kappa_tensor": np.nan}
    if tensor_ok:
        js_t = M.weighted_js(window.tensor, tensor)
        null_t = M.expected_js_under_reference(window.source[valid], window.weights[valid], tensor, key + "|tensor",
                                               settings.null_draws)
        kappa_t = M.fit_kappa(window.tensor, tensor)
        result.update(js_tensor=js_t, js_null_tensor=null_t, js_excess_tensor=js_t - null_t, kappa_tensor=kappa_t,
                      log_kappa_tensor=math.log(kappa_t) if kappa_t > 0 else np.nan)
    return result


def compute_participant(records: pd.DataFrame, native: dict[str, pd.DataFrame], to_canonical: np.ndarray,
                        settings: Settings) -> dict:
    """All windows, metrics and the interictal reference of one participant."""
    windows, recording_rows = _participant_windows(records, native, to_canonical, settings)
    n_states = len(to_canonical)
    interictal = [w for w in windows if w.role == ROLE_INTERICTAL]
    reference = sum((w.counts for w in interictal), np.zeros((n_states, n_states)))
    tensor_reference = sum((w.tensor for w in interictal), np.zeros((4, n_states, n_states)))
    rows, count_rows = [], []
    meta = records.reset_index(drop=True)
    for window in windows:
        record = meta.iloc[window.record]
        own = window.role == ROLE_INTERICTAL
        metrics = _evaluate(window, record.file_key, reference - window.counts if own else reference,
                            tensor_reference - window.tensor if own else tensor_reference, settings)
        if metrics is None:
            continue
        rows.append({
            "participant": record.participant, "file_path": record.file_path, "file_key": record.file_key,
            "group": record.group, "event_type": record.event_type if record.condition == "preict" else "",
            "period": "preictal" if record.condition == "preict" else "interictal",
            "window_role": window.role, "window_end_min": window.end_min, "window_start_s": window.start_s,
            "n_transitions": int(window.counts.sum()), "n_action_transitions": window.n_action,
            **metrics, "csi_z": window.csi, "cpi_z": window.cpi,
            "csi_gradient": window.csi_gradient, "cpi_gradient": window.cpi_gradient})
        for source, destination in zip(*np.nonzero(window.counts)):
            count_rows.append((record.participant, record.file_path, window.role, window.end_min, window.start_s,
                               int(source), int(destination), int(window.counts[source, destination]),
                               *[float(window.tensor[a, source, destination]) for a in range(4)]))
    return {"windows": rows, "counts": count_rows, "recordings": recording_rows, "reference": reference,
            "n_reference_transitions": int(reference.sum())}


# ------------------------------------------------------------------------------ driver
def load_native_by_recording(path: Path, file_keys: set[str]) -> dict[str, pd.DataFrame]:
    parts = []
    for chunk in pd.read_csv(path, usecols=["file_path", "event_time_s", "csi_z", "cpi_z"], chunksize=200_000):
        chunk["file_key"] = normalise_path(chunk.file_path)
        parts.append(chunk[chunk.file_key.isin(file_keys)])
    frame = pd.concat(parts, ignore_index=True)
    return {key: value for key, value in frame.groupby("file_key", sort=False)}


def load_index(transition_index: Path, manifest_csv: Path) -> pd.DataFrame:
    index = pd.read_csv(transition_index)
    if index.file_path.duplicated().any():
        raise ValueError("Duplicate recording in the transition index")
    if index[["participant", "group"]].drop_duplicates().participant.duplicated().any():
        raise ValueError("A participant has conflicting diagnostic groups")
    index["file_key"] = normalise_path(index.file_path)
    index["condition"] = index.condition.astype(str).str.lower()
    manifest = pd.read_csv(manifest_csv)
    manifest["file_key"] = normalise_path(manifest.file_path)
    columns = [c for c in ("file_key", "seizure_type", "patient_id") if c in manifest]
    event = manifest[columns].drop_duplicates()
    if event.file_key.duplicated().any():
        raise ValueError("Conflicting event labels in the manifest")
    index = index.merge(event, on="file_key", how="left", validate="one_to_one")
    seizure = index["seizure_type"] if "seizure_type" in index else pd.Series("", index=index.index)
    index["event_type"] = seizure.astype(str).str.upper().where(lambda s: s.isin(["FS", "ES"]), "")
    return index


def run(transition_index: Path, autonomic_native: Path, manifest_csv: Path, recovery_audit: Path, out_dir: Path,
        settings: Settings, log=print) -> dict:
    """Compute and write ``window_metrics.csv.gz``, ``window_counts.csv.gz``, ``recordings.csv`` and
    ``reference_counts.csv.gz`` (all keyed by source identifiers; the release stage pseudonymises them)."""
    audit = json.loads(recovery_audit.read_text(encoding="utf-8"))
    names, to_canonical = canonical_states(audit["fit_labels"])
    index = load_index(transition_index, manifest_csv)
    native = load_native_by_recording(autonomic_native, set(index.file_key))
    all_rows, all_counts, all_records, reference_rows = [], [], [], []
    participants = sorted(index.participant.unique())
    for number, participant in enumerate(participants, 1):
        result = compute_participant(index[index.participant == participant], native, to_canonical, settings)
        all_rows += result["windows"]
        all_counts += result["counts"]
        all_records += result["recordings"]
        for source in range(len(names)):
            for destination in range(len(names)):
                if source != destination:
                    reference_rows.append((participant, names[source], names[destination],
                                           int(result["reference"][source, destination])))
        if number % 10 == 0 or number == len(participants):
            log(f"[{number}/{len(participants)}] participants evaluated")
    out_dir.mkdir(parents=True, exist_ok=True)
    windows = pd.DataFrame(all_rows)
    if windows.empty:
        raise ValueError("No window met the transition and reference requirements; check the inputs")
    counts = pd.DataFrame(all_counts, columns=["participant", "file_path", "window_role", "window_end_min",
                                               "window_start_s", "from_index", "to_index", "count",
                                               "weight_a1", "weight_a2", "weight_a3", "weight_a4"])
    counts["from_state"] = [names[i] for i in counts.from_index]
    counts["to_state"] = [names[i] for i in counts.to_index]
    recordings = pd.DataFrame(all_records)
    reference = pd.DataFrame(reference_rows, columns=["participant", "from_state", "to_state", "count"])
    windows.to_csv(out_dir / "window_metrics.csv.gz", index=False, compression="gzip")
    counts.drop(columns=["from_index", "to_index"]).to_csv(out_dir / "window_counts.csv.gz", index=False, compression="gzip")
    recordings.to_csv(out_dir / "recordings.csv", index=False)
    reference.to_csv(out_dir / "reference_counts.csv.gz", index=False, compression="gzip")
    summary = {"states": names, "n_participants": len(participants), "n_recordings": int(len(recordings)),
               "n_windows": int(len(windows)), "settings": settings.__dict__}
    (out_dir / "window_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
