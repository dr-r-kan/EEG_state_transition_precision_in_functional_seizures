"""Synthetic de-identified release bundles for testing the statistics package.

The generator follows ``docs/DATA_DICTIONARY.md``: 60-s windows aligned to seizure onset (baseline block
``window_end_min`` 19..10, minutes 9..0, interictal windows), ``events.csv`` and ``participants.csv`` with
per-representation eligibility flags (``eligible_matrix``, ``eligible_tensor``), and
``inclusion_checkpoint.json``.  Effects of FS seizures are planted explicitly:

* ``js_shift`` / ``kappa_shift``: FS-seizure change at the final minute (ES seizures: 0);
* ``js_slope`` / ``kappa_slope``: FS-seizure gradient toward onset, per minute;
* ``a_shift``: FS-seizure CSI increase and CPI decrease at the final minute, so that ``a = (CSI - CPI) / 2``
  changes by ``a_shift``.

Participants have random offsets, random final-minute changes and random gradients; seizures have random
offsets; windows are noisy and a fraction is missing (a further fraction lacks the tensor values, so the two
representations have different eligible samples).  Groups are unbalanced by default, some participants are DD
(their seizures alternate FS, ES), some seizures are ineligible and some participants have no eligible seizure;
ineligible data carry extreme values so that any leakage into the analyses is visible.  The tensor columns share
the planted effects (scaled by ``tensor_scale``) but have independent noise.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ("js_excess_ordinary", "log_kappa_ordinary", "js_excess_tensor", "log_kappa_tensor", "csi_z", "cpi_z")
REPRESENTATION_METRICS = {
    "matrix": ["js_excess_ordinary", "log_kappa_ordinary", "csi_z", "cpi_z"],
    "tensor": ["js_excess_tensor", "log_kappa_tensor", "csi_z", "cpi_z"],
}
OUTLIER = 50.0


@dataclass(frozen=True)
class SyntheticSpec:
    n_fs: int = 14
    n_es: int = 11
    n_dd: int = 4
    n_ineligible_participants: int = 1
    events_per_participant: tuple[int, int] = (1, 3)
    n_interictal_windows: int = 12
    js_shift: float = 0.0
    kappa_shift: float = 0.0
    a_shift: float = 0.0
    js_slope: float = 0.0
    kappa_slope: float = 0.0
    tensor_scale: float = 0.8
    sd_participant: float = 0.5
    sd_change: float = 0.15
    sd_slope: float = 0.02
    sd_event: float = 0.2
    sd_noise: float = 0.3
    missing_window_probability: float = 0.05
    tensor_missing_probability: float = 0.08
    invalid_event_probability: float = 0.1
    seed: int = 0

    def null(self, seed: int) -> "SyntheticSpec":
        """The same design with every group effect set to zero."""
        return replace(self, js_shift=0.0, kappa_shift=0.0, a_shift=0.0, js_slope=0.0, kappa_slope=0.0, seed=seed)


def _fs_effects(spec: SyntheticSpec) -> dict[str, tuple[float, float]]:
    s = spec.tensor_scale
    return {
        "js_excess_ordinary": (spec.js_shift, spec.js_slope),
        "log_kappa_ordinary": (spec.kappa_shift, spec.kappa_slope),
        "js_excess_tensor": (s * spec.js_shift, s * spec.js_slope),
        "log_kappa_tensor": (s * spec.kappa_shift, s * spec.kappa_slope),
        "csi_z": (spec.a_shift, 0.0),
        "cpi_z": (-spec.a_shift, 0.0),
    }


def build_tables(spec: SyntheticSpec) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return ``(windows, events, participants)`` for ``spec``."""
    rng = np.random.default_rng(spec.seed)
    effects = _fs_effects(spec)
    plan = ([("FS", False)] * spec.n_fs + [("ES", False)] * spec.n_es + [("DD", False)] * spec.n_dd
            + [("FS", True)] * spec.n_ineligible_participants)
    window_parts, event_rows, participant_rows = [], [], []
    for index, (group, force_invalid) in enumerate(plan, start=1):
        participant = f"P{index:03d}"
        mu = rng.normal(0, spec.sd_participant, len(METRICS))
        u0 = rng.normal(0, spec.sd_change, len(METRICS))
        u1 = rng.normal(0, spec.sd_slope, len(METRICS))

        interictal = _window_frame(participant, f"{participant}_I01", group, np.nan, "interictal",
                                   np.arange(spec.n_interictal_windows), spec, rng, mu, None)
        window_parts.append(interictal)

        n_events = int(rng.integers(spec.events_per_participant[0], spec.events_per_participant[1] + 1))
        eligible_events = {name: 0 for name in REPRESENTATION_METRICS}
        for number in range(1, n_events + 1):
            event_type = group if group != "DD" else ("FS" if number % 2 else "ES")
            recording = f"{participant}_R{number:02d}"
            invalid = force_invalid or rng.random() < spec.invalid_event_probability
            shift_slope = {m: (effects[m] if event_type == "FS" else (0.0, 0.0)) for m in METRICS}
            offset = {m: (mu[i], rng.normal(0, spec.sd_event), u0[i], u1[i]) for i, m in enumerate(METRICS)}
            frame = _window_frame(participant, recording, group, event_type, "preictal",
                                  np.arange(19, -1, -1), spec, rng, mu, (shift_slope, offset))
            if invalid:  # no usable cardiac data: ineligible for both representations, extreme transition values
                frame.loc[:, ["csi_z", "cpi_z"]] = np.nan
                frame["js_excess_ordinary"] += OUTLIER
                frame["js_excess_tensor"] += OUTLIER
            window_parts.append(frame)
            # semiology of functional seizures: by participant (every fifth participant alternates = "mixed")
            semiology = "" if event_type != "FS" else (
                ("hypo", "hyper")[number % 2] if index % 5 == 0 else ("hypo", "hyper")[index % 2])
            row = {"recording_id": recording, "participant_id": participant, "group": group, "event_type": event_type,
                   "semiology": semiology}
            for name, metrics in REPRESENTATION_METRICS.items():
                complete = frame[metrics].notna().all(axis=1)
                n_baseline = int((complete & (frame["window_end_min"] >= 10)).sum())
                has_final = bool((complete & (frame["window_end_min"] == 0)).any())
                reasons = (["no_complete_final_minute"] if not has_final else []) + (
                    ["insufficient_baseline"] if n_baseline < 5 else [])
                eligible_events[name] += int(not reasons)
                row.update({f"n_baseline_windows_{name}": n_baseline, f"has_final_minute_{name}": has_final,
                            f"eligible_{name}": not reasons, f"exclusion_reason_{name}": ";".join(reasons)})
            event_rows.append(row)
        row = {"participant_id": participant, "group": group, "n_interictal_recordings": 1,
               "n_interictal_windows": len(interictal), "n_interictal_transitions": 40 * len(interictal),
               "n_preictal_recordings": n_events}
        for name, metrics in REPRESENTATION_METRICS.items():
            n_valid = int(interictal[metrics].notna().all(axis=1).sum())
            reasons = (["insufficient_interictal_windows"] if n_valid < 5 else []) + (
                ["no_eligible_event"] if eligible_events[name] == 0 else [])
            row.update({f"n_eligible_events_{name}": eligible_events[name], f"eligible_{name}": not reasons,
                        f"exclusion_reason_{name}": ";".join(reasons)})
        participant_rows.append(row)
    windows = pd.concat(window_parts, ignore_index=True)
    return windows, pd.DataFrame(event_rows), pd.DataFrame(participant_rows)


def _window_frame(participant, recording, group, event_type, period, minutes, spec, rng, mu, preictal):
    n = len(minutes)
    values = {}
    for i, metric in enumerate(METRICS):
        value = mu[i] + rng.normal(0, spec.sd_noise, n)
        if preictal is not None:
            shift_slope, offset = preictal
            _, event_offset, u0, u1 = offset[metric]
            shift, slope = shift_slope[metric]
            t = -minutes.astype(float)
            value = value + event_offset
            in_minute = minutes <= 9
            value = value + np.where(in_minute, shift + u0 + (slope + u1) * t, 0.0)
        values[metric] = value
    role = np.where(minutes >= 10, "baseline", "minute") if preictal is not None else np.full(n, None, dtype=object)
    frame = pd.DataFrame({
        "participant_id": participant, "recording_id": recording, "group": group, "event_type": event_type,
        "period": period, "window_role": role, "window_end_min": minutes.astype(float) if preictal is not None else np.nan,
        "n_transitions": rng.integers(30, 80, n), "n_action_transitions": rng.integers(10, 30, n), **values})
    frame["js_null_ordinary"] = 0.05
    frame["js_ordinary"] = frame["js_excess_ordinary"] + 0.05
    frame["kappa_ordinary"] = np.exp(frame["log_kappa_ordinary"])
    frame["csi_gradient"] = rng.normal(0, 0.01, n)
    frame["cpi_gradient"] = rng.normal(0, 0.01, n)
    tensor_missing = rng.random(n) < spec.tensor_missing_probability
    frame.loc[tensor_missing, ["js_excess_tensor", "log_kappa_tensor"]] = np.nan
    keep = rng.random(n) >= spec.missing_window_probability
    return frame[keep].reset_index(drop=True)


def write_release(directory: str | Path, spec: SyntheticSpec, prereg_id: str = "TEST-PREREG") -> Path:
    """Write a release directory (``windows.csv.gz``, ``events.csv``, ``participants.csv``, checkpoint)."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    windows, events, participants = build_tables(spec)
    windows.to_csv(root / "windows.csv.gz", index=False)
    events.to_csv(root / "events.csv", index=False)
    participants.to_csv(root / "participants.csv", index=False)
    write_checkpoint(root, prereg_id)
    return root


def write_checkpoint(directory: str | Path, prereg_id: str) -> None:
    (Path(directory) / "inclusion_checkpoint.json").write_text(
        json.dumps({"prereg_id": prereg_id, "note": "synthetic test checkpoint"}), encoding="utf-8")
