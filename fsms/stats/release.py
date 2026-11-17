"""Reading the release bundle and deriving the participant- and event-level quantities every analysis uses.

Only the three bundle tables ``windows.csv.gz``, ``events.csv`` and ``participants.csv`` are read (see
``docs/DATA_DICTIONARY.md``).  The module

* verifies the inclusion checkpoint (the statistics gate);
* keeps only the participants and seizures eligible for the representation (``eligible_matrix`` or
  ``eligible_tensor``);
* computes, for each eligible seizure and each outcome,
  the baseline-block mean (``window_end_min`` 10..19, at least ``MIN_BASELINE_WINDOWS`` valid windows), the
  final-minute value (``window_end_min`` 0), the per-minute values (``window_end_min`` 9..0) and the
  change of each from the baseline-block mean;
* computes, for each eligible participant and outcome, the mean over their interictal windows.

An *outcome* is one of ``JS`` (excess Jensen-Shannon divergence), ``kappa`` (log precision), ``CSI`` and ``CPI``
(cardiac sympathetic / parasympathetic indices, z-scored to the participant's interictal recordings).  A
*representation* selects which transition-organisation columns supply JS and kappa.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .errors import ReleaseSchemaError, StatisticsGateError

OUTCOMES: tuple[str, ...] = ("JS", "kappa", "CSI", "CPI")
GROUPS: tuple[str, ...] = ("FS", "ES")
REPRESENTATIONS: Mapping[str, Mapping[str, str]] = {
    "matrix": {"JS": "js_excess_ordinary", "kappa": "log_kappa_ordinary", "CSI": "csi_z", "CPI": "cpi_z"},
    "tensor": {"JS": "js_excess_tensor", "kappa": "log_kappa_tensor", "CSI": "csi_z", "CPI": "cpi_z"},
}
CHECKPOINT_FILE = "inclusion_checkpoint.json"
MIN_BASELINE_WINDOWS = 5
BASELINE_MINUTES = (10, 19)  # inclusive range of window_end_min forming the baseline block
TRAJECTORY_MINUTES = tuple(range(9, -1, -1))  # window_end_min 9 .. 0
FINAL_MINUTE = 0

_WINDOW_COLUMNS = ("participant_id", "recording_id", "group", "event_type", "period", "window_role",
                   "window_end_min")
_EVENT_COLUMNS = ("recording_id", "participant_id", "group", "event_type")
_PARTICIPANT_COLUMNS = ("participant_id", "group")
_TRUE = {"true", "1", "1.0", "t", "yes"}
_FALSE = {"false", "0", "0.0", "f", "no"}


@dataclass(frozen=True)
class ReleaseData:
    """The three release tables, normalised (boolean flags parsed, string columns stripped)."""

    windows: pd.DataFrame
    events: pd.DataFrame
    participants: pd.DataFrame

    @classmethod
    def from_frames(cls, windows: pd.DataFrame, events: pd.DataFrame, participants: pd.DataFrame) -> "ReleaseData":
        _require(windows, _WINDOW_COLUMNS, "windows.csv.gz")
        _require(events, _EVENT_COLUMNS, "events.csv")
        _require(participants, _PARTICIPANT_COLUMNS, "participants.csv")
        windows, events, participants = windows.copy(), events.copy(), participants.copy()
        for frame in (windows, events, participants):
            for column in ("participant_id", "recording_id", "group", "event_type", "period", "window_role"):
                if column in frame:
                    frame[column] = frame[column].astype("string").str.strip().astype(object)
        windows["window_end_min"] = pd.to_numeric(windows["window_end_min"], errors="coerce")
        if "semiology" not in events:                  # bundles written before the column existed
            events["semiology"] = ""
        events["semiology"] = events["semiology"].fillna("").astype(str).str.strip().str.lower()
        for name in REPRESENTATIONS:
            flag = f"eligible_{name}"
            _require(events, (flag,), "events.csv")
            _require(participants, (flag,), "participants.csv")
            events[flag] = _as_bool(events[flag], f"events.csv: {flag}")
            participants[flag] = _as_bool(participants[flag], f"participants.csv: {flag}")
        if events["recording_id"].duplicated().any():
            raise ReleaseSchemaError("events.csv: recording_id is not unique")
        if participants["participant_id"].duplicated().any():
            raise ReleaseSchemaError("participants.csv: participant_id is not unique")
        return cls(windows=windows, events=events, participants=participants)


@dataclass(frozen=True)
class RepresentationData:
    """Eligible-sample quantities for one representation.

    ``events``: one row per eligible seizure and outcome with ``baseline_mean``, ``n_baseline`` (valid baseline
    windows), ``final`` (final-minute value), ``interictal_mean`` (the participant's interictal mean),
    ``change`` (final minus baseline mean) and ``change_vs_interictal`` (final minus interictal mean).
    ``minutes``: one row per eligible seizure, outcome and minute with ``value`` and ``change`` (value minus the
    seizure's baseline mean).
    ``interictal``: one row per eligible participant and outcome with the mean over interictal windows.
    ``participants``: eligible participants with their group.
    """

    name: str
    columns: Mapping[str, str]
    events: pd.DataFrame
    minutes: pd.DataFrame
    interictal: pd.DataFrame
    participants: pd.DataFrame


def _require(frame: pd.DataFrame, columns: tuple[str, ...], label: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ReleaseSchemaError(f"{label}: missing column(s) {missing}")


def _as_bool(series: pd.Series, label: str) -> pd.Series:
    if series.dtype == bool:
        return series
    text = series.astype(str).str.strip().str.lower()
    mapped = text.map({**{k: True for k in _TRUE}, **{k: False for k in _FALSE}})
    if mapped.isna().any():
        raise ReleaseSchemaError(f"{label}: values must be true/false, found {sorted(set(text[mapped.isna()]))}")
    return mapped.astype(bool)


def check_gate(release_dir: str | os.PathLike[str], prereg_id: str) -> dict:
    """Return the inclusion checkpoint, or raise ``StatisticsGateError`` when it is absent or for another id."""
    path = Path(release_dir) / CHECKPOINT_FILE
    if not path.is_file():
        raise StatisticsGateError(f"{path} is missing: run the checkpoint stage for '{prereg_id}' first")
    try:
        checkpoint = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as error:
        raise StatisticsGateError(f"{path} is not valid JSON: {error}") from error
    found = checkpoint.get("prereg_id") if isinstance(checkpoint, dict) else None
    if found != prereg_id:
        raise StatisticsGateError(
            f"{path} was written for preregistration id {found!r}, but statistics were requested for {prereg_id!r}")
    return checkpoint


def load_release(release_dir: str | os.PathLike[str]) -> ReleaseData:
    """Read ``windows.csv.gz``, ``events.csv`` and ``participants.csv`` from a release directory."""
    root = Path(release_dir)
    for name in ("windows.csv.gz", "events.csv", "participants.csv"):
        if not (root / name).is_file():
            raise ReleaseSchemaError(f"{root / name} is missing")
    return ReleaseData.from_frames(
        pd.read_csv(root / "windows.csv.gz", low_memory=False),
        pd.read_csv(root / "events.csv"),
        pd.read_csv(root / "participants.csv"),
    )


def prepare_representation(data: ReleaseData, name: str) -> RepresentationData:
    """Derive the eligible-sample event, minute and interictal quantities for representation ``name``."""
    columns = REPRESENTATIONS[name]
    _require(data.windows, tuple(columns.values()), "windows.csv.gz")
    flag = f"eligible_{name}"
    participants = data.participants.loc[data.participants[flag], ["participant_id", "group"]]
    participants = participants.reset_index(drop=True)
    events = data.events[data.events[flag] & data.events["participant_id"].isin(participants["participant_id"])]
    events = events[["recording_id", "participant_id", "group", "event_type", "semiology"]].reset_index(drop=True)

    windows = data.windows
    preictal = windows[(windows["period"] == "preictal") & windows["recording_id"].isin(events["recording_id"])]
    if preictal.duplicated(["recording_id", "window_end_min"]).any():
        raise ReleaseSchemaError("windows.csv.gz: duplicate (recording_id, window_end_min) rows")
    interictal = windows[(windows["period"] == "interictal") & windows["participant_id"].isin(participants["participant_id"])]

    low, high = BASELINE_MINUTES
    baseline = preictal[(preictal["window_end_min"] >= low) & (preictal["window_end_min"] <= high)]
    final = preictal[preictal["window_end_min"] == FINAL_MINUTE]
    trajectory = preictal[preictal["window_end_min"].isin(TRAJECTORY_MINUTES)]
    group_of = participants.set_index("participant_id")["group"]

    event_parts, minute_parts, interictal_parts = [], [], []
    for outcome, column in columns.items():
        grouped = baseline.groupby("recording_id")[column]
        base = pd.DataFrame({"baseline_mean": grouped.mean(), "n_baseline": grouped.count()})
        table = events.merge(base, left_on="recording_id", right_index=True, how="left")
        table["n_baseline"] = table["n_baseline"].fillna(0).astype(int)
        table.loc[table["n_baseline"] < MIN_BASELINE_WINDOWS, "baseline_mean"] = np.nan
        table["final"] = table["recording_id"].map(final.set_index("recording_id")[column])
        ictal = interictal.groupby("participant_id")[column].agg(interictal_mean="mean", n_windows="count")
        table["interictal_mean"] = table["participant_id"].map(ictal["interictal_mean"])
        table["change"] = table["final"] - table["baseline_mean"]
        table["change_vs_interictal"] = table["final"] - table["interictal_mean"]
        table.insert(5, "outcome", outcome)
        event_parts.append(table)

        minute = trajectory[["recording_id", "window_end_min", column]].rename(columns={column: "value"})
        minute = minute.dropna(subset=["value"]).merge(
            table[["recording_id", "participant_id", "group", "event_type", "baseline_mean"]], on="recording_id")
        minute["change"] = minute["value"] - minute["baseline_mean"]
        minute.insert(1, "outcome", outcome)
        minute_parts.append(minute.drop(columns="baseline_mean"))

        ictal = ictal.reset_index()
        ictal.insert(1, "group", ictal["participant_id"].map(group_of))
        ictal.insert(2, "outcome", outcome)
        interictal_parts.append(ictal.rename(columns={"interictal_mean": "mean"}))

    return RepresentationData(
        name=name,
        columns=columns,
        events=pd.concat(event_parts, ignore_index=True),
        minutes=pd.concat(minute_parts, ignore_index=True),
        interictal=pd.concat(interictal_parts, ignore_index=True),
        participants=participants,
    )
