"""Which seizures and participants enter the statistics.

Eligibility is decided separately for the matrix and the tensor representation, so that the four outcomes of a
representation (JS, kappa, CSI, CPI) always rest on the same sample.

A preictal recording (one seizure) is eligible when

* its participant has an interictal reference grammar (``min_reference_transitions`` transitions),
* its final-minute window (``window_end_min`` 0) has all four outcomes, and
* at least ``min_baseline_windows`` of its ten baseline windows (20 to 10 minutes before onset) have all four.

A participant is eligible when the reference exists, at least ``min_interictal_windows`` interictal windows have
all four outcomes, and at least one of their seizures is eligible.  Every exclusion carries one reason.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REPRESENTATIONS = {"matrix": ("js_excess_ordinary", "log_kappa_ordinary"),
                   "tensor": ("js_excess_tensor", "log_kappa_tensor")}
AUTONOMIC = ("csi_z", "cpi_z")
REASONS = ("no_interictal_reference", "too_few_interictal_windows", "no_final_minute_window",
           "final_minute_incomplete", "too_few_baseline_windows", "no_eligible_seizure")


def complete(windows: pd.DataFrame, representation: str) -> pd.Series:
    """True for windows with JS, kappa, CSI and CPI all finite."""
    columns = [*REPRESENTATIONS[representation], *AUTONOMIC]
    return windows[columns].notna().all(axis=1) & np.isfinite(windows[columns].to_numpy(float)).all(axis=1)


def classify_events(windows: pd.DataFrame, recordings: pd.DataFrame, reference_ok: dict[str, bool],
                    min_baseline_windows: int, recording_key: str = "recording_id",
                    participant_key: str = "participant_id") -> pd.DataFrame:
    """One row per preictal recording with eligibility and a single exclusion reason per representation."""
    preictal = recordings[recordings.condition == "preict"]
    rows = []
    for record in preictal.itertuples(index=False):
        mine = windows[windows[recording_key] == getattr(record, recording_key)]
        row = {recording_key: getattr(record, recording_key), participant_key: getattr(record, participant_key),
               "group": record.group, "event_type": record.event_type}
        for name in REPRESENTATIONS:
            ok = complete(mine, name) if len(mine) else pd.Series(dtype=bool)
            final = mine[(mine.window_end_min == 0)]
            final_ok = bool(len(final)) and bool(ok[final.index].all())
            baseline_ok = int((ok & mine.window_role.eq("baseline")).sum()) if len(mine) else 0
            reason = ""
            if not reference_ok.get(getattr(record, participant_key), False):
                reason = "no_interictal_reference"
            elif final.empty:
                reason = "no_final_minute_window"
            elif not final_ok:
                reason = "final_minute_incomplete"
            elif baseline_ok < min_baseline_windows:
                reason = "too_few_baseline_windows"
            row.update({f"n_baseline_windows_{name}": baseline_ok, f"has_final_minute_{name}": final_ok,
                        f"eligible_{name}": reason == "", f"exclusion_reason_{name}": reason})
        rows.append(row)
    return pd.DataFrame(rows)


def classify_participants(windows: pd.DataFrame, recordings: pd.DataFrame, events: pd.DataFrame,
                          reference_ok: dict[str, bool], min_interictal_windows: int,
                          participant_key: str = "participant_id") -> pd.DataFrame:
    rows = []
    for participant, mine in recordings.groupby(participant_key, sort=True):
        group = mine.group.iloc[0]
        ii = windows[(windows[participant_key] == participant) & windows.window_role.eq("interictal")]
        row = {participant_key: participant, "group": group,
               "n_interictal_recordings": int((mine.condition == "ii").sum()),
               "n_interictal_windows": int(len(ii)), "n_interictal_transitions": int(ii.n_transitions.sum()),
               "n_preictal_recordings": int((mine.condition == "preict").sum())}
        for name in REPRESENTATIONS:
            n_ii = int(complete(ii, name).sum()) if len(ii) else 0
            n_events = int(events[(events[participant_key] == participant)][f"eligible_{name}"].sum()) if len(events) else 0
            reason = ""
            if not reference_ok.get(participant, False):
                reason = "no_interictal_reference"
            elif n_ii < min_interictal_windows:
                reason = "too_few_interictal_windows"
            elif n_events == 0:
                reason = "no_eligible_seizure"
            row.update({f"n_eligible_events_{name}": n_events, f"eligible_{name}": reason == "",
                        f"exclusion_reason_{name}": reason})
        rows.append(row)
    return pd.DataFrame(rows)


def apply_participant_gate(events: pd.DataFrame, participants: pd.DataFrame,
                           participant_key: str = "participant_id") -> pd.DataFrame:
    """A seizure only counts when its participant is eligible as well (interictal sample large enough)."""
    events = events.copy()
    for name in REPRESENTATIONS:
        ok = participants.set_index(participant_key)[f"eligible_{name}"]
        gate = events[participant_key].map(ok).fillna(False).astype(bool)
        blocked = events[f"eligible_{name}"] & ~gate
        events.loc[blocked, f"exclusion_reason_{name}"] = "too_few_interictal_windows"
        events[f"eligible_{name}"] = events[f"eligible_{name}"] & gate
    return events
