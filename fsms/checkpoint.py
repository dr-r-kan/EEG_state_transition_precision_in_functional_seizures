"""Inclusion checkpoint: the final number of participants and seizures, fixed before any statistic is computed.

The checkpoint reads only the release bundle.  It reports, per representation (matrix, tensor) and group, the
participants and seizures that enter the analyses, the reason for every exclusion, and the within-participant
pairs available in dual-diagnosis participants.  It is bound to the preregistration identifier supplied by the
analyst; the statistics refuse to run without a checkpoint for the same identifier.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from . import __version__
from .atomic import atomic_write_json, read_json, sha256_file
from .eligibility import REPRESENTATIONS

CHECKPOINT_NAME = "inclusion_checkpoint.json"
GROUPS = ("FS", "ES", "DD")
_ID = re.compile(r"^[a-z0-9]{5}$")


class CheckpointError(RuntimeError):
    pass


def parse_prereg_id(value: str) -> str:
    """Accept an OSF identifier (``abc12``) or its URL (``https://osf.io/abc12``); return the lower-case identifier."""
    text = str(value or "").strip().rstrip("/")
    candidate = text.rsplit("/", 1)[-1].lower() if text else ""
    if not _ID.fullmatch(candidate):
        raise CheckpointError(f"{value!r} is not a preregistration identifier: expected the 5-character registration "
                              "serial, e.g. 'abc12' or 'https://osf.io/abc12'")
    return candidate


def _semiology_counts(events: pd.DataFrame) -> dict:
    """Eligible FS seizures per semiology category (exploratory analysis E1); empty if the bundle has no column."""
    if "semiology" not in events:
        return {}
    labels = events.loc[events.event_type == "FS", "semiology"].fillna("").astype(str).replace("", "not_recorded")
    return {str(k): int(v) for k, v in labels.value_counts().sort_index().items()}


def compute(release_dir: Path, rules: dict, prereg_id: str) -> dict:
    events = pd.read_csv(release_dir / "events.csv")
    participants = pd.read_csv(release_dir / "participants.csv")
    k_info = {}
    k_path = release_dir / "microstate_solution" / "k_selection.json"
    if k_path.exists():
        k_info = json.loads(k_path.read_text(encoding="utf-8"))
    representations = {}
    for name in REPRESENTATIONS:
        elig_p = participants[participants[f"eligible_{name}"]]
        elig_e = events[events[f"eligible_{name}"]]
        pairs = (elig_e[elig_e.group == "DD"].groupby(["participant_id", "event_type"]).size().unstack(fill_value=0)
                 if (elig_e.group == "DD").any() else pd.DataFrame())
        both = int(((pairs.get("FS", 0) > 0) & (pairs.get("ES", 0) > 0)).sum()) if len(pairs) else 0
        representations[name] = {
            "participants": {g: int((elig_p.group == g).sum()) for g in GROUPS},
            "seizures": {g: int((elig_e.group == g).sum()) for g in GROUPS},
            "seizures_by_type": {t: int((elig_e.event_type == t).sum()) for t in ("FS", "ES")},
            "dd_participants_with_both_types": both,
            "fs_seizures_by_semiology": _semiology_counts(elig_e),
            "participants_total": int(len(elig_p)), "seizures_total": int(len(elig_e)),
            "participant_exclusions": participants[~participants[f"eligible_{name}"]]
            .groupby([f"exclusion_reason_{name}"]).size().to_dict(),
            "seizure_exclusions": events[~events[f"eligible_{name}"]].groupby([f"exclusion_reason_{name}"]).size().to_dict(),
        }
    funnel = {"participants_in_release": {g: int((participants.group == g).sum()) for g in GROUPS},
              "seizures_in_release": {g: int((events.group == g).sum()) for g in GROUPS}}
    return {"schema": "fsms-inclusion-checkpoint/1", "prereg_id": prereg_id,
            "code_version": __version__,
            "rules": rules, "selected_K": k_info.get("selected_K"), "state_labels": k_info.get("labels"),
            "funnel": funnel, "representations": representations,
            "data_manifest_sha256": sha256_file(release_dir / "MANIFEST.json")}


def render(checkpoint: dict) -> str:
    lines = [f"Inclusion checkpoint (preregistration {checkpoint['prereg_id']})"]
    if checkpoint.get("selected_K"):
        lines.append(f"Microstates: K = {checkpoint['selected_K']} selected by the free-energy elbow")
    lines.append("In release: participants " + ", ".join(f"{g} {n}" for g, n in
                 checkpoint["funnel"]["participants_in_release"].items()) + "; seizures " + ", ".join(
                 f"{g} {n}" for g, n in checkpoint["funnel"]["seizures_in_release"].items()))
    for name, info in checkpoint["representations"].items():
        lines.append(f"\n{name} representation: {info['participants_total']} participants, {info['seizures_total']} seizures")
        for g in GROUPS:
            lines.append(f"  {g}: {info['participants'][g]:>4} participants  {info['seizures'][g]:>4} seizures")
        lines.append(f"  dual-diagnosis participants with both seizure types: {info['dd_participants_with_both_types']}")
        if info.get("fs_seizures_by_semiology"):
            lines.append("  FS seizures by semiology (exploratory): " + ", ".join(
                f"{k} {v}" for k, v in info["fs_seizures_by_semiology"].items()))
        for kind in ("participant", "seizure"):
            reasons = info[f"{kind}_exclusions"]
            if reasons:
                lines.append(f"  {kind} exclusions: " + ", ".join(f"{k or 'none'} {v}" for k, v in reasons.items()))
    return "\n".join(lines)


def write(release_dir: Path, rules: dict, prereg_id: str) -> dict:
    checkpoint = compute(release_dir, rules, parse_prereg_id(prereg_id))
    atomic_write_json(release_dir / CHECKPOINT_NAME, checkpoint)
    return checkpoint


def require(release_dir: Path, prereg_id: str) -> dict:
    """Return the checkpoint if it exists for this identifier and the data bundle is unchanged since."""
    wanted = parse_prereg_id(prereg_id)
    path = release_dir / CHECKPOINT_NAME
    if not path.exists():
        raise CheckpointError("No inclusion checkpoint: run the pipeline through the 'checkpoint' stage first")
    checkpoint = read_json(path)
    if checkpoint.get("prereg_id") != wanted:
        raise CheckpointError(f"The checkpoint belongs to preregistration {checkpoint.get('prereg_id')!r}, "
                              f"not {wanted!r}")
    if checkpoint.get("data_manifest_sha256") != sha256_file(release_dir / "MANIFEST.json"):
        raise CheckpointError("The release bundle changed after the checkpoint was written; rerun the checkpoint stage")
    return checkpoint
