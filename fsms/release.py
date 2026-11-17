"""De-identified release bundle: the only input of the statistics, and the part of a run that can be published.

``build_release`` converts the window tables (keyed by source identifiers, recording paths and clinical
identifiers) into the bundle described in ``docs/DATA_DICTIONARY.md``:

* participants and recordings get random pseudonyms; the key stays in ``<run_root>/private`` and is created once;
* file paths, clinical identifiers, dates and free text are dropped, times stay relative to the recording or onset;
* ``audit_directory`` then scans every produced file for any source identifier, file stem, drive path, e-mail
  address or absolute date and fails the release if it finds one.
"""
from __future__ import annotations

import gzip
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from . import eligibility as E
from . import labels as L
from .atomic import atomic_write_json, sha256_file
from .windows import normalise_path

WINDOW_COLUMNS = ["participant_id", "recording_id", "group", "event_type", "period", "window_role", "window_end_min",
                  "n_transitions", "n_action_transitions", "js_ordinary", "js_null_ordinary", "js_excess_ordinary",
                  "kappa_ordinary", "log_kappa_ordinary", "js_excess_tensor", "log_kappa_tensor",
                  "csi_z", "cpi_z", "csi_gradient", "cpi_gradient"]
_DATE = re.compile(r"(?<!\d)(?:19|20)\d{2}[-_/.](?:0?[1-9]|1[0-2])[-_/.](?:0?[1-9]|[12]\d|3[01])(?!\d)")
_DRIVE = re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/]")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_TOKEN = re.compile(r"[A-Za-z0-9]+")


SEMIOLOGY_LEVELS = ("hypo", "hyper")


class LeakageError(RuntimeError):
    pass


class ReleaseError(RuntimeError):
    pass


# ------------------------------------------------------------------------------ semiology
semiology_category = L.semiology_category


def semiology_by_recording(index: pd.DataFrame, labels: pd.DataFrame | None = None) -> dict[str, str]:
    """Normalised recording path -> semiology category.

    With ``labels`` (the table written by the ``labels`` stage) the category comes from the originating workbooks,
    joined on ``(source, source_sheet, source_row, event_number)``.  Without it (bundles built outside the
    pipeline) the manifest column ``hypo_hyper`` is used, and a manifest without that column gives no semiology.
    """
    if "file_path" not in index:
        return {}
    paths = normalise_path(index.file_path)
    if labels is None:
        return dict(zip(paths, index.hypo_hyper.map(semiology_category))) if "hypo_hyper" in index else {}
    missing = [name for name in L.KEY if name not in index]
    if missing:
        raise ReleaseError(f"The manifest lacks the label key column(s) {missing}")
    table = L.lookup(labels)
    keys = zip(*(index[name].map(L.text) for name in L.KEY))
    return {path: table[key] for path, key in zip(paths, keys) if key in table}


# ------------------------------------------------------------------------------ pseudonyms
def pseudonym_key(private_dir: Path, participants: list[str], recordings: dict[str, list[str]], seed: int | None = None
                  ) -> tuple[dict[str, str], dict[str, str]]:
    """Persistent random pseudonyms for participants (``P001``) and recordings (``P001_R03``).

    The key file lives in ``private_dir``; existing entries are reused and new identifiers appended, so a rerun
    (or a later addition of participants) never changes a pseudonym that was already issued.
    """
    path = private_dir / "pseudonym_key.csv"
    known: dict[tuple[str, str], str] = {}
    if path.exists():
        frame = pd.read_csv(path, dtype=str)
        known = {(r.kind, r.source): r.pseudonym for r in frame.itertuples()}
    rng = np.random.default_rng(seed)                  # unseeded in production: the key file is the record
    prefix = next(c for c in "PSQRTUVWXYZ" if not any(re.fullmatch(rf"{c}\d+", p, re.I) for p in participants))
    issued = {v for v in known.values()}
    missing_participants = [p for p in participants if ("participant", p) not in known]
    start = 1 + sum(1 for k in known if k[0] == "participant")
    for offset, source in enumerate(rng.permutation(missing_participants)):
        known[("participant", str(source))] = f"{prefix}{start + offset:03d}"
    for participant in participants:
        pseudonym = known[("participant", participant)]
        have = sum(1 for k in known if k[0] == "recording" and known[k].startswith(pseudonym + "_R"))
        todo = [r for r in recordings.get(participant, []) if ("recording", r) not in known]
        for offset, source in enumerate(rng.permutation(todo)):
            known[("recording", str(source))] = f"{pseudonym}_R{have + offset + 1:02d}"
    assert len(set(known.values())) == len(known) and not (issued - set(known.values()))
    private_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([(k[0], k[1], v) for k, v in sorted(known.items())],
                 columns=["kind", "source", "pseudonym"]).to_csv(path, index=False)
    return ({p: known[("participant", p)] for p in participants},
            {r: known[("recording", r)] for rs in recordings.values() for r in rs})


# ------------------------------------------------------------------------------ leakage audit
def forbidden_tokens(index: pd.DataFrame) -> set[str]:
    """Identifier strings that must not appear anywhere in published files (lower case)."""
    tokens: set[str] = set()
    for column in ("participant", "patient_id"):
        if column in index:
            tokens |= {str(v).strip().lower() for v in index[column].dropna().unique()}
    for column in ("file_path", "source_raw_fif", "source_edf_path", "source_migrated_edf_path"):
        if column in index:
            for value in index[column].dropna().unique():
                stem = Path(str(value).replace("\\", "/")).stem.lower()
                tokens.add(stem)
                tokens |= {t for t in _TOKEN.findall(str(value).lower()) if len(t) >= 8}
    return {t for t in tokens if t and t not in {"nan", "none"}}


def _text_of(path: Path) -> str:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as stream:
        return stream.read()


def _find_identifiers(text: str, tokens: set[str]) -> set[str]:
    """Identifiers occurring in ``text``: whole-token match for digits-only and short ones, substring otherwise.

    Digits-only identifiers shorter than five characters are not checked (they cannot be told apart from counts).
    """
    lowered = text.lower()
    hits = set()
    for token in tokens:
        if token.isdigit() and len(token) < 5:
            continue
        if token.isdigit() or len(token) < 5:
            if re.search(rf"(?<![0-9a-z.]){re.escape(token)}(?![0-9a-z])", lowered):
                hits.add(token)
        elif token in lowered:
            hits.add(token)
    return hits


def audit_directory(directory: Path, forbidden: set[str], pseudonyms: set[str] | None = None) -> dict:
    """Fail if any text file under ``directory`` contains a forbidden identifier, path, e-mail or absolute date."""
    allowed = {p.lower() for p in (pseudonyms or set())}
    findings, scanned = [], 0
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".csv", ".gz", ".json", ".md", ".txt"}:
            continue
        text = _text_of(path)
        scanned += 1
        for pattern, label in ((_DRIVE, "drive path"), (_EMAIL, "e-mail address"), (_DATE, "absolute date")):
            if pattern.search(text):
                findings.append((path.name, label))
        for token in sorted(_find_identifiers(text, forbidden - allowed)):
            findings.append((path.name, f"source identifier {token!r}"))
    if findings:
        raise LeakageError("Leakage audit failed: " + "; ".join(f"{name}: {what}" for name, what in findings[:20]))
    return {"files_scanned": scanned, "identifiers_checked": len(forbidden), "findings": 0}


# ------------------------------------------------------------------------------ bundle
def build_release(analysis_dir: Path, release_dir: Path, private_dir: Path, microstate_dir: Path | None,
                  index: pd.DataFrame, settings: dict, prereg_id: str | None = None,
                  labels: pd.DataFrame | None = None) -> dict:
    """Write the de-identified bundle and return its summary (``MANIFEST.json`` is written last).

    ``labels`` is the event-label table of the ``labels`` stage; when given, every seizure in the bundle must have a
    row in it, so a seizure can never silently lose its semiology.
    """
    windows = pd.read_csv(analysis_dir / "window_metrics.csv.gz")
    counts = pd.read_csv(analysis_dir / "window_counts.csv.gz")
    recordings = pd.read_csv(analysis_dir / "recordings.csv")
    reference = pd.read_csv(analysis_dir / "reference_counts.csv.gz")
    recording_sources = {p: list(g.file_path) for p, g in recordings.groupby("participant")}
    participant_ids, recording_ids = pseudonym_key(private_dir, sorted(recordings.participant.unique()), recording_sources)

    for frame in (windows, counts, recordings):
        frame["participant_id"] = frame.participant.map(participant_ids)
        frame["recording_id"] = frame.file_path.map(recording_ids)
    reference["participant_id"] = reference.participant.map(participant_ids)
    reference_ok = {participant_ids[p]: int(g["count"].sum()) >= settings["min_reference_transitions"]
                    for p, g in reference.groupby("participant")}

    events = E.classify_events(windows, recordings, reference_ok, settings["min_baseline_windows"])
    participants = E.classify_participants(windows, recordings, events, reference_ok, settings["min_interictal_windows"])
    events = E.apply_participant_gate(events, participants)
    semiology = semiology_by_recording(index, labels)
    source_of = {pseudonym: normalise_path(source) for source, pseudonym in recording_ids.items()}
    if labels is not None:
        unlabelled = sorted(r for r in events.recording_id if source_of.get(r, "") not in semiology)
        if unlabelled:
            raise ReleaseError(f"{len(unlabelled)} seizure(s) have no row in the event labels (first: {unlabelled[0]}); "
                               "rerun the labels stage")
    events.insert(list(events.columns).index("event_type") + 1, "semiology",
                  events.recording_id.map(lambda r: semiology.get(source_of.get(r, ""), "")))
    for name in E.REPRESENTATIONS:
        participants[f"n_eligible_events_{name}"] = participants.participant_id.map(
            events.groupby("participant_id")[f"eligible_{name}"].sum()).fillna(0).astype(int)

    release_dir.mkdir(parents=True, exist_ok=True)
    out = windows[WINDOW_COLUMNS].sort_values(["participant_id", "recording_id", "period", "window_end_min"])
    out.to_csv(release_dir / "windows.csv.gz", index=False, compression="gzip", float_format="%.8g")
    events.to_csv(release_dir / "events.csv", index=False, float_format="%.8g")
    participants.to_csv(release_dir / "participants.csv", index=False)
    reference[["participant_id", "from_state", "to_state", "count"]].to_csv(
        release_dir / "reference_grammar.csv", index=False)
    window_counts = counts[["participant_id", "recording_id", "window_role", "window_end_min", "from_state", "to_state",
                            "count", "weight_a1", "weight_a2", "weight_a3", "weight_a4"]]
    window_counts.to_csv(release_dir / "window_transition_counts.csv.gz", index=False, compression="gzip",
                         float_format="%.6g")
    if microstate_dir is not None:
        copy_microstate_solution(microstate_dir, release_dir / "microstate_solution")

    forbidden = forbidden_tokens(index)
    pseudonyms = set(participant_ids.values()) | set(recording_ids.values())
    audit = audit_directory(release_dir, forbidden, pseudonyms)
    atomic_write_json(release_dir / "leakage_audit.json", audit)
    return write_manifest(release_dir, prereg_id)


def copy_microstate_solution(source: Path, target: Path) -> None:
    """Copy the K-selection table, state labels and global centres (no individual data) into the bundle.

    Timestamps (keys ending ``_utc``) are dropped from JSON so the bundle is deterministic and holds no dates.
    """
    target.mkdir(parents=True, exist_ok=True)
    for name in ("k_selection.csv", "k_selection.json", "free_energy_starts.csv", "state_labels.csv", "global_centres.csv"):
        if not (source / name).exists():
            continue
        if name.endswith(".json"):
            content = json.loads((source / name).read_text(encoding="utf-8"))
            content = {k: v for k, v in content.items() if not k.endswith("_utc")}
            text = json.dumps(content, indent=2)
            (target / name).write_text(text + chr(10), encoding="utf-8")
        else:
            (target / name).write_bytes((source / name).read_bytes())


def write_manifest(release_dir: Path, prereg_id: str | None = None) -> dict:
    files = sorted(p for p in release_dir.rglob("*") if p.is_file()
                   and p.name not in {"MANIFEST.json", "inclusion_checkpoint.json"})
    manifest = {"schema": "fsms-release/1", "code_version": __version__, "prereg_id": prereg_id,
                "files": {p.relative_to(release_dir).as_posix(): {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
                          for p in files}}
    atomic_write_json(release_dir / "MANIFEST.json", manifest)
    return manifest
