"""Event labels from the originating workbooks (stage ``labels``).

The analysis database (``analysis_database.csv``) was compiled from two clinical workbooks.  This stage goes back
to those workbooks for every seizure row of the database and

* **verifies** the database against them: the row found at ``(source, source_sheet, source_row)`` must be the same
  patient, with the same diagnostic group and the same seizure type.  Any disagreement stops the pipeline;
* **derives the semiology category** (``hypo`` / ``hyper``) of every seizure from what the workbook records:

  - ``metaeventmap``: the rater's ``hypo/hyper`` column;
  - ``fs_dd_es``: the rater's ``Event N classification`` (the workbook has no hypo/hyper column), mapped by
    :func:`semiology_from_classification`.

The output, ``<run_root>/labels/event_labels.csv``, has one row per seizure row of the database, keyed by
``(source, source_sheet, source_row, event_number)``; it holds no patient identifier.  The release stage joins it to
the manifest by that key and refuses to build a bundle in which a seizure has no label row.

Labels never enter the preprocessing, microstate or window stages: relabelling reruns ``labels`` and ``release``
only.  Without a ``sources.workbooks`` section in the config (test fixtures) the labels are taken from the
database column ``hypo_hyper`` and the audit records that nothing was verified against a workbook.
"""
from __future__ import annotations

import csv
import io
import re
from collections import Counter
from pathlib import Path

from .atomic import atomic_write_bytes, sha256_file

KEY = ("source", "source_sheet", "source_row", "event_number")
COLUMNS = [*KEY, "group", "event_type", "semiology", "semiology_basis"]
LEVELS = ("hypo", "hyper")
MAX_EVENTS = 7

# Seizure classification -> semiology.  A classification naming any motor manifestation is hypermotor; otherwise one
# naming an akinetic, subjective or electrographic-only event is hypomotor; anything else is unclassified.
_HYPER = re.compile(r"motor|myoclon|(?<![a-z])tonic|(?<![a-z])clonic|gelastic|secondar\w*\s+generali|bilateral")
_HYPO = re.compile(r"akinetic|atonic|absen[cs]e|subjective|sensory|aura|psychic|subclinical|electrographic|"
                   r"behaviou?ral\s+arrest")


class LabelError(RuntimeError):
    pass


def text(value) -> str:
    """Cell value as trimmed text (whole-number floats without the decimal point; blanks and NaN empty)."""
    if value is None:
        return ""
    if isinstance(value, float):
        if value != value:
            return ""
        if value.is_integer():
            return str(int(value))
    out = str(value).strip()
    return "" if out.lower() in {"nan", "nat", "none"} else out


def normal_type(value) -> str:
    key = text(value).upper()
    if key in {"NEAD", "FS"}:
        return "FS"
    if key in {"EPILEPSY", "EPILEPTIC SEIZURE", "ES"}:
        return "ES"
    if key in {"DD", "DUAL DIAGNOSIS"}:
        return "DD"
    return text(value)


def semiology_category(value) -> str:
    """A recorded hypo/hyper entry as a category: ``hypo``, ``hyper``, ``unclassified`` (other text) or empty."""
    entry = text(value).lower()
    if entry in {"", "na", "n/a"}:
        return ""
    if entry.startswith("hyper"):
        return "hyper"
    if entry.startswith("hypo"):
        return "hypo"
    return "unclassified"


def semiology_from_classification(value) -> str:
    """Semiology category of a seizure classification (``Focal Motor``, ``Akinetic``, ``Subjective Sensory`` ...)."""
    entry = " ".join(text(value).lower().split())
    if not entry:
        return ""
    if _HYPER.search(entry):
        return "hyper"
    if _HYPO.search(entry):
        return "hypo"
    return "unclassified"


# ------------------------------------------------------------------------------ workbook readers
def _sheet(path: Path, sheet: str):
    import warnings

    import pandas as pd

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")      # openpyxl warns about workbook features it does not read
            return pd.read_excel(path, sheet_name=sheet, dtype=object)
    except Exception as error:   # missing file, missing sheet, unreadable workbook
        raise LabelError(f"Cannot read sheet {sheet!r} of {path}: {error}") from error


def _need(frame, columns, path: Path) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise LabelError(f"{path.name} lacks column(s) {missing}")


def read_fs_dd_es(path: Path, sheet: str) -> dict[tuple[str, str], dict]:
    """(source_row, event_number) -> labels, from the wide ``Database_events`` sheet (one row per patient)."""
    frame = _sheet(path, sheet)
    _need(frame, ["hospital number", "category; FS, DD, ES"], path)
    out: dict[tuple[str, str], dict] = {}
    for position, row in enumerate(frame.to_dict("records")):
        for event in range(1, MAX_EVENTS + 1):
            kind = text(row.get(f"ES/FS Event {event}"))
            if kind not in {"FS", "ES"}:
                continue
            classification = row.get(f"Event {event} classification")
            score = text(row.get(f"Event {event} score (FS)"))
            out[(str(position + 2), str(event))] = {
                "patient_id": text(row.get("hospital number")), "group": normal_type(row.get("category; FS, DD, ES")),
                "event_type": normal_type(kind), "semiology": semiology_from_classification(classification),
                "semiology_basis": "classification", "motor_score": score}
    return out


def read_metaeventmap(path: Path, sheet: str) -> dict[tuple[str, str], dict]:
    """(source_row, event_number) -> labels, from the long ``Sheet1`` (one row per event)."""
    frame = _sheet(path, sheet)
    _need(frame, ["num", "PatientType", "SeizureType", "hypo/hyper", "EventType", "EventNumber"], path)
    out: dict[tuple[str, str], dict] = {}
    for position, row in enumerate(frame.to_dict("records")):
        if text(row.get("EventType")).upper() not in {"ICT", "ICTAL"}:
            continue
        recorded = semiology_category(row.get("hypo/hyper"))
        classification = "; ".join(filter(None, (text(row.get(f"Semiology{n}")) for n in (1, 2, 3))))
        out[(str(position + 2), text(row.get("EventNumber")))] = {
            "patient_id": text(row.get("num")), "group": normal_type(row.get("PatientType")),
            "event_type": normal_type(row.get("SeizureType")),
            "semiology": recorded or semiology_from_classification(classification),
            "semiology_basis": "recorded" if recorded else "classification", "motor_score": ""}
    return out


READERS = {"fs_dd_es": read_fs_dd_es, "metaeventmap": read_metaeventmap}


# ------------------------------------------------------------------------------ build
def read_database(path: Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return [{k: (v or "").strip() for k, v in row.items()} for row in csv.DictReader(handle)]


def _csv_bytes(rows: list[dict]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=COLUMNS, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _write_if_changed(path: Path, data: bytes) -> bool:
    try:
        if path.read_bytes() == data:
            return False
    except OSError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(path, data)
    return True


def build(database_csv: Path, workbooks: dict[str, Path] | None, out_dir: Path) -> dict:
    """Write ``event_labels.csv`` and ``labels_audit.json`` under ``out_dir``; return the audit."""
    import json

    seizures = [row for row in read_database(database_csv) if row.get("event_type") == "Ict"]
    sources = sorted({row["source"] for row in seizures})
    verified = bool(workbooks)
    tables: dict[tuple[str, str], dict] = {}
    if verified:
        for source in sources:
            if source not in workbooks:
                raise LabelError(f"No workbook configured for database source {source!r} (sources.workbooks)")
            if source not in READERS:
                raise LabelError(f"No reader for source {source!r}; known: {sorted(READERS)}")
            for sheet in sorted({row["source_sheet"] for row in seizures if row["source"] == source}):
                tables[(source, sheet)] = READERS[source](Path(workbooks[source]), sheet)

    rows, problems = [], []
    score_checked = score_disagrees = 0
    for row in seizures:
        key = tuple(row.get(name, "") for name in KEY)
        where = f"{key[0]}/{key[1]} row {key[2]} event {key[3]}"
        if verified:
            found = tables[(key[0], key[1])].get((key[2], key[3]))
            if found is None:
                problems.append(f"{where}: no such event in the workbook")
                continue
            for name, have, want in (("patient", row.get("patient_id", ""), found["patient_id"]),
                                     ("group", row.get("patient_type", ""), found["group"]),
                                     ("seizure type", row.get("seizure_type", ""), found["event_type"])):
                if have.upper() != want.upper():
                    problems.append(f"{where}: {name} differs between database and workbook")
            semiology, basis = found["semiology"], found["semiology_basis"]
            if found["motor_score"] and semiology in LEVELS:
                try:
                    score_checked += 1
                    score_disagrees += (float(found["motor_score"]) > 0) != (semiology == "hyper")
                except ValueError:
                    score_checked -= 1
        else:
            semiology, basis = semiology_category(row.get("hypo_hyper")), "database"
        rows.append({**dict(zip(KEY, key)), "group": row.get("patient_type", ""), "event_type": row.get("seizure_type", ""),
                     "semiology": semiology, "semiology_basis": basis if semiology else ""})
    if problems:
        raise LabelError(f"{len(problems)} database seizure row(s) do not match the originating workbooks: "
                         + "; ".join(problems[:10]))
    keys = [tuple(r[name] for name in KEY) for r in rows]
    if len(set(keys)) != len(keys):
        raise LabelError("Seizure rows of the database are not unique on (source, source_sheet, source_row, event_number)")
    rows.sort(key=lambda r: (r["source"], r["source_sheet"], int(r["source_row"] or 0), r["event_number"]))

    out_dir = Path(out_dir)
    changed = _write_if_changed(out_dir / "event_labels.csv", _csv_bytes(rows))
    counts = Counter((r["source"], r["event_type"], r["semiology"] or "not_recorded") for r in rows)
    audit = {
        "schema": "fsms-labels/1", "seizure_rows": len(rows), "verified_against_workbooks": verified,
        "workbooks": {s: {"file": Path(workbooks[s]).name, "sha256": sha256_file(Path(workbooks[s]))}
                      for s in sources} if verified else {},
        "database_sha256": sha256_file(Path(database_csv)),
        "semiology_by_source_and_seizure_type": [
            {"source": s, "event_type": t, "semiology": m, "n": n} for (s, t, m), n in sorted(counts.items())],
        "functional_seizures_without_hypo_or_hyper": sum(
            1 for r in rows if r["event_type"] == "FS" and r["semiology"] not in LEVELS),
        "classification_versus_motor_score": {"compared": score_checked, "disagree": score_disagrees},
        "labels_sha256": sha256_file(out_dir / "event_labels.csv"),
    }
    _write_if_changed(out_dir / "labels_audit.json", (json.dumps(audit, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    audit["changed"] = changed
    return audit


def lookup(labels) -> dict[tuple[str, str, str, str], str]:
    """Key -> semiology category from the ``event_labels.csv`` table (a DataFrame read as text)."""
    return {tuple(text(row[name]) for name in KEY): text(row["semiology"]) for row in labels.to_dict("records")}
