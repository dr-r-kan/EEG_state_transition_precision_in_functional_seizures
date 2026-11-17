"""Event labels from the originating workbooks: verification, semiology rule, idempotence and the release join."""
from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_release_checkpoint import SETTINGS, synthetic_analysis  # noqa: E402

from fsms import labels as L  # noqa: E402
from fsms import pipeline  # noqa: E402
from fsms import release as R  # noqa: E402
from fsms.config import Config  # noqa: E402
from fsms.ledger import Ledger  # noqa: E402

DB_COLUMNS = ["source", "source_sheet", "source_row", "patient_id", "patient_code", "patient_type", "event_type",
              "event_number", "segment_number", "seizure_type", "hypo_hyper"]


def write_sources(root: Path) -> dict[str, Path]:
    """Two small workbooks in the layout of the study's sources, and the database compiled from them."""
    wide = pd.DataFrame([
        {"hospital number": 1001, "category; FS, DD, ES": "FS", "ES/FS Event 1": "FS", "Event 1 classification": "Akinetic",
         "Event 1 score (FS)": 0, "ES/FS Event 2": "FS", "Event 2 classification": "Generalised Motor",
         "Event 2 score (FS)": 1.6},
        {"hospital number": "AB77", "category; FS, DD, ES": "DD", "ES/FS Event 1": "ES",
         "Event 1 classification": "Focal motor with loss of awareness", "Event 1 score (FS)": None,
         "ES/FS Event 2": "FS", "Event 2 classification": "Subjective Sensory", "Event 2 score (FS)": 1.2},
        {"hospital number": 1003, "category; FS, DD, ES": "Sleep disorder", "ES/FS Event 1": "FS",
         "Event 1 classification": "Akinetic", "Event 1 score (FS)": 0},
    ])
    long = pd.DataFrame([
        {"num": 2001, "PatientType": "NEAD", "SeizureType": "", "hypo/hyper": "", "EventType": "II", "EventNumber": 0},
        {"num": 2001, "PatientType": "NEAD", "SeizureType": "NEAD", "hypo/hyper": "hypo", "EventType": "Ict", "EventNumber": 1},
        {"num": 2002, "PatientType": "Epilepsy", "SeizureType": "ES", "hypo/hyper": "Hyper ", "EventType": "Ict",
         "EventNumber": 1, "Semiology1": "Tonic-Clonic"},
    ])
    paths = {"fs_dd_es": root / "wide.xlsx", "metaeventmap": root / "long.xlsx"}
    wide.to_excel(paths["fs_dd_es"], sheet_name="Database_events", index=False)
    long.to_excel(paths["metaeventmap"], sheet_name="Sheet1", index=False)
    rows = [
        ("fs_dd_es", "Database_events", 2, "1001", "P_a", "FS", "II", 0, 1, "", ""),
        ("fs_dd_es", "Database_events", 2, "1001", "P_a", "FS", "Ict", 1, "", "FS", ""),
        ("fs_dd_es", "Database_events", 2, "1001", "P_a", "FS", "Ict", 2, "", "FS", ""),
        ("fs_dd_es", "Database_events", 3, "AB77", "P_b", "DD", "Ict", 1, "", "ES", ""),
        ("fs_dd_es", "Database_events", 3, "AB77", "P_b", "DD", "Ict", 2, "", "FS", ""),
        ("metaeventmap", "Sheet1", 3, "2001", "P_c", "FS", "Ict", 1, "", "FS", "hypo"),
        ("metaeventmap", "Sheet1", 4, "2002", "P_d", "ES", "Ict", 1, "", "ES", "hyper"),
    ]
    with (root / "analysis_database.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(DB_COLUMNS)
        writer.writerows(rows)
    return paths


class ClassificationRuleTests(unittest.TestCase):
    def test_rule(self):
        expected = {
            "Generalised Motor": "hyper", "Focal motor": "hyper", "Generalised; Tonic Clonic": "hyper",
            "generalised; Myoclonic": "hyper", "Focal motor; automotor": "hyper", "Hypermotor": "hyper",
            "Generalised; Myoclonic-atonic": "hyper", "Focal Motor/responding to subjective sensory?": "hyper",
            "Focal Onset with Secondary Generalisation": "hyper", "Focal to Bilateral Tonic-Clonic Seizures": "hyper",
            "Akinetic": "hypo", "Subjective Sensory": "hypo", "Subjective symptoms": "hypo",
            "Aura/Subjective Sensory": "hypo", "Subclinical": "hypo", "Generalised; Atonic": "hypo",
            "Generalised; Atypical Absense": "hypo", "Generalised, Electrographic": "hypo",
            "Unknown onset": "unclassified", "": "", None: "", float("nan"): "",
        }
        for value, category in expected.items():
            self.assertEqual(L.semiology_from_classification(value), category, value)


class BuildTests(unittest.TestCase):
    def test_labels_come_from_the_workbooks_and_are_verified(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            books = write_sources(root)
            audit = L.build(root / "analysis_database.csv", books, root / "labels")
            table = pd.read_csv(root / "labels" / "event_labels.csv", dtype=str, keep_default_na=False)
            self.assertEqual(list(table.columns), L.COLUMNS)
            self.assertEqual(len(table), 6)                                   # every seizure row, no interictal row
            got = {(r.source, r.source_row, r.event_number): (r.semiology, r.semiology_basis) for r in table.itertuples()}
            self.assertEqual(got[("fs_dd_es", "2", "1")], ("hypo", "classification"))
            self.assertEqual(got[("fs_dd_es", "2", "2")], ("hyper", "classification"))
            self.assertEqual(got[("fs_dd_es", "3", "2")], ("hypo", "classification"))
            self.assertEqual(got[("metaeventmap", "3", "1")], ("hypo", "recorded"))
            self.assertEqual(got[("metaeventmap", "4", "1")], ("hyper", "recorded"))
            self.assertTrue(audit["verified_against_workbooks"])
            self.assertEqual(audit["functional_seizures_without_hypo_or_hyper"], 0)
            self.assertEqual(audit["classification_versus_motor_score"], {"compared": 3, "disagree": 1})
            text = (root / "labels" / "event_labels.csv").read_text(encoding="utf-8")
            for identifier in ("1001", "AB77", "2001", "P_a"):
                self.assertNotIn(identifier, text)

    def test_rerun_writes_nothing(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            books = write_sources(root)
            self.assertTrue(L.build(root / "analysis_database.csv", books, root / "labels")["changed"])
            files = [root / "labels" / "event_labels.csv", root / "labels" / "labels_audit.json"]
            before = [(f.stat().st_mtime_ns, f.read_bytes()) for f in files]
            self.assertFalse(L.build(root / "analysis_database.csv", books, root / "labels")["changed"])
            self.assertEqual(before, [(f.stat().st_mtime_ns, f.read_bytes()) for f in files])

    def test_database_that_disagrees_with_the_workbook_is_refused(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            books = write_sources(root)
            database = root / "analysis_database.csv"
            original = database.read_text(encoding="utf-8")
            for old, new, what in (("3,AB77,P_b,DD,Ict,2,,FS", "3,AB77,P_b,DD,Ict,2,,ES", "seizure type"),
                                   ("2,1001,P_a,FS,Ict,1", "2,9999,P_a,FS,Ict,1", "patient"),
                                   ("3,AB77,P_b,DD,Ict,1", "3,AB77,P_b,ES,Ict,1", "group"),
                                   ("4,2002,P_d,ES,Ict,1", "9,2002,P_d,ES,Ict,1", "no such event")):
                self.assertIn(old, original)
                database.write_text(original.replace(old, new), encoding="utf-8")
                with self.assertRaises(L.LabelError, msg=what) as caught:
                    L.build(database, books, root / "labels")
                self.assertIn(what, str(caught.exception))
                self.assertNotIn("AB77", str(caught.exception))

    def test_missing_workbook_or_source_is_refused(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            books = write_sources(root)
            with self.assertRaises(L.LabelError):
                L.build(root / "analysis_database.csv", {"fs_dd_es": books["fs_dd_es"]}, root / "labels")
            with self.assertRaises(L.LabelError):
                L.build(root / "analysis_database.csv", {**books, "metaeventmap": root / "absent.xlsx"}, root / "labels")

    def test_without_workbooks_the_database_column_is_used_and_flagged(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            write_sources(root)
            audit = L.build(root / "analysis_database.csv", None, root / "labels")
            self.assertFalse(audit["verified_against_workbooks"])
            self.assertEqual(audit["functional_seizures_without_hypo_or_hyper"], 3)
            table = pd.read_csv(root / "labels" / "event_labels.csv", dtype=str, keep_default_na=False)
            self.assertEqual(sorted(table.semiology_basis.unique()), ["", "database"])


def keyed_analysis(root: Path):
    """The synthetic analysis tables with label keys on the manifest, and a label table for its seizures."""
    index = synthetic_analysis(root).drop(columns="hypo_hyper")
    recordings = pd.read_csv(root / "recordings.csv")
    index["source"], index["source_sheet"] = "book", "Sheet1"
    index["source_row"] = range(2, 2 + len(index))
    index["event_number"] = 1
    kinds = dict(zip(R.normalise_path(recordings.file_path), recordings.event_type.fillna("")))
    events = index[R.normalise_path(index.file_path).map(kinds).isin(["FS", "ES"])]
    table = pd.DataFrame({"source": "book", "source_sheet": "Sheet1", "source_row": events.source_row.astype(str),
                          "event_number": "1", "group": "", "event_type": "", "semiology": "hyper",
                          "semiology_basis": "recorded"})
    return index, table


class ReleaseJoinTests(unittest.TestCase):
    def test_every_seizure_gets_its_label(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            (tmp / "analysis").mkdir()
            index, table = keyed_analysis(tmp / "analysis")
            table.loc[table.index[0], "semiology"] = "hypo"
            R.build_release(tmp / "analysis", tmp / "release", tmp / "private", None, index, SETTINGS, labels=table)
            events = pd.read_csv(tmp / "release" / "events.csv", keep_default_na=False)
            self.assertEqual(len(events), len(table))
            self.assertEqual(sorted(events.semiology.unique()), ["hyper", "hypo"])
            self.assertEqual(int((events.semiology == "hypo").sum()), 1)

    def test_a_seizure_without_a_label_row_stops_the_release(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            (tmp / "analysis").mkdir()
            index, table = keyed_analysis(tmp / "analysis")
            with self.assertRaises(R.ReleaseError):
                R.build_release(tmp / "analysis", tmp / "release", tmp / "private", None, index, SETTINGS,
                                labels=table.iloc[1:])
            with self.assertRaises(R.ReleaseError):
                R.build_release(tmp / "analysis", tmp / "release", tmp / "private", None,
                                index.drop(columns="source_row"), SETTINGS, labels=table)


class StageTests(unittest.TestCase):
    def context(self, root: Path, books: dict[str, Path] | None):
        raw = {"schema": "fsms-config/1", "name": "t", "paths": {"data_root": str(root), "run_root": str(root / "run")}}
        if books:
            raw["sources"] = {"workbooks": {name: str(path) for name, path in books.items()}}
        config = Config(raw, root / "config.json")
        ctx = pipeline.Context(config, Ledger(config.state_dir))
        ctx.fingerprints["metrics"] = "m"
        return ctx

    def test_order(self):
        order = pipeline.STAGE_ORDER
        self.assertLess(order.index("labels"), order.index("release"))
        self.assertEqual(order[order.index("labels") + 1], "release")

    def test_relabelling_reruns_labels_and_release_only(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            books = write_sources(root)
            ctx = self.context(root, books)
            labels_stage, release_stage = pipeline.STAGES["labels"], pipeline.STAGES["release"]
            first = labels_stage.fingerprint(ctx)
            labels_stage.run(ctx)
            self.assertTrue(all(p.is_file() for p in labels_stage.outputs(ctx)))
            release_first = release_stage.fingerprint(ctx)
            self.assertEqual(labels_stage.fingerprint(ctx), first)

            # a workbook edit that changes no label: labels reruns, the release fingerprint does not move
            frame = pd.read_excel(books["metaeventmap"], sheet_name="Sheet1", dtype=object)
            frame["a note"] = "x"
            frame.to_excel(books["metaeventmap"], sheet_name="Sheet1", index=False)
            self.assertNotEqual(labels_stage.fingerprint(ctx), first)
            labels_stage.run(ctx)
            self.assertEqual(release_stage.fingerprint(ctx), release_first)

            # a changed label moves the release fingerprint
            frame.loc[1, "hypo/hyper"] = "hyper"
            frame.to_excel(books["metaeventmap"], sheet_name="Sheet1", index=False)
            labels_stage.run(ctx)
            self.assertNotEqual(release_stage.fingerprint(ctx), release_first)
            audit = json.loads((ctx.config.labels_dir / "labels_audit.json").read_text(encoding="utf-8"))
            self.assertEqual(audit["seizure_rows"], 6)

    def test_missing_workbook_fails_before_anything_runs(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            books = write_sources(root)
            ctx = self.context(root, {**books, "metaeventmap": root / "absent.xlsx"})
            with self.assertRaises(pipeline.PipelineError):
                pipeline.STAGES["labels"].fingerprint(ctx)

    def test_processing_stages_do_not_depend_on_labels(self):
        source = (Path(pipeline.__file__)).read_text(encoding="utf-8")
        for name in ("_heppy_fp", "_manifest_fp", "_micro_fp", "_auto_fp", "_metrics_fp"):
            body = source.split(f"def {name}(")[1].split("\ndef ")[0]
            self.assertNotIn("labels", body, name)
            self.assertNotIn("workbook", body, name)


if __name__ == "__main__":
    unittest.main()
