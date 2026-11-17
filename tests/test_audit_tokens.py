"""The identifier check on statistics outputs ignores the pipeline's own folder names but not identifiers."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fsms import pipeline  # noqa: E402
from fsms import release as R  # noqa: E402
from fsms.config import Config  # noqa: E402


def config(root: Path) -> Config:
    raw = {"schema": "fsms-config/1", "name": "t",
           "paths": {"data_root": str(root / "StudyData"), "run_root": str(root / "FS_MS_analysis" / "dynamics_run")}}
    return Config(raw, root / "config.json")


class AuditTokenTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        run = self.root / "FS_MS_analysis" / "dynamics_run"
        self.index = pd.DataFrame({
            "participant": ["P_ab12cd34ef", "P_99887766aa"], "patient_id": ["40123456", "QSB48703"],
            "file_path": [str(run / "vg_microstates" / "converted_mat" / "P_ab12cd34ef_preict_row5.mat"),
                          str(run / "vg_microstates" / "converted_mat" / "P_99887766aa_II_row6.mat")],
            "source_edf_path": [r"X:\Anon NHNN\EDFs\All\ABCD1234.edf", r"X:\Anon NHNN\EDFs\All\WXYZ5678.edf"]})
        self.cfg = config(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_folder_words_are_dropped_and_identifiers_kept(self):
        plain = R.forbidden_tokens(self.index)
        self.assertIn("analysis", plain)                      # the problem this fixes
        tokens = pipeline.audit_tokens(self.cfg, self.index)
        self.assertNotIn("analysis", tokens)
        self.assertNotIn("dynamics", tokens)
        for identifier in ("p_ab12cd34ef", "40123456", "qsb48703", "abcd1234", "wxyz5678"):
            self.assertIn(identifier, tokens)

    def test_a_folder_word_that_is_also_an_identifier_is_kept(self):
        index = self.index.assign(patient_id=["analysis", "QSB48703"])
        self.assertIn("analysis", pipeline.audit_tokens(self.cfg, index))

    def test_results_with_an_analysis_column_pass_and_identifiers_still_fail(self):
        out = self.root / "results"
        out.mkdir()
        (out / "table.csv").write_text("analysis,estimate\nexploratory,0.1\n", encoding="utf-8")
        tokens = pipeline.audit_tokens(self.cfg, self.index)
        R.audit_directory(out, tokens)
        with self.assertRaises(R.LeakageError):
            R.audit_directory(out, R.forbidden_tokens(self.index))   # the unfiltered check fails on that file
        (out / "notes.txt").write_text("participant 40123456 and file ABCD1234", encoding="utf-8")
        with self.assertRaises(R.LeakageError):
            R.audit_directory(out, tokens)


if __name__ == "__main__":
    unittest.main()
