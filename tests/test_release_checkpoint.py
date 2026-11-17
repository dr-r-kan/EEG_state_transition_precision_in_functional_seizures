"""Release bundle (pseudonyms, eligibility, leakage audit) and the inclusion checkpoint on synthetic windows."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fsms import checkpoint as cp  # noqa: E402
from fsms import release as R  # noqa: E402

SETTINGS = {"min_reference_transitions": 300, "min_baseline_windows": 5, "min_interictal_windows": 5}


SEMIOLOGY_TEXT = ("Hypomotor ", "hyper", "collapse then shaking")   # source entries; only the category is released


def synthetic_analysis(root: Path, drop_final_for: str | None = None):
    """Window tables for 4 participants (FS, FS, ES, DD) with source identifiers that must never leak."""
    rng = np.random.default_rng(0)
    participants = {"01234567": "FS", "76543210": "FS", "SUBJ-AB": "ES", "DDPAT99": "DD"}
    windows, recordings, counts, reference, index = [], [], [], [], []
    for p, group in participants.items():
        for r, (condition, n_windows) in enumerate((("ii", 12), ("preict", 20), ("preict", 20))):
            path = f"X:\\data\\{p}_2016_10_28_clip{r}.mat"
            event_type = "" if condition == "ii" else ("FS" if group in ("FS", "DD") and r == 1 else "ES" if group != "FS" else "FS")
            recordings.append({"participant": p, "file_path": path, "condition": condition, "group": group,
                               "event_type": event_type})
            index.append({"participant": p, "patient_id": p, "file_path": path.replace("X:", "x:"), "source_edf_path": path,
                          "hypo_hyper": "" if event_type != "FS" else SEMIOLOGY_TEXT[sum(bool(i["hypo_hyper"]) for i in index) % 3]})
            for w in range(n_windows):
                end_min = None if condition == "ii" else w
                role = "interictal" if condition == "ii" else ("minute" if w < 10 else "baseline")
                if drop_final_for == p and end_min == 0 and r == 1:
                    continue
                windows.append({
                    "participant": p, "file_path": path, "group": group, "event_type": event_type,
                    "period": "interictal" if condition == "ii" else "preictal", "window_role": role,
                    "window_end_min": end_min, "window_start_s": 60.0 * w, "n_transitions": 120, "n_action_transitions": 110,
                    "js_ordinary": 0.05, "js_null_ordinary": 0.04, "js_excess_ordinary": rng.normal(0, 0.01),
                    "kappa_ordinary": 1.0, "log_kappa_ordinary": rng.normal(0, 0.1), "js_excess_tensor": rng.normal(0, .01),
                    "log_kappa_tensor": rng.normal(0, .1), "csi_z": rng.normal(), "cpi_z": rng.normal(),
                    "csi_gradient": 0.0, "cpi_gradient": 0.0})
                counts.append((p, path, role, end_min, 60.0 * w, 0, 1, 12, .25, .25, .25, .25, "A", "B"))
        reference += [(p, "A", "B", 400), (p, "B", "A", 380)]
    pd.DataFrame(windows).to_csv(root / "window_metrics.csv.gz", index=False)
    pd.DataFrame(counts, columns=["participant", "file_path", "window_role", "window_end_min", "window_start_s",
                                  "from_index", "to_index", "count", "weight_a1", "weight_a2", "weight_a3", "weight_a4",
                                  "from_state", "to_state"]).to_csv(root / "window_counts.csv.gz", index=False)
    pd.DataFrame(recordings).to_csv(root / "recordings.csv", index=False)
    pd.DataFrame(reference, columns=["participant", "from_state", "to_state", "count"]).to_csv(
        root / "reference_counts.csv.gz", index=False)
    return pd.DataFrame(index)


class ReleaseTests(unittest.TestCase):
    def build(self, tmp: Path, **kwargs):
        analysis = tmp / "analysis"
        analysis.mkdir()
        index = synthetic_analysis(analysis, **kwargs)
        summary = R.build_release(analysis, tmp / "release", tmp / "private", None, index, SETTINGS)
        return index, summary

    def test_bundle_is_deidentified_and_complete(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            _, summary = self.build(tmp)
            release = tmp / "release"
            for name in ("windows.csv.gz", "events.csv", "participants.csv", "reference_grammar.csv",
                         "window_transition_counts.csv.gz", "leakage_audit.json", "MANIFEST.json"):
                self.assertTrue((release / name).exists(), name)
            windows = pd.read_csv(release / "windows.csv.gz")
            self.assertEqual(list(windows.columns), R.WINDOW_COLUMNS)
            self.assertTrue(windows.participant_id.str.fullmatch(r"P\d{3}").all())
            self.assertTrue(windows.recording_id.str.fullmatch(r"P\d{3}_R\d{2}").all())
            self.assertFalse((tmp / "release" / "pseudonym_key.csv").exists())
            self.assertTrue((tmp / "private" / "pseudonym_key.csv").exists())
            self.assertIn("windows.csv.gz", summary["files"])

    def test_pseudonyms_are_stable_across_reruns_and_additions(self):
        with tempfile.TemporaryDirectory() as t:
            private = Path(t)
            first, rec1 = R.pseudonym_key(private, ["a", "b"], {"a": ["a1", "a2"], "b": ["b1"]})
            again, rec2 = R.pseudonym_key(private, ["a", "b", "c"], {"a": ["a1", "a2"], "b": ["b1"], "c": ["c1"]})
            self.assertEqual(first["a"], again["a"])
            self.assertEqual(rec1["a2"], rec2["a2"])
            self.assertEqual(len({*again.values()}), 3)

    def test_eligibility_reasons(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            self.build(tmp, drop_final_for="76543210")
            events = pd.read_csv(tmp / "release" / "events.csv")
            participants = pd.read_csv(tmp / "release" / "participants.csv")
            self.assertEqual(int((events.exclusion_reason_matrix == "no_final_minute_window").sum()), 1)
            self.assertEqual(int(events.eligible_matrix.sum()), 7)
            self.assertTrue(participants.eligible_matrix.all())      # still has one eligible seizure

    def test_leakage_audit_catches_identifiers_dates_and_paths(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            index, _ = self.build(tmp)
            forbidden = R.forbidden_tokens(index)
            self.assertIn("01234567", forbidden)
            release = tmp / "release"
            R.audit_directory(release, forbidden)                    # the clean bundle passes
            for text in ("subject SUBJ-AB seen", "recorded 2016-10-28", "see X:\\data\\file", "mail me@example.org"):
                (release / "notes.txt").write_text(text, encoding="utf-8")
                with self.assertRaises(R.LeakageError, msg=text):
                    R.audit_directory(release, forbidden)
            (release / "notes.txt").unlink()


class CheckpointTests(unittest.TestCase):
    def test_prereg_id_parsing(self):
        self.assertEqual(cp.parse_prereg_id("ABC12"), "abc12")
        self.assertEqual(cp.parse_prereg_id("https://osf.io/nuf95/"), "nuf95")
        for bad in ("", "abc", "abc123", "ab c1", None):
            with self.assertRaises(cp.CheckpointError):
                cp.parse_prereg_id(bad)

    def test_checkpoint_counts_and_gate(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            analysis = tmp / "analysis"
            analysis.mkdir()
            index = synthetic_analysis(analysis)
            R.build_release(analysis, tmp / "release", tmp / "private", None, index, SETTINGS)
            release = tmp / "release"
            with self.assertRaises(cp.CheckpointError):
                cp.require(release, "abc12")                         # no checkpoint yet
            checkpoint = cp.write(release, SETTINGS, "abc12")
            matrix = checkpoint["representations"]["matrix"]
            self.assertEqual(matrix["participants"], {"FS": 2, "ES": 1, "DD": 1})
            self.assertEqual(matrix["seizures"], {"FS": 4, "ES": 2, "DD": 2})
            self.assertEqual(matrix["dd_participants_with_both_types"], 1)
            self.assertEqual(matrix["fs_seizures_by_semiology"], {"hyper": 2, "hypo": 2, "unclassified": 1})
            self.assertIn("FS seizures by semiology", cp.render(checkpoint))
            self.assertIn("matrix representation", cp.render(checkpoint))
            self.assertEqual(cp.require(release, "ABC12")["prereg_id"], "abc12")
            with self.assertRaises(cp.CheckpointError):
                cp.require(release, "zzz99")                         # a different registration
            R.write_manifest(release)                                # bundle rewritten with a new timestamp only
            (release / "events.csv").write_text((release / "events.csv").read_text() + "\n", encoding="utf-8")
            R.write_manifest(release)
            with self.assertRaises(cp.CheckpointError):
                cp.require(release, "abc12")                         # data changed after the checkpoint


if __name__ == "__main__":
    unittest.main()
