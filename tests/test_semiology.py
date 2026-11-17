"""Semiology in the release bundle and the exploratory hypomotor / hypermotor comparison (E1)."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from synthetic_release import SyntheticSpec, build_tables, write_release  # noqa: E402
from test_release_checkpoint import SETTINGS, synthetic_analysis  # noqa: E402

from fsms import release as R  # noqa: E402
from fsms.stats import run_statistics  # noqa: E402
from fsms.stats.release import ReleaseData, prepare_representation  # noqa: E402
from fsms.stats.semiology import run_semiology  # noqa: E402


class ReleaseSemiologyTests(unittest.TestCase):
    def test_categories(self):
        expected = {"Hypomotor ": "hypo", "HYPER": "hyper", "hypermotor": "hyper", "": "", None: "", float("nan"): "",
                    "nan": "", "shaking, eyes closed": "unclassified"}
        for value, category in expected.items():
            self.assertEqual(R.semiology_category(value), category, value)

    def test_events_carry_the_category_and_never_the_source_text(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            (tmp / "analysis").mkdir()
            index = synthetic_analysis(tmp / "analysis")
            R.build_release(tmp / "analysis", tmp / "release", tmp / "private", None, index, SETTINGS)
            events = pd.read_csv(tmp / "release" / "events.csv", keep_default_na=False)
            self.assertEqual(list(events.columns[:5]),
                             ["recording_id", "participant_id", "group", "event_type", "semiology"])
            self.assertTrue((events.loc[events.event_type != "FS", "semiology"] == "").all())
            self.assertEqual(sorted(events.loc[events.event_type == "FS", "semiology"]),
                             ["hyper", "hyper", "hypo", "hypo", "unclassified"])
            text = (tmp / "release" / "events.csv").read_text(encoding="utf-8").lower()
            self.assertNotIn("shaking", text)
            self.assertNotIn("hypomotor", text)
            R.build_release(tmp / "analysis", tmp / "release", tmp / "private", None, index, SETTINGS)   # rerun
            self.assertEqual((tmp / "release" / "events.csv").read_text(encoding="utf-8").lower(), text)

    def test_manifest_without_the_column_gives_empty_semiology(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            (tmp / "analysis").mkdir()
            index = synthetic_analysis(tmp / "analysis").drop(columns="hypo_hyper")
            R.build_release(tmp / "analysis", tmp / "release", tmp / "private", None, index, SETTINGS)
            events = pd.read_csv(tmp / "release" / "events.csv", keep_default_na=False)
            self.assertTrue((events.semiology == "").all())


class ExploratorySemiologyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tables = build_tables(SyntheticSpec(n_fs=30, n_es=8, n_dd=6, seed=11))
        cls.rep = prepare_representation(ReleaseData.from_frames(*cls.tables), "matrix")
        cls.result = run_semiology(cls.rep)

    def test_matches_independent_computation(self):
        events = self.rep.events
        fs = events[(events.event_type == "FS") & events.semiology.isin(["hypo", "hyper"]) & events.change.notna()]
        self.assertFalse(fs.empty)
        for outcome, row in self.result.between.set_index("outcome").iterrows():
            mine = fs[fs.outcome == outcome]
            kinds = mine.groupby("participant_id").semiology.agg(lambda s: set(s))
            means = mine.groupby("participant_id").change.mean()
            hypo = means[[p for p, k in kinds.items() if k == {"hypo"}]]
            hyper = means[[p for p, k in kinds.items() if k == {"hyper"}]]
            mixed = sum(len(k) == 2 for k in kinds)
            self.assertGreater(mixed, 0)
            self.assertEqual((row.n_hypo, row.n_hyper, row.n_mixed_excluded), (len(hypo), len(hyper), mixed))
            reference = stats.ttest_ind(hypo, hyper, equal_var=False)
            self.assertAlmostEqual(row.t, reference.statistic, places=10)
            self.assertAlmostEqual(row.p, reference.pvalue, places=10)
            self.assertAlmostEqual(row.estimate, hypo.mean() - hyper.mean(), places=12)

    def test_only_functional_seizures_of_fs_and_dd_participants_are_included(self):
        table = self.result.participant_change
        self.assertEqual(set(table.group), {"FS", "DD"})
        es_only = set(self.rep.participants.loc[self.rep.participants.group == "ES", "participant_id"])
        self.assertFalse(es_only & set(table.participant_id))

    def test_bundle_without_semiology_still_runs(self):
        windows, events, participants = self.tables
        rep = prepare_representation(ReleaseData.from_frames(windows, events.drop(columns="semiology"), participants),
                                     "tensor")
        result = run_semiology(rep)
        self.assertTrue(result.participant_change.empty)
        self.assertTrue((result.between.n_hypo == 0).all() and result.between.p.isna().all())

    def test_outputs_are_written_and_marked_exploratory(self):
        with tempfile.TemporaryDirectory() as t:
            release = write_release(Path(t) / "release", SyntheticSpec(seed=3), prereg_id="abc12")
            results = run_statistics(release, Path(t) / "out", "abc12", ("matrix",), n_bootstrap=20)
            block = results["representations"]["matrix"]["E1_exploratory_semiology"]
            self.assertEqual(len(block["between"]), 4)
            table = pd.read_csv(Path(t) / "out" / "matrix" / "tables" / "e1_semiology_between.csv")
            self.assertEqual(list(table.columns[:3]), ["prereg_id", "representation", "analysis"])
            self.assertTrue((table.analysis == "exploratory").all())
            self.assertNotIn("p_holm", table.columns)
            self.assertIn("E1 (exploratory)", (Path(t) / "out" / "results_summary.md").read_text(encoding="utf-8"))
            self.assertFalse(any("E1" in f["analysis"] for f in results["representations"]["matrix"]["holm_families"]))


if __name__ == "__main__":
    unittest.main()
