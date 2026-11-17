"""End-to-end check of the window-metric stage (caches -> metrics -> release -> checkpoint) on synthetic recordings."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import savemat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fsms import checkpoint as cp  # noqa: E402
from fsms import release as R  # noqa: E402
from fsms import windows as W  # noqa: E402

LABELS = ["F", "B", "G", "D", "A"]            # fitted-state order; K = 5 with a class (D) the old six-state set lacked
SFREQ = 250.0
SETTINGS = W.Settings(prior_strength=10.0, max_autonomic_gap_s=5.0, min_window_transitions=30,
                      min_reference_transitions=300, null_draws=40)
RULES = {"min_reference_transitions": 300, "min_baseline_windows": 5, "min_interictal_windows": 5}


def reference_grammar(k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.gamma(1.5, size=(k, k))
    np.fill_diagonal(raw, 0)
    return raw / raw.sum(axis=1, keepdims=True)


def simulate(duration_s: float, rate: float, grammar_at, seed: int):
    """Ordered state changes: exponential gaps, destinations from the grammar valid at that time."""
    rng = np.random.default_rng(seed)
    times = np.cumsum(rng.exponential(1 / rate, int(duration_s * rate * 1.3)))
    times = times[times < duration_s - 0.01]
    state = int(rng.integers(len(LABELS)))
    src, dst = [], []
    for t in times:
        row = grammar_at(t)[state]
        nxt = int(rng.choice(len(LABELS), p=row))
        src.append(state)
        dst.append(nxt)
        state = nxt
    samples = np.unique(np.round(times * SFREQ).astype(np.int64) + 1)
    return samples, np.array(src[: len(samples)]) + 1, np.array(dst[: len(samples)]) + 1


def build_study(root: Path, drift: bool):
    """Three participants (FS, FS, ES); preictal grammar drifts away from the interictal one in the last 3 minutes."""
    base = {p: reference_grammar(len(LABELS), i) for i, p in enumerate(("S01", "S02", "S03"))}
    groups = {"S01": "FS", "S02": "FS", "S03": "ES"}
    cache_dir = root / "caches"
    cache_dir.mkdir()
    index, manifest, native = [], [], []
    number = 0
    for seed, (p, group) in enumerate(groups.items()):
        other = reference_grammar(len(LABELS), 50 + seed)
        for condition, duration, repeat in (("II", 1500.0, 1), ("preict", 1500.0, 2)):
            for r in range(repeat):
                number += 1
                path = f"/data/{p}_{condition}_{r}.mat"
                if condition == "preict" and drift:
                    def grammar(t, b=base[p], o=other, d=duration):
                        w = np.clip((t - (d - 180)) / 180, 0, 1)
                        return (1 - w) * b + w * o
                else:
                    def grammar(t, b=base[p]):
                        return b
                samples, src, dst = simulate(duration, 6.0, grammar, 1000 * number + seed)
                cache = cache_dir / f"{number:04d}.mat"
                savemat(cache, {"transition": {"sfreq": SFREQ, "n_samples": int(duration * SFREQ),
                                               "transition_sample": samples, "from_state": src, "to_state": dst}})
                index.append({"manifest_index": number, "participant": p, "condition": condition, "group": group,
                              "file_path": path, "cache_file": str(cache)})
                manifest.append({"file_path": path, "seizure_type": group if condition == "preict" else "",
                                 "patient_id": f"0000{number}77"})
                t = np.arange(8.0, duration - 8, 5.0)
                level = np.sin(t / 60.0) + (np.clip((t - (duration - 180)) / 180, 0, 1) * 1.5 if condition == "preict" and group == "FS" else 0)
                native.append(pd.DataFrame({"file_path": path, "event_time_s": t, "csi_z": level, "cpi_z": -level / 2}))
    pd.DataFrame(index).to_csv(root / "transition_index.csv", index=False)
    pd.DataFrame(manifest).to_csv(root / "manifest.csv", index=False)
    pd.concat(native).to_csv(root / "autonomic.csv.gz", index=False)
    (root / "recovery_audit.json").write_text(json.dumps({"fit_labels": LABELS}), encoding="utf-8")


class WindowPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        build_study(cls.root, drift=True)
        cls.summary = W.run(cls.root / "transition_index.csv", cls.root / "autonomic.csv.gz", cls.root / "manifest.csv",
                            cls.root / "recovery_audit.json", cls.root / "analysis", SETTINGS, log=lambda m: None)
        cls.windows = pd.read_csv(cls.root / "analysis" / "window_metrics.csv.gz")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_states_are_alphabetical_labels(self):
        self.assertEqual(self.summary["states"], ["A", "B", "D", "F", "G"])
        counts = pd.read_csv(self.root / "analysis" / "window_counts.csv.gz")
        self.assertTrue(set(counts.from_state) <= set("ABDFG"))

    def test_window_roles_and_final_minute(self):
        pre = self.windows[self.windows.period == "preictal"]
        self.assertEqual(set(pre.window_role), {"baseline", "minute"})
        self.assertEqual(int(pre.window_end_min.min()), 0)
        self.assertEqual(int(pre.window_end_min.max()), 19)
        self.assertTrue(self.windows[self.windows.period == "interictal"].window_end_min.isna().all())

    def test_drift_raises_js_in_final_minutes_and_csi_for_fs(self):
        pre = self.windows[self.windows.period == "preictal"]
        final = pre[pre.window_end_min == 0].js_excess_ordinary.mean()
        baseline = pre[pre.window_role == "baseline"].js_excess_ordinary.mean()
        self.assertGreater(final, baseline + 0.02)
        fs = pre[(pre.group == "FS") & (pre.window_end_min == 0)].csi_z.mean()
        es = pre[(pre.group == "ES") & (pre.window_end_min == 0)].csi_z.mean()
        self.assertGreater(fs, es + 0.5)

    def test_interictal_excess_js_is_centred_near_zero(self):
        ii = self.windows[self.windows.period == "interictal"].js_excess_ordinary
        self.assertLess(abs(ii.mean()), 0.01)

    def test_release_and_checkpoint_from_these_windows(self):
        index = W.load_index(self.root / "transition_index.csv", self.root / "manifest.csv")
        R.build_release(self.root / "analysis", self.root / "release", self.root / "private", None, index, RULES)
        checkpoint = cp.write(self.root / "release", RULES, "https://osf.io/abc12")
        matrix = checkpoint["representations"]["matrix"]
        self.assertEqual(matrix["participants"], {"FS": 2, "ES": 1, "DD": 0})
        self.assertEqual(matrix["seizures"]["FS"] + matrix["seizures"]["ES"], 6)
        events = pd.read_csv(self.root / "release" / "events.csv")
        self.assertTrue(events.eligible_matrix.all())
        windows = pd.read_csv(self.root / "release" / "windows.csv.gz")
        self.assertNotIn("S01", " ".join(map(str, windows.participant_id.unique())))


if __name__ == "__main__":
    unittest.main()
