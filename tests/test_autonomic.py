"""Cardiac autonomic stage (CSI/CPI): numerical properties and reason-coded QC on synthetic R-peak trains."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import trim_mean

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fsms import autonomic  # noqa: E402


def synthetic_rpeaks(duration_s: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    times, t = [], 0.4
    while t < duration_s:
        times.append(t)
        rr = 0.85 + 0.06 * np.sin(2 * np.pi * t / 90) + 0.03 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 0.012)
        t += max(rr, 0.45)
    return np.asarray(times)


def ar1_series(n: int, rho: float, sigma: float, mean: float, seed: int) -> np.ndarray:
    """Stationary AR(1) with marginal SD ``sigma``: the Poincare cloud has SD1 = sigma*sqrt(1-rho), SD2 = sigma*sqrt(1+rho)."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma * np.sqrt(1 - rho ** 2), n)
    x = np.zeros(n)
    for k in range(1, n):
        x[k] = rho * x[k - 1] + noise[k]
    return mean + x


def write_sidecar(path: Path, peaks: np.ndarray, recovered: np.ndarray | None = None) -> None:
    frame = {"rpeak_time_s": peaks, "rpeak_source": "auto"}
    if recovered is not None:
        frame["is_gap_recovered"] = recovered
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(frame).to_csv(path, index=False)


def build_tree(root: Path, participants: int = 2):
    heppy = root / "heppy_output"
    (heppy / "raw_fif").mkdir(parents=True)
    (heppy / "ecg_qc").mkdir()
    rows = []
    for i in range(participants):
        base = f"P_{i:03d}_snips"
        fif = heppy / "raw_fif" / f"{base}_pp_raw.fif"
        fif.write_bytes(b"")
        peaks = synthetic_rpeaks(2200, seed=i)
        recovered = np.zeros(len(peaks), dtype=int)
        recovered[::97] = 1
        write_sidecar(heppy / "ecg_qc" / f"{base}_rpeaks_corrected.csv", peaks, recovered)
        for condition, start, stop in (("II", 0.0, 500.0), ("preict", 600.0, 2000.0)):
            rows.append({"participant": f"P{i}", "condition": condition, "group": "FS",
                         "file_path": str(root / f"P{i}_{condition}.mat"), "source_raw_fif": str(fif),
                         "crop_start_seconds": start, "crop_end_seconds": stop})
    return heppy, pd.DataFrame(rows)


class RobustCovarianceTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.points = np.column_stack([rng.normal(0.9, 0.05, 400), rng.normal(0.9, 0.05, 400)])
        self.points[:, 1] += 0.6 * (self.points[:, 0] - 0.9)

    def test_symmetric_positive_semidefinite_and_two_by_two(self):
        cov = autonomic.robust_cov_hrv(self.points)
        self.assertEqual(cov.shape, (2, 2))
        np.testing.assert_allclose(cov, cov.T, atol=1e-15)
        self.assertGreaterEqual(np.linalg.eigvalsh(cov).min(), -1e-15)

    def test_scale_equivariance(self):
        base = autonomic.robust_cov_hrv(self.points)
        np.testing.assert_allclose(autonomic.robust_cov_hrv(3.0 * self.points), 9.0 * base, rtol=1e-9)

    def test_translation_invariance(self):
        base = autonomic.robust_cov_hrv(self.points)
        np.testing.assert_allclose(autonomic.robust_cov_hrv(self.points + 5.0), base, rtol=1e-9, atol=1e-14)

    def test_approaches_sample_covariance_for_large_samples_with_real_correlation(self):
        rng = np.random.default_rng(4)
        x = rng.multivariate_normal([0, 0], [[1.0, 0.8], [0.8, 1.5]], size=20000)
        np.testing.assert_allclose(autonomic.robust_cov_hrv(x), np.cov(x, rowvar=False), rtol=0.03)

    def test_shrinks_spurious_correlation_and_contracts_eigenvalue_spread_for_small_samples(self):
        rng = np.random.default_rng(5)
        x = rng.normal(size=(6, 2))
        sample = np.linalg.eigvalsh(np.cov(x, rowvar=False))
        shrunk = np.linalg.eigvalsh(autonomic.robust_cov_hrv(x))
        self.assertLess(shrunk[1] / shrunk[0], sample[1] / sample[0])
        white = rng.normal(size=(5000, 2))   # uncorrelated: off-diagonal is shrunk (almost) entirely to zero
        cov = autonomic.robust_cov_hrv(white)
        self.assertLess(abs(cov[0, 1]) / np.sqrt(cov[0, 0] * cov[1, 1]), 0.02)

    def test_constant_column_gives_diagonal_covariance(self):
        x = np.column_stack([np.linspace(0, 1, 10), np.full(10, 0.7)])
        cov = autonomic.robust_cov_hrv(x)
        self.assertEqual(cov[0, 1], 0.0)
        self.assertGreaterEqual(cov[1, 1], 0.0)
        self.assertTrue(np.isfinite(cov).all())

    def test_rejects_too_few_or_non_finite_observations(self):
        with self.assertRaises(ValueError):
            autonomic.robust_cov_hrv(self.points[:3])
        bad = self.points.copy()
        bad[0, 0] = np.nan
        with self.assertRaises(ValueError):
            autonomic.robust_cov_hrv(bad)
        with self.assertRaises(ValueError):
            autonomic.robust_cov_hrv(self.points[:, 0])


class PoincareFeatureTests(unittest.TestCase):
    def test_recovers_known_sd1_sd2_of_an_ar1_ibi_series(self):
        rho, sigma, mean = 0.9, 0.05, 0.9
        ibi = ar1_series(20000, rho, sigma, mean, seed=1)
        sd1, sd2, distance = autonomic.poincare_features(ibi)
        self.assertAlmostEqual(sd1, sigma * np.sqrt(1 - rho), delta=0.05 * sigma * np.sqrt(1 - rho))
        self.assertAlmostEqual(sd2, sigma * np.sqrt(1 + rho), delta=0.05 * sigma * np.sqrt(1 + rho))
        self.assertLess(sd1, sd2)
        self.assertAlmostEqual(distance, mean * np.sqrt(2), delta=0.01)

    def test_uncorrelated_ibis_give_equal_sd1_and_sd2(self):
        rng = np.random.default_rng(2)
        sd1, sd2, _ = autonomic.poincare_features(0.9 + rng.normal(0, 0.05, 20000))
        self.assertAlmostEqual(sd1, 0.05, delta=0.0025)
        self.assertAlmostEqual(sd2, 0.05, delta=0.0025)

    def test_centre_distance_uses_a_trimmed_mean_robust_to_outliers(self):
        ibi = ar1_series(4000, 0.5, 0.03, 1.0, seed=7)
        ibi[::200] = 3.0                      # isolated long intervals: 0.5 % of the points, below the 2.5 % trim
        points = np.column_stack([ibi[:-1], ibi[1:]])
        expected = np.linalg.norm(trim_mean(points, 0.025, axis=0))
        self.assertAlmostEqual(autonomic.poincare_features(ibi)[2], expected, places=12)
        self.assertLess(expected, np.sqrt(2) * 1.05)

    def test_deterministic(self):
        ibi = ar1_series(500, 0.7, 0.04, 0.8, seed=11)
        self.assertEqual(autonomic.poincare_features(ibi), autonomic.poincare_features(ibi.copy()))


class NativeIndicesTests(unittest.TestCase):
    def setUp(self):
        self.peaks = synthetic_rpeaks(900, seed=0)
        self.recovered = np.zeros(len(self.peaks), dtype=bool)
        self.native = autonomic.native_indices(self.peaks, self.recovered)

    def test_columns_and_window_rules(self):
        native = self.native
        expected = {"event_time_s", "window_start_s", "window_end_s", "sd1", "sd2", "ccd", "n_ibi",
                    "recovered_fraction", "mean_rr_s", "mean_hr_bpm", "rr_first_difference_s", "rr_acceleration_s",
                    "rmssd_s", "sd1_recentred", "sd2_recentred", "ccd_recentred", "cpi_hr_component",
                    "csi_hr_component", "cpi_hrv_component", "csi_hrv_component", "cpi", "csi"}
        self.assertTrue(expected <= set(native.columns))
        self.assertGreater(len(native), 300)
        self.assertTrue((native.n_ibi >= 7).all())
        self.assertTrue(((native.window_end_s - native.window_start_s) >= 0.8 * 15.0 - 1e-9).all())
        self.assertTrue((native.window_end_s - native.window_start_s <= 15.0 + 1e-9).all())
        self.assertTrue(native.event_time_s.is_monotonic_increasing)
        self.assertTrue(((native.event_time_s >= native.window_start_s) & (native.event_time_s <= native.window_end_s)).all())
        np.testing.assert_allclose(native.mean_hr_bpm, 60.0 / native.mean_rr_s)
        self.assertTrue((native.sd1 > 0).all() and (native.sd2 > 0).all())

    def test_recentring_and_cpi_csi_composition(self):
        native = self.native
        rr = np.diff(self.peaks)
        ok = (rr >= 60 / autonomic.MAX_RR_BPM) & (rr <= 60 / autonomic.MIN_RR_BPM)
        pairs = ok[:-1] & ok[1:]
        base = np.column_stack([rr[:-1][pairs], rr[1:][pairs]])
        sd01 = np.sqrt(np.linalg.eigvalsh(autonomic.robust_cov_hrv(base))[0])
        d0 = np.linalg.norm(trim_mean(base, 0.025, axis=0))
        # the record-wide baseline is the mean of the recentred estimates
        self.assertAlmostEqual(native.sd1_recentred.mean(), sd01, places=12)
        self.assertAlmostEqual(native.ccd_recentred.mean(), d0, places=12)
        np.testing.assert_allclose(native.cpi_hr_component, native.ccd_recentred)
        # CSI mirrors CPI's heart-rate part around its mean
        np.testing.assert_allclose(native.cpi_hr_component + native.csi_hr_component,
                                   2 * native.ccd_recentred.mean())
        np.testing.assert_allclose(native.cpi_hrv_component, 10 * native.sd1_recentred + 1)
        np.testing.assert_allclose(native.csi_hrv_component, native.sd2_recentred + 1)
        np.testing.assert_allclose(native.cpi, native.cpi_hr_component + native.cpi_hrv_component)
        np.testing.assert_allclose(native.csi, native.csi_hr_component + native.csi_hrv_component)
        # CPI and the heart-rate part of CSI move in opposite directions by construction
        self.assertLess(np.corrcoef(native.cpi_hr_component, native.csi_hr_component)[0, 1], -0.999)

    def test_time_shift_moves_event_times_and_leaves_indices_unchanged(self):
        shifted = autonomic.native_indices(self.peaks + 123.0, self.recovered)
        self.assertEqual(len(shifted), len(self.native))
        np.testing.assert_allclose(shifted.event_time_s, self.native.event_time_s + 123.0)
        for column in ("sd1", "sd2", "ccd", "cpi", "csi", "mean_rr_s", "rmssd_s"):
            np.testing.assert_allclose(shifted[column], self.native[column], rtol=1e-9, atol=1e-12)

    def test_windows_touching_a_gap_in_the_beat_series_are_dropped(self):
        keep = ~((self.peaks > 300) & (self.peaks < 330))
        gapped = autonomic.native_indices(self.peaks[keep], self.recovered[keep])
        before, after = self.peaks[keep][self.peaks[keep] <= 300].max(), self.peaks[keep][self.peaks[keep] >= 330].min()
        spans_gap = (gapped.window_start_s <= after) & (gapped.window_end_s >= after)
        self.assertFalse(spans_gap.any())
        self.assertLess(len(gapped), len(self.native))
        self.assertGreater(before, 290)

    def test_windows_dominated_by_recovered_peaks_are_dropped(self):
        recovered = (self.peaks >= 500) & (self.peaks <= 700)
        flagged = autonomic.native_indices(self.peaks, recovered)
        inside = (flagged.window_start_s >= 505) & (flagged.window_end_s <= 695)
        self.assertFalse(inside.any())
        self.assertTrue((flagged.recovered_fraction <= 0.10 + 1e-12).all())
        self.assertLess(len(flagged), len(self.native))

    def test_rejects_records_that_cannot_support_the_indices(self):
        with self.assertRaisesRegex(ValueError, "fewer than 20 physiological IBI pairs"):
            autonomic.native_indices(np.arange(0.0, 10.0, 3.0), np.zeros(4, dtype=bool))      # 3-s intervals: 20 bpm
        short = synthetic_rpeaks(30, seed=0)
        with self.assertRaisesRegex(ValueError, "fewer than 30 valid 15-second estimates"):
            autonomic.native_indices(short, np.zeros(len(short), dtype=bool))


class StandardisationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as td:
            _, manifest = build_tree(Path(td))
            cls.native, cls.qc = autonomic.build_native(manifest)
        cls.table = autonomic.standardise_autonomic_to_ii(cls.native)

    def test_interictal_windows_have_zero_mean_and_unit_sd_per_participant(self):
        self.assertTrue(self.qc.autonomic_valid.all())
        interictal = self.table[self.table.condition.eq("ii")]
        for index in ("csi", "cpi", "mean_rr_s", "rmssd_s"):
            grouped = interictal.groupby("participant")[f"{index}_z"]
            np.testing.assert_allclose(grouped.mean(), 0.0, atol=1e-10)
            np.testing.assert_allclose(grouped.std(), 1.0, rtol=1e-10)

    def test_other_conditions_use_the_participants_interictal_mean_and_sd(self):
        for participant, part in self.table.groupby("participant"):
            ii = part[part.condition.eq("ii")].csi
            pre = part[part.condition.eq("preict")]
            np.testing.assert_allclose(pre.csi_ii_mean, ii.mean())
            np.testing.assert_allclose(pre.csi_ii_sd, ii.std())
            np.testing.assert_allclose(pre.csi_z, (pre.csi - ii.mean()) / ii.std())

    def test_rr_surprise_is_standardised_on_the_interictal_baseline(self):
        interictal = self.table[self.table.condition.eq("ii")]
        self.assertGreater(interictal.rr_surprise.notna().mean(), 0.9)    # the first two windows of a file lack lags
        np.testing.assert_allclose(interictal.groupby("participant").rr_surprise_z.mean(), 0.0, atol=1e-10)
        self.assertGreater(self.table.rr_surprise_z.notna().sum(), 100)

    def test_participant_without_interictal_variation_gets_missing_z_scores(self):
        frame = pd.DataFrame({"participant": ["A"] * 4, "file_path": "a", "event_time_s": np.arange(4.0),
                              "condition": ["ii", "ii", "preict", "preict"]})
        for column in ("csi", "cpi", "mean_rr_s", "mean_hr_bpm", "rr_first_difference_s", "rr_acceleration_s",
                       "rmssd_s"):
            frame[column] = [1.0, 1.0, 2.0, 3.0]                                  # constant interictal values: SD 0
        table = autonomic.standardise_autonomic_to_ii(frame)
        self.assertTrue(table.csi_z.isna().all())
        self.assertTrue(table.rr_surprise.isna().all())


class SidecarAndQcTests(unittest.TestCase):
    def test_sidecar_resolution_from_fif_location(self):
        raw_fif = Path("run") / "heppy_output" / "raw_fif"
        for name in ("P_001_snips_pp_raw_keepcfa.fif", "P_001_snips_pp_raw.fif", "P_001_snips.fif"):
            path = autonomic.sidecar_for(str(raw_fif / name))
            self.assertEqual(path, Path("run") / "heppy_output" / "ecg_qc" / "P_001_snips_rpeaks_corrected.csv")

    def test_reader_sorts_drops_unparsable_times_and_defaults_missing_flags(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "s.csv"
            path.write_text("rpeak_time_s,rpeak_source\n3.0,a\nnan,a\n1.0,a\nbad,a\n2.0,a\n", encoding="utf-8")
            peaks, recovered = autonomic.read_rpeaks_with_recovery_flags(path)
            np.testing.assert_array_equal(peaks, [1.0, 2.0, 3.0])
            self.assertFalse(recovered.any())
            path.write_text("rpeak_time_s,is_gap_recovered\n2.0,1\n1.0,0\n", encoding="utf-8")
            peaks, recovered = autonomic.read_rpeaks_with_recovery_flags(path)
            np.testing.assert_array_equal(peaks, [1.0, 2.0])
            np.testing.assert_array_equal(recovered, [False, True])
            peaks, recovered = autonomic.read_rpeaks_with_recovery_flags(Path(td) / "absent.csv")
            self.assertEqual((len(peaks), len(recovered)), (0, 0))

    def test_missing_sidecar_is_reason_coded_not_silent(self):
        with tempfile.TemporaryDirectory() as td:
            heppy, manifest = build_tree(Path(td))
            (heppy / "ecg_qc" / "P_001_snips_rpeaks_corrected.csv").unlink()
            native, qc = autonomic.build_native(manifest)
            missing = qc[qc.participant.eq("P1")]
            self.assertTrue((~missing.autonomic_valid).all())
            self.assertTrue((missing.reason == "no_corrected_rpeaks").all())
            self.assertEqual(set(native.participant), {"P0"})

    def test_all_invalid_raises_with_reasons(self):
        with tempfile.TemporaryDirectory() as td:
            heppy, manifest = build_tree(Path(td), participants=1)
            (heppy / "ecg_qc" / "P_000_snips_rpeaks_corrected.csv").unlink()
            with self.assertRaisesRegex(ValueError, "no_corrected_rpeaks"):
                autonomic.build_native(manifest)

    def test_segment_level_reason_codes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            heppy, manifest = build_tree(root, participants=1)
            fif = manifest.source_raw_fif.iloc[0]
            extra = [
                # fewer than three R-peaks inside the crop (record ends at 2200 s)
                {"participant": "P0", "condition": "preict", "file_path": "no_peaks", "source_raw_fif": fif,
                 "crop_start_seconds": 5000.0, "crop_end_seconds": 5100.0},
                # peaks exist, but the crop holds fewer than 30 complete 15-s estimates
                {"participant": "P0", "condition": "preict", "file_path": "too_short", "source_raw_fif": fif,
                 "crop_start_seconds": 100.0, "crop_end_seconds": 130.0},
                # a condition outside the autonomic analysis is not scored at all
                {"participant": "P0", "condition": "ictal", "file_path": "ictal", "source_raw_fif": fif,
                 "crop_start_seconds": 100.0, "crop_end_seconds": 600.0},
            ]
            frame = pd.concat([manifest, pd.DataFrame(extra)], ignore_index=True)
            native, qc = autonomic.build_native(frame)
            by_file = qc.set_index("file_path")
            self.assertEqual(by_file.loc["no_peaks", "reason"], "fewer than three R peaks")
            self.assertEqual(by_file.loc["too_short", "reason"], "fewer than 30 complete 15-second estimates inside segment")
            self.assertNotIn("ictal", set(qc.file_path))
            self.assertEqual(int(qc.autonomic_valid.sum()), 2)
            self.assertEqual(set(native.file_path), set(manifest.file_path))

    def test_unusable_record_reports_the_estimator_reason(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            heppy, manifest = build_tree(root, participants=2)
            slow = np.arange(0.0, 1500.0, 3.0)                  # 3-s intervals are not physiological
            write_sidecar(heppy / "ecg_qc" / "P_001_snips_rpeaks_corrected.csv", slow, np.zeros(len(slow)))
            _, qc = autonomic.build_native(manifest)
            self.assertTrue((qc[qc.participant.eq("P1")].reason == "fewer than 20 physiological IBI pairs").all())
            self.assertTrue(qc[qc.participant.eq("P0")].autonomic_valid.all())

    def test_native_estimates_are_cropped_to_the_segment_and_re_zeroed(self):
        with tempfile.TemporaryDirectory() as td:
            _, manifest = build_tree(Path(td), participants=1)
            native, _ = autonomic.build_native(manifest)
            for (_, condition), part in native.groupby(["participant", "condition"]):
                crop = manifest[manifest.condition.str.lower().eq(condition)].iloc[0]
                duration = crop.crop_end_seconds - crop.crop_start_seconds
                self.assertTrue((part.window_start_s >= 0).all())
                self.assertTrue((part.window_end_s <= duration + 1e-9).all())
                self.assertTrue((part.record_duration_s == duration).all())
            self.assertEqual(set(native.condition), {"ii", "preict"})

    def test_run_writes_table_qc_and_audit_atomically(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _, manifest = build_tree(root)
            manifest_csv = root / "manifest.csv"
            manifest.to_csv(manifest_csv, index=False)
            audit = autonomic.run(manifest_csv, root / "out")
            self.assertEqual(audit["records_excluded"], 0)
            self.assertEqual(audit["records_valid"], 4)
            self.assertEqual(audit["participants_with_ii_baseline"], 2)
            self.assertEqual(audit["schema"], autonomic.NATIVE_SCHEMA)
            self.assertEqual(len(audit["rpeak_sidecar_sha256"]), 2)
            self.assertTrue((root / "out" / "autonomic_native_ii_standardised.csv.gz").exists())
            self.assertTrue((root / "out" / "autonomic_audit.json").exists())
            self.assertFalse(list((root / "out").glob("*.part")))
            table = pd.read_csv(root / "out" / "autonomic_native_ii_standardised.csv.gz")
            self.assertTrue({"file_path", "event_time_s", "csi_z", "cpi_z"} <= set(table.columns))
            autonomic.run(manifest_csv, root / "out")                           # re-run reproduces the table
            pd.testing.assert_frame_equal(pd.read_csv(root / "out" / "autonomic_native_ii_standardised.csv.gz"), table)

    def test_run_rejects_a_manifest_without_required_columns(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pd.DataFrame({"participant": ["P0"]}).to_csv(root / "m.csv", index=False)
            with self.assertRaisesRegex(ValueError, "required columns"):
                autonomic.run(root / "m.csv", root / "out")


if __name__ == "__main__":
    unittest.main()
