"""Calibration of the statistics under the global null: no group effect anywhere.

Each synthetic dataset has random participant, seizure, slope and window effects but no FS/ES difference in
any outcome.  The proportion of p < 0.05 for each headline test must lie inside the 99.9% two-sided binomial band
for ``N_DATASETS`` Bernoulli(0.05) trials (A4 reports the likelihood-ratio p-value and, for M_L, the Wald p-value;
in a separate run of 800 datasets with about 24 participants each, the likelihood-ratio rate was about 7% and the
Wald rate about 3%, a small-sample departure this band is too wide to resolve).  For the A3 mixed model the Wald z-test of the interaction is checked.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from synthetic_release import SyntheticSpec, build_tables  # noqa: E402

from fsms.stats.change_test import participant_changes  # noqa: E402
from fsms.stats.classification import fit_model_set, predictor_table  # noqa: E402
from fsms.stats.inference import welch_test  # noqa: E402
from fsms.stats.mixed_anova import mixed_anova, participant_values  # noqa: E402
from fsms.stats.release import ReleaseData, prepare_representation  # noqa: E402
from fsms.stats.trajectory_model import fit_trajectory_model, trajectory_table  # noqa: E402

N_DATASETS = 300
ALPHA = 0.05
BASE = SyntheticSpec(n_fs=14, n_es=11, n_dd=2, n_ineligible_participants=0, n_interictal_windows=10)


def collect_null_p_values() -> dict[str, np.ndarray]:
    names = ("A1 interaction (JS)", "A1 interaction (CPI)", "A2 Welch (JS)", "A3 t:group Wald (JS)",
             "A4 M_JS", "A4 M_a", "A4 M_L", "A4 M_L Wald")
    p = {name: [] for name in names}
    for seed in range(N_DATASETS):
        data = ReleaseData.from_frames(*build_tables(BASE.null(seed)))
        rep = prepare_representation(data, "matrix")
        p["A1 interaction (JS)"].append(
            mixed_anova(participant_values(rep, "JS")).set_index("effect").loc["interaction", "p"])
        p["A1 interaction (CPI)"].append(
            mixed_anova(participant_values(rep, "CPI")).set_index("effect").loc["interaction", "p"])
        changes = participant_changes(rep)
        js = changes[changes["outcome"] == "JS"]
        p["A2 Welch (JS)"].append(welch_test(js.loc[js["group"] == "FS", "change"], js.loc[js["group"] == "ES", "change"])["p"])
        fit = fit_trajectory_model(trajectory_table(rep, "JS"), with_lrt=False)
        p["A3 t:group Wald (JS)"].append(fit.fixed.set_index("term").loc["t:group_FS", "p"])
        predictors = predictor_table(changes)
        raw = predictors[["JS_change", "a"]].to_numpy(float)
        y = (predictors["group"] == "FS").to_numpy(float)
        fitted = fit_model_set(raw, y, n_boot=0)
        models = fitted.models.set_index("model")
        for model in ("M_JS", "M_a", "M_L"):
            p[f"A4 {model}"].append(models.loc[model, "p_lr"])
        p["A4 M_L Wald"].append(fitted.coefficients.set_index("model").loc["M_L", "p_wald"])
    return {name: np.array(values, float) for name, values in p.items()}


class NullCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.p = collect_null_p_values()

    def test_false_positive_rates_within_binomial_band(self) -> None:
        for name, values in self.p.items():
            trials = N_DATASETS
            low = stats.binom.ppf(0.0005, trials, ALPHA) / trials
            high = stats.binom.ppf(0.9995, trials, ALPHA) / trials
            with self.subTest(test=name):
                self.assertTrue(np.all(np.isfinite(values)), f"{name}: non-finite p-values")
                rate = float(np.mean(values < ALPHA))
                self.assertGreaterEqual(rate, low, f"{name}: rate {rate:.3f} below band [{low:.3f}, {high:.3f}]")
                self.assertLessEqual(rate, high, f"{name}: rate {rate:.3f} above band [{low:.3f}, {high:.3f}]")

    def test_p_values_close_to_uniform(self) -> None:
        for name, values in self.p.items():
            with self.subTest(test=name):
                self.assertGreater(stats.kstest(values, "uniform").pvalue, 0.001, name)


if __name__ == "__main__":
    unittest.main()
