"""Correctness tests of the statistics package on synthetic release bundles (no real data)."""
from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
import warnings
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))

from synthetic_release import SyntheticSpec, build_tables, write_release  # noqa: E402

from fsms.stats import StatisticsGateError, run_statistics  # noqa: E402
from fsms.stats import logistic, trajectory_model  # noqa: E402
from fsms.stats.__main__ import main as cli_main  # noqa: E402
from fsms.stats.change_test import participant_changes, run_change_test  # noqa: E402
from fsms.stats.classification import fit_model_set, run_classification  # noqa: E402
from fsms.stats.inference import hedges_j, holm, one_sample_test, welch_test  # noqa: E402
from fsms.stats.mixed_anova import mixed_anova, run_mixed_anova  # noqa: E402
from fsms.stats.release import ReleaseData, prepare_representation  # noqa: E402
from fsms.stats.trajectory_model import (fit_mixedlm, fit_trajectory_model, is_singular, run_trajectory_model,  # noqa: E402
                                         trajectory_table)

STRONG = SyntheticSpec(n_fs=26, n_es=20, n_dd=6, js_shift=0.5, kappa_shift=-0.5, a_shift=0.8, js_slope=0.06,
                       kappa_slope=-0.05, seed=11)


def prepared(spec: SyntheticSpec, representation: str = "matrix"):
    data = ReleaseData.from_frames(*build_tables(spec))
    return data, prepare_representation(data, representation)


class HolmTests(unittest.TestCase):
    def test_hand_example(self) -> None:
        np.testing.assert_allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])

    def test_matches_statsmodels_and_is_monotone(self) -> None:
        rng = np.random.default_rng(3)
        for size in (1, 2, 4, 16):
            for _ in range(50):
                p = rng.uniform(0, 1, size) ** rng.integers(1, 4)
                ours = holm(p)
                np.testing.assert_allclose(ours, multipletests(p, method="holm")[1], atol=1e-12)
                self.assertTrue(np.all(np.diff(ours[np.argsort(p)]) >= -1e-15))
                self.assertTrue(np.all(ours >= p - 1e-15) and np.all(ours <= 1))

    def test_nan_is_ignored_in_family_size(self) -> None:
        adjusted = holm([0.01, np.nan, 0.03])
        np.testing.assert_allclose(adjusted[[0, 2]], [0.02, 0.03])
        self.assertTrue(np.isnan(adjusted[1]))


class InferenceTests(unittest.TestCase):
    def test_welch_and_one_sample_match_scipy(self) -> None:
        rng = np.random.default_rng(1)
        x, y = rng.normal(0.4, 1.0, 17), rng.normal(0.0, 2.0, 9)
        ours, reference = welch_test(x, y), stats.ttest_ind(x, y, equal_var=False)
        self.assertAlmostEqual(ours["t"], reference.statistic)
        self.assertAlmostEqual(ours["p"], reference.pvalue)
        self.assertAlmostEqual(ours["df"], reference.df)
        interval = reference.confidence_interval()
        self.assertAlmostEqual(ours["ci_low"], interval.low)
        self.assertAlmostEqual(ours["ci_high"], interval.high)
        one, reference = one_sample_test(x), stats.ttest_1samp(x, 0.0)
        self.assertAlmostEqual(one["t"], reference.statistic)
        self.assertAlmostEqual(one["p"], reference.pvalue)

    def test_hedges_correction(self) -> None:
        for df in (5, 10, 30):
            self.assertAlmostEqual(hedges_j(df), 1 - 3 / (4 * df - 1), places=2)
        self.assertTrue(np.isnan(hedges_j(1)))

    def test_degenerate_inputs_give_nan(self) -> None:
        self.assertTrue(np.isnan(welch_test([1.0], [1.0, 2.0])["p"]))
        self.assertTrue(np.isnan(one_sample_test([1.0, 1.0, 1.0])["p"]))


class LogisticEngineTests(unittest.TestCase):
    def test_fit_matches_statsmodels(self) -> None:
        rng = np.random.default_rng(5)
        x = rng.normal(size=(60, 2))
        y = (rng.random(60) < 1 / (1 + np.exp(-(0.3 + 1.2 * x[:, 0] - 0.7 * x[:, 1])))).astype(float)
        design = np.column_stack([np.ones(60), x])
        ours, reference = logistic.fit_logit(design, y), sm.Logit(y, design).fit(disp=0)
        np.testing.assert_allclose(ours.beta, reference.params, atol=1e-6)
        np.testing.assert_allclose(ours.se, reference.bse, atol=1e-6)
        self.assertAlmostEqual(ours.log_likelihood, reference.llf, places=8)
        self.assertAlmostEqual(logistic.mcfadden_r2(ours.log_likelihood, logistic.null_log_likelihood(y)),
                               reference.prsquared, places=8)
        self.assertTrue(ours.converged and not ours.separated)

    def test_separation_is_flagged(self) -> None:
        x = np.arange(10.0)
        fit = logistic.fit_logit(np.column_stack([np.ones(10), x]), (x >= 5).astype(float))
        self.assertTrue(fit.separated)

    def test_auc_equals_pairwise_definition_and_mann_whitney(self) -> None:
        rng = np.random.default_rng(2)
        score = np.round(rng.normal(size=40), 1)  # ties on purpose
        y = (rng.random(40) < 0.4).astype(float)
        pos, neg = score[y == 1], score[y == 0]
        brute = np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg])
        self.assertAlmostEqual(logistic.auc(score, y), brute)
        u = stats.mannwhitneyu(pos, neg).statistic
        self.assertAlmostEqual(logistic.auc(score, y), u / (len(pos) * len(neg)))
        fpr, tpr, _ = logistic.roc_curve(score, y)
        self.assertAlmostEqual(np.trapezoid(tpr, fpr), brute)

    def test_bootstrap_and_loo_are_reproducible(self) -> None:
        rng = np.random.default_rng(8)
        raw = rng.normal(size=(30, 2))
        y = (rng.random(30) < 1 / (1 + np.exp(-raw[:, 1]))).astype(float)
        first, second = fit_model_set(raw, y, n_boot=100, seed=7), fit_model_set(raw, y, n_boot=100, seed=7)
        pd.testing.assert_frame_equal(first.models, second.models)
        row = first.models.set_index("model").loc["M_L"]
        self.assertAlmostEqual(row["auc_corrected"], row["auc_apparent"] - row["auc_optimism"])
        self.assertTrue(0 <= row["auc_loo"] <= 1)

    def test_index_uses_equal_weights_on_standardised_predictors(self) -> None:
        rng = np.random.default_rng(4)
        raw = rng.normal(2.0, 3.0, size=(25, 2))
        y = (rng.random(25) < 0.5).astype(float)
        y[:2] = [0, 1]
        coefficients = fit_model_set(raw, y, n_boot=0).coefficients.set_index(["model", "term"])
        z = (raw - raw.mean(0)) / raw.std(0, ddof=1)
        direct = logistic.fit_logit(np.column_stack([np.ones(25), z.sum(axis=1)]), y)
        self.assertAlmostEqual(coefficients.loc[("M_L", "L"), "estimate"], direct.beta[1], places=6)
        self.assertAlmostEqual(coefficients.loc[("M_L", "L"), "sd_predictor"], z.sum(axis=1).std(ddof=1))


class DerivationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = SyntheticSpec(n_fs=10, n_es=8, n_dd=3, seed=5)
        cls.windows, cls.events, cls.participants = build_tables(cls.spec)
        cls.data = ReleaseData.from_frames(cls.windows, cls.events, cls.participants)

    def test_change_matches_independent_computation(self) -> None:
        for name, column in (("matrix", "js_excess_ordinary"), ("tensor", "js_excess_tensor")):
            expected = {}
            eligible_participants = set(self.participants.loc[self.participants[f"eligible_{name}"], "participant_id"])
            for _, event in self.events[self.events[f"eligible_{name}"]].iterrows():
                if event["participant_id"] not in eligible_participants or event["group"] not in ("FS", "ES"):
                    continue
                rows = self.windows[self.windows["recording_id"] == event["recording_id"]]
                base = rows[(rows["window_end_min"] >= 10) & (rows["window_end_min"] <= 19)][column].dropna()
                final = rows[rows["window_end_min"] == 0][column].dropna()
                if len(base) >= 5 and len(final) == 1:
                    expected.setdefault(event["participant_id"], []).append(final.iloc[0] - base.mean())
            changes = participant_changes(prepare_representation(self.data, name))
            changes = changes[changes["outcome"] == "JS"].set_index("participant_id")["change"]
            self.assertEqual(set(changes.index), set(expected))
            for participant, values in expected.items():
                self.assertAlmostEqual(changes[participant], np.mean(values), places=10)

    def test_only_representation_eligible_data_are_used_and_dd_is_excluded_from_group_tests(self) -> None:
        matrix, tensor = (prepare_representation(self.data, n) for n in ("matrix", "tensor"))
        self.assertNotEqual(set(matrix.events["recording_id"]), set(tensor.events["recording_id"]))
        for name, rep in (("matrix", matrix), ("tensor", tensor)):
            self.assertTrue(rep.events["recording_id"].isin(
                self.events.loc[self.events[f"eligible_{name}"], "recording_id"]).all())
            self.assertTrue(rep.participants["participant_id"].isin(
                self.participants.loc[self.participants[f"eligible_{name}"], "participant_id"]).all())
            self.assertLess(rep.events["final"].abs().max(), 10)  # extreme values only occur in ineligible seizures
            self.assertLess(rep.interictal["mean"].abs().max(), 10)
            self.assertTrue((rep.events["n_baseline"].where(rep.events["baseline_mean"].notna(), 5) >= 5).all())
            values = run_mixed_anova(rep).values
            self.assertEqual(set(values["group"]), {"FS", "ES"})
            self.assertEqual(set(participant_changes(rep)["group"]), {"FS", "ES"})
            self.assertEqual(set(trajectory_table(rep, "JS")["group"]), {"FS", "ES"})

    def test_baseline_requires_five_valid_windows(self) -> None:
        windows = self.windows.copy()
        recording = self.events.loc[self.events["eligible_matrix"], "recording_id"].iloc[0]
        rows = (windows["recording_id"] == recording) & (windows["window_end_min"] >= 10)
        keep = windows.index[rows][:4]
        windows.loc[windows.index[rows].difference(keep), "js_excess_ordinary"] = np.nan
        rep = prepare_representation(ReleaseData.from_frames(windows, self.events, self.participants), "matrix")
        row = rep.events[(rep.events["recording_id"] == recording) & (rep.events["outcome"] == "JS")].iloc[0]
        self.assertTrue(np.isnan(row["baseline_mean"]) and np.isnan(row["change"]))


class MixedAnovaTests(unittest.TestCase):
    @staticmethod
    def values(n_fs: int, n_es: int, seed: int) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        group = np.array(["FS"] * n_fs + ["ES"] * n_es)
        subject = rng.normal(0, 1.0, len(group))
        interictal = subject + rng.normal(0, 0.5, len(group))
        final = subject + 0.3 + 0.5 * (group == "FS") + rng.normal(0, 0.5, len(group))
        return pd.DataFrame({"participant_id": [f"p{i}" for i in range(len(group))], "group": group,
                             "interictal": interictal, "final": final})

    def test_matches_ols_type3_on_unbalanced_data(self) -> None:
        for n_fs, n_es in ((13, 8), (6, 15), (10, 10)):
            values = self.values(n_fs, n_es, seed=n_fs)
            ours = mixed_anova(values).set_index("effect")
            frame = values.assign(mean_score=(values["interictal"] + values["final"]) / 2,
                                  difference=values["final"] - values["interictal"])
            group = anova_lm(smf.ols("mean_score ~ C(group, Sum)", frame).fit(), typ=3)
            difference = anova_lm(smf.ols("difference ~ C(group, Sum)", frame).fit(), typ=3)
            expected = {"group": group.loc["C(group, Sum)"], "interaction": difference.loc["C(group, Sum)"],
                        "condition": difference.loc["Intercept"]}
            for effect, row in expected.items():
                self.assertAlmostEqual(ours.loc[effect, "f"], row["F"], places=8, msg=effect)
                self.assertAlmostEqual(ours.loc[effect, "p"], row["PR(>F)"], places=10, msg=effect)
                self.assertEqual(ours.loc[effect, "df2"], n_fs + n_es - 2)
            eta = ours["f"] / (ours["f"] + ours["df2"])
            np.testing.assert_allclose(ours["partial_eta_sq"], eta)

    def test_matches_random_intercept_mixed_model_on_long_data(self) -> None:
        values = self.values(13, 8, seed=2)
        long = values.melt(id_vars=["participant_id", "group"], value_vars=["interictal", "final"],
                           var_name="condition", value_name="y")
        long["grp"] = np.where(long["group"] == "FS", 1.0, -1.0)
        long["cnd"] = np.where(long["condition"] == "final", 0.5, -0.5)
        fit = MixedLM.from_formula("y ~ cnd * grp", long, groups="participant_id", re_formula="1").fit(reml=True)
        ours = mixed_anova(values).set_index("effect")
        for name, effect in (("cnd", "condition"), ("cnd:grp", "interaction"), ("grp", "group")):
            self.assertAlmostEqual(fit.tvalues[name] ** 2, ours.loc[effect, "f"], delta=1e-3 * ours.loc[effect, "f"])

    def test_estimates_are_group_differences_in_unweighted_marginal_means(self) -> None:
        values = self.values(13, 8, seed=9)
        ours = mixed_anova(values).set_index("effect")
        means = values.groupby("group")[["interictal", "final"]].mean()
        change = means["final"] - means["interictal"]
        self.assertAlmostEqual(ours.loc["interaction", "estimate"], change["FS"] - change["ES"])
        self.assertAlmostEqual(ours.loc["condition", "estimate"], change.mean())
        self.assertAlmostEqual(ours.loc["group", "estimate"], means.loc["FS"].mean() - means.loc["ES"].mean())

    def test_holm_families_of_the_tables(self) -> None:
        _, rep = prepared(STRONG)
        result = run_mixed_anova(rep)
        for effect, frame in result.omnibus.groupby("effect"):
            np.testing.assert_allclose(frame["p_holm"], multipletests(frame["p"], method="holm")[1])
        for _, frame in result.posthoc.groupby("effect"):
            np.testing.assert_allclose(frame["p_holm_outcomes"], multipletests(frame["p"], method="holm")[1])
        self.assertEqual(len(result.posthoc), 16)
        np.testing.assert_allclose(result.posthoc["p_holm_all"], multipletests(result.posthoc["p"], method="holm")[1])
        self.assertTrue((result.posthoc["p_holm_all"] >= result.posthoc["p_holm_outcomes"] - 1e-12).all())

    def test_posthoc_tests_match_scipy(self) -> None:
        _, rep = prepared(STRONG)
        result = run_mixed_anova(rep)
        values = result.values[result.values["outcome"] == "CSI"]
        fs, es = values[values["group"] == "FS"], values[values["group"] == "ES"]
        row = result.posthoc[(result.posthoc["outcome"] == "CSI")].set_index("effect")
        self.assertAlmostEqual(row.loc["group_at_final_minute", "p"], stats.ttest_ind(fs["final"], es["final"], equal_var=False).pvalue)
        self.assertAlmostEqual(row.loc["condition_in_FS", "p"], stats.ttest_rel(fs["final"], fs["interictal"]).pvalue)
        self.assertAlmostEqual(row.loc["condition_in_ES", "t"], stats.ttest_rel(es["final"], es["interictal"]).statistic)


class RecoveryTests(unittest.TestCase):
    """Planted effects are recovered with the right sign and approximate size."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.data, cls.rep = prepared(STRONG)
        cls.a1, cls.a2 = run_mixed_anova(cls.rep), run_change_test(cls.rep)
        cls.a3, cls.a4 = run_trajectory_model(cls.rep), run_classification(cls.rep, n_boot=100)

    def test_a1_interaction(self) -> None:
        row = self.a1.omnibus[self.a1.omnibus["effect"] == "interaction"].set_index("outcome")
        for outcome, sign in (("JS", 1), ("kappa", -1), ("CSI", 1), ("CPI", -1)):
            self.assertEqual(np.sign(row.loc[outcome, "estimate"]), sign, outcome)
            self.assertLess(row.loc[outcome, "p_holm"], 0.05, outcome)

    def test_a2_change(self) -> None:
        row = self.a2.between.set_index("outcome")
        for outcome, sign in (("JS", 1), ("kappa", -1), ("CSI", 1), ("CPI", -1)):
            self.assertEqual(np.sign(row.loc[outcome, "estimate"]), sign, outcome)
            self.assertLess(row.loc[outcome, "p_holm"], 0.05, outcome)
            self.assertEqual(np.sign(row.loc[outcome, "hedges_g"]), sign)
        self.assertAlmostEqual(row.loc["JS", "estimate"], 0.5, delta=0.2)

    def test_a3_gradient(self) -> None:
        fixed = self.a3.fixed[self.a3.fixed["term"] == "t:group_FS"].set_index("outcome")
        self.assertGreater(fixed.loc["JS", "estimate"], 0)
        self.assertLess(fixed.loc["kappa", "estimate"], 0)
        self.assertAlmostEqual(fixed.loc["JS", "estimate"], 0.06, delta=0.03)
        self.assertAlmostEqual(fixed.loc["kappa", "estimate"], -0.05, delta=0.03)
        self.assertTrue((fixed["p_holm_outcomes"] < 0.05).all())
        self.assertTrue((self.a3.models["lrt_p"] < 0.05).all())
        simple = self.a3.simple[self.a3.simple["outcome"] == "JS"].set_index("quantity")
        self.assertGreater(simple.loc["slope_FS", "estimate"], simple.loc["slope_ES", "estimate"])

    def test_a4_signal_in_a(self) -> None:
        models = self.a4.primary.models.set_index("model")
        coefficients = self.a4.primary.coefficients.set_index(["model", "term"])
        self.assertGreater(coefficients.loc[("M_a", "z_a"), "or_per_sd"], 1)
        self.assertGreater(coefficients.loc[("M_L", "L"), "or_per_sd"], 1)
        self.assertLess(models.loc["M_a", "p_holm_lr"], 0.05)
        self.assertGreater(models.loc["M_a", "auc_apparent"], 0.7)
        self.assertGreater(models.loc["M_a", "auc_corrected"], 0.6)
        self.assertGreater(models.loc["M_a", "auc_loo"], 0.6)
        self.assertGreater(self.a4.event_coefficients.set_index("model").loc["M_a", "or_per_sd"], 1)
        np.testing.assert_allclose(models.loc[["M_JS", "M_a", "M_L"], "p_holm_lr"],
                                   multipletests(models.loc[["M_JS", "M_a", "M_L"], "p_lr"], method="holm")[1])
        self.assertTrue(np.isnan(models.loc["M_JS+a", "p_holm_lr"]))
        pairs = self.a4.dd_pairs.set_index("measure")
        self.assertGreater(pairs.loc["a", "n_pairs"], 1)
        self.assertGreater(pairs.loc["a", "estimate_fs_minus_es"], 0)

    def test_index_definition_is_stored(self) -> None:
        definition = self.a4.l_definition["standardisation"]
        predictors = self.a4.predictors
        self.assertAlmostEqual(definition["JS_change"]["mean"], predictors["JS_change"].mean())
        self.assertAlmostEqual(definition["a"]["sd"], predictors["a"].std(ddof=1))
        np.testing.assert_allclose(predictors["L"], predictors["z_JS"] + predictors["z_a"])
        np.testing.assert_allclose(predictors["a"], (predictors["CSI_change"] - predictors["CPI_change"]) / 2)

    def test_tensor_representation_is_analysed_separately(self) -> None:
        rep = prepare_representation(self.data, "tensor")
        row = run_change_test(rep).between.set_index("outcome")
        self.assertGreater(row.loc["JS", "estimate"], 0)
        self.assertNotAlmostEqual(row.loc["JS", "estimate"], self.a2.between.set_index("outcome").loc["JS", "estimate"])
        self.assertGreater(row.loc["CSI", "n_fs"] + row.loc["CSI", "n_es"], 0)


class TrajectoryModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _, cls.rep = prepared(STRONG)
        cls.table = trajectory_table(cls.rep, "JS")

    def test_t_is_minus_window_end_and_spans_final_ten_minutes(self) -> None:
        self.assertEqual(sorted(self.table["t"].unique()), list(range(-9, 1)))

    def test_random_intercept_and_slope_are_fitted(self) -> None:
        fit = fit_mixedlm(self.table, reml=True)
        self.assertEqual(fit.result.k_re, 2)
        self.assertEqual(np.asarray(fit.result.cov_re).shape, (2, 2))
        self.assertTrue(fit.converged)
        summary = fit_trajectory_model(self.table).summary
        self.assertEqual(summary["random_effects_dim"], 2)
        self.assertIn(summary["fit_status"], ("converged", "converged_after_retry", "singular_fit"))

    def test_fixed_effects_are_the_gls_estimates_for_the_fitted_variance_components(self) -> None:
        """Re-derive estimates and standard errors by generalised least squares from the fitted G and s^2."""
        fit = fit_mixedlm(self.table, reml=True)
        cov_re, scale = np.asarray(fit.result.cov_re), float(fit.result.scale)
        table = self.table
        xtvx, xtvy = np.zeros((4, 4)), np.zeros(4)
        for _, block in table.groupby("participant_id"):
            t = block["t"].to_numpy()
            fs = float(block["group"].iloc[0] == "FS")
            x = np.column_stack([np.ones_like(t), t, np.full_like(t, fs), t * fs])
            z = x[:, :2]
            inverse = np.linalg.inv(z @ cov_re @ z.T + scale * np.eye(len(t)))
            xtvx += x.T @ inverse @ x
            xtvy += x.T @ inverse @ block["change"].to_numpy()
        beta, se = np.linalg.solve(xtvx, xtvy), np.sqrt(np.diag(np.linalg.inv(xtvx)))
        reported = fit_trajectory_model(self.table).fixed
        np.testing.assert_allclose(reported["estimate"], beta, rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(reported["se"], se, rtol=1e-4, atol=1e-6)
        reference = np.asarray(fit.result.fe_params)
        np.testing.assert_allclose(reference, beta, rtol=1e-4, atol=1e-6)

    def test_fit_is_at_least_as_good_as_the_default_statsmodels_fit(self) -> None:
        table = self.table.assign(fs=(self.table["group"] == "FS").astype(float))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reference = smf.mixedlm("change ~ t * fs", table, groups="participant_id", re_formula="~t").fit(reml=True)
        self.assertGreaterEqual(fit_mixedlm(self.table, reml=True).result.llf, reference.llf - 1e-6)

    def test_interaction_lrt_matches_manual_ml_comparison(self) -> None:
        summary = fit_trajectory_model(self.table).summary
        full = fit_mixedlm(self.table, reml=False)
        reduced = fit_mixedlm(self.table, reml=False, interaction=False)
        chi2 = 2 * (full.result.llf - reduced.result.llf)
        self.assertAlmostEqual(summary["lrt_chi2"], chi2, places=6)
        self.assertAlmostEqual(summary["lrt_p"], stats.chi2.sf(chi2, 1), places=8)

    def test_singularity_detection(self) -> None:
        self.assertFalse(is_singular(np.array([[0.2, 0.01], [0.01, 0.05]]), 0.1))
        self.assertTrue(is_singular(np.zeros((2, 2)), 0.1))
        self.assertTrue(is_singular(np.array([[0.2, 0.0], [0.0, 0.0]]), 0.1))
        self.assertTrue(is_singular(np.array([[1.0, 1.0], [1.0, 1.0]]), 0.1))
        self.assertTrue(is_singular(np.array([[1.0, 2.0], [2.0, 1.0]]), 0.1))
        self.assertTrue(is_singular(np.array([[0.2, 0.01], [0.01, 1e-9]]), 0.1))
        self.assertTrue(is_singular(np.array([[np.nan, 0.0], [0.0, 1.0]]), 0.1))

    def test_singular_fit_is_reported_not_hidden(self) -> None:
        rng = np.random.default_rng(0)
        n = 24
        table = pd.DataFrame({
            "participant_id": np.repeat([f"p{i}" for i in range(n)], 10),
            "group": np.repeat(["FS"] * 12 + ["ES"] * 12, 10), "t": np.tile(np.arange(-9.0, 1.0), n)})
        table["change"] = 0.1 + 0.02 * table["t"] + rng.normal(0, 0.3, len(table))  # no participant-level variation
        summary = fit_trajectory_model(table, with_lrt=False).summary
        self.assertEqual(summary["fit_status"], "singular_fit")
        self.assertTrue(summary["singular"])
        self.assertEqual(summary["random_effects_dim"], 2)

    def _patched(self, failing: tuple[str, ...]):
        class Failing(MixedLM):
            def fit(self, *args, **kwargs):
                result = super().fit(*args, **kwargs)
                if kwargs.get("method") in failing:
                    result.converged = False
                return result
        return mock.patch.object(trajectory_model, "MixedLM", Failing)

    def test_retry_sequence_and_status(self) -> None:
        with self._patched(("lbfgs",)):
            fit = fit_trajectory_model(self.table, with_lrt=False)
        self.assertIn(fit.summary["fit_status"], ("converged_after_retry", "singular_fit"))
        self.assertEqual(fit.summary["optimiser"], "powell")
        self.assertTrue(fit.summary["attempts"].startswith("lbfgs:not_converged;powell:converged"))
        self.assertEqual(fit.summary["random_effects_dim"], 2)

    def test_failure_is_reported_and_random_slope_never_dropped(self) -> None:
        with self._patched(trajectory_model.OPTIMISERS):
            fit = fit_trajectory_model(self.table, with_lrt=False)
        self.assertEqual(fit.summary["fit_status"], "not_converged")
        self.assertFalse(fit.summary["converged"])
        self.assertEqual(fit.summary["random_effects_dim"], 2)
        self.assertEqual(fit.summary["attempts"].count("not_converged"), len(trajectory_model.OPTIMISERS))

    def test_insufficient_data(self) -> None:
        fit = fit_trajectory_model(self.table[self.table["group"] == "FS"])
        self.assertEqual(fit.summary["fit_status"], "insufficient_data")
        self.assertTrue(fit.fixed["estimate"].isna().all())

    def test_holm_over_the_two_outcomes(self) -> None:
        result = run_trajectory_model(self.rep)
        for term in ("t:group_FS", "group_FS"):
            frame = result.fixed[result.fixed["term"] == term]
            np.testing.assert_allclose(frame["p_holm_outcomes"], multipletests(frame["p"], method="holm")[1])
        self.assertTrue(result.fixed.loc[~result.fixed["term"].isin(["t:group_FS", "group_FS"]), "p_holm_outcomes"].isna().all())


class GateAndPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.spec = SyntheticSpec(n_fs=9, n_es=8, n_dd=3, js_shift=0.4, a_shift=0.6, seed=21)
        cls.release = write_release(cls.root / "release", cls.spec, "PREREG-A")
        cls.results = run_statistics(cls.release, cls.root / "out", "PREREG-A", n_bootstrap=40)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def test_missing_checkpoint_raises(self) -> None:
        release = write_release(self.root / "no_checkpoint", self.spec, "PREREG-A")
        (release / "inclusion_checkpoint.json").unlink()
        with self.assertRaises(StatisticsGateError):
            run_statistics(release, self.root / "out_none", "PREREG-A", n_bootstrap=5)
        self.assertFalse((self.root / "out_none").exists())

    def test_mismatched_prereg_id_raises(self) -> None:
        with self.assertRaisesRegex(StatisticsGateError, "PREREG-B"):
            run_statistics(self.release, self.root / "out_mismatch", "PREREG-B", n_bootstrap=5)
        self.assertFalse((self.root / "out_mismatch").exists())

    def test_checkpoint_without_id_raises(self) -> None:
        release = write_release(self.root / "blank", self.spec, "PREREG-A")
        (release / "inclusion_checkpoint.json").write_text("{}", encoding="utf-8")
        with self.assertRaises(StatisticsGateError):
            run_statistics(release, self.root / "out_blank", "PREREG-A", n_bootstrap=5)

    def test_unknown_representation_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            run_statistics(self.release, self.root / "out_bad", "PREREG-A", representations=("cube",))

    def test_cli(self) -> None:
        with redirect_stdout(io.StringIO()):
            self.assertEqual(cli_main(["--release", str(self.release), "--out", str(self.root / "cli"), "--prereg-id",
                                       "WRONG"]), 2)
            self.assertEqual(cli_main(["--release", str(self.release), "--out", str(self.root / "cli"), "--prereg-id",
                                       "PREREG-A", "--representations", "matrix", "--n-bootstrap", "10"]), 0)
        self.assertTrue((self.root / "cli" / "matrix" / "tables" / "a1_omnibus.csv").is_file())
        self.assertFalse((self.root / "cli" / "tensor").exists())

    def test_every_output_is_stamped_with_the_prereg_id(self) -> None:
        out = self.root / "out"
        self.assertEqual(self.results["prereg_id"], "PREREG-A")
        self.assertEqual(json.loads((out / "results.json").read_text(encoding="utf-8"))["prereg_id"], "PREREG-A")
        self.assertIn("PREREG-A", (out / "results_summary.md").read_text(encoding="utf-8"))
        for representation in ("matrix", "tensor"):
            tables = sorted((out / representation / "tables").glob("*.csv"))
            self.assertGreaterEqual(len(tables), 20)
            for path in tables:
                frame = pd.read_csv(path)
                self.assertEqual(list(frame.columns[:2]), ["prereg_id", "representation"], path.name)
                self.assertEqual(set(frame["prereg_id"]), {"PREREG-A"}, path.name)
                self.assertEqual(set(frame["representation"]), {representation}, path.name)
            definition = json.loads((out / representation / "L_definition.json").read_text(encoding="utf-8"))
            self.assertEqual(definition["prereg_id"], "PREREG-A")
            self.assertIn("mean", definition["standardisation"]["JS_change"])

    def test_figures_are_300_dpi_pngs(self) -> None:
        from PIL import Image
        for representation in ("matrix", "tensor"):
            names = sorted(p.name for p in (self.root / "out" / representation / "figures").glob("*.png"))
            self.assertEqual(names, ["a1_interaction.png", "a2_change_distributions.png", "a3_trajectories.png",
                                     "a4_roc_curves.png"])
            for name in names:
                with Image.open(self.root / "out" / representation / "figures" / name) as image:
                    self.assertAlmostEqual(image.info["dpi"][0], 300, delta=0.5)
                    self.assertIn("PREREG-A", image.info["Description"])

    def test_results_json_content(self) -> None:
        for representation in ("matrix", "tensor"):
            block = self.results["representations"][representation]
            self.assertEqual(len(block["holm_families"]), 10)
            self.assertEqual(len(block["A1"]["posthoc"]), 16)
            self.assertEqual({r["outcome"] for r in block["A3"]["models"]}, {"JS", "kappa"})
            self.assertEqual(block["n"]["A4_primary"]["participants"],
                             block["n"]["A4_primary"]["FS"] + block["n"]["A4_primary"]["ES"])
            self.assertIn("L_definition", block["A4"])
        matrix, tensor = (self.results["representations"][n]["n"]["eligible_events"] for n in ("matrix", "tensor"))
        events = pd.read_csv(self.release / "events.csv")
        self.assertEqual(matrix["FS"] + matrix["ES"] + matrix["DD"], int(events["eligible_matrix"].sum()))
        self.assertEqual(tensor["FS"] + tensor["ES"] + tensor["DD"], int(events["eligible_tensor"].sum()))
        text = json.dumps(self.results)
        self.assertNotIn("NaN", text)

    def test_rerun_is_deterministic(self) -> None:
        run_statistics(self.release, self.root / "again", "PREREG-A", n_bootstrap=40)
        for path in (self.root / "out").rglob("*.csv"):
            self.assertEqual(path.read_bytes(), (self.root / "again" / path.relative_to(self.root / "out")).read_bytes())
        self.assertEqual((self.root / "out" / "results.json").read_bytes(), (self.root / "again" / "results.json").read_bytes())

    def test_no_dd_participant_in_between_group_tables(self) -> None:
        dd = set(pd.read_csv(self.release / "participants.csv").query("group == 'DD'")["participant_id"])
        for representation in ("matrix", "tensor"):
            for name in ("a1_participant_values", "a2_participant_change", "a3_participant_minute_changes",
                         "a4_participant_predictors"):
                frame = pd.read_csv(self.root / "out" / representation / "tables" / f"{name}.csv")
                self.assertFalse(set(frame["participant_id"]) & dd, name)


if __name__ == "__main__":
    unittest.main()
