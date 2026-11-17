"""Transition-grammar measures and window construction on synthetic data."""
import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fsms import metrics as M  # noqa: E402
from fsms import windows as W  # noqa: E402


def random_reference(k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.gamma(2.0, size=(k, k))
    np.fill_diagonal(raw, 0)
    return raw / raw.sum(axis=1, keepdims=True)


def draw_transitions(reference: np.ndarray, n: int, seed: int, kappa: float = 1.0):
    """Ordered transitions from a Markov chain whose rows are softmax(kappa * log reference)."""
    k = reference.shape[0]
    rng = np.random.default_rng(seed)
    logits = np.where(reference > 0, kappa * np.log(np.clip(reference, 1e-12, None)), -np.inf)
    rows = np.exp(logits - logits.max(axis=1, keepdims=True))
    rows /= rows.sum(axis=1, keepdims=True)
    source, destination, state = [], [], int(rng.integers(k))
    for _ in range(n):
        nxt = int(rng.choice(k, p=rows[state]))
        source.append(state)
        destination.append(nxt)
        state = nxt
    return np.array(source), np.array(destination)


class MetricTests(unittest.TestCase):
    def test_row_probabilities_are_proper_for_any_k(self):
        for k in (3, 5, 7):
            counts = np.random.default_rng(k).integers(0, 9, (k, k)).astype(float)
            np.fill_diagonal(counts, 0)
            probability = M.row_probabilities(counts)
            self.assertTrue(np.allclose(probability.sum(axis=1), 1))
            self.assertTrue(np.allclose(np.diag(probability), 0))

    def test_js_zero_for_identical_grammar_and_positive_otherwise(self):
        reference = random_reference(6, 1)
        counts = reference * 4000
        self.assertLess(M.weighted_js(counts, reference), 1e-3)
        other = random_reference(6, 2)
        self.assertGreater(M.weighted_js(other * 4000, reference), 0.01)

    def test_kappa_recovers_precision(self):
        reference = random_reference(6, 3)
        for true_kappa in (0.5, 1.0, 2.0):
            source, destination = draw_transitions(reference, 20000, 5, true_kappa)
            counts = M.transition_counts(source, destination, 6)
            self.assertAlmostEqual(M.fit_kappa(counts, reference), true_kappa, delta=0.12 * true_kappa + 0.05)

    def test_expected_js_removes_small_sample_bias(self):
        reference = random_reference(5, 7)
        draws = 120
        excess = []
        for seed in range(30):
            source, destination = draw_transitions(reference, 60, 100 + seed)
            counts = M.transition_counts(source, destination, 5)
            null = M.expected_js_under_reference(source, np.ones((len(source), 1)), reference[None], f"w{seed}", draws)
            excess.append(M.weighted_js(counts, reference) - null)
        plug_in = np.mean([M.weighted_js(M.transition_counts(*draw_transitions(reference, 60, 100 + s), 5), reference)
                           for s in range(30)])
        self.assertGreater(plug_in, 0.01)              # the plug-in estimator is biased upwards
        self.assertLess(abs(np.mean(excess)), 0.4 * plug_in)

    def test_expected_js_is_deterministic(self):
        reference = random_reference(6, 9)
        source, _ = draw_transitions(reference, 80, 3)
        weights = np.ones((len(source), 1))
        a = M.expected_js_under_reference(source, weights, reference[None], "key", 50)
        b = M.expected_js_under_reference(source, weights, reference[None], "key", 50)
        self.assertEqual(a, b)

    def test_tensor_counts_conserve_mass(self):
        source, destination = draw_transitions(random_reference(4, 2), 300, 1)
        weights = M.gradient_weights(np.sin(np.arange(300) / 7.0), np.cos(np.arange(300) / 11.0))
        tensor = M.tensor_counts(source, destination, weights, 4)
        self.assertAlmostEqual(tensor.sum(), 300, places=6)
        self.assertTrue(np.allclose(tensor.sum(axis=0), M.transition_counts(source, destination, 4)))

    def test_tensor_window_matches_matrix_when_weights_are_uniform(self):
        reference = random_reference(5, 4)
        source, destination = draw_transitions(reference, 400, 2)
        weights = np.full((400, 4), 0.25)
        tensor = M.tensor_counts(source, destination, weights, 5)
        tensor_reference = np.broadcast_to(reference, (4, 5, 5))
        self.assertAlmostEqual(M.weighted_js(tensor, tensor_reference),
                               M.weighted_js(M.transition_counts(source, destination, 5), reference), delta=0.02)


class WindowTests(unittest.TestCase):
    def test_preictal_windows_are_aligned_to_onset(self):
        specs = W.window_specs("preict", 1500.0)
        self.assertEqual(len(specs), 20)
        roles = {end: role for role, end, _ in specs}
        self.assertEqual(roles[0], "minute")
        self.assertEqual(roles[9], "minute")
        self.assertEqual(roles[10], "baseline")
        self.assertEqual(roles[19], "baseline")
        starts = {end: start for _, end, start in specs}
        self.assertAlmostEqual(starts[0], 1440.0)            # final minute: the 60 s before onset
        self.assertAlmostEqual(starts[19], 1500.0 - 20 * 60)

    def test_short_preictal_recording_yields_fewer_baseline_windows(self):
        specs = W.window_specs("preict", 14 * 60 + 5)
        self.assertEqual(max(end for _, end, _ in specs), 13)
        self.assertEqual(W.window_specs("preict", 40.0), [])

    def test_interictal_windows_are_consecutive_blocks(self):
        specs = W.window_specs("ii", 305.0)
        self.assertEqual([s for _, _, s in specs], [0.0, 60.0, 120.0, 180.0, 240.0])
        self.assertTrue(all(role == "interictal" and end is None for role, end, _ in specs))

    def test_canonical_states_are_alphabetical_for_any_labels(self):
        names, mapping = W.canonical_states(["F", "B", "G", "D"])
        self.assertEqual(names, ["B", "D", "F", "G"])
        self.assertEqual(list(mapping), [2, 0, 3, 1])
        with self.assertRaises(ValueError):
            W.canonical_states(["A", "A", "B"])

    def test_cache_validation_rejects_self_transitions_and_bad_states(self):
        good = {"sfreq": 250.0, "n_samples": 1000, "sample": np.array([10, 20, 30]),
                "from": np.array([1, 2, 3], np.int16), "to": np.array([2, 3, 1], np.int16)}
        W.validate_cache(good, 3)
        for broken in ({"to": np.array([1, 3, 1], np.int16)}, {"to": np.array([2, 3, 4], np.int16)},
                       {"sample": np.array([10, 10, 30])}):
            with self.assertRaises(ValueError):
                W.validate_cache({**good, **broken}, 3)

    def test_autonomic_interpolation_never_bridges_gaps(self):
        import pandas as pd
        native = pd.DataFrame({"event_time_s": [0.0, 5.0, 10.0, 40.0, 45.0], "csi_z": [0, 1, 2, 3, 4.0],
                               "cpi_z": [0, -1, -2, -3, -4.0]})
        csi, cpi, csi_g, _ = W.interpolate_autonomic(native, np.array([2.5, 7.5, 20.0, 42.5]), 5.0)
        self.assertAlmostEqual(csi[0], 0.5)
        self.assertTrue(np.isnan(csi[2]))                    # inside the 30-s gap
        self.assertAlmostEqual(csi_g[3], 0.2)


if __name__ == "__main__":
    unittest.main()
