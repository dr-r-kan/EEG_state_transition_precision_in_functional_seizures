"""Transition-grammar measures for K microstates.

A window of ordered genuine microstate changes is summarised by a K x K matrix of counts (source -> destination,
no self transitions) and, when autonomic information is available, by a tensor of shape (4, K, K): each
transition contributes soft weights to four CPI/CSI-gradient quadrants.  Both are compared with the
participant's interictal reference grammar:

``weighted_js``
    Jensen-Shannon divergence between the window's destination distributions and the reference rows, averaged over
    source states with their transition counts as weights.
``expected_js_under_reference`` (E0)
    Plug-in JS is biased upwards by an amount that depends on the transition count and source mix of the window.
    E0 is the mean JS of destinations drawn from the reference given the window's own source sequence and soft
    weights; ``JS - E0`` ("excess JS") is the analysed outcome.
``fit_kappa``
    Precision (inverse temperature) of the window's destination distributions relative to the reference: the
    ``kappa`` for which ``softmax(kappa * log reference)`` maximises the window's likelihood.  kappa = 1 means the
    window is exactly as precise as the reference, kappa > 1 sharper, kappa < 1 flatter.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit

ACTIONS = ("dCPI+/dCSI+", "dCPI+/dCSI-", "dCPI-/dCSI+", "dCPI-/dCSI-")
KAPPA_BOUNDS = (math.exp(-4), math.exp(4))
_EPS = 1e-12


def off_diagonal(n_states: int) -> np.ndarray:
    return ~np.eye(n_states, dtype=bool)


def action_weights(cpi: np.ndarray, csi: np.ndarray, temperature: float = 0.5) -> np.ndarray:
    """Soft quadrant weights (n, 4) from CPI and CSI values; rows without both values are NaN and sum to one otherwise."""
    output = np.full((len(cpi), 4), np.nan)
    valid = np.isfinite(cpi) & np.isfinite(csi)
    p_cpi = expit(np.clip(cpi[valid] / temperature, -30, 30))
    p_csi = expit(np.clip(csi[valid] / temperature, -30, 30))
    output[valid] = np.column_stack((p_cpi * p_csi, p_cpi * (1 - p_csi), (1 - p_cpi) * p_csi, (1 - p_cpi) * (1 - p_csi)))
    return output


def gradient_weights(cpi_gradient: np.ndarray, csi_gradient: np.ndarray) -> np.ndarray:
    """Soft increasing/decreasing quadrants of the CPI and CSI gradients, scaled by each recording's typical slope."""
    normalised = []
    for values in (cpi_gradient, csi_gradient):
        valid = np.isfinite(values)
        scale = np.nanmedian(np.abs(values[valid])) if valid.any() else np.nan
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = np.nanstd(values[valid]) if valid.any() else np.nan
        normalised.append(values / scale if np.isfinite(scale) and scale > 1e-12 else np.where(valid, 0.0, np.nan))
    return action_weights(normalised[0], normalised[1])


def transition_counts(source: np.ndarray, destination: np.ndarray, n_states: int) -> np.ndarray:
    counts = np.zeros((n_states, n_states), dtype=float)
    np.add.at(counts, (source, destination), 1)
    return counts


def tensor_counts(source: np.ndarray, destination: np.ndarray, weights: np.ndarray, n_states: int) -> np.ndarray:
    counts = np.zeros((4, n_states, n_states), dtype=float)
    valid = np.isfinite(weights).all(axis=1)
    for action in range(4):
        np.add.at(counts[action], (source[valid], destination[valid]), weights[valid, action])
    return counts


def row_probabilities(counts: np.ndarray, prior: np.ndarray | None = None, prior_strength: float = 0.0,
                      jeffreys: float = 0.5) -> np.ndarray:
    """Destination probabilities per source (diagonal zero) with a Jeffreys term and an optional informative prior.

    ``counts`` is (K, K) or (A, K, K); ``prior`` is a matching (K, K) or (A, K, K) probability array that receives
    ``prior_strength`` pseudo-counts per row.
    """
    values = np.asarray(counts, float)
    was_matrix = values.ndim == 2
    if was_matrix:
        values = values[None]
    n_states = values.shape[-1]
    output = np.zeros_like(values)
    for action in range(len(values)):
        for source in range(n_states):
            allowed = np.arange(n_states) != source
            if prior is None:
                p0 = np.full(n_states - 1, 1.0 / (n_states - 1))
            else:
                p0 = np.asarray(prior if np.ndim(prior) == 2 else prior[action])[source, allowed]
                p0 = p0 / p0.sum()
            alpha = np.clip(values[action, source, allowed], 0, None) + prior_strength * p0 + jeffreys
            output[action, source, allowed] = alpha / alpha.sum()
    return output[0] if was_matrix else output


def _rows(values: np.ndarray, probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values, probabilities = np.asarray(values, float), np.asarray(probabilities, float)
    if values.ndim == 2:
        values, probabilities = values[None], probabilities[None]
    mask = off_diagonal(values.shape[-1])
    width = values.shape[-1] - 1
    return values[..., mask].reshape(-1, width), probabilities[..., mask].reshape(-1, width)


def weighted_js(counts: np.ndarray, reference: np.ndarray) -> float:
    x, q = _rows(counts, reference)
    n = x.sum(axis=1)
    valid = n > 0
    if not valid.any():
        return np.nan
    x, q, n = x[valid], q[valid], n[valid]
    width = x.shape[1]
    p = (x + 0.5) / (n[:, None] + 0.5 * width)
    q = np.clip(q, _EPS, None)
    q /= q.sum(axis=1, keepdims=True)
    midpoint = (p + q) / 2
    js = 0.5 * np.sum(p * np.log(p / midpoint), axis=1) + 0.5 * np.sum(q * np.log(q / midpoint), axis=1)
    return float(np.average(js, weights=n))


def weighted_js_batch(counts: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """``weighted_js`` for a stack of count tensors (B, A, K, K) against one reference (A, K, K); returns (B,)."""
    counts = np.asarray(counts, float)
    n_states = counts.shape[-1]
    mask, width = off_diagonal(n_states), n_states - 1
    x = counts[..., mask].reshape(len(counts), -1, width)
    q = np.asarray(reference, float)[..., mask].reshape(-1, width)
    n = x.sum(axis=2)
    p = (x + 0.5) / (n[..., None] + 0.5 * width)
    q = np.clip(q, _EPS, None)
    q /= q.sum(axis=1, keepdims=True)
    midpoint = (p + q) / 2
    js = 0.5 * np.sum(p * np.log(p / midpoint), axis=2) + 0.5 * np.sum(q * np.log(q / midpoint), axis=2)
    total = n.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0, (js * n).sum(axis=1) / total, np.nan)


def fit_kappa(counts: np.ndarray, reference: np.ndarray) -> float:
    x, probabilities = _rows(counts, reference)
    effect = np.log(np.clip(probabilities, _EPS, 1))
    n = x.sum(axis=1)
    valid = n > 0
    if not valid.any() or np.all(np.ptp(effect[valid], axis=1) < 1e-12):
        return np.nan
    x, effect, n = x[valid], effect[valid], n[valid]

    def gradient(kappa: float) -> float:
        logits = kappa * effect
        logits -= logits.max(axis=1, keepdims=True)
        probability = np.exp(logits)
        probability /= probability.sum(axis=1, keepdims=True)
        return float(np.sum(x * effect) - np.sum(n * np.sum(probability * effect, axis=1)))

    lower, upper = KAPPA_BOUNDS
    if gradient(lower) <= 0:
        return lower
    if gradient(upper) >= 0:
        return upper
    return float(brentq(gradient, lower, upper, xtol=1e-8))


def stable_rng(key: str) -> np.random.Generator:
    """Deterministic generator keyed by a string (recording, window, purpose): reruns are bit-identical."""
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little"))


def expected_js_under_reference(source: np.ndarray, weights: np.ndarray, reference: np.ndarray, key: str,
                                draws: int) -> float:
    """E0: mean JS when this window's destinations are drawn from the reference.

    ``weights`` is (n, A) with rows summing to one (the ordinary matrix is A = 1 with unit weights).
    """
    source = np.asarray(source, int)
    weights = np.asarray(weights, float)
    n_actions = weights.shape[1]
    n_states = np.asarray(reference).shape[-1]
    reference = np.asarray(reference, float).reshape(n_actions, n_states, n_states)
    if len(source) == 0:
        return np.nan
    rows = np.einsum("ia,aid->id", weights, reference[:, source, :])
    rows /= rows.sum(axis=1, keepdims=True)
    cdf = np.cumsum(rows, axis=1)
    cdf[:, -1] = 1.0
    uniform = stable_rng(key).random((draws, len(source)))
    destination = (uniform[:, :, None] >= cdf[None, :, : n_states - 1]).sum(axis=2)
    flat = (source[None, :] * n_states + destination + np.arange(draws)[:, None] * n_states ** 2).ravel()
    counts = np.empty((draws, n_actions, n_states, n_states))
    for action in range(n_actions):
        counts[:, action] = np.bincount(flat, weights=np.tile(weights[:, action], draws),
                                        minlength=draws * n_states ** 2).reshape(draws, n_states, n_states)
    return float(np.nanmean(weighted_js_batch(counts, reference)))
