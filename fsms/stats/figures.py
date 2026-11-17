"""Figures (PNG, 300 dpi, colour-blind-safe Okabe-Ito colours; group is also encoded by marker shape).

* ``a1_interaction.png``: per outcome, group x condition means with 95% CI and individual participants;
* ``a2_change_distributions.png``: per outcome, participant change from baseline by group;
* ``a3_trajectories.png``: JS and kappa mean change by minute with the fixed-effect lines of the mixed model;
* ``a4_roc_curves.png``: ROC curves of the single-predictor models.

Every figure carries the preregistration identifier and representation in a footer and in the PNG metadata.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from matplotlib import rc_context
from matplotlib.figure import Figure

from ..atomic import atomic_write_bytes
from .classification import SINGLE_PREDICTOR_MODELS, ClassificationResult
from .change_test import ChangeResult
from .mixed_anova import MixedAnovaResult
from .release import GROUPS, OUTCOMES
from .trajectory_model import A3_OUTCOMES, TrajectoryResult

COLOURS = {"FS": "#D55E00", "ES": "#0072B2"}
MARKERS = {"FS": "o", "ES": "s"}
MODEL_STYLE = {"M_JS": ("#0072B2", "-"), "M_a": ("#D55E00", "--"), "M_L": ("#000000", "-.")}
OUTCOME_LABELS = {"JS": "JS excess", "kappa": "log kappa", "CSI": "CSI (z)", "CPI": "CPI (z)"}
_STYLE = {"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 10,
          "axes.labelsize": 9, "legend.frameon": False, "legend.fontsize": 8}


def _save(fig: Figure, path: Path, prereg_id: str, representation: str) -> Path:
    fig.get_layout_engine().set(rect=(0.0, 0.05, 1.0, 0.95))
    fig.text(0.995, 0.005, f"{prereg_id} | {representation}", ha="right", va="bottom", fontsize=6, color="#555555")
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight",
                metadata={"Description": f"prereg_id={prereg_id}; representation={representation}"})
    atomic_write_bytes(path, buffer.getvalue())
    return path


def _format_p(p: float) -> str:
    return "p = n/a" if not np.isfinite(p) else ("p < 0.001" if p < 0.001 else f"p = {p:.3f}")


def plot_interaction(result: MixedAnovaResult, path: Path, prereg_id: str, representation: str) -> Path:
    """A1: 2 x 2 panels (one per outcome) of group x condition means, 95% CI and individual participants."""
    rng = np.random.default_rng(0)
    with rc_context(_STYLE):
        fig = Figure(figsize=(7.2, 6.4), layout="constrained")
        for axis, outcome in zip(fig.subplots(2, 2).ravel(), OUTCOMES):
            values = result.values[result.values["outcome"] == outcome]
            cells = result.cell_means[result.cell_means["outcome"] == outcome]
            interaction = result.omnibus[(result.omnibus["outcome"] == outcome) & (result.omnibus["effect"] == "interaction")]
            for offset, group in zip((-0.12, 0.12), GROUPS):
                subset = values[values["group"] == group]
                jitter = rng.uniform(-0.04, 0.04, len(subset))
                for x0, x1, y0, y1 in zip(np.full(len(subset), 0 + offset) + jitter, np.full(len(subset), 1 + offset) + jitter,
                                          subset["interictal"], subset["final"]):
                    axis.plot([x0, x1], [y0, y1], color=COLOURS[group], alpha=0.12, lw=0.7, zorder=1)
                axis.scatter(np.r_[np.full(len(subset), 0 + offset), np.full(len(subset), 1 + offset)] + np.r_[jitter, jitter],
                             np.r_[subset["interictal"], subset["final"]], s=9, color=COLOURS[group], alpha=0.35,
                             marker=MARKERS[group], zorder=2, linewidths=0)
                cell = cells[cells["group"] == group].set_index("condition")
                means = cell.loc[["interictal", "final_minute"], "mean"].to_numpy()
                low = cell.loc[["interictal", "final_minute"], "ci_low"].to_numpy()
                high = cell.loc[["interictal", "final_minute"], "ci_high"].to_numpy()
                axis.errorbar([0 + offset, 1 + offset], means, yerr=[means - low, high - means], color=COLOURS[group],
                              marker=MARKERS[group], ms=6, lw=1.8, capsize=3, zorder=3,
                              label=f"{group} (n = {int(cell['n'].iloc[0])})")
            axis.set_xticks([0, 1], ["interictal", "final minute"])
            axis.set_xlim(-0.5, 1.5)
            axis.set_ylabel(OUTCOME_LABELS[outcome])
            axis.set_title(f"{outcome}: interaction {_format_p(float(interaction['p'].iloc[0]))}, "
                           f"Holm {_format_p(float(interaction['p_holm'].iloc[0]))}", fontsize=8)
            axis.legend(loc="best")
        return _save(fig, path, prereg_id, representation)


def plot_change_distributions(result: ChangeResult, path: Path, prereg_id: str, representation: str) -> Path:
    """A2: per outcome, participant change (final minute minus baseline block) by group."""
    rng = np.random.default_rng(0)
    with rc_context(_STYLE):
        fig = Figure(figsize=(7.2, 6.4), layout="constrained")
        for axis, outcome in zip(fig.subplots(2, 2).ravel(), OUTCOMES):
            changes = result.participant_change[result.participant_change["outcome"] == outcome]
            stats_row = result.between[result.between["outcome"] == outcome].iloc[0]
            for position, group in enumerate(GROUPS):
                x = changes.loc[changes["group"] == group, "change"].to_numpy()
                axis.boxplot([x], positions=[position], widths=0.45, showfliers=False, manage_ticks=False,
                             medianprops={"color": "black"}, boxprops={"color": COLOURS[group]},
                             whiskerprops={"color": COLOURS[group]}, capprops={"color": COLOURS[group]})
                axis.scatter(position + rng.uniform(-0.12, 0.12, len(x)), x, s=14, color=COLOURS[group], alpha=0.6,
                             marker=MARKERS[group], linewidths=0, zorder=3)
            axis.axhline(0, color="#888888", lw=0.8, ls=":")
            axis.set_xticks([0, 1], [f"FS (n = {int(stats_row['n_fs'])})", f"ES (n = {int(stats_row['n_es'])})"])
            axis.set_xlim(-0.6, 1.6)
            axis.set_ylabel(f"change in {OUTCOME_LABELS[outcome]}")
            axis.set_title(f"{outcome}: g = {stats_row['hedges_g']:.2f}, {_format_p(float(stats_row['p']))}, "
                           f"Holm {_format_p(float(stats_row['p_holm']))}", fontsize=8)
        return _save(fig, path, prereg_id, representation)


def plot_trajectories(result: TrajectoryResult, path: Path, prereg_id: str, representation: str) -> Path:
    """A3: mean change by minute (mean and SE over participants) with the mixed-model fixed-effect lines."""
    with rc_context(_STYLE):
        fig = Figure(figsize=(7.2, 3.4), layout="constrained")
        for axis, outcome in zip(fig.subplots(1, 2), A3_OUTCOMES):
            data = result.data[result.data["outcome"] == outcome]
            fixed = result.fixed[result.fixed["outcome"] == outcome].set_index("term")["estimate"]
            model = result.models[result.models["outcome"] == outcome].iloc[0]
            interaction = result.fixed[(result.fixed["outcome"] == outcome) & (result.fixed["term"] == "t:group_FS")].iloc[0]
            t_grid = np.linspace(-9, 0, 50)
            for group in GROUPS:
                summary = data[data["group"] == group].groupby("t")["change"].agg(["mean", "sem"]).reset_index()
                axis.errorbar(summary["t"] + (0.08 if group == "FS" else -0.08), summary["mean"], yerr=summary["sem"],
                              fmt=MARKERS[group], color=COLOURS[group], ms=4, capsize=2, lw=1, alpha=0.9,
                              label=f"{group} mean +/- SE")
                is_fs = float(group == "FS")
                line = (fixed["intercept"] + fixed["group_FS"] * is_fs) + (fixed["t"] + fixed["t:group_FS"] * is_fs) * t_grid
                axis.plot(t_grid, line, color=COLOURS[group], lw=1.8, label=f"{group} model")
            axis.axhline(0, color="#888888", lw=0.8, ls=":")
            axis.set_xlabel("minutes relative to onset (t = -window_end_min)")
            axis.set_ylabel(f"change in {OUTCOME_LABELS[outcome]}")
            axis.set_title(f"{outcome}: t x group = {interaction['estimate']:.3g}/min, {_format_p(float(interaction['p']))}\n"
                           f"fit: {model['fit_status']}", fontsize=8)
            axis.legend(loc="best", ncol=2)
        return _save(fig, path, prereg_id, representation)


def plot_roc(result: ClassificationResult, path: Path, prereg_id: str, representation: str) -> Path:
    """A4: ROC curves of M_JS, M_a and M_L (apparent) with apparent and optimism-corrected AUC."""
    with rc_context(_STYLE):
        fig = Figure(figsize=(4.2, 4.2), layout="constrained")
        axis = fig.subplots()
        axis.plot([0, 1], [0, 1], color="#999999", lw=0.8, ls=":")
        for model in SINGLE_PREDICTOR_MODELS:
            curve = result.primary.roc[result.primary.roc["model"] == model]
            row = result.primary.models[result.primary.models["model"] == model].iloc[0]
            colour, style = MODEL_STYLE[model]
            axis.plot(curve["fpr"], curve["tpr"], color=colour, ls=style, lw=1.8,
                      label=f"{model}: AUC {row['auc_apparent']:.2f} (corrected {row['auc_corrected']:.2f})")
        axis.set_xlabel("false-positive rate (ES classified as FS)")
        axis.set_ylabel("true-positive rate (FS classified as FS)")
        axis.set_aspect("equal")
        axis.legend(loc="lower right", fontsize=7)
        return _save(fig, path, prereg_id, representation)


def make_figures(directory: Path, prereg_id: str, representation: str, a1: MixedAnovaResult, a2: ChangeResult,
                 a3: TrajectoryResult, a4: ClassificationResult) -> list[Path]:
    """Write the four figures of one representation into ``directory`` and return their paths."""
    return [
        plot_interaction(a1, directory / "a1_interaction.png", prereg_id, representation),
        plot_change_distributions(a2, directory / "a2_change_distributions.png", prereg_id, representation),
        plot_trajectories(a3, directory / "a3_trajectories.png", prereg_id, representation),
        plot_roc(a4, directory / "a4_roc_curves.png", prereg_id, representation),
    ]
