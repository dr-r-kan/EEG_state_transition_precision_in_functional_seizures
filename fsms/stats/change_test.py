"""A2: change toward seizure onset, FS-only vs ES-only participants.

For every eligible seizure the change of an outcome is the final-minute value minus the mean of the baseline
block (``window_end_min`` 10..19, at least five valid windows).  A participant's change is the mean of their
seizures' changes.  Between groups the change is compared with a Welch t-test (estimate FS minus ES, 95% CI,
Hedges' g) and, as a sensitivity analysis, a Mann-Whitney U test.  Within each group the change is tested
against zero with a one-sample t-test (descriptive).

Multiplicity: Holm across the four outcomes, separately for the Welch test, the Mann-Whitney test and, within
each group, the one-sample tests.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .inference import holm, mann_whitney, one_sample_test, welch_test
from .release import GROUPS, OUTCOMES, RepresentationData

REFERENCES = {"baseline_block": "change", "interictal_mean": "change_vs_interictal"}


@dataclass(frozen=True)
class ChangeResult:
    """Tables of analysis A2."""

    participant_change: pd.DataFrame
    between: pd.DataFrame
    within: pd.DataFrame


def participant_changes(rep: RepresentationData, reference: str = "baseline_block") -> pd.DataFrame:
    """Participant-level change per outcome (mean over eligible seizures) for FS-only and ES-only participants.

    ``reference`` is ``baseline_block`` (final minute minus the 20-10 minute block) or ``interictal_mean``
    (final minute minus the participant's interictal mean).  Columns: ``participant_id``, ``group``,
    ``outcome``, ``change``, ``n_events``.
    """
    column = REFERENCES[reference]
    events = rep.events[rep.events["group"].isin(GROUPS)].dropna(subset=[column])
    table = events.groupby(["participant_id", "group", "outcome"])[column].agg(change="mean", n_events="count")
    return table.reset_index()


def run_change_test(rep: RepresentationData) -> ChangeResult:
    """Run A2 for every outcome."""
    changes = participant_changes(rep)
    between_rows, within_rows = [], []
    for outcome in OUTCOMES:
        subset = changes[changes["outcome"] == outcome]
        fs = subset.loc[subset["group"] == "FS", "change"].to_numpy()
        es = subset.loc[subset["group"] == "ES", "change"].to_numpy()
        welch, rank = welch_test(fs, es), mann_whitney(fs, es)
        between_rows.append({
            "outcome": outcome, "n_fs": welch["n_1"], "n_es": welch["n_2"],
            "mean_fs": welch["mean_1"], "mean_es": welch["mean_2"], "sd_fs": welch["sd_1"], "sd_es": welch["sd_2"],
            "estimate": welch["estimate"], "se": welch["se"], "ci_low": welch["ci_low"], "ci_high": welch["ci_high"],
            "t": welch["t"], "df": welch["df"], "p": welch["p"],
            "hedges_g": welch["hedges_g"], "hedges_g_ci_low": welch["hedges_g_ci_low"],
            "hedges_g_ci_high": welch["hedges_g_ci_high"],
            "mannwhitney_u": rank["u"], "p_mannwhitney": rank["p"],
        })
        for group, values in (("FS", fs), ("ES", es)):
            one = one_sample_test(values)
            within_rows.append({
                "group": group, "outcome": outcome, "n": one["n"], "mean": one["mean"], "sd": one["sd"],
                "se": one["se"], "ci_low": one["ci_low"], "ci_high": one["ci_high"], "t": one["t"],
                "df": one["df"], "p": one["p"], "dz": one["dz"], "hedges_g": one["hedges_g"],
            })
    between = pd.DataFrame(between_rows)
    between["p_holm"] = holm(between["p"])
    between["p_mannwhitney_holm"] = holm(between["p_mannwhitney"])
    within = pd.DataFrame(within_rows)
    within["p_holm"] = np.nan
    for group in GROUPS:
        mask = within["group"] == group
        within.loc[mask, "p_holm"] = holm(within.loc[mask, "p"])
    return ChangeResult(participant_change=changes, between=between, within=within)
