"""E1 (exploratory): change toward seizure onset in hypomotor versus hypermotor functional seizures.

Seizures: eligible functional seizures (``event_type == 'FS'``) whose ``semiology`` is ``hypo`` or ``hyper``, of
FS-only and DD participants.  The change of an outcome is the same quantity as in A2 (final minute minus the mean
of the baseline block).  A participant whose classified functional seizures all have one semiology is assigned to
it and contributes the mean change over those seizures; a participant with seizures of both semiologies is
``mixed`` and is left out of the comparison (counted in ``n_mixed_excluded``).

Hypomotor versus hypermotor participants are compared with a Welch t-test (estimate hypo minus hyper, 95% CI,
Hedges' g) and a Mann-Whitney U test.  The analysis is exploratory: p-values are not adjusted for multiplicity and
are not used for inference.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .inference import mann_whitney, welch_test
from .release import OUTCOMES, RepresentationData

LEVELS = ("hypo", "hyper")
PARTICIPANT_COLUMNS = ["participant_id", "group", "semiology", "outcome", "change", "n_events"]


@dataclass(frozen=True)
class SemiologyResult:
    """Tables of the exploratory analysis E1."""

    participant_change: pd.DataFrame
    between: pd.DataFrame


def participant_changes(rep: RepresentationData) -> pd.DataFrame:
    """Participant-level change per outcome over classified functional seizures.

    Columns: ``participant_id``, ``group``, ``semiology`` (``hypo``, ``hyper`` or ``mixed``), ``outcome``,
    ``change`` (mean over the participant's classified functional seizures), ``n_events``.
    """
    events = rep.events[(rep.events["event_type"] == "FS") & rep.events["semiology"].isin(LEVELS)]
    events = events.dropna(subset=["change"])
    if events.empty:
        return pd.DataFrame(columns=PARTICIPANT_COLUMNS)
    grouped = events.groupby(["participant_id", "group", "outcome"])
    table = grouped["change"].agg(change="mean", n_events="count").reset_index()
    kinds = grouped["semiology"].agg(lambda s: s.iloc[0] if s.nunique() == 1 else "mixed").reset_index()
    return table.merge(kinds, on=["participant_id", "group", "outcome"])[PARTICIPANT_COLUMNS]


def run_semiology(rep: RepresentationData) -> SemiologyResult:
    """Run E1 for every outcome."""
    changes = participant_changes(rep)
    rows = []
    for outcome in OUTCOMES:
        subset = changes[changes["outcome"] == outcome]
        hypo, hyper = (subset[subset["semiology"] == level] for level in LEVELS)
        welch = welch_test(hypo["change"].to_numpy(), hyper["change"].to_numpy())
        rank = mann_whitney(hypo["change"].to_numpy(), hyper["change"].to_numpy())
        rows.append({
            "outcome": outcome, "n_hypo": welch["n_1"], "n_hyper": welch["n_2"],
            "n_mixed_excluded": int((subset["semiology"] == "mixed").sum()),
            "n_events_hypo": int(hypo["n_events"].sum()), "n_events_hyper": int(hyper["n_events"].sum()),
            "mean_hypo": welch["mean_1"], "mean_hyper": welch["mean_2"], "sd_hypo": welch["sd_1"],
            "sd_hyper": welch["sd_2"], "estimate": welch["estimate"], "se": welch["se"], "ci_low": welch["ci_low"],
            "ci_high": welch["ci_high"], "t": welch["t"], "df": welch["df"], "p": welch["p"],
            "hedges_g": welch["hedges_g"], "hedges_g_ci_low": welch["hedges_g_ci_low"],
            "hedges_g_ci_high": welch["hedges_g_ci_high"], "mannwhitney_u": rank["u"], "p_mannwhitney": rank["p"],
        })
    return SemiologyResult(participant_change=changes, between=pd.DataFrame(rows))
