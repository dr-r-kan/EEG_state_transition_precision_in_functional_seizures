"""A1: group (FS, ES; between participants) x condition (interictal, final minute; within participants) mixed ANOVA.

Each participant contributes two numbers per outcome: the mean over their interictal windows (``interictal``)
and the mean, over their eligible seizures, of the final-minute window (``final``).  With two within-participant
conditions the mixed ANOVA reduces exactly to three F-tests on one number per participant, which is how it is
computed here (valid for unequal group sizes):

* group main effect: one-way test of FS vs ES on the participant mean of the two conditions,
  ``(interictal + final) / 2`` (unweighted marginal means);
* interaction: the same test on the difference score ``final - interictal``;
* condition main effect (Type III, unweighted marginal means): test that the intercept of the effect-coded
  model ``difference ~ 1 + group`` is zero, i.e. that the average of the two group means of the difference
  score is zero.

Every effect has df = (1, N - 2) with N the number of participants; partial eta squared is
``F df1 / (F df1 + df2)``.  Post-hoc simple effects: Welch t-tests FS vs ES at each condition and paired t-tests
final minute vs interictal within each group.

Multiplicity: omnibus p-values are Holm-adjusted across the four outcomes separately for each effect.  Post-hoc
p-values are Holm-adjusted across the four outcomes within each simple-effect type (``p_holm_outcomes``) and
across all sixteen post-hoc tests (``p_holm_all``, the headline).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .inference import CONFIDENCE, holm, one_sample_test, welch_test
from .release import GROUPS, OUTCOMES, RepresentationData

OMNIBUS_EFFECTS = ("group", "condition", "interaction")
POSTHOC_EFFECTS = ("group_at_interictal", "group_at_final_minute", "condition_in_FS", "condition_in_ES")
_POSTHOC_LABELS = {
    "group_at_interictal": "FS minus ES, interictal",
    "group_at_final_minute": "FS minus ES, final minute",
    "condition_in_FS": "final minute minus interictal, FS",
    "condition_in_ES": "final minute minus interictal, ES",
}


@dataclass(frozen=True)
class MixedAnovaResult:
    """Tables of analysis A1 (all outcomes of one representation)."""

    values: pd.DataFrame
    cell_means: pd.DataFrame
    omnibus: pd.DataFrame
    posthoc: pd.DataFrame


def participant_values(rep: RepresentationData, outcome: str) -> pd.DataFrame:
    """One row per FS-only / ES-only participant with ``interictal`` and ``final`` values for ``outcome``."""
    events = rep.events[rep.events["outcome"] == outcome].dropna(subset=["final"])
    final = events.groupby("participant_id")["final"].agg(final="mean", n_events="count")
    interictal = rep.interictal[rep.interictal["outcome"] == outcome].set_index("participant_id")
    table = final.join(interictal[["group", "mean", "n_windows"]].rename(
        columns={"mean": "interictal", "n_windows": "n_interictal_windows"}), how="inner")
    table = table[table["group"].isin(GROUPS)].dropna(subset=["interictal", "final"])
    table = table.reset_index()
    table.insert(1, "outcome", outcome)
    return table[["participant_id", "outcome", "group", "interictal", "final", "n_events", "n_interictal_windows"]]


def _two_group_f(score: np.ndarray, is_fs: np.ndarray) -> dict[str, float]:
    """One-way F-test (df 1, N - 2) of FS vs ES on ``score``; ``estimate`` is FS minus ES."""
    first, second = score[is_fs], score[~is_fs]
    n1, n2 = first.size, second.size
    out = {"estimate": np.nan, "se": np.nan, "f": np.nan, "df1": 1.0, "df2": float(n1 + n2 - 2), "p": np.nan}
    if n1 < 1 or n2 < 1 or n1 + n2 < 3:
        return out
    mse = (((first - first.mean()) ** 2).sum() + ((second - second.mean()) ** 2).sum()) / (n1 + n2 - 2)
    out["estimate"] = float(first.mean() - second.mean())
    if mse <= 0:
        return out
    out["se"] = float(np.sqrt(mse * (1 / n1 + 1 / n2)))
    out["f"] = float((out["estimate"] / out["se"]) ** 2)
    out["p"] = float(stats.f.sf(out["f"], 1, n1 + n2 - 2))
    return out


def _condition_f(difference: np.ndarray, is_fs: np.ndarray) -> dict[str, float]:
    """Type III condition effect: intercept of ``difference ~ 1 + group`` under effect coding."""
    first, second = difference[is_fs], difference[~is_fs]
    n1, n2 = first.size, second.size
    out = {"estimate": np.nan, "se": np.nan, "f": np.nan, "df1": 1.0, "df2": float(n1 + n2 - 2), "p": np.nan}
    if n1 < 1 or n2 < 1 or n1 + n2 < 3:
        return out
    mse = (((first - first.mean()) ** 2).sum() + ((second - second.mean()) ** 2).sum()) / (n1 + n2 - 2)
    out["estimate"] = float((first.mean() + second.mean()) / 2)
    if mse <= 0:
        return out
    out["se"] = float(np.sqrt(mse * (1 / n1 + 1 / n2) / 4))
    out["f"] = float((out["estimate"] / out["se"]) ** 2)
    out["p"] = float(stats.f.sf(out["f"], 1, n1 + n2 - 2))
    return out


def mixed_anova(values: pd.DataFrame) -> pd.DataFrame:
    """The three omnibus effects for one outcome.

    ``values`` needs the columns ``group`` (``FS``/``ES``), ``interictal`` and ``final``.  Returns one row per
    effect with ``estimate`` (group: FS minus ES on the condition-averaged score; condition: unweighted mean of
    the two group means of final minus interictal; interaction: FS minus ES in final minus interictal),
    ``se``, ``f``, ``df1``, ``df2``, ``p`` and ``partial_eta_sq``.
    """
    is_fs = (values["group"] == "FS").to_numpy()
    interictal = values["interictal"].to_numpy(float)
    final = values["final"].to_numpy(float)
    difference = final - interictal
    tests = {
        "group": _two_group_f((interictal + final) / 2.0, is_fs),
        "condition": _condition_f(difference, is_fs),
        "interaction": _two_group_f(difference, is_fs),
    }
    rows = []
    for effect in OMNIBUS_EFFECTS:
        row = tests[effect]
        eta = row["f"] * row["df1"] / (row["f"] * row["df1"] + row["df2"]) if np.isfinite(row["f"]) else np.nan
        rows.append({"effect": effect, **row, "partial_eta_sq": float(eta)})
    return pd.DataFrame(rows)


def simple_effects(values: pd.DataFrame) -> pd.DataFrame:
    """The four post-hoc simple effects for one outcome (Welch between groups, paired within groups)."""
    fs = values[values["group"] == "FS"]
    es = values[values["group"] == "ES"]
    records = {
        "group_at_interictal": ("welch", welch_test(fs["interictal"], es["interictal"])),
        "group_at_final_minute": ("welch", welch_test(fs["final"], es["final"])),
        "condition_in_FS": ("paired", one_sample_test(fs["final"] - fs["interictal"])),
        "condition_in_ES": ("paired", one_sample_test(es["final"] - es["interictal"])),
    }
    rows = []
    for effect in POSTHOC_EFFECTS:
        kind, result = records[effect]
        paired = kind == "paired"
        rows.append({
            "effect": effect,
            "comparison": _POSTHOC_LABELS[effect],
            "n_fs": len(fs) if effect != "condition_in_ES" else np.nan,
            "n_es": len(es) if effect != "condition_in_FS" else np.nan,
            "estimate": result["estimate"], "se": result["se"],
            "ci_low": result["ci_low"], "ci_high": result["ci_high"],
            "t": result["t"], "df": result["df"], "p": result["p"],
            "effect_size": result["hedges_g"],
            "effect_size_type": "hedges_dz" if paired else "hedges_g",
            "effect_size_ci_low": result["hedges_g_ci_low"], "effect_size_ci_high": result["hedges_g_ci_high"],
        })
    return pd.DataFrame(rows)


def cell_means(values: pd.DataFrame) -> pd.DataFrame:
    """Mean and 95% t confidence interval of each group x condition cell (participants as units)."""
    rows = []
    for group in GROUPS:
        subset = values[values["group"] == group]
        for condition, column in (("interictal", "interictal"), ("final_minute", "final")):
            x = subset[column].to_numpy(float)
            n = x.size
            mean = float(x.mean()) if n else np.nan
            sd = float(x.std(ddof=1)) if n > 1 else np.nan
            margin = stats.t.ppf(0.5 + CONFIDENCE / 2, n - 1) * sd / np.sqrt(n) if n > 1 else np.nan
            rows.append({"group": group, "condition": condition, "n": n, "mean": mean, "sd": sd,
                         "ci_low": mean - margin, "ci_high": mean + margin})
    return pd.DataFrame(rows)


def run_mixed_anova(rep: RepresentationData) -> MixedAnovaResult:
    """Run A1 for every outcome and apply the Holm families described in the module docstring."""
    values, cells, omnibus, posthoc = [], [], [], []
    for outcome in OUTCOMES:
        table = participant_values(rep, outcome)
        values.append(table)
        for frame, sink in ((cell_means(table), cells), (mixed_anova(table), omnibus), (simple_effects(table), posthoc)):
            frame.insert(0, "outcome", outcome)
            sink.append(frame)
        counts = table["group"].value_counts()
        omnibus[-1]["n_fs"] = int(counts.get("FS", 0))
        omnibus[-1]["n_es"] = int(counts.get("ES", 0))
    omnibus_table = pd.concat(omnibus, ignore_index=True)
    omnibus_table["p_holm"] = np.nan
    for effect in OMNIBUS_EFFECTS:
        mask = omnibus_table["effect"] == effect
        omnibus_table.loc[mask, "p_holm"] = holm(omnibus_table.loc[mask, "p"])
    posthoc_table = pd.concat(posthoc, ignore_index=True)
    posthoc_table["p_holm_outcomes"] = np.nan
    for effect in POSTHOC_EFFECTS:
        mask = posthoc_table["effect"] == effect
        posthoc_table.loc[mask, "p_holm_outcomes"] = holm(posthoc_table.loc[mask, "p"])
    posthoc_table["p_holm_all"] = holm(posthoc_table["p"])
    return MixedAnovaResult(values=pd.concat(values, ignore_index=True), cell_means=pd.concat(cells, ignore_index=True),
                            omnibus=omnibus_table, posthoc=posthoc_table)
