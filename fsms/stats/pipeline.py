"""Orchestration: gate check, load the release bundle, run A1-A4 for each representation, write all outputs.

Output layout under ``out_dir``::

    results.json, results_summary.md            pooled over representations
    <representation>/tables/*.csv               every table (columns documented in docs/STATISTICAL_ANALYSIS.md)
    <representation>/figures/*.png
    <representation>/L_definition.json          standardisation constants and definition of the index L
"""
from __future__ import annotations

import os
import platform
from pathlib import Path

import matplotlib
import numpy
import pandas
import scipy
import statsmodels

from .. import __version__ as package_version
from ..atomic import sha256_file
from .change_test import ChangeResult, run_change_test
from .classification import BOOTSTRAP_SEED, N_BOOTSTRAP, ClassificationResult, run_classification
from .figures import make_figures
from .mixed_anova import MixedAnovaResult, run_mixed_anova
from .outputs import jsonable, markdown_table, records, write_json, write_summary, write_table
from .release import (CHECKPOINT_FILE, GROUPS, OUTCOMES, REPRESENTATIONS, RepresentationData, check_gate,
                      load_release, prepare_representation)
from .semiology import run_semiology
from .trajectory_model import A3_OUTCOMES, TrajectoryResult, run_trajectory_model

HOLM_FAMILIES = [
    {"analysis": "A1", "family": "omnibus group effect", "tests": list(OUTCOMES), "m": 4},
    {"analysis": "A1", "family": "omnibus condition effect", "tests": list(OUTCOMES), "m": 4},
    {"analysis": "A1", "family": "omnibus interaction", "tests": list(OUTCOMES), "m": 4},
    {"analysis": "A1", "family": "post-hoc, within each simple-effect type (p_holm_outcomes)", "tests": list(OUTCOMES), "m": 4},
    {"analysis": "A1", "family": "post-hoc, all 16 simple effects (p_holm_all; headline)",
     "tests": [f"{o}:{e}" for o in OUTCOMES for e in ("group_at_interictal", "group_at_final_minute",
                                                   "condition_in_FS", "condition_in_ES")], "m": 16},
    {"analysis": "A2", "family": "FS vs ES Welch test (and, separately, Mann-Whitney)", "tests": list(OUTCOMES), "m": 4},
    {"analysis": "A2", "family": "one-sample change vs 0, within each group", "tests": list(OUTCOMES), "m": 4},
    {"analysis": "A3", "family": "t:group_FS interaction (headline)", "tests": list(A3_OUTCOMES), "m": 2},
    {"analysis": "A3", "family": "group_FS at the final minute", "tests": list(A3_OUTCOMES), "m": 2},
    {"analysis": "A4", "family": "single-predictor models (LR and, separately, Wald p)", "tests": ["M_JS", "M_a", "M_L"], "m": 3},
]


def _event_counts(rep: RepresentationData) -> dict:
    events = rep.events.drop_duplicates("recording_id")
    return {
        "eligible_participants": {g: int((rep.participants["group"] == g).sum()) for g in (*GROUPS, "DD")},
        "eligible_events": {g: int((events["group"] == g).sum()) for g in (*GROUPS, "DD")},
    }


def _write_representation(rep: RepresentationData, directory: Path, prereg_id: str, n_boot: int
                          ) -> tuple[dict, dict, list[tuple[str, str]]]:
    """Run all analyses for ``rep``; return (results block, file hashes, summary sections)."""
    tables, figures = directory / "tables", directory / "figures"
    a1, a2 = run_mixed_anova(rep), run_change_test(rep)
    a3, a4 = run_trajectory_model(rep), run_classification(rep, n_boot=n_boot)
    e1 = run_semiology(rep)

    files: dict[str, str] = {}

    def table(frame, name: str, **extra: str) -> None:
        files[f"tables/{name}"] = write_table(frame, tables / name, prereg_id, rep.name, **extra)

    table(a1.values, "a1_participant_values.csv")
    table(a1.cell_means, "a1_cell_means.csv")
    table(a1.omnibus, "a1_omnibus.csv")
    table(a1.posthoc, "a1_posthoc.csv")
    table(a2.participant_change, "a2_participant_change.csv")
    table(a2.between, "a2_between_groups.csv")
    table(a2.within, "a2_within_group.csv")
    table(a3.data, "a3_participant_minute_changes.csv")
    table(a3.models, "a3_models.csv")
    table(a3.fixed, "a3_fixed_effects.csv")
    table(a3.variance, "a3_variance_components.csv")
    table(a3.simple, "a3_simple_effects.csv")
    table(a4.predictors, "a4_participant_predictors.csv", change_reference="baseline_block")
    table(a4.primary.models, "a4_models.csv", analysis="primary", change_reference="baseline_block")
    table(a4.primary.coefficients, "a4_coefficients.csv", analysis="primary", change_reference="baseline_block")
    table(a4.primary.roc, "a4_roc_curve_points.csv", analysis="primary", change_reference="baseline_block")
    table(a4.sensitivity.models, "a4_sensitivity_models.csv", analysis="sensitivity", change_reference="interictal_mean")
    table(a4.sensitivity.coefficients, "a4_sensitivity_coefficients.csv", analysis="sensitivity",
          change_reference="interictal_mean")
    table(a4.event_models, "a4_event_level_models.csv", analysis="event_level", change_reference="baseline_block")
    table(a4.event_coefficients, "a4_event_level_coefficients.csv", analysis="event_level",
          change_reference="baseline_block")
    table(a4.dd_pairs, "a4_dd_paired.csv", analysis="dd_paired", change_reference="baseline_block")
    table(e1.participant_change, "e1_semiology_participant_change.csv", analysis="exploratory")
    table(e1.between, "e1_semiology_between.csv", analysis="exploratory")

    definition = {"prereg_id": prereg_id, "representation": rep.name, **a4.l_definition}
    files["L_definition.json"] = write_json(definition, directory / "L_definition.json")
    make_figures(figures, prereg_id, rep.name, a1, a2, a3, a4)

    block = _results_block(rep, a1, a2, a3, a4)
    block["E1_exploratory_semiology"] = {
        "note": "Exploratory; hypo minus hyper; p-values unadjusted and not used for inference.",
        "between": records(e1.between, ["outcome", "n_hypo", "n_hyper", "n_mixed_excluded", "estimate", "ci_low",
                                        "ci_high", "hedges_g", "p", "p_mannwhitney"])}
    sections = _summary_sections(rep.name, a1, a2, a3, a4)
    sections.append((f"## {rep.name}: E1 (exploratory) hypomotor vs hypermotor functional seizures",
                     "`estimate` is hypo minus hyper (Welch) of the participant change (final minute minus baseline "
                     "block). Unadjusted p-values; not used for inference.\n\n" + markdown_table(
                         e1.between, ["outcome", "n_hypo", "n_hyper", "n_mixed_excluded", "mean_hypo", "mean_hyper",
                                      "estimate", "ci_low", "ci_high", "t", "df", "p", "hedges_g", "p_mannwhitney"])))
    return block, files, sections


def _results_block(rep: RepresentationData, a1: MixedAnovaResult, a2: ChangeResult, a3: TrajectoryResult,
                   a4: ClassificationResult) -> dict:
    primary = a4.primary
    return {
        "outcome_columns": dict(rep.columns),
        "n": {
            **_event_counts(rep),
            "A1_participants": {o: {g: int(((a1.values["outcome"] == o) & (a1.values["group"] == g)).sum())
                                   for g in GROUPS} for o in OUTCOMES},
            "A2_participants": {o: {"FS": int(r["n_fs"]), "ES": int(r["n_es"])}
                                for o, r in a2.between.set_index("outcome").iterrows()},
            "A3": {r["outcome"]: {"participants": int(r["n_participants"]), "FS": int(r["n_fs"]), "ES": int(r["n_es"]),
                                  "participant_minute_observations": int(r["n_obs"])} for _, r in a3.models.iterrows()},
            "A4_primary": {"participants": int(primary.models["n"].iloc[0]), "FS": int(primary.models["n_fs"].iloc[0]),
                           "ES": int(primary.models["n_es"].iloc[0])},
            "A4_event_level": records(a4.event_models.head(1), ["n_events", "n_participants", "n_fs_events",
                                                                "n_es_events", "n_events_dd_participants"])[0],
            "A4_dd_pairs": int(a4.dd_pairs["n_pairs"].iloc[0]),
        },
        "holm_families": HOLM_FAMILIES,
        "A1": {
            "omnibus": records(a1.omnibus, ["outcome", "effect", "f", "df1", "df2", "p", "p_holm", "partial_eta_sq"]),
            "posthoc": records(a1.posthoc, ["outcome", "effect", "estimate", "ci_low", "ci_high", "t", "df", "p",
                                            "effect_size", "effect_size_type", "p_holm_outcomes", "p_holm_all"]),
        },
        "A2": {
            "between": records(a2.between, ["outcome", "estimate", "ci_low", "ci_high", "hedges_g", "p", "p_holm",
                                            "p_mannwhitney", "p_mannwhitney_holm"]),
            "within": records(a2.within, ["group", "outcome", "mean", "ci_low", "ci_high", "p", "p_holm"]),
        },
        "A3": {
            "models": records(a3.models, ["outcome", "fit_status", "converged", "singular", "optimiser",
                                          "random_effects_dim", "lrt_p", "lrt_p_holm_outcomes"]),
            "headline_terms": records(a3.fixed[a3.fixed["term"].isin(["t:group_FS", "group_FS"])],
                                      ["outcome", "term", "estimate", "se", "z", "p", "ci_low", "ci_high",
                                       "p_holm_outcomes"]),
        },
        "A4": {
            "models": records(primary.models, ["model", "n", "p_lr", "p_holm_lr", "p_wald", "p_holm_wald",
                                               "mcfadden_r2", "nagelkerke_r2", "aic", "auc_apparent",
                                               "auc_corrected", "auc_loo"]),
            "coefficients": records(primary.coefficients, ["model", "term", "or_per_sd", "or_per_sd_ci_low",
                                                           "or_per_sd_ci_high", "p_wald", "p_lr"]),
            "sensitivity_interictal_reference": records(a4.sensitivity.models, ["model", "n", "p_lr", "p_holm_lr",
                                                                               "auc_apparent", "auc_corrected",
                                                                               "auc_loo"]),
            "event_level": records(a4.event_coefficients, ["model", "term", "or_per_sd", "or_per_sd_ci_low",
                                                           "or_per_sd_ci_high", "p"]),
            "dd_paired": records(a4.dd_pairs, ["measure", "n_pairs", "estimate_fs_minus_es", "p_paired_t",
                                               "p_wilcoxon"]),
            "L_definition": a4.l_definition,
        },
    }


def _summary_sections(name: str, a1: MixedAnovaResult, a2: ChangeResult, a3: TrajectoryResult,
                      a4: ClassificationResult) -> list[tuple[str, str]]:
    effect_note = "`p_holm` is Holm-adjusted across the four outcomes within each effect."
    return [
        (f"## {name}: A1 mixed ANOVA, omnibus effects",
         effect_note + "\n\n" + markdown_table(a1.omnibus, ["outcome", "effect", "estimate", "f", "df1", "df2", "p",
                                                            "p_holm", "partial_eta_sq"])),
        (f"## {name}: A1 post-hoc simple effects",
         "`p_holm_all` is Holm-adjusted across all 16 tests (headline); `p_holm_outcomes` across the four outcomes "
         "within each effect type.\n\n" + markdown_table(
             a1.posthoc, ["outcome", "effect", "estimate", "ci_low", "ci_high", "t", "df", "p", "effect_size",
                          "effect_size_type", "p_holm_outcomes", "p_holm_all"])),
        (f"## {name}: A2 change (final minute minus baseline block), FS vs ES",
         "`estimate` is FS minus ES (Welch). `p_holm` and `p_mannwhitney_holm` are Holm across the four outcomes.\n\n"
         + markdown_table(a2.between, ["outcome", "n_fs", "n_es", "mean_fs", "mean_es", "estimate", "ci_low", "ci_high",
                                       "t", "df", "p", "p_holm", "hedges_g", "p_mannwhitney", "p_mannwhitney_holm"])),
        (f"## {name}: A2 change versus zero within group",
         markdown_table(a2.within, ["group", "outcome", "n", "mean", "ci_low", "ci_high", "t", "df", "p", "p_holm"])),
        (f"## {name}: A3 mixed model fit status",
         markdown_table(a3.models, ["outcome", "fit_status", "optimiser", "singular", "n_participants", "n_obs",
                                    "random_effects_dim", "lrt_chi2", "lrt_p", "lrt_p_holm_outcomes"])),
        (f"## {name}: A3 fixed effects",
         "`p_holm_outcomes` is Holm across JS and kappa for `t:group_FS` and, separately, for `group_FS`.\n\n"
         + markdown_table(a3.fixed, ["outcome", "term", "estimate", "se", "z", "p", "ci_low", "ci_high",
                                     "p_holm_outcomes"])),
        (f"## {name}: A4 logistic models (FS = 1 vs ES = 0; change vs baseline block)",
         "`p_holm_lr` and `p_holm_wald` are Holm across M_JS, M_a and M_L.\n\n" + markdown_table(
             a4.primary.models, ["model", "n", "p_lr", "p_holm_lr", "p_wald", "p_holm_wald", "mcfadden_r2",
                                 "nagelkerke_r2", "aic", "auc_apparent", "auc_corrected", "auc_loo"])),
        (f"## {name}: A4 coefficients",
         markdown_table(a4.primary.coefficients, ["model", "term", "estimate", "se", "or_per_sd", "or_per_sd_ci_low",
                                                  "or_per_sd_ci_high", "p_wald", "p_lr"])),
        (f"## {name}: A4 sensitivity (change vs interictal mean)",
         markdown_table(a4.sensitivity.models, ["model", "n", "p_lr", "p_holm_lr", "auc_apparent", "auc_corrected",
                                                "auc_loo"])),
        (f"## {name}: A4 event-level models (participant-clustered robust SE)",
         markdown_table(a4.event_coefficients, ["model", "term", "estimate", "robust_se", "or_per_sd",
                                                "or_per_sd_ci_low", "or_per_sd_ci_high", "p"])),
        (f"## {name}: A4 DD participants, FS-seizure minus ES-seizure",
         markdown_table(a4.dd_pairs, ["measure", "n_pairs", "mean_fs_events", "mean_es_events", "estimate_fs_minus_es",
                                      "ci_low", "ci_high", "p_paired_t", "p_wilcoxon"])),
    ]


def run_statistics(release_dir: str | os.PathLike[str], out_dir: str | os.PathLike[str], prereg_id: str,
                   representations: tuple[str, ...] = ("matrix", "tensor"), *, n_bootstrap: int = N_BOOTSTRAP) -> dict:
    """Run the statistical analysis on a release bundle and write all outputs.

    Raises ``StatisticsGateError`` unless ``release_dir/inclusion_checkpoint.json`` exists and carries
    ``prereg_id``.  Each representation is analysed completely and separately (own multiplicity families) and
    written to ``out_dir/<representation>/``.  Returns the content of ``out_dir/results.json``.
    """
    unknown = [r for r in representations if r not in REPRESENTATIONS]
    if unknown or not representations:
        raise ValueError(f"representations must be a non-empty subset of {sorted(REPRESENTATIONS)}, got {representations}")
    release, out = Path(release_dir), Path(out_dir)
    check_gate(release, prereg_id)
    data = load_release(release)
    out.mkdir(parents=True, exist_ok=True)

    blocks, files, sections = {}, {}, []
    for name in representations:
        block, file_hashes, rep_sections = _write_representation(
            prepare_representation(data, name), out / name, prereg_id, n_bootstrap)
        blocks[name], files[name] = block, file_hashes
        sections.extend(rep_sections)

    results = {
        "prereg_id": prereg_id,
        "representations": blocks,
        "settings": {"bootstrap_replicates": n_bootstrap, "bootstrap_seed": BOOTSTRAP_SEED, "confidence_level": 0.95},
        "inputs": {name: sha256_file(release / name)
                   for name in ("windows.csv.gz", "events.csv", "participants.csv", CHECKPOINT_FILE)},
        "outputs": files,
        "software": {"fsms": package_version, "python": platform.python_version(), "numpy": numpy.__version__,
                     "pandas": pandas.__version__, "scipy": scipy.__version__, "statsmodels": statsmodels.__version__,
                     "matplotlib": matplotlib.__version__},
    }
    results = jsonable(results)
    write_json(results, out / "results.json")
    write_summary(out / "results_summary.md", prereg_id, sections)
    return results
