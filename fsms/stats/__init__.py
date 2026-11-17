"""Statistical analysis of the release bundle.

``run_statistics`` reads only the de-identified release bundle (``windows.csv.gz``, ``events.csv``,
``participants.csv``) and runs four analyses for each representation of the transition organisation (``matrix``,
``tensor``) on participant-level quantities of FS-only and ES-only participants:

* A1 mixed ANOVA, group x condition (interictal vs final minute) - ``mixed_anova``;
* A2 change from the baseline block, FS vs ES - ``change_test``;
* A3 linear mixed model of the change over the final ten minutes - ``trajectory_model``;
* A4 logistic regression of seizure type on the change in JS and in the autonomic index ``a`` -
  ``classification``.

The method of each analysis is stated in the docstring of its module and in ``docs/STATISTICAL_ANALYSIS.md``.
"""
from .errors import ReleaseSchemaError, StatisticsGateError
from .pipeline import run_statistics

__all__ = ["run_statistics", "StatisticsGateError", "ReleaseSchemaError"]
