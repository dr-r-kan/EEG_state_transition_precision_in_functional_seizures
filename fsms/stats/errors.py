"""Exceptions raised by the statistics package."""
from __future__ import annotations


class StatisticsGateError(RuntimeError):
    """The release bundle has no valid inclusion checkpoint for the requested preregistration identifier."""


class ReleaseSchemaError(ValueError):
    """A release table is missing a required column or violates a structural assumption."""
