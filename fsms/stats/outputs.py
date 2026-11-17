"""Writing tables, the pooled ``results.json`` and the readable ``results_summary.md``.

Every CSV table starts with the columns ``prereg_id`` and ``representation``; ``results.json`` and
``results_summary.md`` carry the preregistration identifier at the top.  JSON output is strict (NaN becomes
``null``) and deterministic (no timestamps), so re-running on the same bundle reproduces identical files.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from ..atomic import atomic_write_json, atomic_write_text, sha256_file


def write_table(frame: pd.DataFrame, path: Path, prereg_id: str, representation: str, **extra: str) -> str:
    """Write ``frame`` as CSV with identifying columns in front; return the SHA-256 of the file."""
    table = frame.copy()
    for position, (name, value) in enumerate(extra.items()):
        table.insert(position, name, value)
    table.insert(0, "representation", representation)
    table.insert(0, "prereg_id", prereg_id)
    atomic_write_text(path, table.to_csv(index=False, lineterminator="\n"))
    return sha256_file(path)


def jsonable(value: Any) -> Any:
    """Recursively convert numpy / pandas values to JSON-safe Python values (non-finite floats become None)."""
    if isinstance(value, pd.DataFrame):
        return jsonable(value.to_dict(orient="records"))
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    return value


def write_json(value: Any, path: Path) -> str:
    atomic_write_json(path, jsonable(value))
    return sha256_file(path)


def records(frame: pd.DataFrame, columns: Iterable[str]) -> list[dict]:
    """Rows of ``frame`` restricted to ``columns`` as JSON-safe dictionaries."""
    return jsonable(frame[list(columns)].to_dict(orient="records"))


def _cell(value: Any) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{value:.3g}"
    return str(value)


def markdown_table(frame: pd.DataFrame, columns: Iterable[str]) -> str:
    """GitHub-flavoured Markdown table of the given columns (floats to three significant digits)."""
    columns = list(columns)
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in frame[columns].itertuples(index=False):
        lines.append("| " + " | ".join(_cell(v) for v in row) + " |")
    return "\n".join(lines)


def write_summary(path: Path, prereg_id: str, sections: list[tuple[str, str]]) -> str:
    """Write ``results_summary.md`` from ``(heading, body)`` pairs; return its SHA-256."""
    parts = [f"# Statistical results\n\nPreregistration id: `{prereg_id}`\n"]
    parts.extend(f"{heading}\n\n{body}\n" for heading, body in sections)
    atomic_write_text(path, "\n".join(parts))
    return sha256_file(path)
