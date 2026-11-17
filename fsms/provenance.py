"""Content hashes of the vendored third-party code (HEPPy, VG-microstates).

The vendored trees are modified derivatives of the authors' own repositories, so the content hash of each tree
is the source of truth for exactly which code produced a result.  ``external/PROVENANCE.json`` records the
hashes; ``verify`` recomputes them.

    python -m fsms.provenance --write     # regenerate external/PROVENANCE.json
    python -m fsms.provenance --verify    # recompute hashes and compare
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from .atomic import atomic_write_json, read_json, sha256_normalised
from .config import REPO

PROVENANCE_PATH = REPO / "external" / "PROVENANCE.json"
EXCLUDED_DIRS = {"__pycache__", ".git", ".idea", ".ipynb_checkpoints", "outputs", ".pytest_cache"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".part"}

EXTERNALS = {
    "HEPPy": {"path": "external/HEPPy", "upstream": "https://github.com/dr-r-kan/HEPPy",
              "relationship": "Modified derivative: extended preprocessing (atomic saves, cardiac-artefact modes, "
                              "PyPREP fallbacks, ICLabel via ONNX) plus ECG peak correction and statistics helpers."},
    "VG-microstates": {"path": "external/VG-microstates", "upstream": "https://github.com/dr-r-kan/VG-microstates",
                       "relationship": "Modified derivative: variational Bayesian Gaussian-mixture microstates with "
                                       "a hierarchical (global, group, participant) fit, K selection by the free-energy "
                                       "elbow, template matching without forced classes and exact backfit."},
}


def tree_files(root: Path) -> list[Path]:
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in EXCLUDED_DIRS for part in relative.parts) or path.suffix in EXCLUDED_SUFFIXES:
            continue
        files.append(path)
    return files


def tree_hash(root: Path) -> tuple[str, int]:
    """SHA-256 over sorted ``relative_posix_path NUL file_sha256`` lines; returns (hash, file_count)."""
    digest = hashlib.sha256()
    files = tree_files(root)
    for path in files:
        digest.update(f"{path.relative_to(root).as_posix()}\0{sha256_normalised(path)}\n".encode("utf-8"))
    return digest.hexdigest(), len(files)


def compute() -> dict:
    record = {"schema": "fsms-external-provenance/1", "externals": {}}
    for name, meta in EXTERNALS.items():
        digest, count = tree_hash(REPO / meta["path"])
        record["externals"][name] = {**meta, "tree_sha256": digest, "file_count": count}
    template = REPO / "external" / "VG-microstates" / "MetaMaps_2023_06.set"
    record["metamaps_template_sha256"] = sha256_normalised(template)
    return record


def verify() -> list[str]:
    """Return a list of problems (empty when the vendored code matches its recorded hash)."""
    if not PROVENANCE_PATH.exists():
        return [f"missing {PROVENANCE_PATH}"]
    recorded = read_json(PROVENANCE_PATH)
    current = compute()
    problems = []
    for name in EXTERNALS:
        if recorded["externals"].get(name, {}).get("tree_sha256") != current["externals"][name]["tree_sha256"]:
            problems.append(f"vendored {name} differs from external/PROVENANCE.json")
    if recorded.get("metamaps_template_sha256") != current["metamaps_template_sha256"]:
        problems.append("MetaMaps template differs from external/PROVENANCE.json")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true")
    group.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    if args.write:
        atomic_write_json(PROVENANCE_PATH, compute())
        print(f"wrote {PROVENANCE_PATH}")
        return 0
    problems = verify()
    for problem in problems:
        print("PROVENANCE PROBLEM:", problem, file=sys.stderr)
    print("external provenance verified" if not problems else "external provenance FAILED")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
