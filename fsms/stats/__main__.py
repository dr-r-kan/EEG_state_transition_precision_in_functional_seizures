"""Command line: ``python -m fsms.stats --release DIR --out DIR --prereg-id ID``."""
from __future__ import annotations

import argparse
import sys

from .classification import N_BOOTSTRAP
from .errors import StatisticsGateError
from .pipeline import run_statistics
from .release import REPRESENTATIONS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m fsms.stats", description=__doc__)
    parser.add_argument("--release", required=True, help="release bundle directory")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--prereg-id", required=True, help="preregistration identifier (must match the checkpoint)")
    parser.add_argument("--representations", nargs="+", default=list(REPRESENTATIONS), choices=list(REPRESENTATIONS))
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP, help="bootstrap replicates for the optimism-corrected AUC")
    args = parser.parse_args(argv)
    try:
        run_statistics(args.release, args.out, args.prereg_id, tuple(args.representations), n_bootstrap=args.n_bootstrap)
    except StatisticsGateError as error:
        print(f"statistics gate: {error}", file=sys.stderr)
        return 2
    print(f"results written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
