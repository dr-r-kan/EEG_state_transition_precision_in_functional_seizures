"""Single entry point for the pipeline.

    python run_pipeline.py preflight [--config CFG] [--deep]            environment checks; reads no study data
    python run_pipeline.py run --config CFG --prereg-id ID [--through STAGE] [--force STAGE]
    python run_pipeline.py status --config CFG
    python run_pipeline.py fixture [--work-root DIR]                    build the synthetic-label TUEG test study

Stages, in order: clips_check, heppy, manifest, microstates, autonomic, metrics, labels, release, checkpoint, stats.

``run`` stops at the ``checkpoint`` stage by default: it prints the final number of participants and seizures
that enter the analyses and waits.  ``--prereg-id`` (the preregistration serial, e.g. ``abc12``) is required from
the checkpoint onwards.  Run again with ``--through stats`` to perform the statistics.  Reruns are idempotent.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fsms import pipeline  # noqa: E402
from fsms.checkpoint import CheckpointError  # noqa: E402
from fsms.config import Config, ConfigError  # noqa: E402
from fsms.release import LeakageError  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("preflight")
    p.add_argument("--config")
    p.add_argument("--deep", action="store_true")
    p = sub.add_parser("run")
    p.add_argument("--config", required=True)
    p.add_argument("--through", default="checkpoint", choices=pipeline.STAGE_ORDER)
    p.add_argument("--prereg-id", help="preregistration serial (5 characters, or its OSF URL); required from the checkpoint on")
    p.add_argument("--force", choices=pipeline.STAGE_ORDER, help="rerun this stage even if current")
    p = sub.add_parser("status")
    p.add_argument("--config", required=True)
    p = sub.add_parser("fixture")
    p.add_argument("--work-root", required=True)
    args = parser.parse_args(argv)

    try:
        if args.command == "preflight":
            config = Config.load(args.config) if args.config else None
            checks = pipeline.preflight(config, deep=args.deep)
            for name, ok, detail in checks:
                print(f"[{'ok' if ok else 'FAIL'}] {name}: {detail}")
            critical = {"python", "pinned packages", "vendored externals", "matlab on PATH"}
            return 0 if all(ok for name, ok, _ in checks if name in critical) else 1
        if args.command == "status":
            for name, state in pipeline.status(Config.load(args.config)).items():
                print(f"{name:12s} {state}")
            return 0
        if args.command == "fixture":
            return pipeline.run_logged([pipeline.PYTHON, "-B", "-m", "fsms.fixture.make_tueg_fixture", "--work-root",
                                        args.work_root], Path(args.work_root) / "fixture_build.log",
                                       Path(__file__).resolve().parent)
        report = pipeline.run(Config.load(args.config), through=args.through, prereg_id=args.prereg_id, force=args.force)
        print(report)
        if args.through == "checkpoint":
            print("\nCheckpoint written. Review the numbers above, then run again with --through stats.")
        return 0
    except (pipeline.PipelineError, ConfigError, CheckpointError, LeakageError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
