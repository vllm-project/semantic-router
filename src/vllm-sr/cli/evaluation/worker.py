"""Fixed process entry point used by the Dashboard evaluation backend."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cli.evaluation.canonical import canonical_json_bytes
from cli.evaluation.orchestrator import load_manifest, run_worker_evaluation
from cli.evaluation.store import WorkerArtifactStore
from cli.evaluation.suite_store import NormalizedSuiteStore
from cli.evaluation.worker_report import WorkerEvent


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a fixed vLLM-SR evaluation manifest"
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--store", required=True, type=Path)
    parser.add_argument("--suite-store", required=True, type=Path)
    parser.add_argument("--events-stdout", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    def emit(event: WorkerEvent) -> None:
        if not args.events_stdout:
            return
        sys.stdout.buffer.write(canonical_json_bytes(event) + b"\n")
        sys.stdout.buffer.flush()

    try:
        manifest = load_manifest(args.manifest)
        # Dashboard gives each worker a unique staging tree. Its explicit store
        # owns only process-local coordination, so no control file enters the
        # evidence bundle.
        store = WorkerArtifactStore(args.store)
        suite_store = NormalizedSuiteStore.open_read_only(args.suite_store)
        run_worker_evaluation(
            manifest,
            store,
            suite_store=suite_store,
            event_sink=emit,
        )
    # The process boundary must turn every worker failure into a non-zero exit;
    # the Dashboard accepts only the fixed stdout protocol.
    except Exception as exc:
        print(f"evaluation worker failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
