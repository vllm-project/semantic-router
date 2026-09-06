"""Mandatory sandbox entry point for Dashboard-owned evaluation execution."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT))

from cli.evaluation.broker_client import install_worker_broker  # noqa: E402
from cli.evaluation.sandbox import (  # noqa: E402
    WorkerSandboxPolicy,
    apply_worker_sandbox,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one isolated evaluation worker")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--store", required=True, type=Path)
    parser.add_argument("--suite-store", required=True, type=Path)
    parser.add_argument("--cpu-seconds", required=True, type=int)
    parser.add_argument("--broker-request-fd", required=True, type=int)
    parser.add_argument("--broker-response-fd", required=True, type=int)
    parser.add_argument("--events-stdout", action="store_true")
    return parser


def _readable_roots() -> tuple[Path, ...]:
    candidates = {
        (PACKAGE_ROOT / "cli").resolve(),
        Path(sys.base_prefix).resolve(),
        Path(sys.prefix).resolve(),
    }
    for raw in (
        # Native Python extensions are imported only after isolation. Their
        # transitive shared objects must remain readable, while Landlock still
        # withholds execute/write access and seccomp forbids a new process.
        "/lib",
        "/lib64",
        "/usr/lib",
        "/usr/lib64",
        "/usr/local/lib",
        "/etc/ld.so.cache",
        "/etc/ssl/certs",
        "/etc/hosts",
        "/etc/nsswitch.conf",
        "/etc/resolv.conf",
        "/etc/localtime",
        "/usr/share/zoneinfo",
        "/dev/null",
        "/dev/urandom",
    ):
        path = Path(raw)
        if path.exists():
            resolved = path.resolve()
            if not resolved.is_symlink():
                candidates.add(resolved)
    return tuple(sorted(candidates, key=str))


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = args.manifest.resolve(strict=True)
    store = args.store.resolve(strict=True)
    suite_store = args.suite_store.resolve(strict=True)
    apply_worker_sandbox(
        WorkerSandboxPolicy(
            writable_root=store.parent,
            suite_store=suite_store,
            readable_roots=_readable_roots(),
            cpu_seconds=args.cpu_seconds,
        )
    )
    install_worker_broker(args.broker_request_fd, args.broker_response_fd)
    # Import the evaluation engine only after filesystem, process, and network
    # restrictions are active. The tiny launcher and sandbox modules are the
    # complete trusted pre-isolation surface.
    worker_main = importlib.import_module("cli.evaluation.worker").main

    worker_args = [
        "--manifest",
        str(manifest),
        "--store",
        str(store),
        "--suite-store",
        str(suite_store),
    ]
    if args.events_stdout:
        worker_args.append("--events-stdout")
    return worker_main(worker_args)


if __name__ == "__main__":
    os.environ.pop("PYTHONPATH", None)
    raise SystemExit(main())
