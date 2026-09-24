"""Dependency-light console entry point for a pinned Decision runtime."""

from __future__ import annotations

import argparse
import importlib
import ipaddress
from collections.abc import Sequence
from pathlib import Path

from .cpu_threads import configure_cpu_threads
from .runtime_factory import MAX_PENDING_ROWS, RuntimeLaunchConfig

_MAX_TCP_PORT = 65535
_GRAPH_PHYSICAL_BATCH = 8


def parse_launch_args(argv: Sequence[str] | None = None) -> RuntimeLaunchConfig:
    """Parse the exact host-side ``decision serve`` launch contract."""

    parser = argparse.ArgumentParser(prog="vllm-sr-decision-runtime")
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--backend", required=True, choices=("cpu", "rocm", "cuda"))
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--artifact-content-id", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-batch", type=int, required=True)
    parser.add_argument("--max-concurrency", type=int, required=True)
    parser.add_argument("--max-queue", type=int, required=True)
    parser.add_argument("--experimental-qwen-rocm-graph-b8", action="store_true")
    args = parser.parse_args(argv)
    try:
        ipaddress.ip_address(args.host)
    except ValueError as error:
        parser.error(f"--host must be an IP address: {error}")
    if not args.artifact_root.is_absolute():
        parser.error("--artifact-root must be absolute")
    if not 1 <= args.port <= _MAX_TCP_PORT:
        parser.error("--port must be between 1 and 65535")
    if not 1 <= args.max_batch <= MAX_PENDING_ROWS:
        parser.error(f"--max-batch must be between 1 and {MAX_PENDING_ROWS}")
    if args.max_concurrency < 1:
        parser.error("--max-concurrency must be positive")
    if args.max_queue < 0:
        parser.error("--max-queue must be non-negative")
    if args.experimental_qwen_rocm_graph_b8 and (
        args.backend != "rocm" or args.max_batch != _GRAPH_PHYSICAL_BATCH
    ):
        parser.error("--experimental-qwen-rocm-graph-b8 requires B8 ROCm")
    return RuntimeLaunchConfig(
        model=args.model,
        revision=args.revision,
        backend=args.backend,
        artifact_root=args.artifact_root,
        artifact_content_id=args.artifact_content_id,
        host=args.host,
        port=args.port,
        max_batch=args.max_batch,
        max_concurrency=args.max_concurrency,
        max_queue=args.max_queue,
        experimental_qwen_rocm_graph_b8=args.experimental_qwen_rocm_graph_b8,
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Start the real model server or explain the missing optional extra."""

    config = parse_launch_args(argv)
    if config.backend == "cpu":
        try:
            configure_cpu_threads()
        except ValueError as error:
            raise SystemExit(str(error)) from error
    try:
        importlib.import_module("fastapi")
        importlib.import_module("uvicorn")
    except ImportError as exc:  # pragma: no cover - exercised in a subprocess
        raise SystemExit(
            "Install the server dependencies with "
            "`pip install 'vllm-sr[decision-runtime]'`."
        ) from exc

    # Load the optional HTTP stack only after checking its error message.
    from .server import run_server  # noqa: PLC0415

    run_server(config)


if __name__ == "__main__":
    main()
