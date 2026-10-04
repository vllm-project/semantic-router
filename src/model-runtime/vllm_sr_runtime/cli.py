"""``vllm-sr-runtime`` command line; ``vllm-sr serve <model>`` delegates to ``serve`` here."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Sequence

from .accel.autotune import AUTOTUNE_ENV
from .config import DEFAULT_PORT, ServeConfig
from .plugins import registry
from .registry import builtin

PROFILES = ("exact", "shared_context", "batching", "max_speed")


def add_serve_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "model",
        help="Hub repository ID, built-in model name, or local package directory",
    )
    parser.add_argument(
        "--revision",
        help="40-hex commit; required for a Hub model that is not built in",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda[:N] or rocm[:N] (default: auto)",
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="TCP bind address (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"TCP port (default: {DEFAULT_PORT})",
    )
    parser.add_argument("--uds", help="serve on this Unix domain socket instead of TCP")
    parser.add_argument(
        "--profile",
        default="exact",
        choices=PROFILES,
        help="numerics profile (default: exact)",
    )
    parser.add_argument(
        "--engine", default="native", help="engine plugin (default: native)"
    )
    parser.add_argument(
        "--family", help="model family plugin (default: detected from the package)"
    )
    parser.add_argument(
        "--served-model-name",
        help="model ID reported by the API (default: the package name)",
    )
    parser.add_argument("--threads", type=int, help="CPU threads for the model")
    parser.add_argument(
        "--memory-budget",
        type=float,
        help="refuse to load a model estimated above this many GiB",
    )
    parser.add_argument(
        "--max-queue",
        type=int,
        default=256,
        help="queued requests before 429 (default: 256)",
    )
    parser.add_argument(
        "--max-queued-tokens",
        type=int,
        default=1 << 22,
        help="queued tokens before 429",
    )
    parser.add_argument(
        "--batch-window-ms",
        type=float,
        default=2.0,
        help="batching window of approximate profiles",
    )
    parser.add_argument(
        "--max-batch-tokens",
        type=int,
        default=65_536,
        help="padded tokens per batched forward",
    )
    parser.add_argument(
        "--max-request-bytes",
        type=int,
        default=8 << 20,
        help="largest accepted request body",
    )
    parser.add_argument(
        "--cache-dir", help="Hugging Face cache directory (default: HF_HUB_CACHE)"
    )
    parser.add_argument("--offline", action="store_true", help="use only cached files")
    parser.add_argument(
        "--base-path", help="local copy of an adapter package's pinned base"
    )
    parser.add_argument(
        "--accept-licence",
        action="append",
        default=[],
        help="accept a restricted licence identifier",
    )
    parser.add_argument(
        "--log-level", default="info", choices=("debug", "info", "warning", "error")
    )
    parser.add_argument(
        "--autotune-cache",
        default=os.environ.get(AUTOTUNE_ENV),
        help=(
            "record and reuse GPU kernel autotuning in this directory, so answers "
            f"repeat across processes (default: ${AUTOTUNE_ENV})"
        ),
    )


def config_from_args(args: argparse.Namespace) -> ServeConfig:
    return ServeConfig(
        model=args.model,
        revision=args.revision,
        device=args.device,
        host=args.host,
        port=args.port,
        uds=args.uds,
        profile=args.profile,
        engine=args.engine,
        family=args.family,
        served_model_name=args.served_model_name,
        threads=args.threads,
        memory_budget_gib=args.memory_budget,
        max_queue=args.max_queue,
        max_queued_tokens=args.max_queued_tokens,
        batch_window_ms=args.batch_window_ms,
        max_batch_tokens=args.max_batch_tokens,
        max_request_bytes=args.max_request_bytes,
        cache_dir=args.cache_dir,
        offline=args.offline,
        base_path=args.base_path,
        accept_licences=tuple(args.accept_licence),
        log_level=args.log_level,
        autotune_cache=args.autotune_cache,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vllm-sr-runtime", description="vLLM Semantic Router model runtime"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser(
        "serve", help="serve one model over HTTP (TCP or a Unix socket)"
    )
    add_serve_arguments(serve)
    commands.add_parser(
        "models", help="list the built-in models and their pinned revisions"
    )
    commands.add_parser(
        "plugins",
        help="list the installed families, engines, accelerators and profiles",
    )
    fixture = commands.add_parser(
        "fixture", help="write a tiny random-weight Decision 2.0 package (tests, E2E)"
    )
    fixture.add_argument("output", help="directory to create")
    fixture.add_argument("--backbone", default="qwen3_5", choices=("qwen3", "qwen3_5"))
    fixture.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "serve":
        logging.basicConfig(
            level=args.log_level.upper(),
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
        from .api.server import serve

        serve(config_from_args(args))
        return 0
    if args.command == "models":
        for model in builtin.all_models():
            print(
                f"{model.repo_id}@{model.revision}  {model.backbone}  {model.loaded_parameters:,} parameters"
            )
        return 0
    if args.command == "plugins":
        print(
            json.dumps(
                {
                    kind: sorted(entries)
                    for kind, entries in registry.discover().items()
                },
                indent=2,
            )
        )
        return 0
    if args.command == "fixture":
        from .testing.fixtures import write_package

        path = write_package(args.output, backbone=args.backbone, seed=args.seed)
        print(path)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
