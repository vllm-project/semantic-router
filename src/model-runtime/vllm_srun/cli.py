"""``vllm-srun`` command line; ``vllm-sr serve <model>`` delegates to ``serve`` here."""

from __future__ import annotations

import argparse
import json
import logging
import os
import pwd
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import fields

from .accel.autotune import AUTOTUNE_ENV
from .config import (
    DEFAULT_MAX_NUM_BATCHED_TOKENS,
    DEFAULT_MAX_NUM_SEQS,
    DTYPES,
    EngineArgs,
    ModelConfig,
    ServeConfig,
    load_models_file,
    spin_count,
    split_revision,
)
from .plugins import registry
from .registry import builtin

log = logging.getLogger("vllm_srun")
SERVE_DEFAULTS = ServeConfig()
ENGINE_DEFAULTS = EngineArgs()
MODEL_DEFAULTS = {field.name: field.default for field in fields(ModelConfig)}
# Flags that were removed, and what replaced them.
REMOVED_FLAGS = {
    "--profile": (
        "batching is on by default; set --max-num-seqs, --max-num-batched-tokens, "
        "--batch-window-ms, --enable-shared-context or --dtype instead"
    ),
    "--max-batch-tokens": "it is --max-num-batched-tokens",
}


class RemovedFlag(argparse.Action):
    """A removed flag: refuse it and name what replaced it."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: object,
        option_string: str | None = None,
    ) -> None:
        flag = option_string or self.dest
        parser.error(f"{flag} was removed; {REMOVED_FLAGS[flag]}")


def add_serve_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "model",
        nargs="*",
        help=(
            "Hub repository ID, built-in model name, or local package directory; "
            "several models share one process, and MODEL@REVISION pins a revision"
        ),
    )
    parser.add_argument(
        "--models",
        dest="models_file",
        help="YAML file listing the models to serve with their own name, revision, device and engine arguments",
    )
    parser.add_argument(
        "--revision",
        help="40-hex commit; required for a Hub model that is not built in (one MODEL only)",
    )
    parser.add_argument(
        "--device",
        default=MODEL_DEFAULTS["device"],
        help="auto, or an installed accelerator with an optional :N: cpu, cuda, npu, rocm, xpu and mps are built in (default: %(default)s)",
    )
    parser.add_argument(
        "--host",
        default=SERVE_DEFAULTS.host,
        help="TCP bind address (default: %(default)s)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=SERVE_DEFAULTS.port,
        help="TCP port (default: %(default)s)",
    )
    parser.add_argument("--uds", help="serve on this Unix domain socket instead of TCP")
    for flag in REMOVED_FLAGS:
        parser.add_argument(flag, nargs="?", action=RemovedFlag, help=argparse.SUPPRESS)
    parser.add_argument(
        "--engine",
        default=MODEL_DEFAULTS["engine"],
        help="engine plugin, or auto: the first that runs the model, native first (default: %(default)s)",
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
        default=SERVE_DEFAULTS.max_queue,
        help="queued requests before 429 (default: %(default)s)",
    )
    parser.add_argument(
        "--max-queued-tokens",
        type=int,
        default=SERVE_DEFAULTS.max_queued_tokens,
        help="queued tokens before 429 (default: %(default)s)",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=ENGINE_DEFAULTS.max_num_seqs,
        help=(
            "most rows one forward holds, across requests: one per question of a "
            "decision request, one per input or window otherwise; 1 runs every row "
            "alone (default: by the device's class, 256 on GPUs with 160 GB or more "
            "and fewer on smaller ones, adjusted by a short startup calibration; "
            f"{DEFAULT_MAX_NUM_SEQS} on the CPU)"
        ),
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=ENGINE_DEFAULTS.max_num_batched_tokens,
        help=(
            "tokens one forward may hold: the sum of its rows, each rounded up to 8, "
            "for models that pack merged rows into one sequence, else its rows times "
            "its longest row (default: by the device's class, 32768 on GPUs with "
            "160 GB or more and less on smaller ones, adjusted by a short startup "
            f"calibration; {DEFAULT_MAX_NUM_BATCHED_TOKENS} on the CPU)"
        ),
    )
    parser.add_argument(
        "--batch-window-ms",
        type=float,
        default=ENGINE_DEFAULTS.batch_window_ms,
        help=(
            "how long the scheduler waits after the first request to collect more, "
            "when --max-num-seqs is above 1 (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--dtype",
        choices=DTYPES,
        default=ENGINE_DEFAULTS.dtype,
        help=(
            "precision the backbone computes in; auto is the precision the model "
            "was published in (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--enable-shared-context",
        action=argparse.BooleanOptionalAction,
        default=ENGINE_DEFAULTS.enable_shared_context,
        help=(
            "compute the shared state of a multi-question decision request once "
            "(default: on where the model family's records show no accuracy loss, "
            "Decision 2.0; off for the others)"
        ),
    )
    parser.add_argument(
        "--media-workers",
        type=int,
        default=SERVE_DEFAULTS.media_workers,
        help=(
            "worker processes that decode and resize request images and videos; "
            "0 decodes them in the request threads (default: one per 16 CPUs the "
            "process may use, at most 4, when a served model reads media)"
        ),
    )
    parser.add_argument(
        "--max-request-bytes",
        type=int,
        default=SERVE_DEFAULTS.max_request_bytes,
        help="largest accepted request body in bytes (default: %(default)s)",
    )
    parser.add_argument(
        "--max-bundle-tasks",
        type=int,
        default=SERVE_DEFAULTS.max_bundle_tasks,
        help="most tasks one /v1/bundle request may carry (default: %(default)s)",
    )
    parser.add_argument(
        "--load-attempts",
        type=int,
        default=SERVE_DEFAULTS.load_attempts,
        help="loads of a model before it stays failed; package and golden failures are never retried (default: %(default)s)",
    )
    parser.add_argument(
        "--load-retry-seconds",
        type=float,
        default=SERVE_DEFAULTS.load_retry_seconds,
        help="wait before the first reload of a model that failed to load, doubling up to 300 s (default: %(default)s)",
    )
    parser.add_argument(
        "--result-cache-entries",
        type=int,
        default=SERVE_DEFAULTS.result_cache_entries,
        help=(
            "item results each model keeps by content hash, for families that allow it; "
            "0 disables (default: %(default)s)"
        ),
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
        "--log-level",
        default=SERVE_DEFAULTS.log_level,
        choices=("debug", "info", "warning", "error"),
    )
    parser.add_argument(
        "--autotune-cache",
        default=os.environ.get(AUTOTUNE_ENV),
        help=(
            "record and reuse GPU kernel autotuning and compiled kernels in this "
            "directory; processes that share a warm cache answer alike, and built-in "
            "models pin their kernel choices on gfx942 (MI300X, MI325X) "
            f"(default: ${AUTOTUNE_ENV})"
        ),
    )


def _engine_args(args: argparse.Namespace) -> EngineArgs:
    try:
        return EngineArgs(
            max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=args.max_num_batched_tokens,
            batch_window_ms=args.batch_window_ms,
            dtype=args.dtype,
            enable_shared_context=args.enable_shared_context,
        )
    except ValueError as exc:
        raise SystemExit(f"--{str(exc).replace('_', '-')}") from exc


def _models(
    args: argparse.Namespace, engine_args: EngineArgs
) -> tuple[ModelConfig, ...]:
    """The served models, from MODEL arguments or --models."""
    positional = list(args.model or [])
    if args.models_file:
        if positional:
            raise SystemExit("give either MODEL arguments or --models, not both")
        if args.revision or args.served_model_name:
            raise SystemExit(
                "--revision and --served-model-name apply to one MODEL argument"
            )
        try:
            return load_models_file(args.models_file, engine_args)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    if not positional:
        raise SystemExit("serve needs a MODEL argument or --models FILE")
    if len(positional) > 1 and (args.revision or args.served_model_name):
        raise SystemExit(
            "--revision and --served-model-name apply to one MODEL argument"
        )
    models = []
    for value in positional:
        model, revision = split_revision(value)
        if revision and args.revision and revision != args.revision:
            raise SystemExit(f"{value} conflicts with --revision {args.revision}")
        models.append(
            ModelConfig(
                model=model,
                revision=revision or args.revision,
                name=args.served_model_name,
                device=args.device,
                engine=args.engine,
                family=args.family,
                memory_budget_gib=args.memory_budget,
            )
        )
    return tuple(models)


def config_from_args(args: argparse.Namespace) -> ServeConfig:
    engine_args = _engine_args(args)
    models = _models(args, engine_args)
    if args.media_workers is not None and args.media_workers < 0:
        raise SystemExit("--media-workers must be a non-negative integer")
    return ServeConfig(
        models=models,
        host=args.host,
        port=args.port,
        uds=args.uds,
        max_bundle_tasks=args.max_bundle_tasks,
        load_attempts=args.load_attempts,
        load_retry_seconds=args.load_retry_seconds,
        result_cache_entries=args.result_cache_entries,
        threads=args.threads,
        memory_budget_gib=args.memory_budget,
        max_queue=args.max_queue,
        max_queued_tokens=args.max_queued_tokens,
        engine_args=engine_args,
        max_request_bytes=args.max_request_bytes,
        cache_dir=args.cache_dir,
        offline=args.offline,
        base_path=args.base_path,
        accept_licences=tuple(args.accept_licence),
        log_level=args.log_level,
        autotune_cache=args.autotune_cache,
        media_workers=args.media_workers,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vllm-srun", description="vLLM Semantic Router model runtime"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser(
        "serve", help="serve one or more models over HTTP (TCP or a Unix socket)"
    )
    add_serve_arguments(serve)
    commands.add_parser(
        "models", help="list the built-in models and their pinned revisions"
    )
    commands.add_parser(
        "plugins",
        help="list the installed families, engines and accelerators",
    )
    commands.add_parser(
        "devices",
        help="list, as JSON, the devices of this host and the one --device auto takes",
    )
    fixture = commands.add_parser(
        "fixture",
        help="write a tiny random-weight package of a family (tests, E2E)",
    )
    fixture.add_argument("output", help="directory to create")
    fixture.add_argument(
        "--family",
        default="decision2",
        help="family whose package format to write (default: decision2)",
    )
    fixture.add_argument(
        "--variant",
        help="family-specific variant, for example a backbone or a head kind",
    )
    fixture.add_argument("--seed", type=int, default=0)
    return parser


def default_identity() -> None:
    """Defaults for a uid with no passwd entry (OpenShift, the router images).

    PyTorch's compile caches name the user (``getpass.getuser``) and live under
    ``HOME``; without a passwd entry, ``USER`` or a writable ``HOME`` the first
    model load fails. Set variables are kept.
    """
    try:
        pwd.getpwuid(os.getuid())
        return
    except KeyError:
        pass
    scratch = tempfile.gettempdir()
    os.environ.setdefault("USER", "vllm-srun")
    home = os.environ.get("HOME")
    if not home or not os.access(home, os.W_OK):
        os.environ["HOME"] = scratch
    os.environ.setdefault(
        "TORCHINDUCTOR_CACHE_DIR", os.path.join(scratch, "torchinductor")
    )


def choose_spin_count(config: ServeConfig) -> None:
    """Set libgomp's spin count for the models this process serves (``spin_count``); a set value is kept.

    libgomp reads it once, when PyTorch loads.
    """
    value = spin_count(config.served_models())
    if value is None or "GOMP_SPINCOUNT" in os.environ:
        return
    if "torch" in sys.modules:
        log.warning(
            "PyTorch is already loaded, so GOMP_SPINCOUNT=%s takes effect only in a new process",
            value,
        )
    os.environ["GOMP_SPINCOUNT"] = value


def main(argv: Sequence[str] | None = None) -> int:
    default_identity()
    args = build_parser().parse_args(argv)
    if args.command == "serve":
        logging.basicConfig(
            level=args.log_level.upper(),
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
        config = config_from_args(args)
        choose_spin_count(config)
        from .api.server import serve

        serve(config)
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
    if args.command == "devices":
        from .devices import report

        print(json.dumps(report(), indent=2))
        return 0
    if args.command == "fixture":
        from .testing.fixtures import write_fixture

        print(
            write_fixture(
                args.output, family=args.family, variant=args.variant, seed=args.seed
            )
        )
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
