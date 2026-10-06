"""Engine mode of ``vllm-sr serve``: serve one or more models with the built-in model runtime."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Sequence

import click
from click.core import ParameterSource

from cli.validator_decision_model import MODEL_RUNTIME_PROFILE

ENGINE_OPTIONS = ("models_file", "revision", "device", "host", "port", "uds")
ROUTER_OPTIONS = (
    "config",
    "replace_active_config",
    "image",
    "router_image",
    "envoy_image",
    "dashboard_image",
    "image_pull_policy",
    "readonly",
    "minimal",
    "platform",
    "algorithm",
    "target",
    "namespace",
    "context",
    "chart_dir",
    "runtime",
    "recipe_env_names",
    "startup_timeout",
)
RUNTIME_LOG_LEVELS = {
    "debug": "debug",
    "info": "info",
    "warn": "warning",
    "warning": "warning",
    "error": "error",
    "dpanic": "error",
    "panic": "error",
    "fatal": "error",
}
INSTALL_HINT = (
    "Engine mode needs the vLLM Semantic Router model runtime (Python package "
    "vllm-srun), which ships in the router images and is not on PyPI. Install "
    "it from a repository checkout with `pip install ./src/model-runtime` "
    "(`make model-runtime-install` in a development checkout), or run "
    "`vllm-sr serve --config ...` for router mode."
)

ENGINE_HELP = """
\b
ENGINE MODE:

\b
  vllm-sr serve MODEL [MODEL ...] [--revision SHA] [--device DEVICE]
                [--host HOST] [--port N | --uds PATH] [--profile PROFILE]
  vllm-sr serve --models models.yaml [--host HOST] [--port N | --uds PATH]

Serves router models with the built-in model runtime instead of starting the
Router: decision models (POST /v1/decisions), classifiers (/v1/classify),
embedders (/v1/embeddings) and rerankers (/v1/rerank), with /v1/bundle,
/v1/models, /health and /metrics. MODEL is a Hub repository, a built-in model
name or a local package directory; MODEL@REVISION pins a revision, and several
MODELs share one process. --models lists models with their own name,
revision, device and profile. In engine mode --profile selects the numerics
profile: exact (default, identical to the released package), shared_context,
batching, max_speed, or one a plugin installs (vllm-srun plugins lists
them). Router mode starts managed runtimes itself for model_runtime
deployments in the config.

On Apple silicon, --platform apple uses a managed native MPS environment and
host-process lifecycle. It needs a local Docker image containing the matching
runtime, exposes loopback TCP, and supports status/logs model-runtime and stop.
MPS is experimental; missing per-model golden records remain unverified.
"""


def _explicit(ctx: click.Context, name: str) -> bool:
    source = ctx.get_parameter_source(name)
    return source not in (None, ParameterSource.DEFAULT)


def _flag(name: str) -> str:
    return "--" + name.replace("_", "-").removesuffix("-names")


def reject_engine_options(ctx: click.Context) -> None:
    """Router mode: engine options need a MODEL argument."""

    for name in ENGINE_OPTIONS:
        if _explicit(ctx, name):
            raise click.UsageError(
                f"{_flag(name)} applies to engine mode; pass a MODEL or --models",
                ctx=ctx,
            )


def _load_runtime_main() -> Callable[[Sequence[str]], int]:
    try:
        module = importlib.import_module("vllm_srun.cli")
    except ImportError as exc:
        raise click.ClickException(INSTALL_HINT) from exc
    return module.main


def engine_arguments(
    models: Sequence[str],
    *,
    models_file: str | None = None,
    revision: str | None,
    device: str | None,
    host: str | None,
    port: int | None,
    uds: str | None,
    profile: str | None,
    log_level: str | None,
) -> list[str]:
    arguments = ["serve", *models]
    if models_file:
        arguments += ["--models", models_file]
    arguments += ["--device", device or "auto", "--profile", profile or "exact"]
    if revision:
        arguments += ["--revision", revision]
    if uds:
        arguments += ["--uds", uds]
    else:
        if host:
            arguments += ["--host", host]
        if port is not None:
            arguments += ["--port", str(port)]
    if log_level:
        arguments += ["--log-level", RUNTIME_LOG_LEVELS[log_level.lower()]]
    return arguments


def run_engine_mode(
    ctx: click.Context,
    models: Sequence[str],
    *,
    models_file: str | None = None,
    revision: str | None,
    device: str | None,
    host: str | None,
    port: int | None,
    uds: str | None,
    profile: str | None,
    log_level: str | None,
) -> None:
    for name in ROUTER_OPTIONS:
        if _explicit(ctx, name):
            raise click.UsageError(
                f"{_flag(name)} applies to router mode; engine mode serves only models",
                ctx=ctx,
            )
    if profile is not None and not MODEL_RUNTIME_PROFILE.match(profile):
        raise click.UsageError(
            f"--profile {profile!r} is not a profile name, such as exact or batching",
            ctx=ctx,
        )
    if models and models_file:
        raise click.UsageError("give MODEL arguments or --models, not both", ctx=ctx)
    if revision and (models_file or len(models) != 1):
        raise click.UsageError(
            "--revision applies to a single MODEL; use MODEL@REVISION for several",
            ctx=ctx,
        )
    if uds and (host or port is not None):
        raise click.UsageError(
            "--uds cannot be combined with --host or --port", ctx=ctx
        )
    runtime_main = _load_runtime_main()
    code = runtime_main(
        engine_arguments(
            models,
            models_file=models_file,
            revision=revision,
            device=device,
            host=host,
            port=port,
            uds=uds,
            profile=profile,
            log_level=log_level,
        )
    )
    ctx.exit(code or 0)
