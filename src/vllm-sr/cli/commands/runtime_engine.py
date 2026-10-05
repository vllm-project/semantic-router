"""Engine mode of ``vllm-sr serve``: serve one model with the built-in model runtime."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Sequence

import click
from click.core import ParameterSource

ENGINE_PROFILES = ("exact", "shared_context", "batching", "max_speed")
ENGINE_OPTIONS = ("revision", "device", "host", "port", "uds")
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
    "vllm_sr_runtime). Install it from a repository checkout with "
    "`pip install ./src/model-runtime`, or run `vllm-sr serve --config ...` for "
    "router mode."
)

ENGINE_HELP = """
\b
ENGINE MODE:

\b
  vllm-sr serve MODEL [--revision SHA] [--device auto|cpu|cuda[:N]|rocm[:N]]
                      [--host HOST] [--port N | --uds PATH] [--profile PROFILE]

Serves one decision model with the built-in model runtime (POST /v1/decisions,
/v1/systemone, GET /v1/models, /health, /metrics) instead of starting the
Router. MODEL is a Hub repository, a built-in model name or a local package
directory. In engine mode --profile selects the numerics profile: exact
(default, identical to the released package), shared_context, batching or
max_speed. Router mode starts managed runtimes itself for model_runtime
deployments in the config.
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
                f"{_flag(name)} applies to engine mode; pass a MODEL to serve one model",
                ctx=ctx,
            )


def _load_runtime_main() -> Callable[[Sequence[str]], int]:
    try:
        module = importlib.import_module("vllm_sr_runtime.cli")
    except ImportError as exc:
        raise click.ClickException(INSTALL_HINT) from exc
    return module.main


def engine_arguments(
    model: str,
    *,
    revision: str | None,
    device: str | None,
    host: str | None,
    port: int | None,
    uds: str | None,
    profile: str | None,
    log_level: str | None,
) -> list[str]:
    arguments = ["serve", model, "--device", device or "auto"]
    arguments += ["--profile", profile or "exact"]
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
    model: str,
    *,
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
                f"{_flag(name)} applies to router mode; engine mode serves only MODEL",
                ctx=ctx,
            )
    if profile is not None and profile not in ENGINE_PROFILES:
        raise click.UsageError(
            f"--profile {profile!r} is not a runtime profile; use one of "
            + ", ".join(ENGINE_PROFILES),
            ctx=ctx,
        )
    if uds and (host or port is not None):
        raise click.UsageError(
            "--uds cannot be combined with --host or --port", ctx=ctx
        )
    runtime_main = _load_runtime_main()
    code = runtime_main(
        engine_arguments(
            model,
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
