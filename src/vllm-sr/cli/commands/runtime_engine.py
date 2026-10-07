"""Engine mode of ``vllm-sr serve``: serve models with the model runtime in a container."""

from __future__ import annotations

from collections.abc import Sequence

import click

from cli.commands.runtime_support import apply_container_runtime_override
from cli.consts import DEFAULT_IMAGE_PULL_POLICY
from cli.container_images import get_container_image
from cli.container_runtime import get_container_runtime
from cli.engine_container import (
    RUNTIME_PORT,
    EngineRequest,
    check_device,
    serve_engine,
)
from cli.validator_decision_model import MODEL_RUNTIME_PROFILE

DEFAULT_HOST = "127.0.0.1"
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

ENGINE_HELP = """
\b
ENGINE MODE:

\b
  vllm-sr serve MODEL [MODEL ...] [--revision SHA] [--device DEVICE]
                [--platform PLATFORM] [--host HOST] [--port N]
                [--runtime-profile PROFILE]
  vllm-sr serve --models models.yaml [--host HOST] [--port N]

Serves router models with the built-in model runtime instead of starting the
Router: decision models (POST /v1/decisions), classifiers (/v1/classify),
embedders (/v1/embeddings) and rerankers (/v1/rerank), with /v1/bundle,
/v1/models, /health and /metrics. The runtime runs in the foreground in a
container from the platform's router image (vllm-sr, vllm-sr-rocm with
--platform amd, vllm-sr-cuda with --platform nvidia), and the host publishes
its port on --host and --port (default 127.0.0.1:8100). Ctrl-C stops it.

MODEL is a Hub repository, a built-in model name or a local package directory,
which the container reads through a read-only mount; MODEL@REVISION pins a
revision, and several MODELs share one process. --models lists models with
their own name, revision, device and profile. --device takes what the image
runs: cpu, rocm[:N] on amd, cuda[:N] on nvidia, or a plugin's accelerator in
an image that has the plugin. --runtime-profile selects the numerics profile:
exact (default, identical to the released package), shared_context, batching,
max_speed, or one a plugin installs. Downloads persist in
~/.cache/vllm-sr/models (VLLM_SR_ENGINE_CACHE_DIR moves it). Router mode starts
managed runtimes itself for model_runtime deployments in the config.
"""


def run_engine_mode(
    ctx: click.Context,
    models: Sequence[str],
    *,
    models_file: str | None,
    revision: str | None,
    device: str | None,
    host: str | None,
    port: int | None,
    runtime_profile: str | None,
    log_level: str | None,
    platform: str,
    image: str | None,
    image_pull_policy: str | None,
    container_runtime: str | None,
) -> None:
    """Validate engine mode's options, then serve the models in a container."""

    if runtime_profile is not None and not MODEL_RUNTIME_PROFILE.match(runtime_profile):
        raise click.UsageError(
            f"--runtime-profile {runtime_profile!r} is not a profile name, such "
            "as exact or batching",
            ctx=ctx,
        )
    if models and models_file:
        raise click.UsageError("give MODEL arguments or --models, not both", ctx=ctx)
    if revision and (models_file or len(models) != 1):
        raise click.UsageError(
            "--revision applies to a single MODEL; use MODEL@REVISION for several",
            ctx=ctx,
        )
    request = EngineRequest(
        models=tuple(models),
        models_file=models_file,
        revision=revision,
        device=device or "auto",
        runtime_profile=runtime_profile or "exact",
        host=host or DEFAULT_HOST,
        port=RUNTIME_PORT if port is None else port,
        log_level=RUNTIME_LOG_LEVELS[log_level.lower()] if log_level else None,
        platform=platform,
    )
    try:
        check_device(request.device, platform)
    except ValueError as error:
        raise click.UsageError(str(error), ctx=ctx) from error
    apply_container_runtime_override(container_runtime)
    runtime = get_container_runtime()
    selected_image = get_container_image(
        image=image,
        pull_policy=image_pull_policy or DEFAULT_IMAGE_PULL_POLICY,
        platform=platform,
    )
    ctx.exit(serve_engine(request, runtime=runtime, image=selected_image))
