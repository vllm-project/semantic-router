"""Single-model/multi-model engine mode using the managed Apple environment."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence
from pathlib import Path

import click
import yaml

from cli import apple_runtime
from cli.apple_runtime_environment import validate_apple_host, validate_local_docker
from cli.commands.serve_options import explicit
from cli.commands.runtime_support import apply_container_runtime_override
from cli.container_images import get_container_image
from cli.container_runtime import get_container_runtime
from cli.deployment_backend import resolve_target
from cli.runtime_lifecycle_lock import acquire_runtime_lifecycle_lock
from cli.runtime_stack import resolve_runtime_stack
from cli.terminal import echo
from cli.validator_decision_model import MODEL_RUNTIME_PROFILE


ROUTER_OPTIONS = (
    "config",
    "replace_active_config",
    "envoy_image",
    "dashboard_image",
    "readonly",
    "minimal",
    "algorithm",
    "gateway",
    "namespace",
    "context",
    "chart_dir",
    "recipe_env_names",
    "startup_timeout",
    "data_parallel_size",
    "device_ids",
    "profile",
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


def _explicit(ctx: click.Context, name: str) -> bool:
    return name in ctx.params and explicit(ctx, name)


def _flag(name: str) -> str:
    return "--" + name.replace("_", "-").removesuffix("-names")


def run_apple_engine(
    ctx: click.Context,
    models: Sequence[str],
    *,
    models_file: str | None,
    revision: str | None,
    device: str | None,
    host: str | None,
    port: int | None,
    uds: str | None,
    profile: str | None,
    log_level: str | None,
    target: str | None,
    runtime: str | None,
    image: str | None,
    pull_policy: str,
) -> None:
    # Apple engine mode shares the target/platform/image lifecycle controls.
    allowed = {
        "platform",
        "target",
        "runtime",
        "container_runtime",
        "image",
        "router_image",
        "image_pull_policy",
    }
    for name in ROUTER_OPTIONS:
        if name not in allowed and _explicit(ctx, name):
            raise click.UsageError(f"{_flag(name)} applies to router mode", ctx=ctx)
    validate_apple_host(resolve_target(target), runtime or "docker")
    apply_container_runtime_override(runtime)
    validate_apple_host(resolve_target(target), get_container_runtime())
    if device not in (None, "mps"):
        raise click.UsageError("--platform apple requires --device mps", ctx=ctx)
    if _explicit(ctx, "envoy_image") or _explicit(ctx, "dashboard_image"):
        raise click.UsageError(
            "Apple engine mode does not start Envoy or Dashboard", ctx=ctx
        )
    if uds or host not in (None, "127.0.0.1", "localhost"):
        raise click.UsageError(
            "Apple engine mode exposes TCP on 127.0.0.1; --uds and other bind addresses are unsupported",
            ctx=ctx,
        )
    if port is not None and not 1 <= port <= 65535:
        raise click.UsageError("--port must be between 1 and 65535", ctx=ctx)
    if models and models_file:
        raise click.UsageError("give MODEL arguments or --models, not both", ctx=ctx)
    if revision and (models_file or len(models) != 1):
        raise click.UsageError("--revision applies to one MODEL", ctx=ctx)
    if profile and not MODEL_RUNTIME_PROFILE.fullmatch(profile):
        raise click.UsageError("invalid runtime profile name", ctx=ctx)
    entries = []
    if models_file:
        document = yaml.safe_load(Path(models_file).expanduser().read_text())
        entries = document.get("models") if isinstance(document, dict) else None
        if not isinstance(entries, list) or not entries:
            raise click.UsageError("--models requires a nonempty models list", ctx=ctx)
        if any(
            not isinstance(entry, dict)
            or entry.get("device", "mps") not in ("mps", "auto")
            for entry in entries
        ):
            raise click.UsageError(
                "Apple --models entries must select mps or auto", ctx=ctx
            )
        entries = [{**entry, "device": "mps"} for entry in entries]
    else:
        for value in models:
            artifact, separator, pinned = value.partition("@")
            if separator and revision and pinned != revision:
                raise click.UsageError(
                    "MODEL@REVISION conflicts with --revision", ctx=ctx
                )
            entry = {"model": artifact, "device": "mps", "profile": profile or "exact"}
            if pinned or revision:
                entry["revision"] = pinned or revision
            entries.append(entry)
    if any(
        not isinstance(entry.get("model"), str) or not entry["model"]
        for entry in entries
    ):
        raise click.UsageError(
            "each model entry requires a nonempty model name", ctx=ctx
        )
    # Resolve against the caller's cwd before hashing or handing entries to
    # the detached supervisor. Hub IDs remain unchanged when no directory exists.
    for entry in entries:
        local = Path(entry["model"]).expanduser()
        if local.is_dir():
            entry["model"] = str(local.resolve())
    key = hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()
    validate_local_docker()
    layout = resolve_runtime_stack()
    with acquire_runtime_lifecycle_lock(runtime="docker", stack_name=layout.stack_name):
        selected = get_container_image(
            image or os.getenv("VLLM_SR_ROUTER_IMAGE"),
            pull_policy=pull_policy,
            platform="apple",
        )
        state = apple_runtime.start_bridge(
            selected,
            engine_port=port or 8000,
            engine_mode=True,
            log_level=RUNTIME_LOG_LEVELS.get(log_level or "info", "info"),
        )
        try:
            answer = apple_runtime.request(
                state, "/engine", method="POST", body={"key": key, "models": entries}
            )
        except BaseException:
            apple_runtime.stop_bridge()
            raise
    echo(f"Apple MPS model runtime started at http://127.0.0.1:{answer['port']}")
    echo(
        "Model loading continues in the background; inspect /health and vllm-sr logs model-runtime."
    )
    echo("Stop with: vllm-sr stop")
