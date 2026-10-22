"""Startup mode and explicit model overrides in one canonical declaration."""

from __future__ import annotations

from copy import deepcopy

import click

from cli.bootstrap import build_bootstrap_config
from cli.decision_model import configured_decision_model, decision_model_deployment
from cli.model_revision import builtin_releases, resolve_model_revision
from cli.utils import get_logger
from cli.validator_decision_model import engine_args_error, public_model_name

log = get_logger(__name__)

MODE_HELP = """
\b
INSTANCE MODES:

\b
  vllm-sr serve
  vllm-sr serve vllm-sr/Decision-2.0-Kai-0.6B --engine
  vllm-sr serve vllm-sr/Vela-2.0-4B --platform rocm -dp 2 --device-ids 0

Without --engine the instance starts in Router mode, including on restart.
--engine (-e) disables recipe routing and serves the model. On the docker
target only the Router container (the frontend, the native System One APIs
and the model runtime) starts; --dashboard, --observability or --no-minimal
add the rest. Without --config, and with no ./config.yaml, Engine mode keeps
each model's config and state in ~/.local/state/vllm-sr/engine/<model> and
prints a short summary (--verbose prints every detail). Saved routing
configuration is retained.

MODEL overrides the configured default judgment deployment's artifact.
Omitting MODEL preserves that deployment (a new configuration uses Vela 2.0
0.3B). Only explicit model, placement and engine options override saved
settings: --max-num-seqs, --max-num-batched-tokens, --batch-window-ms, --dtype
and --enable-shared-context set the deployment's engine arguments.
Backend LLMs, named deployments, listeners and API grants belong in --config.
"""


# The deployment's engine arguments that serve flags set.
ENGINE_ARGUMENTS = (
    "max_num_seqs",
    "max_num_batched_tokens",
    "batch_window_ms",
    "dtype",
    "enable_shared_context",
)


def removed_runtime_profile(ctx, param, value):
    """Refuse the removed --runtime-profile and name the engine options that replace it."""
    if value is not None:
        raise click.UsageError(
            "--runtime-profile was removed; batching is on by default; set "
            "--max-num-seqs, --max-num-batched-tokens, --batch-window-ms, "
            "--enable-shared-context or --dtype instead",
            ctx=ctx,
        )
    return value


def validate_model_options(
    *,
    model,
    revision,
    data_parallel_size=None,
    device_ids=None,
    engine_args=None,
):
    """Keep unspecified values absent so they cannot reset saved resources."""
    if model is not None and (not model or model.strip() != model):
        raise click.UsageError("MODEL must be a non-empty, trimmed artifact")
    if revision is not None and (not revision or revision.strip() != revision):
        raise click.UsageError("--revision must be a non-empty branch, tag or commit")
    engine_args = {
        name: value for name, value in (engine_args or {}).items() if value is not None
    }
    message = engine_args_error(engine_args)
    if message:
        flag, _, rest = message.partition(" ")
        raise click.UsageError(f"--{flag.replace('_', '-')} {rest}")
    ids = None
    if device_ids is not None:
        parts = device_ids.split(",")
        if not parts or any(not item.isdecimal() for item in parts):
            raise click.UsageError(
                "--device-ids must be comma-separated non-negative GPU indices"
            )
        ids = tuple(int(item) for item in parts)
        if len(ids) != len(set(ids)):
            raise click.UsageError(
                "--device-ids must be unique; use -dp to repeat workers on one GPU"
            )
    values = {
        "artifact": model,
        "revision": revision,
        "data_parallel_size": data_parallel_size,
        "device_ids": ids,
        **engine_args,
    }
    return {key: value for key, value in values.items() if value is not None} or None


def _warn_off_release(artifact, revision):
    """Name this CLI's release pin when a saved revision keeps a model off it.

    A saved revision outlives upgrades by design, including one an older CLI
    resolved from the Hub before it pinned the model.
    """
    pin = next(
        (
            pin
            for name, pin in builtin_releases().items()
            if name.lower() == artifact.lower()
        ),
        None,
    )
    if pin and revision != pin:
        log.warning(
            "%s is served at revision %s, saved in this configuration; this vllm-sr "
            "pins %s. Serve with --revision %s to switch.",
            artifact,
            revision[:12],
            pin[:12],
            pin[:12],
        )


def _placements(resource, options, platform, available_devices, authored):
    existing = deepcopy(
        resource.get("replicas")
        or [
            {
                key: resource[key]
                for key in ("device", "endpoint", "served_name")
                if key in resource
            }
        ]
    )
    count = options.get("data_parallel_size", len(existing))
    ids = options.get("device_ids")
    if any(item.get("endpoint") for item in existing):
        raise ValueError(
            "Attached deployments cannot be resized with serve flags; edit their endpoints in --config"
        )
    if ids is not None:
        if platform not in {"cuda", "rocm"}:
            raise ValueError("--device-ids requires a cuda or rocm execution platform")
        if len(ids) not in {1, count}:
            raise ValueError(
                "--device-ids needs one GPU index or one index per replica; set -dp explicitly"
            )
        ordinals = options["device_ordinals"]
        return [
            {
                "device": f"{platform}:{ordinals[0] if len(ids) == 1 else ordinals[index]}"
            }
            for index in range(count)
        ]
    if not authored and platform in {"cuda", "rocm"} and count > 1:
        if len(available_devices) < count:
            raise ValueError(
                f"-dp {count} requires {count} available GPUs; use --device-ids explicitly to share a GPU"
            )
        return [
            {"device": f"{platform}:{index}"} for index in available_devices[:count]
        ]
    return [deepcopy(existing[index % len(existing)]) for index in range(count)]


def apply_instance_options(
    document,
    *,
    engine=False,
    model_options=None,
    platform="cpu",
    available_devices=(),
):
    """Apply startup capabilities without widening grants or replacing saved policy."""
    before = deepcopy(document)
    setup = (document.get("setup") or {}).get("mode") is True
    global_config = document.get("global") or {}
    # Omission already means Router in the canonical contract. Do not turn an
    # ordinary restart into an authored-config change just to repeat defaults.
    if engine or (global_config.get("router") or {}).get("enabled") is False:
        global_config = document.setdefault("global", {})
        global_config.setdefault("router", {})["enabled"] = not engine
    key = configured_decision_model(document)
    catalog = global_config.get("model_catalog") or {}
    deployments = catalog.get("deployments") or {}
    authored = key in deployments
    if model_options or (not authored and platform in {"cuda", "rocm"}):
        catalog = document.setdefault("global", {}).setdefault("model_catalog", {})
        deployments = catalog.setdefault("deployments", {})
        defaults = decision_model_deployment(document)
        resource = (
            deepcopy(deployments[key])
            if authored
            else {
                "provider": "model_runtime",
                "artifact": defaults["artifact"],
                "device": "auto",
            }
        )
        options = model_options or {}
        if resource.get("endpoint") or any(
            item.get("endpoint") for item in resource.get("replicas") or []
        ):
            raise ValueError(
                "serve model overrides require a managed default deployment; attached endpoints belong in --config"
            )
        artifact = options.get("artifact")
        if artifact is not None and artifact != resource.get("artifact"):
            resource.pop("revision", None)
        for name in ("artifact", "revision", *ENGINE_ARGUMENTS):
            if name in options:
                resource[name] = options[name]
        if "data_parallel_size" in options or "device_ids" in options:
            resource["replicas"] = _placements(
                resource,
                options,
                platform,
                available_devices,
                authored and any(name in resource for name in ("device", "replicas")),
            )
            for name in ("device", "endpoint", "served_name"):
                resource.pop(name, None)
        resource.setdefault("provider", "model_runtime")
        revision = resolve_model_revision(
            resource["artifact"], resource.get("revision")
        )
        if revision is None:
            resource.pop("revision", None)
        else:
            resource["revision"] = revision
            if "revision" not in options:
                _warn_off_release(resource["artifact"], revision)
        deployments[key] = resource
        catalog.setdefault("system", {})["decision_model"] = {"deployment": key}
    if setup and engine:
        document.pop("setup", None)
        bootstrap_listeners = build_bootstrap_config()["listeners"]
        listeners = document.get("listeners")
        # A setup marker does not imply that listener policy is still pristine.
        # Only the unmodified bootstrap listener receives an automatic grant.
        if listeners is None or listeners == bootstrap_listeners:
            name = public_model_name(decision_model_deployment(document))
            if not name:
                raise ValueError(
                    "A local MODEL needs an explicit public_name in --config"
                )
            bootstrap_listeners[0]["systemone"] = {"models": [name]}
            document["listeners"] = bootstrap_listeners
    return document != before
