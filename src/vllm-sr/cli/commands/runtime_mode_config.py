"""Instance routing and model selection expressed as canonical configuration."""

from __future__ import annotations

import re

import click

from cli.decision_model import configured_decision_model, decision_model_deployment
from cli.model_runtime_platform import check_device
from cli.validator_decision_model import (
    MODEL_RUNTIME_PROFILE,
    model_runtime_deployment_error,
    public_model_name,
)

MODE_HELP = """
\b
INSTANCE MODES:

\b
  vllm-sr serve --mode engine --model vllm-sr/Decision-2.0-Kai-0.6B
  vllm-sr serve --mode router

Both modes run the same frontend, Dashboard, model management and native APIs.
Engine mode disables recipe routing with global.router.enabled=false. Router
mode enables it again without discarding saved routing configuration. Omitting
--mode keeps the saved setting (routing is enabled by default).

--model configures the selected logical deployment; --decision-model selects
its deployment key. Multiple models and replicas belong in --config under
global.model_catalog.deployments. Listener addresses, ports, API keys and
System One model grants also belong in the canonical config.
"""


def validate_model_options(*, model, revision, device, runtime_profile, platform):
    """Validate model flags before bootstrap creates or changes any files."""
    if model is None:
        for flag, value in (
            ("revision", revision),
            ("device", device),
            ("runtime-profile", runtime_profile),
        ):
            if value is not None:
                raise click.UsageError(f"--{flag} requires --model")
        return None
    if not model or model.strip() != model:
        raise click.UsageError("--model must be a non-empty, trimmed artifact")
    if revision is not None and not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise click.UsageError("--revision must be a pinned 40-hex revision")
    if runtime_profile is not None and not MODEL_RUNTIME_PROFILE.fullmatch(
        runtime_profile
    ):
        raise click.UsageError("--runtime-profile must be a valid profile name")
    try:
        check_device(device or "auto", platform)
    except ValueError as error:
        raise click.UsageError(str(error)) from error
    result = {
        "provider": "model_runtime",
        "artifact": model,
        "device": device or "auto",
        "profile": runtime_profile or "exact",
    }
    if revision:
        result["revision"] = revision.lower()
    error = model_runtime_deployment_error(result)
    if error:
        raise click.UsageError(error)
    return result


def apply_instance_options(
    document, *, mode=None, model_options=None, decision_model=None
):
    """Mutate one candidate; existing routing and listener grants are preserved."""
    if mode is None and model_options is None:
        return False
    setup = (document.get("setup") or {}).get("mode") is True
    global_config = document.setdefault("global", {})
    changed = False
    if mode is not None:
        router = global_config.setdefault("router", {})
        enabled = mode == "router"
        changed = router.get("enabled", True) != enabled
        router["enabled"] = enabled
    if model_options is not None:
        key = decision_model or configured_decision_model(document)
        catalog = global_config.setdefault("model_catalog", {})
        deployments = catalog.setdefault("deployments", {})
        previous = deployments.get(key) or {}
        # Preserve the logical public identity and input policy when replacing
        # an artifact. Placement flags describe a fresh single managed worker.
        resource = {
            name: previous[name]
            for name in ("public_name", "input")
            if name in previous
        }
        resource.update(model_options)
        changed = deployments.get(key) != resource or changed
        deployments[key] = resource
        catalog.setdefault("system", {})["decision_model"] = {"deployment": key}
    if setup and mode == "engine":
        document.pop("setup", None)
        resource = decision_model_deployment(document, decision_model)
        public_name = public_model_name(resource)
        if not public_name:
            raise ValueError(
                "A local --model needs an explicit public_name in --config"
            )
        # Only the first-run bootstrap receives a grant. An authored or active
        # listener is never broadened by changing mode or model selection.
        document["listeners"] = [
            {
                "name": "http-8899",
                "address": "0.0.0.0",
                "port": 8899,
                "systemone": {"models": [public_name]},
            }
        ]
        changed = True
    return changed
