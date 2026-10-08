"""The Router's decision model: global.model_catalog.system.decision_model.

The binding selects one declared model_runtime deployment for the Router's
judgment tasks. Model identity and runtime requirements belong to that resource.
The Router owns defaults; the CLI selects exact keys and validates the resource.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import yaml

from cli.consts import PLATFORM_AMD, PLATFORM_NVIDIA
from cli.model_runtime_defaults import effective_model_deployments_document

DEFAULT_DECISION_MODEL = "primary"
DECISION_MODEL_FIELD = "global.model_catalog.system.decision_model"


def canonical_decision_model(name: str | None) -> str:
    """Validate an exact deployment key, without model-family aliases."""
    if name is None:
        return DEFAULT_DECISION_MODEL
    if not isinstance(name, str) or not name or name.strip() != name:
        raise ValueError(
            "decision_model.deployment must be a non-empty, trimmed deployment key"
        )
    return name


def configured_decision_model(document: dict | None) -> str:
    """Return the authored deployment key, or the canonical default key."""
    system = (((document or {}).get("global") or {}).get("model_catalog") or {}).get(
        "system"
    ) or {}
    if "decision_model" not in system:
        return DEFAULT_DECISION_MODEL
    binding = system["decision_model"]
    if not isinstance(binding, dict) or set(binding) != {"deployment"}:
        raise ValueError(
            "decision_model must be {deployment: <declared deployment key>}"
        )
    return canonical_decision_model(binding["deployment"])


def decision_model_deployment(document: dict | None, name: str | None = None) -> dict:
    """Resolve a declaration using the Router-owned defaults, never an artifact alias."""
    key = (
        canonical_decision_model(name)
        if name is not None
        else configured_decision_model(document)
    )
    deployment = effective_model_deployments_document(document).get(key)
    if deployment is None:
        raise ValueError(f"decision_model.deployment: unknown deployment {key!r}")
    if deployment.get("provider") != "model_runtime":
        raise ValueError(
            f"decision_model.deployment {key!r} must use provider model_runtime"
        )
    return deployment


def set_decision_model(document: dict, name: str) -> bool:
    """Select one declared deployment; retain its resource and every other binding."""
    key = canonical_decision_model(name)
    decision_model_deployment(document, key)
    catalog = document.setdefault("global", {}).setdefault("model_catalog", {})
    system = catalog.setdefault("system", {})
    binding = {"deployment": key}
    if system.get("decision_model") == binding:
        return False
    system["decision_model"] = binding
    return True


def host_has_gpu(platform: str) -> bool:
    """Whether this host exposes the GPU devices a --platform passes through."""

    if platform == PLATFORM_AMD:
        return os.path.exists("/dev/kfd") and os.path.exists("/dev/dri")
    if platform == PLATFORM_NVIDIA:
        return (
            os.path.exists("/dev/nvidiactl") or shutil.which("nvidia-smi") is not None
        )
    return False


def gpu_requirement_error(
    deployment: dict, platform: str, *, local_host: bool
) -> str | None:
    """Validate a managed deployment's explicit accelerator against the host.

    Attached runtimes own their hardware. Automatic device selection and model
    package requirements remain the runtime's responsibility.
    """
    if deployment.get("endpoint"):
        return None
    device = (deployment.get("device") or "auto").split(":", 1)[0]
    expected_platform = {"rocm": PLATFORM_AMD, "cuda": PLATFORM_NVIDIA}.get(device)
    if expected_platform is None:
        return None
    if platform != expected_platform:
        return f"the decision deployment device {device} requires --platform {expected_platform}"
    if local_host and not host_has_gpu(platform):
        return f"the decision deployment device {device} requires a GPU, and this host shows no {platform.upper()} GPU devices"
    return None


def write_decision_model(config_path: Path, name: str) -> bool:
    """Write the decision model into the active config file as its next version.

    The Router records the document it starts with as a configuration version,
    so the previous version stays in the history for rollback. True when the
    file changed.
    """

    document = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not set_decision_model(document, name):
        return False
    temporary = config_path.with_name(f".{config_path.name}.decision-model.tmp")
    temporary.write_text(
        yaml.safe_dump(document, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )
    os.chmod(temporary, config_path.stat().st_mode & 0o777)
    os.replace(temporary, config_path)
    return True


def read_decision_model(config_path: Path) -> str | None:
    """The decision model an active config file names, or None when it cannot be read."""

    try:
        document = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        return configured_decision_model(document)
    except (OSError, ValueError, yaml.YAMLError):
        return None
