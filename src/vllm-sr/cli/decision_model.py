"""The Router's decision model: global.model_catalog.system.decision_model.

The decision model is the Vela model that answers the Router's own questions:
every built-in signal it covers, and every routing.signals.decision question
that names no deployment, in one call per request. The Router owns the
contract (pkg/config/decision_model.go); the CLI checks the name and the
hardware before it serves, and writes the choice into the active config.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import yaml

from cli.consts import PLATFORM_AMD, PLATFORM_NVIDIA

DECISION_MODELS = (
    "Vela-2.0-0.3B",
    "Vela-2.0-0.8B",
    "Vela-2.0-4B",
    "Vela-2.0-9B",
    "Vela-1.0",
)
DEFAULT_DECISION_MODEL = DECISION_MODELS[0]
VELA1_DECISION_MODEL = "Vela-1.0"
GPU_DECISION_MODELS = frozenset({"Vela-2.0-4B", "Vela-2.0-9B"})
DECISION_MODEL_FIELD = "global.model_catalog.system.decision_model"
_DECISION_2_FAMILIES = frozenset({"kai", "eos", "sol", "nox", "lux", "vega"})


def _choices() -> str:
    names = list(DECISION_MODELS)
    return f"{names[0]} (the default), {', '.join(names[1:-1])} or {names[-1]}"


def _is_decision_2(name: str) -> bool:
    lower = name.lower()
    if lower.startswith(("decision-2", "decision2")):
        return True
    words = lower.replace("_", "-").replace(" ", "-").replace(".", "-").split("-")
    return any(word in _DECISION_2_FAMILIES for word in words)


def canonical_decision_model(name: str | None) -> str:
    """The canonical name of a decision model, matched case-insensitively.

    Raises ValueError with the Router's wording for anything else.
    """

    trimmed = (name or "").strip() or DEFAULT_DECISION_MODEL
    for known in DECISION_MODELS:
        if known.lower() == trimmed.lower():
            return known
    base = trimmed.rsplit("/", 1)[-1]
    for known in DECISION_MODELS:
        if known.lower() == base.lower():
            raise ValueError(
                f"decision_model {name!r}: name the model {known}, without a "
                "repository or path"
            )
    if _is_decision_2(base):
        raise ValueError(
            f"decision_model {name!r} is a Decision 2.0 model. The built-in "
            "signals ask the questions a Vela model was trained on, so the "
            f"decision model is {_choices()}. A Decision 2.0 model answers your "
            "own questions: declare it as a model_runtime deployment under "
            "global.model_catalog.deployments and name that deployment in a "
            "routing.signals.decision question"
        )
    raise ValueError(
        f"decision_model {name!r} is not a decision model; choose {_choices()}"
    )


def requires_gpu(name: str) -> bool:
    return canonical_decision_model(name) in GPU_DECISION_MODELS


def configured_decision_model(document: dict | None) -> str:
    """The decision model a config document names, canonical; the default when unset."""

    system = (((document or {}).get("global") or {}).get("model_catalog") or {}).get(
        "system"
    ) or {}
    return canonical_decision_model(system.get("decision_model"))


def set_decision_model(document: dict, name: str) -> bool:
    """Write the decision model into a config document; True when it changed."""

    canonical = canonical_decision_model(name)
    catalog = document.setdefault("global", {}).setdefault("model_catalog", {})
    system = catalog.get("system")
    if not isinstance(system, dict):
        system = {}
        catalog["system"] = system
    if system.get("decision_model") == canonical:
        return False
    system["decision_model"] = canonical
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


def gpu_requirement_error(name: str, platform: str, *, local_host: bool) -> str | None:
    """Why a decision model cannot serve on a platform, or None.

    The 4B and 9B run on a GPU only: --platform cpu (the default) cannot serve
    them, nor can a local host without the platform's GPU devices.
    """

    canonical = canonical_decision_model(name)
    if canonical not in GPU_DECISION_MODELS:
        return None
    if platform not in (PLATFORM_AMD, PLATFORM_NVIDIA):
        return (
            f"the decision model {canonical} needs a GPU; serve it with "
            "--platform amd or --platform nvidia, or choose Vela-2.0-0.3B or "
            "Vela-2.0-0.8B, which run on a CPU"
        )
    if local_host and not host_has_gpu(platform):
        return (
            f"the decision model {canonical} needs a GPU, and this host shows no "
            f"{platform.upper()} GPU devices; serve it on a GPU host, or choose "
            "Vela-2.0-0.3B or Vela-2.0-0.8B, which run on a CPU"
        )
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
