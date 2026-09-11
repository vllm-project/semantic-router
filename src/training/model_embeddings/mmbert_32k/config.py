"""Standard-library configuration helpers for the mmBERT-32K trainers."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
SUPPORTED_FAMILIES = frozenset({"foundation", "embedder", "reranker"})
_UNRESOLVED_ENV = re.compile(r"\$\{[A-Za-z_][A-Za-z0-9_]*\}")


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load and minimally validate a pinned training configuration."""
    path = Path(config_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"{path} schema_version must be {SCHEMA_VERSION}, "
            f"got {payload.get('schema_version')!r}"
        )
    family = payload.get("family")
    if family not in SUPPORTED_FAMILIES:
        raise ValueError(f"{path} has unsupported family {family!r}")
    return payload


def expand_environment(value: Any) -> Any:
    """Expand environment placeholders recursively and reject missing values."""
    if isinstance(value, str):
        expanded = os.path.expandvars(value)
        unresolved = _UNRESOLVED_ENV.findall(expanded)
        if unresolved:
            names = ", ".join(sorted(set(unresolved)))
            raise ValueError(f"required environment variable(s) not set: {names}")
        return expanded
    if isinstance(value, list):
        return [expand_environment(item) for item in value]
    if isinstance(value, dict):
        return {key: expand_environment(item) for key, item in value.items()}
    return value


def arguments_to_argv(arguments: dict[str, Any]) -> list[str]:
    """Convert a config argument mapping to deterministic argparse tokens."""
    argv: list[str] = []
    for key, raw_value in arguments.items():
        value = expand_environment(raw_value)
        option = f"--{key}"
        if value is None or value is False:
            continue
        if value is True:
            argv.append(option)
            continue
        if isinstance(value, list):
            value = ",".join(str(item) for item in value)
        argv.extend((option, str(value)))
    return argv
