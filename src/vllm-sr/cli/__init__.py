"""vLLM Semantic Router CLI package."""

from __future__ import annotations

import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

_PYPROJECT_VERSION_PATTERN = re.compile(
    r'^version = "(?P<version>[^"]+)"$', re.MULTILINE
)


def _load_version() -> str:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    try:
        match = _PYPROJECT_VERSION_PATTERN.search(
            pyproject_path.read_text(encoding="utf-8")
        )
    except FileNotFoundError:
        match = None
    if match is not None:
        return match.group("version")

    try:
        return version("vllm-sr")
    except PackageNotFoundError:
        return "unknown"


__version__ = _load_version()
