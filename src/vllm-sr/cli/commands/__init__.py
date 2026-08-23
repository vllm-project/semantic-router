"""CLI commands for vLLM Semantic Router.

Command modules are loaded lazily so importing one runtime helper does not pull
the complete interactive CLI dependency graph into lightweight router images.
"""

from importlib import import_module
from typing import Any

__all__ = [
    "config_command",
]

_LAZY_EXPORTS = {
    "config_command": (".config", "config_command"),
}


def __getattr__(name: str) -> Any:
    """Load legacy package-level command exports only when requested."""
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    return getattr(import_module(module_name, __name__), attribute_name)
