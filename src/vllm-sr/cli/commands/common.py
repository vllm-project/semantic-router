"""Shared helpers for Click command entrypoints."""

from __future__ import annotations

import sys
from collections.abc import Callable
from functools import wraps
from typing import Any


def exit_with_logged_error(
    log: Any, interrupt_message: str | None = None
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Wrap a CLI command and exit cleanly on logged failures."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except KeyboardInterrupt:
                if interrupt_message:
                    log.info(interrupt_message)
                sys.exit(0)
            except Exception as exc:
                log.error(f"Error: {exc}")
                sys.exit(1)

        return wrapper

    return decorator
