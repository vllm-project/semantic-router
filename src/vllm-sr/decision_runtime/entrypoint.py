"""Dependency-light console entry point for the optional HTTP runtime."""

from __future__ import annotations

import importlib


def main() -> None:
    """Start the server or explain which optional extra is required."""

    try:
        importlib.import_module("fastapi")
        importlib.import_module("uvicorn")
    except ImportError as exc:  # pragma: no cover - exercised in a subprocess
        raise SystemExit(
            "Install the server dependencies with "
            "`pip install 'vllm-sr[decision-runtime]'`."
        ) from exc

    from .server import main as server_main

    server_main()


if __name__ == "__main__":
    main()
