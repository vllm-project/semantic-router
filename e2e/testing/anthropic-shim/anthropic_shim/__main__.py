"""Run the shim with ``python -m anthropic_shim``."""

from __future__ import annotations

import os

import uvicorn


def main() -> None:
    host = os.environ.get("ANTHROPIC_SHIM_HOST", "0.0.0.0")
    port = int(os.environ.get("ANTHROPIC_SHIM_PORT", "9080"))
    uvicorn.run("anthropic_shim.app:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
