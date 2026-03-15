"""Bootstrap helpers for zero-config startup."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from cli.consts import DEFAULT_LISTENER_PORT

SETUP_MODE_ENV = "VLLM_SR_SETUP_MODE"
DASHBOARD_SETUP_MODE_ENV = "DASHBOARD_SETUP_MODE"
SETUP_MODE_KEY = "setup"
DEFAULT_SETUP_LISTENER_PORT = DEFAULT_LISTENER_PORT
DEFAULT_OUTPUT_DIR_NAME = ".vllm-sr"


@dataclass
class BootstrapResult:
    """Result of ensuring bootstrap workspace state."""

    config_path: Path
    output_dir: Path
    setup_mode: bool
    created_config: bool = False
    created_output_dir: bool = False
    created_defaults: bool = False


def _setup_metadata() -> dict[str, Any]:
    return {
        "mode": True,
        "state": "bootstrap",
        "created_by": "vllm-sr serve",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def build_bootstrap_config(port: int = DEFAULT_SETUP_LISTENER_PORT) -> dict[str, Any]:
    """Build the minimal config needed for dashboard-first setup."""

    return {
        "version": "v0.3",
        "listeners": [
            {
                "name": f"http-{port}",
                "address": "0.0.0.0",
                "port": port,
                "timeout": "300s",
            }
        ],
        SETUP_MODE_KEY: _setup_metadata(),
    }


def _load_yaml_dict(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        return {}

    with open(config_path) as f:
        data = yaml.safe_load(f) or {}

    return data if isinstance(data, dict) else {}


def is_setup_mode_config(config_path: str | Path) -> bool:
    """Return True when config.yaml is a bootstrap/setup config."""

    data = _load_yaml_dict(Path(config_path))
    setup_data = data.get(SETUP_MODE_KEY)
    if isinstance(setup_data, dict):
        return bool(setup_data.get("mode"))
    return False


def ensure_bootstrap_workspace(
    config_path: str | Path,
    output_dir_name: str = DEFAULT_OUTPUT_DIR_NAME,
) -> BootstrapResult:
    """Ensure config and output directory exist for setup mode startup."""

    path = Path(config_path)
    output_dir = path.parent / output_dir_name
    created_config = False
    created_output_dir = False
    created_defaults = False

    path.parent.mkdir(parents=True, exist_ok=True)

    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        created_output_dir = True

    if not path.exists():
        bootstrap_config = build_bootstrap_config()
        with open(path, "w") as f:
            yaml.safe_dump(bootstrap_config, f, sort_keys=False)
        created_config = True

    return BootstrapResult(
        config_path=path,
        output_dir=output_dir,
        setup_mode=is_setup_mode_config(path),
        created_config=created_config,
        created_output_dir=created_output_dir,
        created_defaults=created_defaults,
    )
