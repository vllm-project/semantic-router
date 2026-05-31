"""Path helpers for runtime-oriented CLI commands."""

from __future__ import annotations

import os
from pathlib import Path

import yaml


def _runtime_config_output_dir(config_path: Path) -> Path:
    state_root = os.getenv("VLLM_SR_STATE_ROOT_DIR", "").strip()
    base_dir = Path(state_root).expanduser() if state_root else config_path.parent
    runtime_dir = base_dir / ".vllm-sr"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    return runtime_dir


def _runtime_config_output_path(source_config_path: Path) -> Path:
    stack_name = os.getenv("VLLM_SR_STACK_NAME", "").strip()
    filename = "runtime-config.yaml"
    if stack_name:
        filename = f"runtime-config.{stack_name}.yaml"
    return _runtime_config_output_dir(source_config_path) / filename


def _container_runtime_config_path(source_config_path: Path) -> str:
    return f"/app/.vllm-sr/{_runtime_config_output_path(source_config_path).name}"


def _container_source_config_path() -> str:
    return "/app/config.yaml"


def _write_runtime_config(source_config_path: Path, config: dict[str, object]) -> Path:
    runtime_config_path = _runtime_config_output_path(source_config_path)
    with runtime_config_path.open("w") as handle:
        yaml.dump(config, handle, default_flow_style=False, sort_keys=False)
    return runtime_config_path
