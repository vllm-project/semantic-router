"""Support helpers for runtime-oriented CLI commands."""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile
from pathlib import Path

import yaml

from cli.bootstrap import (
    DASHBOARD_SETUP_MODE_ENV,
    SETUP_MODE_ENV,
    BootstrapResult,
)
from cli.utils import getLogger

log = getLogger(__name__)

_temp_dirs: list[str] = []

PASSTHROUGH_ENV_RULES = (
    ("HF_ENDPOINT", False),
    ("HF_TOKEN", True),
    ("HF_HOME", False),
    ("HF_HUB_CACHE", False),
    ("ANTHROPIC_API_KEY", True),
    ("OPENAI_API_KEY", True),
    ("OPENCLAW_BASE_IMAGE", False),
)

ALGORITHM_TYPES = [
    "static",
    "elo",
    "router_dc",
    "automix",
    "hybrid",
    "thompson",
    "gmtrouter",
    "router_r1",
]

ALGORITHM_HINTS = {
    "elo": "  Tip: Submit feedback via POST /api/v1/feedback",
    "router_dc": "  Tip: Ensure models have 'description' fields",
    "automix": "  Tip: Configure model 'pricing' for cost optimization",
    "hybrid": "  Tip: Configure weights in decision.algorithm.hybrid",
    "thompson": "  Tip: Balances exploration vs exploitation automatically",
    "gmtrouter": "  Tip: Learns user preferences via graph neural network",
    "router_r1": "  Tip: Requires Router-R1 server (see training docs)",
}


def _cleanup_temp_dirs() -> None:
    """Clean up temp directories created for transient config translation."""
    for temp_dir in _temp_dirs:
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        except Exception:
            pass


atexit.register(_cleanup_temp_dirs)


def inject_algorithm_into_config(config_path: Path, algorithm: str) -> Path:
    """Create a temporary config with algorithm.type injected into all decisions."""
    with config_path.open() as handle:
        config = yaml.safe_load(handle)

    decisions = config.get("decisions", [])
    for decision in decisions:
        if "algorithm" not in decision:
            decision["algorithm"] = {}
        decision["algorithm"]["type"] = algorithm
        log.info(
            f"  Injected algorithm.type={algorithm} into decision '{decision.get('name', 'unnamed')}'"
        )

    temp_dir = tempfile.mkdtemp(prefix="vllm-sr-")
    _temp_dirs.append(temp_dir)
    temp_config_path = Path(temp_dir) / "config-with-algorithm.yaml"
    with temp_config_path.open("w") as handle:
        yaml.dump(config, handle, default_flow_style=False, sort_keys=False)

    log.info(f"Created config with algorithm: {temp_config_path}")
    return temp_config_path


def log_bootstrap_result(requested_config: str, bootstrap: BootstrapResult) -> None:
    """Report any workspace files created during bootstrap."""
    if bootstrap.created_config:
        log.warning(f"Config file not found: {requested_config}")
        log.info(f"Created bootstrap setup config: {bootstrap.config_path}")
    if bootstrap.created_output_dir:
        log.info(f"Created bootstrap output directory: {bootstrap.output_dir}")
    if bootstrap.created_defaults:
        log.info(
            f"Copied router defaults reference: {bootstrap.output_dir / 'router-defaults.yaml'}"
        )


def validate_setup_mode_flags(setup_mode: bool, minimal: bool, readonly: bool) -> None:
    """Reject option combinations that conflict with dashboard-first bootstrap."""
    if setup_mode and minimal:
        raise ValueError(
            "Setup mode requires the dashboard. Remove --minimal or create a full config first."
        )
    if setup_mode and readonly:
        raise ValueError(
            "Setup mode requires dashboard editing. Remove --readonly or activate a config first."
        )


def append_passthrough_env_vars(env_vars: dict[str, str]) -> None:
    """Pass selected host environment variables into the container runtime."""
    for name, masked in PASSTHROUGH_ENV_RULES:
        value = os.environ.get(name)
        if value is None:
            continue
        env_vars[name] = value
        logged_value = "***" if masked else value
        log.info(f"Passing environment variable: {name}={logged_value}")


def apply_runtime_mode_env_vars(
    env_vars: dict[str, str],
    minimal: bool,
    readonly: bool,
    setup_mode: bool,
    platform: str | None,
) -> None:
    """Apply runtime-mode environment variables derived from CLI flags."""
    if minimal:
        env_vars["DISABLE_DASHBOARD"] = "true"
        log.info("Minimal mode: ENABLED (no dashboard, no observability)")
        if readonly:
            log.warning("--readonly is ignored in minimal mode (dashboard is disabled)")

    if readonly and not minimal:
        env_vars["DASHBOARD_READONLY"] = "true"
        log.info("Dashboard read-only mode: ENABLED")

    if setup_mode:
        env_vars[SETUP_MODE_ENV] = "true"
        env_vars[DASHBOARD_SETUP_MODE_ENV] = "true"
        log.info(
            "Setup mode: starting dashboard-first bootstrap flow without router/envoy"
        )

    if platform:
        env_vars["DASHBOARD_PLATFORM"] = platform
        env_vars["VLLM_SR_PLATFORM"] = platform
        log.info(f"Platform branding: {platform}")


def resolve_effective_config_path(
    config_path: Path, algorithm: str | None, setup_mode: bool
) -> Path:
    """Apply CLI algorithm override translation when appropriate."""
    if not algorithm:
        return config_path
    if setup_mode:
        log.warning(
            f"--algorithm={algorithm} ignored in setup mode until a runnable config is activated"
        )
        return config_path

    normalized = algorithm.lower()
    log.info(f"Model selection algorithm: {normalized}")
    effective_config = inject_algorithm_into_config(config_path, normalized)
    log_algorithm_hint(normalized)
    return effective_config


def log_algorithm_hint(algorithm: str) -> None:
    """Emit a targeted hint for the selected routing algorithm."""
    hint = ALGORITHM_HINTS.get(algorithm)
    if hint:
        log.info(hint)
