"""Tests for the contributor vLLM-SR journey helper."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
JOURNEY_SCRIPT = REPO_ROOT / "tools" / "agent" / "scripts" / "vllm_sr_journey.py"


def _run_journey(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(JOURNEY_SCRIPT), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_detect_env_returns_supported_envs():
    result = _run_journey("detect-env")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert "cpu-local" in payload["supported_envs"]
    assert payload["build_target"]
    assert payload["serve_command"]


def test_validate_journey_starter_example():
    config = REPO_ROOT / "config/recipes/examples/journey-starter/config.yaml"
    recipe_dir = REPO_ROOT / "config/recipes/examples/journey-starter"
    result = _run_journey(
        "validate",
        "--config",
        str(config),
        "--recipe-dir",
        str(recipe_dir),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["passed"] is True
