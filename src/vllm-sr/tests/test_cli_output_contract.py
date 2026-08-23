"""Machine-readable CLI output contracts."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

main = importlib.import_module("cli.main").main


_VALID_CONFIG = {
    "version": "v0.4",
    "listeners": [{"name": "http-8899", "address": "0.0.0.0", "port": 8899}],
    "models": [
        {
            "name": "test-model",
            "card": {
                "description": "Model used by CLI output contract tests.",
                "capabilities": ["chat"],
            },
            "connections": [
                {
                    "provider": "openai-compatible",
                    "endpoint": "http://host.docker.internal:8000/v1",
                    "model": "test-model",
                    "weight": "1",
                }
            ],
        }
    ],
    "recipes": [
        {
            "name": "default",
            "document": {
                "decisions": [
                    {
                        "name": "default-route",
                        "description": "fallback",
                        "priority": 100,
                        "rules": {"operator": "AND", "conditions": []},
                    }
                ]
            },
        }
    ],
    "entrypoints": [
        {
            "name": "vllm-sr/auto",
            "recipe": "default",
            "assignments": {"default-route": {"models": [{"model": "test-model"}]}},
        }
    ],
    "global": {
        "services": {
            "backend_dispatch": {
                "bind_address": "127.0.0.1",
                "port": 8180,
                "audience": "vllm-sr.backend-dispatch",
                "capability_ttl": "30s",
                "max_request_body_bytes": 64 << 20,
            },
            "backend_egress": {"policy_file": "/app/config/backend-egress-policy.yaml"},
        }
    },
}


def _write_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(_VALID_CONFIG, sort_keys=False),
        encoding="utf-8",
    )
    return config_path


def _run_cli_subprocess(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT_ROOT)
    environment["PYDANTIC_DISABLE_PLUGINS"] = "__all__"
    return subprocess.run(
        [sys.executable, "-m", "cli.main", *args],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("config_type", ["router", "envoy"])
def test_config_machine_output_is_pure_yaml_with_click_runner(
    tmp_path: Path, config_type: str
) -> None:
    config_path = _write_config(tmp_path)

    result = CliRunner().invoke(
        main,
        ["config", config_type, "--config", str(config_path)],
    )

    assert result.exit_code == 0, result.stderr
    assert result.stderr == ""
    assert yaml.safe_load(result.stdout)
    assert "Configuration parsed successfully" not in result.stdout
    assert "Configuration validation passed" not in result.stdout
    assert "Generating Envoy config" not in result.stdout
    assert "\x1b[" not in result.stdout
    if config_type == "router":
        assert result.stdout == config_path.read_text(encoding="utf-8")


@pytest.mark.parametrize("config_type", ["router", "envoy"])
def test_config_machine_output_is_pure_yaml_in_subprocess(
    tmp_path: Path, config_type: str
) -> None:
    config_path = _write_config(tmp_path)

    result = _run_cli_subprocess(
        tmp_path,
        "config",
        config_type,
        "--config",
        str(config_path),
    )

    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert yaml.safe_load(result.stdout)
    assert "Configuration parsed successfully" not in result.stdout
    assert "Configuration validation passed" not in result.stdout
    assert "Generating Envoy config" not in result.stdout
    assert "\x1b[" not in result.stdout
    if config_type == "router":
        assert result.stdout == config_path.read_text(encoding="utf-8")
