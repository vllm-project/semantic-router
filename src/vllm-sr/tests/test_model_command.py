"""Tests for `vllm-sr model list` (#1931).

The command must:
- Be discoverable from the top-level CLI help.
- Print provider models + model cards for a valid config (exit 0).
- Mark the default model so users can spot the routing target at a glance.
- Never leak `api_key` / `api_key_env` values from backend refs - those are
  secrets and the whole point of this command is safe inspection.
- Fail with a clear error message when the config file is missing.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import yaml
from click.testing import CliRunner

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

main = importlib.import_module("cli.main").main
README_PATH = PROJECT_ROOT / "README.md"


_VALID_CONFIG = {
    "version": "v0.3",
    "listeners": [{"name": "http-8899", "address": "0.0.0.0", "port": 8899}],
    "providers": {
        "defaults": {"default_model": "gpt-4o-mini"},
        "models": [
            {
                "name": "gpt-4o-mini",
                "provider_model_id": "gpt-4o-mini",
                "backend_refs": [
                    {
                        "name": "openai-primary",
                        "provider": "openai",
                        "base_url": "https://api.openai.com/v1",
                        "protocol": "http",
                        "weight": 100,
                        "api_key": "SECRET_API_KEY_DO_NOT_LEAK",
                    }
                ],
            },
            {
                "name": "llama-3-8b",
                "backend_refs": [
                    {
                        "name": "vllm-local",
                        "provider": "openai",
                        "base_url": "http://localhost:8000/v1",
                        "protocol": "http",
                        "weight": 50,
                        "api_key_env": "SECRET_ENV_VAR_NAME_DO_NOT_LEAK",
                    }
                ],
            },
        ],
    },
    "routing": {
        "modelCards": [
            {
                "name": "gpt-4o-mini",
                "modality": "text",
                "capabilities": ["chat", "tools"],
                "param_size": "8B",
                "context_window_size": 128000,
            },
            {"name": "llama-3-8b", "modality": "text"},
        ],
        "decisions": [
            {
                "name": "default",
                "description": "fallback",
                "priority": 0,
                "rules": {"operator": "AND", "conditions": []},
                "modelRefs": [{"model": "gpt-4o-mini"}],
            }
        ],
    },
}


def _write_config(tmp_path: Path, data: dict | None = None) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(data or _VALID_CONFIG, sort_keys=False))
    return config_path


def test_model_list_registered_on_top_level_help():
    runner = CliRunner()

    result = runner.invoke(main, ["--help"])

    assert result.exit_code == 0
    assert "model" in result.output


def test_model_list_help_describes_subcommand():
    runner = CliRunner()

    result = runner.invoke(main, ["model", "--help"])

    assert result.exit_code == 0
    assert "list" in result.output


def test_readme_documents_model_list_usage():
    content = README_PATH.read_text(encoding="utf-8")

    assert "vllm-sr model list" in content
    assert "vllm-sr model list --config my-config.yaml" in content
    assert "Provider models" in content
    assert "Model cards" in content


def test_model_list_prints_provider_models_and_model_cards(tmp_path: Path, caplog):
    runner = CliRunner()
    config_path = _write_config(tmp_path)

    with caplog.at_level("INFO"):
        result = runner.invoke(main, ["model", "list", "--config", str(config_path)])

    assert result.exit_code == 0
    combined = "\n".join(record.message for record in caplog.records)

    # Provider models present, with the default flagged so users can tell at
    # a glance which one routing falls back to.
    assert "gpt-4o-mini" in combined
    assert "[default]" in combined
    assert "llama-3-8b" in combined
    # Backend identity (safe fields) is printed.
    assert "openai-primary" in combined
    assert "https://api.openai.com/v1" in combined
    # Model cards section is rendered.
    assert "Model cards" in combined
    assert "modality:" in combined


def test_model_list_never_leaks_api_key_or_env_var(tmp_path: Path, caplog):
    runner = CliRunner()
    config_data = yaml.safe_load(yaml.safe_dump(_VALID_CONFIG, sort_keys=False))
    config_data["providers"]["models"][0]["backend_refs"][0][
        "base_url"
    ] = "https://sk-url-secret@api.openai.com/v1?api_key=SECRET_QUERY_KEY&project=public"
    config_data["providers"]["models"][1]["backend_refs"][0][
        "base_url"
    ] = "http://localhost:8000/v1?token=SECRET_QUERY_TOKEN&tenant=dev"
    config_path = _write_config(tmp_path, config_data)

    with caplog.at_level("INFO"):
        result = runner.invoke(main, ["model", "list", "--config", str(config_path)])

    assert result.exit_code == 0
    combined = result.output + "\n".join(record.message for record in caplog.records)
    # The whole point of the command: inspection without credential exposure.
    assert "SECRET_API_KEY_DO_NOT_LEAK" not in combined
    assert "SECRET_ENV_VAR_NAME_DO_NOT_LEAK" not in combined
    assert "sk-url-secret" not in combined
    assert "SECRET_QUERY_KEY" not in combined
    assert "SECRET_QUERY_TOKEN" not in combined
    assert "https://***@api.openai.com/v1?api_key=***&project=public" in combined
    assert "http://localhost:8000/v1?token=***&tenant=dev" in combined


def test_model_list_tolerates_malformed_backend_url(tmp_path: Path, caplog):
    runner = CliRunner()
    config_data = yaml.safe_load(yaml.safe_dump(_VALID_CONFIG, sort_keys=False))
    config_data["providers"]["models"][0]["backend_refs"][0]["base_url"] = "http://[::1"
    config_path = _write_config(tmp_path, config_data)

    with caplog.at_level("INFO"):
        result = runner.invoke(main, ["model", "list", "--config", str(config_path)])

    assert result.exit_code == 0
    combined = "\n".join(record.message for record in caplog.records)
    assert "http://[::1" in combined


def test_model_list_reports_missing_config(tmp_path: Path, caplog):
    runner = CliRunner()
    missing = tmp_path / "does-not-exist.yaml"

    with caplog.at_level("ERROR"):
        result = runner.invoke(main, ["model", "list", "--config", str(missing)])

    assert result.exit_code == 1
    combined = "\n".join(record.message for record in caplog.records)
    assert "Config file not found" in combined
