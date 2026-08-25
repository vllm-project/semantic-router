import importlib
import json
import sys
from pathlib import Path

import yaml
from click.testing import CliRunner

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

main = importlib.import_module("cli.main").main


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def merge_source_payload() -> dict:
    return {
        "models": {
            "providers": {
                "remote": {
                    "baseUrl": "https://api.example.com/v1",
                    "apiKey": "${REMOTE_API_KEY}",
                    "api": "openai-completions",
                    "models": [
                        {"id": "gpt-4o-mini", "input": ["text"]},
                        {"id": "gpt-4.1", "input": ["text", "image"]},
                    ],
                }
            }
        }
    }


def merge_target_payload() -> dict:
    return {
        "version": "v0.3",
        "listeners": [
            {
                "name": "http-18889",
                "address": "0.0.0.0",
                "port": 18889,
                "timeout": "300s",
            }
        ],
        "providers": {
            "models": [
                {
                    "name": "existing-model",
                    "provider_model_id": "existing-model",
                    "backend_refs": [
                        {
                            "provider": "openai-compatible",
                            "base_url": "http://127.0.0.1:8000/v1",
                        }
                    ],
                }
            ]
        },
        "routing": {
            "modelCards": [{"name": "existing-model", "capabilities": ["chat"]}]
        },
        "recipes": [
            {
                "name": "existing",
                "routing": {
                    "signals": {
                        "keywords": [
                            {
                                "name": "keep",
                                "operator": "OR",
                                "method": "bm25",
                                "keywords": ["keep"],
                                "case_sensitive": False,
                                "bm25_threshold": 0.1,
                            }
                        ]
                    },
                    "decisions": [
                        {
                            "name": "existing-default",
                            "description": "keep me",
                            "priority": 100,
                            "rules": {"operator": "AND", "conditions": []},
                        }
                    ],
                },
            }
        ],
        "entrypoints": [
            {
                "model_names": ["vllm-sr/existing"],
                "recipe": "existing",
                "assignments": {
                    "existing-default": {"models": [{"model": "existing-model"}]}
                },
            }
        ],
    }


def test_cli_config_help_lists_import_command() -> None:
    runner = CliRunner()

    result = runner.invoke(main, ["config", "--help"])

    assert result.exit_code == 0
    assert "import" in result.output
    assert "--from openclaw" in result.output


def test_cli_config_import_openclaw_bootstraps_target_and_rewrites_source(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "openclaw.json"
    target_path = tmp_path / "config.yaml"
    write_json(
        source_path,
        {
            "models": {
                "providers": {
                    "vllm": {
                        "baseUrl": "http://10.0.0.7:8000/v1",
                        "apiKey": "not-needed",
                        "api": "openai-completions",
                        "models": [
                            {
                                "id": "qwen3-8b",
                                "name": "Qwen 3 8B",
                                "reasoning": True,
                                "input": ["text"],
                                "contextWindow": 131072,
                            }
                        ],
                    }
                }
            },
            "agents": {
                "defaults": {
                    "model": {
                        "primary": "vllm/qwen3-8b",
                    }
                }
            },
        },
    )

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "config",
            "import",
            "--from",
            "openclaw",
            "--source",
            str(source_path),
            "--target",
            str(target_path),
        ],
    )

    assert result.exit_code == 0
    assert_bootstrapped_import(source_path, target_path)


def assert_bootstrapped_import(source_path: Path, target_path: Path) -> None:
    """Verify the public current-v0.3 output and source rewrite together."""

    imported = yaml.safe_load(target_path.read_text(encoding="utf-8"))
    assert imported["version"] == "v0.3"
    assert imported["listeners"][0]["port"] == 8899
    assert set(imported) == {
        "version",
        "listeners",
        "providers",
        "routing",
        "recipes",
        "entrypoints",
        "global",
    }
    model = imported["providers"]["models"][0]
    assert model["name"] == "qwen3-8b"
    assert set(model) == {
        "name",
        "provider_model_id",
        "api_format",
        "backend_refs",
    }
    assert imported["routing"]["modelCards"][0] == {
        "name": "qwen3-8b",
        "capabilities": ["chat", "reasoning"],
        "description": "Qwen 3 8B",
        "context_window_size": 131072,
    }
    assert model["backend_refs"][0] == {
        "provider": "openai-compatible",
        "base_url": "http://10.0.0.7:8000/v1",
        "protocol": "http",
    }
    decision = imported["recipes"][0]["routing"]["decisions"][0]
    assignments = imported["entrypoints"][0]["assignments"]
    assert assignments[decision["name"]]["models"][0]["model"] == model["name"]
    assert not (
        {"id", "revision", "provider_catalog_revision", "backends"} & set(model)
    )

    rewritten_source = read_json(source_path)
    assert (
        rewritten_source["models"]["providers"]["vllm"]["baseUrl"]
        == "http://127.0.0.1:8899/v1"
    )
    assert rewritten_source["agents"]["defaults"]["model"]["primary"] == (
        "vllm/vllm-sr/imported/qwen3-8b"
    )
    assert (source_path.parent / "openclaw.json.bak").exists()
    assert not (target_path.parent / "config.yaml.bak").exists()


def test_cli_config_import_openclaw_merges_existing_target_and_preserves_sections(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "openclaw.json"
    target_path = tmp_path / "config.yaml"
    write_json(source_path, merge_source_payload())
    target_path.write_text(
        yaml.safe_dump(merge_target_payload(), sort_keys=False),
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "config",
            "import",
            "--from",
            "openclaw",
            "--source",
            str(source_path),
            "--target",
            str(target_path),
        ],
    )

    assert result.exit_code == 0

    merged = yaml.safe_load(target_path.read_text(encoding="utf-8"))
    assert merged["listeners"][0]["port"] == 18889
    assert "config_source" not in merged.get("global", {}).get("router", {})
    existing_recipe = next(
        recipe for recipe in merged["recipes"] if recipe["name"] == "existing"
    )
    assert existing_recipe["routing"]["signals"]["keywords"][0]["name"] == "keep"
    assert existing_recipe["routing"]["decisions"][0]["name"] == "existing-default"

    model_names = {model["name"] for model in merged["providers"]["models"]}
    assert model_names == {"existing-model", "gpt-4o-mini", "gpt-4.1"}

    imported_remote_model = next(
        model
        for model in merged["providers"]["models"]
        if model["name"] == "gpt-4o-mini"
    )
    assert imported_remote_model["backend_refs"][0] == {
        "provider": "openai-compatible",
        "base_url": "https://api.example.com/v1",
        "protocol": "https",
        "api_key_env": "REMOTE_API_KEY",
    }

    rewritten_source = read_json(source_path)
    assert (
        rewritten_source["models"]["providers"]["remote"]["baseUrl"]
        == "http://127.0.0.1:18889/v1"
    )
    assert (target_path.parent / "config.yaml.bak").exists()


def test_cli_config_import_openclaw_prefixes_duplicate_model_ids_and_rewrites_model_refs(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "openclaw.json"
    target_path = tmp_path / "config.yaml"
    write_json(
        source_path,
        {
            "models": {
                "providers": {
                    "openai": {
                        "baseUrl": "https://api.openai.com/v1",
                        "apiKey": "${OPENAI_API_KEY}",
                        "api": "openai-completions",
                        "models": [{"id": "shared-model", "input": ["text"]}],
                    },
                    "local": {
                        "baseUrl": "http://127.0.0.1:9000/v1",
                        "apiKey": "not-needed",
                        "api": "openai-completions",
                        "models": [{"id": "shared-model", "input": ["text", "image"]}],
                    },
                }
            },
            "agents": {
                "defaults": {
                    "model": {
                        "primary": "openai/shared-model",
                    }
                }
            },
        },
    )

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "config",
            "import",
            "--from",
            "openclaw",
            "--source",
            str(source_path),
            "--target",
            str(target_path),
        ],
    )

    assert result.exit_code == 0

    imported = yaml.safe_load(target_path.read_text(encoding="utf-8"))
    model_names = {model["name"] for model in imported["providers"]["models"]}
    assert model_names == {"openai/shared-model", "local/shared-model"}
    provider_model_ids = {
        model["provider_model_id"] for model in imported["providers"]["models"]
    }
    assert provider_model_ids == {"shared-model"}

    rewritten_source = read_json(source_path)
    openai_model = rewritten_source["models"]["providers"]["openai"]["models"][0]
    local_model = rewritten_source["models"]["providers"]["local"]["models"][0]
    assert openai_model["id"] == "vllm-sr/imported/openai/shared-model"
    assert local_model["id"] == "vllm-sr/imported/local/shared-model"
    assert (
        rewritten_source["agents"]["defaults"]["model"]["primary"]
        == "openai/vllm-sr/imported/openai/shared-model"
    )
    assert (
        rewritten_source["models"]["providers"]["openai"]["baseUrl"]
        == "http://127.0.0.1:8899/v1"
    )
    assert (
        rewritten_source["models"]["providers"]["local"]["baseUrl"]
        == "http://127.0.0.1:8899/v1"
    )


def test_cli_config_import_openclaw_uses_env_discovery_when_source_omitted(
    tmp_path: Path, monkeypatch
) -> None:
    source_path = tmp_path / "detected-openclaw.json"
    target_path = tmp_path / "config.yaml"
    write_json(
        source_path,
        {
            "models": {
                "providers": {
                    "vllm": {
                        "baseUrl": "http://127.0.0.1:8000/v1",
                        "api": "openai-completions",
                        "models": [{"id": "demo-model"}],
                    }
                }
            }
        },
    )
    monkeypatch.setenv("OPENCLAW_CONFIG_PATH", str(source_path))

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "config",
            "import",
            "--from",
            "openclaw",
            "--target",
            str(target_path),
        ],
    )

    assert result.exit_code == 0
    imported = yaml.safe_load(target_path.read_text(encoding="utf-8"))
    assert imported["providers"]["models"][0]["name"] == "demo-model"


def test_cli_config_import_openclaw_rejects_unsupported_provider_family_without_rewriting(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "openclaw.json"
    target_path = tmp_path / "config.yaml"
    original_source = {
        "models": {
            "providers": {
                "anthropic": {
                    "baseUrl": "https://api.anthropic.com",
                    "api": "anthropic-messages",
                    "models": [{"id": "claude-3-7-sonnet"}],
                }
            }
        }
    }
    write_json(source_path, original_source)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "config",
            "import",
            "--from",
            "openclaw",
            "--source",
            str(source_path),
            "--target",
            str(target_path),
        ],
    )
    assert result.exit_code != 0
    assert read_json(source_path) == original_source
    assert not target_path.exists()
    assert not (source_path.parent / "openclaw.json.bak").exists()


def test_cli_config_import_openclaw_rejects_custom_transport_headers(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "openclaw.json"
    target_path = tmp_path / "config.yaml"
    source = merge_source_payload()
    source["models"]["providers"]["remote"]["headers"] = {"X-Tenant": "integration"}
    write_json(source_path, source)

    result = CliRunner().invoke(
        main,
        [
            "config",
            "import",
            "--from",
            "openclaw",
            "--source",
            str(source_path),
            "--target",
            str(target_path),
        ],
    )

    assert result.exit_code != 0
    assert "Provider Integration" in result.output
    assert read_json(source_path) == source
    assert not target_path.exists()


def test_cli_config_import_openclaw_requires_force_when_backup_exists(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "openclaw.json"
    target_path = tmp_path / "config.yaml"
    write_json(
        source_path,
        {
            "models": {
                "providers": {
                    "vllm": {
                        "baseUrl": "http://127.0.0.1:8000/v1",
                        "api": "openai-completions",
                        "models": [{"id": "demo-model"}],
                    }
                }
            }
        },
    )
    target_path.write_text(
        yaml.safe_dump(merge_target_payload(), sort_keys=False), encoding="utf-8"
    )
    (source_path.parent / "openclaw.json.bak").write_text("{}", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "config",
            "import",
            "--from",
            "openclaw",
            "--source",
            str(source_path),
            "--target",
            str(target_path),
        ],
    )

    assert result.exit_code != 0

    force_result = runner.invoke(
        main,
        [
            "config",
            "import",
            "--from",
            "openclaw",
            "--source",
            str(source_path),
            "--target",
            str(target_path),
            "--force",
        ],
    )

    assert force_result.exit_code == 0
