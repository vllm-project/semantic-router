import importlib
import re
import sys
from pathlib import Path

import yaml
from click.testing import CliRunner

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BootstrapResult = importlib.import_module("cli.bootstrap").BootstrapResult
runtime_commands = importlib.import_module("cli.commands.runtime")
main = importlib.import_module("cli.main").main

_PYPROJECT_VERSION_PATTERN = re.compile(
    r'^version = "(?P<version>[^"]+)"$', re.MULTILINE
)


def _project_version() -> str:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    match = _PYPROJECT_VERSION_PATTERN.search(
        pyproject_path.read_text(encoding="utf-8")
    )
    assert match is not None
    return match.group("version")


def test_cli_help_lists_registered_commands():
    runner = CliRunner()

    result = runner.invoke(main, ["--help"])

    assert result.exit_code == 0
    for command_name in (
        "serve",
        "config",
        "validate",
        "status",
        "logs",
        "stop",
        "dashboard",
    ):
        assert command_name in result.output
    assert " init" not in result.output


def test_cli_version_matches_project_metadata():
    runner = CliRunner()

    result = runner.invoke(main, ["--version"])
    expected_version = _project_version()

    assert result.exit_code == 0
    assert result.output.strip() == f"vllm-sr version: {expected_version}"


def test_inject_algorithm_into_config_updates_all_decisions(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "version": "v0.3",
                "routing": {
                    "decisions": [
                        {"name": "fast"},
                        {"name": "slow", "algorithm": {"type": "static"}},
                    ]
                },
            },
            sort_keys=False,
        )
    )

    rewritten_path = runtime_commands.inject_algorithm_into_config(config_path, "elo")

    with rewritten_path.open() as handle:
        rewritten = yaml.safe_load(handle)

    assert rewritten_path != config_path
    assert [
        decision["algorithm"]["type"] for decision in rewritten["routing"]["decisions"]
    ] == ["elo", "elo"]


def test_serve_uses_algorithm_translated_config(monkeypatch, tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "version": "v0.3",
                "listeners": [
                    {"name": "http-8899", "address": "0.0.0.0", "port": 8899}
                ],
                "routing": {"decisions": [{"name": "default"}]},
            },
            sort_keys=False,
        )
    )
    bootstrap = BootstrapResult(
        config_path=config_path,
        output_dir=tmp_path / ".vllm-sr",
        setup_mode=False,
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        runtime_commands, "ensure_bootstrap_workspace", lambda _: bootstrap
    )
    monkeypatch.setattr(
        runtime_commands,
        "start_vllm_sr",
        lambda **kwargs: captured.update(kwargs),
    )

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "serve",
            "--config",
            str(config_path),
            "--algorithm",
            "elo",
            "--image-pull-policy",
            "never",
        ],
    )

    assert result.exit_code == 0
    with Path(captured["runtime_config_file"]).open() as handle:
        translated = yaml.safe_load(handle)
    assert Path(captured["config_file"]) == config_path
    assert (
        captured["env_vars"]["VLLM_SR_RUNTIME_CONFIG_PATH"]
        == "/app/.vllm-sr/runtime-config.yaml"
    )
    assert captured["env_vars"]["VLLM_SR_SOURCE_CONFIG_PATH"] == "/app/config.yaml"
    assert captured["env_vars"]["VLLM_SR_ALGORITHM_OVERRIDE"] == "elo"
    assert translated["routing"]["decisions"][0]["algorithm"]["type"] == "elo"
    assert captured["pull_policy"] == "never"
