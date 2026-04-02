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


def test_serve_help_describes_docker_only_runtime():
    runner = CliRunner()

    result = runner.invoke(main, ["serve", "--help"])

    assert result.exit_code == 0
    assert "Local Docker deployment" in result.output
    assert "Podman" not in result.output
    assert "--topology" not in result.output
    assert "--log-level" in result.output


def test_serve_passes_log_level_to_backend_env(monkeypatch, tmp_path: Path):
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

    class _StubBackend:
        def deploy(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        runtime_commands, "ensure_bootstrap_workspace", lambda _: bootstrap
    )
    monkeypatch.setattr(
        runtime_commands, "_build_backend", lambda *a, **kw: _StubBackend()
    )

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "serve",
            "--config",
            str(config_path),
            "--log-level",
            "debug",
            "--image-pull-policy",
            "never",
        ],
    )

    assert result.exit_code == 0
    assert captured["env_vars"]["SR_LOG_LEVEL"] == "debug"


def test_serve_keeps_observability_enabled_in_setup_mode(monkeypatch, tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "version": "v0.3",
                "listeners": [
                    {"name": "http-8899", "address": "0.0.0.0", "port": 8899}
                ],
                "routing": {"decisions": [{"name": "default"}]},
                "setup": {"mode": True},
            },
            sort_keys=False,
        )
    )
    bootstrap = BootstrapResult(
        config_path=config_path,
        output_dir=tmp_path / ".vllm-sr",
        setup_mode=True,
    )
    captured: dict[str, object] = {}

    class _StubBackend:
        def deploy(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        runtime_commands, "ensure_bootstrap_workspace", lambda _: bootstrap
    )
    monkeypatch.setattr(
        runtime_commands, "_build_backend", lambda *a, **kw: _StubBackend()
    )

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "serve",
            "--config",
            str(config_path),
            "--image-pull-policy",
            "never",
        ],
    )

    assert result.exit_code == 0
    assert captured["enable_observability"] is True


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

    class _StubBackend:
        def deploy(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        runtime_commands, "ensure_bootstrap_workspace", lambda _: bootstrap
    )
    monkeypatch.setattr(
        runtime_commands, "_build_backend", lambda *a, **kw: _StubBackend()
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
    effective_config = Path(captured["config_file"])
    with effective_config.open() as handle:
        translated = yaml.safe_load(handle)
    assert captured["source_config_file"] == str(config_path)
    assert captured["runtime_config_file"] == str(effective_config)
    assert (
        captured["env_vars"]["VLLM_SR_RUNTIME_CONFIG_PATH"]
        == "/app/.vllm-sr/runtime-config.yaml"
    )
    assert captured["env_vars"]["VLLM_SR_SOURCE_CONFIG_PATH"] == "/app/config.yaml"
    assert captured["env_vars"]["VLLM_SR_ALGORITHM_OVERRIDE"] == "elo"
    assert captured["source_config_file"] == str(config_path)
    assert captured["runtime_config_file"] == str(effective_config)
    assert translated["routing"]["decisions"][0]["algorithm"]["type"] == "elo"
    assert captured["pull_policy"] == "never"


def test_serve_passes_role_specific_images_to_backend(monkeypatch, tmp_path: Path):
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

    class _StubBackend:
        def deploy(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        runtime_commands, "ensure_bootstrap_workspace", lambda _: bootstrap
    )
    monkeypatch.setattr(
        runtime_commands, "_build_backend", lambda *a, **kw: _StubBackend()
    )

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "serve",
            "--config",
            str(config_path),
            "--router-image",
            "test/router:latest",
            "--envoy-image",
            "test/envoy:latest",
            "--dashboard-image",
            "test/dashboard:latest",
            "--image-pull-policy",
            "never",
        ],
    )

    assert result.exit_code == 0
    assert "topology" not in captured
    assert captured["router_image"] == "test/router:latest"
    assert captured["envoy_image"] == "test/envoy:latest"
    assert captured["dashboard_image"] == "test/dashboard:latest"
