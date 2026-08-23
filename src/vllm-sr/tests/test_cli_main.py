import importlib
import re
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BootstrapResult = importlib.import_module("cli.bootstrap").BootstrapResult
runtime_commands = importlib.import_module("cli.commands.runtime")
main = importlib.import_module("cli.main").main
compiled_bootstrap_lock_module = importlib.import_module("cli.runtime_config_lock")

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


def _runtime_config(*, credential_env: str | None = None) -> dict:
    services: dict[str, object] = {
        "backend_dispatch": {
            "bind_address": "0.0.0.0",
            "port": 8180,
            "audience": "vllm-sr.backend-dispatch",
            "capability_ttl": "30s",
            "max_request_body_bytes": 64 << 20,
        }
    }
    if credential_env:
        services["backend_credentials"] = {
            "custom": {
                "credential_adapter_id": "bearer",
                "secret_env": credential_env,
            }
        }
    return {
        "version": "v0.4",
        "listeners": [{"name": "http-8899", "address": "0.0.0.0", "port": 8899}],
        "models": [],
        "recipes": [],
        "entrypoints": [],
        "global": {"services": services},
    }


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
        "chat",
    ):
        assert command_name in result.output
    assert "  model " not in result.output
    assert " init" not in result.output


def test_runtime_cli_import_graph_excludes_offline_v03_converter():
    script = """
import sys
import cli.main

for module_name in (
    "cli.config_upgrade_v03",
    "cli.config_upgrade_v03_models",
    "cli.config_upgrade_v03_routing",
    "cli.config_upgrade_v03_support",
):
    if module_name in sys.modules:
        raise SystemExit(f"runtime imported offline converter module: {module_name}")
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        check=True,
    )


def test_cli_version_matches_project_metadata():
    runner = CliRunner()

    result = runner.invoke(main, ["--version"])
    expected_version = _project_version()

    assert result.exit_code == 0
    assert result.output.strip() == f"vllm-sr version: {expected_version}"


def test_serve_compiles_bootstrap_under_custom_host_state_root(
    monkeypatch, tmp_path: Path
):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    config_path = source_dir / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            _runtime_config(),
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    state_root = tmp_path / "host-state"
    bootstrap = BootstrapResult(
        config_path=config_path,
        output_dir=source_dir / ".vllm-sr",
    )
    captured: dict[str, object] = {}

    class _StubBackend:
        def deploy(self, **kwargs):
            owned_lock = kwargs["compiled_bootstrap_lock"]
            assert owned_lock._closed is False
            with pytest.raises(
                compiled_bootstrap_lock_module.CompiledBootstrapLockError,
                match="operation is in progress",
            ):
                compiled_bootstrap_lock_module.acquire_compiled_bootstrap_lock(
                    compiled_bootstrap_path=kwargs["compiled_bootstrap_file"],
                    state_root_dir=state_root,
                    stack_name="vllm-sr",
                    timeout_seconds=0,
                )
            captured.update(kwargs)

    monkeypatch.setenv("VLLM_SR_STATE_ROOT_DIR", str(state_root))
    monkeypatch.setattr(
        runtime_commands, "ensure_bootstrap_workspace", lambda *_a, **_kw: bootstrap
    )
    monkeypatch.setattr(
        runtime_commands, "_build_backend", lambda *a, **kw: _StubBackend()
    )

    result = CliRunner().invoke(
        main,
        [
            "serve",
            "--image-pull-policy",
            "never",
        ],
    )

    assert result.exit_code == 0, result.output
    expected = state_root / ".vllm-sr" / "compiled-bootstrap.yaml"
    assert Path(captured["config_file"]) == expected
    assert Path(captured["compiled_bootstrap_file"]) == expected
    assert captured["compiled_bootstrap_lock"]._closed is True
    assert expected.is_file()
    assert not (source_dir / ".vllm-sr" / "compiled-bootstrap.yaml").exists()
    assert "VLLM_SR_STATE_ROOT_DIR" not in captured["env_vars"]


def test_k8s_serve_keeps_non_persistent_source_config_flow(monkeypatch, tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            _runtime_config(),
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    bootstrap = BootstrapResult(
        config_path=config_path,
        output_dir=tmp_path / ".vllm-sr",
    )
    captured: dict[str, object] = {}

    class _StubBackend:
        def deploy(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        runtime_commands, "ensure_bootstrap_workspace", lambda *_a, **_kw: bootstrap
    )
    monkeypatch.setattr(
        runtime_commands, "_build_backend", lambda *a, **kw: _StubBackend()
    )
    monkeypatch.setattr(
        runtime_commands,
        "materialize_compiled_bootstrap",
        lambda *_a, **_kw: (_ for _ in ()).throw(
            AssertionError("Kubernetes must not materialize persistent local state")
        ),
    )
    monkeypatch.setattr(runtime_commands, "DEFAULT_SERVE_CONFIG", str(config_path))

    result = CliRunner().invoke(
        main,
        [
            "serve",
            "--config",
            str(config_path),
            "--target",
            "k8s",
        ],
    )

    assert result.exit_code == 0, result.output
    effective = Path(captured["config_file"])
    assert effective == config_path
    assert captured["config_document"] == yaml.safe_load(
        config_path.read_text(encoding="utf-8")
    )
    assert "VLLM_SR_SOURCE_CONFIG_PATH" not in captured["env_vars"]
    assert "VLLM_SR_RUNTIME_CONFIG_PATH" not in captured["env_vars"]
    assert not (tmp_path / ".vllm-sr").exists()


def test_serve_help_describes_docker_only_runtime():
    runner = CliRunner()

    result = runner.invoke(main, ["serve", "--help"])

    assert result.exit_code == 0
    assert "Local Docker deployment" in result.output
    assert "Podman" not in result.output
    assert "--topology" not in result.output
    assert "--log-level" in result.output
    assert "--algorithm" not in result.output
    assert "--catalog-version" not in result.output
    assert "--config PATH" in result.output
    assert "[MODEL]" not in result.output
    assert "session_aware" not in result.output
    assert "--sim-image" in result.output
    assert "--recipe-env" not in result.output
    assert "router_r1" not in result.output
    assert "thompson" not in result.output


def test_legacy_serve_command_module_is_absent():
    assert find_spec("cli.commands.serve") is None


@pytest.mark.parametrize(
    "arguments",
    (
        ["model"],
        ["serve", "vllm-sr/mom-v1-blend"],
        ["serve", "--catalog-version", "latest"],
        ["serve", "--algorithm", "static"],
        ["serve", "--recipe-env", "CUSTOM_API_KEY"],
    ),
)
def test_removed_model_and_serve_policy_surfaces_are_rejected(arguments):
    result = CliRunner().invoke(main, arguments)

    assert result.exit_code == 2
    assert (
        "No such command" in result.output
        or "No such option" in result.output
        or "unexpected extra argument" in result.output.lower()
    )


def test_serve_rejects_a_missing_explicit_bootstrap_manifest():
    result = CliRunner().invoke(main, ["serve", "--config", "missing-bootstrap.yaml"])

    assert result.exit_code == 1
    assert "bootstrap config file does not exist" in result.output


def test_serve_rejects_generated_private_state_as_config(tmp_path: Path):
    generated = tmp_path / ".vllm-sr" / "compiled-bootstrap.yaml"
    generated.parent.mkdir()
    generated.write_text("version: v0.4\n", encoding="utf-8")

    result = CliRunner().invoke(main, ["serve", "--config", str(generated)])

    assert result.exit_code != 0
    assert "must select an immutable user bootstrap" in result.output


def test_plain_serve_forwards_source_config_credentials_without_recipe_state(
    monkeypatch, tmp_path: Path
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            _runtime_config(credential_env="CUSTOM_API_KEY"),
            sort_keys=False,
        )
    )
    bootstrap = BootstrapResult(
        config_path=config_path,
        output_dir=tmp_path / ".vllm-sr",
    )
    captured: dict[str, object] = {}

    class _StubBackend:
        def deploy(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        runtime_commands, "ensure_bootstrap_workspace", lambda *_a, **_kw: bootstrap
    )
    monkeypatch.setattr(
        runtime_commands, "_build_backend", lambda *a, **kw: _StubBackend()
    )
    monkeypatch.setenv("CUSTOM_API_KEY", "never-print-this-value")

    result = CliRunner().invoke(main, ["serve"])
    assert result.exit_code == 0, result.output
    assert captured["env_vars"]["CUSTOM_API_KEY"] == "never-print-this-value"
    assert "never-print-this-value" not in result.output


def test_serve_passes_log_level_to_backend_env(monkeypatch, tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            _runtime_config(),
            sort_keys=False,
        )
    )
    bootstrap = BootstrapResult(
        config_path=config_path,
        output_dir=tmp_path / ".vllm-sr",
    )
    captured: dict[str, object] = {}

    class _StubBackend:
        def deploy(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        runtime_commands, "ensure_bootstrap_workspace", lambda *_a, **_kw: bootstrap
    )
    monkeypatch.setattr(
        runtime_commands, "_build_backend", lambda *a, **kw: _StubBackend()
    )

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "serve",
            "--log-level",
            "debug",
            "--image-pull-policy",
            "never",
        ],
    )

    assert result.exit_code == 0
    assert captured["env_vars"]["SR_LOG_LEVEL"] == "debug"


def test_serve_does_not_start_observability_by_default(monkeypatch, tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            _runtime_config(),
            sort_keys=False,
        )
    )
    bootstrap = BootstrapResult(
        config_path=config_path,
        output_dir=tmp_path / ".vllm-sr",
    )
    captured: dict[str, object] = {}

    class _StubBackend:
        def deploy(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        runtime_commands, "ensure_bootstrap_workspace", lambda *_a, **_kw: bootstrap
    )
    monkeypatch.setattr(
        runtime_commands, "_build_backend", lambda *a, **kw: _StubBackend()
    )

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "serve",
            "--image-pull-policy",
            "never",
        ],
    )

    assert result.exit_code == 0
    assert captured["enable_observability"] is False


def test_serve_starts_observability_only_when_requested(monkeypatch, tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            _runtime_config(),
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    bootstrap = BootstrapResult(
        config_path=config_path,
        output_dir=tmp_path / ".vllm-sr",
    )
    captured: dict[str, object] = {}

    class _StubBackend:
        def deploy(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        runtime_commands, "ensure_bootstrap_workspace", lambda *_a, **_kw: bootstrap
    )
    monkeypatch.setattr(
        runtime_commands, "_build_backend", lambda *a, **kw: _StubBackend()
    )

    result = CliRunner().invoke(
        main,
        [
            "serve",
            "--with-observability",
            "--image-pull-policy",
            "never",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["enable_observability"] is True
