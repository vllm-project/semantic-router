from pathlib import Path

import pytest
import yaml
from cli.bootstrap import BootstrapResult
from cli.commands import runtime as runtime_commands
from cli.main import main
from click.testing import CliRunner


def _capture_serve_deployment(monkeypatch, tmp_path: Path):
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
        runtime_commands, "_build_backend", lambda *args, **kwargs: _StubBackend()
    )
    return config_path, captured


def test_serve_uses_algorithm_translated_config(monkeypatch, tmp_path: Path):
    config_path, captured = _capture_serve_deployment(monkeypatch, tmp_path)

    result = CliRunner().invoke(
        main,
        [
            "serve",
            "--config",
            str(config_path),
            "--algorithm",
            "multi_factor",
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
    assert (
        captured["env_vars"]["VLLM_SR_SOURCE_CONFIG_PATH"]
        == "/app/.vllm-sr/runtime-config.yaml"
    )
    assert "VLLM_SR_STATE_ROOT_DIR" not in captured["env_vars"]
    assert captured["env_vars"]["VLLM_SR_ALGORITHM_OVERRIDE"] == "multi_factor"
    assert translated["routing"]["decisions"][0]["algorithm"]["type"] == "multi_factor"
    assert captured["pull_policy"] == "never"


def test_serve_passes_role_specific_images_to_backend(monkeypatch, tmp_path: Path):
    config_path, captured = _capture_serve_deployment(monkeypatch, tmp_path)

    result = CliRunner().invoke(
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


@pytest.mark.parametrize("seconds", [None, 7200])
def test_serve_startup_timeout_is_a_host_side_option(monkeypatch, tmp_path, seconds):
    config_path, captured = _capture_serve_deployment(monkeypatch, tmp_path)
    args = ["serve", "--config", str(config_path)]
    if seconds is not None:
        args.extend(["--startup-timeout", str(seconds)])
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    assert captured.get("startup_timeout") == seconds
    assert not any("TIMEOUT" in name for name in captured["env_vars"])
    assert "startup_timeout" not in Path(captured["runtime_config_file"]).read_text()


@pytest.mark.parametrize("seconds", ["0", "-1", "1.5", "nan", "inf", "bad"])
def test_serve_rejects_invalid_startup_timeout_before_mutation(monkeypatch, seconds):
    def unexpected(*args, **kwargs):
        pytest.fail("invalid timeout reached workspace or backend mutation")

    monkeypatch.setattr(runtime_commands, "ensure_bootstrap_workspace", unexpected)
    monkeypatch.setattr(runtime_commands, "_build_backend", unexpected)
    result = CliRunner().invoke(main, ["serve", "--startup-timeout", seconds])
    assert result.exit_code == 2
    assert "--startup-timeout" in result.output


def test_serve_rejects_startup_timeout_for_kubernetes_before_mutation(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("unsupported timeout reached workspace or backend mutation")

    monkeypatch.setattr(runtime_commands, "_resolve_serve_config", unexpected)
    monkeypatch.setattr(runtime_commands, "_build_backend", unexpected)
    result = CliRunner().invoke(
        main, ["serve", "--target", "k8s", "--startup-timeout", "7200"]
    )
    assert result.exit_code == 1
    assert "supported only for local Docker" in result.output
