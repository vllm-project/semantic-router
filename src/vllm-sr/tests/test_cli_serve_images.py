from pathlib import Path

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
