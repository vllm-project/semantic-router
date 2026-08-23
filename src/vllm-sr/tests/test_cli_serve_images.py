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
                "version": "v0.4",
                "listeners": [
                    {"name": "http-8899", "address": "0.0.0.0", "port": 8899}
                ],
                "models": [
                    {
                        "name": "default-model",
                        "card": {"capabilities": ["chat"]},
                        "connections": [
                            {
                                "provider": "vllm",
                                "endpoint": "http://127.0.0.1:8000/v1",
                                "model": "default-model",
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
                                    "name": "default",
                                    "algorithm": {"type": "static"},
                                    "rules": {"operator": "AND", "conditions": []},
                                }
                            ]
                        },
                    }
                ],
                "entrypoints": [
                    {
                        "name": "vllm-sr/default",
                        "aliases": ["default"],
                        "recipe": "default",
                        "assignments": {
                            "default": {"models": [{"model": "default-model"}]}
                        },
                    }
                ],
                "global": {
                    "services": {
                        "backend_egress": {
                            "policy_file": "/app/config/backend-egress-policy.yaml"
                        }
                    }
                },
            },
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
        runtime_commands,
        "ensure_bootstrap_workspace",
        lambda *_args, **_kwargs: bootstrap,
    )
    monkeypatch.setattr(
        runtime_commands, "_build_backend", lambda *args, **kwargs: _StubBackend()
    )
    return config_path, captured


def test_serve_preserves_configured_recipe_algorithm(monkeypatch, tmp_path: Path):
    config_path, captured = _capture_serve_deployment(monkeypatch, tmp_path)

    result = CliRunner().invoke(
        main,
        [
            "serve",
            "--image-pull-policy",
            "never",
        ],
    )

    assert result.exit_code == 0
    effective_config = Path(captured["config_file"])
    with effective_config.open() as handle:
        translated = yaml.safe_load(handle)
    assert captured["source_config_file"] == str(config_path)
    assert captured["compiled_bootstrap_file"] == str(effective_config)
    assert effective_config.name == "compiled-bootstrap.yaml"
    assert "VLLM_SR_RUNTIME_CONFIG_PATH" not in captured["env_vars"]
    assert "VLLM_SR_SOURCE_CONFIG_PATH" not in captured["env_vars"]
    assert "VLLM_SR_STATE_ROOT_DIR" not in captured["env_vars"]
    assert "VLLM_SR_ALGORITHM_OVERRIDE" not in captured["env_vars"]
    assert (
        translated["recipes"][0]["document"]["decisions"][0]["algorithm"]["type"]
        == "static"
    )
    assert captured["pull_policy"] == "never"


def test_serve_passes_role_specific_images_to_backend(monkeypatch, tmp_path: Path):
    _config_path, captured = _capture_serve_deployment(monkeypatch, tmp_path)

    result = CliRunner().invoke(
        main,
        [
            "serve",
            "--router-image",
            "test/router:latest",
            "--envoy-image",
            "test/envoy:latest",
            "--dashboard-image",
            "test/dashboard:latest",
            "--sim-image",
            "test/sim:latest",
            "--image-pull-policy",
            "never",
        ],
    )

    assert result.exit_code == 0
    assert "topology" not in captured
    assert captured["router_image"] == "test/router:latest"
    assert captured["envoy_image"] == "test/envoy:latest"
    assert captured["dashboard_image"] == "test/dashboard:latest"
    assert captured["sim_image"] == "test/sim:latest"
