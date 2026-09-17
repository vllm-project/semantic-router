"""The fresh setup flow preserves the strictly validated product envelope."""

from pathlib import Path

import pytest
import yaml
from cli import container_start
from cli.bootstrap import build_bootstrap_config, ensure_bootstrap_workspace
from cli.commands.runtime_support import realize_runtime_config
from cli.parser import ConfigParseError, parse_user_config
from cli.runtime_stack import resolve_runtime_stack


def test_fresh_amd_bootstrap_reaches_standby_specs_with_real_envoy_render(
    tmp_path: Path, monkeypatch
):
    bootstrap = ensure_bootstrap_workspace(tmp_path / "config.yaml")
    source_bytes = bootstrap.config_path.read_bytes()
    runtime = tmp_path / ".vllm-sr" / "runtime-config.setup-test.yaml"
    realize_runtime_config(
        bootstrap.config_path, runtime, algorithm=None, platform="amd"
    )
    runtime_bytes = runtime.read_bytes()
    layout = resolve_runtime_stack(stack_name="setup-test", port_offset=12000)
    captured = []
    monkeypatch.setattr(container_start, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(
        container_start,
        "get_runtime_images",
        lambda **kwargs: dict.fromkeys(("router", "envoy", "dashboard"), "test-image"),
    )
    docker = tmp_path / "docker"
    docker.touch()
    monkeypatch.setattr(
        container_start, "resolve_container_cli_path", lambda **kwargs: str(docker)
    )

    def capture_specs(specs, *, storage_secret_values):
        captured.extend(specs)
        return 0, "", ""

    monkeypatch.setattr(container_start, "run_container_specs", capture_specs)
    result = container_start.container_start_vllm_sr(
        str(bootstrap.config_path),
        {"VLLM_SR_SETUP_MODE": "true", "DASHBOARD_SETUP_MODE": "true"},
        [{"name": "http-8899", "address": "0.0.0.0", "port": 8899}],
        runtime_config_file=str(runtime),
        stack_layout=layout,
    )
    assert result[0] == 0
    assert bootstrap.config_path.read_bytes() == source_bytes
    assert runtime.read_bytes() == runtime_bytes
    assert yaml.safe_load(runtime_bytes)["setup"]["mode"] is True
    envoy = yaml.safe_load((tmp_path / ".vllm-sr" / "envoy.yaml").read_text())
    assert envoy["static_resources"]["listeners"]
    services = {name: commands for name, _, commands in captured}
    assert set(services) == {"router", "envoy", "dashboard"}
    assert services["router"][0][1] == "create"
    assert services["envoy"][0][1] == "create"
    assert services["dashboard"][0][1] == "run"
    assert not any(
        command[1] == "start" for commands in services.values() for command in commands
    )
    parsed = parse_user_config(str(runtime))
    assert parsed.model_dump().get("setup", {}).get("mode") is True


@pytest.mark.parametrize(
    "setup",
    [{"mode": "true"}, {"mode": 1}, {"mode": True, "unknown": 1}],
)
def test_setup_renderer_rejects_invalid_internal_metadata(tmp_path: Path, setup):
    config = build_bootstrap_config()
    config["setup"] = setup
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ConfigParseError, match="setup"):
        container_start._render_split_envoy_config(
            str(path),
            str(tmp_path / "envoy.yaml"),
            resolve_runtime_stack(),
        )
    assert not (tmp_path / "envoy.yaml").exists()


@pytest.mark.parametrize(
    "invalid", [{"unexpected": True}, {"listeners": [{"port": "invalid"}]}]
)
def test_setup_renderer_validates_remaining_canonical_fields(tmp_path: Path, invalid):
    config = build_bootstrap_config()
    config.update(invalid)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ConfigParseError, match="schema validation"):
        container_start._render_split_envoy_config(
            str(path),
            str(tmp_path / "envoy.yaml"),
            resolve_runtime_stack(),
        )
    assert not (tmp_path / "envoy.yaml").exists()


@pytest.mark.parametrize("mode", [True, False])
def test_regular_renderer_accepts_only_validated_setup_envelope(tmp_path: Path, mode):
    path = tmp_path / "config.yaml"
    config = build_bootstrap_config()
    config["setup"]["mode"] = mode
    path.write_text(yaml.safe_dump(config))
    output = tmp_path / "envoy.yaml"
    container_start._render_split_envoy_config(
        str(path), str(output), resolve_runtime_stack()
    )
    assert yaml.safe_load(output.read_text())["static_resources"]["listeners"]
