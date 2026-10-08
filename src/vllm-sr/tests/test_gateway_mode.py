"""Gateway modes, targets and the option groups of `vllm-sr serve`."""

import pytest
import yaml
from cli.commands import runtime as runtime_commands
from cli.config_translator import translate_config_to_helm_values
from cli.deployment_backend import resolve_target
from cli.gateway_mode import GATEWAY_ENV, apply_gateway_mode, resolve_gateway
from cli.main import main
from click.testing import CliRunner

CONFIG = {
    "version": "v0.3",
    "listeners": [{"name": "http-8899", "address": "0.0.0.0", "port": 8899}],
    "providers": {
        "defaults": {"model": "m"},
        "models": [
            {
                "name": "m",
                "backend_refs": [{"endpoint": "127.0.0.1:8000", "protocol": "http"}],
            }
        ],
    },
    "routing": {"modelCards": [{"name": "m"}]},
}


def test_standalone_is_the_default_gateway():
    assert resolve_gateway(None) == "standalone"
    assert resolve_gateway("ExtProc") == "extproc"
    with pytest.raises(ValueError, match="Invalid gateway"):
        resolve_gateway("native")


def test_apply_gateway_mode_reaches_the_containers_only(monkeypatch):
    monkeypatch.delenv(GATEWAY_ENV, raising=False)
    env_vars = {}
    apply_gateway_mode(env_vars, "standalone")
    assert env_vars == {GATEWAY_ENV: "standalone"}


def test_the_kubernetes_target_keeps_k8s_for_one_release(caplog):
    assert resolve_target(None) == "docker"
    assert resolve_target("Kubernetes") == "kubernetes"
    assert resolve_target("k8s") == "kubernetes"
    assert "renamed to --target kubernetes" in caplog.text
    with pytest.raises(ValueError, match="Invalid deployment target"):
        resolve_target("local")


def test_kubernetes_values_carry_the_gateway_and_a_gpu_platform(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(CONFIG))

    standalone = translate_config_to_helm_values(str(config), gateway="standalone")
    assert standalone["gateway"] == {"mode": "standalone"}
    assert "resources" not in standalone
    assert "repository" not in standalone.get("image", {})

    amd = translate_config_to_helm_values(
        str(config), gateway="extproc", platform="rocm"
    )
    assert amd["gateway"] == {"mode": "extproc"}
    assert (
        amd["image"]["repository"]
        == "ghcr.io/vllm-project/semantic-router/vllm-sr-rocm"
    )
    assert amd["resources"]["limits"] == {"amd.com/gpu": 1}

    nvidia = translate_config_to_helm_values(
        str(config),
        platform="cuda",
        image="registry.example/router:1",
        profile_values={"resources": {"limits": {"memory": "16Gi"}}},
    )
    assert nvidia["image"]["repository"] == "registry.example/router"
    assert nvidia["resources"]["limits"] == {"memory": "16Gi", "nvidia.com/gpu": 1}


@pytest.fixture
def captured_serve(monkeypatch):
    calls = []
    monkeypatch.setattr(
        runtime_commands, "_execute_serve", lambda *args, **kwargs: calls.append(args)
    )
    return calls


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--namespace", "ns"], "--namespace applies to the kubernetes target"),
        (["--chart-dir", "chart"], "--chart-dir applies to the kubernetes target"),
        (["--profile", "dev"], "--profile applies to the kubernetes target"),
        (
            ["--target", "kubernetes", "--router-image", "r:1"],
            "--router-image applies to the docker target",
        ),
        (
            ["--target", "kubernetes", "--container-runtime", "podman"],
            "--container-runtime applies to the docker target",
        ),
        (
            ["--target", "kubernetes", "--startup-timeout", "60"],
            "--startup-timeout applies to the docker target",
        ),
        (["--envoy-image", "e:1"], "--envoy-image applies to --gateway extproc"),
    ],
)
def test_an_option_outside_its_group_names_where_it_applies(
    captured_serve, arguments, message
):
    result = CliRunner().invoke(main, ["serve", *arguments])
    assert result.exit_code == 2
    assert message in result.output
    assert captured_serve == []


def test_common_options_serve_both_targets(captured_serve):
    for target in ("docker", "kubernetes"):
        result = CliRunner().invoke(
            main,
            [
                "serve",
                "--target",
                target,
                "--gateway",
                "extproc",
                "--platform",
                "rocm",
                "--image",
                "img:1",
                "--minimal",
            ],
        )
        assert result.exit_code == 0, result.output
    assert [call[-1] for call in captured_serve] == ["extproc", "extproc"]


def test_extproc_takes_an_envoy_image(captured_serve):
    result = CliRunner().invoke(
        main, ["serve", "--gateway", "extproc", "--envoy-image", "e:1"]
    )
    assert result.exit_code == 0, result.output
    assert captured_serve[0][4] == "e:1"


def test_runtime_is_the_old_name_of_container_runtime(captured_serve):
    result = CliRunner().invoke(main, ["serve", "--runtime", "podman"])
    assert result.exit_code == 0, result.output
    assert "renamed to --container-runtime" in result.output
    conflicting = CliRunner().invoke(
        main, ["serve", "--runtime", "podman", "--container-runtime", "docker"]
    )
    assert conflicting.exit_code == 2


def test_serve_help_lists_the_option_groups():
    result = CliRunner().invoke(main, ["serve", "--help"])
    assert result.exit_code == 0
    for section in (
        "Instance configuration (docker and kubernetes targets):",
        "Docker target:",
        "Kubernetes target:",
        "Instance and model options:",
    ):
        assert section in result.output
    assert "--gateway [standalone|extproc]" in result.output
    assert "--container-runtime" in result.output
    assert "--runtime [" not in result.output, "the old name is hidden"
