"""`vllm-sr serve MODEL [MODEL ...]` delegates to the model runtime; router mode is unchanged."""

import sys
import types

import pytest
from cli.commands import runtime as runtime_commands
from cli.main import main
from click.testing import CliRunner

REVISION = "cd49ea3813fd8ba0928a9a23ef6c9a0f2f0cd764"


@pytest.fixture
def runtime_calls(monkeypatch):
    calls = []
    package = types.ModuleType("vllm_srun")
    module = types.ModuleType("vllm_srun.cli")

    def fake_main(argv):
        calls.append(list(argv))
        return 0

    module.main = fake_main
    package.cli = module
    monkeypatch.setitem(sys.modules, "vllm_srun", package)
    monkeypatch.setitem(sys.modules, "vllm_srun.cli", module)
    return calls


@pytest.fixture
def router_serve(monkeypatch):
    calls = []
    monkeypatch.setattr(
        runtime_commands, "_execute_serve", lambda *args: calls.append(args)
    )
    return calls


def test_engine_mode_delegates_to_the_runtime(runtime_calls, router_serve):
    result = CliRunner().invoke(
        main,
        [
            "serve",
            "vllm-sr/Decision-2.0-Kai-0.6B",
            "--revision",
            REVISION,
            "--device",
            "cpu",
            "--port",
            "8200",
            "--profile",
            "batching",
            "--log-level",
            "warn",
        ],
    )

    assert result.exit_code == 0, result.output
    assert runtime_calls == [
        [
            "serve",
            "vllm-sr/Decision-2.0-Kai-0.6B",
            "--device",
            "cpu",
            "--profile",
            "batching",
            "--revision",
            REVISION,
            "--port",
            "8200",
            "--log-level",
            "warning",
        ]
    ]
    assert router_serve == []


def test_engine_mode_leaves_profile_names_to_the_runtime(runtime_calls):
    result = CliRunner().invoke(
        main, ["serve", "/models/kai", "--profile", "plugin_turbo"]
    )

    assert result.exit_code == 0, result.output
    assert runtime_calls == [
        ["serve", "/models/kai", "--device", "auto", "--profile", "plugin_turbo"]
    ]


def test_engine_mode_defaults_and_unix_socket(runtime_calls, tmp_path):
    socket = str(tmp_path / "runtime.sock")
    result = CliRunner().invoke(main, ["serve", "/models/kai", "--uds", socket])

    assert result.exit_code == 0, result.output
    assert runtime_calls == [
        [
            "serve",
            "/models/kai",
            "--device",
            "auto",
            "--profile",
            "exact",
            "--uds",
            socket,
        ]
    ]


@pytest.mark.parametrize(
    "arguments, message",
    [
        (["--config", "my.yaml"], "--config applies to router mode"),
        (["--target", "k8s"], "--target applies to router mode"),
        (["--recipe-env", "TOKEN"], "--recipe-env applies to router mode"),
        (["--profile", "Fast!"], "is not a profile name"),
        (["--uds", "/tmp/r.sock", "--port", "1"], "--uds cannot be combined"),
    ],
)
def test_engine_mode_rejects_router_options(runtime_calls, arguments, message):
    result = CliRunner().invoke(
        main, ["serve", "vllm-sr/Decision-2.0-Kai-0.6B", *arguments]
    )

    assert result.exit_code == 2
    assert message in result.output
    assert runtime_calls == []


def test_several_models_share_one_runtime(runtime_calls, router_serve):
    result = CliRunner().invoke(
        main,
        [
            "serve",
            f"vllm-sr/Decision-2.0-Kai-0.6B@{REVISION}",
            "vllm-sr/Vela-1.0-Encoder-307M-Domain",
            "--device",
            "cpu",
        ],
    )

    assert result.exit_code == 0, result.output
    assert runtime_calls == [
        [
            "serve",
            f"vllm-sr/Decision-2.0-Kai-0.6B@{REVISION}",
            "vllm-sr/Vela-1.0-Encoder-307M-Domain",
            "--device",
            "cpu",
            "--profile",
            "exact",
        ]
    ]
    assert router_serve == []


def test_a_models_file_selects_engine_mode(runtime_calls, router_serve, tmp_path):
    models = tmp_path / "models.yaml"
    models.write_text("models:\n  - {model: /models/kai, name: kai}\n")
    result = CliRunner().invoke(
        main, ["serve", "--models", str(models), "--port", "8300"]
    )

    assert result.exit_code == 0, result.output
    assert runtime_calls == [
        [
            "serve",
            "--models",
            str(models),
            "--device",
            "auto",
            "--profile",
            "exact",
            "--port",
            "8300",
        ]
    ]
    assert router_serve == []


@pytest.mark.parametrize(
    "arguments, message",
    [
        (["a", "b", "--revision", REVISION], "--revision applies to a single MODEL"),
        (["a", "--models", "models.yaml"], "not both"),
    ],
)
def test_engine_mode_rejects_ambiguous_model_lists(runtime_calls, arguments, message):
    result = CliRunner().invoke(main, ["serve", *arguments])

    assert result.exit_code == 2
    assert message in result.output
    assert runtime_calls == []


def test_router_mode_rejects_engine_options(router_serve):
    result = CliRunner().invoke(main, ["serve", "--device", "cpu"])

    assert result.exit_code == 2
    assert "--device applies to engine mode" in result.output
    assert router_serve == []


def test_router_mode_keeps_the_deployment_profile(router_serve):
    result = CliRunner().invoke(main, ["serve", "--target", "k8s", "--profile", "dev"])

    assert result.exit_code == 0, result.output
    assert len(router_serve) == 1
    assert "dev" in router_serve[0]


def test_engine_mode_without_the_runtime_explains_the_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm_srun", None)
    monkeypatch.setitem(sys.modules, "vllm_srun.cli", None)

    result = CliRunner().invoke(main, ["serve", "vllm-sr/Decision-2.0-Kai-0.6B"])

    assert result.exit_code == 1
    assert "pip install ./src/model-runtime" in result.output


def test_serve_help_documents_engine_mode():
    result = CliRunner().invoke(main, ["serve", "--help"])

    assert result.exit_code == 0
    assert "ENGINE MODE" in result.output
    assert "--revision" in result.output
    assert "--uds" in result.output
