"""`vllm-sr serve MODEL [MODEL ...]` runs the model runtime in a container; router mode is unchanged."""

import os
import signal
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml
from cli import container_run_command, engine_container
from cli.commands import runtime as runtime_commands
from cli.commands import runtime_engine
from cli.main import main
from click.testing import CliRunner

REVISION = "cd49ea3813fd8ba0928a9a23ef6c9a0f2f0cd764"
IMAGE = "registry.example/vllm-sr:1"


class Engine:
    """The container runtime and image engine mode resolves, and what it ran."""

    def __init__(self, monkeypatch, cache_dir: Path):
        self.cache_dir = cache_dir
        self.runtime = "docker"
        self.images: list[dict] = []
        self.commands: list[list[str]] = []
        self.models_files: list[dict] = []
        self.exit_code = 0
        monkeypatch.setattr(
            runtime_engine, "get_container_runtime", lambda: self.runtime
        )
        monkeypatch.setattr(runtime_engine, "get_container_image", self._image)
        monkeypatch.setattr(
            engine_container, "_ensure_port_is_free", lambda *args: None
        )
        monkeypatch.setattr(engine_container, "run_foreground", self._run)

    def _image(self, **kwargs):
        self.images.append(kwargs)
        return IMAGE

    def _run(self, command):
        self.commands.append(list(command))
        for spec in _values(command, "-v"):
            host, inside, _ = spec.split(":")
            if inside == "/app/packages/models.yaml":
                self.models_files.append(yaml.safe_load(Path(host).read_text()))
        return self.exit_code

    @property
    def command(self) -> list[str]:
        (command,) = self.commands
        return command

    def runtime_arguments(self) -> list[str]:
        return self.command[self.command.index(IMAGE) + 1 :]


def _values(command, flag):
    return [command[i + 1] for i, value in enumerate(command) if value == flag]


@pytest.fixture
def engine(monkeypatch, tmp_path):
    for name in ("VLLM_SR_DNS", "HF_TOKEN", "HF_ENDPOINT", "HF_HUB_OFFLINE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("VLLM_SR_PLATFORM", raising=False)
    monkeypatch.delenv("DASHBOARD_PLATFORM", raising=False)
    monkeypatch.setenv("VLLM_SR_ENGINE_CACHE_DIR", str(tmp_path / "cache"))
    return Engine(monkeypatch, tmp_path / "cache")


@pytest.fixture
def router_serve(monkeypatch):
    calls = []
    monkeypatch.setattr(
        runtime_commands, "_execute_serve", lambda *args, **kwargs: calls.append(args)
    )
    return calls


def _serve(*arguments):
    return CliRunner().invoke(main, ["serve", *arguments])


def _text(result) -> str:
    """The output with the terminal's line wrapping undone."""
    return " ".join(result.output.split())


def test_engine_mode_runs_the_runtime_in_the_platform_image(engine, router_serve):
    result = _serve(
        "vllm-sr/Decision-2.0-Kai-0.6B",
        "--revision",
        REVISION,
        "--device",
        "cpu",
        "--port",
        "8200",
        "--runtime-profile",
        "batching",
        "--log-level",
        "warn",
    )

    assert result.exit_code == 0, result.output
    assert engine.command == [
        "docker",
        "run",
        "--rm",
        "--init",
        "--name",
        "vllm-sr-engine-8200",
        "-p",
        "127.0.0.1:8200:8100",
        "-v",
        f"{engine.cache_dir}:/app/models:z",
        "--entrypoint",
        "vllm-srun",
        IMAGE,
        "serve",
        "vllm-sr/Decision-2.0-Kai-0.6B",
        "--device",
        "cpu",
        "--profile",
        "batching",
        "--revision",
        REVISION,
        "--host",
        "0.0.0.0",
        "--port",
        "8100",
        "--cache-dir",
        "/app/models/model-runtime",
        "--log-level",
        "warning",
    ]
    assert engine.images == [{"image": None, "pull_policy": "always", "platform": ""}]
    assert engine.cache_dir.is_dir()
    assert router_serve == []


def test_engine_mode_defaults(engine):
    result = _serve("vllm-sr/Decision-2.0-Kai-0.6B")

    assert result.exit_code == 0, result.output
    assert _values(engine.command, "-p") == ["127.0.0.1:8100:8100"]
    assert engine.runtime_arguments()[:6] == [
        "serve",
        "vllm-sr/Decision-2.0-Kai-0.6B",
        "--device",
        "auto",
        "--profile",
        "exact",
    ]


def test_host_and_port_are_the_host_publication(engine):
    result = _serve("kai", "--host", "0.0.0.0", "--port", "9000")

    assert result.exit_code == 0, result.output
    assert _values(engine.command, "-p") == ["0.0.0.0:9000:8100"]
    arguments = engine.runtime_arguments()
    assert arguments[arguments.index("--host") + 1] == "0.0.0.0"
    assert arguments[arguments.index("--port") + 1] == "8100"


def test_local_packages_are_mounted_read_only(engine, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "kai").mkdir()
    (tmp_path / "pkgs" / "domain").mkdir(parents=True)

    result = _serve(
        "kai",
        str(tmp_path / "pkgs" / "domain") + "/",
        "./kai",
        f"vllm-sr/Vela-1.0-Encoder-307M-Domain@{REVISION}",
    )

    assert result.exit_code == 0, result.output
    assert _values(engine.command, "-v")[1:] == [
        f"{tmp_path / 'kai'}:/app/packages/0/kai:ro,z",
        f"{tmp_path / 'pkgs' / 'domain'}:/app/packages/1/domain:ro,z",
    ]
    assert engine.runtime_arguments()[1:5] == [
        "/app/packages/0/kai",
        "/app/packages/1/domain",
        "/app/packages/0/kai",
        f"vllm-sr/Vela-1.0-Encoder-307M-Domain@{REVISION}",
    ]


def test_a_models_file_names_the_mounts_of_its_local_packages(engine, tmp_path):
    (tmp_path / "kai").mkdir()
    models = tmp_path / "models.yaml"
    models.write_text(
        textwrap.dedent(
            f"""\
            models:
              - {{model: {tmp_path / "kai"}, name: kai, device: cpu}}
              - {{model: vllm-sr/Vela-1.0-Encoder-307M-PII, name: pii}}
            """
        )
    )

    result = _serve("--models", str(models), "--port", "8300")

    assert result.exit_code == 0, result.output
    assert engine.models_files == [
        {
            "models": [
                {"model": "/app/packages/0/kai", "name": "kai", "device": "cpu"},
                {"model": "vllm-sr/Vela-1.0-Encoder-307M-PII", "name": "pii"},
            ]
        }
    ]
    assert engine.runtime_arguments()[:3] == [
        "serve",
        "--models",
        "/app/packages/models.yaml",
    ]
    assert models.read_text().count(str(tmp_path / "kai")) == 1


def test_a_models_file_device_is_checked_against_the_image(engine, tmp_path):
    models = tmp_path / "models.yaml"
    models.write_text("models:\n  - {model: vllm-sr/Kai, device: cuda:0}\n")

    result = _serve("--models", str(models))

    assert result.exit_code == 1
    assert "models[0].device cuda:0 needs --platform nvidia" in _text(result)
    assert engine.commands == []


def test_amd_passes_the_rocm_devices_through(engine, monkeypatch):
    monkeypatch.setattr(container_run_command.os.path, "exists", lambda path: True)

    result = _serve("vllm-sr/Kai", "--platform", "amd", "--device", "rocm:1")

    assert result.exit_code == 0, result.output
    assert engine.images[0]["platform"] == "amd"
    container = engine.command[: engine.command.index(IMAGE)]
    assert _values(container, "--device") == ["/dev/kfd", "/dev/dri"]
    assert _values(container, "--group-add") == ["video"]
    assert engine.runtime_arguments()[2:4] == ["--device", "rocm:1"]


@pytest.mark.parametrize(
    ("runtime", "flags"),
    [
        ("docker", ["--gpus", "all", "--runtime", "nvidia"]),
        ("podman", ["--device", "nvidia.com/gpu=all"]),
    ],
)
def test_nvidia_passes_the_gpus_through(engine, runtime, flags):
    engine.runtime = runtime

    result = _serve("vllm-sr/Kai", "--platform", "nvidia")

    assert result.exit_code == 0, result.output
    assert engine.command[6 : 6 + len(flags)] == flags
    assert engine.command[0] == runtime


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--device", "cuda"], "--device cuda needs --platform nvidia"),
        (["--device", "rocm:0", "--platform", "nvidia"], "needs --platform amd"),
        (["--device", "mps"], "mps needs macOS's Metal"),
        (["--device", "xpu:0", "--platform", "amd"], "no router image runs xpu"),
    ],
)
def test_the_device_must_be_one_the_image_runs(engine, arguments, message):
    result = _serve("vllm-sr/Kai", *arguments)

    assert result.exit_code == 2
    assert message in result.output
    assert engine.images == []


def test_a_plugin_accelerator_is_the_runtimes_to_check(engine):
    result = _serve("vllm-sr/Kai", "--device", "example_host", "--image", "mine:1")

    assert result.exit_code == 0, result.output
    assert engine.images[0]["image"] == "mine:1"
    assert engine.runtime_arguments()[2:4] == ["--device", "example_host"]


def test_hub_settings_are_inherited_by_name(engine, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_secret_value")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    result = _serve("vllm-sr/Kai")

    assert result.exit_code == 0, result.output
    assert _values(engine.command, "-e") == ["HF_HUB_OFFLINE", "HF_TOKEN"]
    assert "hf_secret_value" not in " ".join(engine.command)


def test_container_options_apply_to_engine_mode(engine, monkeypatch):
    # Recorded so the override the command applies is undone afterwards.
    monkeypatch.setenv("CONTAINER_RUNTIME", "docker")

    result = _serve(
        "vllm-sr/Kai",
        "--container-runtime",
        "podman",
        "--image-pull-policy",
        "never",
    )

    assert result.exit_code == 0, result.output
    assert os.environ.get("CONTAINER_RUNTIME") == "podman"
    assert engine.images[0]["pull_policy"] == "never"


def test_the_runtimes_exit_code_is_the_commands(engine):
    engine.exit_code = 3

    result = _serve("vllm-sr/Kai")

    assert result.exit_code == 3


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--config", "my.yaml"], "--config applies to router mode"),
        (["--target", "k8s"], "--target applies to router mode"),
        (["--gateway", "extproc"], "--gateway applies to router mode"),
        (["--recipe-env", "TOKEN"], "--recipe-env applies to the docker target"),
        (["--router-image", "r:1"], "--router-image applies to the docker target"),
        (["--namespace", "ns"], "--namespace applies to the kubernetes target"),
        (["--profile", "batching"], "engine mode takes --runtime-profile"),
        (["--runtime-profile", "Fast!"], "is not a profile name"),
    ],
)
def test_engine_mode_rejects_other_modes_options(engine, arguments, message):
    result = _serve("vllm-sr/Decision-2.0-Kai-0.6B", *arguments)

    assert result.exit_code == 2
    assert message in result.output
    assert engine.commands == []


def test_unix_sockets_are_gone(engine):
    result = _serve("vllm-sr/Kai", "--uds", "/tmp/r.sock")

    assert result.exit_code == 2
    assert "No such option '--uds'" in result.output


@pytest.mark.parametrize(
    "arguments, message",
    [
        (["a", "b", "--revision", REVISION], "--revision applies to a single MODEL"),
        (["a", "--models", "models.yaml"], "not both"),
    ],
)
def test_engine_mode_rejects_ambiguous_model_lists(engine, arguments, message):
    result = _serve(*arguments)

    assert result.exit_code == 2
    assert message in result.output
    assert engine.commands == []


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--device", "cpu"], "--device applies to engine mode"),
        (["--runtime-profile", "exact"], "--runtime-profile applies to engine mode"),
        (["--port", "8100"], "--port applies to engine mode"),
    ],
)
def test_router_mode_rejects_engine_options(router_serve, arguments, message):
    result = _serve(*arguments)

    assert result.exit_code == 2
    assert message in result.output
    assert router_serve == []


def test_router_mode_keeps_the_deployment_profile(router_serve):
    result = _serve("--target", "k8s", "--profile", "dev")

    assert result.exit_code == 0, result.output
    assert len(router_serve) == 1
    assert "dev" in router_serve[0]


def test_engine_mode_on_macos_is_cpu_only(engine, monkeypatch):
    monkeypatch.setattr(runtime_commands.sys, "platform", "darwin")

    result = _serve("vllm-sr/Kai", "--platform", "nvidia")

    assert result.exit_code == 1
    assert "needs a Linux host" in result.output
    assert engine.commands == []


def test_serve_help_documents_engine_mode():
    result = CliRunner().invoke(main, ["serve", "--help"])

    assert result.exit_code == 0
    assert "ENGINE MODE" in result.output
    assert "Engine mode (vllm-sr serve MODEL)" in result.output
    assert "--runtime-profile" in result.output
    assert "--uds" not in result.output


def test_a_running_engine_on_the_port_is_refused(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="true\n")

    monkeypatch.setattr(engine_container.subprocess, "run", fake_run)

    with pytest.raises(ValueError, match="already serves port 8100"):
        engine_container._ensure_port_is_free("docker", "vllm-sr-engine-8100", 8100)
    assert len(calls) == 1


def test_a_stopped_leftover_on_the_port_is_removed(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="false\n")

    monkeypatch.setattr(engine_container.subprocess, "run", fake_run)

    engine_container._ensure_port_is_free("podman", "vllm-sr-engine-8100", 8100)
    assert calls[-1] == ["podman", "rm", "-f", "vllm-sr-engine-8100"]


FOREGROUND = textwrap.dedent(
    """\
    import sys
    from cli.engine_container import run_foreground
    child = '''
    import signal, sys, time
    signal.signal(signal.SIGINT, lambda *_: sys.exit(130))
    print("ready", flush=True)
    if sys.argv[1] == "fail":
        sys.exit(3)
    time.sleep(60)
    sys.exit(4)
    '''
    sys.exit(run_foreground([sys.executable, "-c", child, sys.argv[1]]))
    """
)


def _foreground(mode: str) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-c", FOREGROUND, mode],
        stdout=subprocess.PIPE,
        text=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])},
    )


def test_a_stop_is_forwarded_once_and_exits_zero():
    wrapper = _foreground("serve")
    assert wrapper.stdout.readline().strip() == "ready"

    wrapper.send_signal(signal.SIGINT)

    assert wrapper.wait(timeout=30) == 0


def test_a_runtime_failure_keeps_its_exit_code():
    wrapper = _foreground("fail")

    assert wrapper.wait(timeout=30) == 3
