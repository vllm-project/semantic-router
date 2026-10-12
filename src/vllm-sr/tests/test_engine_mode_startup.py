"""Engine mode starts only what serving a model needs, out of the working directory, with short output."""

import logging
from pathlib import Path

import pytest
from cli import core, runtime_lifecycle, terminal
from cli.commands import runtime
from cli.main import main
from cli.managed_storage_detection import detect_canonical_storage_backends
from cli.runtime_stack import resolve_runtime_stack
from click.testing import CliRunner

MODEL = "vllm-sr/d3"
# The positional slot of _execute_serve that turns the Dashboard and sr-bench off.
MINIMAL = 8


@pytest.fixture
def served(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    calls = []

    def execute(*args, **kwargs):
        calls.append((args, kwargs, terminal.quiet_logs()))

    monkeypatch.setattr(runtime, "_execute_serve", execute)
    return calls


@pytest.mark.parametrize(
    ("flags", "minimal", "observability"),
    [
        (["-e"], True, False),
        (["-e", "--dashboard"], False, False),
        (["-e", "--observability"], True, True),
        (["-e", "--dashboard", "--observability"], False, True),
        (["-e", "--no-minimal"], False, True),
        ([], False, True),
        (["--minimal"], True, False),
        (["--minimal", "--dashboard"], False, False),
    ],
)
def test_engine_mode_is_minimal_unless_components_are_asked_for(
    served, flags, minimal, observability
):
    result = CliRunner().invoke(main, ["serve", MODEL, *flags])
    assert result.exit_code == 0, result.output
    ((args, kwargs, _quiet),) = served
    assert args[MINIMAL] is minimal
    assert kwargs["observability"] is observability


def test_engine_mode_keeps_its_config_out_of_the_working_directory(served, tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        result = runner.invoke(main, ["serve", MODEL, "-e"])
        assert result.exit_code == 0, result.output
        ((args, kwargs, quiet),) = served
        workspace = tmp_path / "state" / "vllm-sr" / "engine" / "vllm-sr--d3"
        assert args[0] == str(workspace / "config.yaml")
        assert kwargs["cli_workspace"] is True and quiet is True
        assert workspace.is_dir() and (workspace.stat().st_mode & 0o077) == 0
        assert not any(Path(cwd).iterdir())
    assert terminal.quiet_logs() is False


@pytest.mark.parametrize(
    ("flags", "config", "announced", "quiet"),
    [
        (["-e", "--config", "mine.yaml"], "mine.yaml", True, True),
        (
            ["-e", "--verbose"],
            "state/vllm-sr/engine/vllm-sr--d3/config.yaml",
            False,
            False,
        ),
        ([], "config.yaml", True, False),
    ],
)
def test_an_explicit_config_verbose_output_and_router_mode(
    served, tmp_path, flags, config, announced, quiet
):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(main, ["serve", MODEL, *flags])
        assert result.exit_code == 0, result.output
    ((args, kwargs, quieted),) = served
    expected = str(tmp_path / config) if config.startswith("state/") else config
    assert args[0] == expected
    assert kwargs["cli_workspace"] is not announced and quieted is quiet


def test_each_model_has_its_own_engine_workspace(served, tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        for arguments in (
            ["vllm-sr/d3-edge"],
            ["vllm-sr/d3-nano"],
            ["vllm-sr/d3-edge"],
            [],
        ):
            result = runner.invoke(main, ["serve", *arguments, "-e"])
            assert result.exit_code == 0, result.output
    engine = tmp_path / "state" / "vllm-sr" / "engine"
    assert [args[0] for args, _, _ in served] == [
        str(engine / name / "config.yaml")
        for name in (
            "vllm-sr--d3-edge",
            "vllm-sr--d3-nano",
            "vllm-sr--d3-edge",
            "default",
        )
    ]


def test_an_existing_config_in_the_working_directory_still_serves_engine_mode(
    served, tmp_path
):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        open("config.yaml", "w").close()
        result = runner.invoke(main, ["serve", MODEL, "-e"])
        assert result.exit_code == 0, result.output
    ((args, kwargs, quiet),) = served
    assert args[0] == "config.yaml" and kwargs["cli_workspace"] is False
    assert quiet is True


def test_the_routing_pipeline_stores_start_only_with_routing():
    config = {"global": {"router": {"enabled": False}}}
    layout = resolve_runtime_stack()
    assert detect_canonical_storage_backends(config, layout) == {"redis"}
    assert detect_canonical_storage_backends(config, layout, routing=False) == set()
    vector = {
        "global": {
            "stores": {"vector_store": {"enabled": True, "metadata_store": "postgres"}}
        }
    }
    assert detect_canonical_storage_backends(vector, layout, routing=False) == {
        "postgres"
    }


@pytest.mark.parametrize(
    ("enabled", "dashboard_disabled", "routing"),
    [(False, True, False), (False, False, True), (None, True, True)],
)
def test_engine_mode_without_a_dashboard_starts_no_routing_storage(
    monkeypatch, enabled, dashboard_disabled, routing
):
    seen = {}
    monkeypatch.setattr(
        core,
        "provision_storage_backends",
        lambda config, layout, **kwargs: seen.update(kwargs) or set(),
    )
    monkeypatch.setattr(core, "start_observability_stack", lambda *args: None)
    monkeypatch.setattr(core, "_stop_observability_containers", lambda layout: None)
    router = {} if enabled is None else {"enabled": enabled}
    core._start_support_services(
        {"global": {"router": router}},
        "net",
        "/state",
        {},
        resolve_runtime_stack(),
        False,
        routing=core._routing_enabled({"global": {"router": router}})
        or not dashboard_disabled,
    )
    assert seen["routing"] is routing


def test_quiet_logs_keep_headlines_warnings_and_errors(capsys):
    logger = logging.getLogger("test-engine-quiet")
    logger.addHandler(terminal.TerminalLogHandler())
    logger.setLevel(logging.INFO)
    terminal.set_quiet_logs(True)
    try:
        logger.info("internal detail")
        logger.info("Pulling container image: example", extra=terminal.HEADLINE)
        logger.warning("worth knowing")
    finally:
        terminal.set_quiet_logs(False)
    logger.info("shown again")
    output = capsys.readouterr().err
    assert "internal detail" not in output
    assert "Pulling container image: example" in output
    assert "Warning: worth knowing" in output and "shown again" in output


def test_quiet_startup_shows_model_download_progress_only(monkeypatch, capsys):
    line = (
        '{"level":"info","msg":"runtime_log","line":"\\rFetching 21 files:  57%|'
        '\u2588\u2588\u2588\u2588\u2588\u258b    | 12/21 [00:18<00:08,  1.11it/s]2026 INFO httpx: GET"}'
    )
    monkeypatch.setattr(
        runtime_lifecycle, "container_logs_since", lambda *a, **kw: (0, line, "")
    )
    downloads = runtime_lifecycle._DownloadProgress()
    terminal.set_quiet_logs(True)
    try:
        for _ in range(2):
            runtime_lifecycle._emit_router_startup_logs("router", 0, downloads)
    finally:
        terminal.set_quiet_logs(False)
    output = capsys.readouterr().err
    assert output.count("Downloading model files: 12 of 21") == 1
    assert "httpx" not in output and "runtime_log" not in output


def test_quiet_engine_summary_is_one_ready_line_and_how_to_stop(capsys):
    terminal.set_quiet_logs(True)
    try:
        runtime_lifecycle.log_runtime_summary(
            [{"name": "http-8899", "port": 8899, "systemone": {"models": [MODEL]}}],
            resolve_runtime_stack(),
            dashboard_disabled=True,
            enable_observability=False,
            config={"global": {"router": {"enabled": False}}},
        )
    finally:
        terminal.set_quiet_logs(False)
    output = capsys.readouterr().out
    assert "Ready: http://localhost:8899/v1/systemone (model vllm-sr/d3)" in output
    assert "Stop with: vllm-sr stop" in output
    assert "Endpoints" not in output and "curl" not in output


def test_status_of_a_minimal_stack_lists_no_dashboard(monkeypatch, capsys):
    layout = resolve_runtime_stack()
    states = {layout.router_container_name: "running"}
    monkeypatch.setattr(
        core, "container_status", lambda name: states.get(name, "not found")
    )
    monkeypatch.setattr(core, "_runtime_stack_status", lambda stack: "running")
    monkeypatch.setattr(core, "_activation_state", lambda stack: None)
    reported = []
    monkeypatch.setattr(
        core, "report_service_status", lambda service, stack: reported.append(service)
    )
    core.show_status("all")
    assert reported == ["router"]
    assert "Detailed logs: vllm-sr logs <router>" in capsys.readouterr().out
