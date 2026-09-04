import pytest
from cli import runtime_lifecycle
from cli.runtime_stack import resolve_runtime_stack


def test_wait_for_router_health_fails_fast_when_router_exits(monkeypatch):
    calls = {"exec": 0, "logs": 0}

    monkeypatch.setattr(
        runtime_lifecycle, "_emit_router_startup_logs", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(runtime_lifecycle, "container_status", lambda _name: "exited")

    def fake_exec(*_args, **_kwargs):
        calls["exec"] += 1
        return 1, "", ""

    def fake_logs(*_args, **_kwargs):
        calls["logs"] += 1

    monkeypatch.setattr(runtime_lifecycle, "container_exec", fake_exec)
    monkeypatch.setattr(runtime_lifecycle, "container_logs", fake_logs)

    with pytest.raises(SystemExit):
        runtime_lifecycle.wait_for_router_health(resolve_runtime_stack())

    assert calls["exec"] == 0
    assert calls["logs"] == 1


def test_wait_for_router_health_uses_configured_management_port(monkeypatch):
    commands = []
    monkeypatch.setattr(
        runtime_lifecycle, "_emit_router_startup_logs", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(runtime_lifecycle, "container_status", lambda _name: "running")

    def fake_exec(_container, command):
        commands.append(command)
        return 0, "", ""

    monkeypatch.setattr(runtime_lifecycle, "container_exec", fake_exec)

    runtime_lifecycle.wait_for_router_health(
        resolve_runtime_stack(), management_port=9090
    )

    assert commands == [["curl", "-f", "-s", "http://localhost:9090/ready"]]


def test_wait_for_router_health_reads_bearer_from_container_environment(monkeypatch):
    commands = []
    monkeypatch.setattr(
        runtime_lifecycle, "_emit_router_startup_logs", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(runtime_lifecycle, "container_status", lambda _name: "running")

    def fake_exec(_container, command):
        commands.append(command)
        return 0, "", ""

    monkeypatch.setattr(runtime_lifecycle, "container_exec", fake_exec)

    runtime_lifecycle.wait_for_router_health(
        resolve_runtime_stack(),
        management_port=9090,
        readiness_token_env="CATALOG_MANAGEMENT_TOKEN",
    )

    assert len(commands) == 1
    command = commands[0]
    assert command[:2] == ["sh", "-c"]
    assert command[-2:] == [
        "CATALOG_MANAGEMENT_TOKEN",
        "http://localhost:9090/ready",
    ]
    assert 'printenv "$1"' in command[2]
    assert "curl -f -s -H @-" in command[2]
    assert "secret-value" not in repr(command)


def test_runtime_summary_is_clean_human_stdout(capsys):
    stack_layout = resolve_runtime_stack(stack_name="terminal-test", port_offset=200)

    runtime_lifecycle.log_runtime_summary(
        [{"name": "http-8899", "port": 8899}],
        stack_layout,
        dashboard_disabled=False,
        enable_observability=True,
        started_backends={"postgres", "redis"},
    )

    captured = capsys.readouterr()
    assert captured.err == ""
    assert "✓ vLLM Semantic Router is running" in captured.out
    assert "Endpoints" in captured.out
    assert stack_layout.dashboard_url in captured.out
    assert "http://localhost:9099" in captured.out
    assert "Storage" in captured.out
    assert "Observability" in captured.out
    assert "Commands" in captured.out
    assert "Try it" in captured.out


def test_router_startup_diagnostics_use_stderr(capsys):
    runtime_lifecycle._print_matching_lines(
        '2026-01-01 router ready caller="startup"\nordinary line'
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert 'caller="startup"' in captured.err
    assert "ordinary line" not in captured.err


def test_setup_mode_keeps_progress_on_stderr_and_summary_on_stdout(monkeypatch, capsys):
    monkeypatch.setattr(runtime_lifecycle, "_wait_for_setup_dashboard", lambda *_: None)
    monkeypatch.setattr(
        runtime_lifecycle, "ensure_runtime_container_not_exited", lambda *_a, **_k: None
    )
    stack_layout = resolve_runtime_stack()

    finished = runtime_lifecycle.maybe_finish_setup_mode(
        setup_mode=True,
        dashboard_disabled=False,
        stack_layout=stack_layout,
    )

    captured = capsys.readouterr()
    assert finished is True
    assert "✓ vLLM Semantic Router setup mode is running" in captured.out
    assert stack_layout.dashboard_url in captured.out
    assert "Next steps" in captured.out
    assert "Setup mode detected" in captured.err
    assert "Waiting for Dashboard" in captured.err
