import pytest
from cli import runtime_lifecycle
from cli.runtime_stack import resolve_runtime_stack


def test_wait_for_router_health_fails_fast_when_router_exits(monkeypatch):
    calls = {"exec": 0, "logs": 0}

    monkeypatch.setattr(
        runtime_lifecycle, "_emit_router_startup_logs", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        runtime_lifecycle, "container_status_strict", lambda _name, **_: "exited"
    )

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
    monkeypatch.setattr(
        runtime_lifecycle, "container_status_strict", lambda _name, **_: "running"
    )

    def fake_exec(_container, command, *, timeout):
        commands.append(command)
        assert 0 < timeout <= 5
        return 0, "", ""

    monkeypatch.setattr(runtime_lifecycle, "container_exec", fake_exec)

    runtime_lifecycle.wait_for_router_health(
        resolve_runtime_stack(), management_port=9090
    )

    assert commands == [
        ["curl", "-f", "-s", "--max-time", "5.000", "http://localhost:9090/ready"]
    ]


def test_wait_for_router_health_reads_bearer_from_container_environment(monkeypatch):
    commands = []
    monkeypatch.setattr(
        runtime_lifecycle, "_emit_router_startup_logs", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        runtime_lifecycle, "container_status_strict", lambda _name, **_: "running"
    )

    def fake_exec(_container, command, *, timeout):
        commands.append(command)
        assert 0 < timeout <= 5
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
    assert command[-3:] == [
        "CATALOG_MANAGEMENT_TOKEN",
        "http://localhost:9090/ready",
        "5.000",
    ]
    assert 'printenv "$1"' in command[2]
    assert 'curl -f -s --max-time "$3" -H @-' in command[2]
    assert "secret-value" not in repr(command)


def test_runtime_summary_is_clean_human_stdout(capsys):
    stack_layout = resolve_runtime_stack(stack_name="terminal-test", port_offset=200)

    runtime_lifecycle.log_runtime_summary(
        [{"name": "http-8899", "port": 8899}],
        stack_layout,
        dashboard_disabled=False,
        enable_observability=True,
        started_backends={"postgres", "redis"},
        config={
            "entrypoints": [{"model_names": ["vllm-sr/balance"], "recipe": "balance"}]
        },
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
    assert '"model": "vllm-sr/balance"' in captured.out
    assert "vllm-sr/auto" not in captured.out


def test_runtime_example_retains_automatic_model_for_default_routing(capsys):
    runtime_lifecycle._print_curl_example(
        [{"port": 8899}], resolve_runtime_stack(), {"routing": {"decisions": []}}
    )
    assert '"model": "vllm-sr/auto"' in capsys.readouterr().out


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


class _Clock:
    now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def _readiness_clock(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(runtime_lifecycle.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(runtime_lifecycle.time, "sleep", clock.sleep)
    # Wall-clock jumps must not extend the elapsed readiness budget.
    monkeypatch.setattr(runtime_lifecycle.time, "time", lambda: 100000 - clock.now * 30)
    monkeypatch.setattr(
        runtime_lifecycle, "_emit_router_startup_logs", lambda *a, **k: None
    )
    monkeypatch.setattr(
        runtime_lifecycle, "container_status_strict", lambda *a, **k: "running"
    )
    monkeypatch.setattr(runtime_lifecycle, "container_logs", lambda *a, **k: None)
    for name in ("container_stop_container", "container_remove_container"):
        monkeypatch.setattr(
            runtime_lifecycle,
            name,
            lambda *a, **k: pytest.fail("readiness wait stopped a container"),
        )
    return clock


@pytest.mark.parametrize("startup_timeout", [None, 1804])
def test_readiness_can_wait_past_default_without_changing_default(
    monkeypatch, startup_timeout
):
    clock = _readiness_clock(monkeypatch)
    monkeypatch.setattr(
        runtime_lifecycle,
        "container_exec",
        lambda *a, **k: (0 if clock.now >= 1802 else 1, "", ""),
    )
    if startup_timeout is None:
        with pytest.raises(SystemExit) as exc:
            runtime_lifecycle.wait_for_router_health(resolve_runtime_stack())
        assert exc.value.code == 1
        assert clock.now == 1800
    else:
        runtime_lifecycle.wait_for_router_health(
            resolve_runtime_stack(), startup_timeout=startup_timeout
        )
        assert clock.now == 1802


def test_readiness_io_and_sleep_share_remaining_deadline(monkeypatch):
    clock = _readiness_clock(monkeypatch)
    budgets = []

    def logs(*args, timeout):
        budgets.append(("logs", timeout))
        clock.sleep(0.4)

    def status(*args, timeout):
        budgets.append(("status", timeout))
        clock.sleep(0.4)
        return "running"

    def execute(*args, timeout):
        budgets.append(("exec", timeout))
        clock.sleep(timeout)
        # Even a late success cannot publish ready after the deadline.
        return 0, "", ""

    monkeypatch.setattr(runtime_lifecycle, "_emit_router_startup_logs", logs)
    monkeypatch.setattr(runtime_lifecycle, "container_status_strict", status)
    monkeypatch.setattr(runtime_lifecycle, "container_exec", execute)
    with pytest.raises(SystemExit):
        runtime_lifecycle.wait_for_router_health(
            resolve_runtime_stack(), startup_timeout=1
        )
    assert [name for name, _ in budgets] == ["logs", "status", "exec"]
    assert [value for _, value in budgets] == pytest.approx([1, 0.6, 0.2])
    assert clock.now == 1


def test_setup_dashboard_uses_selected_startup_timeout(monkeypatch):
    clock = _readiness_clock(monkeypatch)
    commands = []

    def execute(container, command, *, timeout):
        commands.append(command)
        return (0 if clock.now >= 2 else 1), "", ""

    monkeypatch.setattr(runtime_lifecycle, "container_exec", execute)
    assert runtime_lifecycle.maybe_finish_setup_mode(
        True, False, resolve_runtime_stack(), startup_timeout=3
    )
    assert clock.now == 2
    assert commands[-1] == [
        "curl",
        "-f",
        "-s",
        "--max-time",
        "1.000",
        "http://localhost:8700/healthz",
    ]


@pytest.mark.parametrize(
    "value", [0, -1, True, 1.5, float("nan"), float("inf"), "7200", 10**1000]
)
def test_startup_timeout_validation_rejects_invalid_direct_calls(value):
    with pytest.raises(ValueError, match="finite positive integer"):
        runtime_lifecycle.validate_startup_timeout(value)


def test_inspect_timeout_retries_within_startup_deadline(monkeypatch):
    clock = _readiness_clock(monkeypatch)
    inspections = []
    probes = []

    def inspect(container, *, timeout):
        inspections.append(timeout)
        if len(inspections) == 1:
            clock.sleep(timeout)
            raise RuntimeError("managed container status inspection failed")
        return "running"

    def execute(*args, **kwargs):
        probes.append(clock.now)
        return 0, "", ""

    monkeypatch.setattr(runtime_lifecycle, "container_status_strict", inspect)
    monkeypatch.setattr(runtime_lifecycle, "container_exec", execute)
    runtime_lifecycle.wait_for_router_health(
        resolve_runtime_stack(), startup_timeout=10
    )
    assert inspections == [5, 3]
    assert probes == [7]
    assert clock.now == 7


def test_inspection_failure_after_ready_cannot_report_success(monkeypatch):
    _readiness_clock(monkeypatch)
    inspections = []

    def inspect(container, *, timeout):
        inspections.append(container)
        if len(inspections) > 1:
            raise RuntimeError("managed container status inspection failed")
        return "running"

    monkeypatch.setattr(runtime_lifecycle, "container_status_strict", inspect)
    monkeypatch.setattr(
        runtime_lifecycle, "container_exec", lambda *a, **k: (0, "", "")
    )
    with pytest.raises(SystemExit) as exc:
        runtime_lifecycle.wait_and_verify_runtime(
            resolve_runtime_stack(), dashboard_disabled=True, startup_timeout=10
        )
    assert exc.value.code == 1
    assert len(inspections) == 2
