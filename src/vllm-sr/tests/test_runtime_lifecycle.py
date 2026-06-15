import pytest
from cli import runtime_lifecycle
from cli.runtime_stack import resolve_runtime_stack


def test_wait_for_router_health_fails_fast_when_router_exits(monkeypatch):
    calls = {"exec": 0, "logs": 0}

    monkeypatch.setattr(
        runtime_lifecycle, "_emit_router_startup_logs", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        runtime_lifecycle, "docker_container_status", lambda _name: "exited"
    )

    def fake_exec(*_args, **_kwargs):
        calls["exec"] += 1
        return 1, "", ""

    def fake_logs(*_args, **_kwargs):
        calls["logs"] += 1

    monkeypatch.setattr(runtime_lifecycle, "docker_exec", fake_exec)
    monkeypatch.setattr(runtime_lifecycle, "docker_logs", fake_logs)

    with pytest.raises(SystemExit):
        runtime_lifecycle.wait_for_router_health(resolve_runtime_stack())

    assert calls["exec"] == 0
    assert calls["logs"] == 1
