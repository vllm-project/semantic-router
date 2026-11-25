"""A restart must resolve its images before it stops the stack it replaces.

The serve sequence used to stop and remove the running Router first and pull
the replacement images afterwards. With the default `always` pull policy a
registry it cannot read therefore left the stack with no Router at all, even
though the image it had been serving was still in the local cache. These tests
pin the order the sequence now uses, and that a failed pull leaves the running
stack untouched.
"""

import pytest

from cli import core
from cli.gateway_mode import GATEWAY_EXTPROC


class _AfterCleanup(Exception):
    """Raised where the serve sequence stops for these tests."""


class _StubStack:
    """The layout fields the serve sequence reads before it starts containers."""

    stack_name = "vllm-sr"
    network_name = "vllm-sr-net"
    data_network_name = "vllm-sr-data"
    runtime_container_names = ("vllm-sr-router", "vllm-sr-dashboard", "vllm-sr-envoy")

    def host_port(self, port, name=None):
        return port


def _stub_serve_prefix(monkeypatch, tmp_path, events):
    """Patch everything the serve sequence touches before it starts containers."""
    monkeypatch.setattr(
        core,
        "_preflight_runtime_config",
        lambda *_args, **_kwargs: ({}, [{"name": "main", "port": 8888}]),
    )
    monkeypatch.setattr(core, "read_pending_activation", lambda *_a, **_k: None)
    monkeypatch.setattr(core, "clear_pending_activation", lambda *_a, **_k: None)
    monkeypatch.setattr(core, "_configured_management_port", lambda _config: 8080)
    monkeypatch.setattr(core, "_readiness_token_env", lambda *_a, **_k: None)
    monkeypatch.setattr(core, "log_startup_banner", lambda *_a, **_k: None)
    monkeypatch.setattr(core, "resolve_state_root_dir", lambda *_a, **_k: str(tmp_path))
    monkeypatch.setattr(
        core, "ensure_shared_network", lambda _name: events.append("network")
    )
    monkeypatch.setattr(core, "ensure_data_network", lambda _name: None)
    monkeypatch.setattr(
        core,
        "ensure_clean_runtime_container",
        lambda name: events.append(("clean", name)),
    )

    def stop_after_cleanup(*_args, **_kwargs):
        raise _AfterCleanup

    monkeypatch.setattr(core, "_start_support_services", stop_after_cleanup)


def _run_locked(monkeypatch, tmp_path, events, **overrides):
    kwargs = {
        "source_config_file": "config.yaml",
        "runtime_config_file": "config.yaml",
        "env_vars": {},
        "stack_layout": _StubStack(),
        "runtime_topology": "split",
        "state_root_dir": str(tmp_path),
        "image": None,
        "router_image": None,
        "envoy_image": None,
        "dashboard_image": None,
        "pull_policy": "always",
        "enable_observability": False,
        "gateway": GATEWAY_EXTPROC,
    }
    kwargs.update(overrides)
    return core._start_vllm_sr_locked(**kwargs)


def test_images_are_resolved_before_the_running_stack_is_stopped(monkeypatch, tmp_path):
    events = []
    resolved = {}

    def fake_get_runtime_images(**kwargs):
        events.append("images")
        resolved.update(kwargs)
        return {"router": "router:test"}

    monkeypatch.setattr(core, "get_runtime_images", fake_get_runtime_images)
    _stub_serve_prefix(monkeypatch, tmp_path, events)

    with pytest.raises(_AfterCleanup):
        _run_locked(monkeypatch, tmp_path, events)

    assert "images" in events, "the serve sequence must resolve its images"
    cleaned = [event for event in events if isinstance(event, tuple)]
    assert cleaned, "the serve sequence must still clean the runtime containers"
    for name in _StubStack.runtime_container_names:
        assert ("clean", name) in events
    # Every image is resolved before any container is stopped or removed.
    assert events.index("images") < events.index(("clean", "vllm-sr-router"))


def test_a_failed_image_pull_leaves_the_running_stack_alone(monkeypatch, tmp_path):
    events = []

    def failing_pull(**_kwargs):
        events.append("images")
        raise SystemExit(1)

    monkeypatch.setattr(core, "get_runtime_images", failing_pull)
    _stub_serve_prefix(monkeypatch, tmp_path, events)

    with pytest.raises(SystemExit):
        _run_locked(monkeypatch, tmp_path, events)

    assert "images" in events, "the serve sequence must resolve its images"
    cleaned = [event for event in events if isinstance(event, tuple)]
    assert (
        cleaned == []
    ), "a failed pull must stop the command before it stops or removes anything"


def test_preflight_carries_the_requested_pull_policy(monkeypatch, tmp_path):
    events = []
    resolved = {}

    def fake_get_runtime_images(**kwargs):
        events.append("images")
        resolved.update(kwargs)
        return {"router": "router:test"}

    monkeypatch.setattr(core, "get_runtime_images", fake_get_runtime_images)
    _stub_serve_prefix(monkeypatch, tmp_path, events)

    for policy in ("always", "ifnotpresent", "never"):
        resolved.clear()
        with pytest.raises(_AfterCleanup):
            _run_locked(monkeypatch, tmp_path, events, pull_policy=policy)
        assert resolved["pull_policy"] == policy
