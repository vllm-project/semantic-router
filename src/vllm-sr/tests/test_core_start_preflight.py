import pytest
from cli import core


def _write_config(tmp_path, listener_address, *, listener_port=8899):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "version: v0.1\nlisteners:\n"
        "  - name: http-8899\n"
        f"    address: {listener_address}\n"
        f"    port: {listener_port}\n"
    )
    return config_path


def _guard_startup_mutations(monkeypatch):
    calls = []

    def guard(name):
        def fail_if_called(*args, **kwargs):
            calls.append(name)
            pytest.fail(f"startup mutation reached before preflight: {name}")

        return fail_if_called

    for name in (
        "ensure_clean_runtime_container",
        "_prepare_runtime_network",
        "_start_support_services",
        "_start_runtime_containers",
        "connect_runtime_container",
        "maybe_finish_setup_mode",
        "_wait_and_verify_runtime",
        "recover_openclaw_containers",
        "log_runtime_summary",
    ):
        monkeypatch.setattr(core, name, guard(name))
    monkeypatch.setattr(core, "print_vllm_logo", lambda: None)
    return calls


@pytest.mark.parametrize(
    (
        "listener_address",
        "listener_port",
        "internal_bind",
        "port_offset",
        "error_pattern",
    ),
    [
        ("0.0.0.0", 8899, "localhost:8080", None, "VLLM_SR_INTERNAL_BIND_ADDRESS"),
        ("localhost", 8899, None, None, "listener 'http-8899' address"),
        ("0.0.0.0", 8899, None, "60000", "router extproc host port"),
        ("0.0.0.0", 65000, None, "1000", "listener 'http-8899' host port"),
        ("0.0.0.0", 8700, None, None, "host port publication conflict"),
    ],
)
def test_public_restart_preflight_rejects_invalid_publication_before_mutation(
    tmp_path,
    monkeypatch,
    listener_address,
    listener_port,
    internal_bind,
    port_offset,
    error_pattern,
):
    config_path = _write_config(tmp_path, listener_address, listener_port=listener_port)
    calls = _guard_startup_mutations(monkeypatch)
    monkeypatch.setenv("VLLM_SR_TOPOLOGY", "split")
    if internal_bind is None:
        monkeypatch.delenv("VLLM_SR_INTERNAL_BIND_ADDRESS", raising=False)
    else:
        monkeypatch.setenv("VLLM_SR_INTERNAL_BIND_ADDRESS", internal_bind)
    if port_offset is None:
        monkeypatch.delenv("VLLM_SR_PORT_OFFSET", raising=False)
    else:
        monkeypatch.setenv("VLLM_SR_PORT_OFFSET", port_offset)

    with pytest.raises(ValueError, match=error_pattern):
        core.start_vllm_sr(str(config_path), env_vars={}, enable_observability=False)

    assert calls == []
    assert not (tmp_path / ".vllm-sr").exists()


def test_public_restart_preflight_validates_every_listener_before_mutation(
    tmp_path, monkeypatch
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "version: v0.1\nlisteners:\n"
        "  - name: public\n"
        "    address: 0.0.0.0\n"
        "    port: 8899\n"
        "  - name: invalid-second-listener\n"
        "    address: localhost\n"
        "    port: 8900\n"
    )
    calls = _guard_startup_mutations(monkeypatch)
    monkeypatch.setenv("VLLM_SR_TOPOLOGY", "split")
    monkeypatch.delenv("VLLM_SR_INTERNAL_BIND_ADDRESS", raising=False)
    monkeypatch.delenv("VLLM_SR_PORT_OFFSET", raising=False)

    with pytest.raises(ValueError, match="invalid-second-listener"):
        core.start_vllm_sr(str(config_path), env_vars={}, enable_observability=False)

    assert calls == []


@pytest.mark.parametrize("kind", ["missing", "directory"])
def test_public_start_rejects_invalid_dashboard_blocklist_before_mutation(
    tmp_path, monkeypatch, kind
):
    config_path = _write_config(tmp_path, "0.0.0.0")
    blocklist = tmp_path / kind
    if kind == "directory":
        blocklist.mkdir()
    calls = _guard_startup_mutations(monkeypatch)
    monkeypatch.setenv("VLLM_SR_TOPOLOGY", "split")

    with pytest.raises(ValueError, match="DASHBOARD_PASSWORD_BLOCKLIST_PATH"):
        core.start_vllm_sr(
            str(config_path),
            env_vars={"DASHBOARD_PASSWORD_BLOCKLIST_PATH": str(blocklist)},
            enable_observability=False,
        )

    assert calls == []
    assert not (tmp_path / ".vllm-sr").exists()


def test_public_start_passes_one_normalized_immutable_plan_downstream(
    tmp_path, monkeypatch
):
    config_path = _write_config(tmp_path, '" 127.0.0.1 "')
    received_plans = []
    received_auth_plans = []
    support_envs = []
    monkeypatch.setenv("VLLM_SR_TOPOLOGY", "split")
    monkeypatch.setenv("VLLM_SR_PORT_OFFSET", "200")
    monkeypatch.setenv("VLLM_SR_INTERNAL_BIND_ADDRESS", " ::1 ")
    monkeypatch.setattr(core, "print_vllm_logo", lambda: None)
    monkeypatch.setattr(core, "ensure_clean_runtime_container", lambda _name: None)
    monkeypatch.setattr(
        core,
        "_prepare_runtime_network",
        lambda *args: ("test-network", str(tmp_path)),
    )

    def capture_support_services(*args):
        plan = args[-1]
        received_plans.append(plan)
        support_envs.append(args[3])
        monkeypatch.setenv("VLLM_SR_INTERNAL_BIND_ADDRESS", "invalid-after-preflight")
        return set(), "test-network", False

    def capture_runtime_containers(*args):
        received_plans.append(args[-1])
        received_auth_plans.append(args[-2])
        return 0, "", ""

    monkeypatch.setattr(core, "_start_support_services", capture_support_services)
    monkeypatch.setattr(core, "_start_runtime_containers", capture_runtime_containers)
    monkeypatch.setattr(core, "connect_runtime_container", lambda *args: None)
    monkeypatch.setattr(core, "maybe_finish_setup_mode", lambda *args: True)

    core.start_vllm_sr(
        str(config_path),
        env_vars={
            "DASHBOARD_JWT_SECRET": "jwt-secret-value",
            "DASHBOARD_ADMIN_PASSWORD": "admin-password-value",
        },
        enable_observability=False,
    )

    assert len(received_plans) == 2
    assert received_plans[0] is received_plans[1]
    plan = received_plans[0]
    assert plan.internal_bind_address == "::1"
    assert plan.listeners[0].bind_address == "127.0.0.1"
    assert plan.listeners[0].container_port == 8899
    assert plan.listeners[0].host_port == 9099
    assert len(received_auth_plans) == 1
    assert received_auth_plans[0].container_env["DASHBOARD_JWT_SECRET"] == (
        "jwt-secret-value"
    )
    assert "DASHBOARD_JWT_SECRET" not in support_envs[0]
    assert "DASHBOARD_ADMIN_PASSWORD" not in support_envs[0]
