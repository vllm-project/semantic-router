"""Exercise listener TLS preflight through the public Docker serve command."""

from types import SimpleNamespace

import pytest
import yaml
from cli import (
    container_backend,
    container_start,
    container_start_runner,
    core,
    runtime_lifecycle,
)
from cli.commands import runtime_serve_config
from cli.commands.runtime_paths import (
    _runtime_config_output_path,
    _runtime_config_provenance_path,
    materialize_runtime_config,
)
from cli.commands.runtime_support import build_effective_config_bytes
from cli.main import main
from cli.runtime_stack import resolve_runtime_stack
from click.testing import CliRunner


@pytest.mark.parametrize(
    ("missing", "active_state"),
    [
        ("cert_file", "different"),
        ("key_file", "different"),
        ("cert_file", "unchanged"),
        ("cert_file", "absent"),
        ("cert_file", "cold"),
        ("cert_file", "during-validation"),
        (None, "different"),
    ],
)
def test_serve_checks_listener_tls_before_replacing_active_config(
    tmp_path, monkeypatch, missing, active_state
):
    state_dir = tmp_path / "state"
    monkeypatch.setenv("VLLM_SR_STATE_ROOT_DIR", str(state_dir))
    monkeypatch.setenv("VLLM_SR_STACK_NAME", "tls-replace")
    monkeypatch.setenv("VLLM_SR_TOPOLOGY", "split")

    source = tmp_path / "config.yaml"
    document = {
        "version": "v0.5",
        "listeners": [{"name": "public", "address": "0.0.0.0", "port": 8899}],
        "global": {"services": {"observability": {"tracing": {"enabled": False}}}},
    }
    tls_files = {"cert_file": "certs/tls.crt", "key_file": "certs/tls.key"}
    if active_state == "unchanged":
        document["listeners"][0]["tls"] = tls_files
    source.write_bytes(yaml.safe_dump(document).encode())
    old = build_effective_config_bytes(source, None, False)
    stack = resolve_runtime_stack()
    active = _runtime_config_output_path(
        source, state_root_dir=state_dir, stack_name=stack.stack_name
    )
    if active_state != "cold":
        materialize_runtime_config(
            source, old, state_root_dir=state_dir, stack_name=stack.stack_name
        )
    if active_state == "absent":
        active.unlink()
    provenance = _runtime_config_provenance_path(active)
    old_provenance = provenance.read_bytes() if provenance.exists() else None

    document["listeners"][0]["tls"] = tls_files
    candidate = yaml.safe_dump(document).encode()
    source.write_bytes(candidate)
    (tmp_path / "certs").mkdir()
    for field, filename in tls_files.items():
        if field != missing or active_state == "during-validation":
            (tmp_path / filename).write_text("fixture", encoding="utf-8")
    if active_state == "during-validation":
        monkeypatch.setattr(
            runtime_serve_config, "is_setup_mode_config", lambda _path: False
        )

        def run_image_check(*_args):
            (tmp_path / tls_files[missing]).unlink()
            return []

        monkeypatch.setattr(runtime_serve_config, "_router_refusal", run_image_check)

    states = {} if active_state == "cold" else {stack.router_container_name: "running"}
    commands = []

    def status(name):
        return states.get(name, "not found")

    def stop(name):
        assert active.read_bytes() == old
        commands.append("stop")
        states[name] = "exited"
        return True

    def remove(name):
        commands.append("remove")
        del states[name]
        return True

    def docker(command, **_kwargs):
        assert command[0] == "docker"
        if command[1] == "create":
            name = command[command.index("--name") + 1]
            assert name not in states
            commands.append("create")
            states[name] = "created"
        elif command[1] == "start":
            name = command[2]
            assert states[name] == "created"
            commands.append("start")
            states[name] = "running"
        else:
            assert command[1:3] == ["network", "connect"]
        return SimpleNamespace(stdout="fixture-container\n", stderr="", returncode=0)

    for module in (core, runtime_lifecycle, container_backend):
        monkeypatch.setattr(module, "container_status", status)
    for module in (core, runtime_lifecycle):
        monkeypatch.setattr(module, "container_status_strict", status)
    for module in (runtime_lifecycle, container_start, container_backend):
        monkeypatch.setattr(module, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(runtime_lifecycle, "container_stop_container", stop)
    monkeypatch.setattr(runtime_lifecycle, "container_remove_container", remove)
    monkeypatch.setattr(
        container_start, "get_runtime_images", lambda **_: {"router": "fixture:local"}
    )
    monkeypatch.setattr(
        container_start_runner, "subprocess", SimpleNamespace(run=docker)
    )
    monkeypatch.setattr(container_backend, "attach_controller", lambda *args: None)
    monkeypatch.setattr(
        core, "_prepare_runtime_network", lambda *a, **k: ("shared", str(state_dir))
    )
    monkeypatch.setattr(
        core, "_start_support_services", lambda *a, **k: ([], "runtime")
    )
    monkeypatch.setattr(core, "connect_runtime_container", lambda *a: None)
    monkeypatch.setattr(core, "_wait_and_verify_runtime", lambda *a, **k: None)

    result = CliRunner().invoke(
        main,
        [
            "serve",
            "--target",
            "docker",
            "--gateway",
            "standalone",
            "--minimal",
            "--platform",
            "cpu",
            "--config",
            str(source),
            "--replace-active-config",
        ],
    )

    if missing:
        assert result.exit_code == 1, result.output
        assert f"listener 'public': tls.{missing}" in result.output
        assert commands == []
        if active_state in {"absent", "cold"}:
            assert not active.exists()
            assert (
                provenance.read_bytes() if provenance.exists() else None
            ) == old_provenance
        else:
            assert active.read_bytes() == old
    else:
        assert result.exit_code == 0, result.output
        assert commands == ["stop", "remove", "create", "start"]
        assert yaml.safe_load(active.read_bytes())["listeners"][0]["tls"] == tls_files
    assert states.get(stack.router_container_name) == (
        None if active_state == "cold" else "running"
    )
