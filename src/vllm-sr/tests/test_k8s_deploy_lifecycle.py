"""Tests for K8s deployment, rollback cleanup, and Secret garbage collection."""

import json
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock

import pytest

TESTS_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_ROOT.parent
for path in (PROJECT_ROOT, TESTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from _k8s_test_helpers import (  # noqa: E402
    k8s_backend,
    managed_secret_payload,
)
from cli.config_translator import write_helm_values_file  # noqa: E402


def install_ordered_deploy_stubs(
    monkeypatch,
    backend,
    current_name,
    payload,
    events,
    written_values,
    values_directories,
    lock_state,
):
    """Install observability stubs for the lock/order contract test."""

    def get_secret(_):
        assert lock_state["held"] is True
        events.append(("snapshot", current_name))
        return payload

    @contextmanager
    def release_lock():
        assert lock_state["held"] is False
        lock_state["held"] = True
        events.append(("lock", "acquire"))
        try:
            yield
        finally:
            events.append(("lock", "release"))
            lock_state["held"] = False

    def assert_lock():
        assert lock_state["held"] is True

    def create(plan):
        assert lock_state["held"] is True
        events.append(("create", plan.active_name))
        return plan.active_name

    def run(cmd, **kwargs):
        assert lock_state["held"] is True
        events.append(("helm", cmd))
        return MagicMock(returncode=0)

    def garbage_collect():
        assert lock_state["held"] is True
        events.append(("gc", None))
        return True

    monkeypatch.setattr(backend, "_require_tool", lambda _: None)
    monkeypatch.setattr(backend, "_current_router_secret_refs", lambda: [current_name])
    monkeypatch.setattr(backend, "_get_secret_json", get_secret)
    monkeypatch.setattr(backend, "_release_operation_lock", release_lock)
    monkeypatch.setattr(backend, "_assert_release_operation_lock", assert_lock)
    monkeypatch.setattr("cli.k8s_backend.print_vllm_logo", lambda: None)

    def write_values(values, dest_dir):
        written_values.update(values)
        values_directories.append(Path(dest_dir))
        return write_helm_values_file(values, dest_dir)

    monkeypatch.setattr(
        "cli.k8s_backend.write_helm_values_file",
        write_values,
    )
    monkeypatch.setattr(backend, "_create_planned_secret", create)
    monkeypatch.setattr(backend, "_run", run)
    monkeypatch.setattr(backend, "_garbage_collect_managed_secrets", garbage_collect)
    monkeypatch.setattr(backend, "_wait_for_pods", lambda: None)
    monkeypatch.setattr(backend, "_log_k8s_summary", lambda: None)


class TestK8sBackend:
    def test_deploy_orders_create_atomic_helm_then_bounded_gc(
        self, monkeypatch, tmp_path
    ):
        backend = k8s_backend()
        config = tmp_path / "config.yaml"
        config.write_text("listeners: []\n")
        current_name = "alpha-vsr-env-current"
        old = "a" * 64
        new = "b" * 64
        payload = managed_secret_payload(
            current_name,
            "alpha",
            {"VLLM_SR_LOOPER_SHARED_SECRET": old},
        )
        events = []
        written_values = {}
        values_directories = []
        lock_state = {"held": False}
        install_ordered_deploy_stubs(
            monkeypatch,
            backend,
            current_name,
            payload,
            events,
            written_values,
            values_directories,
            lock_state,
        )

        backend.deploy(
            str(config),
            env_vars={"VLLM_SR_LOOPER_SHARED_SECRET": new},
        )

        assert [event[0] for event in events] == [
            "lock",
            "snapshot",
            "create",
            "helm",
            "gc",
            "lock",
        ]
        helm_cmd = events[3][1]
        assert "--atomic" in helm_cmd
        assert "--wait" in helm_cmd
        assert helm_cmd[helm_cmd.index("--history-max") + 1] == "10"
        assert written_values["runtimeCredentials"]["recreateForLooperRotation"] is True
        serialized_values = json.dumps(written_values)
        assert old not in serialized_values
        assert new not in serialized_values
        assert len(values_directories) == 1
        assert not values_directories[0].exists()

    def test_atomic_failure_deletes_only_new_secret_and_retains_old(
        self, monkeypatch, tmp_path
    ):
        backend = k8s_backend()
        config = tmp_path / "config.yaml"
        config.write_text("listeners: []\n")
        current_name = "alpha-vsr-env-current"
        payload = managed_secret_payload(
            current_name,
            "alpha",
            {"HF_TOKEN": "old"},
        )
        events = []
        values_directories = []
        monkeypatch.setattr(backend, "_require_tool", lambda _: None)
        monkeypatch.setattr(
            backend, "_current_router_secret_refs", lambda: [current_name]
        )
        monkeypatch.setattr(backend, "_get_secret_json", lambda _: payload)
        monkeypatch.setattr("cli.k8s_backend.print_vllm_logo", lambda: None)

        def write_values(values, dest_dir):
            values_directories.append(Path(dest_dir))
            return write_helm_values_file(values, dest_dir)

        monkeypatch.setattr(
            "cli.k8s_backend.write_helm_values_file",
            write_values,
        )

        def create(plan):
            events.append(("create", plan.active_name))
            return plan.active_name

        def fail_helm(cmd, **kwargs):
            events.append(("helm", cmd))
            raise subprocess.CalledProcessError(1, cmd)

        monkeypatch.setattr(backend, "_create_planned_secret", create)
        monkeypatch.setattr(backend, "_run", fail_helm)
        monkeypatch.setattr(
            backend,
            "_delete_managed_secret_if_unreferenced",
            lambda name: events.append(("delete", name)) or True,
        )

        with pytest.raises(subprocess.CalledProcessError):
            backend.deploy(str(config), env_vars={"HF_TOKEN": "new"})

        assert [event[0] for event in events] == ["create", "helm", "delete"]
        assert events[-1][1] == events[0][1]
        assert current_name not in [
            event[1] for event in events if event[0] == "delete"
        ]
        assert len(values_directories) == 1
        assert not values_directories[0].exists()

    def test_values_writer_exception_cleans_cli_owned_directory(
        self, monkeypatch, tmp_path
    ):
        backend = k8s_backend()
        config = tmp_path / "config.yaml"
        config.write_text("listeners: []\n")
        current_name = "alpha-vsr-env-current"
        payload = managed_secret_payload(
            current_name,
            "alpha",
            {"VLLM_SR_LOOPER_SHARED_SECRET": "a" * 64},
        )
        values_directories = []
        monkeypatch.setattr(backend, "_require_tool", lambda _: None)
        monkeypatch.setattr(
            backend, "_current_router_secret_refs", lambda: [current_name]
        )
        monkeypatch.setattr(backend, "_get_secret_json", lambda _: payload)
        monkeypatch.setattr("cli.k8s_backend.print_vllm_logo", lambda: None)

        def fail_write(_values, dest_dir):
            values_directory = Path(dest_dir)
            values_directories.append(values_directory)
            (values_directory / "partial-values.yaml").write_text("partial")
            raise RuntimeError("values serialization failed")

        monkeypatch.setattr(
            "cli.k8s_backend.write_helm_values_file",
            fail_write,
        )

        with pytest.raises(RuntimeError, match="values serialization failed"):
            backend.deploy(str(config), env_vars=None)

        assert len(values_directories) == 1
        assert not values_directories[0].exists()

    def test_cleanup_rechecks_latest_rollback_references(self, monkeypatch):
        backend = k8s_backend()
        name = "alpha-vsr-env-concurrent"
        refs = [name]
        deleted = []
        monkeypatch.setattr(backend, "_protected_secret_refs", lambda: set(refs))
        monkeypatch.setattr(
            backend,
            "_delete_secret_if_exists",
            lambda value, _candidate: deleted.append(value) or True,
        )

        candidate = MagicMock()
        backend._delete_managed_secret_if_unreferenced(name, candidate)
        refs.clear()
        backend._delete_managed_secret_if_unreferenced(name, candidate)

        assert deleted == [name]

    def test_omitted_optional_credentials_reuse_exact_generation(
        self, monkeypatch, tmp_path
    ):
        backend = k8s_backend()
        config = tmp_path / "config.yaml"
        config.write_text("listeners: []\n")
        current_name = "alpha-vsr-env-current"
        payload = managed_secret_payload(
            current_name,
            "alpha",
            {"VLLM_SR_LOOPER_SHARED_SECRET": "a" * 64},
        )
        events = []
        values = {}
        monkeypatch.setattr(backend, "_require_tool", lambda _: None)
        monkeypatch.setattr(
            backend, "_current_router_secret_refs", lambda: [current_name]
        )
        monkeypatch.setattr(backend, "_get_secret_json", lambda _: payload)
        monkeypatch.setattr("cli.k8s_backend.print_vllm_logo", lambda: None)
        monkeypatch.setattr(
            "cli.k8s_backend.write_helm_values_file",
            lambda rendered, _dest_dir: (values.update(rendered) or "/tmp/values.yaml"),
        )

        def create_secret(plan):
            assert plan.active_name == current_name
            assert plan.creates_secret is False

        monkeypatch.setattr(backend, "_create_planned_secret", create_secret)
        monkeypatch.setattr(
            backend,
            "_run",
            lambda cmd, **kwargs: events.append("helm") or MagicMock(returncode=0),
        )
        monkeypatch.setattr(
            backend,
            "_garbage_collect_managed_secrets",
            lambda: events.append("gc") or True,
        )
        monkeypatch.setattr(backend, "_wait_for_pods", lambda: None)
        monkeypatch.setattr(backend, "_log_k8s_summary", lambda: None)

        backend.deploy(str(config), env_vars=None)

        assert events == ["helm", "gc"]
        assert values["runtimeCredentials"] == {
            "generation": current_name,
            "recreateForLooperRotation": True,
        }
