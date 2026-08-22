"""Kubernetes credential Secret rotation and cleanup transaction tests."""

from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock

import pytest
from cli.k8s_env_secret import (
    LEGACY_ENV_SECRET_NAME,
    EnvSecretPlan,
    env_secret_owner,
)
from k8s_env_secret_test_support import _backend, _patch_deploy_io, _plan


def test_secret_create_uses_stdin_without_argv_or_log_leak(monkeypatch, caplog):
    backend = _backend()
    canary = "credential-value-canary"
    plan = EnvSecretPlan(
        owner=env_secret_owner(backend.namespace, backend.release_name),
        name=_plan(backend).name,
        manifest=json.dumps({"stringData": {"TOKEN": canary}}),
        key_count=1,
        keys=frozenset({"TOKEN"}),
    )
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    with caplog.at_level("DEBUG", logger="cli.k8s_backend"):
        backend._create_env_secret(plan)

    assert calls == [
        (
            ["kubectl", "create", "-f", "-"],
            {
                "check": True,
                "input": plan.manifest,
                "text": True,
                "capture_output": False,
            },
        )
    ]
    assert canary not in repr(calls[0][0])
    assert canary not in caplog.text


def test_secret_inventory_uses_only_metadata_names(monkeypatch):
    backend = _backend()
    valid = _plan(backend).name
    commands = []

    def fake_run(cmd, **kwargs):
        commands.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 0, stdout=f"{valid}\nignored\n")

    monkeypatch.setattr(backend, "_run", fake_run)

    assert backend._list_managed_env_secrets() == {valid}
    command, kwargs = commands[0]
    assert command[command.index("-o") + 1].startswith("jsonpath=")
    assert "json" not in command
    assert kwargs == {"capture_output": True}


def test_release_reference_query_uses_router_metadata_only(monkeypatch):
    backend = _backend()
    valid = _plan(backend).name
    commands = []

    def fake_run(cmd, **kwargs):
        commands.append((cmd, kwargs))
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout=f"{valid}\n{LEGACY_ENV_SECRET_NAME}\noperator-db\n",
        )

    monkeypatch.setattr(backend, "_run", fake_run)

    assert backend._current_release_env_secret_refs() == {
        valid,
        LEGACY_ENV_SECRET_NAME,
    }
    command, kwargs = commands[0]
    selector = command[command.index("-l") + 1]
    assert "app.kubernetes.io/instance=test-release" in selector
    assert "app.kubernetes.io/component=router" in selector
    assert command[command.index("-o") + 1].startswith("jsonpath=")
    assert kwargs == {"capture_output": True}


def test_successful_rotation_stages_helm_verifies_then_cleans(monkeypatch, tmp_path):
    backend = _backend()
    plan = _plan(backend)
    old = _plan(backend, "b" * 32).name
    events: list[str] = []
    helm_commands = _patch_deploy_io(
        monkeypatch,
        backend,
        tmp_path,
        plan=plan,
        current_refs=[{old}, {plan.name}],
        previous_managed={old},
        events=events,
    )

    backend.deploy("config.yaml", env_vars={"HF_TOKEN": "secret"})

    assert events == [
        "ensure",
        "current",
        "list",
        "create",
        "helm",
        "current",
        "cleanup",
        "wait",
        "summary",
    ]
    assert "--atomic" in helm_commands[0]
    assert "--cleanup-on-fail" in helm_commands[0]
    assert "--wait" in helm_commands[0]


def test_create_failure_cleans_only_staged_revision_and_never_calls_helm(
    monkeypatch, tmp_path
):
    backend = _backend()
    plan = _plan(backend)
    events: list[str] = []
    helm_commands = _patch_deploy_io(
        monkeypatch,
        backend,
        tmp_path,
        plan=plan,
        current_refs=[set()],
        previous_managed=set(),
        events=events,
    )
    monkeypatch.setattr(
        backend,
        "_create_env_secret",
        lambda _plan: (_ for _ in ()).throw(
            subprocess.CalledProcessError(1, ["kubectl", "create"])
        ),
    )
    deleted = []
    monkeypatch.setattr(
        backend,
        "_delete_secret_if_exists",
        lambda name: deleted.append(name) or True,
    )

    with pytest.raises(subprocess.CalledProcessError):
        backend.deploy("config.yaml", env_vars={"HF_TOKEN": "secret"})

    assert helm_commands == []
    assert deleted == [plan.name]
    assert "cleanup" not in events


def test_helm_failure_is_atomic_and_cleans_staged_without_old_secret_cleanup(
    monkeypatch, tmp_path
):
    backend = _backend()
    plan = _plan(backend)
    old = _plan(backend, "b" * 32).name
    events: list[str] = []
    helm_commands = _patch_deploy_io(
        monkeypatch,
        backend,
        tmp_path,
        plan=plan,
        current_refs=[{old}],
        previous_managed={old},
        events=events,
    )

    def fail_helm(cmd, **_kwargs):
        helm_commands.append(cmd)
        events.append("helm")
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(backend, "_run", fail_helm)
    monkeypatch.setattr(backend, "_namespace_references_secret", lambda _name: False)
    deleted = []
    monkeypatch.setattr(
        backend,
        "_delete_secret_if_exists",
        lambda name: deleted.append(name) or True,
    )

    with pytest.raises(subprocess.CalledProcessError):
        backend.deploy("config.yaml", env_vars={"HF_TOKEN": "secret"})

    assert "--atomic" in helm_commands[0]
    assert "--cleanup-on-fail" in helm_commands[0]
    assert deleted == [plan.name]
    assert old not in deleted
    assert "cleanup" not in events


def test_helm_failure_preserves_staged_secret_when_rollback_left_a_reference(
    monkeypatch, tmp_path, caplog
):
    backend = _backend()
    plan = _plan(backend)
    events: list[str] = []
    helm_commands = _patch_deploy_io(
        monkeypatch,
        backend,
        tmp_path,
        plan=plan,
        current_refs=[set()],
        previous_managed=set(),
        events=events,
    )

    def fail_helm(cmd, **_kwargs):
        helm_commands.append(cmd)
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(backend, "_run", fail_helm)
    monkeypatch.setattr(backend, "_namespace_references_secret", lambda _name: True)
    deleted = MagicMock(return_value=True)
    monkeypatch.setattr(backend, "_delete_secret_if_exists", deleted)

    with (
        caplog.at_level("WARNING", logger="cli.k8s_backend"),
        pytest.raises(subprocess.CalledProcessError),
    ):
        backend.deploy("config.yaml", env_vars={"HF_TOKEN": "secret"})

    deleted.assert_not_called()
    assert "Failed to remove staged Kubernetes credential Secret" in caplog.text


def test_helm_failure_is_not_masked_when_staged_cleanup_also_fails(
    monkeypatch, tmp_path, caplog
):
    backend = _backend()
    plan = _plan(backend)
    events: list[str] = []
    _patch_deploy_io(
        monkeypatch,
        backend,
        tmp_path,
        plan=plan,
        current_refs=[set()],
        previous_managed=set(),
        events=events,
    )
    monkeypatch.setattr(
        backend,
        "_run",
        MagicMock(side_effect=RuntimeError("primary helm failure")),
    )
    monkeypatch.setattr(
        backend,
        "_delete_secret_if_exists",
        MagicMock(side_effect=RuntimeError("secondary cleanup failure")),
    )

    with (
        caplog.at_level("WARNING", logger="cli.k8s_backend"),
        pytest.raises(RuntimeError, match="primary helm failure"),
    ):
        backend.deploy("config.yaml", env_vars={"HF_TOKEN": "secret"})

    assert "Failed to remove staged Kubernetes credential Secret" in caplog.text
    assert "secondary cleanup failure" not in caplog.text


def test_post_helm_verification_failure_preserves_all_secrets(monkeypatch, tmp_path):
    backend = _backend()
    plan = _plan(backend)
    old = _plan(backend, "b" * 32).name
    events: list[str] = []
    _patch_deploy_io(
        monkeypatch,
        backend,
        tmp_path,
        plan=plan,
        current_refs=[{old}, {old}],
        previous_managed={old},
        events=events,
    )
    deleted = []
    monkeypatch.setattr(
        backend,
        "_delete_secret_if_exists",
        lambda name: deleted.append(name) or True,
    )

    with pytest.raises(RuntimeError, match="could not be verified"):
        backend.deploy("config.yaml", env_vars={"HF_TOKEN": "secret"})

    assert events[-1] == "current"
    assert deleted == []


def test_removing_all_credentials_commits_helm_before_old_cleanup(
    monkeypatch, tmp_path
):
    backend = _backend()
    old = _plan(backend).name
    events: list[str] = []
    _patch_deploy_io(
        monkeypatch,
        backend,
        tmp_path,
        plan=None,
        current_refs=[{old}, set()],
        previous_managed={old},
        events=events,
    )

    backend.deploy("config.yaml")

    assert "create" not in events
    assert events.index("helm") < events.index("cleanup")


def test_cleanup_deletes_only_snapshot_revisions_not_currently_referenced(
    monkeypatch,
):
    backend = _backend()
    old = _plan(backend).name
    active = _plan(backend, "b" * 32).name
    concurrent = _plan(backend, "c" * 32).name
    deleted = []
    monkeypatch.setattr(
        backend,
        "_namespace_references_secret",
        lambda name: name == active,
    )
    monkeypatch.setattr(
        backend,
        "_delete_secret_if_exists",
        lambda name, required=False: deleted.append((name, required)) or True,
    )

    backend._cleanup_obsolete_env_secrets(
        active_secret_name=active,
        previous_secret_refs={old},
        previous_managed_secrets={old, active},
    )

    assert deleted == [(old, True)]
    assert concurrent not in {name for name, _required in deleted}


def test_legacy_secret_is_preserved_while_any_namespace_deployment_references_it(
    monkeypatch,
):
    backend = _backend()
    deleted = MagicMock(return_value=True)
    monkeypatch.setattr(backend, "_namespace_references_secret", lambda _name: True)
    monkeypatch.setattr(backend, "_delete_secret_if_exists", deleted)

    backend._cleanup_obsolete_env_secrets(
        active_secret_name=None,
        previous_secret_refs={LEGACY_ENV_SECRET_NAME},
        previous_managed_secrets=set(),
    )

    deleted.assert_not_called()


def test_legacy_secret_is_deleted_only_after_namespace_references_are_gone(
    monkeypatch,
):
    backend = _backend()
    deleted = []
    monkeypatch.setattr(backend, "_namespace_references_secret", lambda _name: False)
    monkeypatch.setattr(
        backend,
        "_delete_secret_if_exists",
        lambda name, required=False: deleted.append((name, required)) or True,
    )

    backend._cleanup_obsolete_env_secrets(
        active_secret_name=None,
        previous_secret_refs={LEGACY_ENV_SECRET_NAME},
        previous_managed_secrets=set(),
    )

    assert deleted == [(LEGACY_ENV_SECRET_NAME, True)]


def test_cleanup_failure_is_reported_after_helm_success(monkeypatch):
    backend = _backend()
    old = _plan(backend).name
    monkeypatch.setattr(backend, "_namespace_references_secret", lambda _name: False)
    monkeypatch.setattr(
        backend,
        "_delete_secret_if_exists",
        MagicMock(side_effect=RuntimeError("cleanup failed")),
    )

    with pytest.raises(RuntimeError, match="cleanup failed"):
        backend._cleanup_obsolete_env_secrets(
            active_secret_name=None,
            previous_secret_refs={old},
            previous_managed_secrets={old},
        )
