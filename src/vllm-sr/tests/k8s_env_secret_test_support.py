"""Shared fixtures for Kubernetes credential Secret tests."""

from __future__ import annotations

import subprocess
from pathlib import Path

from cli.k8s_backend import K8sBackend
from cli.k8s_env_secret import EnvSecretPlan, env_secret_owner


def _backend(*, namespace: str = "test-ns", release: str = "test-release"):
    backend = K8sBackend.__new__(K8sBackend)
    backend.namespace = namespace
    backend.context = None
    backend.release_name = release
    backend.profile = None
    backend.chart_dir = "/chart"
    return backend


def _plan(backend: K8sBackend, revision: str = "a" * 32) -> EnvSecretPlan:
    owner = env_secret_owner(backend.namespace, backend.release_name)
    name = f"vllm-sr-env-{owner}-{revision}"
    return EnvSecretPlan(
        owner=owner,
        name=name,
        manifest="{}",
        key_count=1,
        keys=frozenset({"HF_TOKEN"}),
    )


def _patch_deploy_io(
    monkeypatch,
    backend: K8sBackend,
    tmp_path: Path,
    *,
    plan: EnvSecretPlan | None,
    current_refs: list[set[str]],
    previous_managed: set[str],
    events: list[str],
) -> list[list[str]]:
    helm_commands: list[list[str]] = []
    refs = iter(current_refs)
    monkeypatch.setattr(backend, "_require_tool", lambda _name: None)
    monkeypatch.setattr("cli.k8s_backend.print_vllm_logo", lambda: None)
    monkeypatch.setattr(backend, "_plan_env_secret", lambda *_args: plan)
    monkeypatch.setattr("cli.k8s_backend.load_profile_values", lambda *_args: None)
    monkeypatch.setattr(
        "cli.k8s_backend.translate_config_to_helm_values",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        "cli.config_translator.write_helm_values_file",
        lambda _values, _dest_dir: str(tmp_path / "values.yaml"),
    )
    monkeypatch.setattr(backend, "_ensure_namespace", lambda: events.append("ensure"))

    def release_refs():
        events.append("current")
        return next(refs)

    monkeypatch.setattr(backend, "_current_release_env_secret_refs", release_refs)
    monkeypatch.setattr(
        backend,
        "_list_managed_env_secrets",
        lambda: events.append("list") or set(previous_managed),
    )
    monkeypatch.setattr(
        backend,
        "_create_env_secret",
        lambda _plan: events.append("create"),
    )

    def run(cmd, **_kwargs):
        if cmd[0] == "helm":
            events.append("helm")
            helm_commands.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="")

    monkeypatch.setattr(backend, "_run", run)
    monkeypatch.setattr(
        backend,
        "_cleanup_obsolete_env_secrets",
        lambda **_kwargs: events.append("cleanup"),
    )
    monkeypatch.setattr(backend, "_wait_for_pods", lambda: events.append("wait"))
    monkeypatch.setattr(backend, "_log_k8s_summary", lambda: events.append("summary"))
    return helm_commands
