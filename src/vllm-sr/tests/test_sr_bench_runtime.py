"""The benchmark worker survives UI/runtime reloads without resending work."""

from __future__ import annotations

import json
import sqlite3
import stat
import subprocess
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from cli import container_start, container_start_runner, core, runtime_lifecycle
from cli.commands.benchmark import benchmark
from cli.runtime_stack import resolve_runtime_stack
from cli.sr_bench import client
from cli.sr_bench_runtime import (
    BENCH_IDENTITY_LABEL,
    _remove_idle_bench_container,
    dashboard_bench_env,
    prepare_bench_runtime,
    reconcile_bench_container,
)
from click.testing import CliRunner


def test_managed_service_token_is_private_stable_and_not_in_common_mounts(tmp_path):
    stack = resolve_runtime_stack(stack_name="bench-test", port_offset=1000)
    first = prepare_bench_runtime(str(tmp_path), stack, {})
    second = prepare_bench_runtime(str(tmp_path), stack, {})
    assert first == second
    assert first.store == tmp_path / ".sr-bench/bench-test/store"
    assert len(first.secrets[first.token_env]) >= 32
    assert stat.S_IMODE((first.store.parent / "service-token").stat().st_mode) == 0o600
    assert str(first.store).find(".vllm-sr") == -1
    assert stack.sr_bench_port == 9090
    assert stack.sr_bench_container_name not in stack.runtime_container_names
    assert dashboard_bench_env(first)[first.token_env] == ""


def test_external_service_does_not_create_or_restart_a_container(tmp_path):
    stack = resolve_runtime_stack()
    configured = prepare_bench_runtime(
        str(tmp_path),
        stack,
        {
            "SR_BENCH_URL": "https://bench.example",
            "SR_BENCH_TOKEN_ENV": "BENCH_ACCESS",
            "BENCH_ACCESS": "private",
        },
    )
    assert not configured.managed
    assert configured.origin == "https://bench.example"
    assert not (tmp_path / ".sr-bench").exists()
    with pytest.raises(ValueError, match="missing"):
        prepare_bench_runtime(
            str(tmp_path), stack, {"SR_BENCH_URL": "https://bench.example"}
        )


def test_target_credentials_are_loaded_only_from_registered_references(tmp_path):
    stack = resolve_runtime_stack()
    initial = prepare_bench_runtime(str(tmp_path), stack, {})
    (initial.store / "targets.json").write_text(
        json.dumps([{"id": "single", "api_key_env": "MODEL_TOKEN"}])
    )
    with pytest.raises(ValueError, match="Missing"):
        prepare_bench_runtime(str(tmp_path), stack, {})
    configured = prepare_bench_runtime(
        str(tmp_path), stack, {"MODEL_TOKEN": "model-secret"}
    )
    assert configured.secrets["MODEL_TOKEN"] == "model-secret"
    assert "MODEL_TOKEN" not in dashboard_bench_env(configured)


def test_hugging_face_credential_only_reaches_managed_preparation_worker(tmp_path):
    stack = resolve_runtime_stack()
    bench = prepare_bench_runtime(
        str(tmp_path), stack, {"HF_TOKEN": "fixture-gated-source-token"}
    )
    assert bench.secrets["HF_TOKEN"] == "fixture-gated-source-token"
    assert "HF_TOKEN" not in dashboard_bench_env(bench)
    spec = container_start._build_bench_runtime_spec(
        runtime="docker",
        image="dashboard:test",
        nofile_limit=10000,
        network_name=stack.network_name,
        stack_layout=stack,
        bench=bench,
    )
    command, health = spec[2]
    assert "HF_TOKEN" in command
    assert "fixture-gated-source-token" not in " ".join(command + health)
    assert (
        "fixture-gated-source-token"
        not in (bench.store.parent / "service-token").read_text()
    )
    external = prepare_bench_runtime(
        str(tmp_path / "external"),
        stack,
        {
            "SR_BENCH_URL": "https://bench.example",
            "SR_BENCH_TOKEN": "fixture-service-token",
            "HF_TOKEN": "fixture-gated-source-token",
        },
    )
    assert external.secrets == {"SR_BENCH_TOKEN": "fixture-service-token"}
    assert "HF_TOKEN" not in dashboard_bench_env(external)
    assert not (tmp_path / "external" / ".sr-bench").exists()


def test_bench_container_has_no_runtime_socket_or_gpu_and_token_never_enters_argv(
    tmp_path,
):
    stack = resolve_runtime_stack()
    bench = prepare_bench_runtime(str(tmp_path), stack, {})
    name, container, commands = container_start._build_bench_runtime_spec(
        runtime="docker",
        image="dashboard@sha256:test",
        nofile_limit=10000,
        network_name=stack.network_name,
        stack_layout=stack,
        bench=bench,
    )
    command, health = commands
    assert name == "sr-bench" and container == stack.sr_bench_container_name
    assert "127.0.0.1:8090:8090" in command
    assert "--user" in command
    assert "--device" not in command and "--gpus" not in command
    assert not any("docker.sock" in item for item in command)
    assert "SR_BENCH_TOKEN" in command
    assert bench.secrets[bench.token_env] not in " ".join(command + health)
    assert ":ro" not in next(
        item for item in command if item.startswith(str(bench.store) + ":")
    )


def test_reuse_requires_running_matching_worker_and_never_starts_a_stopped_one(
    monkeypatch,
):
    command = ["docker", "run", "--label", f"{BENCH_IDENTITY_LABEL}=same"]
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {"status": "running", "labels": {BENCH_IDENTITY_LABEL: "same"}}
            ),
        ),
    )
    assert reconcile_bench_container(command, "bench", {})
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {"status": "exited", "labels": {BENCH_IDENTITY_LABEL: "same"}}
            ),
        ),
    )
    with pytest.raises(ValueError, match="stopped"):
        reconcile_bench_container(command, "bench", {})
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {"status": "running", "labels": {BENCH_IDENTITY_LABEL: "other"}}
            ),
        ),
    )
    with pytest.raises(ValueError, match="different"):
        reconcile_bench_container(command, "bench", {})


def test_reused_worker_is_not_owned_by_failed_dashboard_start_rollback(monkeypatch):
    commands = []
    removed = []

    def run(command, **kwargs):
        commands.append(command)
        if command[1] == "run":
            raise subprocess.CalledProcessError(125, command, stderr="name conflict")
        return SimpleNamespace(stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(
        container_start_runner, "reconcile_bench_container", lambda *a: "reuse"
    )
    monkeypatch.setattr(
        container_start_runner,
        "_cleanup_started_containers",
        removed.extend,
    )
    status, _, _ = container_start_runner.run_container_specs(
        [
            (
                "sr-bench",
                "bench",
                (["docker", "run"], ["docker", "exec", "bench", "health"]),
            ),
            ("dashboard", "dashboard", (["docker", "run"],)),
        ],
        storage_secret_values={},
        bench_secret_values={"SR_BENCH_TOKEN": "secret"},
    )
    assert status == 125
    assert removed == []
    assert commands.count(["docker", "run"]) == 2


def test_runtime_reload_preserves_bench_but_explicit_stop_accounts_for_it(monkeypatch):
    stack = resolve_runtime_stack()
    observed = []
    monkeypatch.setattr(
        core, "container_status", lambda name: observed.append(name) or "not found"
    )
    core._managed_container_statuses(stack)
    assert stack.sr_bench_container_name in observed
    assert stack.sr_bench_container_name not in stack.runtime_container_names
    stopped = []
    monkeypatch.setattr(
        runtime_lifecycle,
        "acquire_runtime_lifecycle_lock",
        lambda **kwargs: nullcontext(),
    )
    monkeypatch.setattr(runtime_lifecycle, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(
        runtime_lifecycle,
        "container_status_strict",
        lambda name: "exited" if name in stopped else "running",
    )
    monkeypatch.setattr(
        runtime_lifecycle,
        "container_stop_container",
        lambda name: stopped.append(name) or True,
    )
    runtime_lifecycle.stop_runtime_before_config_replacement(stack)
    assert set(stopped) == set(stack.runtime_container_names)
    assert stack.sr_bench_container_name not in stopped


def test_reconciliation_failure_rolls_back_only_new_runtime_containers(monkeypatch):
    removed = []

    def run(command, **kwargs):
        if command[-1] == "worker":
            raise subprocess.CalledProcessError(125, command, stderr="name conflict")
        return SimpleNamespace(stdout="", stderr="")

    def reconcile(*args):
        raise ValueError("Stopped worker requires saved-ledger reconciliation")

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(container_start_runner, "reconcile_bench_container", reconcile)
    monkeypatch.setattr(
        container_start_runner,
        "_cleanup_started_containers",
        removed.extend,
    )
    status, _, error = container_start_runner.run_container_specs(
        [
            ("router", "router", (["docker", "create", "router"],)),
            ("envoy", "envoy", (["docker", "run", "envoy"],)),
            ("sr-bench", "worker", (["docker", "run", "worker"],)),
        ],
        storage_secret_values={},
        bench_secret_values={"SR_BENCH_TOKEN": "secret"},
    )
    assert status == 1
    assert "reconciliation" in error
    assert removed == ["router", "envoy"]


@pytest.mark.parametrize(
    "saved_status", ["completed", "failed", "cancelled", "interrupted"]
)
def test_serve_replaces_only_idle_image_upgrade_and_preserves_journal(
    tmp_path, monkeypatch, saved_status
):
    stack = resolve_runtime_stack(stack_name="upgrade-test", port_offset=1000)
    bench = prepare_bench_runtime(str(tmp_path), stack, {})
    with sqlite3.connect(bench.store / "journal.sqlite3") as db:
        db.execute("CREATE TABLE runs(id TEXT, status TEXT)")
        db.execute("INSERT INTO runs VALUES(?, ?)", ("saved-result", saved_status))
    kwargs = {
        "runtime": "docker",
        "nofile_limit": 10000,
        "network_name": stack.network_name,
        "stack_layout": stack,
        "bench": bench,
    }
    old = container_start._build_bench_runtime_spec(image="dashboard:old", **kwargs)
    new = container_start._build_bench_runtime_spec(image="dashboard:new", **kwargs)
    old_identity = next(
        arg.split("=", 1)[1]
        for arg in old[2][0]
        if arg.startswith(BENCH_IDENTITY_LABEL + "=")
    )
    state = {"exists": True, "paused": False, "image": "dashboard:old"}
    actions = []

    def run(command, **kwargs):
        action = command[1]
        actions.append(action)
        if action == "run":
            if state["exists"]:
                raise subprocess.CalledProcessError(
                    125, command, stderr="name conflict"
                )
            state.update(exists=True, image="dashboard:new")
        elif action == "inspect":
            info = {
                "id": "owned-container-id",
                "status": "running",
                "labels": {BENCH_IDENTITY_LABEL: old_identity},
                "image": "dashboard:old",
            }
            return SimpleNamespace(returncode=0, stdout=json.dumps(info), stderr="")
        elif action == "pause":
            assert command[-1] == "owned-container-id"
            state["paused"] = True
        elif action == "rm":
            assert state["paused"] and command[-1] == "owned-container-id"
            state.update(exists=False, paused=False)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    code, _, error = container_start_runner.run_container_specs(
        [new], storage_secret_values={}, bench_secret_values=bench.secrets
    )
    assert code == 0, error
    assert state == {"exists": True, "paused": False, "image": "dashboard:new"}
    assert actions == ["run", "inspect", "image", "pause", "rm", "run", "exec"]
    with sqlite3.connect(bench.store / "journal.sqlite3") as db:
        assert db.execute("SELECT id, status FROM runs").fetchall() == [
            ("saved-result", saved_status)
        ]


@pytest.mark.parametrize("status", ["queued", "running", "planning"])
def test_image_upgrade_preserves_active_runs_and_unpauses_worker(
    tmp_path, monkeypatch, status
):
    with sqlite3.connect(tmp_path / "journal.sqlite3") as db:
        db.execute("CREATE TABLE runs(status TEXT)")
        db.execute("INSERT INTO runs VALUES(?)", (status,))
    actions = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: actions.append(cmd))
    with pytest.raises(ValueError, match="active runs"):
        _remove_idle_bench_container("docker", "owned-id", tmp_path)
    assert actions == [
        ["docker", "pause", "owned-id"],
        ["docker", "unpause", "owned-id"],
    ]


@pytest.mark.parametrize("status", ["queued", "running", "completed", "failed"])
def test_image_upgrade_only_replaces_worker_after_preparation_finishes(
    tmp_path, monkeypatch, status
):
    with sqlite3.connect(tmp_path / "journal.sqlite3") as db:
        db.execute("CREATE TABLE runs(status TEXT)")
    journal = tmp_path / "dataset-preparations"
    journal.mkdir()
    job = {"id": "prep-" + "ab" * 16, "status": status}
    path = journal / (job["id"] + ".json")
    actions = []

    def run(command, **kwargs):
        actions.append(command)
        if command[1] == "pause":
            # Admission just before the freeze must be visible to the idle check.
            path.write_text(json.dumps(job))

    monkeypatch.setattr(subprocess, "run", run)
    if status in {"queued", "running"}:
        with pytest.raises(ValueError, match="active dataset preparations"):
            _remove_idle_bench_container("docker", "owned-id", tmp_path)
        assert actions == [
            ["docker", "pause", "owned-id"],
            ["docker", "unpause", "owned-id"],
        ]
    else:
        _remove_idle_bench_container("docker", "owned-id", tmp_path)
        assert actions == [
            ["docker", "pause", "owned-id"],
            ["docker", "rm", "--force", "owned-id"],
        ]
    assert json.loads(path.read_text()) == job


@pytest.mark.parametrize(
    "failure",
    [
        "invalid-json",
        "unknown-status",
        "mismatched-id",
        "symlink",
        "unreadable-directory",
    ],
)
def test_image_upgrade_keeps_worker_when_preparation_journal_cannot_be_verified(
    tmp_path, monkeypatch, failure
):
    with sqlite3.connect(tmp_path / "journal.sqlite3") as db:
        db.execute("CREATE TABLE runs(status TEXT)")
    journal = tmp_path / "dataset-preparations"
    journal.mkdir()
    identifier = "prep-" + "ab" * 16
    path = journal / (identifier + ".json")
    if failure == "invalid-json":
        path.write_text("incomplete fixture")
    elif failure == "symlink":
        destination = tmp_path / "foreign-job.json"
        destination.write_text(json.dumps({"id": identifier, "status": "completed"}))
        path.symlink_to(destination)
    elif failure == "unreadable-directory":
        original_iterdir = type(journal).iterdir

        def iterdir(directory):
            if directory == journal:
                raise PermissionError("fixture denied preparation journal read")
            return original_iterdir(directory)

        monkeypatch.setattr(type(journal), "iterdir", iterdir)
    else:
        path.write_text(
            json.dumps(
                {
                    "id": "wrong-id" if failure == "mismatched-id" else identifier,
                    "status": "unknown" if failure == "unknown-status" else "completed",
                }
            )
        )
    actions = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: actions.append(cmd))
    with pytest.raises(ValueError, match="Cannot verify dataset preparations"):
        _remove_idle_bench_container("docker", "owned-id", tmp_path)
    assert actions == [
        ["docker", "pause", "owned-id"],
        ["docker", "unpause", "owned-id"],
    ]


@pytest.mark.parametrize(
    "failure", ["missing-journal", "invalid-journal", "remove-failed"]
)
def test_image_upgrade_failure_resumes_previous_worker(tmp_path, monkeypatch, failure):
    if failure == "invalid-journal":
        (tmp_path / "journal.sqlite3").write_text("not sqlite")
    elif failure == "remove-failed":
        with sqlite3.connect(tmp_path / "journal.sqlite3") as db:
            db.execute("CREATE TABLE runs(status TEXT)")
    actions = []

    def run(cmd, **kwargs):
        actions.append(cmd[1])
        if cmd[1] == "rm":
            raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        _remove_idle_bench_container("docker", "owned-id", tmp_path)
    assert actions[-1] == "unpause"
    assert actions == (
        ["pause", "rm", "unpause"]
        if failure == "remove-failed"
        else ["pause", "unpause"]
    )


@pytest.mark.parametrize(
    "difference", ["credential", "store", "stack", "port", "missing-label"]
)
def test_image_upgrade_never_replaces_different_worker(
    tmp_path, monkeypatch, difference
):
    stack = resolve_runtime_stack(stack_name="upgrade-test", port_offset=1000)
    bench = prepare_bench_runtime(str(tmp_path), stack, {})
    kwargs = {
        "runtime": "docker",
        "nofile_limit": 10000,
        "network_name": stack.network_name,
        "stack_layout": stack,
        "bench": bench,
    }
    old = container_start._build_bench_runtime_spec(image="dashboard:old", **kwargs)[2][
        0
    ]
    new = container_start._build_bench_runtime_spec(image="dashboard:new", **kwargs)[2][
        0
    ]
    old_identity = next(
        arg.split("=", 1)[1]
        for arg in old
        if arg.startswith(BENCH_IDENTITY_LABEL + "=")
    )
    credentials = dict(bench.secrets)
    if difference == "credential":
        credentials[bench.token_env] = "different-test-token"
    elif difference == "store":
        new[new.index("--store") + 1] += "-other"
    elif difference == "stack":
        new[new.index("--name") + 1] = "different-stack"
    elif difference == "port":
        new[new.index("-p") + 1] = "127.0.0.1:9199:8090"
    info = {
        "id": "foreign-id",
        "status": "running",
        "image": "dashboard:old",
        "labels": (
            {}
            if difference == "missing-label"
            else {BENCH_IDENTITY_LABEL: old_identity}
        ),
    }
    actions = []

    def run(command, **kwargs):
        actions.append(command[1])
        return SimpleNamespace(returncode=0, stdout=json.dumps(info), stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(ValueError, match="different"):
        reconcile_bench_container(new, stack.sr_bench_container_name, credentials)
    assert actions == ["inspect"]


def test_managed_bench_port_override_matches_container_and_cli_discovery(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("VLLM_SR_BENCH_PORT", "8091")
    monkeypatch.setenv("VLLM_SR_PORT_OFFSET", "1000")
    monkeypatch.setenv("VLLM_SR_STATE_ROOT_DIR", str(tmp_path))
    stack = resolve_runtime_stack()
    bench = prepare_bench_runtime(str(tmp_path), stack, {})
    spec = container_start._build_bench_runtime_spec(
        runtime="docker",
        image="dashboard:test",
        nofile_limit=10000,
        network_name=stack.network_name,
        stack_layout=stack,
        bench=bench,
    )
    assert "127.0.0.1:8091:8090" in spec[2][0]
    assert bench.origin == f"http://{stack.sr_bench_container_name}:8090"
    urls = []

    def request(self, method, path):
        urls.append(self.url)
        return {"runs": []}

    monkeypatch.setattr(client.Client, "request", request)
    result = CliRunner().invoke(benchmark, ["runs"])
    assert result.exit_code == 0, result.output
    assert urls == ["http://127.0.0.1:8091"]


@pytest.mark.parametrize("port", ["0", "65536", "-1", "not-a-port", "8090.5"])
def test_managed_bench_port_override_rejects_invalid_values(monkeypatch, port):
    monkeypatch.setenv("VLLM_SR_BENCH_PORT", port)
    with pytest.raises(ValueError, match="VLLM_SR_BENCH_PORT"):
        resolve_runtime_stack()


def test_replaced_worker_is_owned_by_failed_health_rollback(monkeypatch):
    commands = []
    removed = []

    def run(command, **kwargs):
        commands.append(command)
        if len(commands) == 1 or command[1] == "exec":
            raise subprocess.CalledProcessError(125, command, stderr="fixture failure")
        return SimpleNamespace(stdout="new-container", stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(
        container_start_runner, "reconcile_bench_container", lambda *a: "replace"
    )
    monkeypatch.setattr(
        container_start_runner, "_cleanup_started_containers", removed.extend
    )
    code, _, _ = container_start_runner.run_container_specs(
        [
            (
                "sr-bench",
                "owned-bench",
                (["docker", "run"], ["docker", "exec", "health"]),
            )
        ],
        storage_secret_values={},
        bench_secret_values={"SR_BENCH_TOKEN": "fixture"},
    )
    assert code == 125
    assert removed == ["owned-bench"]
    assert len(commands) == 3


def test_pause_failure_never_removes_worker(tmp_path, monkeypatch):
    actions = []

    def run(cmd, **kwargs):
        actions.append(cmd[1])
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError):
        _remove_idle_bench_container("docker", "owned-id", tmp_path)
    assert actions == ["pause"]


def test_pause_timeout_attempts_resume_without_removing_worker(tmp_path, monkeypatch):
    actions = []

    def run(cmd, **kwargs):
        actions.append(cmd[1])
        if cmd[1] == "pause":
            raise subprocess.TimeoutExpired(cmd, 10)

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(subprocess.TimeoutExpired):
        _remove_idle_bench_container("docker", "owned-id", tmp_path)
    assert actions == ["pause", "unpause"]
