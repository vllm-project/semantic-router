"""Taking over storage containers that predate per-stack credentials.

Split out of ``test_storage_backends``: a takeover has its own fixture -- real
credential state on disk plus a recorded transcript of every runtime step -- and
every test here asserts on the *order* of those steps, which is what proves the
credential state is committed only after both containers have accepted it.
"""

import subprocess

import pytest as _pytest
from cli import storage_backends
from cli.runtime_stack import resolve_runtime_stack
from cli.storage_backends import start_storage_backends
from cli.storage_secrets import (
    StorageVolumes,
    default_storage_volume_names,
    load_storage_secrets,
    storage_state_path,
)


def _takeover_environment(monkeypatch, tmp_path, events):
    """Record every step of a takeover, with real credential state on disk."""

    layout = resolve_runtime_stack()
    state_path = storage_state_path(str(tmp_path), stack_layout=layout)

    def note(step, **detail):
        events.append((step, {"state_committed": state_path.exists(), **detail}))

    monkeypatch.setattr(
        storage_backends,
        "adopt_storage_volumes",
        lambda _layout: note("inspect-mounts")
        or StorageVolumes(postgres="pg-anon-hex", redis="redis-anon-hex"),
    )
    monkeypatch.setattr(storage_backends, "container_status", lambda _name: "running")
    monkeypatch.setattr(
        storage_backends,
        "container_stop_container",
        lambda name: note("park", container=name) or True,
    )
    monkeypatch.setattr(
        storage_backends,
        "container_start_redis",
        lambda _network, _layout, **kwargs: note("start-redis", **kwargs)
        or (0, "", ""),
    )
    monkeypatch.setattr(
        storage_backends,
        "container_start_postgres",
        lambda _network, _layout, **kwargs: note("start-postgres", **kwargs)
        or (0, "", ""),
    )
    monkeypatch.setattr(
        storage_backends, "container_remove_container", lambda _name: True
    )
    monkeypatch.setattr(
        storage_backends, "_postgres_credentials_match", lambda *_a: True
    )
    monkeypatch.setattr(storage_backends, "_wait_for_postgres", lambda *_a: True)
    monkeypatch.setattr(
        storage_backends,
        "rekey_postgres_role",
        lambda name, secret: note("alter-role", container=name) or (0, "", ""),
    )
    return layout, state_path


def test_takeover_commits_credential_state_only_after_the_containers_accept_it(
    monkeypatch, tmp_path
):

    events = []
    layout, state_path = _takeover_environment(monkeypatch, tmp_path, events)
    assert not state_path.exists()

    started = start_storage_backends(
        {"redis", "postgres"}, layout, state_root_dir=str(tmp_path)
    )

    assert started == {"redis", "postgres"}
    steps = [step for step, _detail in events]
    # Mounts are read while the old containers still exist, the containers are
    # rebuilt and re-keyed, and only then is the state written.
    assert steps == ["inspect-mounts", "start-redis", "start-postgres", "alter-role"]
    assert not any(detail["state_committed"] for _step, detail in events)
    assert state_path.exists()

    secrets = load_storage_secrets(str(tmp_path), stack_layout=layout)
    assert secrets.postgres.volume == "pg-anon-hex"
    assert secrets.redis.volume == "redis-anon-hex"
    detail = dict(events)["start-postgres"]
    assert detail["data_volume"] == "pg-anon-hex"
    assert detail["postgres_password_file"].endswith("postgres-password")
    # A running container keeps the credentials it started with, so a takeover
    # has to rebuild rather than reuse.
    assert detail["recreate"] is True
    assert dict(events)["start-redis"]["recreate"] is True


def test_takeover_announces_that_the_shared_credential_is_revoked_without_printing_it(
    monkeypatch, tmp_path, caplog
):

    events = []
    layout, _state_path = _takeover_environment(monkeypatch, tmp_path, events)

    with caplog.at_level("INFO"):
        start_storage_backends(
            {"redis", "postgres"}, layout, state_root_dir=str(tmp_path)
        )

    assert "treated as compromised" in caplog.text
    assert "revoked" in caplog.text
    secrets = load_storage_secrets(str(tmp_path), stack_layout=layout)
    assert secrets.postgres.password not in caplog.text
    assert secrets.redis.password not in caplog.text


def test_a_restart_reuses_the_recorded_credentials_and_does_not_re_key(
    monkeypatch, tmp_path
):

    events = []
    layout, state_path = _takeover_environment(monkeypatch, tmp_path, events)
    start_storage_backends({"redis", "postgres"}, layout, state_root_dir=str(tmp_path))
    committed = state_path.read_bytes()
    events.clear()

    start_storage_backends({"redis", "postgres"}, layout, state_root_dir=str(tmp_path))

    steps = [step for step, _detail in events]
    assert steps == ["start-redis", "start-postgres"]
    # Unchanged credentials mean a running container may simply be reused.
    assert dict(events)["start-redis"]["recreate"] is False
    assert dict(events)["start-postgres"]["recreate"] is False
    assert state_path.read_bytes() == committed


def test_a_failed_re_key_leaves_no_credential_state_behind(monkeypatch, tmp_path):

    events = []
    layout, state_path = _takeover_environment(monkeypatch, tmp_path, events)
    monkeypatch.setattr(
        storage_backends,
        "rekey_postgres_role",
        lambda _name, _secret: (1, "", "connection refused"),
    )

    with _pytest.raises(SystemExit):
        start_storage_backends(
            {"redis", "postgres"}, layout, state_root_dir=str(tmp_path)
        )

    assert not state_path.exists()


@_pytest.mark.parametrize("unattached", [0, 2])
def test_a_fresh_stack_names_its_own_volumes_and_warns_only_with_unattached_ones(
    monkeypatch, tmp_path, caplog, unattached
):

    events = []
    layout, _state_path = _takeover_environment(monkeypatch, tmp_path, events)
    defaults = default_storage_volume_names(layout)
    monkeypatch.setattr(
        storage_backends, "adopt_storage_volumes", lambda _layout: defaults
    )
    monkeypatch.setattr(storage_backends, "container_status", lambda _name: "not found")
    monkeypatch.setattr(
        storage_backends, "_unattached_volume_count", lambda: unattached
    )

    with caplog.at_level("INFO"):
        start_storage_backends({"redis"}, layout, state_root_dir=str(tmp_path))

    # A fresh host has no volume an older stack could have left behind.
    if unattached:
        assert "empty data volumes" in caplog.text
        assert "2 data volume(s) on this host belong to no container" in caplog.text
    else:
        assert "empty data volumes" not in caplog.text
        assert "older" not in caplog.text
    assert dict(events)["start-redis"]["data_volume"] == defaults.redis


def test_takeover_re_keys_a_container_the_config_no_longer_requires(
    monkeypatch, tmp_path
):

    events = []
    layout, _state_path = _takeover_environment(monkeypatch, tmp_path, events)

    # Only Redis is required, but a Postgres container from an older stack is
    # still here and its volume is about to be recorded in the state file.
    started = start_storage_backends({"redis"}, layout, state_root_dir=str(tmp_path))

    assert started == {"redis"}
    steps = [step for step, _detail in events]
    assert steps == [
        "inspect-mounts",
        "start-redis",
        "start-postgres",
        "alter-role",
        "park",
    ]
    assert dict(events)["park"]["container"] == layout.postgres_container_name
    secrets = load_storage_secrets(str(tmp_path), stack_layout=layout)
    assert secrets.postgres.volume == "pg-anon-hex"


def test_a_restart_leaves_a_container_the_config_no_longer_requires_alone(
    monkeypatch, tmp_path
):

    events = []
    layout, _state_path = _takeover_environment(monkeypatch, tmp_path, events)
    start_storage_backends({"redis"}, layout, state_root_dir=str(tmp_path))
    events.clear()

    start_storage_backends({"redis"}, layout, state_root_dir=str(tmp_path))

    assert [step for step, _detail in events] == ["start-redis"]


@_pytest.mark.parametrize("initial_pg_status", ["not found", "exited"])
@_pytest.mark.parametrize("fail_first_rekey", [False, True])
def test_redis_created_state_reconciles_later_postgres_retained_volume(
    monkeypatch, tmp_path, initial_pg_status, fail_first_rekey
):
    events = []
    layout, state_path = _takeover_environment(monkeypatch, tmp_path, events)
    statuses = {
        layout.redis_container_name: "not found",
        layout.postgres_container_name: "not found",
    }
    monkeypatch.setattr(
        storage_backends, "container_status", lambda name: statuses[name]
    )
    monkeypatch.setattr(storage_backends, "_unattached_volume_count", lambda: 0)
    defaults = default_storage_volume_names(layout)
    monkeypatch.setattr(
        storage_backends, "adopt_storage_volumes", lambda _layout: defaults
    )
    start_storage_backends({"redis"}, layout, state_root_dir=str(tmp_path))
    committed = state_path.read_bytes()
    canonical = load_storage_secrets(str(tmp_path), stack_layout=layout)
    statuses[layout.postgres_container_name] = initial_pg_status
    volume_password = "prior-stack-password"
    attempts = []
    launches = []

    def start_pg(_network, _layout, **kwargs):
        launches.append(kwargs)
        statuses[layout.postgres_container_name] = "running"
        return 0, "", ""

    def rekey(_name, secret):
        nonlocal volume_password
        attempts.append(secret)
        if fail_first_rekey and len(attempts) == 1:
            return 1, "", "temporary rekey failure"
        volume_password = secret.password
        return 0, "", ""

    def stop_pg(name):
        assert name == layout.postgres_container_name
        statuses[name] = "exited"
        return True

    def remove_pg(name):
        assert name == layout.postgres_container_name
        statuses[name] = "not found"
        return True

    monkeypatch.setattr(storage_backends, "container_start_postgres", start_pg)
    monkeypatch.setattr(storage_backends, "rekey_postgres_role", rekey)
    monkeypatch.setattr(storage_backends, "container_stop_container", stop_pg)
    monkeypatch.setattr(storage_backends, "container_remove_container", remove_pg)
    monkeypatch.setattr(
        storage_backends,
        "_postgres_credentials_match",
        lambda _name, secret: volume_password == secret.password,
    )
    if fail_first_rekey:
        with _pytest.raises(SystemExit):
            start_storage_backends({"postgres"}, layout, state_root_dir=str(tmp_path))
        assert statuses[layout.postgres_container_name] == "not found"
        assert state_path.read_bytes() == committed
        assert volume_password == "prior-stack-password"
    assert start_storage_backends(
        {"postgres"}, layout, state_root_dir=str(tmp_path)
    ) == {"postgres"}
    assert volume_password == canonical.postgres.password
    assert state_path.read_bytes() == committed
    assert all(call["data_volume"] == defaults.postgres for call in launches)
    assert all(attempt == canonical.postgres for attempt in attempts)
    before = len(attempts)
    start_storage_backends({"postgres"}, layout, state_root_dir=str(tmp_path))
    assert len(attempts) == before


def test_failed_rekey_and_failed_cleanup_cannot_be_reused_as_ready(
    monkeypatch, tmp_path
):
    events = []
    layout, state_path = _takeover_environment(monkeypatch, tmp_path, events)
    start_storage_backends({"redis", "postgres"}, layout, state_root_dir=str(tmp_path))
    committed = state_path.read_bytes()
    status = "exited"

    def start_pg(*_args, **_kwargs):
        nonlocal status
        status = "running"
        return 0, "", ""

    monkeypatch.setattr(storage_backends, "container_status", lambda _name: status)
    monkeypatch.setattr(storage_backends, "container_start_postgres", start_pg)
    monkeypatch.setattr(
        storage_backends, "container_stop_container", lambda _name: False
    )
    monkeypatch.setattr(
        storage_backends, "rekey_postgres_role", lambda *_args: (1, "", "failed")
    )
    monkeypatch.setattr(
        storage_backends, "_postgres_credentials_match", lambda *_args: False
    )
    for _ in range(2):
        with _pytest.raises(SystemExit):
            start_storage_backends({"postgres"}, layout, state_root_dir=str(tmp_path))
        assert state_path.read_bytes() == committed
    assert status == "running"


@_pytest.mark.parametrize(
    "outcome", ["success", "rejected", "timeout", "runtime_error"]
)
def test_reused_postgres_check_uses_tcp_and_never_exposes_password(
    monkeypatch, tmp_path, caplog, outcome
):
    verify = storage_backends._postgres_credentials_match
    events = []
    layout, _state_path = _takeover_environment(monkeypatch, tmp_path, events)
    start_storage_backends({"redis", "postgres"}, layout, state_root_dir=str(tmp_path))
    secret = load_storage_secrets(str(tmp_path), stack_layout=layout).postgres
    monkeypatch.setattr(storage_backends, "get_container_runtime", lambda: "docker")
    commands = []

    def run(command, **kwargs):
        assert secret.password not in " ".join(command)
        assert kwargs["env"]["PGPASSWORD"] == secret.password
        assert "--no-password" in command
        assert command[command.index("-h") + 1] == "127.0.0.1"
        commands.append(command)
        if outcome == "timeout":
            raise subprocess.TimeoutExpired(command, 10, output=secret.password)
        if outcome == "runtime_error":
            raise OSError(secret.password)
        return subprocess.CompletedProcess(
            command, 0 if outcome == "success" else 1, stderr=secret.password
        )

    monkeypatch.setattr(storage_backends.subprocess, "run", run)
    assert verify(layout.postgres_container_name, secret) == (outcome == "success")
    assert len(commands) == 1
    assert secret.password not in caplog.text
