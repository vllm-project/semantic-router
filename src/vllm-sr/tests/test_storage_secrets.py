import json
import logging
import stat
import subprocess
from pathlib import Path

import pytest
from cli import container_mounts, storage_secrets
from cli.commands import runtime_paths
from cli.commands.runtime_paths import write_private_state_bytes
from cli.consts import DEFAULT_STACK_NAME
from cli.runtime_stack import resolve_runtime_stack
from cli.storage_secrets import StorageSecretError


@pytest.fixture(autouse=True)
def blocked_container_runtime(monkeypatch):
    """Fail loudly instead of reaching a real daemon from a unit test."""

    def forbidden(*_args, **_kwargs):
        raise AssertionError("unit tests must not invoke the container runtime")

    monkeypatch.setattr(storage_secrets, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(storage_secrets.subprocess, "run", forbidden)
    monkeypatch.setattr(container_mounts, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(container_mounts.subprocess, "run", forbidden)


def _layout(stack_name: str = DEFAULT_STACK_NAME):
    return resolve_runtime_stack(stack_name=stack_name)


def _ensure(state_root: Path, layout, **kwargs):
    kwargs.setdefault("volumes", storage_secrets.default_storage_volume_names(layout))
    return storage_secrets.ensure_storage_secrets(
        state_root, stack_layout=layout, **kwargs
    )


def _file_mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _state_document(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_fresh_stack_generates_private_state_and_credential_files(tmp_path: Path):
    layout = _layout()

    secrets = _ensure(tmp_path, layout)

    directory = storage_secrets.storage_secrets_directory(tmp_path)
    state_path = storage_secrets.storage_state_path(tmp_path, stack_layout=layout)
    password_path = storage_secrets.postgres_password_path(
        tmp_path, stack_layout=layout
    )
    conf_path = storage_secrets.redis_conf_path(tmp_path, stack_layout=layout)

    assert _file_mode(directory) == 0o700
    assert _file_mode(state_path) == 0o600
    assert _file_mode(password_path) == 0o600
    # The Redis image reads its config as uid 999, so this file stays group and
    # world readable while the 0700 directory keeps it private on the host.
    assert _file_mode(conf_path) == 0o644

    assert secrets.stack == DEFAULT_STACK_NAME
    assert secrets.postgres.user == "router"
    assert secrets.postgres.database == "vsr"
    assert secrets.postgres.volume == "vllm-sr-postgres-data"
    assert secrets.redis.volume == "vllm-sr-redis-data"
    assert secrets.postgres.password != secrets.redis.password
    assert len(secrets.postgres.password) >= 40

    assert password_path.read_bytes() == secrets.postgres.password.encode("utf-8")
    assert not password_path.read_bytes().endswith(b"\n")
    assert conf_path.read_text(encoding="utf-8") == (
        f"requirepass {secrets.redis.password}\n"
    )

    assert _state_document(state_path) == {
        "schema": "vllm-sr/storage-secrets/v1",
        "stack": DEFAULT_STACK_NAME,
        "postgres": {
            "user": "router",
            "database": "vsr",
            "password": secrets.postgres.password,
            "volume": "vllm-sr-postgres-data",
        },
        "redis": {
            "password": secrets.redis.password,
            "volume": "vllm-sr-redis-data",
        },
    }


def test_restart_reuses_the_same_credentials_without_rewriting_state(tmp_path: Path):
    layout = _layout()
    applied = []

    first = _ensure(tmp_path, layout, apply_secrets=applied.append)
    state_path = storage_secrets.storage_state_path(tmp_path, stack_layout=layout)
    committed = state_path.read_bytes()

    second = _ensure(tmp_path, layout, apply_secrets=applied.append)

    assert second == first
    assert state_path.read_bytes() == committed
    assert [entry.postgres.password for entry in applied] == [first.postgres.password]


def test_restart_restores_a_deleted_credential_file(tmp_path: Path):
    layout = _layout()
    secrets = _ensure(tmp_path, layout)
    conf_path = storage_secrets.redis_conf_path(tmp_path, stack_layout=layout)
    conf_path.unlink()

    reused = _ensure(tmp_path, layout)

    assert reused == secrets
    assert conf_path.read_text(encoding="utf-8") == (
        f"requirepass {secrets.redis.password}\n"
    )
    assert _file_mode(conf_path) == 0o644


def test_multiple_stacks_share_one_state_root_without_collision(tmp_path: Path):
    default_layout = _layout()
    audit_layout = _layout("audit")

    default_secrets = _ensure(tmp_path, default_layout)
    directory = storage_secrets.storage_secrets_directory(tmp_path)
    default_bytes = {
        entry.name: entry.read_bytes() for entry in sorted(directory.iterdir())
    }

    audit_secrets = _ensure(tmp_path, audit_layout)

    assert sorted(entry.name for entry in directory.iterdir()) == [
        "postgres-password",
        "postgres-password.audit",
        "redis.audit.conf",
        "redis.conf",
        "secrets.audit.json",
        "secrets.json",
    ]
    for name, content in default_bytes.items():
        assert (directory / name).read_bytes() == content

    assert audit_secrets.postgres.password != default_secrets.postgres.password
    assert audit_secrets.redis.password != default_secrets.redis.password
    assert audit_secrets.postgres.volume == "audit-vllm-sr-postgres-data"
    assert audit_secrets.redis.volume == "audit-vllm-sr-redis-data"


def test_rotation_only_touches_the_requested_stack(tmp_path: Path):
    default_layout = _layout()
    audit_layout = _layout("audit")
    _ensure(tmp_path, default_layout)
    _ensure(tmp_path, audit_layout)

    directory = storage_secrets.storage_secrets_directory(tmp_path)
    neighbour_bytes = {
        entry.name: entry.read_bytes()
        for entry in directory.iterdir()
        if "audit" not in entry.name
    }

    storage_secrets.rotate_storage_secrets(tmp_path, stack_layout=audit_layout)

    for name, content in neighbour_bytes.items():
        assert (directory / name).read_bytes() == content


def test_rotation_replaces_both_values_and_keeps_the_recorded_volumes(tmp_path: Path):
    layout = _layout()
    original = _ensure(tmp_path, layout)
    applied = []

    rotated = storage_secrets.rotate_storage_secrets(
        tmp_path, stack_layout=layout, apply_secrets=applied.append
    )

    assert rotated.postgres.password != original.postgres.password
    assert rotated.redis.password != original.redis.password
    assert rotated.postgres.user == original.postgres.user
    assert rotated.postgres.database == original.postgres.database
    assert rotated.postgres.volume == original.postgres.volume
    assert rotated.redis.volume == original.redis.volume
    assert applied == [rotated]

    assert storage_secrets.load_storage_secrets(tmp_path, stack_layout=layout) == (
        rotated
    )
    conf_path = storage_secrets.redis_conf_path(tmp_path, stack_layout=layout)
    assert original.redis.password not in conf_path.read_text(encoding="utf-8")


def test_generation_writes_state_only_after_the_containers_are_rekeyed(
    tmp_path: Path,
):
    layout = _layout()
    state_path = storage_secrets.storage_state_path(tmp_path, stack_layout=layout)
    observed = {}

    def failing_apply(secrets):
        observed["state_existed_during_apply"] = state_path.exists()
        raise RuntimeError("ALTER ROLE failed")

    with pytest.raises(RuntimeError, match="ALTER ROLE failed"):
        _ensure(tmp_path, layout, apply_secrets=failing_apply)

    assert observed == {"state_existed_during_apply": False}
    assert not state_path.exists()

    # The next serve replays the whole takeover instead of trusting a partial one.
    retried = _ensure(tmp_path, layout)
    assert state_path.exists()
    assert storage_secrets.load_storage_secrets(tmp_path, stack_layout=layout) == (
        retried
    )


def test_failed_rotation_reports_that_the_backends_may_disagree(tmp_path: Path):
    """A half-applied rotation is not self-healing and has to say so.

    `ALTER ROLE` may already have landed when the Redis rebuild fails. The
    committed state still names the previous value, so a silent failure would
    let the next serve restore that stale value and hand Router a credential
    Postgres no longer accepts.
    """

    layout = _layout()
    original = _ensure(tmp_path, layout)
    state_path = storage_secrets.storage_state_path(tmp_path, stack_layout=layout)
    committed = state_path.read_bytes()

    def failing_apply(_secrets):
        raise RuntimeError("rebuild failed")

    with pytest.raises(storage_secrets.StorageSecretError) as raised:
        storage_secrets.rotate_storage_secrets(
            tmp_path, stack_layout=layout, apply_secrets=failing_apply
        )
    assert "failed partway through" in str(raised.value)
    assert "vllm-sr serve" in str(raised.value)
    assert isinstance(raised.value.__cause__, RuntimeError)

    assert state_path.read_bytes() == committed
    assert storage_secrets.load_storage_secrets(tmp_path, stack_layout=layout) == (
        original
    )

    # The credential files are put back at once rather than at the next serve,
    # so the files on disk never disagree with the state that was committed.
    conf_path = storage_secrets.redis_conf_path(tmp_path, stack_layout=layout)
    password_path = storage_secrets.postgres_password_path(
        tmp_path, stack_layout=layout
    )
    assert conf_path.read_text(encoding="utf-8") == (
        f"requirepass {original.redis.password}\n"
    )
    assert password_path.read_bytes() == original.postgres.password.encode("utf-8")


def _malformed_payloads() -> dict[str, bytes]:
    valid = {
        "schema": "vllm-sr/storage-secrets/v1",
        "stack": DEFAULT_STACK_NAME,
        "postgres": {
            "user": "router",
            "database": "vsr",
            "password": "pg-value",
            "volume": "vllm-sr-postgres-data",
        },
        "redis": {"password": "redis-value", "volume": "vllm-sr-redis-data"},
    }

    truncated = json.dumps(valid)[:40].encode("utf-8")

    wrong_schema = json.loads(json.dumps(valid))
    wrong_schema["schema"] = "vllm-sr/storage-secrets/v2"

    empty_field = json.loads(json.dumps(valid))
    empty_field["redis"]["password"] = ""

    foreign_stack = json.loads(json.dumps(valid))
    foreign_stack["stack"] = "audit"

    extra_field = json.loads(json.dumps(valid))
    extra_field["generation"] = 2

    extra_section_field = json.loads(json.dumps(valid))
    extra_section_field["redis"]["rotated_at"] = "2026-01-01T00:00:00Z"

    missing_section = json.loads(json.dumps(valid))
    missing_section["postgres"] = "router:pg-value"

    return {
        "truncated": truncated,
        "wrong-schema": json.dumps(wrong_schema).encode("utf-8"),
        "empty-field": json.dumps(empty_field).encode("utf-8"),
        "foreign-stack": json.dumps(foreign_stack).encode("utf-8"),
        "extra-top-level-field": json.dumps(extra_field).encode("utf-8"),
        "extra-section-field": json.dumps(extra_section_field).encode("utf-8"),
        "section-is-not-an-object": json.dumps(missing_section).encode("utf-8"),
    }


@pytest.mark.parametrize("case", sorted(_malformed_payloads()))
def test_malformed_state_fails_closed_and_is_never_regenerated(
    tmp_path: Path, case: str
):
    layout = _layout()
    state_path = storage_secrets.storage_state_path(tmp_path, stack_layout=layout)
    payload = _malformed_payloads()[case]
    write_private_state_bytes(state_path, payload)

    with pytest.raises(StorageSecretError):
        _ensure(tmp_path, layout)
    with pytest.raises(StorageSecretError):
        storage_secrets.load_storage_secrets(tmp_path, stack_layout=layout)

    assert state_path.read_bytes() == payload


def test_state_owned_by_another_user_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    layout = _layout()
    _ensure(tmp_path, layout)
    state_path = storage_secrets.storage_state_path(tmp_path, stack_layout=layout)
    committed = state_path.read_bytes()

    current_user_id = runtime_paths._current_posix_user_id()
    if current_user_id is None:
        pytest.skip("POSIX ownership does not apply on this platform")
    monkeypatch.setattr(
        runtime_paths, "_current_posix_user_id", lambda: current_user_id + 1
    )

    with pytest.raises(StorageSecretError, match="owned by the current user"):
        storage_secrets.load_storage_secrets(tmp_path, stack_layout=layout)

    assert state_path.read_bytes() == committed


def test_group_readable_state_is_hardened_and_still_loads(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    layout = _layout()
    secrets = _ensure(tmp_path, layout)
    state_path = storage_secrets.storage_state_path(tmp_path, stack_layout=layout)
    state_path.chmod(0o644)

    with caplog.at_level(logging.WARNING, logger=runtime_paths.log.name):
        reloaded = storage_secrets.load_storage_secrets(tmp_path, stack_layout=layout)

    assert reloaded == secrets
    assert _file_mode(state_path) == 0o600
    assert any("owner-only permissions" in record.message for record in caplog.records)


def test_state_symlink_fails_closed(tmp_path: Path):
    layout = _layout()
    _ensure(tmp_path, layout)
    state_path = storage_secrets.storage_state_path(tmp_path, stack_layout=layout)
    target = state_path.with_name("elsewhere.json")
    target.write_bytes(state_path.read_bytes())
    state_path.unlink()
    state_path.symlink_to(target)

    with pytest.raises(StorageSecretError, match="symbolic link"):
        storage_secrets.load_storage_secrets(tmp_path, stack_layout=layout)


def test_generated_values_stay_out_of_logs_and_reprs(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    layout = _layout()

    with caplog.at_level(logging.DEBUG):
        secrets = _ensure(tmp_path, layout)
        rotated = storage_secrets.rotate_storage_secrets(tmp_path, stack_layout=layout)

    values = (
        secrets.postgres.password,
        secrets.redis.password,
        rotated.postgres.password,
        rotated.redis.password,
    )
    logged = "\n".join(record.getMessage() for record in caplog.records)
    rendered = f"{rotated!r} {rotated.postgres!r} {rotated.redis!r}"
    for value in values:
        assert value not in logged
        assert value not in rendered


def test_adoption_takes_over_the_anonymous_volume_of_an_existing_container(
    monkeypatch: pytest.MonkeyPatch,
):
    mounts = {
        "vllm-sr-postgres": [
            {"Type": "bind", "Destination": "/etc/hosts", "Name": ""},
            {
                "Type": "volume",
                "Destination": "/var/lib/postgresql/data",
                "Name": "409cea11",
            },
        ],
        "vllm-sr-redis": [
            {"Type": "volume", "Destination": "/data", "Name": "7b2f00aa"}
        ],
    }
    commands = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(
            command, 0, stdout=json.dumps(mounts[command[-1]]), stderr=""
        )

    monkeypatch.setattr(container_mounts.subprocess, "run", fake_run)

    volumes = storage_secrets.adopt_storage_volumes(_layout())

    assert volumes == storage_secrets.StorageVolumes(
        postgres="409cea11", redis="7b2f00aa"
    )
    assert commands[0] == [
        "docker",
        "inspect",
        "--format",
        "{{json .Mounts}}",
        "vllm-sr-postgres",
    ]


@pytest.mark.parametrize("mounts_output", ["[]", "null", '[{"Type":"bind"}]'])
def test_an_image_without_a_volume_declaration_creates_a_fresh_named_volume(
    monkeypatch: pytest.MonkeyPatch, mounts_output: str
):
    monkeypatch.setattr(
        container_mounts.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command, 0, stdout=mounts_output, stderr=""
        ),
    )

    assert storage_secrets.adopt_storage_volumes(_layout()) == (
        storage_secrets.StorageVolumes(
            postgres="vllm-sr-postgres-data", redis="vllm-sr-redis-data"
        )
    )


def test_an_absent_container_creates_a_fresh_named_volume(
    monkeypatch: pytest.MonkeyPatch,
):
    def fake_run(command, **_kwargs):
        raise subprocess.CalledProcessError(1, command, stderr="No such object")

    monkeypatch.setattr(container_mounts.subprocess, "run", fake_run)

    assert storage_secrets.adopt_storage_volumes(_layout("audit")) == (
        storage_secrets.StorageVolumes(
            postgres="audit-vllm-sr-postgres-data", redis="audit-vllm-sr-redis-data"
        )
    )


def test_a_runtime_that_cannot_answer_fails_instead_of_minting_a_new_volume(
    monkeypatch: pytest.MonkeyPatch,
):
    """An unreachable daemon must not be read as "this container is gone".

    Doing so would create an empty volume next to data that is still on disk,
    which is the exact silent-orphan outcome adoption exists to prevent.
    """

    def fake_run(command, **_kwargs):
        raise subprocess.CalledProcessError(
            1, command, stderr="Cannot connect to the Docker daemon. Is it running?"
        )

    monkeypatch.setattr(container_mounts.subprocess, "run", fake_run)

    with pytest.raises(storage_secrets.StorageSecretError, match="Cannot inspect"):
        storage_secrets.adopt_storage_volumes(_layout())


def test_postgres_rekey_sends_the_statement_on_stdin_never_in_argv(
    monkeypatch: pytest.MonkeyPatch,
):
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["input"] = kwargs.get("input")
        captured["check"] = kwargs.get("check")
        captured["timeout"] = kwargs.get("timeout")
        return subprocess.CompletedProcess(command, 0, stdout="ALTER ROLE\n", stderr="")

    monkeypatch.setattr(storage_secrets.subprocess, "run", fake_run)
    secret = storage_secrets.PostgresSecret(
        user="router", database="vsr", password="tOkEn-value_1", volume="v"
    )

    assert storage_secrets.rekey_postgres_role("vllm-sr-postgres", secret) == (
        0,
        "ALTER ROLE\n",
        "",
    )
    assert captured["command"] == [
        "docker",
        "exec",
        "-i",
        "vllm-sr-postgres",
        "psql",
        "-v",
        "ON_ERROR_STOP=1",
        "-U",
        "router",
        "-d",
        "vsr",
        "-f",
        "-",
    ]
    assert secret.password not in " ".join(captured["command"])
    assert captured["input"] == "ALTER ROLE \"router\" WITH PASSWORD 'tOkEn-value_1';\n"
    assert captured["check"] is True
    assert captured["timeout"] == 10


def test_postgres_rekey_redacts_the_value_from_reported_output(
    monkeypatch: pytest.MonkeyPatch,
):
    secret = storage_secrets.PostgresSecret(
        user="router", database="vsr", password="tOkEn-value_1", volume="v"
    )

    def fake_run(command, **_kwargs):
        raise subprocess.CalledProcessError(
            1, command, output="", stderr=f"ERROR near '{secret.password}'"
        )

    monkeypatch.setattr(storage_secrets.subprocess, "run", fake_run)

    return_code, stdout, stderr = storage_secrets.rekey_postgres_role(
        "vllm-sr-postgres", secret
    )

    assert (return_code, stdout) == (1, "")
    assert secret.password not in stderr
    assert "***" in stderr
