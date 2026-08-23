import logging

from cli import storage_secrets
from cli.commands import runtime as runtime_commands
from cli.commands import storage as storage_command
from cli.main import main
from cli.runtime_stack import resolve_runtime_stack
from cli.storage_secrets import (
    StorageVolumes,
    ensure_storage_secrets,
    load_storage_secrets,
    storage_state_path,
)
from click.testing import CliRunner


def _provision_state(state_root):
    layout = resolve_runtime_stack()
    ensure_storage_secrets(
        str(state_root),
        stack_layout=layout,
        volumes=StorageVolumes(postgres="pg-volume", redis="redis-volume"),
    )
    return layout


def test_rotate_does_not_create_a_redis_container_for_a_stack_without_one(
    monkeypatch, tmp_path
):
    """Rotating a credential must not start a service the stack does not run.

    Redis holds its password only in the config file it started with, so when
    no container exists there is nothing left holding the previous value and
    rewriting that file already completes the rotation.
    """

    layout = _provision_state(tmp_path)
    previous = load_storage_secrets(str(tmp_path), stack_layout=layout)
    started = []

    def status(name):
        return "not found" if name == layout.redis_container_name else "running"

    monkeypatch.setattr(storage_command, "container_status", status)
    monkeypatch.setattr(
        storage_command, "rekey_managed_postgres", lambda _name, _secret: None
    )
    monkeypatch.setattr(
        storage_command,
        "container_start_redis",
        lambda *args, **kwargs: started.append(kwargs) or (0, "", ""),
    )

    result = CliRunner().invoke(
        main, ["storage", "rotate", "--config", str(tmp_path / "config.yaml")]
    )

    assert result.exit_code == 0, result.output
    assert started == []
    rotated = load_storage_secrets(str(tmp_path), stack_layout=layout)
    assert rotated.redis.password != previous.redis.password
    conf = storage_secrets.redis_conf_path(str(tmp_path), stack_layout=layout)
    assert conf.read_text(encoding="utf-8") == f"requirepass {rotated.redis.password}\n"


def test_rotate_refuses_when_postgres_is_not_running(monkeypatch, tmp_path):
    layout = _provision_state(tmp_path)
    before = storage_state_path(str(tmp_path), stack_layout=layout).read_bytes()
    monkeypatch.setattr(storage_command, "container_status", lambda _name: "exited")

    result = CliRunner().invoke(
        main, ["storage", "rotate", "--config", str(tmp_path / "config.yaml")]
    )

    assert result.exit_code != 0
    assert "is not running" in result.output
    assert storage_state_path(str(tmp_path), stack_layout=layout).read_bytes() == before


def test_rotate_redis_only_stack_does_not_require_postgres(monkeypatch, tmp_path):
    layout = _provision_state(tmp_path)
    rebuilt = []

    def status(name):
        return "not found" if name == layout.postgres_container_name else "running"

    monkeypatch.setattr(storage_command, "container_status", status)
    monkeypatch.setattr(
        storage_command,
        "container_start_redis",
        lambda _network, _layout, **kwargs: rebuilt.append(kwargs) or (0, "", ""),
    )

    result = CliRunner().invoke(
        main, ["storage", "rotate", "--config", str(tmp_path / "config.yaml")]
    )

    assert result.exit_code == 0, result.output
    assert len(rebuilt) == 1


def test_rotate_re_keys_postgres_and_rebuilds_redis_without_re_serving(
    monkeypatch, tmp_path, caplog
):
    layout = _provision_state(tmp_path)
    previous = load_storage_secrets(str(tmp_path), stack_layout=layout)
    steps = []

    monkeypatch.setattr(storage_command, "container_status", lambda _name: "running")
    monkeypatch.setattr(
        storage_command,
        "rekey_managed_postgres",
        lambda name, secret: steps.append(("alter-role", name, secret.password)),
    )
    monkeypatch.setattr(
        storage_command,
        "container_start_redis",
        lambda _network, _layout, **kwargs: steps.append(("rebuild-redis", kwargs))
        or (0, "", ""),
    )

    config_path = str(tmp_path / "config.yaml")
    with caplog.at_level(logging.WARNING, logger=storage_command.log.name):
        result = CliRunner().invoke(
            main, ["storage", "rotate", "--config", config_path]
        )

    assert result.exit_code == 0, result.output
    # Postgres first: restarting Router before the role accepts the new value
    # would start it on a credential Postgres has not taken yet.
    assert [step[0] for step in steps] == ["alter-role", "rebuild-redis"]
    rotated = load_storage_secrets(str(tmp_path), stack_layout=layout)
    assert rotated.postgres.password != previous.postgres.password
    assert rotated.redis.password != previous.redis.password
    # Rotation keeps the data where it is.
    assert rotated.postgres.volume == "pg-volume"
    assert rotated.redis.volume == "redis-volume"
    assert steps[0][2] == rotated.postgres.password
    # Redis reads `requirepass` once at startup, so reuse would keep serving
    # the revoked value.
    assert steps[1][1]["recreate"] is True
    assert steps[1][1]["data_volume"] == "redis-volume"
    assert layout.stack_name in result.output
    # Router keeps its credentials from container-create time, so the operator
    # is told to re-create it with the command they actually served with.
    assert "Router is still running on the previous storage credentials" in caplog.text
    assert "`vllm-sr serve`" in caplog.text
    assert rotated.postgres.password not in result.output
    assert rotated.redis.password not in result.output
    assert rotated.postgres.password not in caplog.text
    assert rotated.redis.password not in caplog.text


def test_rotate_needs_existing_credential_state(monkeypatch, tmp_path):
    monkeypatch.setattr(storage_command, "container_status", lambda _name: "running")

    result = CliRunner().invoke(
        main, ["storage", "rotate", "--config", str(tmp_path / "config.yaml")]
    )

    assert result.exit_code != 0


def test_rotate_never_re_serves_the_stack_on_the_operator_s_behalf(
    monkeypatch, tmp_path
):
    layout = _provision_state(tmp_path)
    invocations = []

    monkeypatch.setattr(storage_command, "container_status", lambda _name: "running")
    monkeypatch.setattr(
        storage_command, "rekey_managed_postgres", lambda _name, _secret: None
    )
    monkeypatch.setattr(
        storage_command,
        "container_start_redis",
        lambda _network, _layout, **_kwargs: (0, "", ""),
    )
    # Watch the work `serve` delegates to rather than the command object, so a
    # resurrected `ctx.invoke(serve, ...)` is caught however it is spelled.
    monkeypatch.setattr(
        runtime_commands,
        "_execute_serve",
        lambda config, *args, **kwargs: invocations.append((config, args, kwargs)),
    )

    config_path = str(tmp_path / "config.yaml")
    result = CliRunner().invoke(main, ["storage", "rotate", "--config", config_path])

    assert result.exit_code == 0, result.output
    # Re-serving from here would rebuild the stack out of click's defaults,
    # discarding the images, `--profile`, `--minimal`, and `--recipe-env` the
    # operator actually started it with -- and would simply fail for a stack
    # whose Recipe needs env bindings.
    assert invocations == []
    assert not hasattr(storage_command, "serve")
    assert layout.stack_name in result.output
