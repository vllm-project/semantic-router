import stat
from types import SimpleNamespace

from cli import container_openclaw_support


def _socket_stat(*, gid: int, mode: int = 0o660) -> SimpleNamespace:
    return SimpleNamespace(st_gid=gid, st_mode=stat.S_IFSOCK | mode)


def test_dashboard_mounts_runtime_socket_with_private_nonroot_group(
    tmp_path, monkeypatch
):
    mounts = []
    socket_path = tmp_path / "runtime.sock"
    socket_path.write_text("")
    monkeypatch.setattr(
        container_openclaw_support,
        "_socket_candidates",
        lambda _runtime: [str(socket_path)],
    )
    monkeypatch.setattr(
        container_openclaw_support,
        "_runtime_socket_is_group_safe",
        lambda _path: True,
    )

    attached = container_openclaw_support._attach_container_socket(mounts, "docker")

    assert attached is True
    assert mounts == [f"{socket_path}:/var/run/docker.sock"]


def test_dashboard_skips_runtime_socket_owned_by_root_group(tmp_path, monkeypatch):
    mounts = []
    socket_path = tmp_path / "runtime.sock"
    socket_path.write_text("")
    monkeypatch.setattr(
        container_openclaw_support,
        "_socket_candidates",
        lambda _runtime: [str(socket_path)],
    )
    monkeypatch.setattr(
        container_openclaw_support,
        "_runtime_socket_is_group_safe",
        lambda _path: False,
    )

    attached = container_openclaw_support._attach_container_socket(mounts, "docker")

    assert attached is False
    assert mounts == []


def test_runtime_socket_group_safety_policy():
    for info, expected in (
        (_socket_stat(gid=1234), True),
        (_socket_stat(gid=0), False),
        (SimpleNamespace(st_gid=1234, st_mode=stat.S_IFREG | 0o660), False),
        (_socket_stat(gid=1234, mode=0o600), False),
        (_socket_stat(gid=1234, mode=0o666), False),
        (SimpleNamespace(st_gid=1234, st_mode=stat.S_IFLNK | 0o777), False),
    ):
        assert (
            container_openclaw_support._runtime_socket_is_group_safe(
                "/runtime.sock", lstat_path=lambda _path, current=info: current
            )
            is expected
        )

    assert not container_openclaw_support._runtime_socket_is_group_safe(
        "/runtime.sock",
        lstat_path=lambda _path: (_ for _ in ()).throw(FileNotFoundError()),
    )


def test_dashboard_skips_unsafe_runtime_socket(tmp_path, monkeypatch):
    mounts = []
    socket_path = tmp_path / "runtime.sock"
    socket_path.write_text("")
    monkeypatch.setattr(
        container_openclaw_support,
        "_socket_candidates",
        lambda _runtime: [str(socket_path)],
    )
    monkeypatch.setattr(
        container_openclaw_support,
        "_runtime_socket_is_group_safe",
        lambda _path: False,
    )

    assert (
        container_openclaw_support._attach_container_socket(mounts, "podman") is False
    )
    assert mounts == []


def test_dashboard_marks_openclaw_runtime_disabled_for_unsafe_socket(
    tmp_path, monkeypatch
):
    mounts = []
    env_vars = {}
    socket_path = tmp_path / "runtime.sock"
    socket_path.write_text("")
    monkeypatch.setattr(
        container_openclaw_support,
        "_socket_candidates",
        lambda _runtime: [str(socket_path)],
    )

    container_openclaw_support.configure_openclaw_support(
        mounts,
        env_vars,
        str(tmp_path),
        None,
        "docker",
        SimpleNamespace(network_name="test-network"),
        resolve_container_cli=lambda _candidate: (_ for _ in ()).throw(
            AssertionError("container CLI must not be resolved without a safe socket")
        ),
    )

    assert (
        env_vars[container_openclaw_support.OPENCLAW_CONTAINER_RUNTIME_DISABLED_ENV]
        == "true"
    )
    assert "OPENCLAW_CONTAINER_RUNTIME" not in env_vars
    assert not any("docker.sock" in mount for mount in mounts)
