from __future__ import annotations

import resource
from pathlib import Path
from types import SimpleNamespace

import pytest
from cli.evaluation import sandbox


class _KernelTaskLimit:
    """Model the kernel's UID accounting independently of visible /proc tasks."""

    def __init__(self, tasks: int, inherited: tuple[int, int]) -> None:
        self.tasks = tasks
        self.limits = inherited
        self.changes: list[tuple[int, int]] = []
        self.started = 0
        self.joined = 0

    def getrlimit(self, key: int) -> tuple[int, int]:
        assert key == resource.RLIMIT_NPROC
        return self.limits

    def setrlimit(self, key: int, limits: tuple[int, int]) -> None:
        assert key == resource.RLIMIT_NPROC
        self.limits = limits
        self.changes.append(limits)

    def thread(self) -> object:
        kernel = self

        class Probe:
            def start(self) -> None:
                if kernel.limits[0] <= kernel.tasks:
                    raise RuntimeError("can't start new thread")
                kernel.started += 1

            def join(self) -> None:
                kernel.joined += 1

        return Probe()


def _install_kernel(monkeypatch: pytest.MonkeyPatch, kernel: _KernelTaskLimit) -> None:
    monkeypatch.setattr(sandbox.resource, "getrlimit", kernel.getrlimit)
    monkeypatch.setattr(sandbox.resource, "setrlimit", kernel.setrlimit)
    monkeypatch.setattr(sandbox.threading, "Thread", kernel.thread)
    monkeypatch.setattr(sandbox, "_require_single_worker_thread", lambda: None)
    monkeypatch.setattr(sandbox, "_wait_for_probe_exit", lambda probe: probe.join())


@pytest.mark.parametrize(
    ("tasks", "inherited", "expected"),
    [
        (341, (4096, 8192), 598),
        (341, (500, 8192), 500),
        (341, (500, 500), 500),
        (1_048_576, (resource.RLIM_INFINITY, resource.RLIM_INFINITY), 1_048_833),
    ],
)
def test_task_limit_measures_hidden_uid_tasks_and_respects_inherited_caps(
    monkeypatch: pytest.MonkeyPatch,
    tasks: int,
    inherited: tuple[int, int],
    expected: int,
) -> None:
    kernel = _KernelTaskLimit(tasks, inherited)
    _install_kernel(monkeypatch, kernel)

    assert sandbox._worker_task_limit() == expected
    assert kernel.limits == inherited
    assert kernel.started == kernel.joined
    assert kernel.started > 0
    assert len(kernel.changes) < 64
    assert all(hard == inherited[1] for _, hard in kernel.changes)
    if inherited[0] != resource.RLIM_INFINITY:
        assert all(soft <= inherited[0] for soft, _ in kernel.changes)


def test_task_limit_fails_closed_if_inherited_capacity_is_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel = _KernelTaskLimit(341, (300, 8192))
    _install_kernel(monkeypatch, kernel)

    with pytest.raises(sandbox.SandboxUnavailableError, match="inherited task limit"):
        sandbox._worker_task_limit()
    assert kernel.limits == (300, 8192)
    assert kernel.started == kernel.joined == 0
    assert kernel.changes == [(256, 8192), (300, 8192), (300, 8192)]
    assert all(soft <= 300 for soft, _ in kernel.changes)


def test_task_limit_restores_inherited_limits_on_unexpected_probe_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel = _KernelTaskLimit(341, (4096, 8192))
    _install_kernel(monkeypatch, kernel)

    class BrokenProbe:
        def start(self) -> None:
            raise RuntimeError("unexpected probe failure")

    monkeypatch.setattr(sandbox.threading, "Thread", BrokenProbe)
    with pytest.raises(RuntimeError, match="unexpected probe failure"):
        sandbox._worker_task_limit()
    assert kernel.limits == (4096, 8192)


def test_task_limit_rejects_zero_capacity_without_changing_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel = _KernelTaskLimit(1, (0, 8192))
    _install_kernel(monkeypatch, kernel)

    with pytest.raises(sandbox.SandboxUnavailableError, match="no capacity"):
        sandbox._worker_task_limit()
    assert kernel.changes == []


def test_worker_seals_measured_task_headroom_after_restoring_inherited_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel = _KernelTaskLimit(341, (4096, 8192))
    _install_kernel(monkeypatch, kernel)

    def getrlimit(key: int) -> tuple[int, int]:
        if key == resource.RLIMIT_NPROC:
            return kernel.getrlimit(key)
        return resource.RLIM_INFINITY, resource.RLIM_INFINITY

    def setrlimit(key: int, limits: tuple[int, int]) -> None:
        if key == resource.RLIMIT_NPROC:
            kernel.setrlimit(key, limits)

    monkeypatch.setattr(sandbox.resource, "getrlimit", getrlimit)
    monkeypatch.setattr(sandbox.resource, "setrlimit", setrlimit)
    monkeypatch.setattr(sandbox.threading, "Thread", kernel.thread)
    monkeypatch.setattr(
        sandbox.ctypes,
        "CDLL",
        lambda *_args, **_kwargs: SimpleNamespace(prctl=lambda *_: 0),
    )
    sandbox._apply_resource_limits(
        sandbox.WorkerSandboxPolicy(
            writable_root=Path("/"),
            suite_store=Path("/"),
            readable_roots=(),
            cpu_seconds=30,
        )
    )

    assert kernel.changes[-2:] == [(4096, 8192), (598, 598)]
    assert all(hard == 8192 for _, hard in kernel.changes[:-1])
    assert kernel.started == kernel.joined


def test_task_limit_rejects_preexisting_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sandbox.os, "listdir", lambda _path: ["1", "2"])
    with pytest.raises(sandbox.SandboxUnavailableError, match="one thread"):
        sandbox._worker_task_limit()


def test_task_probe_waits_for_kernel_task_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    exists = iter([True, False])
    sleeps = []
    joined = []
    probe = SimpleNamespace(
        native_id=123,
        join=lambda **kwargs: joined.append(kwargs),
        is_alive=lambda: False,
    )
    monkeypatch.setattr(sandbox.Path, "exists", lambda _path: next(exists))
    monkeypatch.setattr(sandbox.time, "sleep", sleeps.append)

    sandbox._wait_for_probe_exit(probe)

    assert joined == [{"timeout": 1.0}]
    assert sleeps == [0.001]


def test_task_probe_fails_closed_when_kernel_task_lingers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = iter([0.0, 1.0])
    probe = SimpleNamespace(
        native_id=123, join=lambda **_kwargs: None, is_alive=lambda: False
    )
    monkeypatch.setattr(sandbox.Path, "exists", lambda _path: True)
    monkeypatch.setattr(sandbox.time, "monotonic", lambda: next(clock))

    with pytest.raises(sandbox.SandboxUnavailableError, match="exit the kernel"):
        sandbox._wait_for_probe_exit(probe)
