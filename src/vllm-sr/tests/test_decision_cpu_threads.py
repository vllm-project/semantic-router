"""CPU thread limits for the isolated Decision process."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from decision_runtime import cpu_threads, entrypoint  # noqa: E402


@pytest.fixture
def many_visible_cpus(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cpu_threads.os, "sched_getaffinity", lambda _pid: set(range(64))
    )
    monkeypatch.setattr(cpu_threads.os, "cpu_count", lambda: 64)


def test_default_thread_budget_observes_cgroup_v2_quota(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, many_visible_cpus: None
) -> None:
    (tmp_path / "cpu.max").write_text("800000 100000\n", encoding="ascii")
    monkeypatch.delenv("DECISION_CPU_THREADS", raising=False)

    assert cpu_threads.configure_cpu_threads(cgroup_root=tmp_path) == 8
    assert all(
        cpu_threads.os.environ[key] == "8" for key in cpu_threads.CPU_THREAD_ENVIRONMENT
    )


def test_default_thread_budget_observes_cgroup_v1_and_affinity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, many_visible_cpus: None
) -> None:
    cpu = tmp_path / "cpu"
    cpu.mkdir()
    (cpu / "cpu.cfs_quota_us").write_text("250000\n", encoding="ascii")
    (cpu / "cpu.cfs_period_us").write_text("100000\n", encoding="ascii")
    monkeypatch.delenv("DECISION_CPU_THREADS", raising=False)

    assert cpu_threads.configure_cpu_threads(cgroup_root=tmp_path) == 2
    monkeypatch.setattr(cpu_threads.os, "sched_getaffinity", lambda _pid: {0, 1})
    assert cpu_threads.available_cpu_count(tmp_path) == 2


def test_default_thread_budget_caps_large_unlimited_host(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, many_visible_cpus: None
) -> None:
    (tmp_path / "cpu.max").write_text("max 100000\n", encoding="ascii")
    monkeypatch.delenv("DECISION_CPU_THREADS", raising=False)

    assert cpu_threads.configure_cpu_threads(cgroup_root=tmp_path) == 8


def test_explicit_cpu_threads_override_default_quota(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, many_visible_cpus: None
) -> None:
    (tmp_path / "cpu.max").write_text("200000 100000\n", encoding="ascii")
    monkeypatch.setenv("DECISION_CPU_THREADS", "12")

    assert cpu_threads.configure_cpu_threads(cgroup_root=tmp_path) == 12
    assert all(
        cpu_threads.os.environ[key] == "12"
        for key in cpu_threads.CPU_THREAD_ENVIRONMENT
    )


@pytest.mark.parametrize("value", ("0", "257", "abc", " 8", "8 ", "1.5"))
def test_invalid_explicit_cpu_threads_fail_before_environment_change(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, value: str
) -> None:
    monkeypatch.setenv("DECISION_CPU_THREADS", value)
    monkeypatch.setenv("OMP_NUM_THREADS", "7")

    with pytest.raises(ValueError, match="DECISION_CPU_THREADS"):
        cpu_threads.configure_cpu_threads(cgroup_root=tmp_path)

    assert cpu_threads.os.environ["OMP_NUM_THREADS"] == "7"


def test_entrypoint_configures_cpu_threads_before_server_imports(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events: list[str] = []
    server = ModuleType("decision_runtime.server")
    server.run_server = lambda _config: events.append("server")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "decision_runtime.server", server)
    monkeypatch.setattr(
        entrypoint,
        "configure_cpu_threads",
        lambda: events.append("threads"),
    )
    monkeypatch.setattr(
        entrypoint.importlib,
        "import_module",
        lambda name: events.append(name) or ModuleType(name),
    )

    entrypoint.main(
        [
            "--model",
            "fixture",
            "--revision",
            "a" * 40,
            "--backend",
            "cpu",
            "--artifact-root",
            str(tmp_path),
            "--artifact-content-id",
            "b" * 64,
            "--max-batch",
            "1",
            "--max-concurrency",
            "1",
            "--max-queue",
            "0",
        ]
    )

    assert events == ["threads", "fastapi", "uvicorn", "server"]


def test_gpu_entrypoint_does_not_change_cpu_thread_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events: list[str] = []
    server = ModuleType("decision_runtime.server")
    server.run_server = lambda _config: events.append("server")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "decision_runtime.server", server)
    monkeypatch.setattr(
        entrypoint,
        "configure_cpu_threads",
        lambda: events.append("threads"),
    )
    monkeypatch.setattr(
        entrypoint.importlib,
        "import_module",
        lambda name: events.append(name) or ModuleType(name),
    )

    entrypoint.main(
        [
            "--model",
            "fixture",
            "--revision",
            "a" * 40,
            "--backend",
            "rocm",
            "--artifact-root",
            str(tmp_path),
            "--artifact-content-id",
            "b" * 64,
            "--max-batch",
            "1",
            "--max-concurrency",
            "1",
            "--max-queue",
            "0",
        ]
    )

    assert events == ["fastapi", "uvicorn", "server"]
