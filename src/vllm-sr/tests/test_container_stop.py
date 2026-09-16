"""Bounded stop tests using neutral subprocesses, without a container daemon."""

import os
import subprocess
import sys
import textwrap
import time

import pytest
from cli import container_services


@pytest.mark.parametrize("runtime", ["docker", "podman"])
def test_stop_requests_router_drain_grace_and_bounds_command(monkeypatch, runtime):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(container_services, "get_container_runtime", lambda: runtime)
    monkeypatch.setattr(container_services.subprocess, "run", run)

    assert container_services.container_stop_container("owned-router") is True
    assert calls == [
        (
            [runtime, "stop", "--time", "40", "owned-router"],
            {"check": True, "capture_output": True, "timeout": 50},
        )
    ]


@pytest.mark.parametrize(
    "failure",
    [OSError("runtime unavailable"), subprocess.CalledProcessError(1, ["stop"])],
)
def test_stop_failure_is_not_reported_as_stopped(monkeypatch, failure):
    def fail(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(container_services, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(container_services.subprocess, "run", fail)
    assert container_services.container_stop_container("owned-router") is False


def _runtime_executable(tmp_path, body):
    runtime = tmp_path / "runtime"
    runtime.write_text(f"#!{sys.executable}\n" + textwrap.dedent(body))
    runtime.chmod(0o700)
    return str(runtime)


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX process signals")
def test_stop_allows_sigterm_cleanup_before_forced_exit(monkeypatch, tmp_path):
    # Scale seconds in the stand-in daemon, while asserting the real 40/50s
    # contract above. The child actually receives SIGTERM and closes its owner.
    (tmp_path / "worker.py").write_text(
        textwrap.dedent(
            """
            import pathlib
            import signal
            import sys
            import time

            def close_owner(_signum, _frame):
                time.sleep(0.2)
                pathlib.Path(sys.argv[1]).write_text("closed")
                sys.exit(0)

            signal.signal(signal.SIGTERM, close_owner)
            print("ready", flush=True)
            signal.pause()
            """
        )
    )
    runtime = _runtime_executable(
        tmp_path,
        """
        import pathlib
        import signal
        import subprocess
        import sys

        worker = str(pathlib.Path(__file__).with_name("worker.py"))
        grace = int(sys.argv[sys.argv.index("--time") + 1]) if "--time" in sys.argv else 10
        child = subprocess.Popen(
            [sys.executable, worker, sys.argv[-1]],
            stdout=subprocess.PIPE, text=True,
        )
        try:
            assert child.stdout.readline().strip() == "ready"
            child.send_signal(signal.SIGTERM)
            try:
                status = child.wait(timeout=grace / 100)
            except subprocess.TimeoutExpired:
                child.kill()
                status = child.wait(timeout=2)
            sys.exit(0 if status == 0 else 1)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=2)
        """,
    )
    monkeypatch.setattr(container_services, "get_container_runtime", lambda: runtime)
    closed = tmp_path / "owner-closed"

    assert container_services.container_stop_container(str(closed)) is True
    assert closed.read_text() == "closed"


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX process signals")
def test_stop_hung_runtime_is_bounded_reaped_and_unconfirmed(monkeypatch, tmp_path):
    runtime = _runtime_executable(
        tmp_path,
        """
        import signal

        signal.pause()
        """,
    )
    monkeypatch.setattr(container_services, "get_container_runtime", lambda: runtime)
    monkeypatch.setattr(container_services, "CONTAINER_STOP_COMMAND_TIMEOUT_SECONDS", 1)
    process_ids = []
    real_popen = subprocess.Popen

    def track_process(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        process_ids.append(process.pid)
        return process

    monkeypatch.setattr(container_services.subprocess, "Popen", track_process)
    started = time.monotonic()

    assert container_services.container_stop_container("owned-router") is False
    assert time.monotonic() - started < 5
    assert len(process_ids) == 1
    with pytest.raises(ProcessLookupError):
        os.kill(process_ids[0], 0)
