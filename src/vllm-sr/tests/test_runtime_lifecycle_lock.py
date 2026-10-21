import json
import os
import signal
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest
from cli.runtime_lifecycle_lock import (
    RuntimeLifecycleLockError,
    _resolve_lock_directory,
    acquire_runtime_lifecycle_lock,
)
from cli.pending_activation import heartbeat_file

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX file lock contract")


@pytest.fixture
def private_tmp_path():
    # Lifecycle locks reject writable ancestors, including Linux /tmp. Keep
    # this fixture under the user-owned home so the tests reach the lock itself.
    with tempfile.TemporaryDirectory(
        prefix=".vllm-sr-lock-test-", dir=Path.home().resolve()
    ) as directory:
        yield Path(directory)


def test_runtime_lifecycle_lock_is_private_noninheritable_and_reusable(
    private_tmp_path: Path,
):
    lock_root = private_tmp_path / "locks"
    with acquire_runtime_lifecycle_lock(
        runtime="docker",
        stack_name="audit-a",
        lock_root=lock_root,
    ) as lock:
        info = lock.lock_path.stat()
        assert stat.S_IMODE(lock_root.stat().st_mode) == 0o700
        assert stat.S_IMODE(info.st_mode) == 0o600
        assert info.st_nlink == 1
        assert os.get_inheritable(lock._lock_fd) is False

    with acquire_runtime_lifecycle_lock(
        runtime="docker",
        stack_name="audit-a",
        lock_root=lock_root,
    ):
        pass


def test_runtime_lifecycle_lock_contends_across_working_directories(
    private_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    lock_root = private_tmp_path / "locks"
    other_workspace = private_tmp_path / "other-workspace"
    other_workspace.mkdir()

    with acquire_runtime_lifecycle_lock(
        runtime="docker",
        stack_name="audit-a",
        lock_root=lock_root,
    ):
        monkeypatch.chdir(other_workspace)
        source_root = Path(__file__).resolve().parents[1]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(source_root)
        contender = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; "
                    "from cli.runtime_lifecycle_lock import "
                    "acquire_runtime_lifecycle_lock; "
                    "acquire_runtime_lifecycle_lock(runtime='docker', "
                    "stack_name='audit-a', lock_root=sys.argv[1])"
                ),
                str(lock_root),
            ],
            cwd=other_workspace,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )

    assert contender.returncode != 0
    assert "another lifecycle operation in progress" in contender.stderr
    assert f"pid {os.getpid()}" in contender.stderr
    assert "command 'vllm-sr lifecycle'" in contender.stderr
    assert "state active" in contender.stderr
    assert "started " in contender.stderr
    assert f"kill -TERM {os.getpid()}" in contender.stderr


def _serve_holder(private_tmp_path: Path, state: str, script_name: str = "vllm-sr"):
    lock_root = private_tmp_path / "locks"
    config = private_tmp_path / "runtime-config.yaml"
    script = private_tmp_path / script_name
    script.write_text(
        """import signal
import sys
import time
from cli.pending_activation import serve_heartbeat, SERVE_WAITING, heartbeat_file
from cli.runtime_lifecycle_lock import acquire_runtime_lifecycle_lock

signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
with acquire_runtime_lifecycle_lock(
    runtime='docker', stack_name='audit-a', lock_root=sys.argv[2],
    command='vllm-sr serve', state='starting',
    heartbeat_path=heartbeat_file(sys.argv[3]),
):
    with serve_heartbeat(sys.argv[3], interval=60) as set_state:
        if sys.argv[4] == 'waiting':
            set_state(SERVE_WAITING)
        while True:
            time.sleep(1)
""",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    process = subprocess.Popen(
        [sys.executable, str(script), "serve", str(lock_root), str(config), state],
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + 5
    heartbeat = heartbeat_file(config)
    while not heartbeat.exists() and time.monotonic() < deadline:
        if process.poll() is not None:
            pytest.fail(f"serve holder exited: {process.stderr.read()}")
        time.sleep(0.01)
    assert heartbeat.exists()
    return process, lock_root, heartbeat


@pytest.mark.skipif(
    not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"),
    reason="safe process signaling requires Linux pidfd",
)
@pytest.mark.parametrize("operation", ["serve", "stop"])
def test_waiting_serve_is_taken_over(private_tmp_path: Path, operation: str):
    process, lock_root, _heartbeat = _serve_holder(private_tmp_path, "waiting")
    try:
        with acquire_runtime_lifecycle_lock(
            runtime="docker",
            stack_name="audit-a",
            lock_root=lock_root,
            command=f"vllm-sr {operation}",
            take_over_waiting=True,
        ):
            assert process.wait(timeout=2) == 0
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
        process.stderr.close()


@pytest.mark.skipif(
    not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"),
    reason="safe process signaling requires Linux pidfd",
)
@pytest.mark.parametrize(
    "invalid", ["starting", "stale", "reused-pid", "wrong-command"]
)
def test_takeover_refuses_unverified_holder(private_tmp_path: Path, invalid: str):
    state = "starting" if invalid == "starting" else "waiting"
    process, lock_root, heartbeat = _serve_holder(
        private_tmp_path,
        state,
        script_name="other-cli" if invalid == "wrong-command" else "vllm-sr",
    )
    try:
        if invalid == "stale":
            old = time.time() - 30
            os.utime(heartbeat, (old, old))
        elif invalid == "reused-pid":
            record = json.loads(heartbeat.read_text(encoding="utf-8"))
            record["start_ticks"] += 1
            heartbeat.write_text(json.dumps(record), encoding="utf-8")
        with pytest.raises(RuntimeLifecycleLockError, match="another lifecycle"):
            acquire_runtime_lifecycle_lock(
                runtime="docker",
                stack_name="audit-a",
                lock_root=lock_root,
                command="vllm-sr stop",
                take_over_waiting=True,
            )
        assert process.poll() is None
    finally:
        process.terminate()
        process.wait(timeout=5)
        process.stderr.close()


def test_runtime_lifecycle_lock_separates_runtime_and_stack_keys(
    private_tmp_path: Path,
):
    lock_root = private_tmp_path / "locks"
    with (
        acquire_runtime_lifecycle_lock(
            runtime="docker", stack_name="audit-a", lock_root=lock_root
        ),
        acquire_runtime_lifecycle_lock(
            runtime="docker", stack_name="audit-b", lock_root=lock_root
        ),
        acquire_runtime_lifecycle_lock(
            runtime="podman", stack_name="audit-a", lock_root=lock_root
        ),
    ):
        pass


def test_runtime_lifecycle_lock_releases_after_exception(private_tmp_path: Path):
    lock_root = private_tmp_path / "locks"
    with (
        pytest.raises(RuntimeError, match="deployment failed"),
        acquire_runtime_lifecycle_lock(
            runtime="docker", stack_name="audit-a", lock_root=lock_root
        ),
    ):
        raise RuntimeError("deployment failed")

    with acquire_runtime_lifecycle_lock(
        runtime="docker", stack_name="audit-a", lock_root=lock_root
    ):
        pass


def test_runtime_lifecycle_lock_rejects_symlinked_or_linked_lock_file(
    private_tmp_path: Path,
):
    lock_root = private_tmp_path / "locks"
    with acquire_runtime_lifecycle_lock(
        runtime="docker", stack_name="audit-a", lock_root=lock_root
    ) as lock:
        lock_path = lock.lock_path

    outside = private_tmp_path / "outside-lock"
    outside.write_text("", encoding="utf-8")
    lock_path.unlink()
    lock_path.symlink_to(outside)
    with pytest.raises(RuntimeLifecycleLockError, match="cannot be opened safely"):
        acquire_runtime_lifecycle_lock(
            runtime="docker", stack_name="audit-a", lock_root=lock_root
        )

    lock_path.unlink()
    os.link(outside, lock_path)
    with pytest.raises(RuntimeLifecycleLockError, match="private regular file"):
        acquire_runtime_lifecycle_lock(
            runtime="docker", stack_name="audit-a", lock_root=lock_root
        )


def test_runtime_lifecycle_lock_rejects_symlinked_directory(private_tmp_path: Path):
    real_root = private_tmp_path / "real-locks"
    real_root.mkdir()
    symlink_root = private_tmp_path / "linked-locks"
    symlink_root.symlink_to(real_root, target_is_directory=True)

    with pytest.raises(RuntimeLifecycleLockError, match="path cannot be opened safely"):
        acquire_runtime_lifecycle_lock(
            runtime="docker", stack_name="audit-a", lock_root=symlink_root
        )


def test_resolve_lock_directory_falls_through_when_xdg_runtime_dir_is_unusable(
    tmp_path: Path,
):
    # WSL images export XDG_RUNTIME_DIR=/run/user/<uid> without a systemd
    # session, so the directory is named but never created. Resolution falls
    # back to the state-home location instead of aborting the serve flow.
    state_home = tmp_path / "state"
    state_home.mkdir()
    missing_runtime = tmp_path / "missing-runtime"

    lock_directory = _resolve_lock_directory(
        missing_runtime, str(missing_runtime), str(state_home)
    )

    assert lock_directory == state_home / "vllm-sr" / "locks"


def test_resolve_lock_directory_falls_through_when_xdg_runtime_dir_is_relative(
    tmp_path: Path,
):
    state_home = tmp_path / "state"
    state_home.mkdir()

    lock_directory = _resolve_lock_directory(
        tmp_path / "missing-runtime", "relative/runtime", str(state_home)
    )

    assert lock_directory == state_home / "vllm-sr" / "locks"


def test_resolve_lock_directory_prefers_usable_xdg_runtime_dir(tmp_path: Path):
    runtime_home = tmp_path / "runtime"
    runtime_home.mkdir(mode=0o700)

    lock_directory = _resolve_lock_directory(
        tmp_path / "missing-runtime", str(runtime_home), str(tmp_path / "state")
    )

    assert lock_directory == runtime_home / "vllm-sr" / "locks"


def test_resolve_lock_directory_rejects_unusable_state_home(tmp_path: Path):
    with pytest.raises(RuntimeLifecycleLockError, match="absolute path"):
        _resolve_lock_directory(
            tmp_path / "missing-runtime", "relative/runtime", "relative/state"
        )


def test_resolve_lock_directory_prefers_linux_runtime_dir(tmp_path: Path):
    linux_runtime = tmp_path / "run-user-0"
    linux_runtime.mkdir(mode=0o700)

    lock_directory = _resolve_lock_directory(
        linux_runtime, str(tmp_path / "runtime"), str(tmp_path / "state")
    )

    assert lock_directory == linux_runtime / "vllm-sr" / "locks"
