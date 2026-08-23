import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest
from cli.runtime_lifecycle_lock import (
    RuntimeLifecycleLockError,
    acquire_runtime_lifecycle_lock,
)

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX file lock contract")


def test_runtime_lifecycle_lock_is_private_noninheritable_and_reusable(
    tmp_path: Path,
):
    lock_root = tmp_path / "locks"
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    lock_root = tmp_path / "locks"
    other_workspace = tmp_path / "other-workspace"
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


def test_runtime_lifecycle_lock_separates_runtime_and_stack_keys(tmp_path: Path):
    lock_root = tmp_path / "locks"
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


def test_runtime_lifecycle_lock_releases_after_exception(tmp_path: Path):
    lock_root = tmp_path / "locks"
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
    tmp_path: Path,
):
    lock_root = tmp_path / "locks"
    with acquire_runtime_lifecycle_lock(
        runtime="docker", stack_name="audit-a", lock_root=lock_root
    ) as lock:
        lock_path = lock.lock_path

    outside = tmp_path / "outside-lock"
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


def test_runtime_lifecycle_lock_rejects_symlinked_directory(tmp_path: Path):
    real_root = tmp_path / "real-locks"
    real_root.mkdir()
    symlink_root = tmp_path / "linked-locks"
    symlink_root.symlink_to(real_root, target_is_directory=True)

    with pytest.raises(RuntimeLifecycleLockError, match="path cannot be opened safely"):
        acquire_runtime_lifecycle_lock(
            runtime="docker", stack_name="audit-a", lock_root=symlink_root
        )


def test_runtime_lifecycle_lock_rejects_user_owned_writable_ancestor(tmp_path: Path):
    writable_ancestor = tmp_path / "shared"
    writable_ancestor.mkdir(mode=0o770)
    writable_ancestor.chmod(0o770)

    with pytest.raises(RuntimeLifecycleLockError, match="not safely owned"):
        acquire_runtime_lifecycle_lock(
            runtime="docker",
            stack_name="audit-a",
            lock_root=writable_ancestor / "runtime" / "locks",
        )
