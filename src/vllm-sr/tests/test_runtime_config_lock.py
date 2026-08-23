from __future__ import annotations

import os
import stat
import time
from pathlib import Path

import pytest
import yaml
from cli.bootstrap import BootstrapResult
from cli.commands import runtime as runtime_commands
from cli.main import main
from cli.runtime_config_lock import (
    CompiledBootstrapLockError,
    LOCK_FILENAME,
    acquire_compiled_bootstrap_lock,
)
from click.testing import CliRunner

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX file lock contract")


def test_compiled_bootstrap_lock_is_private_non_inheritable_and_reusable(
    tmp_path: Path,
):
    compiled_bootstrap = tmp_path / ".vllm-sr" / "compiled-bootstrap.yaml"
    compiled_bootstrap.parent.mkdir()

    with acquire_compiled_bootstrap_lock(
        compiled_bootstrap_path=compiled_bootstrap,
        state_root_dir=tmp_path,
        stack_name="audit",
    ) as lock:
        lock.assert_matches(
            compiled_bootstrap_path=compiled_bootstrap,
            state_root_dir=tmp_path,
            stack_name="audit",
        )
        assert os.get_inheritable(lock._lock_fd) is False
        lock_path = lock.coordination_dir / LOCK_FILENAME
        assert stat.S_IMODE(lock_path.stat().st_mode) == 0o600

    with acquire_compiled_bootstrap_lock(
        compiled_bootstrap_path=compiled_bootstrap,
        state_root_dir=tmp_path,
        stack_name="audit",
        timeout_seconds=0,
    ):
        lock_path = tmp_path / ".vllm-sr" / "runtime-locks" / "audit" / LOCK_FILENAME
        lock_path.chmod(0o660)

    with acquire_compiled_bootstrap_lock(
        compiled_bootstrap_path=compiled_bootstrap,
        state_root_dir=tmp_path,
        stack_name="audit",
        timeout_seconds=0,
    ):
        assert stat.S_IMODE(lock_path.stat().st_mode) == 0o660


def test_compiled_bootstrap_lock_repairs_parent_traversal_without_exposing_stack(
    tmp_path: Path,
):
    runtime_dir = tmp_path / ".vllm-sr"
    lock_parent = runtime_dir / "runtime-locks"
    lock_parent.mkdir(parents=True)
    runtime_dir.chmod(0o700)
    lock_parent.chmod(0o700)

    with acquire_compiled_bootstrap_lock(
        compiled_bootstrap_path=runtime_dir / "compiled-bootstrap.yaml",
        state_root_dir=tmp_path,
        stack_name="audit",
    ) as lock:
        runtime_mode = stat.S_IMODE(runtime_dir.stat().st_mode)
        parent_mode = stat.S_IMODE(lock_parent.stat().st_mode)
        stack_mode = stat.S_IMODE(lock.coordination_dir.stat().st_mode)

    assert runtime_mode == 0o700
    assert parent_mode == 0o711
    assert stack_mode == 0o700
    assert parent_mode & stat.S_IRWXO == stat.S_IXOTH


def test_compiled_bootstrap_lock_rejects_contention_and_token_mismatch(tmp_path: Path):
    compiled_bootstrap = tmp_path / ".vllm-sr" / "compiled-bootstrap.yaml"
    compiled_bootstrap.parent.mkdir()

    with acquire_compiled_bootstrap_lock(
        compiled_bootstrap_path=compiled_bootstrap,
        state_root_dir=tmp_path,
        stack_name="audit",
    ) as lock:
        with pytest.raises(
            CompiledBootstrapLockError, match="operation is in progress"
        ):
            acquire_compiled_bootstrap_lock(
                compiled_bootstrap_path=compiled_bootstrap,
                state_root_dir=tmp_path,
                stack_name="audit",
                timeout_seconds=0,
            )
        with pytest.raises(
            CompiledBootstrapLockError, match="does not match this stack"
        ):
            lock.assert_matches(
                compiled_bootstrap_path=compiled_bootstrap,
                state_root_dir=tmp_path,
                stack_name="other",
            )


def test_serve_fails_fast_when_compiled_bootstrap_is_locked(
    monkeypatch, tmp_path: Path
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "version": "v0.4",
                "listeners": [
                    {"name": "http-8899", "address": "0.0.0.0", "port": 8899}
                ],
                "global": {
                    "services": {
                        "backend_egress": {
                            "policy_file": "/app/config/backend-egress-policy.yaml"
                        }
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    state_root = tmp_path / "host-state"
    state_root.mkdir(mode=0o700)
    bootstrap = BootstrapResult(
        config_path=config_path,
        output_dir=tmp_path / ".vllm-sr",
    )
    stack_name = runtime_commands.resolve_runtime_stack().stack_name
    compiled_bootstrap_path = state_root / ".vllm-sr" / "compiled-bootstrap.yaml"

    monkeypatch.setenv("VLLM_SR_STATE_ROOT_DIR", str(state_root))
    monkeypatch.setattr(
        runtime_commands,
        "ensure_bootstrap_workspace",
        lambda *_args, **_kwargs: bootstrap,
    )
    monkeypatch.setattr(
        runtime_commands,
        "_build_backend",
        lambda *_a, **_kw: (_ for _ in ()).throw(
            AssertionError("a contending serve must not reach deployment")
        ),
    )

    with acquire_compiled_bootstrap_lock(
        compiled_bootstrap_path=compiled_bootstrap_path,
        state_root_dir=state_root,
        stack_name=stack_name,
        timeout_seconds=0,
    ):
        started_at = time.monotonic()
        result = CliRunner().invoke(
            main,
            ["serve", "--image-pull-policy", "never"],
        )
        elapsed = time.monotonic() - started_at

    assert result.exit_code != 0
    assert "operation is in progress" in result.output
    assert elapsed < 1.0


def test_compiled_bootstrap_lock_rejects_symlinked_coordination_component(
    tmp_path: Path,
):
    runtime_dir = tmp_path / ".vllm-sr"
    runtime_dir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (runtime_dir / "runtime-locks").symlink_to(outside, target_is_directory=True)

    with pytest.raises(CompiledBootstrapLockError, match="must be a real directory"):
        acquire_compiled_bootstrap_lock(
            compiled_bootstrap_path=runtime_dir / "compiled-bootstrap.yaml",
            state_root_dir=tmp_path,
            stack_name="audit",
        )


def test_compiled_bootstrap_lock_rejects_symlinked_or_linked_lock_file(
    tmp_path: Path,
):
    runtime_dir = tmp_path / ".vllm-sr"
    lock_dir = runtime_dir / "runtime-locks" / "audit"
    lock_dir.mkdir(parents=True)
    outside = tmp_path / "outside-lock"
    outside.write_text("sentinel", encoding="utf-8")
    lock_path = lock_dir / LOCK_FILENAME
    lock_path.symlink_to(outside)

    with pytest.raises(CompiledBootstrapLockError):
        acquire_compiled_bootstrap_lock(
            compiled_bootstrap_path=runtime_dir / "compiled-bootstrap.yaml",
            state_root_dir=tmp_path,
            stack_name="audit",
        )
    assert outside.read_text(encoding="utf-8") == "sentinel"

    lock_path.unlink()
    lock_path.write_text("", encoding="utf-8")
    linked = tmp_path / "linked-lock"
    os.link(lock_path, linked)
    with pytest.raises(CompiledBootstrapLockError, match="private regular file"):
        acquire_compiled_bootstrap_lock(
            compiled_bootstrap_path=runtime_dir / "compiled-bootstrap.yaml",
            state_root_dir=tmp_path,
            stack_name="audit",
        )


def test_compiled_bootstrap_lock_rejects_access_for_other_users(tmp_path: Path):
    runtime_dir = tmp_path / ".vllm-sr"
    lock_dir = runtime_dir / "runtime-locks" / "audit"
    lock_dir.mkdir(parents=True)
    lock_path = lock_dir / LOCK_FILENAME
    lock_path.write_text("", encoding="utf-8")
    lock_path.chmod(0o606)

    with pytest.raises(CompiledBootstrapLockError, match="other users"):
        acquire_compiled_bootstrap_lock(
            compiled_bootstrap_path=runtime_dir / "compiled-bootstrap.yaml",
            state_root_dir=tmp_path,
            stack_name="audit",
        )
