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
    LOCK_FILENAME,
    RuntimeConfigLockError,
    acquire_runtime_config_lock,
)
from cli.runtime_stack import resolve_runtime_stack
from click.testing import CliRunner

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX file lock contract")


def test_runtime_config_lock_is_private_non_inheritable_and_reusable(
    tmp_path: Path,
):
    runtime_config = tmp_path / ".vllm-sr" / "runtime-config.yaml"
    runtime_config.parent.mkdir()

    with acquire_runtime_config_lock(
        runtime_config_path=runtime_config,
        state_root_dir=tmp_path,
        stack_name="audit",
    ) as lock:
        lock.assert_matches(
            runtime_config_path=runtime_config,
            state_root_dir=tmp_path,
            stack_name="audit",
        )
        assert os.get_inheritable(lock._lock_fd) is False
        lock_path = lock.store_dir / LOCK_FILENAME
        assert stat.S_IMODE(lock_path.stat().st_mode) == 0o600

    with acquire_runtime_config_lock(
        runtime_config_path=runtime_config,
        state_root_dir=tmp_path,
        stack_name="audit",
        timeout_seconds=0,
    ):
        lock_path = tmp_path / ".vllm-sr" / "recipe-store" / "audit" / LOCK_FILENAME
        lock_path.chmod(0o660)

    with acquire_runtime_config_lock(
        runtime_config_path=runtime_config,
        state_root_dir=tmp_path,
        stack_name="audit",
        timeout_seconds=0,
    ):
        assert stat.S_IMODE(lock_path.stat().st_mode) == 0o660


def test_runtime_config_lock_repairs_parent_traversal_without_exposing_stack(
    tmp_path: Path,
):
    runtime_dir = tmp_path / ".vllm-sr"
    store_parent = runtime_dir / "recipe-store"
    store_parent.mkdir(parents=True)
    runtime_dir.chmod(0o700)
    store_parent.chmod(0o700)

    with acquire_runtime_config_lock(
        runtime_config_path=runtime_dir / "runtime-config.yaml",
        state_root_dir=tmp_path,
        stack_name="audit",
    ) as lock:
        runtime_mode = stat.S_IMODE(runtime_dir.stat().st_mode)
        parent_mode = stat.S_IMODE(store_parent.stat().st_mode)
        stack_mode = stat.S_IMODE(lock.store_dir.stat().st_mode)

    assert runtime_mode == 0o700
    assert parent_mode == 0o711
    assert stack_mode == 0o700
    assert parent_mode & stat.S_IRWXO == stat.S_IXOTH


def test_runtime_config_lock_rejects_contention_and_token_mismatch(tmp_path: Path):
    runtime_config = tmp_path / ".vllm-sr" / "runtime-config.yaml"
    runtime_config.parent.mkdir()

    with acquire_runtime_config_lock(
        runtime_config_path=runtime_config,
        state_root_dir=tmp_path,
        stack_name="audit",
    ) as lock:
        with pytest.raises(RuntimeConfigLockError, match="operation is in progress"):
            acquire_runtime_config_lock(
                runtime_config_path=runtime_config,
                state_root_dir=tmp_path,
                stack_name="audit",
                timeout_seconds=0,
            )
        with pytest.raises(RuntimeConfigLockError, match="does not match this stack"):
            lock.assert_matches(
                runtime_config_path=runtime_config,
                state_root_dir=tmp_path,
                stack_name="other",
            )


def test_serve_fails_fast_when_runtime_config_is_already_mutating(
    monkeypatch, tmp_path: Path
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "version": "v0.3",
                "listeners": [
                    {"name": "http-8899", "address": "0.0.0.0", "port": 8899}
                ],
                "routing": {"decisions": [{"name": "default"}]},
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
        setup_mode=False,
    )
    stack_name = resolve_runtime_stack().stack_name
    runtime_config_path = state_root / ".vllm-sr" / "runtime-config.yaml"

    monkeypatch.setenv("VLLM_SR_STATE_ROOT_DIR", str(state_root))
    monkeypatch.setattr(
        runtime_commands, "ensure_bootstrap_workspace", lambda _: bootstrap
    )
    monkeypatch.setattr(
        runtime_commands,
        "_build_backend",
        lambda *_a, **_kw: (_ for _ in ()).throw(
            AssertionError("a contending serve must not reach deployment")
        ),
    )

    with acquire_runtime_config_lock(
        runtime_config_path=runtime_config_path,
        state_root_dir=state_root,
        stack_name=stack_name,
        timeout_seconds=0,
    ):
        started_at = time.monotonic()
        result = CliRunner().invoke(
            main,
            ["serve", "--config", str(config_path), "--image-pull-policy", "never"],
        )
        elapsed = time.monotonic() - started_at

    assert result.exit_code != 0
    assert "operation is in progress" in result.output
    assert elapsed < 1.0


def test_runtime_config_lock_rejects_symlinked_store_component(tmp_path: Path):
    runtime_dir = tmp_path / ".vllm-sr"
    runtime_dir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (runtime_dir / "recipe-store").symlink_to(outside, target_is_directory=True)

    with pytest.raises(RuntimeConfigLockError, match="must be a real directory"):
        acquire_runtime_config_lock(
            runtime_config_path=runtime_dir / "runtime-config.yaml",
            state_root_dir=tmp_path,
            stack_name="audit",
        )


def test_runtime_config_lock_rejects_symlinked_or_linked_lock_file(tmp_path: Path):
    runtime_dir = tmp_path / ".vllm-sr"
    store_dir = runtime_dir / "recipe-store" / "audit"
    store_dir.mkdir(parents=True)
    outside = tmp_path / "outside-lock"
    outside.write_text("sentinel", encoding="utf-8")
    lock_path = store_dir / LOCK_FILENAME
    lock_path.symlink_to(outside)

    with pytest.raises(RuntimeConfigLockError):
        acquire_runtime_config_lock(
            runtime_config_path=runtime_dir / "runtime-config.yaml",
            state_root_dir=tmp_path,
            stack_name="audit",
        )
    assert outside.read_text(encoding="utf-8") == "sentinel"

    lock_path.unlink()
    lock_path.write_text("", encoding="utf-8")
    linked = tmp_path / "linked-lock"
    os.link(lock_path, linked)
    with pytest.raises(RuntimeConfigLockError, match="private regular file"):
        acquire_runtime_config_lock(
            runtime_config_path=runtime_dir / "runtime-config.yaml",
            state_root_dir=tmp_path,
            stack_name="audit",
        )


def test_runtime_config_lock_rejects_access_for_other_users(tmp_path: Path):
    runtime_dir = tmp_path / ".vllm-sr"
    store_dir = runtime_dir / "recipe-store" / "audit"
    store_dir.mkdir(parents=True)
    lock_path = store_dir / LOCK_FILENAME
    lock_path.write_text("", encoding="utf-8")
    lock_path.chmod(0o606)

    with pytest.raises(RuntimeConfigLockError, match="other users"):
        acquire_runtime_config_lock(
            runtime_config_path=runtime_dir / "runtime-config.yaml",
            state_root_dir=tmp_path,
            stack_name="audit",
        )
