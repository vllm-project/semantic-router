import os
import stat
from pathlib import Path

import pytest
import yaml
from cli.commands import runtime_paths
from cli.commands.runtime_paths import (
    _container_runtime_config_path,
    _runtime_config_output_path,
    _write_runtime_config,
)
from cli.runtime_stack import normalize_stack_name


@pytest.mark.parametrize(
    ("raw_stack_name", "expected_filename"),
    [
        (None, "runtime-config.yaml"),
        ("", "runtime-config.yaml"),
        ("vllm-sr", "runtime-config.yaml"),
        (" audit a ", "runtime-config.audit-a.yaml"),
        ("audit/a", "runtime-config.audit-a.yaml"),
        (r"audit\a", "runtime-config.audit-a.yaml"),
        ("../audit/../../escape", "runtime-config.audit-..-..-escape.yaml"),
        ("audit\u4f60\u597d", "runtime-config.audit.yaml"),
    ],
)
def test_runtime_config_path_uses_canonical_stack_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raw_stack_name: str | None,
    expected_filename: str,
):
    if raw_stack_name is None:
        monkeypatch.delenv("VLLM_SR_STACK_NAME", raising=False)
    else:
        monkeypatch.setenv("VLLM_SR_STACK_NAME", raw_stack_name)

    config_path = tmp_path / "config.yaml"
    output_path = _runtime_config_output_path(config_path)

    assert output_path == tmp_path / ".vllm-sr" / expected_filename
    assert output_path.parent.resolve() == (tmp_path / ".vllm-sr").resolve()
    assert _container_runtime_config_path(config_path) == (
        f"/app/.vllm-sr/{expected_filename}"
    )


@pytest.mark.parametrize("raw_stack_name", ["...", "\u4f60\u597d", "---___..."])
def test_normalized_stack_name_rejects_nonempty_invalid_values(
    raw_stack_name: str,
):
    with pytest.raises(ValueError, match="ASCII letter or digit"):
        normalize_stack_name(raw_stack_name)


def test_runtime_config_path_rejects_long_values_before_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("VLLM_SR_STACK_NAME", "a" * 300)

    with pytest.raises(ValueError, match="filesystem limit"):
        _write_runtime_config(tmp_path / "config.yaml", {"version": "v0.3"})

    runtime_dir = tmp_path / ".vllm-sr"
    assert list(runtime_dir.iterdir()) == []


def test_normalization_collisions_share_one_runtime_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    config_path = tmp_path / "config.yaml"

    monkeypatch.setenv("VLLM_SR_STACK_NAME", "audit/a")
    slash_path = _runtime_config_output_path(config_path)
    monkeypatch.setenv("VLLM_SR_STACK_NAME", "audit a")
    space_path = _runtime_config_output_path(config_path)

    assert slash_path == space_path
    assert slash_path.name == "runtime-config.audit-a.yaml"


def test_write_runtime_config_uses_same_directory_atomic_private_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    config_path = tmp_path / "config.yaml"
    runtime_path = _runtime_config_output_path(config_path)
    runtime_path.write_text("version: old\n", encoding="utf-8")
    runtime_path.chmod(0o644)
    old_inode = runtime_path.stat().st_ino

    observed: dict[str, object] = {}
    real_replace = os.replace

    def record_replace(source: str | os.PathLike[str], target: str | os.PathLike[str]):
        source_path = Path(source)
        target_path = Path(target)
        observed["source_parent"] = source_path.parent
        observed["target"] = target_path
        observed["temp_mode"] = stat.S_IMODE(source_path.stat().st_mode)
        observed["old_content"] = target_path.read_text(encoding="utf-8")
        real_replace(source, target)

    monkeypatch.setattr(runtime_paths.os, "replace", record_replace)

    result = _write_runtime_config(config_path, {"version": "v0.3"})

    assert result == runtime_path
    assert observed == {
        "source_parent": runtime_path.parent,
        "target": runtime_path,
        "temp_mode": 0o600,
        "old_content": "version: old\n",
    }
    assert yaml.safe_load(runtime_path.read_text(encoding="utf-8")) == {
        "version": "v0.3"
    }
    assert stat.S_IMODE(runtime_path.stat().st_mode) == 0o600
    assert runtime_path.stat().st_ino != old_inode
    assert list(runtime_path.parent.glob(f".{runtime_path.name}.*.tmp")) == []


def test_write_runtime_config_rejects_file_symlink_without_following_it(
    tmp_path: Path,
):
    config_path = tmp_path / "config.yaml"
    runtime_path = _runtime_config_output_path(config_path)
    outside_path = tmp_path / "outside.yaml"
    outside_path.write_text("sentinel\n", encoding="utf-8")
    runtime_path.symlink_to(outside_path)

    with pytest.raises(ValueError, match="symbolic link"):
        _write_runtime_config(config_path, {"version": "v0.3"})

    assert runtime_path.is_symlink()
    assert outside_path.read_text(encoding="utf-8") == "sentinel\n"


def test_runtime_config_output_rejects_symlinked_owned_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    state_root = tmp_path / "state"
    outside_dir = tmp_path / "outside"
    state_root.mkdir()
    outside_dir.mkdir()
    (state_root / ".vllm-sr").symlink_to(outside_dir, target_is_directory=True)
    monkeypatch.setenv("VLLM_SR_STATE_ROOT_DIR", str(state_root))

    with pytest.raises(ValueError, match="symbolic link"):
        _write_runtime_config(tmp_path / "config.yaml", {"version": "v0.3"})

    assert list(outside_dir.iterdir()) == []


def test_atomic_write_failure_preserves_target_and_removes_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    config_path = tmp_path / "config.yaml"
    runtime_path = _runtime_config_output_path(config_path)
    runtime_path.write_text("version: old\n", encoding="utf-8")

    def fail_dump(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("serialization failed")

    monkeypatch.setattr(runtime_paths.yaml, "dump", fail_dump)

    with pytest.raises(RuntimeError, match="serialization failed"):
        _write_runtime_config(config_path, {"version": "v0.3"})

    assert runtime_path.read_text(encoding="utf-8") == "version: old\n"
    assert list(runtime_path.parent.glob(f".{runtime_path.name}.*.tmp")) == []


def test_replace_failure_preserves_target_and_removes_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    config_path = tmp_path / "config.yaml"
    runtime_path = _runtime_config_output_path(config_path)
    runtime_path.write_text("version: old\n", encoding="utf-8")

    def fail_replace(*_args: object, **_kwargs: object) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(runtime_paths.os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        _write_runtime_config(config_path, {"version": "v0.3"})

    assert runtime_path.read_text(encoding="utf-8") == "version: old\n"
    assert list(runtime_path.parent.glob(f".{runtime_path.name}.*.tmp")) == []
