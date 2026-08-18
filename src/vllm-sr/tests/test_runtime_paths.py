import os
import stat
from pathlib import Path

import pytest
import yaml
from cli.commands import runtime_paths
from cli.commands.runtime_paths import (
    _container_runtime_config_path,
    _runtime_config_output_path,
    _runtime_config_provenance_path,
    _write_runtime_config,
    materialize_runtime_config,
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


def test_private_runtime_state_subdirectory_rejects_symlinked_child(
    tmp_path: Path,
):
    state_root = tmp_path / "state"
    outside_dir = tmp_path / "outside"
    runtime_dir = state_root / ".vllm-sr"
    runtime_dir.mkdir(parents=True)
    outside_dir.mkdir()
    (runtime_dir / "catalog-sources").symlink_to(outside_dir, target_is_directory=True)

    with pytest.raises(ValueError, match="symbolic link"):
        runtime_paths.private_runtime_state_subdirectory(state_root, "catalog-sources")

    assert list(outside_dir.iterdir()) == []


@pytest.mark.skipif(os.name != "posix", reason="POSIX permissions only")
def test_private_runtime_state_subdirectory_hardens_existing_owned_directories(
    tmp_path: Path,
):
    state_root = tmp_path / "state"
    runtime_dir = state_root / ".vllm-sr"
    child_dir = runtime_dir / "catalog-sources"
    child_dir.mkdir(parents=True)
    runtime_dir.chmod(0o755)
    child_dir.chmod(0o777)

    result = runtime_paths.private_runtime_state_subdirectory(
        state_root, "catalog-sources"
    )

    assert result == child_dir
    assert stat.S_IMODE(runtime_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(child_dir.stat().st_mode) == 0o700


@pytest.mark.skipif(os.name != "posix", reason="POSIX ownership only")
def test_runtime_config_output_rejects_directory_owned_by_another_user(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    runtime_dir = tmp_path / ".vllm-sr"
    runtime_dir.mkdir()
    runtime_dir.chmod(0o755)
    actual_owner = runtime_dir.stat().st_uid
    monkeypatch.setattr(
        runtime_paths, "_current_posix_user_id", lambda: actual_owner + 1
    )

    with pytest.raises(ValueError, match="owned by the current user"):
        _runtime_config_output_path(tmp_path / "config.yaml")

    assert stat.S_IMODE(runtime_dir.stat().st_mode) == 0o755


def test_runtime_config_output_skips_posix_permissions_when_unsupported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    runtime_dir = tmp_path / ".vllm-sr"
    runtime_dir.mkdir()
    runtime_dir.chmod(0o755)
    monkeypatch.setattr(runtime_paths, "_current_posix_user_id", lambda: None)

    def unexpected_chmod(*_args: object, **_kwargs: object) -> None:
        pytest.fail("POSIX chmod must not run on an unsupported platform")

    monkeypatch.setattr(runtime_paths.os, "chmod", unexpected_chmod)

    output_path = _runtime_config_output_path(tmp_path / "config.yaml")

    assert output_path.parent == runtime_dir


def test_runtime_config_output_rejects_existing_non_directory(tmp_path: Path):
    runtime_path = tmp_path / ".vllm-sr"
    runtime_path.write_text("not a directory\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must be a real directory"):
        _runtime_config_output_path(tmp_path / "config.yaml")


def test_private_runtime_state_subdirectory_rejects_existing_file(
    tmp_path: Path,
):
    runtime_dir = tmp_path / ".vllm-sr"
    runtime_dir.mkdir()
    child_path = runtime_dir / "catalog-sources"
    child_path.write_text("not a directory\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must be a real directory"):
        runtime_paths.private_runtime_state_subdirectory(tmp_path, "catalog-sources")


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


def test_materialize_runtime_config_creates_active_and_provenance_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source = tmp_path / "config.yaml"
    source.write_text("version: source\n", encoding="utf-8")
    effective = b"version: effective\n"
    replacements: list[tuple[Path, Path]] = []
    real_replace = os.replace

    def record_replace(source_path, target_path):
        replacements.append((Path(source_path), Path(target_path)))
        real_replace(source_path, target_path)

    monkeypatch.setattr(runtime_paths.os, "replace", record_replace)

    active = materialize_runtime_config(source, effective)
    provenance = _runtime_config_provenance_path(active)

    assert active == tmp_path / ".vllm-sr" / "runtime-config.yaml"
    assert active.read_bytes() == effective
    assert provenance.is_file()
    assert [target for _, target in replacements] == [active, provenance]
    assert all(
        source_path.parent == target.parent for source_path, target in replacements
    )
    assert not list(active.parent.glob(".*.tmp"))


def test_materialize_refreshes_unchanged_active_when_source_changes(tmp_path: Path):
    source = tmp_path / "config.yaml"
    source.write_text("version: first\n", encoding="utf-8")
    active = materialize_runtime_config(source, b"version: first-effective\n")

    source.write_text("version: second\n", encoding="utf-8")
    refreshed = materialize_runtime_config(source, b"version: second-effective\n")

    assert refreshed == active
    assert active.read_bytes() == b"version: second-effective\n"


def test_materialize_preserves_dashboard_change_when_source_changes(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    source = tmp_path / "config.yaml"
    source.write_text("version: first\n", encoding="utf-8")
    active = materialize_runtime_config(source, b"version: first-effective\n")
    active.write_text("version: dashboard-edit\n", encoding="utf-8")
    source.write_text("version: second\n", encoding="utf-8")

    preserved = materialize_runtime_config(source, b"version: second-effective\n")

    assert preserved == active
    assert active.read_text(encoding="utf-8") == "version: dashboard-edit\n"
    assert "Preserving Dashboard or package changes" in caplog.text


def test_materialize_uses_custom_host_state_without_container_path_leak(
    tmp_path: Path,
):
    source = tmp_path / "source" / "config.yaml"
    source.parent.mkdir()
    source.write_text("version: v0.3\n", encoding="utf-8")
    state_root = tmp_path / "host-state"

    active = materialize_runtime_config(
        source,
        source.read_bytes(),
        state_root_dir=state_root,
        stack_name="audit-a",
    )

    assert active == state_root / ".vllm-sr" / "runtime-config.audit-a.yaml"
    assert not (source.parent / ".vllm-sr").exists()
