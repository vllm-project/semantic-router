import os
import stat
from pathlib import Path

import pytest
import yaml
from cli.commands import runtime_paths
from cli.commands.runtime_paths import (
    _compiled_bootstrap_output_path,
    _container_compiled_bootstrap_path,
    _write_compiled_bootstrap,
    assert_user_bootstrap_source,
    materialize_compiled_bootstrap,
)
from cli.runtime_stack import normalize_stack_name


@pytest.mark.parametrize(
    ("raw_stack_name", "expected_filename"),
    [
        (None, "compiled-bootstrap.yaml"),
        ("", "compiled-bootstrap.yaml"),
        ("vllm-sr", "compiled-bootstrap.yaml"),
        (" audit a ", "compiled-bootstrap.audit-a.yaml"),
        ("audit/a", "compiled-bootstrap.audit-a.yaml"),
        (r"audit\a", "compiled-bootstrap.audit-a.yaml"),
        ("../audit/../../escape", "compiled-bootstrap.audit-..-..-escape.yaml"),
        ("audit\u4f60\u597d", "compiled-bootstrap.audit.yaml"),
    ],
)
def test_compiled_bootstrap_path_uses_canonical_stack_identity(
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
    output_path = _compiled_bootstrap_output_path(config_path)

    assert output_path == tmp_path / ".vllm-sr" / expected_filename
    assert output_path.parent.resolve() == (tmp_path / ".vllm-sr").resolve()
    assert _container_compiled_bootstrap_path(config_path) == (
        f"/app/.vllm-sr/{expected_filename}"
    )


@pytest.mark.parametrize("raw_stack_name", ["...", "\u4f60\u597d", "---___..."])
def test_normalized_stack_name_rejects_nonempty_invalid_values(
    raw_stack_name: str,
):
    with pytest.raises(ValueError, match="ASCII letter or digit"):
        normalize_stack_name(raw_stack_name)


def test_compiled_bootstrap_path_rejects_long_values_before_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("VLLM_SR_STACK_NAME", "a" * 300)

    with pytest.raises(ValueError, match="filesystem limit"):
        _write_compiled_bootstrap(tmp_path / "config.yaml", {"version": "v0.3"})

    runtime_dir = tmp_path / ".vllm-sr"
    assert list(runtime_dir.iterdir()) == []


def test_normalization_collisions_share_one_runtime_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    config_path = tmp_path / "config.yaml"

    monkeypatch.setenv("VLLM_SR_STACK_NAME", "audit/a")
    slash_path = _compiled_bootstrap_output_path(config_path)
    monkeypatch.setenv("VLLM_SR_STACK_NAME", "audit a")
    space_path = _compiled_bootstrap_output_path(config_path)

    assert slash_path == space_path
    assert slash_path.name == "compiled-bootstrap.audit-a.yaml"


def test_write_compiled_bootstrap_uses_same_directory_atomic_private_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    config_path = tmp_path / "config.yaml"
    runtime_path = _compiled_bootstrap_output_path(config_path)
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

    result = _write_compiled_bootstrap(config_path, {"version": "v0.3"})

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


def test_write_compiled_bootstrap_rejects_file_symlink_without_following_it(
    tmp_path: Path,
):
    config_path = tmp_path / "config.yaml"
    runtime_path = _compiled_bootstrap_output_path(config_path)
    outside_path = tmp_path / "outside.yaml"
    outside_path.write_text("sentinel\n", encoding="utf-8")
    runtime_path.symlink_to(outside_path)

    with pytest.raises(ValueError, match="symbolic link"):
        _write_compiled_bootstrap(config_path, {"version": "v0.3"})

    assert runtime_path.is_symlink()
    assert outside_path.read_text(encoding="utf-8") == "sentinel\n"


def test_compiled_bootstrap_output_rejects_symlinked_owned_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    state_root = tmp_path / "state"
    outside_dir = tmp_path / "outside"
    state_root.mkdir()
    outside_dir.mkdir()
    (state_root / ".vllm-sr").symlink_to(outside_dir, target_is_directory=True)
    monkeypatch.setenv("VLLM_SR_STATE_ROOT_DIR", str(state_root))

    with pytest.raises(ValueError, match="symbolic link"):
        _write_compiled_bootstrap(tmp_path / "config.yaml", {"version": "v0.3"})

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


def test_private_runtime_state_nested_directory_rejects_symlinked_child(
    tmp_path: Path,
):
    state_root = tmp_path / "state"
    outside_dir = tmp_path / "outside"
    parent = state_root / ".vllm-sr" / "catalog-sources"
    parent.mkdir(parents=True)
    outside_dir.mkdir()
    (parent / "recipe-deadbeef").symlink_to(outside_dir, target_is_directory=True)

    with pytest.raises(ValueError, match="symbolic link"):
        runtime_paths.private_runtime_state_nested_directory(
            state_root, "catalog-sources", "recipe-deadbeef"
        )

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
def test_compiled_bootstrap_output_rejects_directory_owned_by_another_user(
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
        _compiled_bootstrap_output_path(tmp_path / "config.yaml")

    assert stat.S_IMODE(runtime_dir.stat().st_mode) == 0o755


def test_compiled_bootstrap_output_skips_posix_permissions_when_unsupported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    runtime_dir = tmp_path / ".vllm-sr"
    runtime_dir.mkdir()
    runtime_dir.chmod(0o755)
    monkeypatch.setattr(runtime_paths, "_current_posix_user_id", lambda: None)

    def unexpected_chmod(*_args: object, **_kwargs: object) -> None:
        pytest.fail("POSIX chmod must not run on an unsupported platform")

    monkeypatch.setattr(runtime_paths.os, "chmod", unexpected_chmod)

    output_path = _compiled_bootstrap_output_path(tmp_path / "config.yaml")

    assert output_path.parent == runtime_dir


def test_compiled_bootstrap_output_rejects_existing_non_directory(tmp_path: Path):
    runtime_path = tmp_path / ".vllm-sr"
    runtime_path.write_text("not a directory\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must be a real directory"):
        _compiled_bootstrap_output_path(tmp_path / "config.yaml")


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
    runtime_path = _compiled_bootstrap_output_path(config_path)
    runtime_path.write_text("version: old\n", encoding="utf-8")

    def fail_dump(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("serialization failed")

    monkeypatch.setattr(runtime_paths.yaml, "dump", fail_dump)

    with pytest.raises(RuntimeError, match="serialization failed"):
        _write_compiled_bootstrap(config_path, {"version": "v0.3"})

    assert runtime_path.read_text(encoding="utf-8") == "version: old\n"
    assert list(runtime_path.parent.glob(f".{runtime_path.name}.*.tmp")) == []


def test_replace_failure_preserves_target_and_removes_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    config_path = tmp_path / "config.yaml"
    runtime_path = _compiled_bootstrap_output_path(config_path)
    runtime_path.write_text("version: old\n", encoding="utf-8")

    def fail_replace(*_args: object, **_kwargs: object) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(runtime_paths.os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        _write_compiled_bootstrap(config_path, {"version": "v0.3"})

    assert runtime_path.read_text(encoding="utf-8") == "version: old\n"
    assert list(runtime_path.parent.glob(f".{runtime_path.name}.*.tmp")) == []


def test_materialize_compiled_bootstrap_replaces_staging_atomically(
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

    compiled = materialize_compiled_bootstrap(source, effective)

    assert compiled == tmp_path / ".vllm-sr" / "compiled-bootstrap.yaml"
    assert compiled.read_bytes() == effective
    assert [target for _, target in replacements] == [compiled]
    assert all(
        source_path.parent == target.parent for source_path, target in replacements
    )
    assert not list(compiled.parent.glob(".*.tmp"))
    assert not list(compiled.parent.glob("*.provenance.json"))


def test_materialize_compiled_bootstrap_tracks_source_changes(tmp_path: Path):
    source = tmp_path / "config.yaml"
    source.write_text("version: first\n", encoding="utf-8")
    compiled = materialize_compiled_bootstrap(source, b"version: first-effective\n")

    source.write_text("version: second\n", encoding="utf-8")
    refreshed = materialize_compiled_bootstrap(source, b"version: second-effective\n")

    assert refreshed == compiled
    assert compiled.read_bytes() == b"version: second-effective\n"


def test_materialize_compiled_bootstrap_rejects_stale_dashboard_authority(
    tmp_path: Path,
):
    source = tmp_path / "config.yaml"
    source.write_text("version: first\n", encoding="utf-8")
    compiled = materialize_compiled_bootstrap(source, b"version: first-effective\n")
    compiled.write_text("version: dashboard-edit\n", encoding="utf-8")
    source.write_text("version: second\n", encoding="utf-8")

    refreshed = materialize_compiled_bootstrap(source, b"version: second-effective\n")

    assert refreshed == compiled
    assert compiled.read_text(encoding="utf-8") == "version: second-effective\n"


def test_materialize_compiled_bootstrap_uses_custom_host_state_without_path_leak(
    tmp_path: Path,
):
    source = tmp_path / "source" / "config.yaml"
    source.parent.mkdir()
    source.write_text("version: v0.3\n", encoding="utf-8")
    state_root = tmp_path / "host-state"

    compiled = materialize_compiled_bootstrap(
        source,
        source.read_bytes(),
        state_root_dir=state_root,
        stack_name="audit-a",
    )

    assert compiled == (state_root / ".vllm-sr" / "compiled-bootstrap.audit-a.yaml")
    assert not (source.parent / ".vllm-sr").exists()


def test_generated_private_state_cannot_be_selected_as_user_bootstrap(tmp_path: Path):
    generated = tmp_path / ".vllm-sr" / "compiled-bootstrap.yaml"
    generated.parent.mkdir()
    generated.write_text("version: v0.3\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must select an immutable user bootstrap"):
        assert_user_bootstrap_source(generated)
