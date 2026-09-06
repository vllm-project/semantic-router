"""Runtime knowledge-base bootstrap state for CLI local serve."""

from __future__ import annotations

import os
import posixpath
import re
import shutil
import tempfile
from contextlib import suppress
from pathlib import Path

import yaml

from cli.commands.runtime_paths import _runtime_config_output_dir
from cli.utils import get_logger

log = get_logger(__name__)

KB_RUNTIME_ROOT = "knowledge_bases"
KB_BOOTSTRAP_STATE_FILE = ".bootstrap-state.yaml"
MIN_KB_SOURCE_PATH_PARTS = 2
_SAFE_KB_PATH_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _runtime_kb_state_dir(config_path: Path) -> Path:
    kb_dir = _runtime_config_output_dir(config_path) / KB_RUNTIME_ROOT
    if kb_dir.is_symlink():
        raise ValueError("Runtime knowledge-base directory must not be a symlink")
    kb_dir.mkdir(parents=True, exist_ok=True)
    if kb_dir.is_symlink() or not kb_dir.is_dir():
        raise ValueError("Runtime knowledge-base path must be an owned directory")
    return kb_dir


def _clean_kb_source_path(value: str) -> str:
    cleaned = posixpath.normpath(str(value or "").strip().replace("\\", "/"))
    if cleaned == ".":
        return ""
    return cleaned.rstrip("/")


def _runtime_kb_bootstrap_state_path(config_path: Path) -> Path:
    return _runtime_kb_state_dir(config_path) / KB_BOOTSTRAP_STATE_FILE


def _load_runtime_kb_bootstrap_state(config_path: Path) -> set[str]:
    state_path = _runtime_kb_bootstrap_state_path(config_path)
    if state_path.is_symlink():
        raise ValueError("Runtime KB bootstrap state must be a regular file")
    if not state_path.exists():
        return set()
    if not state_path.is_file():
        raise ValueError("Runtime KB bootstrap state must be a regular file")

    try:
        data = yaml.safe_load(state_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        log.warning(
            "Ignoring invalid runtime KB bootstrap state %s: %s", state_path, exc
        )
        return set()
    processed = data.get("processed")
    if not isinstance(processed, list):
        return set()

    return {
        cleaned
        for item in processed
        if isinstance(item, str)
        if (cleaned := _clean_kb_source_path(item))
    }


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except OSError:
        with suppress(FileNotFoundError):
            os.unlink(temp_name)
        raise


def _write_runtime_kb_bootstrap_state(
    config_path: Path, processed_runtime_paths: set[str]
) -> None:
    state_path = _runtime_kb_bootstrap_state_path(config_path)
    _atomic_write_text(
        state_path,
        yaml.safe_dump(
            {
                "version": 1,
                "processed": sorted(processed_runtime_paths),
            },
            sort_keys=False,
        ),
    )


def _safe_relative_kb_source_path(source_path: str) -> Path:
    raw = str(source_path or "").strip()
    if not raw or "\\" in raw:
        raise ValueError("Knowledge-base source.path must be a relative POSIX path")
    source_root = Path(raw)
    parts = raw.rstrip("/").split("/")
    if source_root.is_absolute() or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(
            "Knowledge-base source.path must not be absolute or contain dot segments"
        )
    return Path(*parts)


def _candidate_kb_source_roots(
    config_path: Path, source_path: str
) -> list[tuple[Path, Path]]:
    source_root = _safe_relative_kb_source_path(source_path)
    candidates: list[tuple[Path, Path]] = [
        (config_path.parent, config_path.parent / source_root)
    ]
    for ancestor in Path(__file__).resolve().parents:
        config_root = ancestor / "config"
        if config_root.is_dir():
            candidates.append((config_root, config_root / source_root))
    return candidates


def _assert_no_symlink_components(root: Path, candidate: Path) -> None:
    relative = candidate.relative_to(root)
    current = root
    for component in relative.parts:
        current /= component
        if current.is_symlink():
            raise ValueError(
                "Knowledge-base source paths must not contain symbolic links"
            )


def _resolve_kb_source_root(config_path: Path, source_path: str) -> Path | None:
    for root, candidate in _candidate_kb_source_roots(config_path, source_path):
        if not candidate.exists():
            continue
        if not candidate.is_dir():
            raise ValueError("Knowledge-base source.path must resolve to a directory")
        resolved_root = root.resolve(strict=True)
        resolved_candidate = candidate.resolve(strict=True)
        try:
            resolved_candidate.relative_to(resolved_root)
        except ValueError as exc:
            raise ValueError(
                "Knowledge-base source.path escapes its allowed source directory"
            ) from exc
        _assert_no_symlink_components(root, candidate)
        return resolved_candidate
    return None


def _configured_knowledge_bases(config: dict[str, object]) -> list[dict[str, object]]:
    global_config = config.get("global")
    if not isinstance(global_config, dict):
        return []

    model_catalog = global_config.get("model_catalog")
    if not isinstance(model_catalog, dict):
        return []

    kb_configs = model_catalog.get("kbs")
    if not isinstance(kb_configs, list):
        return []

    return [kb_config for kb_config in kb_configs if isinstance(kb_config, dict)]


def _kb_source_spec(
    kb_config: dict[str, object],
) -> tuple[str, dict[str, object], str] | None:
    source = kb_config.get("source")
    if not isinstance(source, dict):
        return None

    source_path = str(source.get("path") or "").strip()
    kb_name = str(kb_config.get("name") or "").strip()
    if not source_path or not kb_name:
        return None

    return kb_name, source, source_path


def _runtime_kb_relative_path(source_path: str, kb_name: str) -> str:
    _safe_relative_kb_source_path(source_path)
    cleaned = _clean_kb_source_path(source_path)
    runtime_root = _clean_kb_source_path(KB_RUNTIME_ROOT)

    if cleaned == runtime_root:
        return f"{runtime_root}/{kb_name}"
    if cleaned.startswith(f"{runtime_root}/"):
        return cleaned

    leaf_name = Path(cleaned).name
    if leaf_name in {"", ".", runtime_root}:
        leaf_name = kb_name
    return f"{runtime_root}/{leaf_name}"


def _runtime_kb_target_root(config_path: Path, runtime_relative_path: str) -> Path:
    runtime_root = _runtime_kb_state_dir(config_path)
    target = _runtime_config_output_dir(config_path) / Path(runtime_relative_path)
    try:
        target.relative_to(runtime_root)
    except ValueError as exc:
        raise ValueError(
            "Runtime knowledge-base target escapes the owned runtime directory"
        ) from exc
    if target == runtime_root:
        raise ValueError("Runtime knowledge-base target must name a child directory")
    current = runtime_root
    for component in target.relative_to(runtime_root).parts:
        current /= component
        if current.is_symlink():
            raise ValueError(
                "Runtime knowledge-base target paths must not contain symbolic links"
            )
    return target


def _assert_copy_is_non_recursive(source_root: Path, runtime_target: Path) -> None:
    resolved_source = source_root.resolve(strict=True)
    resolved_target = runtime_target.absolute()
    try:
        resolved_target.relative_to(resolved_source)
    except ValueError:
        return
    raise ValueError(
        "Knowledge-base source directory must not contain its runtime target"
    )


def _assert_source_tree_has_no_symlinks(source_root: Path) -> None:
    for root, directories, filenames in os.walk(source_root, followlinks=False):
        root_path = Path(root)
        for entry in [*directories, *filenames]:
            if (root_path / entry).is_symlink():
                raise ValueError(
                    "Knowledge-base source trees must not contain symbolic links"
                )


def _validate_package_kb_paths(config: dict[str, object]) -> bool:
    """Validate packaged KB references without reading or changing runtime state."""

    kb_configs = _configured_knowledge_bases(config)
    for kb_config in kb_configs:
        kb_source_spec = _kb_source_spec(kb_config)
        if kb_source_spec is None:
            continue
        _kb_name, _source, source_path = kb_source_spec
        relative = _safe_relative_kb_source_path(source_path)
        parts = relative.parts
        if (
            len(parts) < MIN_KB_SOURCE_PATH_PARTS
            or parts[0] != KB_RUNTIME_ROOT
            or any(not _SAFE_KB_PATH_COMPONENT.fullmatch(part) for part in parts[1:])
        ):
            raise ValueError(
                "Packaged knowledge-base source.path must use the canonical "
                "knowledge_bases/<name>/ form"
            )
    return bool(kb_configs)


def _sync_runtime_kb_store(
    config: dict[str, object],
    config_path: Path,
) -> tuple[bool, bool]:
    kb_configs = _configured_knowledge_bases(config)
    if not kb_configs:
        return False, False

    changed = False
    state_changed = False
    bootstrapped = _load_runtime_kb_bootstrap_state(config_path)

    for kb_config in kb_configs:
        kb_source_spec = _kb_source_spec(kb_config)
        if kb_source_spec is None:
            continue

        kb_name, source, source_path = kb_source_spec
        runtime_relative_path = _runtime_kb_relative_path(source_path, kb_name)
        runtime_target = _runtime_kb_target_root(config_path, runtime_relative_path)

        normalized_source_path = f"{runtime_relative_path}/"
        if source.get("path") != normalized_source_path:
            source["path"] = normalized_source_path
            changed = True

        if runtime_target.exists():
            if not runtime_target.is_dir():
                raise ValueError(
                    "Runtime knowledge-base target must be a real directory"
                )
            if runtime_relative_path not in bootstrapped:
                bootstrapped.add(runtime_relative_path)
                state_changed = True
            continue

        if runtime_relative_path in bootstrapped:
            continue

        resolved_source_root = _resolve_kb_source_root(config_path, source_path)
        if resolved_source_root is None:
            log.warning(
                "Runtime KB bootstrap: could not resolve KB source.path=%s for %s",
                source_path,
                kb_name,
            )
            continue

        _assert_copy_is_non_recursive(resolved_source_root, runtime_target)
        _assert_source_tree_has_no_symlinks(resolved_source_root)
        runtime_target.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copytree(resolved_source_root, runtime_target)
            bootstrapped.add(runtime_relative_path)
            state_changed = True
        except OSError as exc:
            log.warning(
                "Runtime KB bootstrap: failed to import %s from %s to %s: %s",
                kb_name,
                resolved_source_root,
                runtime_target,
                exc,
            )

    if state_changed:
        _write_runtime_kb_bootstrap_state(config_path, bootstrapped)

    return True, changed or state_changed
