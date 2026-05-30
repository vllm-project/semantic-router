"""Runtime knowledge-base bootstrap state for CLI local serve."""

from __future__ import annotations

import os
import posixpath
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


def _runtime_kb_state_dir(config_path: Path) -> Path:
    kb_dir = _runtime_config_output_dir(config_path) / KB_RUNTIME_ROOT
    kb_dir.mkdir(parents=True, exist_ok=True)
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
    if not state_path.exists():
        return set()

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


def _candidate_kb_source_roots(config_path: Path, source_path: str) -> list[Path]:
    source_root = Path(source_path)
    if source_root.is_absolute():
        return [source_root]

    candidates: list[Path] = [config_path.parent / source_root]
    for ancestor in Path(__file__).resolve().parents:
        config_root = ancestor / "config"
        if config_root.is_dir():
            candidates.append(config_root / source_root)
    return candidates


def _resolve_kb_source_root(config_path: Path, source_path: str) -> Path | None:
    for candidate in _candidate_kb_source_roots(config_path, source_path):
        if candidate.exists():
            return candidate
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
    return _runtime_config_output_dir(config_path) / Path(runtime_relative_path)


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
