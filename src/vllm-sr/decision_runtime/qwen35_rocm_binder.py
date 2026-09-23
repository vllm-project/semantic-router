"""Bind a released Qwen3.5 ROCm normalization profile to installed FLA.

FLA 0.5.2's ``STRICT`` cache mode still autotunes on a missing key.  This
module verifies the installed sources and the release's data-only launch table,
then guards the one profiled kernel before it can reach that fallback.  One
profile may be active in a process; loading a different profile requires a new
process.  No Python from the model artifact is imported or executed.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import sys
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .qwen35_torch import (
    QwenRocmProfileBinding,
    ValidatedQwenRocmProfile,
    _validate_kernel_profile,
)

_FLA_SOURCE_PATHS = ("modules/l2norm.py", "ops/utils/cache.py")
_OTHER_L2NORM_KERNELS = (
    "l2norm_fwd_kernel1",
    "l2norm_bwd_kernel",
    "l2norm_bwd_kernel1",
)
_CONFIG_FIELDS = frozenset(
    {
        "kwargs",
        "num_warps",
        "num_stages",
        "num_ctas",
        "maxnreg",
        "pre_hook",
        "ir_override",
    }
)
_BIND_LOCK = threading.RLock()
_ACTIVE: _ActiveBinding | None = None


class QwenRocmBindingError(RuntimeError):
    """The published profile cannot be enforced by this process's FLA."""


@dataclass(slots=True)
class _ActiveBinding:
    profile: ValidatedQwenRocmProfile
    process_id: int
    package_root: Path
    kernel: Any
    cache_module: Any
    guarded_run: Any
    rejected_runs: tuple[tuple[Any, Any], ...]
    receipt: QwenRocmProfileBinding


class StrictQwenRocmProfileBinder:
    """Install a hash-bound strict FLA launch guard before model loading."""

    def bind(self, profile: ValidatedQwenRocmProfile) -> QwenRocmProfileBinding:
        global _ACTIVE  # noqa: PLW0603 - one process-wide FLA profile guard

        with _BIND_LOCK:
            entries = _profile_entries(profile)
            if _ACTIVE is not None:
                _verify_active(_ACTIVE, profile)
                return _ACTIVE.receipt

            if any(name == "fla" or name.startswith("fla.") for name in sys.modules):
                raise QwenRocmBindingError(
                    "FLA was imported before the strict Qwen profile was bound"
                )
            package_root = _fla_package_root()
            _verify_sources(package_root, profile)
            expected_runtime = dict(profile.runtime)
            torch = _import_module("torch")
            triton = _import_module("triton")
            if (
                str(torch.__version__) != expected_runtime["torch"]
                or torch.version.hip != expected_runtime["hip"]
                or triton.__version__ != expected_runtime["triton"]
            ):
                raise QwenRocmBindingError(
                    "Torch, HIP, or Triton version differs from profile"
                )
            if any(name == "fla" or name.startswith("fla.") for name in sys.modules):
                raise QwenRocmBindingError(
                    "FLA was imported before the strict Qwen profile was bound"
                )

            config_dir = str(profile.kernel_config_path.parent.resolve(strict=True))
            _set_fla_environment(config_dir)
            fla = _import_module("fla")
            l2norm = _import_module("fla.modules.l2norm")
            cache_module = _import_module("fla.ops.utils.cache")
            if (
                fla.__version__ != expected_runtime["fla"]
                or Path(fla.__file__).resolve(strict=True).parent != package_root
                or Path(l2norm.__file__).resolve(strict=True)
                != package_root / _FLA_SOURCE_PATHS[0]
                or Path(cache_module.__file__).resolve(strict=True)
                != package_root / _FLA_SOURCE_PATHS[1]
            ):
                raise QwenRocmBindingError(
                    "installed FLA identity differs from profile"
                )
            _verify_sources(package_root, profile)
            if cache_module.FLA_CACHE_MODE is not cache_module.FlaCacheMode.STRICT:
                raise QwenRocmBindingError("FLA strict cache mode was not activated")

            kernel = l2norm.l2norm_fwd_kernel
            if (
                not isinstance(kernel, cache_module.CachedAutotuner)
                or kernel.kernel_name != "l2norm_fwd_kernel"
                or kernel.keys != ["D", "NB"]
                or not isinstance(kernel.cache, dict)
                or kernel.cache
            ):
                raise QwenRocmBindingError(
                    "FLA l2norm kernel is not a fresh supported autotuner"
                )
            other_kernels = tuple(
                (name, getattr(l2norm, name)) for name in _OTHER_L2NORM_KERNELS
            )
            if any(
                not callable(getattr(other, "run", None)) for _, other in other_kernels
            ):
                raise QwenRocmBindingError("FLA l2norm kernel surface changed")

            guard = _StrictRunGuard(
                profile=profile,
                kernel=kernel,
                cache_module=cache_module,
                original_run=kernel.run,
                entries=entries,
            )
            rejected = tuple(
                (other, _reject_unprofiled_kernel(name))
                for name, other in other_kernels
            )
            original_runs = [(kernel, kernel.run)] + [
                (other, other.run) for _, other in other_kernels
            ]
            try:
                kernel.run = guard
                for other, reject in rejected:
                    other.run = reject
            except Exception:
                for target, original in original_runs:
                    target.run = original
                raise

            receipt = _receipt(profile)
            _ACTIVE = _ActiveBinding(
                profile=profile,
                process_id=os.getpid(),
                package_root=package_root,
                kernel=kernel,
                cache_module=cache_module,
                guarded_run=guard,
                rejected_runs=rejected,
                receipt=receipt,
            )
            return receipt


def create_qwen_rocm_profile_binder() -> StrictQwenRocmProfileBinder:
    """Return the owned binder without importing Torch, Triton, or FLA."""

    return StrictQwenRocmProfileBinder()


def _import_module(name: str) -> Any:
    return importlib.import_module(name)


def _profile_entries(
    profile: ValidatedQwenRocmProfile,
) -> dict[tuple[Any, ...], dict[str, Any]]:
    if _sha256_file(profile.profile_path) != profile.profile_sha256:
        raise QwenRocmBindingError("Qwen ROCm profile changed before binding")
    if _sha256_file(profile.kernel_config_path) != profile.kernel_config_sha256:
        raise QwenRocmBindingError("Qwen ROCm kernel config changed before binding")
    try:
        data = json.loads(profile.kernel_config_path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QwenRocmBindingError("Qwen ROCm kernel config is unreadable") from error
    if not isinstance(data, dict):
        raise QwenRocmBindingError("Qwen ROCm kernel config is invalid")
    _validate_kernel_profile(
        data,
        minimum=profile.normalization_blocks_min,
        maximum=profile.normalization_blocks_max,
    )
    if (
        data.get("kernel_name") != "l2norm_fwd_kernel"
        or data.get("triton_version") != dict(profile.runtime)["triton"]
        or data.get("default_config") is not None
        or set(data)
        not in (
            {"kernel_name", "triton_version", "autotune_entries"},
            {"kernel_name", "triton_version", "autotune_entries", "default_config"},
        )
    ):
        raise QwenRocmBindingError("Qwen ROCm kernel config has unsupported metadata")
    entries: dict[tuple[Any, ...], dict[str, Any]] = {}
    for item in data["autotune_entries"].values():
        key = tuple(item["autotune_key"])
        config = item["config"]
        if (
            key in entries
            or set(config) != _CONFIG_FIELDS
            or not isinstance(config["kwargs"], dict)
            or set(config["kwargs"]) != {"BT"}
            or type(config["kwargs"]["BT"]) is not int
            or any(
                type(config[field]) is not int
                for field in ("num_warps", "num_stages", "num_ctas")
            )
        ):
            raise QwenRocmBindingError("Qwen ROCm kernel config entry is ambiguous")
        entries[key] = config
    return entries


def _fla_package_root() -> Path:
    specification = importlib.util.find_spec("fla")
    if specification is None or specification.origin is None:
        raise QwenRocmBindingError("installed FLA package is unavailable")
    origin = Path(specification.origin).resolve(strict=True)
    if origin.name != "__init__.py":
        raise QwenRocmBindingError("installed FLA package has unsupported layout")
    return origin.parent


def _verify_sources(root: Path, profile: ValidatedQwenRocmProfile) -> None:
    expected = dict(profile.fla_source_sha256)
    if set(expected) != set(_FLA_SOURCE_PATHS):
        raise QwenRocmBindingError("FLA source identity is incomplete")
    for relative in _FLA_SOURCE_PATHS:
        source = (root / relative).resolve(strict=True)
        if (
            not source.is_relative_to(root)
            or _sha256_file(source) != expected[relative]
        ):
            raise QwenRocmBindingError(f"installed FLA source differs: {relative}")


def _set_fla_environment(config_dir: str) -> None:
    required = {"FLA_CACHE_MODE": "strict", "FLA_CONFIG_DIR": config_dir}
    for name, value in required.items():
        if os.environ.get(name) not in (None, value):
            raise QwenRocmBindingError(
                f"conflicting {name} prevents strict FLA binding"
            )
    os.environ.update(required)


def _verify_active(active: _ActiveBinding, profile: ValidatedQwenRocmProfile) -> None:
    if (
        active.process_id != os.getpid()
        or active.profile != profile
        or active.kernel.run is not active.guarded_run
        or any(other.run is not reject for other, reject in active.rejected_runs)
        or active.cache_module.FLA_CACHE_MODE
        is not active.cache_module.FlaCacheMode.STRICT
        or os.environ.get("FLA_CACHE_MODE") != "strict"
        or os.environ.get("FLA_CONFIG_DIR")
        != str(profile.kernel_config_path.parent.resolve(strict=True))
    ):
        raise QwenRocmBindingError("a different or modified FLA profile is active")
    _verify_sources(active.package_root, profile)


def _receipt(profile: ValidatedQwenRocmProfile) -> QwenRocmProfileBinding:
    return QwenRocmProfileBinding(
        profile_sha256=profile.profile_sha256,
        kernel_config_sha256=profile.kernel_config_sha256,
        guard_contract_sha256=profile.guard_contract_sha256,
        runtime=profile.runtime,
        fla_source_sha256=profile.fla_source_sha256,
        physical_batch_size_min=profile.physical_batch_size_min,
        physical_batch_size_max=profile.physical_batch_size_max,
        normalization_blocks_min=profile.normalization_blocks_min,
        normalization_blocks_max=profile.normalization_blocks_max,
        strict=True,
        source_hashes_verified=True,
        strict_cache_enforced=True,
        unknown_key_guard_enforced=True,
    )


def _reject_unprofiled_kernel(name: str):
    def reject(*args: Any, **kwargs: Any) -> None:
        raise QwenRocmBindingError(f"unprofiled FLA normalization kernel: {name}")

    return reject


class _StrictRunGuard:
    def __init__(
        self,
        *,
        profile: ValidatedQwenRocmProfile,
        kernel: Any,
        cache_module: Any,
        original_run: Any,
        entries: dict[tuple[Any, ...], dict[str, Any]],
    ) -> None:
        self.profile = profile
        self.process_id = os.getpid()
        self.kernel = kernel
        self.cache_module = cache_module
        self.original_run = original_run
        self.entries = entries
        self.config_dir = str(profile.kernel_config_path.parent.resolve(strict=True))
        self.config_file_identity = _file_identity(profile.kernel_config_path)
        if _sha256_file(profile.kernel_config_path) != profile.kernel_config_sha256:
            raise QwenRocmBindingError("Qwen ROCm kernel config changed before binding")
        if _file_identity(profile.kernel_config_path) != self.config_file_identity:
            raise QwenRocmBindingError("Qwen ROCm kernel config changed during binding")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        cache = self.cache_module
        if (
            os.getpid() != self.process_id
            or cache.FLA_CACHE_MODE is not cache.FlaCacheMode.STRICT
            or os.environ.get("FLA_CACHE_MODE") != "strict"
            or os.environ.get("FLA_CONFIG_DIR") != self.config_dir
        ):
            raise QwenRocmBindingError(
                "FLA strict configuration changed during inference"
            )
        identity = _file_identity(self.profile.kernel_config_path)
        if identity != self.config_file_identity:
            if (
                _sha256_file(self.profile.kernel_config_path)
                != self.profile.kernel_config_sha256
                or _file_identity(self.profile.kernel_config_path) != identity
            ):
                raise QwenRocmBindingError(
                    "FLA strict configuration changed during inference"
                )
            self.config_file_identity = identity

        key = cache.AutotuneKey.build(
            self.kernel.arg_names, self.kernel.keys, args, kwargs
        )
        selected = self.entries.get(key.autotune_key)
        if selected is None:
            raise QwenRocmBindingError("unprofiled FLA l2norm autotune key")
        prior = self.kernel.cache.get(key.autotune_key)
        if prior is None:
            # In FLA 0.5.2, CachedAutotuner.run reads its config only when
            # this key is absent from the in-process cache. Populate it once
            # from the already verified launch table, before run can autotune.
            loaded = cache.load_cached_config(self.kernel.kernel_name, key)
            if _config_fields(loaded) != selected:
                raise QwenRocmBindingError(
                    "FLA strict cache did not return the profiled config"
                )
            self.kernel.maybe_load_cached_config(key)
        elif _config_fields(prior) != selected:
            raise QwenRocmBindingError("conflicting FLA in-process autotune cache")
        if _config_fields(self.kernel.cache.get(key.autotune_key)) != selected:
            raise QwenRocmBindingError("FLA did not install the exact profiled config")
        result = self.original_run(*args, **kwargs)
        if _config_fields(self.kernel.cache.get(key.autotune_key)) != selected:
            raise QwenRocmBindingError(
                "FLA changed the profiled config during dispatch"
            )
        return result


def _config_fields(config: Any) -> dict[str, Any] | None:
    if isinstance(config, Mapping):
        if set(config) != _CONFIG_FIELDS:
            return None
        return {field: config[field] for field in _CONFIG_FIELDS}
    if config is None:
        return None
    return {field: getattr(config, field, object()) for field in _CONFIG_FIELDS}


def _sha256_file(path: Path) -> str:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError as error:
        raise QwenRocmBindingError(
            "profile or FLA source file is unreadable"
        ) from error


def _file_identity(path: Path) -> tuple[int, int, int, int, int]:
    """Detect edits and replacements without hashing the table on every launch."""

    try:
        metadata = path.stat(follow_symlinks=False)
    except OSError as error:
        raise QwenRocmBindingError(
            "profile or FLA source file is unreadable"
        ) from error
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )
