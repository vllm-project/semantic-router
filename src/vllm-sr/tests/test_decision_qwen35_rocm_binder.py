"""Dependency-free contract tests for the owned Qwen ROCm FLA binder."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from decision_runtime import qwen35_rocm_binder as binder  # noqa: E402
from decision_runtime.qwen35_torch import ValidatedQwenRocmProfile  # noqa: E402


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _profile(tmp_path: Path) -> tuple[ValidatedQwenRocmProfile, Path, dict]:
    package = tmp_path / "fla"
    (package / "modules").mkdir(parents=True)
    (package / "ops" / "utils").mkdir(parents=True)
    (package / "__init__.py").write_bytes(b"fake FLA package")
    sources = {
        "modules/l2norm.py": b"verified l2norm source",
        "ops/utils/cache.py": b"verified cache source",
    }
    for relative, content in sources.items():
        (package / relative).write_bytes(content)

    profile_path = tmp_path / "profile.json"
    profile_bytes = b"already validated release profile"
    profile_path.write_bytes(profile_bytes)
    kernel_path = tmp_path / "l2norm_fwd_kernel.json"
    entries = {}
    configs = {}
    for nb in (1, 2):
        key = [128, nb, "torch.bfloat16", "torch.bfloat16", "torch.float32"]
        config = {
            "kwargs": {"BT": 8},
            "num_warps": 1,
            "num_stages": 3,
            "num_ctas": 1,
            "maxnreg": None,
            "pre_hook": None,
            "ir_override": None,
        }
        digest = hashlib.md5(
            json.dumps(key, sort_keys=True, separators=(",", ":")).encode(),
            usedforsecurity=False,
        ).hexdigest()
        entries[digest] = {"autotune_key": key, "config": config}
        configs[tuple(key)] = config
    kernel_bytes = json.dumps(
        {
            "kernel_name": "l2norm_fwd_kernel",
            "triton_version": "3.7.1",
            "autotune_entries": entries,
        }
    ).encode()
    kernel_path.write_bytes(kernel_bytes)
    profile = ValidatedQwenRocmProfile(
        profile_path=profile_path,
        kernel_config_path=kernel_path,
        profile_sha256=_sha256(profile_bytes),
        kernel_config_sha256=_sha256(kernel_bytes),
        guard_contract_sha256="a" * 64,
        runtime=tuple(
            sorted(
                {
                    "torch": "2.12.0+git6bbd260",
                    "hip": "7.2.53211",
                    "triton": "3.7.1",
                    "fla": "0.5.2",
                    "gpu_arch": "gfx942",
                }.items()
            )
        ),
        fla_source_sha256=tuple(
            (relative, _sha256(content)) for relative, content in sources.items()
        ),
        physical_batch_size_min=1,
        physical_batch_size_max=8,
        padded_tokens_max=16_384,
        normalization_blocks_min=1,
        normalization_blocks_max=2,
        normalization_blocks_formula="ceil(B*padded_tokens*16/65536)",
        unknown_key_policy="raise before kernel/autotune",
    )
    return profile, package, configs


def _fake_fla_modules(package: Path, configs: dict):
    calls = []

    class FakeKey:
        @staticmethod
        def build(arg_names, key_names, args, kwargs):
            return SimpleNamespace(autotune_key=kwargs["key"])

    class FakeKernel:
        def __init__(self):
            self.kernel_name = "l2norm_fwd_kernel"
            self.keys = ["D", "NB"]
            self.arg_names = ["x", "y", "rstd", "D", "NB"]
            self.cache = {}

        def run(self, *args, **kwargs):
            calls.append(kwargs["key"])
            return "launched"

        def maybe_load_cached_config(self, key):
            self.cache[key.autotune_key] = SimpleNamespace(**configs[key.autotune_key])

    kernel = FakeKernel()
    strict = object()
    cache = SimpleNamespace(
        __file__=str(package / "ops" / "utils" / "cache.py"),
        FlaCacheMode=SimpleNamespace(STRICT=strict),
        FLA_CACHE_MODE=strict,
        CachedAutotuner=FakeKernel,
        AutotuneKey=FakeKey,
        load_cached_config=lambda name, key: configs.get(key.autotune_key),
    )
    l2norm = SimpleNamespace(
        __file__=str(package / "modules" / "l2norm.py"),
        l2norm_fwd_kernel=kernel,
        **{
            name: SimpleNamespace(run=lambda *args, **kwargs: "unprofiled")
            for name in binder._OTHER_L2NORM_KERNELS
        },
    )
    modules = {
        "torch": SimpleNamespace(
            __version__="2.12.0+git6bbd260",
            version=SimpleNamespace(hip="7.2.53211"),
        ),
        "triton": SimpleNamespace(__version__="3.7.1"),
        "fla": SimpleNamespace(
            __version__="0.5.2", __file__=str(package / "__init__.py")
        ),
        "fla.modules.l2norm": l2norm,
        "fla.ops.utils.cache": cache,
    }
    return modules, kernel, l2norm, calls


def test_strict_binding_guards_exact_keys_and_one_profile_per_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile, package, configs = _profile(tmp_path)
    modules, kernel, l2norm, calls = _fake_fla_modules(package, configs)
    monkeypatch.setattr(binder, "_ACTIVE", None)
    monkeypatch.setattr(binder, "sys", SimpleNamespace(modules={}))
    monkeypatch.setattr(binder, "_fla_package_root", lambda: package)
    monkeypatch.setattr(binder, "_import_module", modules.__getitem__)
    monkeypatch.delenv("FLA_CACHE_MODE", raising=False)
    monkeypatch.delenv("FLA_CONFIG_DIR", raising=False)

    selected = binder.create_qwen_rocm_profile_binder()
    receipt = selected.bind(profile)
    assert receipt.strict and receipt.source_hashes_verified
    assert receipt.strict_cache_enforced and receipt.unknown_key_guard_enforced
    assert selected.bind(profile) is receipt

    legal = (128, 1, "torch.bfloat16", "torch.bfloat16", "torch.float32")
    assert kernel.run(key=legal) == "launched"
    assert calls == [legal]
    with pytest.raises(binder.QwenRocmBindingError, match="unprofiled.*key"):
        kernel.run(key=(128, 3, *legal[2:]))
    with pytest.raises(binder.QwenRocmBindingError, match="unprofiled.*kernel"):
        l2norm.l2norm_fwd_kernel1.run()
    assert calls == [legal]

    kernel.cache[legal] = SimpleNamespace(**{**configs[legal], "num_warps": 8})
    with pytest.raises(binder.QwenRocmBindingError, match="conflicting"):
        kernel.run(key=legal)
    assert calls == [legal]

    second_path = tmp_path / "another-profile.json"
    second_path.write_bytes(b"another validated profile")
    second = replace(
        profile,
        profile_path=second_path,
        profile_sha256=_sha256(second_path.read_bytes()),
    )
    with pytest.raises(binder.QwenRocmBindingError, match="different.*active"):
        selected.bind(second)


def test_installed_fla_source_drift_fails_before_environment_or_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile, package, _ = _profile(tmp_path)
    (package / "ops" / "utils" / "cache.py").write_bytes(b"changed cache source")
    imported = []
    monkeypatch.setattr(binder, "_ACTIVE", None)
    monkeypatch.setattr(binder, "sys", SimpleNamespace(modules={}))
    monkeypatch.setattr(binder, "_fla_package_root", lambda: package)
    monkeypatch.setattr(binder, "_import_module", lambda name: imported.append(name))
    monkeypatch.delenv("FLA_CACHE_MODE", raising=False)
    monkeypatch.delenv("FLA_CONFIG_DIR", raising=False)

    with pytest.raises(
        binder.QwenRocmBindingError, match="installed FLA source differs"
    ):
        binder.create_qwen_rocm_profile_binder().bind(profile)
    assert imported == []
    assert "FLA_CACHE_MODE" not in os.environ
    assert "FLA_CONFIG_DIR" not in os.environ


def test_kernel_config_hash_drift_fails_before_fla_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile, package, _ = _profile(tmp_path)
    profile.kernel_config_path.write_bytes(b"modified launch table")
    monkeypatch.setattr(binder, "_ACTIVE", None)
    monkeypatch.setattr(binder, "sys", SimpleNamespace(modules={}))
    monkeypatch.setattr(binder, "_fla_package_root", lambda: package)
    with pytest.raises(binder.QwenRocmBindingError, match="kernel config changed"):
        binder.create_qwen_rocm_profile_binder().bind(profile)


def test_preimported_fla_refuses_profile_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile, package, _ = _profile(tmp_path)
    monkeypatch.setattr(binder, "_ACTIVE", None)
    monkeypatch.setattr(binder, "sys", SimpleNamespace(modules={"fla": object()}))
    monkeypatch.setattr(binder, "_fla_package_root", lambda: package)
    monkeypatch.delenv("FLA_CACHE_MODE", raising=False)
    monkeypatch.delenv("FLA_CONFIG_DIR", raising=False)

    with pytest.raises(binder.QwenRocmBindingError, match="imported before"):
        binder.create_qwen_rocm_profile_binder().bind(profile)
    assert "FLA_CACHE_MODE" not in os.environ
    assert "FLA_CONFIG_DIR" not in os.environ
