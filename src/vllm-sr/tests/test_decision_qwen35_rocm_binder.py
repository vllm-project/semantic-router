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
            self.cache_lookups = 0
            self.cache_installs = 0

        def run(self, *args, **kwargs):
            # FLA v0.5.2 loads the config only on a cache miss; without a
            # config, its original run would enter an autotune fallback.
            if kwargs["key"] not in self.cache:
                self.maybe_load_cached_config(
                    SimpleNamespace(autotune_key=kwargs["key"])
                )
            if kwargs["key"] not in self.cache:
                raise RuntimeError("original run would autotune")
            calls.append(kwargs["key"])
            return "launched"

        def maybe_load_cached_config(self, key):
            self.cache_installs += 1
            selected = cache.load_cached_config(self.kernel_name, key)
            if selected is not None:
                self.cache[key.autotune_key] = SimpleNamespace(**selected)

    kernel = FakeKernel()
    strict = object()

    def load_cached_config(name, key):
        kernel.cache_lookups += 1
        return configs.get(key.autotune_key)

    cache = SimpleNamespace(
        __file__=str(package / "ops" / "utils" / "cache.py"),
        FlaCacheMode=SimpleNamespace(STRICT=strict),
        FLA_CACHE_MODE=strict,
        CachedAutotuner=FakeKernel,
        AutotuneKey=FakeKey,
        load_cached_config=load_cached_config,
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
    with pytest.raises(binder.QwenRocmBindingError, match=r"unprofiled.*key"):
        kernel.run(key=(128, 3, *legal[2:]))
    with pytest.raises(binder.QwenRocmBindingError, match=r"unprofiled.*kernel"):
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
    with pytest.raises(binder.QwenRocmBindingError, match=r"different.*active"):
        selected.bind(second)


def test_verified_warm_key_avoids_profile_hash_and_fla_config_lookup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile, package, configs = _profile(tmp_path)
    modules, kernel, _, calls = _fake_fla_modules(package, configs)
    monkeypatch.setattr(binder, "_ACTIVE", None)
    monkeypatch.setattr(binder, "sys", SimpleNamespace(modules={}))
    monkeypatch.setattr(binder, "_fla_package_root", lambda: package)
    monkeypatch.setattr(binder, "_import_module", modules.__getitem__)
    monkeypatch.delenv("FLA_CACHE_MODE", raising=False)
    monkeypatch.delenv("FLA_CONFIG_DIR", raising=False)
    binder.create_qwen_rocm_profile_binder().bind(profile)

    original_hash = binder._sha256_file
    hash_calls = []

    def count_hash(path):
        hash_calls.append(path)
        return original_hash(path)

    monkeypatch.setattr(binder, "_sha256_file", count_hash)
    legal = (128, 1, "torch.bfloat16", "torch.bfloat16", "torch.float32")
    for _ in range(48):
        assert kernel.run(key=legal) == "launched"
    assert calls == [legal] * 48
    assert hash_calls == [profile.kernel_config_path] * 48
    assert kernel.cache_lookups == 2  # first exact check and install only
    assert kernel.cache_installs == 1

    with pytest.raises(binder.QwenRocmBindingError, match="unprofiled.*key"):
        kernel.run(key=(128, 3, *legal[2:]))
    assert calls == [legal] * 48
    assert hash_calls == [profile.kernel_config_path] * 49
    assert kernel.cache_lookups == 2


@pytest.mark.parametrize("mutation", ["in_place", "replace"])
def test_config_file_mutation_stops_cached_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    profile, package, configs = _profile(tmp_path)
    modules, kernel, _, calls = _fake_fla_modules(package, configs)
    monkeypatch.setattr(binder, "_ACTIVE", None)
    monkeypatch.setattr(binder, "sys", SimpleNamespace(modules={}))
    monkeypatch.setattr(binder, "_fla_package_root", lambda: package)
    monkeypatch.setattr(binder, "_import_module", modules.__getitem__)
    monkeypatch.delenv("FLA_CACHE_MODE", raising=False)
    monkeypatch.delenv("FLA_CONFIG_DIR", raising=False)
    binder.create_qwen_rocm_profile_binder().bind(profile)

    legal = (128, 1, "torch.bfloat16", "torch.bfloat16", "torch.float32")
    assert kernel.run(key=legal) == "launched"
    if mutation == "in_place":
        original = profile.kernel_config_path.read_bytes()
        profile.kernel_config_path.write_bytes(b"x" * len(original))
    else:
        alternate = tmp_path / "replacement.json"
        alternate.write_bytes(b"x" * profile.kernel_config_path.stat().st_size)
        alternate.replace(profile.kernel_config_path)
    with pytest.raises(binder.QwenRocmBindingError, match="configuration changed"):
        kernel.run(key=legal)
    assert calls == [legal]


def test_installed_fla_source_drift_fails_before_environment_or_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile, package, _ = _profile(tmp_path)
    (package / "ops" / "utils" / "cache.py").write_bytes(b"changed cache source")
    imported = []
    monkeypatch.setattr(binder, "_ACTIVE", None)
    monkeypatch.setattr(binder, "sys", SimpleNamespace(modules={}))
    monkeypatch.setattr(binder, "_fla_package_root", lambda: package)
    monkeypatch.setattr(binder, "_import_module", imported.append)
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
