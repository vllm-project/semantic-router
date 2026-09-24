"""Dependency-free fail-closed tests for the owned Qwen Torch loader."""

from __future__ import annotations

import hashlib
import json
import sys
from functools import wraps
from pathlib import Path
from types import FunctionType, SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from decision_runtime import qwen35_torch  # noqa: E402
from decision_runtime.qwen35_torch import (  # noqa: E402
    QWEN_PROMPT_VERSION,
    RELEASED_MAX_INPUT_TOKENS,
    Qwen35RuntimeError,
    Qwen35TorchRuntime,
    QwenRocmProfileBinding,
    _bind_rocm_profile,
    _install_instance_native_gated_delta_kernels,
    _validate_configuration,
    _validate_device,
    _validated_rocm_profile,
)


def _artifact(tmp_path: Path, *, prompt_version: str = QWEN_PROMPT_VERSION) -> Path:
    root = tmp_path / "artifact"
    (root / "backbone").mkdir(parents=True)
    (root / "backbone/config.json").write_text("{}")
    (root / "decision_config.json").write_text(
        json.dumps(
            {
                "prompt_version": prompt_version,
                "head_dim": 256,
                "base_model": "Qwen/Qwen3.5-2B",
            }
        )
    )
    (root / "decision_head.safetensors").write_bytes(b"not loaded in this test")
    (root / "runtime.json").write_text("{}")
    (root / "tokenizer.json").write_text("{}")
    (root / "tokenizer_config.json").write_text("{}")
    return root


def test_loader_validates_release_contract_before_gpu_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    imported = []
    monkeypatch.setattr(
        "decision_runtime.qwen35_torch.importlib.import_module",
        imported.append,
    )
    root = _artifact(tmp_path, prompt_version="wrong")

    with pytest.raises(Qwen35RuntimeError, match="prompt version"):
        Qwen35TorchRuntime.load(
            root,
            temperature=1.0,
            max_length=16384,
            backend="rocm",
        )

    assert imported == []


@pytest.mark.parametrize(
    ("temperature", "max_length", "backend", "message"),
    (
        (0.0, 16384, "rocm", "temperature"),
        (float("nan"), 16384, "rocm", "temperature"),
        (1.0, 0, "rocm", "length"),
        (1.0, RELEASED_MAX_INPUT_TOKENS + 1, "rocm", "length"),
        (1.0, 16384, "mlx", "cpu, rocm, or cuda"),
    ),
)
def test_loader_rejects_invalid_profile_values_before_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    temperature,
    max_length,
    backend,
    message,
) -> None:
    imported = []
    monkeypatch.setattr(
        "decision_runtime.qwen35_torch.importlib.import_module",
        imported.append,
    )

    with pytest.raises(Qwen35RuntimeError, match=message):
        Qwen35TorchRuntime.load(
            _artifact(tmp_path),
            temperature=temperature,
            max_length=max_length,
            backend=backend,
        )

    assert imported == []


def test_loader_rejects_non_sdpa_attention_before_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    imported = []
    monkeypatch.setattr(
        "decision_runtime.qwen35_torch.importlib.import_module",
        imported.append,
    )

    with pytest.raises(Qwen35RuntimeError, match="SDPA"):
        Qwen35TorchRuntime.load(
            _artifact(tmp_path),
            temperature=1.0,
            max_length=16384,
            backend="rocm",
            attention="eager",
        )

    assert imported == []


def _profiled_artifact(tmp_path: Path, *, status: str = "hardware-qualified") -> Path:
    root = _artifact(tmp_path)
    profile_root = root / "runtime-profile"
    profile_root.mkdir()
    entries = {}
    for batch_size in range(1, 33):
        key = [
            128,
            batch_size,
            "torch.bfloat16",
            "torch.bfloat16",
            "torch.float32",
        ]
        encoded = json.dumps(key, sort_keys=True, separators=(",", ":"))
        entries[hashlib.md5(encoded.encode(), usedforsecurity=False).hexdigest()] = {
            "autotune_key": key,
            "config": {
                "kwargs": {"BT": 8},
                "num_warps": 1,
                "num_stages": 3,
                "num_ctas": 1,
                "maxnreg": None,
                "pre_hook": None,
                "ir_override": None,
            },
        }
    kernel_bytes = json.dumps(
        {"default_config": None, "autotune_entries": entries}, sort_keys=True
    ).encode()
    (profile_root / "l2norm_fwd_kernel.json").write_bytes(kernel_bytes)
    kernel_sha = hashlib.sha256(kernel_bytes).hexdigest()
    source_hashes = {
        "modules/l2norm.py": "1" * 64,
        "ops/utils/cache.py": "2" * 64,
    }
    profile_bytes = json.dumps(
        {
            "format": "decision-fla-l2norm-profile-v1",
            "status": status,
            "cache_mode": "strict",
            "supported": {
                "head_dimension": 128,
                "key_heads": 16,
                "body_dtype": "bfloat16",
                "output_dtype": "bfloat16",
                "rstd_dtype": "float32",
                "batch_size_min": 1,
                "batch_size_max": 8,
                "padded_tokens_max": 16384,
                "NB_min": 1,
                "NB_max": 32,
                "NB_formula": "ceil(B*padded_tokens*16/65536)",
                "unknown_key": "raise before kernel/autotune",
            },
            "runtime": {
                "torch": "2.12.0+git6bbd260",
                "hip": "7.2.53211",
                "triton": "3.7.1",
                "fla": "0.5.2",
                "gpu_arch": "gfx942",
            },
            "files": [{"file": "l2norm_fwd_kernel.json", "sha256": kernel_sha}],
            "fla_source_sha256": source_hashes,
        },
        sort_keys=True,
    ).encode()
    (profile_root / "profile.json").write_bytes(profile_bytes)
    profile_sha = hashlib.sha256(profile_bytes).hexdigest()
    guard_bytes = b"published guard contract; never executed"
    guard_path = root / "code" / "profile_guard.py"
    guard_path.parent.mkdir()
    guard_path.write_bytes(guard_bytes)
    guard_sha = hashlib.sha256(guard_bytes).hexdigest()
    (root / "runtime.json").write_text(
        json.dumps(
            {
                "normalization_profile": {
                    "kind": "decision-fla-l2norm-profile-v1",
                    "profile_file": "runtime-profile/profile.json",
                    "profile_sha256": profile_sha,
                    "guard_file": "code/profile_guard.py",
                    "guard_sha256": guard_sha,
                    "validated_arch": "gfx942",
                }
            }
        )
    )
    return root


def test_candidate_label_does_not_override_owned_kernel_contract(
    tmp_path: Path,
) -> None:
    root = _profiled_artifact(tmp_path, status="diagnostic-only")
    metadata = json.loads((root / "decision_config.json").read_text())

    assert _validated_rocm_profile(root, metadata) is not None


def test_rocm_profile_rejects_uncovered_physical_batch_before_torch_import(
    tmp_path: Path,
) -> None:
    root = _profiled_artifact(tmp_path, status="diagnostic-only")
    with pytest.raises(Qwen35RuntimeError, match="kernel envelope"):
        Qwen35TorchRuntime.load(
            root,
            temperature=1.3,
            max_length=16384,
            backend="rocm",
            physical_batch_size=9,
        )


def test_profile_validates_supported_envelope_and_fla_sources(tmp_path: Path) -> None:
    root = _profiled_artifact(tmp_path)
    profile_path = root / "runtime-profile/profile.json"
    profile = json.loads(profile_path.read_text())
    profile["supported"]["batch_size_max"] = 64
    profile_bytes = json.dumps(profile, sort_keys=True).encode()
    profile_path.write_bytes(profile_bytes)
    runtime = json.loads((root / "runtime.json").read_text())
    runtime["normalization_profile"]["profile_sha256"] = hashlib.sha256(
        profile_bytes
    ).hexdigest()
    (root / "runtime.json").write_text(json.dumps(runtime))
    metadata = json.loads((root / "decision_config.json").read_text())

    with pytest.raises(Qwen35RuntimeError, match="execution envelope"):
        _validated_rocm_profile(root, metadata)


def test_binder_receipt_must_prove_source_cache_and_guard_enforcement(
    tmp_path: Path,
) -> None:
    root = _profiled_artifact(tmp_path)
    metadata = json.loads((root / "decision_config.json").read_text())
    profile = _validated_rocm_profile(root, metadata)
    assert profile is not None

    class IncompleteBinder:
        def bind(self, selected):
            return QwenRocmProfileBinding(
                profile_sha256=selected.profile_sha256,
                kernel_config_sha256=selected.kernel_config_sha256,
                guard_contract_sha256=selected.guard_contract_sha256,
                runtime=selected.runtime,
                fla_source_sha256=selected.fla_source_sha256,
                physical_batch_size_min=selected.physical_batch_size_min,
                physical_batch_size_max=selected.physical_batch_size_max,
                normalization_blocks_min=selected.normalization_blocks_min,
                normalization_blocks_max=selected.normalization_blocks_max,
                strict=True,
                source_hashes_verified=True,
                strict_cache_enforced=False,
                unknown_key_guard_enforced=True,
            )

    with pytest.raises(Qwen35RuntimeError, match="invalid receipt"):
        _bind_rocm_profile(IncompleteBinder(), profile)


def test_profile_keeps_physical_batch_and_normalization_blocks_distinct(
    tmp_path: Path,
) -> None:
    root = _profiled_artifact(tmp_path)
    metadata = json.loads((root / "decision_config.json").read_text())
    profile = _validated_rocm_profile(root, metadata)
    assert profile is not None

    # Eight request rows at a short padded length still select one l2norm
    # normalization block; NB is a kernel shape, not a request count.
    assert (
        profile.normalization_blocks_for(physical_batch_size=8, padded_tokens=256) == 1
    )
    assert (
        profile.normalization_blocks_for(physical_batch_size=8, padded_tokens=16_384)
        == 32
    )
    with pytest.raises(Qwen35RuntimeError, match="physical batch size"):
        profile.normalization_blocks_for(physical_batch_size=9, padded_tokens=256)


def test_profiled_rocm_release_requires_owned_binder_before_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    imported = []
    monkeypatch.setattr(
        "decision_runtime.qwen35_torch.importlib.import_module",
        imported.append,
    )

    with pytest.raises(Qwen35RuntimeError, match="owned strict ROCm profile binder"):
        Qwen35TorchRuntime.load(
            _profiled_artifact(tmp_path),
            temperature=1.0,
            max_length=16384,
            backend="rocm",
        )

    assert imported == []


def test_optional_graph_refuses_unbound_rocm_model_before_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    imported = []
    monkeypatch.setattr(
        "decision_runtime.qwen35_torch.importlib.import_module", imported.append
    )
    with pytest.raises(Qwen35RuntimeError, match="verified strict B8 profile"):
        Qwen35TorchRuntime.load(
            _artifact(tmp_path),
            temperature=1.0,
            max_length=1024,
            backend="rocm",
            physical_batch_size=8,
            enable_rocm_graph=True,
            artifact_content_id="a" * 64,
            graph_model_id=qwen35_torch.EXPERIMENTAL_SOL_GRAPH_MODEL_ID,
        )
    assert imported == []


@pytest.mark.parametrize(
    "model_id",
    (
        "llm-semantic-router/Decision-1.0-Nox-4B",
        "llm-semantic-router/Decision-1.0-Lux-9B",
    ),
)
def test_optional_graph_refuses_non_sol_model_before_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model_id: str
) -> None:
    imported = []
    monkeypatch.setattr(
        "decision_runtime.qwen35_torch.importlib.import_module", imported.append
    )
    with pytest.raises(Qwen35RuntimeError, match="verified strict B8 profile"):
        Qwen35TorchRuntime.load(
            _profiled_artifact(tmp_path),
            temperature=1.0,
            max_length=16384,
            backend="rocm",
            physical_batch_size=8,
            enable_rocm_graph=True,
            artifact_content_id="a" * 64,
            graph_model_id=model_id,
        )
    assert imported == []


def test_profile_hash_mismatch_fails_before_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _profiled_artifact(tmp_path)
    runtime = json.loads((root / "runtime.json").read_text())
    runtime["normalization_profile"]["profile_sha256"] = "0" * 64
    (root / "runtime.json").write_text(json.dumps(runtime))
    imported = []
    monkeypatch.setattr(
        "decision_runtime.qwen35_torch.importlib.import_module",
        imported.append,
    )

    with pytest.raises(Qwen35RuntimeError, match="profile hash mismatch"):
        Qwen35TorchRuntime.load(
            root,
            temperature=1.0,
            max_length=16384,
            backend="rocm",
        )

    assert imported == []


def test_profile_path_traversal_fails_before_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _profiled_artifact(tmp_path)
    runtime = json.loads((root / "runtime.json").read_text())
    runtime["normalization_profile"]["profile_file"] = "../profile.json"
    (root / "runtime.json").write_text(json.dumps(runtime))
    imported = []
    monkeypatch.setattr(
        "decision_runtime.qwen35_torch.importlib.import_module",
        imported.append,
    )

    with pytest.raises(Qwen35RuntimeError, match=r"unsafe.*profile path"):
        Qwen35TorchRuntime.load(
            root,
            temperature=1.0,
            max_length=16384,
            backend="rocm",
        )

    assert imported == []


def test_cpu_loader_requires_pinned_manifest_before_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    imported = []
    monkeypatch.setattr(
        "decision_runtime.qwen35_torch.importlib.import_module",
        imported.append,
    )

    with pytest.raises(Qwen35RuntimeError, match="pinned release manifest"):
        Qwen35TorchRuntime.load(
            _artifact(tmp_path),
            temperature=1.0,
            max_length=1024,
            backend="cpu",
        )

    assert imported == []


def test_cpu_loader_rejects_wrong_manifest_before_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _artifact(tmp_path)
    (root / "MODEL_MANIFEST.json").write_text("{}")
    imported = []
    monkeypatch.setattr(
        "decision_runtime.qwen35_torch.importlib.import_module",
        imported.append,
    )

    with pytest.raises(Qwen35RuntimeError, match="manifest digest mismatch"):
        Qwen35TorchRuntime.load(
            root,
            temperature=1.0,
            max_length=1024,
            backend="cpu",
            expected_manifest_sha256="0" * 64,
        )

    assert imported == []


def test_cpu_loader_converts_verified_body_to_fp32(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _artifact(tmp_path)
    calls = []
    monkeypatch.setattr(
        qwen35_torch,
        "verify_release_manifest",
        lambda *args, **kwargs: calls.append(kwargs["manifest_name"]),
    )

    class FakeModule:
        def __init__(self, dtype="fp32"):
            self.dtype = dtype
            self.config = SimpleNamespace(hidden_size=16, use_cache=True)

        def load_state_dict(self, *args, **kwargs):
            pass

        def to(self, device):
            calls.append(device.type)
            return self

        def eval(self):
            return self

        def parameters(self):
            return (SimpleNamespace(dtype=self.dtype),)

    def from_pretrained(*args, **kwargs):
        calls.append(kwargs["dtype"])
        return FakeModule(dtype=kwargs["dtype"])

    fake_torch = SimpleNamespace(
        float32="fp32",
        bfloat16="bf16",
        device=lambda name: SimpleNamespace(type=name),
    )
    fake_transformers = SimpleNamespace(
        __version__=qwen35_torch.SUPPORTED_TRANSFORMERS_VERSION,
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: object()),
    )
    fake_safetensors = SimpleNamespace(load_file=lambda *a, **k: {})
    modules = {
        "torch": fake_torch,
        "transformers": fake_transformers,
        "safetensors.torch": fake_safetensors,
    }
    monkeypatch.setattr(qwen35_torch, "_required_module", modules.__getitem__)
    monkeypatch.setattr(
        qwen35_torch.importlib,
        "import_module",
        lambda name: SimpleNamespace(
            Qwen3_5TextModel=SimpleNamespace(from_pretrained=from_pretrained)
        ),
    )
    monkeypatch.setattr(qwen35_torch, "_candidate_head", lambda *a: FakeModule())
    monkeypatch.setattr(qwen35_torch, "_decision_model", lambda *a: FakeModule())
    monkeypatch.setattr(
        qwen35_torch,
        "_install_instance_native_gated_delta_kernels",
        lambda *a: calls.append("cpu-reference"),
    )

    runtime = Qwen35TorchRuntime.load(
        root,
        temperature=1.0,
        max_length=1024,
        backend="cpu",
        expected_manifest_sha256="a" * 64,
    )

    assert runtime.device.type == "cpu"
    assert runtime.rocm_profile_binding is None
    assert calls == ["MODEL_MANIFEST.json", "fp32", "cpu", "cpu-reference"]
    with pytest.raises(Qwen35RuntimeError, match="CPU device"):
        _validate_device(
            fake_torch,
            SimpleNamespace(type="cuda"),
            backend="cpu",
            rocm_profile=None,
        )


@pytest.mark.parametrize("head_dtype", ("fp32", "bf16"))
def test_native_rocm_loader_keeps_bf16_body_and_fp32_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, head_dtype: str
) -> None:
    root = _artifact(tmp_path)
    calls = []
    monkeypatch.setattr(qwen35_torch, "verify_release_manifest", lambda *a, **k: None)

    class FakeModule:
        def __init__(self, dtype="fp32"):
            self.dtype = dtype
            self.config = SimpleNamespace(hidden_size=16, use_cache=True)

        def load_state_dict(self, *args, **kwargs):
            pass

        def to(self, device):
            return self

        def eval(self):
            return self

        def parameters(self):
            return (SimpleNamespace(dtype=self.dtype),)

    def from_pretrained(*args, **kwargs):
        calls.append(("backbone", kwargs["dtype"]))
        return FakeModule(kwargs["dtype"])

    fake_torch = SimpleNamespace(
        float32="fp32",
        bfloat16="bf16",
        device=lambda name: SimpleNamespace(type="cuda"),
        version=SimpleNamespace(hip="7.2"),
        cuda=SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: True),
    )
    fake_transformers = SimpleNamespace(
        __version__=qwen35_torch.SUPPORTED_TRANSFORMERS_VERSION,
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: object()),
    )
    modules = {
        "torch": fake_torch,
        "transformers": fake_transformers,
        "safetensors.torch": SimpleNamespace(load_file=lambda *a, **k: {}),
    }
    monkeypatch.setattr(qwen35_torch, "_required_module", modules.__getitem__)
    monkeypatch.setattr(
        qwen35_torch.importlib,
        "import_module",
        lambda name: SimpleNamespace(
            Qwen3_5TextModel=SimpleNamespace(from_pretrained=from_pretrained)
        ),
    )
    monkeypatch.setattr(
        qwen35_torch, "_candidate_head", lambda *a: FakeModule(head_dtype)
    )
    monkeypatch.setattr(qwen35_torch, "_decision_model", lambda *a: FakeModule())
    monkeypatch.setattr(
        qwen35_torch,
        "_install_instance_native_gated_delta_kernels",
        lambda *a: calls.append(("native", None)),
    )

    options = {
        "temperature": 1.0,
        "max_length": 1024,
        "backend": "rocm",
        "gated_delta_kernel_policy": "native_torch",
        "native_rocm_max_physical_batch_size": 8,
        "physical_batch_size": 8,
        "expected_manifest_sha256": "a" * 64,
    }
    if head_dtype == "bf16":
        with pytest.raises(Qwen35RuntimeError, match="candidate-head storage"):
            Qwen35TorchRuntime.load(root, **options)
    else:
        runtime = Qwen35TorchRuntime.load(root, **options)
        assert runtime.gated_delta_kernel_policy == "native_torch"
        assert runtime.rocm_profile_binding is None
    assert calls == [("backbone", "bf16"), ("native", None)]


def _native_kernel_surface():
    """Small pinned-signature source stand-ins with an accelerated dispatch."""

    def source(function):
        return FunctionType(
            function.__code__.replace(co_filename="modeling_qwen3_5.py"),
            function.__globals__,
            function.__name__,
            function.__defaults__,
        )

    def conv_fn(hidden_states, weight, bias=None, activation=None, **kwargs):
        return "native"

    def conv_update(hidden_states, conv_state, weight, bias=None, activation=None):
        return "native"

    def chunk(
        query,
        key,
        value,
        g,
        beta,
        chunk_size=64,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
        **kwargs,
    ):
        return "native"

    def recurrent(
        query,
        key,
        value,
        g,
        beta,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
        **kwargs,
    ):
        return "native"

    def norm_forward(self, hidden_states, gate):
        return "native norm"

    conv_fn = source(conv_fn)
    conv_update = source(conv_update)
    chunk = source(chunk)
    recurrent = source(recurrent)
    norm_forward = source(norm_forward)

    @wraps(conv_fn)
    def accelerated_conv(*args, **kwargs):
        return "accelerated"

    @wraps(norm_forward)
    def accelerated_norm(*args, **kwargs):
        return "accelerated norm"

    def original_forward(
        self, hidden_states, cache_params=None, attention_mask=None, **kwargs
    ):
        _ = (
            causal_conv1d_update,  # noqa: F821
            torch_chunk_gated_delta_rule,  # noqa: F821
            torch_recurrent_gated_delta_rule,  # noqa: F821
        )
        return causal_conv1d_fn(hidden_states, None)  # noqa: F821

    original_forward = FunctionType(
        original_forward.__code__.replace(co_filename="modeling_qwen3_5.py"),
        {
            "causal_conv1d_fn": accelerated_conv,
            "causal_conv1d_update": accelerated_conv,
            "torch_chunk_gated_delta_rule": accelerated_conv,
            "torch_recurrent_gated_delta_rule": accelerated_conv,
        },
        "forward",
        original_forward.__defaults__,
    )

    @wraps(original_forward)
    def accelerated_forward(*args, **kwargs):
        return "accelerated"

    class FakeNorm:
        forward = accelerated_norm

    class FakeLayer:
        forward = accelerated_forward

        def __init__(self):
            self.conv1d = object()
            self.norm = FakeNorm()

    modeling = SimpleNamespace(
        Qwen3_5GatedDeltaNet=FakeLayer,
        Qwen3_5RMSNormGated=FakeNorm,
        causal_conv1d_fn=accelerated_conv,
        causal_conv1d_update=conv_update,
        torch_chunk_gated_delta_rule=chunk,
        torch_recurrent_gated_delta_rule=recurrent,
    )
    return modeling, FakeLayer(), FakeLayer()


def test_native_gated_delta_binding_is_instance_local_in_fla_process() -> None:
    modeling, native_layer, accelerated_layer = _native_kernel_surface()
    model = SimpleNamespace(modules=lambda: (native_layer,))

    _install_instance_native_gated_delta_kernels(model, modeling)

    assert native_layer.forward(None) == "native"
    assert native_layer.norm.forward(None, None) == "native norm"
    assert accelerated_layer.forward(None) == "accelerated"
    assert accelerated_layer.norm.forward(None, None) == "accelerated norm"
    assert modeling.Qwen3_5GatedDeltaNet.forward(None, None) == "accelerated"
    assert modeling.Qwen3_5RMSNormGated.forward(None, None, None) == "accelerated norm"
    assert modeling.causal_conv1d_fn(None, None) == "accelerated"


@pytest.mark.parametrize("changed", ("forward", "norm", "kernel"))
def test_native_binding_rejects_changed_source_signature_before_mutation(changed):
    modeling, layer, other = _native_kernel_surface()
    if changed == "forward":

        def incompatible_forward(self, hidden_states):
            return "native"

        modeling.Qwen3_5GatedDeltaNet.forward = incompatible_forward
    elif changed == "norm":

        def incompatible_norm(self, hidden_states):
            return "native"

        modeling.Qwen3_5RMSNormGated.forward = incompatible_norm
    else:

        def incompatible_kernel(query, key):
            return "native"

        modeling.torch_chunk_gated_delta_rule = incompatible_kernel

    with pytest.raises(Qwen35RuntimeError, match="native Qwen"):
        _install_instance_native_gated_delta_kernels(
            SimpleNamespace(modules=lambda: (layer,)), modeling
        )

    assert "forward" not in layer.__dict__
    assert "forward" not in layer.norm.__dict__
    assert "forward" not in other.__dict__


def test_native_binding_rejects_mixed_subclass_layers_before_mutation():
    modeling, layer, _ = _native_kernel_surface()

    class ModifiedLayer(modeling.Qwen3_5GatedDeltaNet):
        pass

    modified = ModifiedLayer()
    with pytest.raises(Qwen35RuntimeError, match="modified or offloaded"):
        _install_instance_native_gated_delta_kernels(
            SimpleNamespace(modules=lambda: (layer, modified)), modeling
        )

    assert "forward" not in layer.__dict__
    assert "forward" not in layer.norm.__dict__
    assert "forward" not in modified.__dict__


def test_eos_native_rocm_policy_ignores_unneeded_fla_profile(tmp_path: Path) -> None:
    root = _artifact(tmp_path)
    (root / "runtime.json").write_text(
        json.dumps({"normalization_profile": {"kind": "untrusted-fla-profile"}})
    )
    assert (
        _validate_configuration(
            root,
            temperature=1.0,
            max_length=1024,
            backend="rocm",
            gated_delta_kernel_policy="native_torch",
            native_rocm_max_physical_batch_size=8,
            attention="sdpa",
            physical_batch_size=8,
        )
        is None
    )
    with pytest.raises(Qwen35RuntimeError, match="qualified maximum 8"):
        _validate_configuration(
            root,
            temperature=1.0,
            max_length=1024,
            backend="rocm",
            gated_delta_kernel_policy="native_torch",
            native_rocm_max_physical_batch_size=8,
            attention="sdpa",
            physical_batch_size=16,
        )
    with pytest.raises(Qwen35RuntimeError, match="profile specification"):
        _validate_configuration(
            root,
            temperature=1.0,
            max_length=1024,
            backend="rocm",
            gated_delta_kernel_policy="accelerated",
            attention="sdpa",
            physical_batch_size=8,
        )


def test_native_rocm_batch_cap_does_not_restrict_cpu(tmp_path: Path) -> None:
    assert (
        _validate_configuration(
            _artifact(tmp_path),
            temperature=1.0,
            max_length=1024,
            backend="cpu",
            gated_delta_kernel_policy="native_torch",
            attention="sdpa",
            physical_batch_size=16,
        )
        is None
    )
