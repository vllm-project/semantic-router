"""Dependency-free fail-closed tests for the owned Vela Torch loader."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from decision_runtime import vela_torch  # noqa: E402
from decision_runtime.vela_inputs import EncodedVelaRow  # noqa: E402
from decision_runtime.vela_torch import (  # noqa: E402
    REQUIRED_HEAD_STRUCTURE,
    SUPPORTED_TRANSFORMERS_VERSION,
    ValidatedVelaArtifact,
    VelaRuntimeError,
    VelaTorchRuntime,
    _validate_device,
)


def _artifact(tmp_path: Path, **overrides) -> Path:
    root = tmp_path / "native"
    for directory in (root / "encoder", root / "tokenizer"):
        directory.mkdir(parents=True)
    config = {
        "architecture": "vela_decision_score_path_capacity_v1",
        "schema": "decision.nano.score_path_capacity.v1",
        "arm": "all22",
        "training_arm": "S22",
        "head": {
            **REQUIRED_HEAD_STRUCTURE,
            "head_dropout": 0.1,
            "attention_mode": "sdpa",
            "gradient_checkpointing": True,
        },
        "transformers_version": SUPPORTED_TRANSFORMERS_VERSION,
        "weight_dtype": "float32",
        "inference_precision": "fp32",
        "base_model": "llm-semantic-router/Vela-1.0-Encoder-307M",
        "branch_policy": (
            "frozen_integrated_choice_noul_score_private_suffix_full_batch_v1"
        ),
        "calibration": "none; raw probabilities",
        "parameters": 571909635,
        "rocm_contiguous_layout": True,
        "type_order": ["choice", "noul", "score"],
        "packing": {"max_length": 8192, "state_truncation": "error"},
    }
    config.update(overrides)
    (root / "decision_config.json").write_text(json.dumps(config))
    (root / "STATE_LAYOUT.json").write_text(json.dumps({"training_run": "fixture"}))
    (root / "INVENTORY.json").write_text(
        json.dumps(
            {
                "counts": {"unique_tensors": 489},
                "encoder_shapes": {"embeddings.weight": [2, 2]},
                "choice_suffix_shapes": {"choice_blocks.0.weight": [2, 2]},
                "score_suffix_shapes": {"score_blocks.0.weight": [2, 2]},
                "head_shapes": {"type_embedding.weight": [3, 2]},
            }
        )
    )
    for path in (
        root / "encoder/config.json",
        root / "encoder/model.safetensors",
        root / "decision_heads.safetensors",
        root / "choice_encoder.safetensors",
        root / "score_encoder.safetensors",
        root / "tokenizer/tokenizer.json",
        root / "tokenizer/tokenizer_config.json",
    ):
        path.write_bytes(b"fixture")
    return root


def test_loader_rejects_head_structure_mismatch_before_gpu_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    imported = []
    monkeypatch.setattr(
        "decision_runtime.vela_torch.importlib.import_module",
        imported.append,
    )

    with pytest.raises(VelaRuntimeError, match="head structure"):
        VelaTorchRuntime.load(
            _artifact(tmp_path, head={**REQUIRED_HEAD_STRUCTURE, "head_heads": 8}),
            max_length=1024,
            backend="rocm",
        )

    assert imported == []


def test_provenance_metadata_does_not_gate_compatible_checkpoint(
    tmp_path: Path,
) -> None:
    root = _artifact(
        tmp_path,
        transformers_version="4.57.7",
        architecture="renamed-architecture-label",
        schema="renamed-schema-label",
        arm="new-training-arm",
        training_arm="new-training-stage",
        base_model="updated-model-origin",
        branch_policy="updated-training-recipe",
        calibration="updated-training-calibration-note",
        parameters=1,
        rocm_contiguous_layout=False,
        inference_precision="updated-precision-note",
        weight_dtype="updated-weight-note",
        state_layout={"updated": "description"},
        head={
            **REQUIRED_HEAD_STRUCTURE,
            "head_dropout": 0.2,
            "attention_mode": "updated-implementation-note",
            "gradient_checkpointing": False,
        },
    )

    artifact = vela_torch._validate_artifact(root, max_length=1024, backend="rocm")

    assert artifact.config["transformers_version"] == "4.57.7"
    assert artifact.inventory["encoder_shapes"] == {"embeddings.weight": (2, 2)}


def test_packing_capacity_can_grow_without_a_runtime_update(tmp_path: Path) -> None:
    artifact = vela_torch._validate_artifact(
        _artifact(tmp_path, packing={"max_length": 16384, "state_truncation": "error"}),
        max_length=12288,
        backend="rocm",
    )

    assert artifact.config["packing"]["max_length"] == 16384


def test_loader_rejects_type_order_mismatch_before_gpu_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    imported = []
    monkeypatch.setattr(
        "decision_runtime.vela_torch.importlib.import_module", imported.append
    )

    with pytest.raises(VelaRuntimeError, match="type order"):
        VelaTorchRuntime.load(
            _artifact(tmp_path, type_order=["score", "noul", "choice"]),
            max_length=1024,
            backend="rocm",
        )

    assert imported == []


def test_loader_rejects_tensor_shape_mismatch_against_model(
    tmp_path: Path,
) -> None:
    artifact = vela_torch._validate_artifact(
        _artifact(tmp_path), max_length=1024, backend="rocm"
    )
    state = {
        "encoder.embeddings.weight": SimpleNamespace(shape=(3, 2), dtype="fp32"),
        "choice_blocks.0.weight": SimpleNamespace(shape=(2, 2), dtype="fp32"),
        "score_blocks.0.weight": SimpleNamespace(shape=(2, 2), dtype="fp32"),
        "type_embedding.weight": SimpleNamespace(shape=(3, 2), dtype="fp32"),
    }

    with pytest.raises(VelaRuntimeError, match="tensor shape mismatch"):
        vela_torch._validate_model_inventory(
            SimpleNamespace(float32="fp32"), state, artifact.inventory
        )


def test_encoder_compile_provenance_does_not_gate_structural_config() -> None:
    encoder_config = SimpleNamespace(
        model_type="modernbert",
        hidden_size=768,
        num_hidden_layers=22,
        num_attention_heads=12,
        max_position_embeddings=32768,
        reference_compile=True,
    )

    vela_torch._validate_encoder_config(encoder_config)

    encoder_config.hidden_size = 1024
    with pytest.raises(VelaRuntimeError, match="hidden_size"):
        vela_torch._validate_encoder_config(encoder_config)


@pytest.mark.parametrize(
    ("max_length", "backend", "message"),
    (
        (0, "rocm", "length"),
        (1024, "mlx", "cpu, rocm, or cuda"),
    ),
)
def test_loader_rejects_invalid_profile_values_before_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    max_length,
    backend,
    message,
) -> None:
    imported = []
    monkeypatch.setattr(
        "decision_runtime.vela_torch.importlib.import_module",
        imported.append,
    )

    with pytest.raises(VelaRuntimeError, match=message):
        VelaTorchRuntime.load(
            _artifact(tmp_path), max_length=max_length, backend=backend
        )

    assert imported == []


def test_loader_rejects_packing_contract_drift_before_gpu_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    imported = []
    monkeypatch.setattr(
        "decision_runtime.vela_torch.importlib.import_module",
        imported.append,
    )

    with pytest.raises(VelaRuntimeError, match="packing contract"):
        VelaTorchRuntime.load(
            _artifact(
                tmp_path, packing={"max_length": 8192, "state_truncation": "clip"}
            ),
            max_length=1024,
            backend="rocm",
        )

    assert imported == []


def test_executor_rejects_mixed_type_physical_batch_before_tensor_work() -> None:
    runtime = VelaTorchRuntime(
        model=None,
        tokenizer=None,
        torch=None,
        device=None,
        max_length=1024,
        backend="rocm",
    )
    rows = (
        EncodedVelaRow(
            question_id="choice",
            type="choice",
            input_ids=(1, 2, 3),
            marker_positions=(1,),
            candidate_ids=("a",),
            state_tokens=1,
        ),
        EncodedVelaRow(
            question_id="score",
            type="score",
            input_ids=(1, 2, 3),
            marker_positions=(1,),
            candidate_ids=("0",),
            state_tokens=1,
        ),
    )

    with pytest.raises(ValueError, match="exactly one question type"):
        runtime.predict_encoded(rows)


def test_cpu_loader_requires_pinned_manifest_before_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    imported = []
    monkeypatch.setattr(
        "decision_runtime.vela_torch.importlib.import_module",
        imported.append,
    )

    with pytest.raises(VelaRuntimeError, match="pinned release manifest"):
        VelaTorchRuntime.load(_artifact(tmp_path), max_length=1024, backend="cpu")

    assert imported == []


def test_cpu_loader_uses_cpu_device_without_rocm_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _artifact(tmp_path)
    calls = []
    monkeypatch.setattr(
        vela_torch,
        "_validate_artifact",
        lambda *args, **kwargs: ValidatedVelaArtifact(
            config={"head": {}},
            inventory={
                "encoder_shapes": {},
                "head_shapes": {},
                "choice_suffix_shapes": {},
                "score_suffix_shapes": {},
            },
        ),
    )
    monkeypatch.setattr(
        vela_torch,
        "verify_release_manifest",
        lambda *args, **kwargs: calls.append(kwargs["manifest_name"]),
    )
    monkeypatch.setattr(vela_torch, "_validate_encoder_config", lambda config: None)
    monkeypatch.setattr(vela_torch, "_validate_tensor_inventory", lambda *a, **k: None)
    monkeypatch.setattr(vela_torch, "_validate_model_inventory", lambda *a: None)
    monkeypatch.setattr(
        vela_torch,
        "_install_rocm_layout_guard",
        lambda *args: pytest.fail("ROCm guard installed on CPU"),
    )

    class FakeModule:
        def load_state_dict(self, *args, **kwargs):
            pass

        def state_dict(self):
            return {}

        def to(self, device):
            calls.append(device.type)
            return self

        def eval(self):
            return self

        def parameters(self):
            return (SimpleNamespace(dtype="fp32"),)

    fake_model = FakeModule()
    fake_torch = SimpleNamespace(
        float32="fp32",
        device=lambda name: SimpleNamespace(type=name),
    )
    fake_transformers = SimpleNamespace(
        __version__=SUPPORTED_TRANSFORMERS_VERSION,
        AutoConfig=SimpleNamespace(from_pretrained=lambda *a, **k: object()),
        AutoModel=SimpleNamespace(from_config=lambda *a, **k: FakeModule()),
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: object()),
    )
    fake_safetensors = SimpleNamespace(load_file=lambda *a, **k: {})
    modules = {
        "torch": fake_torch,
        "transformers": fake_transformers,
        "safetensors.torch": fake_safetensors,
    }
    monkeypatch.setattr(vela_torch, "_required_module", modules.__getitem__)
    monkeypatch.setattr(vela_torch, "_vela_model", lambda *a: fake_model)

    runtime = VelaTorchRuntime.load(
        root,
        max_length=1024,
        backend="cpu",
        expected_manifest_sha256="a" * 64,
    )

    assert runtime.device.type == "cpu"
    assert calls == ["MANIFEST.json", "cpu"]
    with pytest.raises(VelaRuntimeError, match="CPU device"):
        _validate_device(fake_torch, SimpleNamespace(type="cuda"), backend="cpu")
