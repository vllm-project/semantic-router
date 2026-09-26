"""LoRA coverage, source identity, and resume contracts without a GPU."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from training.model.infer import checkpoint_fingerprint
from training.model.lora import (
    lora_metadata,
    select_target_modules,
    verify_adapter_config,
)
from training.model.plan import validate_resume_state
from training.model.source import source_fingerprint, verify_source


def write_adapter_weights(
    path: Path, shapes: dict[str, list[int]], fill: bytes = b"\0"
) -> None:
    offset = 0
    header = {}
    for name, shape in sorted(shapes.items()):
        size = math.prod(shape) * 4
        header[name] = {
            "dtype": "F32",
            "shape": shape,
            "data_offsets": [offset, offset + size],
        }
        offset += size
    encoded = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + fill * offset)


class Linear:
    pass


class HybridBackbone:
    def __init__(self, *, missing: str | None = None, bad_type: str | None = None):
        self.config = SimpleNamespace(
            layer_types=["linear_attention", "full_attention"], num_hidden_layers=2
        )
        names = [
            "layers.0.mlp.gate_proj",
            "layers.0.mlp.up_proj",
            "layers.0.mlp.down_proj",
            "layers.0.linear_attn.in_proj_qkv",
            "layers.0.linear_attn.in_proj_z",
            "layers.0.linear_attn.in_proj_b",
            "layers.0.linear_attn.in_proj_a",
            "layers.0.linear_attn.out_proj",
            "layers.1.mlp.gate_proj",
            "layers.1.mlp.up_proj",
            "layers.1.mlp.down_proj",
            "layers.1.self_attn.q_proj",
            "layers.1.self_attn.k_proj",
            "layers.1.self_attn.v_proj",
            "layers.1.self_attn.o_proj",
        ]
        self.modules = {
            name: (object() if name == bad_type else Linear())
            for name in names
            if name != missing
        }

    def named_modules(self):
        return iter(self.modules.items())


class LoRAContractTest(unittest.TestCase):
    def test_hybrid_qwen35_target_coverage_is_explicit(self):
        targets = select_target_modules(
            HybridBackbone(), lambda module: isinstance(module, Linear)
        )
        self.assertEqual(len(targets), 15)
        self.assertIn("layers.0.linear_attn.in_proj_qkv", targets)
        self.assertIn("layers.1.self_attn.q_proj", targets)
        self.assertIn("layers.0.mlp.down_proj", targets)
        with self.assertRaisesRegex(ValueError, "in_proj_a"):
            select_target_modules(
                HybridBackbone(missing="layers.0.linear_attn.in_proj_a"),
                lambda module: isinstance(module, Linear),
            )
        with self.assertRaisesRegex(ValueError, "q_proj"):
            select_target_modules(
                HybridBackbone(bad_type="layers.1.self_attn.q_proj"),
                lambda module: isinstance(module, Linear),
            )

    def test_posttrained_source_stage_is_recorded(self):
        contract = lora_metadata(
            rank=8,
            alpha=16,
            dropout=0.05,
            target_modules=["layers.0.self_attn.q_proj"],
            target_dimensions={"layers.0.self_attn.q_proj": [4, 5]},
            source_kind="posttrained",
            source_fingerprint={"files_sha256": {"model.safetensors": "a" * 64}},
            base_revision="immutable-posttrained-commit",
        )
        self.assertEqual(contract["source_kind"], "posttrained")
        self.assertEqual(contract["base_revision"], "immutable-posttrained-commit")

    def test_adapter_and_source_hashes_form_full_inference_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            (source / "backbone").mkdir(parents=True)
            (source / "backbone" / "model.safetensors").write_bytes(b"base")
            (source / "decision_config.json").write_text('{"revision":"immutable"}')
            fingerprint = source_fingerprint(source)
            verify_source(source, fingerprint)
            checkpoint = root / "checkpoint"
            (checkpoint / "adapter").mkdir(parents=True)
            targets = ["layers.0.self_attn.q_proj", "layers.1.self_attn.q_proj"]
            dimensions = {name: [4, 5] for name in targets}
            contract = lora_metadata(
                rank=16,
                alpha=32,
                dropout=0.05,
                target_modules=targets,
                target_dimensions=dimensions,
                source_kind="decision1",
                source_fingerprint=fingerprint,
                base_revision="immutable",
            )
            (checkpoint / "decision_config.json").write_text(
                json.dumps({"checkpoint_format": "peft-lora/1", "lora": contract})
            )
            (checkpoint / "decision_head.safetensors").write_bytes(b"head")
            (checkpoint / "adapter" / "adapter_config.json").write_text(
                json.dumps(
                    {
                        "peft_type": "LORA",
                        "r": 16,
                        "lora_alpha": 32,
                        "lora_dropout": 0.05,
                        "bias": "none",
                        "target_modules": ["q_proj"],
                    }
                )
            )
            shapes = {}
            for name in targets:
                shapes[f"base_model.model.{name}.lora_A.weight"] = [16, 4]
                shapes[f"base_model.model.{name}.lora_B.weight"] = [5, 16]
            weights = checkpoint / "adapter" / "adapter_model.safetensors"
            write_adapter_weights(weights, shapes)
            verify_adapter_config(checkpoint / "adapter", contract)
            with self.assertRaisesRegex(ValueError, "source-path"):
                checkpoint_fingerprint(checkpoint)
            first = checkpoint_fingerprint(checkpoint, source)
            self.assertIn("source/backbone/model.safetensors", first["files_sha256"])
            self.assertIn(
                "checkpoint/adapter/adapter_model.safetensors", first["files_sha256"]
            )
            write_adapter_weights(weights, shapes, fill=b"\1")
            self.assertNotEqual(
                first["model_sha256"],
                checkpoint_fingerprint(checkpoint, source)["model_sha256"],
            )
            write_adapter_weights(
                weights,
                {key: value for key, value in shapes.items() if "layers.1" not in key},
            )
            with self.assertRaisesRegex(ValueError, "tensor coverage"):
                verify_adapter_config(checkpoint / "adapter", contract)
            write_adapter_weights(weights, shapes)
            (source / "backbone" / "model.safetensors").write_bytes(b"mutated-base")
            with self.assertRaisesRegex(ValueError, "differ"):
                checkpoint_fingerprint(checkpoint, source)
            with self.assertRaisesRegex(ValueError, "differ"):
                verify_source(source, fingerprint)

    def test_resume_contract_includes_adapter_and_immutable_source(self):
        contract = {
            "planned_updates": 4,
            "train_count": 8,
            "replay_pool_count": 0,
            "replay_fraction": 0.0,
            "microbatch": 1,
            "accumulation": 2,
            "epochs": 1,
            "model_source": {"files_sha256": {"model.safetensors": "a" * 64}},
            "lora": {
                "rank": 16,
                "alpha": 32,
                "dropout": 0.05,
                "lr": 1e-4,
                "target_modules": ["layers.0.self_attn.q_proj"],
            },
        }
        state = {
            "contract": contract,
            "code_sha256": {"lora.py": "b" * 64},
            "step": 2,
            "next_epoch": 0,
            "next_batch": 4,
        }
        validate_resume_state(state, contract, {"lora.py": "b" * 64})
        changed = dict(contract, lora=dict(contract["lora"], rank=32))
        with self.assertRaisesRegex(ValueError, "contract"):
            validate_resume_state(state, changed, {"lora.py": "b" * 64})
        changed = dict(
            contract, model_source={"files_sha256": {"model.safetensors": "c" * 64}}
        )
        with self.assertRaisesRegex(ValueError, "contract"):
            validate_resume_state(state, changed, {"lora.py": "b" * 64})


if __name__ == "__main__":
    unittest.main()
