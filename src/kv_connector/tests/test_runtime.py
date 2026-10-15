"""Contract checks for an artifact before it is eligible for vLLM loading."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file

from src.kv_connector.runtime import MapperArtifact
from src.kv_connector.transform import map_source_cache, qwen3_rope
from src.training.kv_mapper.artifact import (
    CompatibilitySpec,
    Manifest,
    write_artifact,
)


class MapperRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.compat = CompatibilitySpec(
            source_model="source",
            source_revision="a" * 40,
            target_model="target",
            target_revision="b" * 40,
            variant="full_head",
            precision="bf16",
            source_tp=1,
            target_tp=1,
            head_order="contiguous",
            num_kv_heads=2,
            head_dim=2,
        )
        self.manifest = Manifest(
            mapper_id="synthetic",
            compatibility=self.compat,
            topk=2,
            ridge_alpha=1.0,
            centered_inputs=True,
            rope_stripped_on_keys=True,
            source_layers_per_target={"k": {"0": [0, 1]}, "v": {"0": [0, 1]}},
        )
        weight = np.zeros((8, 4), dtype=np.float32)
        weight[:4] = np.eye(4, dtype=np.float32)
        weight[4:] = np.eye(4, dtype=np.float32)
        self.tensors = {
            f"target.0.{channel}.{part}": value
            for channel in ("k", "v")
            for part, value in (("W", weight), ("b", np.ones(4, dtype=np.float32)))
        }
        write_artifact(self.path, self.manifest, self.tensors)

    def test_full_head_apply_and_compatibility(self) -> None:
        artifact = MapperArtifact.open(self.path, self.compat)
        source = {
            0: torch.arange(8, dtype=torch.float32).reshape(2, 2, 2),
            1: torch.ones((2, 2, 2), dtype=torch.float32) * 2,
        }
        mapped = artifact.apply_layer(
            0, "k", source, device="cpu", dtype=torch.bfloat16
        )
        torch.testing.assert_close(mapped.float(), source[0] + source[1] + 1)

    def test_rejects_wrong_tensor_shape_and_precision(self) -> None:
        bad = dict(self.tensors)
        bad["target.0.k.W"] = np.zeros((4, 4), dtype=np.float32)
        save_file(bad, str(self.path / "weights.safetensors"))

        def digest(name: str) -> str:
            return hashlib.sha256((self.path / name).read_bytes()).hexdigest()

        (self.path / "SHA256SUMS").write_text(
            f"{digest('manifest.json')}  manifest.json\n"
            f"{digest('weights.safetensors')}  weights.safetensors\n"
        )
        with self.assertRaisesRegex(ValueError, "invalid mapper tensor"):
            MapperArtifact.open(self.path, self.compat)
        write_artifact(self.path, self.manifest, self.tensors)
        fp16 = CompatibilitySpec(**{**self.compat.__dict__, "precision": "fp16"})
        with self.assertRaisesRegex(ValueError, "precision"):
            MapperArtifact.open(self.path, fp16)

    def _refresh_checksums(self) -> None:
        names = ("manifest.json", "weights.safetensors")
        (self.path / "SHA256SUMS").write_text(
            "".join(
                f"{hashlib.sha256((self.path / name).read_bytes()).hexdigest()}  {name}\n"
                for name in names
            )
        )

    def test_rejects_non_finite_mapper_tensors(self) -> None:
        for name in self.tensors:
            for value in (np.nan, np.inf):
                with self.subTest(tensor=name, value=value):
                    bad = {key: tensor.copy() for key, tensor in self.tensors.items()}
                    bad[name].flat[0] = value
                    save_file(bad, str(self.path / "weights.safetensors"))
                    self._refresh_checksums()
                    with self.assertRaisesRegex(ValueError, "invalid mapper tensor"):
                        MapperArtifact.open(self.path, self.compat)

    def test_rejects_different_k_v_source_lists(self) -> None:
        manifest = self.manifest.to_dict()
        manifest["source_layers_per_target"]["v"]["0"] = [1, 0]
        (self.path / "manifest.json").write_text(json.dumps(manifest))
        self._refresh_checksums()
        with self.assertRaisesRegex(ValueError, "invalid shared source layers"):
            MapperArtifact.open(self.path, self.compat)

    def test_post_rope_source_is_unrotated_before_mapping(self) -> None:
        artifact = MapperArtifact.open(self.path, self.compat)
        positions = torch.arange(3)
        first = torch.arange(12, dtype=torch.float32).reshape(3, 2, 2) / 10
        second = torch.ones_like(first) / 4
        source = {
            0: (qwen3_rope(first, positions, theta=1_000_000), first),
            1: (qwen3_rope(second, positions, theta=1_000_000), second),
        }
        mapped = map_source_cache(
            artifact,
            source,
            positions,
            source_rope_theta=1_000_000,
            target_rope_theta=1_000_000,
            device="cpu",
            dtype=torch.float32,
        )
        expected_key = qwen3_rope(first + second + 1, positions, theta=1_000_000)
        torch.testing.assert_close(mapped[0][0], expected_key, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(mapped[0][1], first + second + 1)


if __name__ == "__main__":
    unittest.main()
