from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

# Direct unittest discovery also runs this file without installing the package.
# ruff: noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from src.training.kv_mapper.artifact import (
    CompatibilitySpec,
    Manifest,
    _write_checksums,
    read_artifact,
    tensor_keys_for_layers,
    verify_compatibility,
    write_artifact,
)
from src.training.kv_mapper.mapper_id import make_mapper_id

COMPAT_MISMATCHES = (
    ("num_kv_heads", 4),
    ("head_dim", 64),
    ("precision", "bf16"),
    ("source_tp", 2),
    ("target_tp", 2),
    ("head_order", "other"),
    ("variant", "per_head"),
    ("source_revision", "mismatch"),
)


def _compat(**overrides) -> CompatibilitySpec:
    base = {
        "source_model": "Qwen/Qwen3-14B",
        "source_revision": "abc123",
        "target_model": "Qwen/Qwen3-32B",
        "target_revision": "def456",
        "variant": "full_head",
        "precision": "fp16",
        "source_tp": 1,
        "target_tp": 1,
        "head_order": "contiguous",
        "num_kv_heads": 2,
        "head_dim": 4,
    }
    base.update(overrides)
    return CompatibilitySpec(**base)


def _synthetic_manifest() -> Manifest:
    compat = _compat()
    mapper_id = make_mapper_id(
        pair_slug="qwen3-14b-32b",
        variant="full_head",
        precision="fp16",
        source_revision=compat.source_revision,
        target_revision=compat.target_revision,
        source_tp=1,
        target_tp=1,
        n_kv_heads=2,
    )
    return Manifest(
        mapper_id=mapper_id,
        compatibility=compat,
        topk=3,
        ridge_alpha=0.01,
        centered_inputs=True,
        rope_stripped_on_keys=True,
        source_layers_per_target={
            "k": {str(i): [0, 1, 2] for i in range(3)},
            "v": {str(i): [0, 1, 2] for i in range(3)},
        },
        calibration={"corpus": "synthetic", "seed": 0},
    )


def _synthetic_tensors(
    target_layers: int = 3, dx: int = 24, dy: int = 8
) -> dict[str, np.ndarray]:
    tensors: dict[str, np.ndarray] = {}
    for key in tensor_keys_for_layers(target_layers):
        if key.endswith(".W"):
            tensors[key] = np.linspace(0, 1, dx * dy, dtype=np.float32).reshape(dx, dy)
        else:
            tensors[key] = np.zeros(dy, dtype=np.float32)
    return tensors


class ArtifactContractTests(unittest.TestCase):
    def test_rejects_wrong_tensor_layout(self) -> None:
        manifest = _synthetic_manifest()
        tensors = _synthetic_tensors()
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "artifact"
            for bad in (
                {**tensors, "target.0.k.W": np.zeros((8, 8), dtype=np.float32)},
                {
                    name: value
                    for name, value in tensors.items()
                    if name != "target.0.k.b"
                },
            ):
                with self.subTest(keys=set(bad)), self.assertRaisesRegex(
                    ValueError, "mapper tensor"
                ):
                    write_artifact(out, manifest, bad)

            write_artifact(out, manifest, tensors)
            bad = dict(tensors)
            bad["target.0.k.W"] = np.zeros((8, 8), dtype=np.float32)
            save_file(bad, str(out / "weights.safetensors"))
            _write_checksums(out)
            with self.assertRaisesRegex(ValueError, "invalid mapper tensor"):
                read_artifact(out)

    def test_rejects_invalid_source_layer_selection(self) -> None:
        manifest = _synthetic_manifest()
        manifest.source_layers_per_target["v"]["0"] = [0, 1, 1]
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(
            ValueError, "shared source layers"
        ):
            write_artifact(Path(directory), manifest, _synthetic_tensors())

    def test_roundtrip_three_layers(self) -> None:
        manifest = _synthetic_manifest()
        tensors = _synthetic_tensors()
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / manifest.mapper_id
            write_artifact(out, manifest, tensors)
            got_manifest, got_tensors = read_artifact(out)
        self.assertEqual(got_manifest.mapper_id, manifest.mapper_id)
        self.assertEqual(got_manifest.topk, 3)
        for key, arr in tensors.items():
            np.testing.assert_allclose(got_tensors[key], arr)

    def test_verify_compatibility_rejects_mismatch(self) -> None:
        manifest = _synthetic_manifest()
        for field, value in COMPAT_MISMATCHES:
            with self.subTest(field=field):
                deployment = _compat(**{field: value})
                with self.assertRaisesRegex(ValueError, field):
                    verify_compatibility(manifest, deployment)

    def test_checksum_tamper_detected(self) -> None:
        manifest = _synthetic_manifest()
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "artifact"
            write_artifact(out, manifest, _synthetic_tensors())
            manifest_path = out / "manifest.json"
            data = json.loads(manifest_path.read_text())
            data["topk"] = 99
            manifest_path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "checksum"):
                read_artifact(out)

    def test_missing_or_duplicate_checksum_entries_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "artifact"
            write_artifact(out, _synthetic_manifest(), _synthetic_tensors())
            lines = (out / "SHA256SUMS").read_text().splitlines()
            for contents in (
                "",
                lines[0] + "\n",
                lines[1] + "\n",
                "\n".join([*lines, lines[0]]) + "\n",
            ):
                with self.subTest(contents=contents):
                    (out / "SHA256SUMS").write_text(contents)
                    with self.assertRaisesRegex(ValueError, "checksum"):
                        read_artifact(out)

    def test_precision_aliases_are_canonical(self) -> None:
        manifest = _synthetic_manifest()
        verify_compatibility(manifest, _compat(precision="float16"))
        self.assertEqual(manifest.compatibility.precision, "fp16")
        self.assertEqual(_compat(precision="bfloat16").precision, "bf16")
        self.assertIn(
            "-bf16-",
            make_mapper_id(
                pair_slug="pair",
                variant="full_head",
                precision="bfloat16",
                source_revision="abc123",
                target_revision="def456",
                source_tp=1,
                target_tp=1,
                n_kv_heads=8,
            ),
        )
        with self.assertRaisesRegex(ValueError, "unsupported mapper precision"):
            _compat(precision="float128")


if __name__ == "__main__":
    unittest.main()
