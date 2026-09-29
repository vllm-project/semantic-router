"""bf16z codec: exact round trip for every dtype the packages use (numpy + zstandard)."""

from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    import zstandard  # noqa: F401

    HAVE = True
except ImportError:
    HAVE = False

if HAVE:
    from v2.release.runtime import bf16z


def write_safetensors(path: Path, tensors: dict, pad: int = 0) -> bytes:
    header, blobs, offset = {"__metadata__": {"format": "pt"}}, [], 0
    for name, (dtype, shape, data) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + len(data)],
        }
        blobs.append(data)
        offset += len(data)
    encoded = json.dumps(header).encode() + b" " * pad
    raw = struct.pack("<Q", len(encoded)) + encoded + b"".join(blobs)
    path.write_bytes(raw)
    return raw


def sample(rng) -> dict:
    weights = rng.normal(0, 0.02, size=(64, 96)).astype(np.float32)
    bf16 = (weights.view(np.uint32) >> 16).astype(np.uint16).tobytes()
    return {
        "layers.0.mlp.up_proj.weight": ("BF16", [64, 96], bf16),
        "embed_tokens.weight": (
            "F32",
            [50, 32],
            rng.normal(0, 1, size=(50, 32)).astype(np.float32).tobytes(),
        ),
        "norm.weight": ("F32", [32], np.ones(32, np.float32).tobytes()),
        "half": ("F16", [7, 3], rng.normal(size=(7, 3)).astype(np.float16).tobytes()),
        "ids": ("I64", [5], np.arange(5, dtype=np.int64).tobytes()),
        "mask": ("BOOL", [9], bytes([1, 0, 1, 1, 0, 0, 1, 0, 1])),
        "bytes": ("U8", [4], b"\x00\x7f\x80\xff"),
        "empty": ("F32", [0, 4], b""),
    }


@unittest.skipUnless(HAVE, "numpy and zstandard are needed")
class RoundTripTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.rng = np.random.default_rng(20260930)

    def tearDown(self):
        self.tmp.cleanup()

    def test_every_dtype_restores_bit_for_bit(self):
        for level in (1, 19):
            source = self.root / f"m{level}.safetensors"
            raw = write_safetensors(source, sample(self.rng), pad=5)
            packed = self.root / f"m{level}.safetensors.bf16z"
            entry = bf16z.compress_file(source, packed, level=level)
            restored = self.root / f"out{level}.safetensors"
            done = bf16z.decompress_file(packed, restored)
            self.assertEqual(restored.read_bytes(), raw)
            self.assertEqual(done["sha256"], entry["source_sha256"])
            self.assertEqual(bf16z.decompress_file(packed)["bytes"], len(raw))
            self.assertFalse((self.root / f"out{level}.safetensors.pending").exists())

    def test_planes_split_bf16_and_fp32_and_header_is_readable(self):
        source = self.root / "m.safetensors"
        write_safetensors(source, sample(self.rng))
        packed = self.root / "m.safetensors.bf16z"
        bf16z.compress_file(source, packed)
        index, _, _ = bf16z.read_index(packed)
        planes = {row["name"]: len(row["planes"]) for row in index["tensors"]}
        self.assertEqual(planes["layers.0.mlp.up_proj.weight"], 2)
        self.assertEqual(planes["embed_tokens.weight"], 4)
        self.assertEqual(planes["mask"], 1)
        self.assertEqual(bf16z.original_header(packed)["norm.weight"]["shape"], [32])
        codecs = {p["codec"] for row in index["tensors"] for p in row["planes"]}
        self.assertLessEqual(codecs, {"zstd", "raw"})

    def test_corruption_is_refused(self):
        source = self.root / "m.safetensors"
        write_safetensors(source, sample(self.rng))
        packed = self.root / "m.safetensors.bf16z"
        bf16z.compress_file(source, packed)
        data = bytearray(packed.read_bytes())
        data[-3] ^= 0x01
        packed.write_bytes(bytes(data))
        with self.assertRaises(Exception):
            bf16z.decompress_file(packed, self.root / "bad.safetensors")
        self.assertFalse((self.root / "bad.safetensors").exists())

    def test_holes_are_refused(self):
        source = self.root / "m.safetensors"
        header = {"a": {"dtype": "F32", "shape": [1], "data_offsets": [4, 8]}}
        encoded = json.dumps(header).encode()
        source.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"\0" * 8)
        with self.assertRaises(ValueError):
            bf16z.compress_file(source, self.root / "m.safetensors.bf16z")

    def test_checkpoint_compress_verify_and_materialize(self):
        ckpt = self.root / "ckpt"
        (ckpt / "backbone").mkdir(parents=True)
        shards = {}
        for i in (1, 2):
            name = f"model-0000{i}-of-00002.safetensors"
            shards[name] = write_safetensors(ckpt / "backbone" / name, sample(self.rng))
        (ckpt / "backbone" / "config.json").write_text("{}")
        (ckpt / "decision_config.json").write_text('{"a": 1}')
        (ckpt / "decision_head.safetensors").write_bytes(b"head")
        out = self.root / "ckpt-bf16z"
        result = bf16z.compress_checkpoint(ckpt, out, level=3)
        self.assertEqual(
            sorted(p.name for p in (out / "backbone").iterdir()),
            ["config.json"] + sorted(f"{n}.bf16z" for n in shards),
        )
        entries = {
            name: {"sha256": e["sha256"], "restored_sha256": e["source_sha256"]}
            for name, e in result["files"].items()
        }
        self.assertTrue(bf16z.verify_dir(out, entries)["passed"])
        manifest = {
            "identity": {"model_sha256": "f" * 64},
            "model_files": sorted(
                [
                    "backbone/config.json",
                    "decision_config.json",
                    "decision_head.safetensors",
                ]
                + list(entries)
            ),
            "storage": {"codec": bf16z.FORMAT, "files": entries},
        }
        cache = self.root / "cache"
        restored = bf16z.materialize(out, manifest, cache)
        for name, raw in shards.items():
            self.assertEqual((restored / "backbone" / name).read_bytes(), raw)
        self.assertEqual((restored / "decision_head.safetensors").read_bytes(), b"head")
        self.assertEqual(bf16z.materialize(out, manifest, cache), restored)
        bad = json.loads(json.dumps(manifest))
        bad["identity"]["model_sha256"] = "e" * 64
        first = sorted(entries)[0]
        bad["storage"]["files"][first]["restored_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            bf16z.materialize(out, bad, cache)
        self.assertEqual(
            sorted(p.name for p in (cache / "bf16z").iterdir()), ["f" * 64]
        )


@unittest.skipUnless(HAVE, "numpy and zstandard are needed")
class PipelineTest(unittest.TestCase):
    """Builder, layout and runtime hooks on a tiny full checkpoint."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        rng = np.random.default_rng(7)
        self.ckpt = self.root / "ckpt"
        (self.ckpt / "backbone").mkdir(parents=True)
        for i in (1, 2):
            write_safetensors(
                self.ckpt / "backbone" / f"model-0000{i}-of-00002.safetensors",
                sample(rng),
            )
        (self.ckpt / "backbone" / "model.safetensors.index.json").write_text("{}")
        (self.ckpt / "backbone" / "config.json").write_text("{}")
        (self.ckpt / "decision_config.json").write_text('{"checkpoint_format": "full"}')
        write_safetensors(self.ckpt / "decision_head.safetensors", sample(rng))
        (self.ckpt / "tokenizer.json").write_text("{}")
        (self.ckpt / "combination_manifest.json").write_text("{}")
        self.out = self.root / "ckpt-bf16z"
        result = bf16z.compress_checkpoint(self.ckpt, self.out, level=3)
        receipt = bf16z.make_receipt(result, 3, self.ckpt)
        self.receipt = self.root / "bf16z.json"
        self.receipt.write_text(json.dumps(receipt))
        report = bf16z.verify_dir(self.out, receipt["files"])
        self.verify = self.root / "verify.json"
        self.verify.write_text(json.dumps(report))

    def tearDown(self):
        self.tmp.cleanup()

    def spec(self) -> dict:
        from v2.release.layout import sha_file

        return {
            "bf16z": {
                "receipt": str(self.receipt),
                "sha256": sha_file(self.receipt),
                "verify": str(self.verify),
                "verify_sha256": sha_file(self.verify),
            }
        }

    def test_identity_from_hashes_equals_the_real_fingerprint(self):
        from training.model.infer import checkpoint_fingerprint
        from v2.release.build import bf16z_files, identity_from_hashes

        restored, storage = bf16z_files(self.spec(), self.out)
        self.assertEqual(
            identity_from_hashes(restored), checkpoint_fingerprint(self.ckpt, None)
        )
        self.assertEqual(len(storage["files"]), 2)
        self.assertEqual(storage["codec"], bf16z.FORMAT)

    def test_tampered_compressed_file_is_refused(self):
        from v2.release.build import bf16z_files

        target = next((self.out / "backbone").glob("*.bf16z"))
        data = bytearray(target.read_bytes())
        data[-1] ^= 1
        target.write_bytes(bytes(data))
        with self.assertRaises(ValueError):
            bf16z_files(self.spec(), self.out)

    def test_counts_and_pointer_read_compressed_headers(self):
        from v2.release import layout
        from v2.release.runtime.api import _tensor_count

        files = layout.select_model_files("qwen-full", self.out)
        self.assertIn("backbone/model-00001-of-00002.safetensors.bf16z", files)
        groups = layout.parameter_files("qwen-full", files)
        plain = layout.select_model_files("qwen-full", self.ckpt)
        plain_groups = layout.parameter_files("qwen-full", plain)
        count = lambda root, names: sum(
            layout.safetensors_count(root / n) for n in names
        )
        self.assertEqual(
            count(self.out, groups["backbone"]),
            count(self.ckpt, plain_groups["backbone"]),
        )
        self.assertEqual(
            sum(_tensor_count(self.out / n) for n in groups["backbone"]),
            count(self.ckpt, plain_groups["backbone"]),
        )
        pointer = layout.pointer(
            "qwen-full",
            "DEV2.0-27B",
            files,
            calibration=None,
            base=None,
            max_input_tokens=8,
        )
        self.assertEqual(pointer["weight_storage"]["codec"], "bf16z/1")
        plain_pointer = layout.pointer(
            "qwen-full",
            "DEV2.0-27B",
            plain,
            calibration=None,
            base=None,
            max_input_tokens=8,
        )
        self.assertNotIn("weight_storage", plain_pointer)


if __name__ == "__main__":
    unittest.main()
