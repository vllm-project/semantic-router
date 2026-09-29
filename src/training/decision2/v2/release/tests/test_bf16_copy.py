import json
import tempfile
import unittest
from pathlib import Path

from v2.release.bf16_copy import storage_dtype

try:
    import torch
    from safetensors.torch import load_file, save_file
except ImportError:
    torch = None


class StorageDtypeTest(unittest.TestCase):
    def test_only_linear_projection_weights_become_bf16(self):
        for name in (
            "layers.0.self_attn.q_proj.weight",
            "layers.3.self_attn.o_proj.weight",
            "layers.1.linear_attn.in_proj_qkv.weight",
            "layers.1.linear_attn.in_proj_z.weight",
            "layers.1.linear_attn.in_proj_a.weight",
            "layers.1.linear_attn.in_proj_b.weight",
            "layers.1.linear_attn.out_proj.weight",
            "layers.2.mlp.gate_proj.weight",
            "layers.2.mlp.up_proj.weight",
            "layers.31.mlp.down_proj.weight",
        ):
            self.assertEqual(storage_dtype(name, [8, 4], "F32"), "BF16", name)
        for name, shape in (
            ("embed_tokens.weight", [16, 4]),
            ("norm.weight", [4]),
            ("layers.0.input_layernorm.weight", [4]),
            ("layers.0.self_attn.q_norm.weight", [4]),
            ("layers.1.linear_attn.A_log", [2]),
            ("layers.1.linear_attn.dt_bias", [2]),
            ("layers.1.linear_attn.conv1d.weight", [8, 1, 4]),
            ("layers.1.linear_attn.norm.weight", [4]),
            ("layers.0.self_attn.q_proj.bias", [8]),
            ("query.weight", [4, 4]),
        ):
            self.assertEqual(storage_dtype(name, shape, "F32"), "F32", name)

    def test_refuses_a_source_that_is_not_fp32(self):
        with self.assertRaises(ValueError):
            storage_dtype("layers.0.mlp.up_proj.weight", [8, 4], "BF16")


@unittest.skipIf(torch is None, "needs torch and safetensors")
class ConvertTest(unittest.TestCase):
    def test_casts_projections_and_copies_everything_else_exactly(self):
        from v2.release.bf16_copy import convert

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src"
            (src / "backbone").mkdir(parents=True)
            gen = torch.Generator().manual_seed(0)
            shard_name = "model-00001-of-00001.safetensors"
            shard = {
                "embed_tokens.weight": torch.randn(16, 4, generator=gen),
                "layers.0.mlp.up_proj.weight": torch.randn(8, 4, generator=gen),
                "layers.0.input_layernorm.weight": torch.randn(4, generator=gen),
                "layers.0.linear_attn.conv1d.weight": torch.randn(
                    8, 1, 4, generator=gen
                ),
            }
            save_file(
                shard, str(src / "backbone" / shard_name), metadata={"format": "pt"}
            )
            (src / "backbone" / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "metadata": {"total_size": 0},
                        "weight_map": {name: shard_name for name in shard},
                    }
                )
            )
            (src / "backbone" / "config.json").write_text("{}\n")
            save_file(
                {"query.weight": torch.randn(4, 4, generator=gen)},
                str(src / "decision_head.safetensors"),
            )
            (src / "decision_config.json").write_text("{}\n")
            out = Path(tmp) / "out"

            result = convert(src, out)

            stored = load_file(str(out / "backbone" / shard_name))
            projection = "layers.0.mlp.up_proj.weight"
            self.assertEqual(stored[projection].dtype, torch.bfloat16)
            self.assertTrue(
                torch.equal(stored[projection], shard[projection].to(torch.bfloat16))
            )
            for name in (
                "embed_tokens.weight",
                "layers.0.input_layernorm.weight",
                "layers.0.linear_attn.conv1d.weight",
            ):
                self.assertEqual(stored[name].dtype, torch.float32)
                self.assertTrue(torch.equal(stored[name], shard[name]))
            self.assertEqual(result["numel_by_storage_dtype"], {"BF16": 32, "F32": 100})
            index = json.loads(
                (out / "backbone" / "model.safetensors.index.json").read_text()
            )
            self.assertEqual(index["metadata"]["total_size"], 32 * 2 + 100 * 4)
            self.assertEqual(
                (out / "decision_head.safetensors").read_bytes(),
                (src / "decision_head.safetensors").read_bytes(),
            )
            self.assertFalse((Path(tmp) / "out.pending").exists())
            with self.assertRaises(FileExistsError):
                convert(src, out)

    def test_single_file_backbone(self):
        from v2.release.bf16_copy import convert

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src"
            (src / "backbone").mkdir(parents=True)
            gen = torch.Generator().manual_seed(1)
            weights = {
                "embed_tokens.weight": torch.randn(16, 4, generator=gen),
                "layers.0.self_attn.q_proj.weight": torch.randn(8, 4, generator=gen),
                "layers.0.self_attn.q_norm.weight": torch.randn(4, generator=gen),
            }
            save_file(weights, str(src / "backbone" / "model.safetensors"))
            (src / "backbone" / "config.json").write_text("{}\n")
            (src / "score_bias.json").write_text("{}\n")
            out = Path(tmp) / "out"

            result = convert(src, out)

            self.assertEqual(result["layout"], "single")
            stored = load_file(str(out / "backbone" / "model.safetensors"))
            q = "layers.0.self_attn.q_proj.weight"
            self.assertEqual(stored[q].dtype, torch.bfloat16)
            self.assertTrue(torch.equal(stored[q], weights[q].to(torch.bfloat16)))
            self.assertTrue(
                torch.equal(
                    stored["layers.0.self_attn.q_norm.weight"],
                    weights["layers.0.self_attn.q_norm.weight"],
                )
            )
            self.assertFalse(
                (out / "backbone" / "model.safetensors.index.json").exists()
            )
            self.assertTrue(result["files"]["score_bias.json"]["verbatim"])
            self.assertFalse(result["files"]["backbone/model.safetensors"]["verbatim"])


class LayoutTest(unittest.TestCase):
    def test_layout_and_ignore_only_backbone_weights(self):
        from v2.release.bf16_copy import backbone_layout, backbone_weights

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp)
            (src / "backbone").mkdir()
            with self.assertRaises(FileNotFoundError):
                backbone_layout(src)
            (src / "backbone" / "model.safetensors").write_bytes(b"x")
            self.assertEqual(
                backbone_layout(src)[:2], ("single", ["model.safetensors"])
            )
            ignore = backbone_weights(src)
            names = [
                "model.safetensors",
                "config.json",
                "model-00001-of-00002.safetensors",
            ]
            self.assertEqual(
                ignore(str(src / "backbone"), names),
                {"model.safetensors", "model-00001-of-00002.safetensors"},
            )
            self.assertEqual(ignore(str(src), names), set())


class RebindTest(unittest.TestCase):
    def test_same_offsets_new_hash_with_provenance(self):
        from training.model.score_bias import validate_score_bias
        from v2.release.bf16_copy import rebind_score_bias

        report = {
            "format": "dev2-score-bias-v1",
            "model_sha256": "a" * 64,
            "offsets": {"5": [0.1, 0.2, 0.0, -0.1, -0.2]},
            "fit": {"lam": 0.5},
        }
        rebound = rebind_score_bias(report, "b" * 64, "f" * 64, "a" * 64)
        self.assertEqual(rebound["offsets"], report["offsets"])
        self.assertEqual(report["model_sha256"], "a" * 64)
        self.assertEqual(rebound["fit"]["bf16_rebind"]["source_model_sha256"], "a" * 64)
        self.assertEqual(rebound["fit"]["lam"], 0.5)
        validate_score_bias(rebound, "b" * 64)
        with self.assertRaises(ValueError):
            rebind_score_bias(report, "b" * 64, "f" * 64, "c" * 64)


class BuilderLinkTest(unittest.TestCase):
    """build.verify_bf16_copy + check_scored_score_bias with a BF16 package identity."""

    def setUp(self):
        from v2.release import layout

        self.scratch = tempfile.TemporaryDirectory()
        root = Path(self.scratch.name)
        self.checkpoint = root / "ckpt"
        (self.checkpoint / "backbone").mkdir(parents=True)
        (self.checkpoint / "backbone" / "model.safetensors").write_bytes(b"w")
        (self.checkpoint / "decision_head.safetensors").write_bytes(b"h")
        files = {
            name: {"sha256": layout.sha_file(self.checkpoint / name)}
            for name in ("backbone/model.safetensors", "decision_head.safetensors")
        }
        self.receipt = root / "bf16-copy.json"
        self.receipt.write_text(
            json.dumps(
                {
                    "schema": "dev2-release-bf16-copy/1",
                    "source_model_sha256": "a" * 64,
                    "model_sha256": "b" * 64,
                    "files": files,
                    "score_bias": {"sha256": "s" * 64},
                }
            )
        )
        self.spec = {
            "expected_identity": {"model_sha256": "b" * 64},
            "score_bias": {"path": "unused", "sha256": "s" * 64},
            "bf16_copy": {
                "receipt": str(self.receipt),
                "sha256": layout.sha_file(self.receipt),
            },
        }
        self.native = {
            "model_sha256": "a" * 64,
            "score_bias_sha256": "d" * 64,
            "score_bias": {"file_sha256": "d" * 64, "offsets": {"5": [0.1]}},
        }
        self.bias = {"sha256": "s" * 64, "offsets": {"5": [0.1]}}

    def tearDown(self):
        self.scratch.cleanup()

    def test_link_binds_scored_identity(self):
        from v2.release.build import check_scored_score_bias, verify_bf16_copy

        link = verify_bf16_copy(self.spec, self.checkpoint)
        self.assertEqual(link["source_model_sha256"], "a" * 64)
        self.assertEqual(
            check_scored_score_bias(self.spec, self.native, self.bias), "d" * 64
        )
        without = {k: v for k, v in self.spec.items() if k != "bf16_copy"}
        self.assertIsNone(verify_bf16_copy(without, self.checkpoint))
        with self.assertRaises(ValueError):
            check_scored_score_bias(without, self.native, self.bias)

    def test_link_refuses_mismatches(self):
        from v2.release.build import verify_bf16_copy

        cases = (
            lambda s: s["bf16_copy"].update(sha256="0" * 64),
            lambda s: s["expected_identity"].update(model_sha256="c" * 64),
            lambda s: s["score_bias"].update(sha256="t" * 64),
            lambda s: (self.checkpoint / "extra.json").write_text("{}"),
        )
        for change in cases:
            with self.subTest():
                spec = json.loads(json.dumps(self.spec))
                change(spec)
                with self.assertRaises(ValueError):
                    verify_bf16_copy(spec, self.checkpoint)
                (self.checkpoint / "extra.json").unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
