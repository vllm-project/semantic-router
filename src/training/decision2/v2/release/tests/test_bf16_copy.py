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


if __name__ == "__main__":
    unittest.main()
