from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import torch

from d25.omni.model import checkpoint, tiny
from d25.omni.model.assemble import assemble


def digests(directory: Path, prefix: str = "") -> dict[str, str]:
    result = {}
    for key, tensor in checkpoint.iter_tensors(directory):
        name = checkpoint.canonical_name(key)
        if name is not None and name.startswith(prefix):
            result[name] = checkpoint.tensor_digest(tensor)
    return result


class CanonicalNameTest(unittest.TestCase):
    def test_layouts(self) -> None:
        cases = {
            "model.visual.blocks.0.attn.qkv.weight": "visual.blocks.0.attn.qkv.weight",
            "visual.merger.linear_fc1.bias": "visual.merger.linear_fc1.bias",
            "model.language_model.layers.3.mlp.up_proj.weight": "language_model.layers.3.mlp.up_proj.weight",
            "layers.0.input_layernorm.weight": "language_model.layers.0.input_layernorm.weight",
            "embed_tokens.weight": "language_model.embed_tokens.weight",
            "backbone.language_model.norm.weight": "language_model.norm.weight",
            "_orig_mod.backbone.visual.pos_embed.weight": "visual.pos_embed.weight",
            "readout.weight": "readout.weight",
        }
        for key, expected in cases.items():
            self.assertEqual(checkpoint.canonical_name(key), expected, key)
        for dropped in (
            "lm_head.weight",
            "mtp.fc.weight",
            "mtp.layers.0.mlp.up_proj.weight",
        ):
            self.assertIsNone(checkpoint.canonical_name(dropped))
        with self.assertRaises(ValueError):
            checkpoint.canonical_name("something.else")

    def test_components(self) -> None:
        self.assertEqual(
            checkpoint.component("visual.merger.norm.weight"), "vision_merger"
        )
        self.assertEqual(
            checkpoint.component("visual.blocks.1.mlp.linear_fc1.weight"),
            "vision_encoder",
        )
        self.assertEqual(
            checkpoint.component("visual.patch_embed.proj.weight"), "vision_encoder"
        )
        self.assertEqual(
            checkpoint.component("language_model.embed_tokens.weight"), "language"
        )


class AssembleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.paths = tiny.fixtures()
        cls.stock_vision = digests(cls.paths["stock"], "visual.")
        cls.tmp = Path(tempfile.mkdtemp(prefix="d25-omni-assemble-"))

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def assert_stock_vision(self, out: Path) -> None:
        self.assertEqual(digests(out, "visual."), self.stock_vision)

    def test_stock_init(self) -> None:
        out = self.paths["init_stock"]
        decision = checkpoint.read_decision_config(out)
        self.assertEqual(decision["format_id"], "d25-omni-code-readout-v1")
        self.assertEqual(decision["prompt"], "d25-vega")
        self.assertEqual(len(decision["codes"]), 255)
        self.assertEqual(decision["attention_mode"], "causal")
        self.assertEqual(decision["max_length"], 16_384)
        self.assertEqual(decision["images"]["max_pixels"], 1_638_400)
        provenance = decision["provenance"]
        self.assertEqual(provenance["init"]["kind"], "stock")
        self.assertEqual(provenance["vision"]["status"], "stock")
        self.assertTrue(provenance["vision"]["bit_equal_to_stock"])
        self.assertEqual(provenance["vision"]["tensors"], len(self.stock_vision))
        for name, sha in provenance["shards_sha256"].items():
            self.assertEqual(checkpoint.file_sha256(out / name), sha)
        self.assert_stock_vision(out)
        names = set(checkpoint.weight_files(out))
        self.assertFalse(any(n.startswith(("lm_head", "mtp", "model.")) for n in names))
        from transformers import Qwen3_5Config

        expected = checkpoint.expected_backbone_shapes(
            Qwen3_5Config.from_pretrained(str(out))
        )
        self.assertEqual(names, set(expected))
        ((_, head),) = checkpoint.iter_tensors(self.paths["stock"], {"lm_head.weight"})
        readout = checkpoint.load_readout(out)
        self.assertEqual(readout.dtype, torch.float32)
        self.assertTrue(torch.equal(readout, head[decision["token_ids"]].float()))
        processor = json.loads((out / "processor_config.json").read_text())
        self.assertEqual(
            processor["image_processor"]["size"]["longest_edge"], 1_638_400
        )

    def test_vega_unchanged_vision_is_kept(self) -> None:
        out = self.paths["init_vega"]
        decision = checkpoint.read_decision_config(out)
        provenance = decision["provenance"]
        self.assertEqual(provenance["init"]["kind"], "vega")
        self.assertEqual(provenance["vision"]["status"], "kept")
        self.assertEqual(
            provenance["init_provenance"], {"run": "tiny-vega", "init": "tiny-stock"}
        )
        self.assert_stock_vision(out)
        self.assertEqual(
            digests(out, "language_model."),
            digests(self.paths["vega"], "language_model."),
        )
        self.assertTrue(
            torch.equal(
                checkpoint.load_readout(out),
                checkpoint.load_readout(self.paths["vega"]),
            )
        )
        vega_shard = checkpoint.file_sha256(self.paths["vega"] / "model.safetensors")
        self.assertEqual(
            provenance["init"]["files_sha256"]["model.safetensors"], vega_shard
        )

    def test_changed_missing_partial_and_text_vision_are_reattached(self) -> None:
        merger = sum(1 for n in self.stock_vision if n.startswith("visual.merger."))
        cases = {
            "changed": ("changed", "backbone", 0, ["visual.blocks.0.attn.qkv.weight"]),
            "merger_changed": (
                "merger_changed",
                "backbone",
                0,
                ["visual.merger.linear_fc2.weight"],
            ),
            "missing": ("missing", "backbone", len(self.stock_vision), []),
            "partial": ("partial", "backbone", merger, []),
            "text": ("unchanged", "text", len(self.stock_vision), []),
        }
        for label, (vision, layout, missing, changed) in cases.items():
            with self.subTest(label):
                vega = tiny.write_vega(
                    self.tmp / f"vega-{label}",
                    self.paths["stock"],
                    vision=vision,
                    layout=layout,
                )
                out = self.tmp / f"out-{label}"
                decision = assemble(self.paths["stock"], out, vega, stock_pin="skip")
                status = decision["provenance"]["vision"]
                self.assertEqual(status["status"], "reattached")
                self.assertEqual(status["missing"], missing)
                self.assertEqual(status["changed"], changed)
                self.assert_stock_vision(out)
                self.assertEqual(
                    digests(out, "language_model."), digests(vega, "language_model.")
                )

    def test_refusals(self) -> None:
        busy = self.tmp / "busy"
        busy.mkdir()
        (busy / "x").write_text("x")
        with self.assertRaises(FileExistsError):
            assemble(self.paths["stock"], busy, stock_pin="skip")
        with self.assertRaisesRegex(ValueError, "not a file of"):
            assemble(self.paths["stock"], self.tmp / "pinned", stock_pin="verify")
        wrong = self.tmp / "vega-wrong-revision"
        shutil.copytree(self.paths["vega"], wrong)
        config = json.loads((wrong / "decision_config.json").read_text())
        config["revision"] = "0" * 40
        (wrong / "decision_config.json").write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "revision"):
            assemble(
                self.paths["stock"], self.tmp / "out-wrong", wrong, stock_pin="skip"
            )
        codes = self.tmp / "vega-wrong-codes"
        shutil.copytree(self.paths["vega"], codes)
        config = json.loads((codes / "decision_config.json").read_text())
        config["token_ids"] = list(reversed(config["token_ids"]))
        (codes / "decision_config.json").write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "answer codes"):
            assemble(
                self.paths["stock"], self.tmp / "out-codes", codes, stock_pin="skip"
            )


if __name__ == "__main__":
    unittest.main()
