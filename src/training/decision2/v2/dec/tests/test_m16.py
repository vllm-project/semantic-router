"""CPU tests for decoder M16: the release / arm interpolation (m16_interp) and the finalist pick (m16_rules)."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parents[1]


def load(name: str):
    spec = importlib.util.spec_from_file_location(
        name, HERE / "ops" / "m16" / f"{name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def checkpoint(root: Path, scale: float, shards: dict[str, list[str]], **meta) -> Path:
    (root / "backbone").mkdir(parents=True)
    config = {
        "architecture": "toy",
        "prompt_version": "p1",
        "head_dim": 2,
        "checkpoint_format": "full",
        "max_options": 4,
        "full_training_source": {"kind": "decision1", "source_name": "s"},
        "soup": {"members": []},
    }
    config.update(meta)
    (root / "decision_config.json").write_text(json.dumps(config))
    (root / "tokenizer.json").write_text('{"v": 1}')
    for shard, keys in shards.items():
        save_file(
            {k: torch.full((2, 3), scale, dtype=torch.bfloat16) for k in keys},
            str(root / "backbone" / shard),
        )
    save_file({"w": torch.full((2,), scale)}, str(root / "decision_head.safetensors"))
    return root


class InterpTest(unittest.TestCase):
    def setUp(self) -> None:
        self.m = load("m16_interp")
        self.tmp = Path(tempfile.mkdtemp())
        self.release = checkpoint(
            self.tmp / "r", 1.0, {"model.safetensors": ["a", "b"]}
        )
        self.arm = checkpoint(
            self.tmp / "x",
            3.0,
            {
                "model-00001-of-00002.safetensors": ["a"],
                "model-00002-of-00002.safetensors": ["b"],
            },
        )

    def test_lineage_passes_across_shard_layouts(self) -> None:
        check = self.m.lineage(self.release, self.arm)
        self.assertTrue(check["pass"], check["reasons"])
        self.assertEqual(check["backbone_tensors"], 2)

    def test_lineage_fails_on_another_source(self) -> None:
        other = checkpoint(
            self.tmp / "o",
            3.0,
            {"model.safetensors": ["a", "b"]},
            full_training_source={"kind": "decision1", "source_name": "other"},
        )
        check = self.m.lineage(self.release, other)
        self.assertFalse(check["pass"])
        self.assertIn(
            "decision_config differs in full_training_source", check["reasons"]
        )

    def test_lineage_fails_on_missing_tensor(self) -> None:
        other = checkpoint(self.tmp / "o", 3.0, {"model.safetensors": ["a"]})
        self.assertFalse(self.m.lineage(self.release, other)["pass"])

    def test_build_mixes_every_tensor_in_fp32(self) -> None:
        out = self.tmp / "w"
        result = self.m.build(self.release, self.arm, 0.25, out)
        self.assertEqual(result["alpha"], 0.25)
        first = load_file(str(out / "backbone" / "model-00001-of-00002.safetensors"))
        second = load_file(str(out / "backbone" / "model-00002-of-00002.safetensors"))
        self.assertEqual(first["a"].dtype, torch.float32)
        self.assertTrue(torch.allclose(first["a"], torch.full((2, 3), 1.5)))
        self.assertTrue(torch.allclose(second["b"], torch.full((2, 3), 1.5)))
        head = load_file(str(out / "decision_head.safetensors"))
        self.assertTrue(torch.allclose(head["w"], torch.full((2,), 1.5)))
        meta = json.loads((out / "decision_config.json").read_text())
        self.assertEqual(meta["interpolation"]["alpha"], 0.25)
        self.assertEqual(
            meta["initialization"], "linear-interpolation-of-full-checkpoints"
        )
        self.assertNotIn("soup", meta)
        self.assertFalse((self.tmp / "w.pending").exists())

    def test_build_refuses_alpha_outside_the_open_interval(self) -> None:
        with self.assertRaises(ValueError):
            self.m.build(self.release, self.arm, 1.0, self.tmp / "w")


def row(
    point: str, delta: float | None, eligible: bool = True, gain: bool = False
) -> dict:
    r = load("m16_rules")
    line, alpha = r.line_alpha(point)
    return {
        "point": point,
        "line": line,
        "alpha": alpha,
        "eligible": eligible,
        "ib_dev": {"transfer_delta": delta},
        "htdev2": {"verdict": "GAIN" if gain else "TIE"},
    }


class PickTest(unittest.TestCase):
    def setUp(self) -> None:
        self.r = load("m16_rules")

    def test_second_finalist_comes_from_another_line(self) -> None:
        rows = [
            row("2b-RAUP-a75", 0.12),
            row("2b-RAUP-a50", 0.10),
            row("2b-RA-a25", 0.03),
            row("2b-RASD-a75", 0.20, eligible=False),
        ]
        self.assertEqual(self.r.pick(rows), ["2b-RAUP-a75", "2b-RA-a25"])

    def test_same_line_when_no_other_line_passes(self) -> None:
        rows = [row("4b-LHA10UP-a25", 0.01), row("4b-LHA10UP-a50", 0.02)]
        self.assertEqual(self.r.pick(rows), ["4b-LHA10UP-a50", "4b-LHA10UP-a25"])

    def test_ties_prefer_gain_then_smaller_alpha(self) -> None:
        rows = [
            row("08b-RA-a75", 0.05),
            row("08b-RA-a50", 0.05),
            row("08b-RASD-a75", 0.05, gain=True),
        ]
        self.assertEqual(self.r.pick(rows), ["08b-RASD-a75", "08b-RA-a50"])

    def test_no_eligible_point(self) -> None:
        self.assertEqual(self.r.pick([row("08b-RA-a25", 0.3, eligible=False)]), [])

    def test_point_names(self) -> None:
        self.assertEqual(self.r.line_alpha("4b-LHA10SD-a75"), ("4b-LHA10SD", 0.75))
        with self.assertRaises(ValueError):
            self.r.line_alpha("4b-LHA10SD")


if __name__ == "__main__":
    unittest.main()
