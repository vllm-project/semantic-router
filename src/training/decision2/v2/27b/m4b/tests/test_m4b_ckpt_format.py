import importlib
import json
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

ckpt_format = importlib.import_module("v2.27b.m4b.ckpt_format")
M4B = Path(__file__).resolve().parents[1]


def safetensors(path: Path, shapes: dict[str, list[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = {
        name: {"dtype": "F32", "shape": shape, "data_offsets": [0, 0]}
        for name, shape in shapes.items()
    }
    header["__metadata__"] = {"format": "pt"}
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw)


def full_checkpoint(root: Path) -> Path:
    ckpt = root / "full"
    (ckpt / "backbone").mkdir(parents=True)
    (ckpt / "decision_config.json").write_text(
        json.dumps({"checkpoint_format": "full"})
    )
    (ckpt / "backbone/model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "a": "model-1.safetensors",
                    "b": "model-2.safetensors",
                    "c": "model-1.safetensors",
                }
            }
        )
    )
    safetensors(ckpt / "backbone/model-1.safetensors", {"a": [1000, 30], "c": [7]})
    safetensors(ckpt / "backbone/model-2.safetensors", {"b": [2, 3, 4]})
    safetensors(ckpt / "decision_head.safetensors", {"w": [256, 10], "b": [10]})
    return ckpt


def lora_checkpoint(root: Path) -> tuple[Path, Path]:
    base = root / "base"
    safetensors(
        base / "model-1.safetensors",
        {
            "model.layers.0.w": [100, 20],
            "lm_head.weight": [50, 20],
            "model.visual.x": [9],
        },
    )
    safetensors(base / "model-2.safetensors", {"model.embed_tokens.weight": [50, 20]})
    ckpt = root / "lora"
    (ckpt / "decision_config.json").parent.mkdir(parents=True)
    (ckpt / "decision_config.json").write_text(
        json.dumps(
            {
                "checkpoint_format": "peft-lora/1",
                "lora": {
                    "rank": 256,
                    "source_kind": "posttrained",
                    "source_fingerprint": {
                        "files_sha256": {
                            "model-1.safetensors": "x",
                            "model-2.safetensors": "y",
                            "config.json": "z",
                        }
                    },
                },
            }
        )
    )
    safetensors(
        ckpt / "adapter/adapter_model.safetensors", {"A": [256, 20], "B": [100, 256]}
    )
    safetensors(ckpt / "decision_head.safetensors", {"w": [256, 10], "b": [10]})
    return ckpt, base


class CheckpointFormatTest(unittest.TestCase):
    def test_check_accepts_only_the_named_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            full = full_checkpoint(Path(tmp))
            lora, _ = lora_checkpoint(Path(tmp))
            self.assertEqual(ckpt_format.check(full, "full"), "full")
            self.assertEqual(ckpt_format.check(lora, "peft-lora/1"), "peft-lora/1")
            with self.assertRaises(SystemExit):
                ckpt_format.check(lora, "full")
            with self.assertRaises(SystemExit):
                ckpt_format.check(full, "peft-lora/1")
            with self.assertRaises(ValueError):
                ckpt_format.check(full, "lora")
            (lora / "adapter/adapter_model.safetensors").unlink()
            with self.assertRaises(SystemExit):
                ckpt_format.check(lora, "peft-lora/1")

    def test_full_count_and_source_string(self):
        with tempfile.TemporaryDirectory() as tmp:
            full = full_checkpoint(Path(tmp))
            loaded, source = ckpt_format.params(full, "full")
            self.assertEqual(loaded, 30000 + 7 + 24 + 2570)
            self.assertEqual(
                source,
                "full FP32 checkpoint: text backbone 30,031 + head 2,570 (safetensors headers)",
            )

    def test_lora_count_uses_the_base_text_tensors(self):
        with tempfile.TemporaryDirectory() as tmp:
            lora, base = lora_checkpoint(Path(tmp))
            loaded, source = ckpt_format.params(lora, "peft-lora/1", base)
            self.assertEqual(loaded, (2000 + 1000) + (5120 + 25600) + 2570)
            self.assertEqual(
                source,
                "pinned base text backbone 3,000 + LoRA rank 256 30,720 + head 2,570 (safetensors headers)",
            )
            with self.assertRaises(ValueError):
                ckpt_format.params(lora, "peft-lora/1")

    def test_drivers_name_the_format(self):
        readout = (M4B / "run_readout.sh").read_text(encoding="utf-8")
        formal = (M4B / "run_formal.sh").read_text(encoding="utf-8")
        for text in (readout, formal):
            self.assertIn("CHECKPOINT_FORMAT=${CHECKPOINT_FORMAT:-full}", text)
            self.assertIn("v2.27b.m4b.ckpt_format", text)
        self.assertIn(
            "full) LOADED_PARAMETERS=${LOADED_PARAMETERS:-25629863936} ;;", formal
        )
        self.assertIn('"checkpoint_format": checkpoint_format,', formal)
        for name in ("run_readout.sh", "run_formal.sh"):
            result = subprocess.run(
                ["bash", "-n", str(M4B / name)], capture_output=True, text=True
            )
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
