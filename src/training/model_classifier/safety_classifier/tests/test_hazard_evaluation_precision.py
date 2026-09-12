"""Hazard checkpoint selection must use the explicitly recorded precision."""

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.training.model_classifier.safety_classifier import train_vela_hazard as train


class EvaluationReachedError(Exception):
    pass


class HazardEvaluationPrecisionTests(unittest.TestCase):
    def test_entrypoint_records_and_passes_precision_without_changing_default(self):
        parameter = SimpleNamespace(requires_grad=True, numel=lambda: 1)
        model = SimpleNamespace(
            config=SimpleNamespace(
                problem_type="multi_label_classification",
                max_position_embeddings=32,
                to_dict=lambda: {"problem_type": "multi_label_classification"},
            ),
            parameters=lambda: iter([parameter]),
            gradient_checkpointing_enable=Mock(),
            to=Mock(),
            train=Mock(),
        )
        fake_torch = SimpleNamespace(
            set_num_threads=Mock(),
            manual_seed=Mock(),
            optim=SimpleNamespace(AdamW=Mock()),
            version=SimpleNamespace(hip="fixture"),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            adapter = root / "adapter"
            adapter.mkdir()
            (adapter / "adapter_model.safetensors").write_bytes(b"fixture")
            for partition in ["train", "dev"]:
                rows = [
                    {
                        "id": f"{partition}-{label}",
                        "group_id": f"{partition}-{label}",
                        "text": f"{partition} fixture {label}",
                        "label": label,
                        "targets": [int(label == "unsafe")],
                        "label_mask": [1],
                    }
                    for label in ["safe", "unsafe"]
                ]
                (root / f"{partition}.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in rows)
                )
            arguments = [
                "train",
                "--base",
                "fixture",
                "--base-revision",
                "fixture",
                "--adapter",
                str(adapter),
                "--contract",
                "fixture",
                "--train",
                str(root / "train.jsonl"),
                "--dev",
                str(root / "dev.jsonl"),
                "--max-length",
                "32",
            ]
            for explicit in [None, "bfloat16", "float32"]:
                output = root / str(explicit)
                argv = [*arguments, "--output", str(output)]
                if explicit is not None:
                    argv += ["--evaluation-dtype", explicit]
                with patch.dict(sys.modules, {"torch": fake_torch}), patch.object(
                    sys, "argv", argv
                ), patch.object(
                    train,
                    "load_model",
                    return_value=(
                        model,
                        lambda *args, **kwargs: {"input_ids": [1, 2]},
                        {"risk": 0},
                        {0: "risk"},
                    ),
                ), patch.object(
                    train, "evaluate", side_effect=EvaluationReachedError
                ) as evaluate, patch.object(
                    train.importlib.metadata, "version", return_value="fixture"
                ), self.assertRaises(
                    EvaluationReachedError
                ):
                    train.main()
                expected = explicit or "bfloat16"
                self.assertEqual(evaluate.call_args.kwargs["dtype"], expected)
                receipt = json.loads((output / "run.json").read_text())
                self.assertEqual(receipt["evaluation_dtype"], expected)
                self.assertIn("BF16 autocast", receipt["precision"])
            with patch.dict(sys.modules, {"torch": fake_torch}), patch.object(
                sys,
                "argv",
                [
                    *arguments,
                    "--output",
                    str(root / "invalid"),
                    "--evaluation-dtype",
                    "float16",
                ],
            ), patch.object(train, "load_model") as load, redirect_stderr(
                io.StringIO()
            ):
                with self.assertRaises(SystemExit) as failure:
                    train.main()
                self.assertEqual(failure.exception.code, 2)
                load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
