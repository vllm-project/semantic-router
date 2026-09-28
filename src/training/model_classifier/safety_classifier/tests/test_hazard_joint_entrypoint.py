"""The public training entrypoint must honor an explicit joint policy sidecar."""

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.training.model_classifier.safety_classifier import train_vela_hazard as train
from src.training.model_classifier.safety_classifier.vela_hazard import (
    score_predictions,
)


class SelectionReachedError(Exception):
    pass


@unittest.skipUnless(
    importlib.util.find_spec("numpy") and importlib.util.find_spec("sklearn"),
    "NumPy and scikit-learn are optional for dependency-light data checks",
)
class JointEntrypointTests(unittest.TestCase):
    def test_cli_dispatches_joint_policy_and_binds_sidecar(self):
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
        torch = SimpleNamespace(
            set_num_threads=Mock(),
            manual_seed=Mock(),
            optim=SimpleNamespace(AdamW=Mock()),
            version=SimpleNamespace(hip="fixture"),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            datasets = {}
            for split in ("train", "dev"):
                rows = [
                    {
                        "id": f"{split}-{label}",
                        "group_id": f"{split}-{label}",
                        "text": f"{split} fixture {label}",
                        "label": label,
                        "targets": [int(label == "unsafe")] * 2,
                        "label_mask": [1, 1],
                    }
                    for label in ("safe", "unsafe")
                ]
                path = root / f"{split}.jsonl"
                path.write_text("".join(json.dumps(row) + "\n" for row in rows))
                datasets[split] = rows
            sidecar = root / "groups.json"
            sidecar.write_text(json.dumps({"boundary": ["dev-safe"]}))
            scores = [[0.1, 0.1], [0.8, 0.8]]
            metrics = score_predictions(datasets["dev"], scores, ["a", "b"])
            output = root / "output"
            arguments = [
                "train",
                "--base",
                "fixture",
                "--base-id",
                "example/tiny-base",
                "--base-revision",
                "fixture",
                "--method",
                "full",
                "--contract",
                "fixture",
                "--max-length",
                "32",
                "--train",
                str(root / "train.jsonl"),
                "--dev",
                str(root / "dev.jsonl"),
                "--output",
                str(output),
                "--selection",
                "joint-fp-budget-macro-f1",
                "--selection-safe-groups",
                str(sidecar),
                "--evaluation-dtype",
                "float32",
            ]
            with patch.dict(sys.modules, {"torch": torch}), patch.object(
                sys, "argv", arguments
            ), patch.object(
                train,
                "load_trainable_model",
                return_value=(
                    model,
                    lambda *a, **k: {"input_ids": [1, 2]},
                    {"a": 0, "b": 1},
                    {0: "a", 1: "b"},
                ),
            ), patch.object(
                train, "initial_artifact_receipt", return_value={"base_files": {}}
            ), patch.object(
                train, "optimizer_groups", return_value=[]
            ), patch.object(
                train, "evaluate", return_value=(metrics, scores)
            ) as evaluate, patch.object(
                train,
                "select_joint_operating_point",
                wraps=train.select_joint_operating_point,
            ) as joint, patch.object(
                train,
                "select_operating_point",
                side_effect=AssertionError("wrong shared selector"),
            ), patch.object(
                train, "save_training_checkpoint", side_effect=SelectionReachedError
            ), patch.object(
                train.importlib.metadata, "version", return_value="fixture"
            ), self.assertRaises(
                SelectionReachedError
            ):
                train.main()
            self.assertEqual(
                joint.call_args.kwargs["safe_groups"], {"boundary": ["dev-safe"]}
            )
            self.assertEqual(evaluate.call_args.kwargs["dtype"], "float32")
            result = json.loads(
                (output / "dev-operating-point-step-0.json").read_text()
            )
            self.assertEqual(result["selection_score"], 1)
            self.assertEqual(
                [group["name"] for group in result["limits"]], ["all_safe", "boundary"]
            )
            receipt = json.loads((output / "run.json").read_text())
            self.assertEqual(
                receipt["selection_safe_groups"][0]["sha256"],
                hashlib.sha256(sidecar.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                receipt["arguments"]["selection"], "joint-fp-budget-macro-f1"
            )


if __name__ == "__main__":
    unittest.main()
