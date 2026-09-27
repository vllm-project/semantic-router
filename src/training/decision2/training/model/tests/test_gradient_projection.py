import copy
import importlib.util
import math
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec("torch"), "CPU torch unavailable")
class ProjectionTest(unittest.TestCase):
    def test_opt_in_trainer_contract_freezes_full466_without_changing_default(self):
        from training.model import train, train_gradient_projection
        from training.model.data import file_sha256
        from training.model.gradient_projection_preflight import CONTROL_TRAIN_SHA256

        self.assertEqual(file_sha256(Path(train.__file__)), CONTROL_TRAIN_SHA256)

        flags = [
            "train",
            "--model-path",
            "pinned-source",
            "--base-revision",
            train.TYPED_HEAD_SOURCE_REVISION,
            "--train",
            "train.jsonl",
            "--select",
            "select.jsonl",
            "--cal",
            "cal.jsonl",
            "--output",
            "fresh-output",
            "--objective",
            "ce_brier",
            "--epochs",
            "1",
            "--microbatch",
            "1",
            "--accumulation",
            "16",
            "--eval-batch",
            "2",
            "--max-length",
            "8192",
            "--backbone-lr",
            "2e-5",
            "--head-lr",
            "2e-4",
            "--save-every",
            "64",
        ]
        with patch.object(sys, "argv", flags), patch.object(
            train_gradient_projection.torch.cuda, "is_available", return_value=True
        ), patch.object(
            train_gradient_projection.torch.cuda, "is_bf16_supported", return_value=True
        ):
            default = train_gradient_projection.parse_args()
            with self.assertRaisesRegex(ValueError, "requires --gradient-projection"):
                train_gradient_projection.validate_args(default)
            with patch.object(sys, "argv", flags + ["--gradient-projection"]):
                treatment = train_gradient_projection.parse_args()
            train_gradient_projection.validate_args(treatment)
            treatment.backbone_lr = 2.1e-5
            with self.assertRaisesRegex(ValueError, "full466 control"):
                train_gradient_projection.validate_args(treatment)

    def test_three_type_original_reference_order_and_norm_match(self):
        import torch

        from training.model.gradient_projection import project_backbone_gradients

        vectors = {
            "choice": [torch.tensor([1.0, 0.0])],
            "noul": [torch.tensor([-0.5, 0.5])],
            "score": [torch.tensor([0.0, -0.25])],
        }
        counts = dict.fromkeys(vectors, 1)
        ordinary, ordinary_info = project_backbone_gradients(
            vectors, counts, enabled=False
        )
        projected, info = project_backbone_gradients(vectors, counts, enabled=True)
        expected_ordinary = torch.tensor([0.5, 0.25])
        torch.testing.assert_close(ordinary[0], expected_ordinary, rtol=0, atol=0)
        self.assertEqual(ordinary_info["projected_pairs"], 0)
        self.assertEqual(info["task_counts"], counts)
        self.assertGreater(info["projected_pairs"], 0)
        self.assertLess(info["pairwise_cosines"]["choice_noul"], 0)
        self.assertAlmostEqual(
            projected[0].norm().item(), expected_ordinary.norm().item(), places=6
        )
        self.assertFalse(torch.allclose(projected[0], expected_ordinary))
        # Independently calculate each task's fixed-order projections against
        # the original, unmodified other-task vectors.
        original = {key: value[0].tolist() for key, value in vectors.items()}
        order = ("choice", "noul", "score")
        projected_python = [0.0, 0.0]
        for task in order:
            current = original[task][:]
            for reference in order:
                if task == reference:
                    continue
                other = original[reference]
                dot = sum(a * b for a, b in zip(current, other))
                norm_sq = sum(b * b for b in other)
                coeff = min(0.0, dot) / (norm_sq + 1e-12)
                current = [a - coeff * b for a, b in zip(current, other)]
            projected_python = [a + b for a, b in zip(projected_python, current)]
        scale = expected_ordinary.norm().item() / math.sqrt(
            sum(value * value for value in projected_python)
        )
        torch.testing.assert_close(
            projected[0],
            torch.tensor([value * scale for value in projected_python]),
            rtol=1e-6,
            atol=1e-6,
        )
        # Originals remain intact for every reference projection.
        torch.testing.assert_close(vectors["choice"][0], torch.tensor([1.0, 0.0]))

    def test_accumulator_disabled_matches_ordinary_and_head_stays_ordinary(self):
        import torch

        from training.model.decision_model import collate
        from training.model.gradient_projection import TaskGradientAccumulator
        from training.model.loss import per_example_loss

        class TinyDecision(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = torch.nn.Linear(3, 4)
                self.head = torch.nn.Linear(4, 2)

            def forward(self, input_ids, candidate_mask, **unused):
                encoded = torch.nn.functional.one_hot(
                    input_ids[:, 0] % 3, num_classes=3
                ).float()
                return self.head(torch.tanh(self.backbone(encoded))).masked_fill(
                    ~candidate_mask, -float("inf")
                )

        def items():
            result = []
            for index in range(16):
                result.append(
                    {
                        "id": f"toy-{index}",
                        "ids": [index % 3, 1, 2],
                        "candidate_positions": [0, 1],
                        "query_position": 2,
                        "keys": ["0", "1"],
                        "label": index % 2,
                        "task_type": ("choice", "noul", "score")[index % 3],
                        "teacher_probs": None,
                    }
                )
            return result

        def backward(model, item):
            batch = collate([item], 0)
            logits = model(**batch)
            loss = (
                per_example_loss(
                    logits,
                    batch["labels"],
                    batch["candidate_mask"],
                    objective="ce_brier",
                    brier_weight=0.5,
                )["total"].sum()
                / 16
            )
            loss.backward()

        torch.manual_seed(20260926)
        initial = TinyDecision()
        ordinary = copy.deepcopy(initial)
        grouped = copy.deepcopy(initial)
        projected = copy.deepcopy(initial)
        for item in items():
            backward(ordinary, item)
        grouped_acc = TaskGradientAccumulator(grouped)
        projected_acc = TaskGradientAccumulator(projected)
        for item in items():
            grouped.zero_grad(set_to_none=True)
            backward(grouped, item)
            grouped_acc.capture(item["task_type"])
            projected.zero_grad(set_to_none=True)
            backward(projected, item)
            projected_acc.capture(item["task_type"])
        grouped.zero_grad(set_to_none=True)
        projected.zero_grad(set_to_none=True)
        disabled = grouped_acc.finalize(enabled=False)
        enabled = projected_acc.finalize(enabled=True)
        self.assertEqual(disabled["task_counts"], {"choice": 6, "noul": 5, "score": 5})
        for standard, reconstructed in zip(ordinary.parameters(), grouped.parameters()):
            torch.testing.assert_close(
                standard.grad, reconstructed.grad, rtol=0, atol=1e-6
            )
        for standard, treated in zip(
            ordinary.head.parameters(), projected.head.parameters()
        ):
            torch.testing.assert_close(standard.grad, treated.grad, rtol=0, atol=1e-6)
        self.assertTrue(math.isfinite(enabled["final_backbone_norm"]))
        self.assertAlmostEqual(
            enabled["ordinary_backbone_norm"], enabled["final_backbone_norm"], places=5
        )

    def test_zero_norm_falls_back_to_ordinary_and_nonfinite_fails(self):
        import torch

        from training.model.gradient_projection import project_backbone_gradients

        vectors = {
            "choice": [torch.tensor([1.0, 0.0])],
            "noul": [torch.tensor([-1.0, 0.0])],
            "score": [torch.tensor([0.0, 0.0])],
        }
        result, info = project_backbone_gradients(
            vectors, {"choice": 1, "noul": 1, "score": 0}, enabled=True
        )
        torch.testing.assert_close(result[0], torch.zeros(2), rtol=0, atol=0)
        self.assertEqual(info["norm_scale"], 1.0)
        vectors["noul"][0][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            project_backbone_gradients(
                vectors, {"choice": 1, "noul": 1, "score": 0}, enabled=True
            )

    def test_tiny_reference_uses_norm_threshold_not_squared_threshold(self):
        import torch

        from training.model.gradient_projection import project_backbone_gradients

        vectors = {
            "choice": [torch.tensor([1.0, 0.0])],
            "noul": [torch.tensor([-1e-8, 0.0])],
            "score": [torch.tensor([0.0, 1.0])],
        }
        _, info = project_backbone_gradients(
            vectors, dict.fromkeys(vectors, 1), enabled=True
        )
        self.assertGreater(info["projected_pairs"], 0)
        self.assertGreater(info["task_norms"]["noul"], 1e-12)
        self.assertLess(info["task_norms"]["noul"] ** 2, 1e-12)


if __name__ == "__main__":
    unittest.main()
