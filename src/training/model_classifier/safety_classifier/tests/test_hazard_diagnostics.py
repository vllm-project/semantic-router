# Torch remains optional for local dependency-light checks.
# ruff: noqa: PLC0415

import importlib.util
import unittest

from src.training.model_classifier.safety_classifier.vela_hazard import masked_loss
from src.training.model_classifier.safety_classifier.vela_hazard_diagnostics import (
    SupervisionDiagnostics,
    logit_loss_gradient,
)


@unittest.skipUnless(
    importlib.util.find_spec("torch"),
    "Torch optional locally; required on training node",
)
class GradientDiagnosticsTests(unittest.TestCase):
    def test_observed_mean_gradient_matches_autograd(self):
        import torch

        logits = torch.tensor([[0.2, -0.4, 0.8], [1.2, 0.1, -0.7]], requires_grad=True)
        targets = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        mask = torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        loss = masked_loss(logits, targets, mask) * (2 / 8)
        loss.backward()
        torch.testing.assert_close(
            logit_loss_gradient(logits, targets, mask, 8), logits.grad
        )
        self.assertEqual(logits.grad[0, 1:].tolist(), [0.0, 0.0])

    def test_source_counts_and_actual_head_gradient(self):
        import torch

        model = torch.nn.Module()
        model.classifier = torch.nn.Linear(4, 2)
        logits = model.classifier(torch.ones(2, 4))
        targets = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
        mask = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
        rows = [
            {"id": "p", "source": "partial", "targets": [1, 0], "label_mask": [1, 0]},
            {"id": "s", "source": "safe", "targets": [0, 0], "label_mask": [1, 1]},
        ]
        diagnostics = SupervisionDiagnostics(["a", "b"])
        masked_loss(logits, targets, mask).backward()
        diagnostics.observe(rows, logits, targets, mask, 2)
        diagnostics.observe_head_gradients(model)
        saved = diagnostics.snapshot()
        self.assertEqual(
            saved["sources"]["partial"]["observed_count_distribution"], {1: 1}
        )
        self.assertEqual(saved["sources"]["partial"]["per_label"]["b"]["unknown"], 1)
        self.assertEqual(saved["sources"]["safe"]["per_label"]["b"]["negative"], 1)
        self.assertEqual(saved["gradient_steps"], 1)
        self.assertAlmostEqual(
            saved["head_gradient_norm_sums"]["classifier.weight"]["a"],
            model.classifier.weight.grad[0].norm().item(),
        )


if __name__ == "__main__":
    unittest.main()
