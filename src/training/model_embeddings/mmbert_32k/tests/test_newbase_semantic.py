"""Absolute coordinates cannot be preserved by pairwise relations alone."""

import unittest
from types import SimpleNamespace

try:
    import torch

    from src.training.model_embeddings.mmbert_32k.newbase_semantic import (
        all_exit_anchor_loss,
        full_batch_relation_loss,
        pointwise_cosine_loss,
    )
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "requires torch")
class SemanticAnchorTest(unittest.TestCase):
    def test_layer_anchors_preserve_depth_identity_weights_and_detached_gradients(self):
        torch.manual_seed(23)
        weighted = [((3, 4), 0.2), ((3, 2), 0.3), ((22, 4), 0.4), ((22, 2), 0.1)]
        exits = SimpleNamespace(
            layers=[3, 22], dimensions=[4, 2], weighted=lambda: weighted
        )
        teacher = torch.randn(3, 2, 4, requires_grad=True)
        values = {
            key: torch.randn(3, key[1], requires_grad=True) for key, _ in weighted
        }
        actual = all_exit_anchor_loss(values, teacher, exits, target_layers=[22, 3])
        full_depth = 22
        expected = sum(
            weight
            * (
                1
                - torch.nn.functional.cosine_similarity(
                    values[key],
                    teacher.detach()[:, 0 if key[0] == full_depth else 1, : key[1]],
                    dim=-1,
                )
            ).mean()
            for key, weight in weighted
        )
        torch.testing.assert_close(actual, expected)
        expected_gradients = torch.autograd.grad(expected, tuple(values.values()))
        actual.backward()
        for value, expected_gradient in zip(
            values.values(), expected_gradients, strict=True
        ):
            torch.testing.assert_close(value.grad, expected_gradient)
            self.assertGreater(value.grad.norm().item(), 0)
        self.assertIsNone(teacher.grad)
        reordered = all_exit_anchor_loss(
            values, teacher.flip(1), exits, target_layers=[3, 22]
        )
        torch.testing.assert_close(reordered, actual)

    def test_layer_anchor_declarations_cannot_broadcast_or_substitute_missing_depths(
        self,
    ):
        exits = SimpleNamespace(
            layers=[3, 22],
            dimensions=[4],
            weighted=lambda: [((3, 4), 0.5), ((22, 4), 0.5)],
        )
        values = {(3, 4): torch.ones(2, 4), (22, 4): torch.ones(2, 4)}
        reference = torch.ones(2, 2, 4)
        for layers in ([], [3], [3, 3], [3, 21], [True, 22], [3.0, 22], "3,22"):
            with self.subTest(layers=layers), self.assertRaises(ValueError):
                all_exit_anchor_loss(values, reference, exits, target_layers=layers)
        for invalid in (reference[0], reference[:, :1], reference[:, :, :2]):
            with self.subTest(shape=invalid.shape), self.assertRaises(ValueError):
                all_exit_anchor_loss(values, invalid, exits, target_layers=[3, 22])
        with self.assertRaises(ValueError):
            all_exit_anchor_loss(values, reference, exits)
        # Equal layer targets reproduce the existing shared-matrix objective exactly.
        torch.testing.assert_close(
            all_exit_anchor_loss(values, reference, exits, target_layers=[3, 22]),
            all_exit_anchor_loss(values, reference[:, 0], exits),
        )

    def test_rotation_preserves_relations_but_anchor_restores_coordinates(self):
        teacher = torch.eye(3, requires_grad=True)
        student = teacher.detach().roll(1, dims=1).requires_grad_()
        ids = ["one", "two", "three"]
        components = {
            key: {"normalized_sha256": key, "parent_groups": [key]} for key in ids
        }
        relation = full_batch_relation_loss(student, teacher, ids, components)
        torch.testing.assert_close(relation, torch.tensor(0.0))
        anchor = pointwise_cosine_loss(student, teacher)
        torch.testing.assert_close(anchor, torch.tensor(1.0))
        anchor.backward()
        self.assertGreater(float(student.grad.abs().sum()), 0)
        self.assertIsNone(teacher.grad)
        self.assertLess(
            float(pointwise_cosine_loss(student - student.grad, teacher)), 1
        )

    def test_scale_invariance_and_no_hidden_width_division(self):
        student = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        teacher = torch.tensor([[0.0, 9.0, 0.0, 0.0]])
        torch.testing.assert_close(
            pointwise_cosine_loss(student, teacher), torch.tensor(1.0)
        )
        torch.testing.assert_close(
            pointwise_cosine_loss(student * 7, teacher * 3), torch.tensor(1.0)
        )
        for bad in (torch.zeros_like(teacher), teacher * float("nan"), teacher[:, :2]):
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                pointwise_cosine_loss(student, bad)

    def test_stronger_teacher_may_have_different_relation_width(self):
        torch.manual_seed(7)
        student = torch.randn(5, 3, requires_grad=True)
        teacher = torch.randn(5, 8, requires_grad=True)
        ids = list("abcde")
        components = {
            key: {"normalized_sha256": key, "parent_groups": [key]} for key in ids
        }
        loss = full_batch_relation_loss(student, teacher, ids, components)
        loss.backward()
        self.assertGreater(float(student.grad.abs().sum()), 0)
        self.assertIsNone(teacher.grad)


if __name__ == "__main__":
    unittest.main()
