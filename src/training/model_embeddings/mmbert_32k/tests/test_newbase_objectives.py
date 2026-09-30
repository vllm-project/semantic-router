"""Numerical contracts for relevance masks and the single loss interventions."""

import unittest

try:
    import torch

    from src.training.model_embeddings.mmbert_32k.newbase_data import retrieval_masks
    from src.training.model_embeddings.mmbert_32k.newbase_objectives import (
        cosent_loss,
        lambda_loss,
        multi_positive_loss,
        order_distillation,
        ranking_terms,
        relational_cosine_loss,
    )
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "requires torch")
class NewBaseObjectivesTest(unittest.TestCase):
    def test_cosine_relations_allow_different_model_dimensions(self):
        torch.manual_seed(49)
        left = torch.randn(3, 4, requires_grad=True)
        right = torch.randn(4, 4, requires_grad=True)
        teacher_left = torch.randn(3, 7, requires_grad=True)
        teacher_right = torch.randn(4, 7, requires_grad=True)
        valid = torch.tensor(
            [[True, False, True, False], [False, True, False, False], [False] * 4]
        )
        actual = relational_cosine_loss(left, right, teacher_left, teacher_right, valid)
        rows = []
        for i in range(len(left)):
            terms = [
                (
                    torch.nn.functional.cosine_similarity(left[i], right[j], dim=0)
                    - torch.nn.functional.cosine_similarity(
                        teacher_left[i], teacher_right[j], dim=0
                    )
                ).square()
                for j in range(len(right))
                if valid[i, j]
            ]
            if terms:
                rows.append(torch.stack(terms).mean())
        torch.testing.assert_close(actual, torch.stack(rows).mean())
        actual.backward()
        self.assertIsNone(teacher_left.grad)
        self.assertIsNone(teacher_right.grad)
        self.assertGreater(left.grad[:2].norm().item(), 0)
        self.assertEqual(left.grad[2].norm().item(), 0)
        self.assertEqual(right.grad[3].norm().item(), 0)
        for invalid in (teacher_right[:, :6], teacher_right[:3], teacher_right[:, :0]):
            with self.assertRaisesRegex(ValueError, "geometry"):
                relational_cosine_loss(left, right, teacher_left, invalid, valid)
        invalid = teacher_right.detach().clone()
        invalid[3, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            relational_cosine_loss(left, right, teacher_left, invalid, valid)

    def test_shared_parent_preference_has_gradient_without_absolute_gold(self):
        components = {
            key: {"normalized_sha256": key, "parent_groups": ["matched-background"]}
            for key in ("positive", "alternative")
        }
        row = {
            "positive_component_ids": ["positive"],
            "judged_negative_component_ids": [],
            "unjudged_component_ids": ["alternative"],
            "candidate_component_ids": list(components),
        }
        scores = torch.tensor([[0.2, 0.5]], requires_grad=True)
        positive, valid = retrieval_masks([row], list(components), components)
        before = multi_positive_loss(
            scores, torch.tensor(positive), torch.tensor(valid)
        )
        before.backward()
        self.assertEqual(before.item(), 0)
        torch.testing.assert_close(scores.grad, torch.zeros_like(scores))
        scores.grad.zero_()
        row["contrastive_preference_component_ids"] = ["alternative"]
        positive, valid = retrieval_masks([row], list(components), components)
        after = multi_positive_loss(scores, torch.tensor(positive), torch.tensor(valid))
        after.backward()
        self.assertGreater(after.item(), 0)
        self.assertLess(scores.grad[0, 0].item(), 0)
        self.assertGreater(scores.grad[0, 1].item(), 0)
        scores.grad.zero_()
        labels = torch.tensor([[1.0, -1.0]])
        terms = ranking_terms(
            scores,
            labels,
            torch.tensor(valid),
            torch.zeros_like(labels, dtype=torch.bool),
        )
        (terms["pairwise"] + terms["bce"]).backward()
        torch.testing.assert_close(scores.grad, torch.zeros_like(scores))

    def test_multiple_positives_are_marginalized_and_masked_scores_have_no_gradient(
        self,
    ):
        scores = torch.tensor([[2.0, 1.0, 9.0, -1.0]], requires_grad=True)
        positive = torch.tensor([[True, True, False, False]])
        valid = torch.tensor([[True, True, False, True]])
        actual = multi_positive_loss(scores, positive, valid)
        expected = torch.logsumexp(scores[0, [0, 1, 3]], 0) - torch.logsumexp(
            scores[0, :2], 0
        )
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertEqual(scores.grad[0, 2].item(), 0)
        self.assertLess(scores.grad[0, 0].item(), 0)
        self.assertGreater(scores.grad[0, 3].item(), 0)

    def test_distillation_detaches_teacher_and_ignores_invalid_alternatives(self):
        student = torch.tensor([[0.0, 1.0, 100.0]], requires_grad=True)
        teacher = torch.tensor([[2.0, 0.0, -100.0]], requires_grad=True)
        valid = torch.tensor([[True, True, False]])
        loss = order_distillation(student, teacher, valid)
        loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertEqual(student.grad[0, 2].item(), 0)
        self.assertLess(student.grad[0, 0].item(), 0)
        single = order_distillation(
            student, teacher, torch.tensor([[True, False, False]])
        )
        self.assertEqual(single.item(), 0)

    def test_unknown_has_no_judged_loss_but_explicit_preference_can_train_it(self):
        scores = torch.tensor([[0.2, -0.5, 5.0]], requires_grad=True)
        labels = torch.tensor([[1.0, 0.0, -1.0]])
        valid, judged = torch.ones_like(labels, dtype=torch.bool), labels >= 0
        terms = ranking_terms(scores, labels, valid, judged)
        (terms["pairwise"] + 0.2 * terms["bce"]).backward(retain_graph=True)
        self.assertEqual(scores.grad[0, 2].item(), 0)
        scores.grad.zero_()
        terms["unjudged_preference"].backward()
        self.assertGreater(scores.grad[0, 2].item(), 0)
        with self.assertRaisesRegex(ValueError, "Unjudged"):
            ranking_terms(scores, labels, valid, valid)

    def test_filtered_unknown_keeps_soft_targets_without_hard_preference(self):
        scores = torch.tensor([[0.2, -0.5, 5.0, 0.8]], requires_grad=True)
        labels = torch.tensor([[1.0, 0.0, -1.0, -1.0]])
        valid, judged = torch.ones_like(labels, dtype=torch.bool), labels >= 0
        legacy = ranking_terms(scores, labels, valid, judged)
        explicit = ranking_terms(
            scores, labels, valid, judged, preference_mask=labels == -1
        )
        for name in legacy:
            torch.testing.assert_close(legacy[name], explicit[name], atol=0, rtol=0)
        filtered = ranking_terms(
            scores,
            labels,
            valid,
            judged,
            preference_mask=torch.tensor([[False, False, False, True]]),
        )
        for name in ("bce", "pairwise"):
            torch.testing.assert_close(legacy[name], filtered[name], atol=0, rtol=0)
        sum(filtered.values()).backward(retain_graph=True)
        self.assertEqual(scores.grad[0, 2].item(), 0)
        self.assertGreater(scores.grad[0, 3].item(), 0)
        scores.grad.zero_()
        order_distillation(scores, torch.zeros_like(scores), valid).backward()
        self.assertNotEqual(scores.grad[0, 2].item(), 0)
        for mask in (valid, valid.float(), valid[:, :2]):
            with self.assertRaisesRegex(ValueError, "preferences"):
                ranking_terms(scores, labels, valid, judged, preference_mask=mask)

    def test_lambda_unknown_removal_preserves_discounts_and_gradients(self):
        scores = torch.tensor([[0.1, 100.0, -0.2, 0.5]], requires_grad=True)
        labels = torch.tensor([[1.0, -1.0, 0.0, 0.0]])
        actual = lambda_loss(
            scores, labels, torch.ones_like(labels, dtype=torch.bool), labels >= 0
        )
        compact = scores.detach()[:, [0, 2, 3]].requires_grad_()
        expected = lambda_loss(
            compact,
            labels[:, [0, 2, 3]],
            torch.ones_like(compact, dtype=torch.bool),
            torch.ones_like(compact, dtype=torch.bool),
        )
        actual.backward()
        expected.backward()
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(
            scores.grad[:, [0, 2, 3]], compact.grad, atol=0, rtol=0
        )
        self.assertEqual(scores.grad[0, 1].item(), 0)
        self.assertEqual(
            lambda_loss(
                compact,
                labels[:, [0, 2, 3]],
                torch.ones_like(compact, dtype=torch.bool),
                torch.ones_like(compact, dtype=torch.bool),
                k=1,
            ).item(),
            0,
        )

    def test_cosent_respects_order_not_absolute_label_scale(self):
        scores = torch.tensor([0.1, 0.5, -0.2], requires_grad=True)
        labels = torch.tensor([0.0, 1.0, 0.0])
        loss = cosent_loss(scores, labels)
        torch.testing.assert_close(
            loss, cosent_loss(scores, labels * 5 + 2), atol=0, rtol=0
        )
        loss.backward()
        self.assertLess(scores.grad[1].item(), 0)

    def test_query_weight_is_independent_of_number_of_judged_candidates(self):
        scores = torch.tensor([[0.2, -0.1, 0.0], [0.3, 0.4, 0.8]])
        labels = torch.tensor([[1.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
        valid, judged = torch.ones_like(labels, dtype=torch.bool), labels >= 0
        combined = ranking_terms(scores, labels, valid, judged)
        singles = [
            ranking_terms(
                scores[i : i + 1],
                labels[i : i + 1],
                valid[i : i + 1],
                judged[i : i + 1],
            )
            for i in range(2)
        ]
        for key in ("pairwise", "bce"):
            torch.testing.assert_close(
                combined[key], (singles[0][key] + singles[1][key]) / 2
            )


if __name__ == "__main__":
    unittest.main()
