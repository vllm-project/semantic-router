"""Compare score mathematics and gradients against installed official losses."""

import unittest

try:
    import torch
    from sentence_transformers.cross_encoder.losses import LambdaLoss, NDCGLoss2PPScheme
    from sentence_transformers.losses import CoSENTLoss
    from transformers import BatchEncoding

    from src.training.model_embeddings.mmbert_32k.newbase_objectives import (
        cosent_loss,
        lambda_loss,
    )
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "requires sentence-transformers for reference parity")
class OfficialLossParityTest(unittest.TestCase):
    def test_lambda_single_query_value_and_gradient_match_official(self):
        class Scores(torch.nn.Module):
            num_labels = 1
            device = torch.device("cpu")

            def __init__(self):
                super().__init__()
                self.scores = torch.nn.Parameter(torch.tensor([0.3, -0.4, 0.8, 0.1]))

            def tokenizer(self, pairs, **_):
                return BatchEncoding(
                    {"indices": torch.tensor([int(right) for _, right in pairs])}
                )

            def forward(self, indices):
                return (self.scores[indices, None],)

        reference = Scores()
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
        official = LambdaLoss(
            reference,
            weighting_scheme=NDCGLoss2PPScheme(mu=10),
            k=10,
            sigma=1,
            eps=1e-10,
            reduction_log="binary",
            activation_fn=torch.nn.Identity(),
        )
        expected = official([["query"], [["0", "1", "2", "3"]]], [labels])
        scores = reference.scores.detach().clone().unsqueeze(0).requires_grad_()
        mask = torch.ones_like(scores, dtype=torch.bool)
        actual = lambda_loss(scores, labels.unsqueeze(0), mask, mask)
        actual.backward()
        expected.backward()
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)
        torch.testing.assert_close(
            scores.grad[0], reference.scores.grad, atol=1e-7, rtol=1e-6
        )

    def test_cosent_value_and_gradient_match_official(self):
        class Embeddings(torch.nn.Module):
            def forward(self, features):
                return {"sentence_embedding": features["values"]}

        left = torch.tensor([[1.0, 0.2], [0.3, 1.0], [0.2, 0.5]], requires_grad=True)
        right = torch.tensor([[0.1, 1.0], [0.2, 1.0], [1.0, 0.0]])
        labels = torch.tensor([0.0, 1.0, 0.5])
        scores = torch.nn.functional.cosine_similarity(left, right)
        actual = cosent_loss(scores, labels)
        expected = CoSENTLoss(Embeddings(), scale=20)(
            [{"values": left}, {"values": right}], labels
        )
        actual_grad = torch.autograd.grad(actual, left, retain_graph=True)[0]
        expected_grad = torch.autograd.grad(expected, left)[0]
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(actual_grad, expected_grad, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
