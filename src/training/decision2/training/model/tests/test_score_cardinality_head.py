"""The 9B residual leaves the shared System One readout unchanged at step zero."""

import pytest
import torch

from training.model.decision_model import CandidateHead
from training.model.score_cardinality_head import ScoreCardinalityHead


def test_zero_step_matches_shared_head_and_is_order_invariant() -> None:
    torch.manual_seed(42)
    shared = CandidateHead(16, 8)
    torch.manual_seed(42)
    treatment = ScoreCardinalityHead(16, 8)
    candidates = torch.randn(3, 5, 16)
    query = torch.randn(3, 16)
    mask = torch.tensor(
        [
            [True, True, True, False, False],
            [True] * 5,
            [True, True, False, False, False],
        ]
    )
    types = torch.tensor([2, 2, 1])
    levels = torch.tensor([[2, 0, 1, 0, 0], [4, 1, 0, 3, 2], [0, 1, 0, 0, 0]])
    baseline = shared(candidates, query)
    observed = treatment(candidates, query, types, mask, levels)
    torch.testing.assert_close(observed, baseline, rtol=0, atol=0)

    with torch.no_grad():
        treatment.score_bias[1, 1] = 0.75
    changed = treatment(candidates, query, types, mask, levels)
    torch.testing.assert_close(changed[0, 2], baseline[0, 2] + 0.75)
    torch.testing.assert_close(changed[1], baseline[1], rtol=0, atol=0)
    torch.testing.assert_close(changed[2], baseline[2], rtol=0, atol=0)


def test_rejects_noncontiguous_score_levels() -> None:
    treatment = ScoreCardinalityHead(8, 4)
    with pytest.raises(ValueError, match="enumerate"):
        treatment(
            torch.zeros(1, 3, 8),
            torch.zeros(1, 8),
            torch.tensor([2]),
            torch.ones(1, 3, dtype=torch.bool),
            torch.tensor([[0, 1, 1]]),
        )
