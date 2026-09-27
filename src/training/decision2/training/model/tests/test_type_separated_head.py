"""CPU-only structural checks for the prospective typed-head ablation."""

from __future__ import annotations

import pytest
import torch

from training.model.type_separated_head import TypeSeparatedCandidateHead


def test_mixed_types_are_routed_and_padded_options_are_masked() -> None:
    torch.manual_seed(27)
    head = TypeSeparatedCandidateHead(hidden_size=16, head_dim=8)
    candidates = torch.randn(3, 4, 16, requires_grad=True)
    query = torch.randn(3, 16, requires_grad=True)
    type_ids = torch.tensor([0, 1, 2])
    mask = torch.tensor(
        [
            [True, True, True, True],
            [True, True, False, False],
            [True, True, True, False],
        ]
    )
    logits = head(candidates, query, type_ids, mask)
    assert logits.shape == (3, 4)
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits[mask]).all()
    assert torch.isneginf(logits[~mask]).all()
    for row, type_id in enumerate(type_ids.tolist()):
        direct = head.heads[type_id](candidates[row : row + 1], query[row : row + 1])
        torch.testing.assert_close(logits[row, mask[row]], direct[0, mask[row]])
    logits.log_softmax(-1).gather(1, torch.tensor([[0], [1], [2]])).sum().backward()
    assert torch.isfinite(candidates.grad[mask]).all()
    assert torch.all(candidates.grad[~mask] == 0)
    assert torch.isfinite(query.grad).all()
    assert all(parameter.grad is not None for parameter in head.parameters())


def test_type_heads_are_independent_and_dynamic_option_order_is_preserved() -> None:
    torch.manual_seed(28)
    head = TypeSeparatedCandidateHead(hidden_size=12, head_dim=6)
    candidates = torch.randn(1, 5, 12)
    query = torch.randn(1, 12)
    mask = torch.ones((1, 5), dtype=torch.bool)
    source = head(candidates, query, torch.tensor([0]), mask)
    order = torch.tensor([4, 2, 0, 3, 1])
    reordered = head(candidates[:, order], query, torch.tensor([0]), mask)
    torch.testing.assert_close(reordered, source[:, order])
    with torch.no_grad():
        head.heads[2].scalar.weight.add_(1.0)
    unchanged_choice = head(candidates, query, torch.tensor([0]), mask)
    torch.testing.assert_close(unchanged_choice, source)


def test_native_cardinality_limits() -> None:
    head = TypeSeparatedCandidateHead(hidden_size=8, head_dim=4)
    choice = head(
        torch.zeros(1, 255, 8),
        torch.zeros(1, 8),
        torch.tensor([0]),
        torch.ones(1, 255, dtype=torch.bool),
    )
    score = head(
        torch.zeros(1, 10, 8),
        torch.zeros(1, 8),
        torch.tensor([2]),
        torch.ones(1, 10, dtype=torch.bool),
    )
    assert choice.shape == (1, 255)
    assert score.shape == (1, 10)
    with pytest.raises(ValueError, match="Noul needs two"):
        head(
            torch.zeros(1, 3, 8),
            torch.zeros(1, 8),
            torch.tensor([1]),
            torch.ones(1, 3, dtype=torch.bool),
        )
    with pytest.raises(ValueError, match="Score needs"):
        head(
            torch.zeros(1, 11, 8),
            torch.zeros(1, 8),
            torch.tensor([2]),
            torch.ones(1, 11, dtype=torch.bool),
        )


@pytest.mark.parametrize("bad_id", [-1, 3])
def test_invalid_type_rejected(bad_id: int) -> None:
    head = TypeSeparatedCandidateHead(hidden_size=8, head_dim=4)
    with pytest.raises(ValueError, match="Unsupported System One"):
        head(
            torch.zeros(1, 2, 8),
            torch.zeros(1, 8),
            torch.tensor([bad_id]),
            torch.ones(1, 2, dtype=torch.bool),
        )
