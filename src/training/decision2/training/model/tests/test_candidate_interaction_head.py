"""CPU contracts for the prospective 0.6B candidate-set readout."""

from __future__ import annotations

import pytest
import torch

from training.model.candidate_interaction_head import CandidateInteractionHead
from training.model.decision_model import CandidateHead, collate, encode


def test_zero_step_matches_the_shared_control_and_noul_remains_exact() -> None:
    torch.manual_seed(19)
    control = CandidateHead(hidden_size=16, head_dim=8)
    torch.manual_seed(19)
    treatment = CandidateInteractionHead(hidden_size=16, head_dim=8)
    for name, tensor in control.state_dict().items():
        torch.testing.assert_close(treatment.state_dict()[name], tensor, rtol=0, atol=0)
    candidates = torch.randn(3, 4, 16)
    query = torch.randn(3, 16)
    kinds = torch.tensor([0, 1, 2])
    mask = torch.tensor(
        [[True] * 4, [True, True, False, False], [True, True, True, False]]
    )
    baseline = control(candidates, query).masked_fill(~mask, -float("inf"))
    actual = treatment(candidates, query, kinds, mask)
    torch.testing.assert_close(actual, baseline, rtol=0, atol=0)

    with torch.no_grad():
        treatment.interaction_out.weight.fill_(0.1)
    shifted = treatment(candidates, query, kinds, mask)
    torch.testing.assert_close(shifted[1], baseline[1], rtol=0, atol=0)
    assert not torch.allclose(shifted[0], baseline[0])
    assert not torch.allclose(shifted[2, :3], baseline[2, :3])


@pytest.mark.parametrize("kind,count", [(0, 2), (0, 255), (1, 2), (2, 2), (2, 10)])
def test_native_cardinality_and_probability_shape(kind: int, count: int) -> None:
    head = CandidateInteractionHead(hidden_size=8, head_dim=4, interaction_dim=4)
    candidates = torch.randn(1, count, 8)
    logits = head(
        candidates,
        torch.randn(1, 8),
        torch.tensor([kind]),
        torch.ones(1, count, dtype=torch.bool),
    )
    assert logits.shape == (1, count)
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()
    torch.testing.assert_close(logits.softmax(-1).sum(-1), torch.ones(1))


def test_set_interaction_is_equivariant_for_fixed_candidate_vectors() -> None:
    torch.manual_seed(11)
    head = CandidateInteractionHead(hidden_size=12, head_dim=6, interaction_dim=4)
    with torch.no_grad():
        head.interaction_out.weight.fill_(0.1)
    candidates = torch.randn(2, 5, 12)
    query = torch.randn(2, 12)
    kinds = torch.tensor([0, 2])
    mask = torch.ones(2, 5, dtype=torch.bool)
    original = head(candidates, query, kinds, mask)
    permutation = torch.tensor([4, 2, 0, 3, 1])
    changed = head(candidates[:, permutation], query, kinds, mask)
    torch.testing.assert_close(changed, original[:, permutation], rtol=1e-6, atol=1e-6)


def test_padding_cannot_change_valid_logits_and_gradients_are_finite() -> None:
    torch.manual_seed(21)
    head = CandidateInteractionHead(hidden_size=12, head_dim=6, interaction_dim=4)
    with torch.no_grad():
        head.interaction_out.weight.fill_(0.1)
    candidates = torch.randn(1, 5, 12, requires_grad=True)
    query = torch.randn(1, 12, requires_grad=True)
    mask = torch.tensor([[True, True, True, False, False]])
    logits = head(candidates, query, torch.tensor([0]), mask)
    changed_padding = candidates.detach().clone()
    changed_padding[:, 3:] += 1000
    control = head(changed_padding, query, torch.tensor([0]), mask)
    torch.testing.assert_close(logits[:, :3], control[:, :3], rtol=0, atol=0)
    assert torch.isneginf(logits[:, 3:]).all()
    (-logits.log_softmax(-1)[0, 1]).backward()
    assert torch.isfinite(candidates.grad[:, :3]).all()
    assert torch.all(candidates.grad[:, 3:] == 0)
    assert torch.isfinite(query.grad).all()
    assert head.interaction_out.weight.grad is not None
    assert torch.isfinite(head.interaction_out.weight.grad).all()


@pytest.mark.parametrize(
    "kind,count,match", [(0, 256, "2..255"), (1, 3, "Noul"), (2, 11, "Score")]
)
def test_rejects_out_of_contract_cardinality(kind: int, count: int, match: str) -> None:
    head = CandidateInteractionHead(hidden_size=8, head_dim=4, interaction_dim=4)
    with pytest.raises(ValueError, match=match):
        head(
            torch.zeros(1, count, 8),
            torch.zeros(1, 8),
            torch.tensor([kind]),
            torch.ones(1, count, dtype=torch.bool),
        )


def test_rejects_missing_or_unknown_question_type() -> None:
    head = CandidateInteractionHead(hidden_size=8, head_dim=4, interaction_dim=4)
    candidates = torch.zeros(1, 2, 8)
    query = torch.zeros(1, 8)
    mask = torch.ones(1, 2, dtype=torch.bool)
    with pytest.raises(ValueError, match="task_type_ids"):
        head(candidates, query, torch.tensor([0, 1]), mask)
    with pytest.raises(ValueError, match="Unsupported System One"):
        head(candidates, query, torch.tensor([3]), mask)


def test_noul_loss_does_not_train_the_interaction_path() -> None:
    head = CandidateInteractionHead(hidden_size=8, head_dim=4, interaction_dim=4)
    logits = head(
        torch.randn(1, 2, 8),
        torch.randn(1, 8),
        torch.tensor([1]),
        torch.ones(1, 2, dtype=torch.bool),
    )
    (-logits.log_softmax(-1)[0, 0]).backward()
    assert head.interaction_out.weight.grad is None
    assert head.scalar.weight.grad is not None


def test_native_renderer_preserves_dynamic_choice_keys_and_score_order() -> None:
    class _Tokenizer:
        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            assert not add_special_tokens
            return [ord(char) % 37 + 1 for char in text]

    tokenizer = _Tokenizer()
    choice = {
        "id": "dynamic-choice",
        "state": {"case": "review"},
        "instructions": ["Pick an option"],
        "options": [
            {"key": f"team/{index:03d}", "description": {"rule": index}}
            for index in range(255)
        ],
        "task_type": "choice",
        "family": "contract",
        "label": 254,
    }
    score = {
        **choice,
        "id": "ordered-score",
        "options": [
            {"key": str(index), "description": f"level {index}"} for index in range(10)
        ],
        "task_type": "score",
        "label": 9,
    }
    rendered = [encode(row, tokenizer, max_length=100_000) for row in (choice, score)]
    batch = collate(rendered, pad_id=0)
    assert rendered[0]["keys"] == [f"team/{index:03d}" for index in range(255)]
    assert rendered[1]["keys"] == [str(index) for index in range(10)]
    assert batch["candidate_mask"].sum(-1).tolist() == [255, 10]
    assert batch["labels"].tolist() == [254, 9]
