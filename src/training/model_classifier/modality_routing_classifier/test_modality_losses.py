"""Tests for the distillation loss."""

import pytest

torch = pytest.importorskip("torch")

from modality_losses import DistillationSettings, distillation_loss  # noqa: E402


def logits(rows):
    return torch.tensor(rows, dtype=torch.float32)


def test_alpha_zero_is_the_hard_loss():
    student, teacher = logits([[1.0, 2.0, 3.0]]), logits([[3.0, 2.0, 1.0]])
    hard = torch.tensor(0.7)
    result = distillation_loss(student, teacher, hard, DistillationSettings(3.0, 0.0))
    assert result.item() == pytest.approx(0.7)


def test_matching_teacher_adds_no_soft_loss():
    student = logits([[1.0, 2.0, 3.0], [0.5, 0.1, 0.2]])
    hard = torch.tensor(0.4)
    result = distillation_loss(
        student, student.clone(), hard, DistillationSettings(3.0, 0.5)
    )
    assert result.item() == pytest.approx(0.5 * 0.4, abs=1e-6)


def test_soft_term_is_scaled_by_temperature_squared():
    student, teacher = logits([[2.0, 0.0, -1.0]]), logits([[0.0, 2.0, 1.0]])
    hard = torch.tensor(0.0)
    at_one = distillation_loss(student, teacher, hard, DistillationSettings(1.0, 1.0))
    soft = torch.nn.functional.kl_div(
        torch.log_softmax(student / 2.0, dim=-1),
        torch.softmax(teacher / 2.0, dim=-1),
        reduction="batchmean",
    )
    at_two = distillation_loss(student, teacher, hard, DistillationSettings(2.0, 1.0))
    assert at_two.item() == pytest.approx((soft * 4.0).item(), rel=1e-5)
    assert at_one.item() > 0
