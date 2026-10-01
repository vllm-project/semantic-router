"""Loss functions for the modality-routing trainer."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: N812  (PyTorch convention)


@dataclass(frozen=True)
class DistillationSettings:
    """How much a student learns from its teacher.

    Attributes:
        temperature: Softmax temperature for the soft-label KL term.
        alpha: Weight of the soft-label term; the hard-label term gets 1 - alpha.
    """

    temperature: float = 3.0
    alpha: float = 0.5


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    hard_loss: torch.Tensor,
    settings: DistillationSettings,
) -> torch.Tensor:
    """Combine a temperature-scaled KL term with the hard-label loss.

    The KL term follows Hinton et al. (2015), including the temperature squared
    factor that keeps its gradient scale comparable to the hard-label term.

    Args:
        student_logits: Student logits of shape [N, C].
        teacher_logits: Teacher logits of shape [N, C], in the same class order.
        hard_loss: Already computed loss of the student against the true labels.
        settings: Temperature and mixing weight.

    Returns:
        alpha * soft_loss + (1 - alpha) * hard_loss.
    """
    temp = settings.temperature
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / temp, dim=-1),
        F.softmax(teacher_logits / temp, dim=-1),
        reduction="batchmean",
    ) * (temp**2)
    return settings.alpha * soft_loss + (1 - settings.alpha) * hard_loss
