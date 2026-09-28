"""Readout heads over cached frozen features.

* ``DualProjection``: separate state and action MLP projections into a shared
  L2-normalized space, scored by a learned-scale cosine (the CLM topology,
  newly initialized; no CLM weights are loaded).
* ``RawCosine``: cosine in the frozen encoder space at a fixed scale.
* ``OrdinalScoreHead``: an absolute cumulative-link Score readout. The state
  maps to one scalar position and the rubric's adjacent level pairs define
  strictly increasing thresholds, so ``P(level >= k) = sigmoid(z - t_k)`` is a
  proper ordinal distribution rather than a softmax of relative similarities.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

RAW_SCALE = 100.0
MAX_SCALE = 100.0


def projection_mlp(
    hidden: int, width: int, depth: int, out: int, layernorm: bool
) -> nn.Module:
    if depth < 2:
        raise ValueError("Projection depth must be at least two")
    layers: list[nn.Module] = [nn.Linear(hidden, width), nn.GELU()]
    for _ in range(depth - 2):
        layers += [
            nn.Linear(width, width),
            nn.LayerNorm(width) if layernorm else nn.Identity(),
            nn.GELU(),
        ]
    layers.append(nn.Linear(width, out))
    return nn.Sequential(*layers)


class DualProjection(nn.Module):
    def __init__(
        self,
        hidden: int,
        width: int = 1536,
        depth: int = 3,
        out: int = 512,
        layernorm: bool = True,
    ):
        super().__init__()
        self.state = projection_mlp(hidden, width, depth, out, layernorm)
        self.action = projection_mlp(hidden, width, depth, out, layernorm)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))
        self.out = out

    def encode_state(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.state(F.normalize(x.float(), dim=-1)), dim=-1)

    def encode_action(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.action(F.normalize(x.float(), dim=-1)), dim=-1)

    def scale(self) -> torch.Tensor:
        return self.logit_scale.exp().clamp(max=MAX_SCALE)


class RawCosine(nn.Module):
    def encode_state(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x.float(), dim=-1)

    encode_action = encode_state

    def scale(self) -> torch.Tensor:
        return torch.tensor(RAW_SCALE)


class OrdinalScoreHead(nn.Module):
    def __init__(self, dim: int, hidden: int = 256):
        super().__init__()
        self.state_norm = nn.LayerNorm(dim)
        self.level_norm = nn.LayerNorm(dim)
        self.state_proj = nn.Linear(dim, hidden)
        self.level_proj = nn.Linear(dim, hidden)
        self.rubric_proj = nn.Linear(hidden, hidden, bias=False)
        self.latent = nn.Linear(hidden, 1)
        self.first = nn.Linear(2 * hidden + 2, 1)
        self.gap = nn.Linear(2 * hidden + 2, 1)

    def forward(
        self, state: torch.Tensor, levels: torch.Tensor, level_mask: torch.Tensor
    ) -> torch.Tensor:
        """Cumulative logits ``z - t_k`` for thresholds k=1..Kmax-1, ``-inf`` where absent.

        ``levels`` must already be sorted by ordinal level; ``level_mask`` marks
        the K offered levels of each row as a prefix.
        """
        with torch.autocast(device_type=state.device.type, enabled=False):
            counts = level_mask.sum(-1)
            if torch.any(counts < 2):
                raise ValueError("Score rows need at least two levels")
            s = F.gelu(self.state_proj(self.state_norm(state.float())))
            lv = F.gelu(self.level_proj(self.level_norm(levels.float())))
            weights = level_mask.float()[..., None]
            rubric = (lv * weights).sum(1) / weights.sum(1)
            z = self.latent(F.gelu(s + self.rubric_proj(rubric)))
            width = levels.shape[1] - 1
            position = torch.arange(
                1, width + 1, device=state.device, dtype=torch.float32
            )[None, :]
            span = (counts - 1).clamp(min=1).float()[:, None]
            features = torch.cat(
                [
                    lv[:, :-1],
                    lv[:, 1:],
                    (position / span)[..., None].expand(-1, -1, 1),
                    (1.0 / span)[..., None].expand(-1, width, 1),
                ],
                dim=-1,
            )
            first = self.first(features[:, :1]).squeeze(-1)
            gaps = F.softplus(self.gap(features[:, 1:]).squeeze(-1)) + 1e-3
            thresholds = torch.cat([first, first + gaps.cumsum(-1)], dim=-1)
            logits = z - thresholds
            valid = position < counts[:, None].float()
            return logits.masked_fill(~valid, -float("inf"))


def ordinal_probabilities(
    cumulative: torch.Tensor, counts: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    """Level probabilities [B, Kmax] from cumulative logits [B, Kmax-1]."""
    with torch.autocast(device_type=cumulative.device.type, enabled=False):
        precise = (
            cumulative if cumulative.dtype == torch.float64 else cumulative.float()
        )
        at_least = torch.sigmoid(precise / temperature)
        ones = torch.ones_like(at_least[:, :1])
        zeros = torch.zeros_like(at_least[:, :1])
        upper = torch.cat([ones, at_least], dim=-1)
        lower = torch.cat([at_least, zeros], dim=-1)
        probabilities = (upper - lower).clamp_min(0.0)
        width = probabilities.shape[1]
        mask = torch.arange(width, device=cumulative.device)[None, :] < counts[:, None]
        probabilities = probabilities * mask
        return probabilities / probabilities.sum(-1, keepdim=True)
