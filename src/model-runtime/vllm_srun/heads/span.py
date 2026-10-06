"""The span head (Vela 2.0 decoders): word x label scores, in FP32 outside autocast.

Words are layer-normed and mean-centred over the block; labels are
layer-normed block means plus a label-slot embedding, mean-centred when there
are two or more. ``score[w, l] = <K w, Q l> / sqrt(d) + v . GELU(M w + N l)``,
the candidate head's form over word x label pairs. Parameter names are the
packages' (``span2.*`` for the router head, ``span_broad.*`` for the broad one).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class SpanHead(nn.Module):
    def __init__(self, hidden: int, projection: int = 256, slots: int = 64):
        super().__init__()
        self.d = projection
        self.wn, self.ln = nn.LayerNorm(hidden), nn.LayerNorm(hidden)
        self.slot = nn.Embedding(slots, hidden)
        self.K = nn.Linear(hidden, projection, bias=False)
        self.Q = nn.Linear(hidden, projection, bias=False)
        self.M = nn.Linear(hidden, projection)
        self.N = nn.Linear(hidden, projection, bias=False)
        self.v = nn.Linear(projection, 1, bias=False)
        # An outside-label readout the packages train and ship but never read.
        self.o = nn.Linear(hidden, 1)

    def forward(self, words: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Scores ``(W, L)`` of raw word states ``(W, H)`` against label states ``(L, H)``."""
        with torch.autocast(device_type=words.device.type, enabled=False):
            word = self.wn(words.float())
            word = word - word.mean(0, keepdim=True)
            count = labels.shape[0]
            slots = torch.arange(count, device=labels.device).clamp(
                max=self.slot.num_embeddings - 1
            )
            label = self.ln(labels.float()) + self.slot(slots)
            if count > 1:
                label = label - label.mean(0, keepdim=True)
            bilinear = (self.K(word) @ self.Q(label).T) / math.sqrt(self.d)
            mlp = self.v(
                F.gelu(self.M(word)[:, None, :] + self.N(label)[None, :, :])
            ).squeeze(-1)
            scores: torch.Tensor = bilinear + mlp
            return scores
