"""The marker head (Vela 2.0 0.3B): options and words read at marker tokens, in FP32 outside autocast.

An option marker is scored against its question marker and the mean of the
parts the question reads: ``u = normalize(W_o LN(h[O]) + W_q LN(h[Q]))``,
``v = normalize(W_p LN(mean(h[part])))``, logit ``u . v / tau`` (cosine), plus
``MLP(h[O])`` (mlp); ``both`` sums them. A word start is scored against a
label marker: ``normalize(W_t LN(h[word])) . normalize(W_e LN(h[E])) / tau_span``.
Parameter names are the packages' (``norm``, ``w_*``, ``log_tau*``, ``cls_mlp``).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

READOUTS = ("cosine", "mlp", "both")


class MarkerHead(nn.Module):
    def __init__(self, hidden: int, projection: int, readout: str):
        super().__init__()
        if readout not in READOUTS:
            raise ValueError(f"unknown marker readout {readout!r}")
        self.readout = readout
        self.norm = nn.LayerNorm(hidden)
        self.w_o = nn.Linear(hidden, projection)
        self.w_q = nn.Linear(hidden, projection)
        self.w_p = nn.Linear(hidden, projection)
        self.log_tau = nn.Parameter(torch.zeros(()))
        self.w_t = nn.Linear(hidden, projection)
        self.w_e = nn.Linear(hidden, projection)
        self.log_tau_span = nn.Parameter(torch.zeros(()))
        if readout in ("mlp", "both"):
            # Index 2 is the packages' (training-only) dropout slot; parameter names keep it.
            self.cls_mlp = nn.Sequential(
                nn.Linear(hidden, 2 * hidden),
                nn.ReLU(),
                nn.Identity(),
                nn.Linear(2 * hidden, 1),
            )

    def forward(
        self,
        hidden: torch.Tensor,
        q_index: torch.Tensor,
        opt_index: torch.Tensor,
        unit_index: torch.Tensor,
        ent_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Option logits ``(M,)`` and word x label logits ``(K, E)`` from padded hidden states.

        ``q_index`` rows are ``[row, [Q], pool start, pool end)``, ``opt_index``
        ``[row, marker, owning question]``, ``unit_index`` ``[row, word start]``
        and ``ent_index`` ``[row, [E]]`` (the published graph's inputs).
        """
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            states = hidden.float()
            options = states[opt_index[:, 0], opt_index[:, 1]]
            owner = opt_index[:, 2]
            logits = torch.zeros(
                options.shape[0], dtype=states.dtype, device=states.device
            )
            if self.readout in ("cosine", "both"):
                cumulative = F.pad(states.cumsum(1), (0, 0, 1, 0))
                row, query, start, end = q_index.unbind(1)
                pooled = (cumulative[row, end] - cumulative[row, start]) / (
                    (end - start).clamp(min=1).unsqueeze(-1).float()
                )
                part = F.normalize(self.w_p(self.norm(pooled)), dim=-1)
                question = self.w_q(self.norm(states[row, query]))
                option = F.normalize(
                    self.w_o(self.norm(options)) + question[owner], dim=-1
                )
                logits = logits + (option * part[owner]).sum(-1) / self.log_tau.exp()
            if self.readout in ("mlp", "both"):
                logits = logits + self.cls_mlp(options).squeeze(-1)
            words = F.normalize(
                self.w_t(self.norm(states[unit_index[:, 0], unit_index[:, 1]])), dim=-1
            )
            labels = F.normalize(
                self.w_e(self.norm(states[ent_index[:, 0], ent_index[:, 1]])), dim=-1
            )
            return logits, (words @ labels.T) / self.log_tau_span.exp()
