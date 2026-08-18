"""Elastic Weight Consolidation for the mmBERT foundation recipe."""
# Derived from Model-training@3bc41e1322ee5a53e08d18eb940855dec53c1539.
# Upstream blob: a8ef9416fb4ce6e6374e5d92f5fefb4dd27221e0.

from __future__ import annotations

import logging

import torch
from torch import nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)
_FISHER_SEQUENCE_LENGTH = 4096


class EWCRegularizer:
    """Protect parameters important to the original masked-language model."""

    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        n_samples: int = 200,
        ewc_lambda: float = 1000.0,
    ):
        self.ewc_lambda = ewc_lambda
        self.device = device
        self.reference_weights = {
            name: parameter.data.clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        logger.info("Computing Fisher Information from %s samples...", n_samples)
        self.fisher = self._compute_fisher(model, dataloader, n_samples)
        logger.info("Fisher Information computed.")

    def _accumulate_sample(self, model, batch, index: int, fisher: dict) -> bool:
        input_ids = batch["input_ids"][index : index + 1, :_FISHER_SEQUENCE_LENGTH].to(
            self.device
        )
        attention_mask = batch["attention_mask"][
            index : index + 1, :_FISHER_SEQUENCE_LENGTH
        ].to(self.device)
        labels = batch["labels"][index : index + 1, :_FISHER_SEQUENCE_LENGTH].to(
            self.device
        )
        model.zero_grad()
        try:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
            outputs.loss.backward()
        except torch.cuda.OutOfMemoryError:
            logger.warning("OOM during Fisher computation, skipping sample")
            torch.cuda.empty_cache()
            return False
        for name, parameter in model.named_parameters():
            if parameter.requires_grad and parameter.grad is not None:
                fisher[name] += parameter.grad.data.pow(2)
        del input_ids, attention_mask, labels, outputs
        torch.cuda.empty_cache()
        return True

    def _compute_fisher(self, model, dataloader, n_samples: int):
        fisher = {
            name: torch.zeros_like(parameter.data)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        model.eval()
        samples_seen = 0
        for batch in dataloader:
            if samples_seen >= n_samples:
                break
            for index in range(batch["input_ids"].shape[0]):
                if samples_seen >= n_samples:
                    break
                samples_seen += self._accumulate_sample(model, batch, index, fisher)
            if samples_seen % 50 == 0:
                logger.info("  Fisher: %s/%s", samples_seen, n_samples)
        for name in fisher:
            fisher[name] /= max(samples_seen, 1)
        model.train()
        return fisher

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """Compute the weighted EWC penalty for the current parameters."""
        loss = 0.0
        for name, parameter in model.named_parameters():
            if name not in self.fisher or name not in self.reference_weights:
                continue
            reference = self.reference_weights[name].to(parameter.device)
            loss += (self.fisher[name] * (parameter - reference).pow(2)).sum()
        return self.ewc_lambda * loss
