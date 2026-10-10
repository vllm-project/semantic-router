"""Whole-input batching and all-exit objective normalization."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import torch

from .newbase_data import retrieval_masks
from .newbase_objectives import (
    cosent_loss,
    lambda_loss,
    multi_positive_loss,
    order_distillation,
    paraphrase_loss,
    ranking_terms,
)
from .newbase_semantic import all_exit_anchor_loss, full_batch_relation_loss
from .newbase_teacher import embedding_teacher_loss


@dataclass(frozen=True)
class ObjectiveConfig:
    """Explicit mathematical choices; source names and experiment IDs are metadata."""

    kind: str
    scale: float = 20.0
    temperature: float = 2.0
    distillation_weight: float = 0.0
    paraphrase_margin: float = 0.5
    pairwise_weight: float = 1.0
    bce_weight: float = 0.2
    preference_weight: float = 0.25
    lambda_weight: float = 0.0
    lambda_k: int = 10
    maximum_candidates: int = 8
    anchor_scale: float = 1.0

    def validate(self, task: str):
        allowed = (
            {"retrieval", "cosent", "paraphrase", "representation"}
            if task == "embedding"
            else {"ranking"}
        )
        if task not in {"embedding", "reranker"} or self.kind not in allowed:
            raise ValueError("Objective kind differs from the model task")
        for field in (
            "scale",
            "temperature",
            "distillation_weight",
            "pairwise_weight",
            "bce_weight",
            "preference_weight",
            "lambda_weight",
            "anchor_scale",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(
                    "Objective weights/scales must be finite and nonnegative"
                )
        if self.scale == 0 or self.temperature == 0:
            raise ValueError("Objective scales must be positive")
        if (
            not math.isfinite(self.paraphrase_margin)
            or not -1 <= self.paraphrase_margin <= 1
        ):
            raise ValueError("Paraphrase margin must be in [-1,1]")
        if any(
            type(value) is not int or value < 1
            for value in (self.lambda_k, self.maximum_candidates)
        ):
            raise ValueError("Candidate bound and Lambda k must be positive integers")


@dataclass(frozen=True)
class TokenBatch:
    rows: tuple[tuple[int, ...], ...]
    pad_id: int

    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(
                {"rows": self.rows, "pad_id": self.pad_id}, separators=(",", ":")
            ).encode()
        ).hexdigest()

    @classmethod
    def encode(cls, tokenizer, left: list[str], right: list[str] | None, maximum: int):
        if not left or (right is not None and len(right) != len(left)):
            raise ValueError("A batch needs complete matched input strings")
        encoded = tokenizer(
            left, right, padding=False, truncation=False, add_special_tokens=True
        )
        rows = tuple(tuple(row) for row in encoded["input_ids"])
        if any(not row or len(row) > maximum for row in rows):
            raise ValueError(
                "Input exceeds its frozen token budget; truncation is forbidden"
            )
        if tokenizer.padding_side != "right" or tokenizer.pad_token_id is None:
            raise ValueError(
                "Training requires explicit right padding with a real pad ID"
            )
        return cls(rows, tokenizer.pad_token_id)

    def partitions(self, token_budget: int):
        """Stable length sort; do not change logical candidates or loss weights."""
        pending, longest = [], 0
        for index in sorted(
            range(len(self.rows)), key=lambda index: (len(self.rows[index]), index)
        ):
            length = len(self.rows[index])
            if length > token_budget:
                raise ValueError(
                    "One complete input exceeds the microbatch token budget"
                )
            if pending and max(longest, length) * (len(pending) + 1) > token_budget:
                yield pending
                pending, longest = [], 0
            pending.append(index)
            longest = max(longest, length)
        if pending:
            yield pending

    def tensors(self, indices: list[int], device):
        maximum = max(len(self.rows[index]) for index in indices)
        ids = torch.full(
            (len(indices), maximum), self.pad_id, dtype=torch.long, device=device
        )
        mask = torch.zeros_like(ids)
        for row, index in enumerate(indices):
            values = self.rows[index]
            ids[row, : len(values)] = torch.tensor(
                values, dtype=torch.long, device=device
            )
            mask[row, : len(values)] = 1
        return ids, mask


def forward_complete(model, batch: TokenBatch, device, token_budget: int, *, amp: bool):
    """Keep every graph until the logical objective sees its complete candidates."""
    outputs = {key: [None] * len(batch.rows) for key, _ in model.exits.weighted()}
    for indices in batch.partitions(token_budget):
        ids, mask = batch.tensors(indices, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            values = model(ids, mask)
        for key, rows in values.items():
            if rows.dtype != torch.float32 or not torch.isfinite(rows).all():
                raise ValueError(
                    "Every readout must be finite FP32 before any conversion"
                )
            for local, global_index in enumerate(indices):
                outputs[key][global_index] = rows[local]
    return {key: torch.stack(values) for key, values in outputs.items()}


def _candidate_ids(record, components, maximum_candidates):
    values = record.get("candidate_component_ids")
    if (
        not values
        or len(set(values)) != len(values)
        or any(key not in components for key in values)
    ):
        raise ValueError(
            "Retrieval training requires an explicitly frozen candidate list"
        )
    if len(values) > maximum_candidates:
        raise ValueError("Frozen candidate list exceeds the configured bound")
    partition = (
        set(record["positive_component_ids"])
        | set(record["judged_negative_component_ids"])
        | set(record["unjudged_component_ids"])
    )
    if not set(values) <= partition:
        raise ValueError("Every candidate requires explicit relevance membership")
    if not set(record["positive_component_ids"]) & set(values):
        raise ValueError("Frozen candidates omit every relevant document")
    return values


def embedding_step(
    model,
    tokenizer,
    records,
    components,
    *,
    device,
    token_budget: int,
    maximum: int,
    objective: ObjectiveConfig,
    amp: bool,
    teacher_cache=None,
    teacher_weight: float = 0.0,
    teacher_temperature: float = 2.0,
    anchor_cache=None,
    anchor_weight: float = 0.0,
    anchor_target_layers=None,
):
    objective.validate("embedding")
    source = records[0]["source"]
    if any(row["source"] != source for row in records):
        raise ValueError("A logical batch cannot mix source objective contracts")
    semantic = objective.kind in {"cosent", "paraphrase"}
    representation = objective.kind == "representation"
    if representation and not anchor_weight and not teacher_weight:
        raise ValueError(
            "Representation training requires explicit teacher supervision"
        )
    if representation:
        component_ids = [row["component_id"] for row in records]
    elif semantic:
        component_ids = [key for row in records for key in row["pair_component_ids"]]
    else:
        query_ids = [row["query_component_id"] for row in records]
        # Deduplicate text, retaining the first physical ID and complete text.
        selected = {}
        for row in records:
            for key in _candidate_ids(row, components, objective.maximum_candidates):
                selected.setdefault(components[key]["normalized_sha256"], key)
        documents = list(selected.values())
        component_ids = query_ids + documents
    batch = TokenBatch.encode(
        tokenizer, [components[key]["text"] for key in component_ids], None, maximum
    )
    values = forward_complete(model, batch, device, token_budget, amp=amp)
    losses, intervention = {}, None
    if representation:
        losses = {key: vectors.sum() * 0.0 for key, vectors in values.items()}
        intervention = next(iter(losses.values()))
    elif semantic:
        labels = torch.tensor(
            [row["label"] for row in records], dtype=torch.float32, device=device
        )
        loss_function = cosent_loss if objective.kind == "cosent" else paraphrase_loss
        loss_parameter = (
            objective.scale
            if objective.kind == "cosent"
            else objective.paraphrase_margin
        )
        losses = {
            key: loss_function(
                (vectors[0::2] * vectors[1::2]).sum(-1), labels, loss_parameter
            )
            for key, vectors in values.items()
        }
        intervention = next(iter(losses.values())) * 0.0
    else:
        positives, valid = retrieval_masks(records, documents, components)
        positives, valid = (
            torch.tensor(positives, device=device),
            torch.tensor(valid, device=device),
        )
        count = len(records)
        scores = {
            key: (vectors[:count] @ vectors[count:].T) * objective.scale
            for key, vectors in values.items()
        }
        teacher = scores[model.exits.layers[-1], model.exits.dimensions[0]]
        shallow_terms = []
        for key, _ in model.exits.weighted():
            losses[key] = multi_positive_loss(scores[key], positives, valid)
            if key[0] != model.exits.layers[-1]:
                student = (
                    scores[key].detach()
                    if objective.distillation_weight == 0
                    else scores[key]
                )
                shallow_terms.append(
                    order_distillation(student, teacher, valid, objective.temperature)
                )
        intervention = (
            torch.stack(shallow_terms).mean() if shallow_terms else teacher.sum() * 0.0
        )
    total = sum(weight * losses[key] for key, weight in model.exits.weighted())
    total = total + objective.distillation_weight * intervention
    metrics = {
        "base_loss": sum(
            weight * losses[key].detach().item()
            for key, weight in model.exits.weighted()
        ),
        "intervention": intervention.detach().item(),
        "input_count": len(batch.rows),
        "maximum_tokens": max(map(len, batch.rows)),
        "token_sha256": batch.digest(),
    }
    if teacher_weight:
        if teacher_cache is None:
            raise ValueError("Nonzero external teacher weight requires a cache")
        reference = teacher_cache.lookup(
            [(key,) for key in component_ids], components, batch, device
        )
        full = values[model.exits.layers[-1], model.exits.dimensions[0]]
        extra = (
            full_batch_relation_loss(full, reference, component_ids, components)
            if representation
            else embedding_teacher_loss(
                full,
                reference,
                component_ids,
                components,
                query_count=None if semantic else len(records),
            )
        )
        total = total + teacher_weight * extra
        metrics["external_teacher_loss"] = extra.detach().item()
    if anchor_weight:
        if anchor_cache is None:
            raise ValueError("Nonzero anchor weight requires a complete cache")
        reference = anchor_cache.lookup(
            [(key,) for key in component_ids], components, batch, device
        )
        anchor = all_exit_anchor_loss(
            values, reference, model.exits, target_layers=anchor_target_layers
        )
        total = total + anchor_weight * objective.anchor_scale * anchor
        metrics["pointwise_anchor_loss"] = anchor.detach().item()
    return total, metrics


def reranker_step(
    model,
    tokenizer,
    records,
    components,
    *,
    device,
    token_budget: int,
    maximum: int,
    objective: ObjectiveConfig,
    amp: bool,
    teacher_cache=None,
    teacher_weight: float = 0.0,
    teacher_temperature: float = 2.0,
    teacher_exit_supervision: str = "full",
):
    objective.validate("reranker")
    if teacher_exit_supervision not in ("full", "all"):
        raise ValueError("Teacher exit_supervision must be full or all for rerankers")
    candidates = [
        _candidate_ids(row, components, objective.maximum_candidates) for row in records
    ]
    left = [
        components[row["query_component_id"]]["text"]
        for row, ids in zip(records, candidates, strict=True)
        for _ in ids
    ]
    right = [components[key]["text"] for ids in candidates for key in ids]
    batch = TokenBatch.encode(tokenizer, left, right, maximum)
    values = forward_complete(model, batch, device, token_budget, amp=amp)
    width = max(map(len, candidates))
    labels = torch.full((len(records), width), -1.0, device=device)
    valid, judged = (
        torch.zeros_like(labels, dtype=torch.bool),
        torch.zeros_like(labels, dtype=torch.bool),
    )
    for index, (row, ids) in enumerate(zip(records, candidates, strict=True)):
        valid[index, : len(ids)] = True
        for column, key in enumerate(ids):
            if key in row["positive_component_ids"]:
                labels[index, column] = 1.0
                judged[index, column] = row.get("positive_bce_judged", True)
            elif key in row["judged_negative_component_ids"]:
                labels[index, column] = 0.0
                judged[index, column] = True
    preferences = (labels == -1) & valid
    for row_index, (row, ids) in enumerate(zip(records, candidates, strict=True)):
        if "unjudged_preference_component_ids" in row:
            allowed = row["unjudged_preference_component_ids"]
            if (
                not isinstance(allowed, list)
                or len(set(allowed)) != len(allowed)
                or not set(allowed) <= set(row["unjudged_component_ids"]) & set(ids)
            ):
                raise ValueError(
                    "Weak preferences require explicit unjudged candidates"
                )
            preferences[row_index] = False
            for column, key in enumerate(ids):
                preferences[row_index, column] = key in allowed
    total, auxiliary = next(iter(values.values())).sum() * 0.0, 0.0
    full_scores = None
    exit_scores = {}
    for key, weight in model.exits.weighted():
        scores = torch.zeros_like(labels)
        offset = 0
        for row, ids in enumerate(candidates):
            scores[row, : len(ids)] = values[key][offset : offset + len(ids)]
            offset += len(ids)
        terms = ranking_terms(
            scores, labels, valid, judged, preference_mask=preferences
        )
        extra = lambda_loss(scores, labels, valid, judged, k=objective.lambda_k)
        base = (
            objective.pairwise_weight * terms["pairwise"]
            + objective.bce_weight * terms["bce"]
            + objective.preference_weight * terms["unjudged_preference"]
        )
        total = total + weight * (base + objective.lambda_weight * extra)
        auxiliary += weight * extra.detach().item()
        if key == (model.exits.layers[-1], model.exits.dimensions[0]):
            full_scores = scores
        if teacher_weight and teacher_exit_supervision == "all":
            exit_scores[key] = scores
    metrics = {
        "intervention": auxiliary,
        "input_count": len(batch.rows),
        "maximum_tokens": max(map(len, batch.rows)),
        "token_sha256": batch.digest(),
    }
    if teacher_weight:
        if teacher_cache is None:
            raise ValueError("Nonzero external teacher weight requires a cache")
        inputs = [
            (row["query_component_id"], key)
            for row, ids in zip(records, candidates, strict=True)
            for key in ids
        ]
        cached = teacher_cache.lookup(inputs, components, batch, device).flatten()
        reference, offset = torch.zeros_like(labels), 0
        for row, ids in enumerate(candidates):
            reference[row, : len(ids)] = cached[offset : offset + len(ids)]
            offset += len(ids)
        extra = (
            sum(
                weight
                * order_distillation(
                    exit_scores[key], reference, valid, teacher_temperature
                )
                for key, weight in model.exits.weighted()
            )
            if teacher_exit_supervision == "all"
            else order_distillation(full_scores, reference, valid, teacher_temperature)
        )
        total = total + teacher_weight * extra
        metrics["external_teacher_loss"] = extra.detach().item()
    return total, metrics
