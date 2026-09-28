"""Frozen external teacher outputs keyed by complete training inputs.

This module validates cache coverage and soft supervision. It does not acquire
data, choose a teacher, or assign relevance labels to unjudged candidates.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file

from .newbase_data import file_digest, read_jsonl, text_digest
from .newbase_objectives import relational_cosine_loss

_SHA256_HEX_LENGTH = 64
_MATRIX_DIMENSIONS = 2


def input_identity(component_ids, components) -> dict:
    """Keep ordered query/document identity separate from tokenizer identity."""
    return {
        "components": [
            {"id": key, "text_sha256": text_digest(components[key]["text"])}
            for key in component_ids
        ]
    }


def identity_digest(identity: dict) -> str:
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def token_digest(tokens, pad_id: int) -> str:
    return identity_digest({"input_ids": list(tokens), "pad_id": pad_id})


def record_inputs(record, task: str) -> list[tuple[str, ...]]:
    """Enumerate the actual frozen candidates; never mine or relabel them."""
    if task == "embedding":
        if "component_id" in record:
            return [(record["component_id"],)]
        if "pair_component_ids" in record:
            return [(key,) for key in record["pair_component_ids"]]
        return [(record["query_component_id"],)] + [
            (key,) for key in record["candidate_component_ids"]
        ]
    if task == "reranker":
        return [
            (record["query_component_id"], key)
            for key in record["candidate_component_ids"]
        ]
    raise ValueError("Teacher cache task must be embedding or reranker")


def validate_teacher_config(config: dict, task: str, *, anchor: bool = False) -> None:
    expected = "relational_cosine" if task == "embedding" else "query_order"
    if anchor:
        if task != "embedding":
            raise ValueError("Pointwise anchors apply to embedding coordinates")
        expected = "pointwise_cosine"
    if config.get("objective") != expected:
        raise ValueError("External teacher objective differs from the student task")
    if "target_layers" in config:
        if not anchor or task != "embedding":
            raise ValueError("Layer targets apply only to embedding pointwise anchors")
        _validate_target_layers(config["target_layers"])
    if "exit_supervision" in config and (
        task != "reranker" or config["exit_supervision"] not in ("full", "all")
    ):
        raise ValueError("Teacher exit_supervision must be full or all for rerankers")
    weight, temperature = config.get("weight"), config.get("temperature", 2.0)
    if (
        isinstance(weight, bool)
        or not isinstance(weight, (int, float))
        or not math.isfinite(weight)
        or weight < 0
        or isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError("Teacher weight and temperature must be finite and valid")
    if weight and (not config.get("directory") or not config.get("manifest_sha256")):
        raise ValueError("A nonzero teacher requires an explicitly hashed cache")


def _validate_target_layers(layers) -> None:
    if (
        not isinstance(layers, list)
        or not layers
        or any(type(layer) is not int or layer <= 0 for layer in layers)
        or len(set(layers)) != len(layers)
    ):
        raise ValueError(
            "Teacher target_layers must be ordered unique positive integers"
        )


@dataclass(frozen=True)
class TeacherCache:
    """A complete immutable cache for one corpus and one draw sequence.

    Entry token_sha256 binds the student's complete input tokens. A teacher's
    own tokenizer and prompt identities belong separately in manifest.teacher;
    teacher vectors need not have the student's embedding dimension.
    """

    entries: dict[str, dict]
    values: torch.Tensor
    manifest_sha256: str
    manifest: dict

    @classmethod
    def load(
        cls,
        directory: Path,
        expected_sha256: str,
        *,
        task: str,
        corpus,
        draws: list[dict],
        draws_sha256: str,
        target_layers: list[int] | None = None,
    ) -> TeacherCache:
        raw = (directory / "manifest.json").read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        manifest = json.loads(raw)
        if (
            digest != expected_sha256
            or manifest.get("task") != task
            or manifest.get("train_manifest_sha256") != corpus.manifest_sha256
            or manifest.get("source_split") != corpus.manifest["split"]
            or manifest.get("draws_sha256") != draws_sha256
            or manifest.get("complete") is not True
            or manifest.get("target_layers") != target_layers
        ):
            raise ValueError("Teacher cache differs from the configured training run")
        if target_layers is not None:
            _validate_target_layers(target_layers)
            if task != "embedding":
                raise ValueError("Layer targets apply only to embedding caches")
        teacher = manifest.get("teacher", {})
        if any(
            not teacher.get(key)
            for key in ("repo_id", "revision", "files", "representation")
        ):
            raise ValueError("Teacher model files and representation require identity")
        for name in ("entries.jsonl", "values.safetensors"):
            if file_digest(directory / name) != manifest.get("files", {}).get(name):
                raise ValueError("Frozen teacher cache file changed")
        by_id = {row["id"]: row for row in corpus.records}
        selected = {key for draw in draws for key in draw["record_ids"]}
        if not selected <= set(by_id):
            raise ValueError("Teacher draws refer outside the configured corpus")
        expected = {}
        for key in sorted(selected):
            for inputs in record_inputs(by_id[key], task):
                identity = input_identity(inputs, corpus.components)
                expected[identity_digest(identity)] = identity
        entries = {}
        for index, row in enumerate(read_jsonl(directory / "entries.jsonl")):
            key = row.get("key")
            if (
                key in entries
                or key not in expected
                or row.get("identity") != expected[key]
                or row.get("value_index") != index
                or not isinstance(row.get("token_sha256"), str)
                or len(row["token_sha256"]) != _SHA256_HEX_LENGTH
            ):
                raise ValueError("Teacher input identity or row index differs")
            entries[key] = row
        if set(entries) != set(expected):
            raise ValueError("Teacher cache must cover every complete training input")
        tensors = load_file(directory / "values.safetensors", device="cpu")
        if set(tensors) != {"values"}:
            raise ValueError("Teacher cache must contain exactly its values tensor")
        values = tensors["values"]
        dimension = manifest.get("dimensions")
        expected_shape = (
            (len(entries), dimension)
            if target_layers is None
            else (len(entries), len(target_layers), dimension)
        )
        if (
            type(dimension) is not int
            or dimension <= 0
            or values.dtype != torch.float32
            or values.shape != expected_shape
            or not torch.isfinite(values).all()
            or (task == "reranker" and dimension != 1)
        ):
            raise ValueError("Teacher values must have finite FP32 declared geometry")
        if task == "embedding" and not torch.allclose(
            values.norm(dim=-1), torch.ones(values.shape[:-1]), atol=1e-4, rtol=1e-4
        ):
            raise ValueError("Embedding teacher cache must contain unit vectors")
        for name in ("entries.jsonl", "values.safetensors"):
            if file_digest(directory / name) != manifest["files"][name]:
                raise ValueError("Teacher cache changed during loading")
        return cls(entries, values.detach(), digest, manifest)

    def lookup(self, inputs, components, batch, device) -> torch.Tensor:
        if len(inputs) != len(batch.rows):
            raise ValueError("Teacher lookup and complete token batch differ")
        indices = []
        for component_ids, tokens in zip(inputs, batch.rows, strict=True):
            key = identity_digest(input_identity(component_ids, components))
            entry = self.entries.get(key)
            if entry is None or entry["token_sha256"] != token_digest(
                tokens, batch.pad_id
            ):
                raise ValueError("Teacher input tokens differ from the frozen cache")
            indices.append(entry["value_index"])
        return self.values[indices].to(device).detach()

    def validate_tokens(self, tokenizer, components, maximum: int) -> None:
        """Check complete student tokens before evaluation or optimizer updates."""
        if tokenizer.padding_side != "right" or tokenizer.pad_token_id is None:
            raise ValueError(
                "Teacher inputs require an explicit right-padding tokenizer"
            )
        entries = list(self.entries.values())
        for offset in range(0, len(entries), 64):
            rows = entries[offset : offset + 64]
            inputs = [
                [value["id"] for value in row["identity"]["components"]] for row in rows
            ]
            left = [components[ids[0]]["text"] for ids in inputs]
            right = (
                [components[ids[1]]["text"] for ids in inputs]
                if self.manifest["task"] == "reranker"
                else None
            )
            tokens = tokenizer(
                left, right, padding=False, truncation=False, add_special_tokens=True
            )["input_ids"]
            for row, ids in zip(rows, tokens, strict=True):
                if (
                    not ids
                    or len(ids) > maximum
                    or token_digest(ids, tokenizer.pad_token_id) != row["token_sha256"]
                ):
                    raise ValueError(
                        "Teacher input tokens differ from the frozen cache"
                    )


def _unrelated(left, right, components, device) -> torch.Tensor:
    return torch.tensor(
        [
            [
                components[a]["normalized_sha256"] != components[b]["normalized_sha256"]
                and not set(components[a]["parent_groups"])
                & set(components[b]["parent_groups"])
                for b in right
            ]
            for a in left
        ],
        dtype=torch.bool,
        device=device,
    )


def embedding_teacher_loss(
    student, teacher, component_ids, components, *, query_count: int | None
) -> torch.Tensor:
    """Balance relation families rather than weighting by candidate count.

    Retrieval uses eligible query/document and document/document relations.
    Related or duplicate components do not create off-diagonal supervision.
    Semantic pairs retain their explicitly paired relation in addition to the
    unrelated off-diagonal relations. No relation changes an absolute label.
    """
    if (
        student.ndim != _MATRIX_DIMENSIONS
        or teacher.ndim != _MATRIX_DIMENSIONS
        or len(component_ids) != len(student)
        or len(teacher) != len(student)
        or student.shape[1] == 0
        or teacher.shape[1] == 0
    ):
        raise ValueError("Embedding teacher and student input geometry differ")
    if not torch.isfinite(student).all() or not torch.isfinite(teacher).all():
        raise ValueError("Embedding teacher and student inputs must be finite")
    teacher = teacher.detach()
    if query_count is None:
        if len(student) % 2:
            raise ValueError("Semantic teacher supervision requires complete pairs")
        paired = (
            (
                (student[0::2] * student[1::2]).sum(-1)
                - (teacher[0::2] * teacher[1::2]).sum(-1)
            )
            .square()
            .mean()
        )
        mask = _unrelated(component_ids, component_ids, components, student.device)
        if not mask.any():
            return paired
        return (
            paired + relational_cosine_loss(student, student, teacher, teacher, mask)
        ) / 2
    queries, documents = component_ids[:query_count], component_ids[query_count:]
    if not queries or not documents:
        raise ValueError("Retrieval teacher supervision requires queries and documents")
    terms = []
    for left, right, a, b in (
        (queries, documents, slice(None, query_count), slice(query_count, None)),
        (documents, documents, slice(query_count, None), slice(query_count, None)),
    ):
        mask = _unrelated(left, right, components, student.device)
        if mask.any():
            terms.append(
                relational_cosine_loss(
                    student[a], student[b], teacher[a], teacher[b], mask
                )
            )
    return torch.stack(terms).mean() if terms else student.sum() * 0.0
