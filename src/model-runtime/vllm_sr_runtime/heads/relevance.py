"""Relevance head: query-document scores from per-exit pair scorers (Vela Reranker).

A cross-encoder reads ``<bos> query <eos> document <eos>`` and scores the
final-normed CLS state at a layer exit, truncated to a Matryoshka width, with
a trained MLP per (layer, dimension) exit (``classification_heads.safetensors``:
``Linear(D, D/2)``, exact GELU, ``Linear(D/2, 1)``, FP32). Scores are raw
logits; ``relevance_score`` is their sigmoid. All pairs of a request run as
one batch; the query is tokenized once.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ..errors import (
    INVALID_INPUT,
    INVALID_MODEL_OUTPUT,
    MAX_LENGTH_EXCEEDED,
    PackageError,
)
from ..plugins.base import DEADLINE, RerankInfo, SurfacePlan, SurfaceRequest
from ..registry.artifacts import safetensors_header
from .embedding import content_key

MAX_DOCUMENTS = 1024
CONTRACT = {
    "version": 1,
    "pooling": "cls",
    "intermediate_normalization": "final_norm",
    "final_normalization": "final_norm",
    "head_dtype": "float32",
}
HEADS_FILE = "classification_heads.safetensors"
LAYOUT_FILE = "matryoshka_config.json"
_MIN_DIMENSION = 2


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PackageError(f"unreadable {path.name}: {exc}") from exc


@dataclass(frozen=True)
class RerankRequest:
    query: str
    documents: tuple[str, ...]
    exit: tuple[int, int]
    top_n: int
    return_documents: bool
    max_tokens: int


@dataclass(frozen=True)
class PairItem:
    """One query-document pair: its token IDs, exit and content key."""

    index: int
    ids: list[int]
    exit: tuple[int, int]
    usage: dict[str, Any]
    cache_key: str


@dataclass
class RerankPlanState:
    request: RerankRequest
    slots: list[int | str]


@dataclass(frozen=True)
class RelevanceHead:
    """The pair-scorer exits a package trains; ``graphs`` holds the ONNX graph per exit."""

    exits: tuple[tuple[int, int], ...]
    default: tuple[int, int]
    weights: Path
    graphs: Mapping[tuple[int, int], Path] = field(default_factory=dict)

    @classmethod
    def detect(
        cls,
        root: Path,
        config: dict[str, Any],
        selection: tuple[int, int] | None = None,
    ) -> RelevanceHead | None:
        """The head of a reranker package; None when ``root`` holds no pair scorers.

        ``selection`` pins the served default exit (the deployment's pair scorer).
        """
        if not (root / LAYOUT_FILE).is_file() or not (root / HEADS_FILE).is_file():
            return None
        if (
            config.get("architectures") != ["ModernBertModel"]
            or config.get("representation_contract") != CONTRACT
        ):
            raise PackageError("unsupported reranker representation contract")
        layout = _json(root / LAYOUT_FILE)
        hidden, layers = int(config["hidden_size"]), int(config["num_hidden_layers"])
        layer_indices, dim_indices = layout.get("layer_indices"), layout.get(
            "dim_indices"
        )
        if (
            layout.get("hidden_size") != hidden
            or layout.get("num_layers") != layers
            or layout.get("pooling_strategy") != "cls"
            or layout.get("has_final_norm") is False
            or layout.get("representation_contract", CONTRACT) != CONTRACT
            or not layer_indices
            or not dim_indices
            or len(set(layer_indices)) != len(layer_indices)
            or len(set(dim_indices)) != len(dim_indices)
            or not all(1 <= layer <= layers for layer in layer_indices)
            or not all(_MIN_DIMENSION <= dim <= hidden for dim in dim_indices)
        ):
            raise PackageError("invalid trained reranker exit layout")
        header = safetensors_header(root / HEADS_FILE)
        exits = tuple(
            (layer, dim)
            for layer in sorted(layer_indices)
            for dim in sorted(dim_indices, reverse=True)
            if _has_head(header, layer, dim)
        )
        if not exits:
            raise PackageError("the reranker ships no complete pair-scorer head")
        default = selection or (layers, hidden)
        if default not in exits:
            raise PackageError(
                f"no trained pair scorer at layer {default[0]}, dimension {default[1]}"
            )
        graphs = {}
        for layer, dim in exits:
            candidates = [
                root / "onnx" / f"model_layer_{layer}_dim_{dim}.onnx",
                root / "onnx" / f"layer-{layer}" / f"dim-{dim}" / "model.onnx",
            ]
            if (layer, dim) == (layers, hidden):
                candidates.insert(0, root / "onnx" / "model.onnx")
            found = next((path for path in candidates if path.is_file()), None)
            if found is not None:
                graphs[(layer, dim)] = found
        return cls(
            exits=exits, default=default, weights=root / HEADS_FILE, graphs=graphs
        )

    def info(self, served: Sequence[tuple[int, int]] | None = None) -> RerankInfo:
        return RerankInfo(exits=tuple(served or self.exits), default=self.default)

    def load(
        self, exits: Sequence[tuple[int, int]], device: torch.device
    ) -> nn.ModuleDict:
        """The FP32 scorer MLP of every exit in ``exits``, keyed ``"<layer>_<dim>"``."""
        from safetensors.torch import load_file

        tensors = load_file(str(self.weights))
        scorers = nn.ModuleDict()
        for layer, dim in exits:
            prefix = f"{layer}.{dim}"
            scorer = nn.Sequential(
                nn.Linear(dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, 1)
            )
            scorer[0].weight.data = tensors[f"{prefix}.0.weight"]
            scorer[0].bias.data = tensors[f"{prefix}.0.bias"]
            scorer[2].weight.data = tensors[f"{prefix}.3.weight"]
            scorer[2].bias.data = tensors[f"{prefix}.3.bias"]
            scorers[f"{layer}_{dim}"] = scorer
        return scorers.float().to(device).eval()

    def parse(
        self,
        request: SurfaceRequest,
        served: Sequence[tuple[int, int]],
        max_input_tokens: int,
    ) -> RerankRequest:
        """Validate a rerank request (``ValueError`` -> 400)."""
        body, options = request.body, request.options
        query, documents = body.get("query"), body.get("documents")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a nonempty string")
        if (
            not isinstance(documents, list)
            or not documents
            or not all(isinstance(doc, str) for doc in documents)
        ):
            raise ValueError("documents must be a nonempty list of strings")
        if len(documents) > MAX_DOCUMENTS:
            raise ValueError(f"documents holds at most {MAX_DOCUMENTS} items")
        layer = _positive(body.get("layer"), "layer") or self.default[0]
        dim = _positive(body.get("dimensions"), "dimensions") or self.default[1]
        if (layer, dim) not in served:
            raise ValueError(
                f"layer {layer} and dimensions {dim} are not a served pair scorer: "
                + ", ".join(
                    f"{exit_layer}/{exit_dim}" for exit_layer, exit_dim in served
                )
            )
        top_n = _positive(body.get("top_n"), "top_n") or len(documents)
        return_documents = body.get("return_documents", False)
        if not isinstance(return_documents, bool):
            raise ValueError("return_documents must be a boolean")
        if options.get("overflow", "reject") != "reject":
            raise ValueError(
                "pair scoring rejects over-long pairs (options.overflow: reject)"
            )
        max_tokens = (
            _positive(options.get("max_tokens"), "options.max_tokens")
            or max_input_tokens
        )
        if max_tokens > max_input_tokens:
            raise ValueError(f"options.max_tokens must be at most {max_input_tokens}")
        return RerankRequest(
            query, tuple(documents), (layer, dim), top_n, return_documents, max_tokens
        )

    def plan(
        self,
        request: SurfaceRequest,
        tokenizer: Any,
        served: Sequence[tuple[int, int]],
        max_input_tokens: int,
        model_sha256: str,
    ) -> SurfacePlan:
        """Every pair tokenized with the tokenizer's pair template; the query once."""
        parsed = self.parse(request, served, max_input_tokens)
        query = tokenizer.encode(parsed.query, add_special_tokens=False)
        specials = tokenizer.num_special_tokens_to_add(True)
        documents = tokenizer.encode_batch(
            list(parsed.documents), add_special_tokens=False
        )
        items: list[PairItem] = []
        slots: list[int | str] = []
        for index, (text, document) in enumerate(
            zip(parsed.documents, documents, strict=True)
        ):
            tokens = len(query.ids) + len(document.ids) + specials
            if not text.strip():
                slots.append(INVALID_INPUT)
            elif tokens > parsed.max_tokens:
                slots.append(MAX_LENGTH_EXCEEDED)
            else:
                ids = list(tokenizer.post_process(query, document).ids)
                key = content_key(model_sha256, "relevance", *parsed.exit, ids)
                usage = {
                    "tokens": tokens,
                    "processed_tokens": tokens,
                    "truncated": False,
                }
                slots.append(len(items))
                items.append(PairItem(index, ids, parsed.exit, usage, key))
        tokens = sum(len(item.ids) for item in items)
        return SurfacePlan(
            request.surface, items, tokens, RerankPlanState(parsed, slots)
        )

    @staticmethod
    def readout(
        scorers: nn.ModuleDict, cls: torch.Tensor, exit: tuple[int, int]
    ) -> torch.Tensor:
        """Logits ``[rows]`` from final-normed CLS states ``[rows, hidden]`` at ``exit``."""
        layer, dim = exit
        return scorers[f"{layer}_{dim}"](cls[:, :dim].float()).squeeze(-1)

    @staticmethod
    def finish(plan: SurfacePlan, results: Any) -> dict[str, Any]:
        """Results by descending logit (ties by input order), ``top_n`` of them, then failed pairs."""
        state: RerankPlanState = plan.state
        scored, failed = [], []
        for index, slot in enumerate(state.slots):
            if isinstance(slot, str):
                failed.append({"index": index, "error": slot})
                continue
            value = DEADLINE if results is DEADLINE else results[slot]
            if value is DEADLINE or value is None or not math.isfinite(value):
                code = (
                    "deadline_exceeded" if value is DEADLINE else INVALID_MODEL_OUTPUT
                )
                failed.append({"index": index, "error": code})
                continue
            entry: dict[str, Any] = {
                "index": index,
                "relevance_score": 1.0 / (1.0 + math.exp(-value)),
                "logit": value,
                "input": dict(plan.items[slot].usage),
            }
            if state.request.return_documents:
                entry["document"] = state.request.documents[index]
            scored.append(entry)
        scored.sort(key=lambda entry: (-entry["logit"], entry["index"]))
        return {
            "results": scored[: state.request.top_n] + failed,
            "usage": {"input_tokens": plan.input_tokens, "output_tokens": 0},
            "meta": {
                "pair_scorer": {
                    "layer": state.request.exit[0],
                    "dimension": state.request.exit[1],
                }
            },
        }


def _positive(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _has_head(header: dict[str, Any], layer: int, dim: int) -> bool:
    prefix = f"{layer}.{dim}"
    shapes = {
        "0.weight": [dim // 2, dim],
        "0.bias": [dim // 2],
        "3.weight": [1, dim // 2],
        "3.bias": [1],
    }
    for suffix, shape in shapes.items():
        meta = header.get(f"{prefix}.{suffix}")
        if meta is None:
            return False
        if meta.get("dtype") != "F32" or meta.get("shape") != shape:
            raise PackageError(f"reranker head {prefix}.{suffix} must be FP32 {shape}")
    return True
