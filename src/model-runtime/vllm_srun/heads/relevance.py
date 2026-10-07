"""Relevance head: query-document scores from per-exit pair scorers (Vela Reranker).

A cross-encoder reads ``<bos> query <eos> document <eos>`` and scores the
final-normed CLS state at a layer exit, truncated to a Matryoshka width, with
a trained MLP per (layer, dimension) exit (``classification_heads.safetensors``:
``Linear(D, D/2)``, exact GELU, ``Linear(D/2, 1)``, FP32). Scores are raw
logits; ``relevance_score`` is their sigmoid. All pairs of a request run in
one forward and the query is tokenized once. An engine that runs the
package's exit graphs returns the logits directly.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import torch
from torch import nn

from ..errors import (
    DEADLINE_EXCEEDED,
    INVALID_INPUT,
    INVALID_MODEL_OUTPUT,
    MAX_LENGTH_EXCEEDED,
    PackageError,
)
from ..plugins.base import DEADLINE, RerankInfo, SurfacePlan, SurfaceRequest
from ..registry.artifacts import read_json, safetensors_header
from .task import Head, Item, Rows, cache_key, positive_option

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
LOGITS = "logits"
WEIGHT_DATA = "onnx/weights.data"
_MIN_DIMENSION = 2


def is_reranker(root: Path) -> bool:
    """Cheap test: a Matryoshka reranker layout with its trained scorers."""
    return (root / LAYOUT_FILE).is_file() and (root / HEADS_FILE).is_file()


def exit_name(exit: tuple[int, int]) -> str:
    return f"relevance@{exit[0]}x{exit[1]}"


@dataclass(frozen=True)
class RelevanceLayout:
    """The pair-scorer exits a package trains; ``graphs`` holds the ONNX graph per exit."""

    exits: tuple[tuple[int, int], ...]
    default: tuple[int, int]
    weights: Path
    graphs: Mapping[tuple[int, int], Path] = field(default_factory=dict)

    @classmethod
    def read(
        cls,
        root: Path,
        config: dict[str, Any],
        selection: tuple[int, int] | None = None,
    ) -> RelevanceLayout:
        """The layout of a reranker package; ``selection`` pins the served default exit."""
        if (
            config.get("architectures") != ["ModernBertModel"]
            or config.get("representation_contract") != CONTRACT
        ):
            raise PackageError("unsupported reranker representation contract")
        layout = read_json(root / LAYOUT_FILE)
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
            if _has_scorer(header, layer, dim)
        )
        if not exits:
            raise PackageError("the reranker ships no complete pair scorer")
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

    def files(self, root: Path, served: Sequence[tuple[int, int]]) -> tuple[str, ...]:
        """What the family loads: the layout, the scorers and the served exits' graphs."""
        graphs = [
            self.graphs[exit].relative_to(root).as_posix()
            for exit in served
            if exit in self.graphs
        ]
        data = (WEIGHT_DATA,) if graphs and (root / WEIGHT_DATA).is_file() else ()
        return (LAYOUT_FILE, HEADS_FILE, *graphs, *data)

    def info(self, served: Sequence[tuple[int, int]]) -> RerankInfo:
        return RerankInfo(exits=tuple(served), default=self.default)

    def scorers(
        self, exits: Sequence[tuple[int, int]]
    ) -> dict[tuple[int, int], nn.Module]:
        """The FP32 scorer MLP of every exit in ``exits``; the logit is output column 0.

        The last layer carries a second, zero output: oneDNN computes a
        one-output linear differently at different batch sizes, two outputs
        the same in any batch.
        """
        from safetensors.torch import load_file

        tensors = load_file(str(self.weights))
        out: dict[tuple[int, int], nn.Module] = {}
        for layer, dim in exits:
            prefix = f"{layer}.{dim}"
            scorer = nn.Sequential(
                nn.Linear(dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, 2)
            )
            scorer[0].weight.data = tensors[f"{prefix}.0.weight"]
            scorer[0].bias.data = tensors[f"{prefix}.0.bias"]
            weight, bias = tensors[f"{prefix}.3.weight"], tensors[f"{prefix}.3.bias"]
            scorer[2].weight.data = torch.cat([weight, torch.zeros_like(weight)])
            scorer[2].bias.data = torch.cat([bias, torch.zeros_like(bias)])
            out[(layer, dim)] = scorer.float().eval()
        return out


class RelevanceHead(Head):
    """One pair-scorer exit: logits from final-normed CLS rows, or from the exit graph's output."""

    kind: ClassVar[str] = "relevance"
    surface: ClassVar[str] = "rerank"

    def __init__(self, exit: tuple[int, int], scorer: nn.Module | None):
        super().__init__(exit_name(exit), exit[0])
        self.exit = exit
        self.scorer = scorer

    def readout(self, rows: Rows, sequences: Sequence[int]) -> list[Any]:
        if LOGITS in rows.outputs:
            logits = rows.outputs[LOGITS][list(sequences)].reshape(-1)
        else:
            assert self.scorer is not None
            cls = rows.first(sequences, self.layer)[:, : self.exit[1]].float()
            logits = self.scorer(cls)[:, 0]
        return [float(value) for value in logits.float().cpu()]

    def to(self, device: torch.device) -> RelevanceHead:
        if self.scorer is not None:
            self.scorer = self.scorer.to(device)
        return self


@dataclass(frozen=True)
class RerankRequest:
    query: str
    documents: tuple[str, ...]
    exit: tuple[int, int]
    top_n: int
    return_documents: bool
    max_tokens: int


@dataclass
class RerankPlanState:
    request: RerankRequest
    slots: list[int | str]
    usages: list[dict[str, Any] | None]


class RerankSurface:
    """``/v1/rerank`` over a model's relevance heads, one per served exit."""

    def __init__(
        self,
        layout: RelevanceLayout,
        tokenizer: Any,
        heads: Mapping[tuple[int, int], RelevanceHead],
    ):
        self.layout = layout
        self.tokenizer = tokenizer
        self.heads = dict(heads)
        self.info = layout.info(list(heads))

    def parse(self, request: SurfaceRequest, max_input_tokens: int) -> RerankRequest:
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
        layer = positive_option(body.get("layer"), "layer") or self.layout.default[0]
        dim = (
            positive_option(body.get("dimensions"), "dimensions")
            or self.layout.default[1]
        )
        if (layer, dim) not in self.heads:
            served = ", ".join(
                f"{exit_layer}/{exit_dim}" for exit_layer, exit_dim in self.heads
            )
            raise ValueError(
                f"layer {layer} and dimensions {dim} are not a served pair scorer: {served}"
            )
        top_n = positive_option(body.get("top_n"), "top_n") or len(documents)
        return_documents = body.get("return_documents", False)
        if not isinstance(return_documents, bool):
            raise ValueError("return_documents must be a boolean")
        if options.get("overflow", "reject") != "reject":
            raise ValueError(
                "pair scoring rejects over-long pairs (options.overflow: reject)"
            )
        max_tokens = (
            positive_option(options.get("max_tokens"), "options.max_tokens")
            or max_input_tokens
        )
        if max_tokens > max_input_tokens:
            raise ValueError(f"options.max_tokens must be at most {max_input_tokens}")
        return RerankRequest(
            query, tuple(documents), (layer, dim), top_n, return_documents, max_tokens
        )

    def plan(
        self, request: SurfaceRequest, identity: str, max_input_tokens: int
    ) -> SurfacePlan[Item]:
        """Every pair tokenized with the tokenizer's pair template; the query once."""
        parsed = self.parse(request, max_input_tokens)
        head = self.heads[parsed.exit]
        tokenizer = self.tokenizer
        query = tokenizer.encode(parsed.query, add_special_tokens=False)
        specials = tokenizer.num_special_tokens_to_add(True)
        documents = tokenizer.encode_batch(
            list(parsed.documents), add_special_tokens=False
        )
        items: list[Item] = []
        slots: list[int | str] = []
        usages: list[dict[str, Any] | None] = []
        for text, document in zip(parsed.documents, documents, strict=True):
            tokens = len(query.ids) + len(document.ids) + specials
            if not text.strip() or tokens > parsed.max_tokens:
                slots.append(INVALID_INPUT if not text.strip() else MAX_LENGTH_EXCEEDED)
                usages.append(None)
                continue
            ids = tuple(tokenizer.post_process(query, document).ids)
            slots.append(len(items))
            usages.append(
                {"tokens": tokens, "processed_tokens": tokens, "truncated": False}
            )
            items.append(
                Item(
                    ids,
                    head.name,
                    head.layer,
                    cache_key(identity, head.name, head.layer, ids),
                )
            )
        tokens = sum(len(item.ids) for item in items)
        return SurfacePlan(
            request.surface, items, tokens, RerankPlanState(parsed, slots, usages)
        )

    @staticmethod
    def finish(plan: SurfacePlan[Item], results: Any) -> dict[str, Any]:
        """Results by descending logit (ties by input order), ``top_n`` of them, then failed pairs."""
        state: RerankPlanState = plan.state
        scored, failed = [], []
        for index, (slot, usage) in enumerate(
            zip(state.slots, state.usages, strict=True)
        ):
            if isinstance(slot, str):
                failed.append({"index": index, "error": slot})
                continue
            value = DEADLINE if results is DEADLINE else results[slot]
            if value is DEADLINE or value is None or not math.isfinite(value):
                code = DEADLINE_EXCEEDED if value is DEADLINE else INVALID_MODEL_OUTPUT
                failed.append({"index": index, "error": code})
                continue
            entry: dict[str, Any] = {
                "index": index,
                "relevance_score": 1.0 / (1.0 + math.exp(-value)),
                "logit": value,
                "input": dict(usage or {}),
            }
            if state.request.return_documents:
                entry["document"] = state.request.documents[index]
            scored.append(entry)
        scored.sort(key=lambda entry: (-entry["logit"], entry["index"]))
        layer, dim = state.request.exit
        return {
            "results": scored[: state.request.top_n] + failed,
            "usage": {"input_tokens": plan.input_tokens, "output_tokens": 0},
            "meta": {"pair_scorer": {"layer": layer, "dimension": dim}},
        }


def _has_scorer(header: dict[str, Any], layer: int, dim: int) -> bool:
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
            raise PackageError(
                f"reranker scorer {prefix}.{suffix} must be FP32 {shape}"
            )
    return True
