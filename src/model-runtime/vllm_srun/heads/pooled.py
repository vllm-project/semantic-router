"""Pooled head: sentence embeddings from an encoder or decoder layer exit.

Serves Vela Embedding (ModernBERT, mean pooling, 2D Matryoshka: layer exits
and dimensions) and sentence-transformers packages such as Qwen3-Embedding
(Qwen3, last-token pooling, instruction prompts per ``input_type``). The
package declares everything: ``1_Pooling/config.json`` the pooling,
``modules.json`` the normalization, ``config_sentence_transformers.json`` the
prompts, ``config.json``'s ``representation_contract`` whether intermediate
exits are final-normalized and whether dimensions truncate before L2, and the
exit graphs it ships (``onnx/model_layer_<L>.onnx``) its layer exits.

A forward's readout is the full-width FP32 pooled vector of each sequence at
the item's exit; the requested Matryoshka view (truncate, then L2) is applied
when the response is assembled, so one cached forward serves every dimension.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import torch

from ..errors import PackageError
from ..plugins.base import EmbeddingInfo, SurfacePlan, SurfaceRequest
from ..registry.artifacts import read_json
from . import embedding
from .task import Head, Item, Rows, cache_key

# Matryoshka views of a model trained for them when the package names none (the legacy default).
MATRYOSHKA_DIMENSIONS = (768, 512, 256, 128, 64)
POOLING_FILE = "1_Pooling/config.json"
MODULES_FILE = "modules.json"
PROMPTS_FILE = "config_sentence_transformers.json"
WEIGHT_DATA = "onnx/weights.data"
_POOLING_FLAGS = {
    "pooling_mode_mean_tokens": "mean",
    "pooling_mode_cls_token": "cls",
    "pooling_mode_lasttoken": "last_token",
}
_UNSUPPORTED_FLAGS = (
    "pooling_mode_max_tokens",
    "pooling_mode_mean_sqrt_len_tokens",
    "pooling_mode_weightedmean_tokens",
)
_EXIT_GRAPH = re.compile(r"^model_layer_(\d+)\.onnx$")
_CONTRACT = {
    "version": 1,
    "final_normalization": "final_norm",
    "pooling": "attention_mask_mean",
    "pooling_accumulation_dtype": "float32",
    "truncate_before_l2_normalize": True,
}


def is_pooled(root: Path) -> bool:
    """Cheap test: a sentence-transformers package (pooling and module files)."""
    return (root / POOLING_FILE).is_file() and (root / MODULES_FILE).is_file()


@dataclass(frozen=True)
class PooledLayout:
    """How a package turns hidden states into its embedding.

    ``layers`` are the declared exits (the last is the full model);
    ``dimensions`` the Matryoshka views (the first is the full width);
    ``normalize_exits`` whether intermediate exits are final-normalized;
    ``prompts`` the text prefix per ``input_type``; ``graphs`` the ONNX graph
    per exit, for engines that run graphs; ``files`` what the family loads.
    """

    pooling: str
    layers: tuple[int, ...]
    dimensions: tuple[int, ...]
    normalize: bool = True
    normalize_exits: bool = False
    prompts: Mapping[str, str] = field(default_factory=dict)
    graphs: Mapping[int, Path] = field(default_factory=dict)
    files: tuple[str, ...] = ()

    @classmethod
    def read(cls, root: Path, config: dict[str, Any]) -> PooledLayout:
        """The layout of a sentence-transformers package; refuses what the head cannot reproduce."""
        flags = read_json(root / POOLING_FILE)
        modes = [mode for key, mode in _POOLING_FLAGS.items() if flags.get(key)]
        if len(modes) != 1 or any(flags.get(key) for key in _UNSUPPORTED_FLAGS):
            raise PackageError(f"unsupported pooling configuration {flags}")
        modules = {module.get("type") for module in read_json(root / MODULES_FILE)}
        hidden = int(config["hidden_size"])
        if int(flags.get("word_embedding_dimension", hidden)) != hidden:
            raise PackageError(
                "the pooling width differs from the backbone's hidden size"
            )
        layers = int(config["num_hidden_layers"])
        contract = config.get("representation_contract")
        normalize_exits = False
        dimensions: tuple[int, ...] = (hidden,)
        if contract is not None:
            if {k: contract.get(k) for k in _CONTRACT} != _CONTRACT or modes != [
                "mean"
            ]:
                raise PackageError(
                    f"unsupported embedding representation contract {contract}"
                )
            if contract.get("intermediate_normalization") not in ("none", "final_norm"):
                raise PackageError("unsupported intermediate_normalization")
            normalize_exits = contract["intermediate_normalization"] == "final_norm"
            dimensions = (hidden, *(d for d in MATRYOSHKA_DIMENSIONS if d < hidden))
        graphs = {
            int(match.group(1)): path
            for path in sorted((root / "onnx").glob("model_layer_*.onnx"))
            if (match := _EXIT_GRAPH.match(path.name))
        }
        if (root / "onnx" / "model.onnx").is_file():
            graphs[layers] = root / "onnx" / "model.onnx"
        if any(not 1 <= layer <= layers for layer in graphs):
            raise PackageError("an exit graph lies outside the backbone's layers")
        prompts: dict[str, str] = {}
        files = [POOLING_FILE, MODULES_FILE]
        if (root / PROMPTS_FILE).is_file():
            files.append(PROMPTS_FILE)
            declared = read_json(root / PROMPTS_FILE).get("prompts") or {}
            prompts = {k: v for k, v in declared.items() if k in ("query", "document")}
        files += [path.relative_to(root).as_posix() for path in graphs.values()]
        if graphs and (root / WEIGHT_DATA).is_file():
            files.append(WEIGHT_DATA)
        return cls(
            pooling=modes[0],
            layers=tuple(sorted({*graphs, layers})),
            dimensions=dimensions,
            normalize="sentence_transformers.models.Normalize" in modules,
            normalize_exits=normalize_exits,
            prompts=prompts,
            graphs=graphs,
            files=tuple(files),
        )

    def info(self, layers: Sequence[int] | None = None) -> EmbeddingInfo:
        """The card descriptor; ``layers`` narrows the exits to the ones the engine serves."""
        return EmbeddingInfo(
            dimensions=self.dimensions,
            layers=tuple(layers or self.layers),
            normalized=self.normalize,
            pooling=self.pooling,
            input_types=tuple(sorted(self.prompts)),
        )


class PooledHead(Head):
    """One exit's pooled readout: full-width FP32 vectors of each sequence, on the host."""

    kind: ClassVar[str] = "pooled"
    surface: ClassVar[str] = "embeddings"

    def __init__(self, layout: PooledLayout, layer: int):
        super().__init__(f"pooled@{layer}", layer)
        self.layout = layout

    def readout(self, rows: Rows, sequences: Sequence[int]) -> list[Any]:
        pool = {"mean": rows.mean, "cls": rows.first, "last_token": rows.last}
        vectors = pool[self.layout.pooling](sequences, self.layer).float().cpu()
        return list(vectors.unbind(0))


class EmbeddingSurface:
    """``/v1/embeddings`` over a model's pooled heads, one per served exit."""

    def __init__(self, layout: PooledLayout, tokenizer: Any, layers: Sequence[int]):
        self.layout = layout
        self.tokenizer = tokenizer
        self.heads = {layer: PooledHead(layout, layer) for layer in layers}
        self.info = layout.info(layers)

    def plan(
        self, request: SurfaceRequest, identity: str, max_input_tokens: int
    ) -> SurfacePlan[Item]:
        """Tokenize every input (with its ``input_type`` prompt) within the request's budget."""
        parsed = embedding.parse_request(request, self.info, max_input_tokens)
        head = self.heads[parsed.layer]
        prompt = self.layout.prompts.get(parsed.input_type or "", "")
        entries: list[tuple[Item, dict[str, Any]] | str] = []
        for entry in parsed.inputs:
            assert entry.text is not None
            encoded = embedding.encode_text(
                self.tokenizer, prompt + entry.text, parsed.max_tokens, parsed.overflow
            )
            if isinstance(encoded, str):
                entries.append(encoded)
                continue
            ids, usage = encoded
            key = cache_key(identity, head.kind, head.layer, ids)
            entries.append((Item(tuple(ids), head.name, head.layer, key), usage))
        rep = embedding.representation(
            identity, parsed.layer, parsed.dimension, self.layout.normalize
        )
        return embedding.plan(request, parsed, entries, rep)

    def finish(self, plan: SurfacePlan[Item], results: Any) -> dict[str, Any]:
        """The response, each vector the request's Matryoshka view of its pooled vector."""
        dimension = plan.state.request.dimension
        normalize = self.layout.normalize

        def view(vector: torch.Tensor) -> list[float]:
            return embedding.matryoshka(vector[None], dimension, normalize)[0].tolist()

        return embedding.finish(plan, results, view)
