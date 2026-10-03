"""Pooled head: sentence embeddings from an encoder or decoder layer exit.

Serves Vela Embedding (ModernBERT, mean pooling, 2D Matryoshka: layer exits
and dimensions) and sentence-transformers packages such as Qwen3-Embedding
(Qwen3, last-token pooling, instruction prompts per ``input_type``). The
package declares everything: ``1_Pooling/config.json`` the pooling,
``modules.json`` the normalization, ``config_sentence_transformers.json`` the
prompts, ``config.json``'s ``representation_contract`` whether intermediate
exits are final-normalized and whether dimensions truncate before L2, and the
exit graphs it ships (``onnx/model_layer_<L>.onnx``) its layer exits.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from ..errors import PackageError
from ..plugins.base import EmbeddingInfo, SurfacePlan, SurfaceRequest
from . import embedding

# Matryoshka views of a model trained for them when the package names none (the legacy default).
MATRYOSHKA_DIMENSIONS = (768, 512, 256, 128, 64)
_POOLING_FLAGS = {
    "pooling_mode_mean_tokens": "mean",
    "pooling_mode_cls_token": "cls",
    "pooling_mode_lasttoken": "last_token",
}
_EXIT_GRAPH = re.compile(r"^model_layer_(\d+)\.onnx$")
_CONTRACT = {
    "version": 1,
    "final_normalization": "final_norm",
    "pooling": "attention_mask_mean",
    "pooling_accumulation_dtype": "float32",
    "truncate_before_l2_normalize": True,
}


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PackageError(f"unreadable {path.name}: {exc}") from exc


@dataclass(frozen=True)
class PooledHead:
    """How a package turns hidden states into its embedding.

    ``layers`` are the declared exits (the last is the full model);
    ``dimensions`` the Matryoshka views (the first is the full width);
    ``normalize_exits`` whether intermediate exits are final-normalized;
    ``prompts`` the text prefix per ``input_type``; ``graphs`` the ONNX graph
    per exit, for engines that run graphs.
    """

    pooling: str
    layers: tuple[int, ...]
    dimensions: tuple[int, ...]
    normalize: bool = True
    normalize_exits: bool = False
    prompts: Mapping[str, str] = field(default_factory=dict)
    graphs: Mapping[int, Path] = field(default_factory=dict)

    @classmethod
    def detect(cls, root: Path, config: dict[str, Any]) -> PooledHead | None:
        """The head of a sentence-transformers package; None when ``root`` is not one."""
        pooling_path = root / "1_Pooling" / "config.json"
        if not pooling_path.is_file() or not (root / "modules.json").is_file():
            return None
        flags = _json(pooling_path)
        modes = [mode for key, mode in _POOLING_FLAGS.items() if flags.get(key)]
        if len(modes) != 1 or any(
            flags.get(key)
            for key in (
                "pooling_mode_max_tokens",
                "pooling_mode_mean_sqrt_len_tokens",
                "pooling_mode_weightedmean_tokens",
            )
        ):
            raise PackageError(f"unsupported pooling configuration {flags}")
        modules = {module.get("type") for module in _json(root / "modules.json")}
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
            dimensions = tuple(d for d in MATRYOSHKA_DIMENSIONS if d <= hidden)
            if dimensions[0] != hidden:
                dimensions = (hidden, *dimensions)
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
        st_config = root / "config_sentence_transformers.json"
        if st_config.is_file():
            declared = _json(st_config).get("prompts") or {}
            prompts = {
                key: value
                for key, value in declared.items()
                if key in ("query", "document")
            }
        return cls(
            pooling=modes[0],
            layers=tuple(sorted({*graphs, layers})),
            dimensions=dimensions,
            normalize="sentence_transformers.models.Normalize" in modules,
            normalize_exits=normalize_exits,
            prompts=prompts,
            graphs=graphs,
        )

    def info(self, modalities: tuple[str, ...] = ("text",)) -> EmbeddingInfo:
        return EmbeddingInfo(
            dimensions=self.dimensions,
            layers=self.layers,
            modalities=modalities,
            normalized=self.normalize,
            pooling=self.pooling,
            input_types=tuple(sorted(self.prompts)),
        )

    def plan(
        self,
        request: SurfaceRequest,
        tokenizer: Any,
        max_input_tokens: int,
        model_sha256: str,
    ) -> SurfacePlan:
        """Tokenize every input (with the prompt of its ``input_type``) within the request's budget."""
        parsed = embedding.parse_request(request, self.info(), max_input_tokens)
        prompt = self.prompts.get(parsed.input_type or "", "")
        items: list[embedding.EmbedItem | str] = []
        for entry in parsed.inputs:
            assert entry.text is not None
            encoded = embedding.encode_text(
                tokenizer, prompt + entry.text, parsed.max_tokens, parsed.overflow
            )
            if isinstance(encoded, str):
                items.append(encoded)
                continue
            ids, usage = encoded
            key = embedding.content_key(
                model_sha256, "pooled", parsed.layer, parsed.dimension, ids
            )
            items.append(
                PooledItem(
                    entry.index,
                    "text",
                    ids,
                    usage,
                    key,
                    layer=parsed.layer,
                    dimension=parsed.dimension,
                )
            )
        rep = embedding.representation(
            model_sha256, parsed.layer, parsed.dimension, self.normalize
        )
        return embedding.plan(request, parsed, items, rep)

    def readout(
        self, hidden: torch.Tensor, mask: torch.Tensor, dimension: int
    ) -> torch.Tensor:
        """One exit's right-padded hidden states to the requested view, ``[rows, dimension]`` FP32."""
        return embedding.matryoshka(
            embedding.pool(hidden, mask, self.pooling), dimension, self.normalize
        )


@dataclass(frozen=True)
class PooledItem(embedding.EmbedItem):
    """A text to embed at one exit and dimension."""

    layer: int = 0
    dimension: int = 0
