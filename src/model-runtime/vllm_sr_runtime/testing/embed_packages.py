"""Tiny packages shaped like Vela Embedding, Vela Reranker and Qwen3-Embedding.

Random-weight backbones from ``fixtures.random_backbone`` in the checkpoints'
root namespace, the packages' own metadata files (sentence-transformers
pooling, representation contracts, Matryoshka layouts, FP32 reranker heads,
prompts) and a word-level tokenizer with each package's special-token
template, so detection, planning and readouts run against real file layouts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from .fixtures import modernbert_config, qwen3_config, random_backbone

WORDS = [
    "<pad>",
    "<eos>",
    "<bos>",
    "<unk>",
    "<mask>",
    "how",
    "do",
    "i",
    "reset",
    "my",
    "password",
    "open",
    "settings",
    "security",
    "offices",
    "are",
    "closed",
    "today",
    "a",
    "cat",
    "is",
    "an",
    "animal",
    "the",
    "router",
    "query",
    "instruct",
    "given",
    "search",
    "hello",
]
QWEN_PROMPTS = {"query": "Instruct: given a search query\nQuery:", "document": ""}
EMBED_CONTRACT = {
    "version": 1,
    "intermediate_normalization": "none",
    "final_normalization": "final_norm",
    "pooling": "attention_mask_mean",
    "pooling_accumulation_dtype": "float32",
    "truncate_before_l2_normalize": True,
}
RERANK_CONTRACT = {
    "version": 1,
    "pooling": "cls",
    "intermediate_normalization": "final_norm",
    "final_normalization": "final_norm",
    "head_dtype": "float32",
}


def write_tokenizer(root: Path, *, gemma: bool = True) -> None:
    """``<bos> A <eos>`` / ``<bos> A <eos> B <eos>`` (Vela) or ``A <eos>`` / ``A B <eos>`` (Qwen3)."""
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors

    tokenizer = Tokenizer(
        models.WordLevel({w: i for i, w in enumerate(WORDS)}, unk_token="<unk>")
    )
    tokenizer.normalizer = normalizers.Lowercase()
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    if gemma:
        tokenizer.post_processor = processors.TemplateProcessing(
            single="<bos> $A <eos>",
            pair="<bos> $A <eos> $B <eos>",
            special_tokens=[("<bos>", 2), ("<eos>", 1)],
        )
    else:
        tokenizer.post_processor = processors.TemplateProcessing(
            single="$A <eos>", pair="$A $B <eos>", special_tokens=[("<eos>", 1)]
        )
    root.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(root / "tokenizer.json"))
    (root / "tokenizer_config.json").write_text(
        json.dumps(
            {"pad_token": "<pad>", "eos_token": "<eos>", "padding_side": "right"}
        ),
        encoding="utf-8",
    )
    if gemma:
        special = {"bos_token": "<bos>", "eos_token": "<eos>", "pad_token": "<pad>"}
        (root / "special_tokens_map.json").write_text(
            json.dumps(special), encoding="utf-8"
        )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _sentence_transformers(
    root: Path, pooling: str, width: int, prompts: dict[str, str] | None
) -> None:
    flags = {
        "word_embedding_dimension": width,
        "pooling_mode_cls_token": pooling == "cls",
        "pooling_mode_mean_tokens": pooling == "mean",
        "pooling_mode_max_tokens": False,
        "pooling_mode_mean_sqrt_len_tokens": False,
        "pooling_mode_lasttoken": pooling == "last_token",
    }
    _write_json(root / "1_Pooling" / "config.json", flags)
    _write_json(
        root / "modules.json",
        [
            {
                "idx": 0,
                "name": "0",
                "path": "",
                "type": "sentence_transformers.models.Transformer",
            },
            {
                "idx": 1,
                "name": "1",
                "path": "1_Pooling",
                "type": "sentence_transformers.models.Pooling",
            },
            {
                "idx": 2,
                "name": "2",
                "path": "2_Normalize",
                "type": "sentence_transformers.models.Normalize",
            },
        ],
    )
    if prompts is not None:
        _write_json(root / "config_sentence_transformers.json", {"prompts": prompts})


def write_embedding_package(
    root: Path, *, exits: tuple[int, ...] = (1, 2), graphs: bool = False, seed: int = 0
) -> Path:
    """A Vela-Embedding-shaped package: mean pooling, raw intermediate exits, exit graphs.

    The exit graphs are empty stubs unless ``graphs`` writes toy hidden-state
    graphs with the real ports, the full-model ``model.onnx`` included
    (``testing.onnx_graphs``, needs the onnx package).
    """
    config = modernbert_config(
        len(WORDS), hidden_size=64, representation_contract=EMBED_CONTRACT
    )
    root.mkdir(parents=True, exist_ok=True)
    _write_json(root / "config.json", config)
    save_file(
        random_backbone("modernbert", config, seed), str(root / "model.safetensors")
    )
    _sentence_transformers(root, "mean", 64, None)
    write_tokenizer(root)
    names = [f"model_layer_{layer}.onnx" for layer in exits]
    names += ["model.onnx"] if graphs else []
    for position, name in enumerate(names):
        (root / "onnx").mkdir(exist_ok=True)
        if graphs:
            from .onnx_graphs import token_graph

            token_graph(
                root / "onnx" / name, vocab=len(WORDS), hidden=64, seed=position
            )
        else:
            (root / "onnx" / name).write_bytes(b"")
    return root


def write_reranker_package(
    root: Path,
    *,
    layers: tuple[int, ...] = (2, 4),
    dims: tuple[int, ...] = (64, 32),
    graphs: bool = False,
    seed: int = 0,
) -> Path:
    """A Vela-Reranker-shaped package: Matryoshka layout and an FP32 scorer per exit.

    ``graphs`` adds toy logit graphs for the default exit and one other
    (``model.onnx``, ``model_layer_2_dim_32.onnx``) with the pair-scorer metadata.
    """
    config = modernbert_config(len(WORDS), representation_contract=RERANK_CONTRACT)
    root.mkdir(parents=True, exist_ok=True)
    _write_json(root / "config.json", config)
    save_file(
        random_backbone("modernbert", config, seed), str(root / "model.safetensors")
    )
    _write_json(
        root / "matryoshka_config.json",
        {
            "layer_indices": list(layers),
            "dim_indices": list(dims),
            "hidden_size": config["hidden_size"],
            "num_layers": config["num_hidden_layers"],
            "pooling_strategy": "cls",
            "representation_contract": RERANK_CONTRACT,
        },
    )
    generator = torch.Generator().manual_seed(seed + 1)
    heads = {}
    for layer in layers:
        for dim in dims:
            prefix = f"{layer}.{dim}"
            heads[f"{prefix}.0.weight"] = (
                torch.randn(dim // 2, dim, generator=generator) * 0.2
            )
            heads[f"{prefix}.0.bias"] = torch.randn(dim // 2, generator=generator) * 0.1
            heads[f"{prefix}.3.weight"] = (
                torch.randn(1, dim // 2, generator=generator) * 0.2
            )
            heads[f"{prefix}.3.bias"] = torch.randn(1, generator=generator) * 0.1
    save_file(heads, str(root / "classification_heads.safetensors"))
    write_tokenizer(root)
    if graphs:
        from .onnx_graphs import token_graph

        for name, (layer, dim) in (
            ("model.onnx", (4, 64)),
            ("model_layer_2_dim_32.onnx", (2, 32)),
        ):
            contract = {
                "version": 1,
                "layer": layer,
                "dimension": dim,
                "score_type": "relevance_logit",
            }
            token_graph(
                root / "onnx" / name,
                vocab=len(WORDS),
                hidden=64,
                scorer=dim,
                output="logits",
                metadata={"semantic_router.pair_scorer": json.dumps(contract)},
                seed=layer,
            )
    return root


def write_qwen3_embedding_package(root: Path, *, seed: int = 0) -> Path:
    """A Qwen3-Embedding-shaped package: last-token pooling and query / document prompts."""
    config = qwen3_config(len(WORDS))
    config["architectures"] = ["Qwen3ForCausalLM"]
    root.mkdir(parents=True, exist_ok=True)
    _write_json(root / "config.json", config)
    save_file(random_backbone("qwen3", config, seed), str(root / "model.safetensors"))
    _sentence_transformers(root, "last_token", config["hidden_size"], QWEN_PROMPTS)
    write_tokenizer(root, gemma=False)
    return root
