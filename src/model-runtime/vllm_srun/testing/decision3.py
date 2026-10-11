"""Tiny random-weight Decision 3.0 packages for tests and the CPU E2E profile.

A package follows ``d3-package-manifest/1`` exactly: a ``Qwen3_5Model``
checkpoint in BF16 (``language_model.*`` and, unless the ``text`` variant,
``visual.*``), the 255-way readout, ``decision_config.json``, a byte-level BPE
tokenizer in which every answer code is one token, the Qwen3.5 chat template
the released packages ship, the image and video processor settings and a
manifest with the scored identity and parameter counts. The bundled ``d3_runtime.py`` raises
on import: the runtime must never execute package code.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import string
from importlib.resources import files
from pathlib import Path
from typing import Any

import torch

from ..families.decision3 import package as pkg
from .fixtures import qwen3_5_config, save

VARIANTS = ("vision", "text", "pruned")
# A pruned backbone keeps an irregular subset of its layers, as d3-edge does; full_attention_interval no
# longer describes them, so the layers must follow layer_types.
PRUNED_LAYERS = (
    "linear_attention",
    "linear_attention",
    "full_attention",
    "linear_attention",
    "full_attention",
)
SPECIALS = (
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|image_pad|>",
    "<|video_pad|>",
    "<think>",
    "</think>",
)
# Special tokens that only tokenizer_config.json lists, as in the released packages.
CONFIG_ONLY_SPECIALS = ("<|audio_pad|>",)
WORDS = (
    "State",
    "Question",
    "Options",
    "Reply",
    "with",
    "only",
    "the",
    "code",
    "of",
    "best",
    "option",
    "You",
    "are",
    "decision",
    "engine",
    "Treat",
    "state",
    "as",
    "data",
    "not",
    "instructions",
    "Read",
    "question",
    "and",
    "every",
    "then",
    "reply",
    "Choose",
    "matching",
    "No",
    "false",
    "Yes",
    "true",
    "system",
    "user",
    "assistant",
    "image",
    "request",
)
VISION_CONFIG = {
    "model_type": "qwen3_5_vision",
    "deepstack_visual_indexes": [],
    "depth": 2,
    "hidden_act": "gelu_pytorch_tanh",
    "hidden_size": 32,
    "in_channels": 3,
    "intermediate_size": 64,
    "num_heads": 2,
    "num_position_embeddings": 16,
    "patch_size": 16,
    "rope_parameters": {"rope_theta": 10000.0, "rope_type": "axial"},
    "spatial_merge_size": 2,
    "temporal_patch_size": 2,
}
PROCESSOR_CONFIG = {
    "size": {"longest_edge": 16777216, "shortest_edge": 65536},
    "patch_size": 16,
    "temporal_patch_size": 2,
    "merge_size": 2,
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
    "processor_class": "Qwen3VLProcessor",
    "image_processor_type": "Qwen2VLImageProcessorFast",
}
VIDEO_PROCESSOR_CONFIG = {
    "size": {"longest_edge": 25165824, "shortest_edge": 4096},
    "patch_size": 16,
    "temporal_patch_size": 2,
    "merge_size": 2,
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
    "processor_class": "Qwen3VLProcessor",
    "video_processor_type": "Qwen3VLVideoProcessor",
}
PACKAGE_CODE = 'raise RuntimeError("the runtime must never import package code")\n'


def answer_codes() -> list[str]:
    letters = string.ascii_uppercase
    two = ("".join(pair) for pair in itertools.product(letters, repeat=2))
    return [*letters, *itertools.islice(two, pkg.MAX_OPTIONS - len(letters))]


def write_tokenizer(root: Path) -> tuple[dict[str, int], list[str]]:
    """A byte-level BPE in which each answer code and each word above is one token; returns specials and codes."""
    from tokenizers import (
        AddedToken,
        Tokenizer,
        decoders,
        models,
        normalizers,
        pre_tokenizers,
    )

    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {symbol: index for index, symbol in enumerate(alphabet)}
    merges: list[tuple[str, str]] = []

    def add(word: str) -> None:
        current = word[0]
        for char in word[1:]:
            merged = current + char
            if merged not in vocab:
                merges.append((current, char))
                vocab[merged] = len(vocab)
            current = merged

    codes = answer_codes()
    for word in (*codes, *WORDS):
        add(word)
    tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=merges))
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens(
        [AddedToken(token, special=True, normalized=False) for token in SPECIALS]
    )
    tokenizer.save(str(root / "tokenizer.json"))
    ids = {token: tokenizer.token_to_id(token) for token in SPECIALS}
    decoder = {
        str(ids[token]): {
            "content": token,
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": True,
        }
        for token in SPECIALS
    }
    for offset, token in enumerate(CONFIG_ONLY_SPECIALS):
        ids[token] = tokenizer.get_vocab_size() + offset
        decoder[str(ids[token])] = {
            "content": token,
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": True,
        }
    config = {
        "add_bos_token": False,
        "add_prefix_space": False,
        "added_tokens_decoder": decoder,
        "bos_token": None,
        "eos_token": "<|im_end|>",
        "pad_token": "<|endoftext|>",
        "tokenizer_class": "Qwen2Tokenizer",
    }
    (root / "tokenizer_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    return ids, codes


def random_tensors(
    module: torch.nn.Module, prefix: str, seed: int
) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    tensors = {}
    for name, parameter in module.named_parameters():
        value = torch.randn(parameter.shape) * 0.05
        if name.endswith("A_log"):
            value = torch.log(torch.rand(parameter.shape) * 15 + 1)
        elif name.endswith(("dt_bias", "linear_attn.norm.weight")) or (
            name.endswith(("norm1.weight", "norm2.weight", "merger.norm.weight"))
        ):
            value = torch.ones(parameter.shape) + value
        tensors[prefix + name] = value.to(torch.bfloat16)
    return tensors


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_fixture(output: str | Path, variant: str | None, seed: int) -> Path:
    """The fixture command's writer: a package that reads images (``vision``), text only (``text``) or images
    with a pruned backbone (``pruned``)."""
    from ..engines.native import models

    variant = variant or VARIANTS[0]
    vision = variant != "text"
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    ids, codes = write_tokenizer(root)
    vocab = max(ids.values()) + 1
    text = qwen3_5_config(vocab)
    if variant == "pruned":
        text |= {
            "layer_types": list(PRUNED_LAYERS),
            "num_hidden_layers": len(PRUNED_LAYERS),
        }
    tensors = random_tensors(
        models.build("qwen3_5_text", text), "language_model.", seed
    )
    config: dict[str, Any] = {
        "architectures": ["Qwen3_5Model"],
        "model_type": "qwen3_5",
        "text_config": text,
        "tie_word_embeddings": True,
        "image_token_id": ids["<|image_pad|>"],
        "video_token_id": ids["<|video_pad|>"],
        "vision_start_token_id": ids["<|vision_start|>"],
        "vision_end_token_id": ids["<|vision_end|>"],
    }
    if vision:
        config["vision_config"] = {
            **VISION_CONFIG,
            "out_hidden_size": text["hidden_size"],
        }
        vision_module = models.build("qwen3_5_vision", config["vision_config"])
        tensors |= random_tensors(vision_module, "visual.", seed + 1)
        (root / "preprocessor_config.json").write_text(
            json.dumps(PROCESSOR_CONFIG, indent=2) + "\n", encoding="utf-8"
        )
        (root / "video_preprocessor_config.json").write_text(
            json.dumps(VIDEO_PROCESSOR_CONFIG, indent=2) + "\n", encoding="utf-8"
        )
    save(tensors, root / "model.safetensors")
    torch.manual_seed(seed + 2)
    save(
        {"weight": torch.randn(pkg.MAX_OPTIONS, text["hidden_size"]) * 0.5},
        root / "readout.safetensors",
    )
    (root / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    template = (
        files("vllm_srun.testing")
        .joinpath("data/qwen3_5_chat_template.jinja")
        .read_bytes()
    )
    (root / "chat_template.jinja").write_bytes(template)
    from tokenizers import Tokenizer

    backend = Tokenizer.from_file(str(root / "tokenizer.json"))
    decision = {
        "format_version": pkg.FORMAT_VERSION,
        "format_id": pkg.FORMAT_ID,
        "prompt": "d3",
        "base_model": "fixture",
        "revision": "0" * 40,
        "codes": codes,
        "token_ids": [backend.token_to_id(code) for code in codes],
        "temperature": 0.75,
        "attention_mode": "noncausal_full_attention",
        "pooling": "last",
        "max_length": 4096,
        "readout_dtype": "float32",
    }
    (root / "decision_config.json").write_text(
        json.dumps(decision, indent=2) + "\n", encoding="utf-8"
    )
    (root / "d3_runtime.py").write_text(PACKAGE_CODE, encoding="utf-8")
    counts = {"text": 0, "vision": 0}
    for name, tensor in tensors.items():
        counts["vision" if name.startswith("visual.") else "text"] += tensor.numel()
    readout = pkg.MAX_OPTIONS * text["hidden_size"]
    identity_files = {
        name: sha256(root / name) for name in ("model.safetensors", *pkg.IDENTITY_FILES)
    }
    names = sorted(path.name for path in root.iterdir() if path.is_file())
    manifest = {
        "schema": pkg.MANIFEST_SCHEMA,
        "model_name": root.name,
        "format_id": pkg.FORMAT_ID,
        "prompt": "d3",
        "attention_mode": decision["attention_mode"],
        "max_input_tokens": decision["max_length"],
        "identity": {
            "model_sha256": pkg.model_identity(identity_files, decision),
            "files_sha256": identity_files,
            "decision": {key: decision[key] for key in pkg.INFERENCE_FIELDS},
        },
        "parameters": {
            **counts,
            "readout": readout,
            "loaded": counts["text"] + counts["vision"] + readout,
        },
        "licence": {
            "spdx": "apache-2.0",
            "components": [{"component": root.name, "licence": "apache-2.0"}],
        },
        "files_sha256": {name: sha256(root / name) for name in names},
        "files_bytes": {name: (root / name).stat().st_size for name in names},
    }
    (root / pkg.MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=1) + "\n", encoding="utf-8"
    )
    return root
