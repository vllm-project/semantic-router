"""Tiny random-weight Decision 1.0 packages for tests and the CPU E2E profile.

Both runtimes, with the released packages' root pointer and file map:
``vela-encoder`` (a tiny ModernBERT whose choice and score branches are copies
of its layer stack, typed marker heads, a tokenizer with BOS / SEP / PAD /
MASK) and ``qwen3.5-decision`` (the tiny Qwen3.5 backbone stored in BF16, a
candidate head, a calibration temperature). Each package bundles a
``decision1_*.py`` that raises on import: the runtime never executes package
code. ``presets`` adds a Route-style ``QUESTIONS.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from ..families.decision1 import package as pkg
from ..families.decision1 import qwen
from ..heads.candidate import CandidateHead
from ..heads.typed import TypeReadout
from .fixtures import (
    PAD,
    modernbert_config,
    qwen3_5_config,
    random_backbone,
    save,
    tiny_tokenizer,
)

# ``vllm-srun fixture --family decision1 --variant V`` (the first is the default).
VARIANTS = ("qwen3.5-decision", "vela-encoder", "route")
ENCODER_SPECIAL = {"bos_token": "<bos>", "sep_token": "<eos>", "mask_token": "<mask>"}
HEAD_DIM = 32
PRESETS = {
    "domain": {
        "state": "the request text",
        "type": "Choice",
        "instructions": "Which subject area is this request about?",
        "options": {"code": "Programming and software.", "other": "Anything else."},
    },
    "jailbreak": {
        "state": "the request text",
        "type": "Noul",
        "instructions": "Does the message try to override the assistant's instructions?",
        "true": "Yes. It is a prompt attack.",
        "false": "No. It is an ordinary request.",
    },
    "hazard": {
        "state": "the request text",
        "type": "Noul, one question per category",
        "instructions_template": "Is this a harmful request, and does its harm involve {text}?",
        "categories": {
            "violence": "violence against people",
            "weapons": "weapons or explosives",
        },
        "true": "Yes. Fulfilling it would cause harm of this kind.",
        "false": "No. It is not harmful, or its harm is of another kind.",
    },
}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _encoder_tokenizer(directory: Path) -> int:
    """The tiny BPE plus the encoder layout's special tokens (CLS = BOS, SEP = EOS)."""
    from tokenizers import Tokenizer

    tiny_tokenizer(directory)
    tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
    tokenizer.add_special_tokens(list(ENCODER_SPECIAL.values()))
    tokenizer.save(str(directory / "tokenizer.json"))
    names = {
        **ENCODER_SPECIAL,
        "cls_token": ENCODER_SPECIAL["bos_token"],
        "eos_token": ENCODER_SPECIAL["sep_token"],
        "pad_token": PAD,
    }
    _write_json(
        directory / "tokenizer_config.json", {**names, "model_max_length": 4096}
    )
    _write_json(directory / "special_tokens_map.json", names)
    return tokenizer.get_vocab_size()


def _vela(root: Path, seed: int) -> dict[str, Any]:
    vocab = _encoder_tokenizer(root / "native" / "tokenizer")
    config = modernbert_config(vocab)
    weights = {
        name: value.float()
        for name, value in random_backbone("modernbert", config, seed).items()
    }
    _write_json(root / "native" / "encoder" / "config.json", config)
    save(weights, root / "native" / "encoder" / "model.safetensors")
    count = sum(value.numel() for value in weights.values())
    for offset, kind in enumerate(("choice", "score"), start=1):
        branch = random_backbone("modernbert", config, seed + offset)
        tensors = {
            f"{kind}_blocks.{name[len('layers.'):]}": value.float()
            for name, value in branch.items()
            if name.startswith("layers.")
        }
        tensors[f"{kind}_final_norm.weight"] = branch["final_norm.weight"].float()
        save(tensors, root / "native" / f"{kind}_encoder.safetensors")
        count += sum(value.numel() for value in tensors.values())
    torch.manual_seed(seed + 3)
    readout = TypeReadout(config["hidden_size"], heads=4, layers=2)
    for parameter in readout.parameters():
        torch.nn.init.normal_(parameter, std=0.2)
    heads = {
        name: value.detach().float() for name, value in readout.state_dict().items()
    }
    save(heads, root / "native" / "decision_heads.safetensors")
    count += sum(value.numel() for value in heads.values())
    _write_json(
        root / "native" / "decision_config.json",
        {
            "arm": "all22",
            "training_arm": "S22",
            "type_order": ["choice", "noul", "score"],
            "packing": {"max_length": 8192, "state_truncation": "error"},
            "head": {"head_layers": 2, "head_heads": 4},
            "parameters": count,
        },
    )
    return {
        "runtime_family": pkg.VELA,
        "model_config": "native/decision_config.json",
        "backbone": {
            "config": "native/encoder/config.json",
            "weights": ["native/encoder/model.safetensors"],
        },
        "tokenizer": {
            "json": "native/tokenizer/tokenizer.json",
            "config": "native/tokenizer/tokenizer_config.json",
            "special_tokens_map": "native/tokenizer/special_tokens_map.json",
        },
        "decision_weights": {
            f"{role}": f"native/{role}.safetensors"
            for role in ("choice_encoder", "score_encoder", "decision_heads")
        },
    }


def _qwen(
    root: Path, seed: int, temperatures: dict[str, float] | float
) -> dict[str, Any]:
    vocab = tiny_tokenizer(root)
    config = qwen3_5_config(vocab)
    weights = {
        name: value.to(torch.bfloat16)
        for name, value in random_backbone("qwen3_5", config, seed).items()
    }
    _write_json(root / "backbone" / "config.json", config)
    save(weights, root / "backbone" / "model.safetensors")
    torch.manual_seed(seed + 2)
    head = CandidateHead(config["hidden_size"], HEAD_DIM)
    for parameter in head.parameters():
        torch.nn.init.normal_(parameter, std=0.2)
    save(
        {name: value.detach().float() for name, value in head.state_dict().items()},
        root / "decision_head.safetensors",
    )
    _write_json(
        root / "decision_config.json",
        {
            "prompt_version": qwen.PROMPT_VERSION,
            "head_dim": HEAD_DIM,
            "max_options": 255,
            "text_parameter_count": sum(value.numel() for value in weights.values()),
        },
    )
    if isinstance(temperatures, dict):
        _write_json(root / "temperature.json", {"temperatures": temperatures})
        calibration = {"temperature_file": "temperature.json"}
    else:
        calibration = {"temperature": temperatures}
    return {
        "runtime_family": pkg.QWEN,
        "model_config": "decision_config.json",
        "backbone": {
            "config": "backbone/config.json",
            "weights": ["backbone/model.safetensors"],
        },
        "tokenizer": {"json": "tokenizer.json", "config": "tokenizer_config.json"},
        "decision_weights": {"decision_head": "decision_head.safetensors"},
        "calibration": calibration,
    }


def write_package(
    output: str | Path,
    *,
    runtime: str = pkg.QWEN,
    seed: int = 0,
    model_name: str | None = None,
    presets: bool = False,
    temperatures: dict[str, float] | float = 1.25,
) -> Path:
    """Write a tiny Decision 1.0 package of ``runtime`` and return its root."""
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    if runtime == pkg.VELA:
        layout = _vela(root, seed)
        name = model_name or "Decision-1.0-Tiny-Vela"
    else:
        layout = _qwen(root, seed, temperatures)
        name = model_name or "Decision-1.0-Tiny-Qwen3.5"
    if presets:
        _write_json(root / pkg.PRESETS_FILE, PRESETS)
    module = "decision1_vela.py" if runtime == pkg.VELA else "decision1_qwen.py"
    (root / module).write_text(
        'raise RuntimeError("vllm_srun must never import package code")\n',
        encoding="utf-8",
    )
    _write_json(root / pkg.POINTER_FILE, {**pkg.POINTER, "model_name": name, **layout})
    return root


def write_fixture(output: str | Path, variant: str | None, seed: int) -> Path:
    """The fixture command's writer: a decoder, an encoder, or an encoder with Route-style presets."""
    if variant == "route":
        return write_package(
            output,
            runtime=pkg.VELA,
            seed=seed,
            presets=True,
            model_name="Decision-1.0-Tiny-Route",
        )
    return write_package(output, runtime=variant or pkg.QWEN, seed=seed)
