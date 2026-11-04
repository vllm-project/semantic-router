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

# ``prefix_bidir``, ``prefix_causal`` and ``causal`` read images and declare that state-once layout. The
# ``validated`` variant reads images and declares a validated length below its maximum, so a question longer than
# it needs long context. ``turbo`` has d3-turbo's card: ``prefix_bidir``, the validated length, and the
# ``systemone`` block of its decision_config (Box and track, question and criteria images, the video forms' clip
# limit and a Box-confidence map). ``ultra`` has d3-ultra's card: turbo's, with the ``header-v1`` prompt layout
# and d3-ultra's temperature per request size.
VARIANTS = (
    "vision",
    "text",
    "pruned",
    "validated",
    "turbo",
    "ultra",
    *pkg.STATE_ONCE_MODES,
)
VALIDATED_LENGTH = 512
# Long enough for the System One SDK suites' requests without the long-context header, as d3-turbo's 32,768 is.
TURBO_VALIDATED_LENGTH, TURBO_MAX_LENGTH = 4096, 32768
TURBO_SYSTEMONE = {
    "question_types": ["box", "track"],
    "image_placements": ["state", "instructions", "criteria"],
    "track_rates": [1, 2],
    "box_video_max_seconds": 60,
    "box_calibration": [[0.0, 0.0], [0.5, 0.4], [1.0, 1.0]],
}
ULTRA_TEMPERATURE = 1.0
ULTRA_TEMPERATURES = {
    "max_questions": [1, 4, 16],
    "temperature": [1.046, 1.0, 1.0, 1.0],
}
# A pruned backbone keeps an irregular subset of its layers, as d3-edge does; full_attention_interval no
# longer describes them, so the layers must follow layer_types.
PRUNED_LAYERS = (
    "linear_attention",
    "linear_attention",
    "full_attention",
    "linear_attention",
    "full_attention",
)
# A state-once package attends fully mid-stack, every fourth layer as the backbone does, so its layout shapes
# what the later layers read; its readout is small enough that answers are not saturated.
STATE_ONCE_LAYERS = (
    *(["linear_attention"] * 3),
    "full_attention",
    *(["linear_attention"] * 3),
    "full_attention",
)
STATE_ONCE_READOUT_SCALE = 0.05
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


def write_fixture(
    output: str | Path,
    variant: str | None,
    seed: int,
    validated: bool = False,
    overrides: dict[str, Any] | None = None,
) -> Path:
    """The fixture command's writer: a package that reads images (``vision``), text only (``text``), images
    with a pruned backbone (``pruned``), images with a validated length below its maximum (``validated``),
    images under a state-once layout (``prefix_bidir``, ``prefix_causal``, ``causal``), d3-turbo's card on
    ``prefix_bidir`` (``turbo``) or d3-ultra's (``ultra``). ``validated`` gives any variant the validated length;
    ``overrides`` sets fields of its decision_config (None removes one).
    """
    from ..engines.native import models

    variant = variant or VARIANTS[0]
    vision = variant != "text"
    ultra = variant == "ultra"
    turbo = variant == "turbo" or ultra
    attention = (
        "prefix_bidir"
        if turbo
        else variant if variant in pkg.STATE_ONCE_MODES else pkg.NONCAUSAL
    )
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
    if attention != pkg.NONCAUSAL:
        text |= {
            "layer_types": list(STATE_ONCE_LAYERS),
            "num_hidden_layers": len(STATE_ONCE_LAYERS),
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
    scale = 0.5 if attention == pkg.NONCAUSAL else STATE_ONCE_READOUT_SCALE
    save(
        {"weight": torch.randn(pkg.MAX_OPTIONS, text["hidden_size"]) * scale},
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
        "attention_mode": attention,
        "pooling": "last",
        "max_length": TURBO_MAX_LENGTH if turbo else 4096,
        "readout_dtype": "float32",
    }
    if variant == "validated" or validated:
        decision["validated_length"] = VALIDATED_LENGTH
    if turbo:
        decision["validated_length"] = TURBO_VALIDATED_LENGTH
        decision["systemone"] = TURBO_SYSTEMONE
    if ultra:
        decision["temperature"] = ULTRA_TEMPERATURE
        decision["temperature_by_questions"] = ULTRA_TEMPERATURES
        decision["prompt_layout"] = pkg.HEADER_V1
    for key, value in (overrides or {}).items():
        if value is None:
            decision.pop(key, None)
        else:
            decision[key] = value
    (root / "decision_config.json").write_text(
        json.dumps(decision, indent=2) + "\n", encoding="utf-8"
    )
    (root / "d3_runtime.py").write_text(PACKAGE_CODE, encoding="utf-8")
    counts = {"text": 0, "vision": 0}
    for name, tensor in tensors.items():
        counts["vision" if name.startswith("visual.") else "text"] += tensor.numel()
    write_manifest(root, counts)
    return root


def write_manifest(root: Path, counts: dict[str, int]) -> None:
    """``MODEL_MANIFEST.json`` over every file in ``root``, with ``counts`` text and vision parameters."""
    decision = json.loads((root / "decision_config.json").read_text(encoding="utf-8"))
    hidden = json.loads((root / "config.json").read_text(encoding="utf-8"))[
        "text_config"
    ]["hidden_size"]
    readout = pkg.MAX_OPTIONS * hidden
    identity_files = {
        name: sha256(root / name) for name in ("model.safetensors", *pkg.IDENTITY_FILES)
    }
    names = sorted(
        path.name
        for path in root.iterdir()
        if path.is_file() and path.name != pkg.MANIFEST_NAME
    )
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
            "decision": {
                key: decision[key] for key in pkg.INFERENCE_FIELDS if key in decision
            },
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


# The projections released quantized d3 packages hold (compressed-tensors targets).
INT4_TARGETS = (
    r"re:(.*\.)?layers\.\d+\.mlp\.(gate|up|down)_proj$",
    r"re:(.*\.)?layers\.\d+\.self_attn\.(q|k|v|o)_proj$",
    r"re:(.*\.)?layers\.\d+\.linear_attn\.(in_proj_qkv|in_proj_z|out_proj)$",
)


def pack_int4(values: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Signed 4-bit ``values`` packed as compressed-tensors does: eight per int32 word along ``dim``."""
    if dim == 0:
        rows = -(-values.shape[0] // 8) * 8
        padded = torch.zeros(rows, values.shape[1], dtype=values.dtype)
        padded[: values.shape[0]] = values
        return pack_int4(padded.t()).t().contiguous()
    nibbles = (values.to(torch.int64) + 8).reshape(values.shape[0], -1, 8)
    words = sum(nibbles[..., i] << (4 * i) for i in range(8))
    return torch.where(words >= 1 << 31, words - (1 << 32), words).to(torch.int32)


def quantize_int4(
    weight: torch.Tensor, group: int, symmetric: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Round-to-nearest int4 per group: (values in [-8, 7], BF16 scales, zero points or None)."""
    rows, columns = weight.shape
    grouped = weight.float().reshape(rows, columns // group, group)
    if symmetric:
        scale = (grouped.abs().amax(-1) / 7).clamp(min=1e-8)
        zero = None
    else:
        low, high = grouped.amin(-1), grouped.amax(-1)
        scale = ((high - low) / 15).clamp(min=1e-8)
        zero = (torch.round(-low / scale) - 8).clamp(-8, 7)
    scale = scale.to(torch.bfloat16)
    shift = 0 if zero is None else zero[..., None]
    values = (torch.round(grouped / scale.float()[..., None]) + shift).clamp(-8, 7)
    return (
        values.reshape(rows, columns).to(torch.int8),
        scale,
        None if zero is None else zero.to(torch.int8),
    )


# The calibrated input global scale of an NVFP4 fixture: FP8 x FP4 range over an activation amax of 32.
NVFP4_INPUT_GLOBAL = 448.0 * 6.0 / 32.0


def write_quantized_fixture(
    source: Path,
    output: Path,
    scheme: str = "int4",
    group: int = 32,
    symmetric: bool = True,
    dequantized: bool = False,
) -> Path:
    """``source`` with its projections in compressed-tensors int4 (W4A16), FP8 (``scheme="float8"``,
    FP8_DYNAMIC) or NVFP4 (``scheme="nvfp4"``), or (``dequantized``) a BF16 package holding exactly the
    weights the quantized one holds. NVFP4 activations get a calibrated global scale of
    ``NVFP4_INPUT_GLOBAL`` (blocks up to 32 in magnitude quantize without saturating).
    """
    import shutil

    from safetensors.torch import load_file

    from ..quantization import INT4, QuantizedWeights

    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    for path in source.iterdir():
        if path.is_file() and path.name not in ("model.safetensors", pkg.MANIFEST_NAME):
            shutil.copy(path, root / path.name)
    holds = QuantizedWeights(INT4, targets=INT4_TARGETS, ignore=()).holds
    tensors: dict[str, torch.Tensor] = {}
    counts = {"text": 0, "vision": 0}
    for name, tensor in load_file(str(source / "model.safetensors")).items():
        counts["vision" if name.startswith("visual.") else "text"] += tensor.numel()
        layer = name.removesuffix(".weight")
        if not (name.endswith(".weight") and holds(layer)):
            tensors[name] = tensor
            continue
        if scheme == "nvfp4":
            from ..accel import nvfp4

            global_ = nvfp4.global_scale(tensor)
            packed, scale = nvfp4.quantize(tensor, global_)
            if dequantized:
                weight = nvfp4.dequantize(packed, scale, global_)
                tensors[name] = weight.to(torch.bfloat16)
            else:
                tensors[f"{layer}.weight_packed"] = packed
                tensors[f"{layer}.weight_scale"] = scale
                tensors[f"{layer}.weight_global_scale"] = global_
                tensors[f"{layer}.input_global_scale"] = torch.tensor(
                    [NVFP4_INPUT_GLOBAL]
                )
            continue
        if scheme == "float8":
            scale = (tensor.float().abs().amax(1, keepdim=True) / 448).clamp(min=1e-12)
            weight = (tensor.float() / scale).to(torch.float8_e4m3fn)
            if dequantized:
                tensors[name] = (weight.float() * scale).to(torch.bfloat16)
            else:
                tensors[name] = weight
                tensors[f"{layer}.weight_scale"] = scale
            continue
        values, scale, zero = quantize_int4(tensor, group, symmetric)
        if dequantized:
            shift = 0 if zero is None else zero.float().repeat_interleave(group, 1)
            weight = (values.float() - shift) * scale.float().repeat_interleave(
                group, 1
            )
            tensors[name] = weight.to(torch.bfloat16)
            continue
        tensors[f"{layer}.weight_packed"] = pack_int4(values)
        tensors[f"{layer}.weight_scale"] = scale
        tensors[f"{layer}.weight_shape"] = torch.tensor(tensor.shape)
        if zero is not None:
            tensors[f"{layer}.weight_zero_point"] = pack_int4(zero, dim=0)
    save(tensors, root / "model.safetensors")
    if not dequantized:
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        int4 = {
            "weights": {
                "num_bits": 4,
                "type": "int",
                "symmetric": symmetric,
                "strategy": "group",
                "group_size": group,
                "dynamic": False,
                "actorder": None,
            },
            "input_activations": None,
        }
        float8 = {
            "weights": {
                "num_bits": 8,
                "type": "float",
                "symmetric": True,
                "strategy": "channel",
                "dynamic": False,
            },
            "input_activations": {
                "num_bits": 8,
                "type": "float",
                "symmetric": True,
                "strategy": "token",
                "dynamic": True,
            },
        }
        fp4 = {
            "num_bits": 4,
            "type": "float",
            "symmetric": True,
            "strategy": "tensor_group",
            "group_size": 16,
            "scale_dtype": "torch.float8_e4m3fn",
            "block_structure": None,
            "actorder": None,
        }
        nvfp4_args = {
            "weights": {**fp4, "dynamic": False},
            "input_activations": {**fp4, "dynamic": "local"},
        }
        formats = {"float8": "float-quantized", "nvfp4": "nvfp4-pack-quantized"}
        args = {"float8": float8, "nvfp4": nvfp4_args}
        config["quantization_config"] = {
            "quant_method": "compressed-tensors",
            "format": formats.get(scheme, "pack-quantized"),
            "config_groups": {
                "group_0": {
                    "targets": list(INT4_TARGETS),
                    **args.get(scheme, int4),
                    "output_activations": None,
                }
            },
            "ignore": ["re:.*visual\\..*"],
        }
        (root / "config.json").write_text(
            json.dumps(config, indent=2) + "\n", encoding="utf-8"
        )
    write_manifest(root, counts)
    return root
