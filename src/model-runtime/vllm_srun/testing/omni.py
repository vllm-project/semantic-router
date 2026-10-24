"""Tiny Vela Omni packages: the published repository's layout with random towers, and prepared bundles.

``write_snapshot`` writes what the published repository holds and the family
reads (``package.FILES``): the variant's ``config.json``, tiny BERT or Qwen3,
SigLIP, Whisper and CLAP configs with random weights in one
``model.safetensors`` (Mini's text weights in BF16, as published), a toy
tokenizer and the real preprocessor configs. Only the readout widths are the
published ones. ``write_bundle`` writes a prepared bundle whose manifest,
inventory and parity receipt follow ``tools/models/vela_omni`` exactly and
whose four graphs are small projections with the real ports
(``testing.onnx_graphs``). Both serve on a laptop CPU in a second.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..families.multimodal_embedding import audio
from ..families.multimodal_embedding import bundle as bundles
from ..families.multimodal_embedding import package as snapshots
from ..registry.artifacts import inventory
from . import onnx_graphs
from .fixtures import qwen3_config, save, tiny_tokenizer

SOURCE = {
    "repo_id": "vllm-sr/Vela-1.0-Omni-Nano",
    "revision": "2ff2d66385dbdd661a560ec3e8bcb45a0527d92e",
}
VOCAB = [
    "[PAD]",
    "[UNK]",
    "[CLS]",
    "[SEP]",
    "route",
    "this",
    "request",
    "to",
    "a",
    "model",
    "hello",
]


def mel_filters(bins: int, mels: int) -> list[list[float]]:
    """Triangular filters over ``bins`` frequency bins: nonnegative, none empty."""
    edges = np.linspace(0, bins - 1, mels + 2)
    positions = np.arange(bins)[:, None]
    left, centre, right = edges[:-2], edges[1:-1], edges[2:]
    rising = (positions - left) / np.maximum(centre - left, 1e-9)
    falling = (right - positions) / np.maximum(right - centre, 1e-9)
    return np.clip(np.minimum(rising, falling), 0, None).round(6).tolist()


def audio_config() -> dict[str, Any]:
    common = {
        "window": "periodic_hann",
        "center": True,
        "pad_mode": "reflect",
        "power": 2,
        "floor": 1e-10,
    }
    return {
        "format_version": 1,
        "whisper": {
            **common,
            "sampling_rate": 16000,
            "n_fft": 400,
            "hop_length": 160,
            "n_samples": 480000,
            "n_frames": 3000,
            "mel_filters": mel_filters(201, 80),
            "log": "log10",
            "range": 8,
            "affine": [0.25, 1.0],
        },
        "clap": {
            **common,
            "sampling_rate": 48000,
            "n_fft": 1024,
            "hop_length": 480,
            "n_samples": 480000,
            "n_frames": 1001,
            "mel_filters": mel_filters(513, 64),
            "log": "db",
            "reference": 1,
            "min_value": 1e-10,
            "db_range": None,
            "padding": "repeatpad",
        },
        "windows": "endpoint_cover_v1",
    }


def write_tokenizer(directory: Path) -> None:
    from tokenizers import Tokenizer, models, pre_tokenizers, processors

    tokenizer = Tokenizer(
        models.WordLevel({w: i for i, w in enumerate(VOCAB)}, unk_token="[UNK]")
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)]
    )
    directory.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(directory / "tokenizer.json"))


def write_bundle(
    root: Path,
    *,
    variant: str = "nano",
    normalize: bool = True,
    source: dict[str, str] | None = None,
    seed: int = 0,
) -> Path:
    """Write a verified bundle of ``variant``'s contract at ``root``.

    ``source`` defaults to the variant's built-in pin; an unpinned source keeps
    the pin's recorded golden answers away from the random weights.
    """
    contract = bundles.VARIANTS[variant]
    size, dimension = contract.image_size, contract.dimension
    root.mkdir(parents=True, exist_ok=True)
    graphs = {
        "text": (
            {"input_ids": [1, "sequence"], "attention_mask": [1, "sequence"]},
            dimension,
        ),
        "image": ({"pixel_values": [1, 3, size, size]}, dimension),
        "clap": ({"input_features": [1, 1, 1001, 64]}, bundles.CLAP_DIMENSION),
        "audio": (
            {
                "input_features": [1, 80, 3000],
                "clap_embedding": [1, bundles.CLAP_DIMENSION],
            },
            dimension,
        ),
    }
    for index, (name, (inputs, width)) in enumerate(graphs.items()):
        onnx_graphs.projection_graph(
            root / f"onnx/{name}.onnx",
            inputs,
            width,
            seed=seed + index,
            normalize=normalize,
        )
    write_tokenizer(root / "components/text")
    (root / "processors").mkdir(exist_ok=True)
    (root / "processors/audio.json").write_text(
        json.dumps(audio_config()), encoding="utf-8"
    )
    if source is None:
        source = dict(SOURCE)
        if variant == "mini":
            source = {
                "repo_id": "vllm-sr/Vela-1.0-Omni-Mini",
                "revision": "801bae3ad28df6891408f0e0441c676b30e132e3",
            }
    checks = sorted(bundles.required_checks(variant))
    receipt = {
        "passed": True,
        "source": source,
        "variant": variant,
        "tests": [{"name": name, "passed": True} for name in checks],
    }
    (root / "reference_parity.json").write_text(json.dumps(receipt), encoding="utf-8")
    draft = {
        "source": source,
        "processors": {"text": {"padding_side": "right", "pad_token_id": 0}},
        "files": {},
    }
    manifest = bundles.expected_manifest(variant, draft)
    manifest["files"] = inventory(root)
    (root / bundles.MANIFEST).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return root


# Tiny towers. Only the widths the published readouts fix are real: Nano's text
# tower is its 384-wide output (CLS, no projection), Mini's is 1,024 wide (its
# readout keeps the first 768), and Nano reads 512 positions.
def text_config(variant: str, vocab: int) -> dict[str, Any]:
    if variant == "mini":
        return {
            **qwen3_config(vocab),
            "architectures": ["Qwen3ForCausalLM"],
            "hidden_size": 1024,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "layer_types": ["full_attention"],
        }
    return {
        "architectures": ["BertModel"],
        "model_type": "bert",
        "vocab_size": vocab,
        "hidden_size": 384,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "intermediate_size": 32,
        "hidden_act": "gelu",
        "max_position_embeddings": 512,
        "type_vocab_size": 2,
        "layer_norm_eps": 1e-12,
        "pad_token_id": 0,
        "position_embedding_type": "absolute",
    }


def vision_config(size: int) -> dict[str, Any]:
    return {
        "model_type": "siglip_vision_model",
        "image_size": size,
        "patch_size": size // 4,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
    }


WHISPER_CONFIG = {
    "model_type": "whisper",
    "d_model": 16,
    "encoder_layers": 1,
    "encoder_attention_heads": 2,
    "encoder_ffn_dim": 32,
    "num_mel_bins": 80,
    "max_source_positions": 1500,
    "activation_function": "gelu",
    "scale_embedding": False,
}
CLAP_CONFIG = {
    "model_type": "clap_audio_model",
    "spec_size": 256,
    "num_mel_bins": 64,
    "patch_size": 4,
    "patch_stride": [4, 4],
    "patch_embeds_hidden_size": 8,
    "patch_embed_input_channels": 1,
    "depths": [2, 1],
    "num_attention_heads": [1, 2],
    "window_size": 8,
    "mlp_ratio": 2.0,
    "hidden_act": "gelu",
    "layer_norm_eps": 1e-5,
    "enable_fusion": False,
    "enable_patch_layer_norm": True,
    "flatten_patch_embeds": True,
    "qkv_bias": True,
    "projection_dim": 8,
    "projection_hidden_act": "relu",
}
NATIVE_TYPES = {"whisper": "whisper_encoder"}


def _random_state(
    module: torch.nn.Module, generator: torch.Generator
) -> dict[str, torch.Tensor]:
    """Random weights for every parameter and persistent buffer; norms near one, variances positive."""
    norms = {
        f"{name}.weight"
        for name, layer in module.named_modules()
        if isinstance(layer, torch.nn.LayerNorm)
        or name.rpartition(".")[2].endswith("norm")
    }
    state = {}
    for name, tensor in module.state_dict().items():
        value = torch.randn(tensor.shape, generator=generator) * 0.05
        if name in norms or name.endswith("running_var"):
            value = 1 + value.abs()
        state[name] = value
    return state


def write_snapshot(root: Path, *, variant: str = "nano", seed: int = 0) -> Path:
    """Write a tiny published Omni snapshot of ``variant`` at ``root`` (every file the family reads)."""
    from ..engines.native import models

    layout, contract = snapshots.LAYOUTS[variant], bundles.VARIANTS[variant]
    size = contract.image_size
    text = root / snapshots.TEXT
    if variant == "mini":
        vocab = tiny_tokenizer(text)
    else:
        write_tokenizer(text)
        vocab = len(VOCAB)
    generator = torch.Generator().manual_seed(seed)
    configs = {
        "text": text_config(variant, vocab),
        "image": vision_config(size),
        "speech": WHISPER_CONFIG,
        "clap": CLAP_CONFIG,
    }
    prefixes = {
        "text": layout.text,
        "image": layout.image,
        "speech": layout.speech,
        "clap": layout.clap,
    }
    tensors: dict[str, torch.Tensor] = {}
    parameters = 0
    for part, config in configs.items():
        module = models.build(
            NATIVE_TYPES.get(config["model_type"], config["model_type"]), config
        )
        parameters += sum(parameter.numel() for parameter in module.parameters())
        for name, value in _random_state(module, generator).items():
            stored = (
                value.to(torch.bfloat16)
                if part == "text" and variant == "mini"
                else value
            )
            tensors[prefixes[part] + name] = stored
    # The published checkpoint also stores CLAP's position indices and BatchNorm counter.
    for name, buffer in models.build("clap_audio_model", CLAP_CONFIG).named_buffers():
        if name.endswith("relative_position_index"):
            tensors[layout.clap + name] = buffer
    tensors[layout.clap + "batch_norm.num_batches_tracked"] = torch.tensor(0)
    clap_width = CLAP_CONFIG["patch_embeds_hidden_size"] * 2 ** (
        len(CLAP_CONFIG["depths"]) - 1
    )
    width = CLAP_CONFIG["projection_dim"]
    readouts = {
        f"{layout.image_projection}weight": (
            contract.dimension,
            configs["image"]["hidden_size"],
        ),
        f"{layout.image_projection}bias": (contract.dimension,),
        f"{layout.speech_projection}weight": (
            contract.dimension,
            WHISPER_CONFIG["d_model"],
        ),
        f"{layout.speech_projection}bias": (contract.dimension,),
        f"{layout.clap_projection}linear1.weight": (width, clap_width),
        f"{layout.clap_projection}linear1.bias": (width,),
        f"{layout.clap_projection}linear2.weight": (width, width),
        f"{layout.clap_projection}linear2.bias": (width,),
        f"{layout.residual}weight": (contract.dimension, width),
    }
    for name, shape in readouts.items():
        tensors[name] = torch.randn(shape, generator=generator) * 0.05
        parameters += tensors[name].numel()
    tensors[f"{layout.residual}mean"] = torch.randn(width, generator=generator) * 0.1
    tensors[f"{layout.residual}scale"] = 1 + torch.rand(width, generator=generator)
    save(tensors, root / snapshots.WEIGHTS)
    files: dict[str, Any] = {
        snapshots.CONFIG: {**layout.config, "parameter_count": parameters},
        f"{snapshots.TEXT}/config.json": configs["text"],
        f"{snapshots.IMAGE}/config.json": {
            "model_type": "siglip",
            "vision_config": configs["image"],
        },
        f"{snapshots.IMAGE}/preprocessor_config.json": {
            "image_processor_type": "SiglipImageProcessor",
            "do_resize": True,
            "do_rescale": True,
            "do_normalize": True,
            "rescale_factor": 1 / 255,
            "resample": 3 if contract.resample == "bicubic" else 2,
            "size": {"height": size, "width": size},
            "image_mean": [0.5] * 3,
            "image_std": [0.5] * 3,
        },
        f"{snapshots.SPEECH}/config.json": WHISPER_CONFIG,
        f"{snapshots.SPEECH}/preprocessor_config.json": dict(audio.WHISPER_FEATURES),
        f"{snapshots.CLAP}/config.json": {
            "model_type": "clap",
            "audio_config": CLAP_CONFIG,
        },
        f"{snapshots.CLAP}/preprocessor_config.json": dict(audio.CLAP_FEATURES),
    }
    for name, value in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return root
