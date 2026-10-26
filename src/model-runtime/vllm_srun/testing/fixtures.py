"""Tiny random-weight Decision 2.0 packages for tests and the CPU E2E profile.

The packages follow ``dev2-package/1`` exactly (pointer, manifest with
per-file SHA-256, identity, parameter counts) with a small byte-level BPE
tokenizer, so every verification and serving path runs on a laptop CPU in
seconds. The bundled ``decision2/`` package raises on import: the runtime
must never execute package code.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from typing import Any

import torch

from ..families.decision2 import package as pkg
from ..heads.candidate import CandidateHead
from ..plugins import registry
from ..registry.artifacts import safetensors_elements, sha256_file
from ..systemone import MAX_OPTIONS

PAD = "<|endoftext|>"
CORPUS = [
    "Context:\nTask type: choice\nQuestion:\nOptions:\n<option>\n</option>\n\nSelect the single option best supported "
    "by the context and instructions.\nDecision:",
    '{"description":"Yes","key":"true"} {"description":"No","key":"false"} {"key":"0"} {"key":"1"} {"key":"2"}',
    "Write a Python function that merges two sorted lists and explain its running time.",
    "Which domain does this request belong to? Programming Mathematics Anything else",
    "Does answering this request need multi-step reasoning? How difficult is this request? Trivial Moderate Hard",
    "Route the request to the model that answers it best. code math chat creative writing translation summary",
    "The quick brown fox jumps over the lazy dog. 0123456789 ,.;:!?'\"()[]{}<>/\\|-_+=*&^%$#@~`",
]
FAKE_BASE_REVISION = "0" * 39 + "1"


def tiny_tokenizer(directory: Path, vocab_size: int = 384) -> int:
    """Train a deterministic byte-level BPE; returns its vocabulary size."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False, use_regex=True
    )
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[PAD],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train_from_iterator(CORPUS * 4, trainer=trainer)
    directory.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(directory / "tokenizer.json"))
    (directory / "tokenizer_config.json").write_text(
        json.dumps(
            {"eos_token": PAD, "pad_token": PAD, "model_max_length": 4096}, indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    return tokenizer.get_vocab_size()


def qwen3_config(vocab: int) -> dict[str, Any]:
    return {
        "architectures": ["Qwen3Model"],
        "model_type": "qwen3",
        "attention_bias": False,
        "head_dim": 16,
        "hidden_act": "silu",
        "hidden_size": 64,
        "intermediate_size": 128,
        "layer_types": ["full_attention", "full_attention"],
        "max_position_embeddings": 4096,
        "num_attention_heads": 4,
        "num_hidden_layers": 2,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {"rope_theta": 1000000, "rope_type": "default"},
        "tie_word_embeddings": True,
        "vocab_size": vocab,
    }


def qwen3_5_config(vocab: int, **overrides: Any) -> dict[str, Any]:
    config = {
        "architectures": ["Qwen3_5TextModel"],
        "model_type": "qwen3_5_text",
        "attention_bias": False,
        "attn_output_gate": True,
        "full_attention_interval": 4,
        "head_dim": 32,
        "hidden_act": "silu",
        "hidden_size": 64,
        "intermediate_size": 128,
        "layer_types": [
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ],
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 16,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "linear_value_head_dim": 16,
        "max_position_embeddings": 4096,
        "num_attention_heads": 4,
        "num_hidden_layers": 4,
        "num_key_value_heads": 2,
        "partial_rotary_factor": 0.25,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {
            "mrope_interleaved": True,
            "mrope_section": [2, 1, 1],
            "partial_rotary_factor": 0.25,
            "rope_theta": 10000000,
            "rope_type": "default",
        },
        "tie_word_embeddings": True,
        "vocab_size": vocab,
    }
    config.update(overrides)
    return config


def modernbert_config(vocab: int, **overrides: Any) -> dict[str, Any]:
    """A tiny Vela-shaped ModernBERT: global and local layers, YaRN rotary, a 4-token local half-window."""
    config = {
        "architectures": ["ModernBertModel"],
        "model_type": "modernbert",
        "attention_bias": False,
        "bos_token_id": 2,
        "cls_token_id": 1,
        "eos_token_id": 1,
        "global_attn_every_n_layers": 3,
        "global_rope_theta": 160000,
        "hidden_activation": "gelu",
        "hidden_size": 64,
        "intermediate_size": 96,
        "local_attention": 8,
        "local_rope_theta": 160000,
        "max_position_embeddings": 32768,
        "mlp_bias": False,
        "norm_bias": False,
        "norm_eps": 1e-5,
        "num_attention_heads": 4,
        "num_hidden_layers": 4,
        "pad_token_id": 0,
        "rope_scaling": {
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "factor": 4.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "yarn",
            "truncate": True,
        },
        "sep_token_id": 1,
        "vocab_size": vocab,
    }
    config.update(overrides)
    return config


def random_backbone(
    backbone: str, config: dict[str, Any], seed: int
) -> dict[str, torch.Tensor]:
    """Random FP32 weights with Linear weights rounded to BF16, as the released checkpoints store them."""
    from ..engines.native import models

    torch.manual_seed(seed)
    module = models.build(config["model_type"], config)
    state = {}
    linear = {
        f"{name}.weight"
        for name, layer in module.named_modules()
        if isinstance(layer, torch.nn.Linear)
    }
    for name, parameter in module.named_parameters():
        value = torch.randn(parameter.shape) * 0.05
        if name.endswith("A_log"):
            value = torch.log(torch.rand(parameter.shape) * 15 + 1)
        elif (
            name.endswith("dt_bias")
            or name.endswith("linear_attn.norm.weight")
            or (backbone in ("qwen3", "modernbert") and name.endswith("norm.weight"))
        ):
            value = torch.ones(parameter.shape) + value
        state[name] = value.to(torch.bfloat16) if name in linear else value
    return state


def save(tensors: dict[str, torch.Tensor], path: Path) -> None:
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {name: tensor.contiguous() for name, tensor in tensors.items()}, str(path)
    )


def write_fixture(
    output: str | Path, *, family: str, variant: str | None = None, seed: int = 0
) -> Path:
    """Write a tiny package of an installed ``family`` with the writer it names (``ModelFamily.fixture_writer``).

    A writer module exposes ``write_fixture(output, variant, seed) -> Path``
    and lists its variants in ``VARIANTS`` (the first is the default).
    """
    try:
        target = registry.plugin("families", family).load().fixture_writer
    except KeyError:
        target = None
    if not target:
        raise ValueError(f"no fixture writer for the {family!r} family")
    module = importlib.import_module(target)
    variants = tuple(getattr(module, "VARIANTS", ()))
    if variant is not None and variants and variant not in variants:
        raise ValueError(
            f"{family} fixtures are {', '.join(variants)}, not {variant!r}"
        )
    return module.write_fixture(
        output, variant or (variants[0] if variants else None), seed
    )


def write_package(
    output: str | Path,
    *,
    backbone: str = "qwen3_5",
    seed: int = 0,
    score_bias: bool = True,
    max_input_tokens: int = 2048,
    adapter: bool = False,
) -> Path:
    """Write a tiny package (full checkpoint, or LoRA over a tiny pinned base) and return its root."""
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    vocab = tiny_tokenizer(root)
    config = qwen3_config(vocab) if backbone == "qwen3" else qwen3_5_config(vocab)
    architecture = next(
        name for name, kind in pkg.ARCHITECTURES.items() if kind == config["model_type"]
    )
    head_dim = 32
    weights = random_backbone(backbone, config, seed)
    base_root = None
    decision_config: dict[str, Any] = {
        "architecture": architecture,
        "head_variant": "shared",
        "backbone_model_type": "qwen3" if backbone == "qwen3" else "qwen3_5",
        "prompt_version": pkg.PROMPT_VERSION,
        "head_dim": head_dim,
        "max_options": MAX_OPTIONS,
        "parameter_dtype": "float32",
        "autocast_dtype": "bfloat16",
        "head_compute_dtype": "float32",
        "attention": "sdpa",
    }
    if adapter:
        base_root = root.parent / f"{root.name}-base"
        base_root.mkdir(parents=True, exist_ok=False)
        full = {
            "model_type": "qwen3_5",
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "text_config": config,
        }
        (base_root / "config.json").write_text(
            json.dumps(full, indent=2) + "\n", encoding="utf-8"
        )
        base_tensors = {
            f"model.language_model.{name}": tensor for name, tensor in weights.items()
        }
        base_tensors["model.visual.patch_embed.weight"] = torch.zeros(
            4, 4, dtype=torch.bfloat16
        )
        save(base_tensors, base_root / "model-00001-of-00001.safetensors")
        targets = [
            f"layers.{i}.mlp.{proj}"
            for i in range(config["num_hidden_layers"])
            for proj in ("gate_proj", "up_proj", "down_proj")
        ]
        rank = 4
        torch.manual_seed(seed + 1)
        lora_tensors = {}
        dimensions = {}
        for target in targets:
            out_features, in_features = weights[f"{target}.weight"].shape
            dimensions[target] = [in_features, out_features]
            lora_tensors[f"base_model.model.{target}.lora_A.weight"] = (
                torch.randn(rank, in_features) * 0.05
            )
            lora_tensors[f"base_model.model.{target}.lora_B.weight"] = (
                torch.randn(out_features, rank) * 0.05
            )
        save(lora_tensors, root / "adapter" / "adapter_model.safetensors")
        adapter_config = {
            "peft_type": "LORA",
            "r": rank,
            "lora_alpha": 2 * rank,
            "lora_dropout": 0.05,
            "bias": "none",
            "target_modules": sorted({name.rsplit(".", 1)[-1] for name in targets}),
        }
        (root / "adapter" / "adapter_config.json").write_text(
            json.dumps(adapter_config, indent=2) + "\n", encoding="utf-8"
        )
        decision_config.update(
            checkpoint_format=pkg.LORA_FORMAT,
            lora={
                "rank": rank,
                "alpha": 2 * rank,
                "dropout": 0.05,
                "target_modules": targets,
                "target_dimensions": dimensions,
                "source_kind": "posttrained",
                "source_fingerprint": {
                    "source_name": base_root.name,
                    "files_sha256": pkg.source_files(base_root),
                },
                "base_revision": FAKE_BASE_REVISION,
            },
        )
    else:
        decision_config["checkpoint_format"] = "full"
        (root / "backbone").mkdir()
        (root / "backbone" / "config.json").write_text(
            json.dumps(config, indent=2) + "\n", encoding="utf-8"
        )
        save(weights, root / "backbone" / "model.safetensors")
    torch.manual_seed(seed + 2)
    head = CandidateHead(config["hidden_size"], head_dim)
    for parameter in head.parameters():
        torch.nn.init.normal_(parameter, std=0.2)
    save(
        {name: tensor.detach().float() for name, tensor in head.state_dict().items()},
        root / "decision_head.safetensors",
    )
    (root / "decision_config.json").write_text(
        json.dumps(decision_config, indent=2) + "\n", encoding="utf-8"
    )

    model_sha256 = pkg.model_identity(root, decision_config, base_root)
    bias_entry = None
    if score_bias and not adapter:
        offsets = {"3": [0.05, -0.02, 0.01], "5": [0.04, 0.2, 0.08, -0.15, -0.17]}
        report = {
            "format": pkg.SCORE_BIAS_FORMAT,
            "model_sha256": model_sha256,
            "offsets": offsets,
            "fit": {"source": "fixture"},
        }
        (root / "score_bias.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        bias_entry = {
            "file": "score_bias.json",
            "offsets": offsets,
            "sha256": sha256_file(root / "score_bias.json"),
        }
    # Bundled package code that must never run.
    (root / "decision2").mkdir()
    (root / "decision2" / "__init__.py").write_text(
        'raise RuntimeError("vllm_srun must never import package code")\n',
        encoding="utf-8",
    )
    (root / "README.md").write_text(
        "# Tiny Decision 2.0 fixture\n\nRandom weights for tests.\n", encoding="utf-8"
    )

    model_name = f"Decision-2.0-Tiny-{'Qwen3' if backbone == 'qwen3' else 'Qwen3.5'}{'-LoRA' if adapter else ''}"
    weight_groups = {"head": ["decision_head.safetensors"], "residual": []}
    if adapter:
        weight_groups["adapter"] = ["adapter/adapter_model.safetensors"]
    else:
        weight_groups["backbone"] = ["backbone/model.safetensors"]
    packaged = {
        group: sum(safetensors_elements(root / name) for name in names)
        for group, names in weight_groups.items()
    }
    base_text = sum(tensor.numel() for tensor in weights.values())
    loaded = packaged["head"] + (
        packaged["adapter"] + base_text if adapter else packaged["backbone"]
    )
    pointer = {
        **pkg.POINTER,
        "model_name": model_name,
        "runtime_family": (
            "decision2-qwen-adapter" if adapter else "decision2-qwen-full"
        ),
        "manifest": pkg.MANIFEST_NAME,
        "max_input_tokens": max_input_tokens,
    }
    (root / "config.json").write_text(
        json.dumps(pointer, indent=2) + "\n", encoding="utf-8"
    )
    files = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    manifest: dict[str, Any] = {
        "schema": pkg.MANIFEST_SCHEMA,
        "kind": "fixture",
        "model_name": model_name,
        "profile": "qwen-adapter" if adapter else "qwen-full",
        "files_sha256": files,
        "identity": {"model_sha256": model_sha256},
        "max_input_tokens": max_input_tokens,
        "licence": {"spdx": "apache-2.0", "components": []},
        "calibration": None,
        "parameters": {
            "loaded": loaded,
            "packaged": packaged,
            "packaged_files": weight_groups,
        },
    }
    if bias_entry:
        manifest["score_bias"] = bias_entry
    if adapter:
        manifest["base"] = {
            "repo_id": "vllm-sr-fixtures/tiny-base",
            "revision": FAKE_BASE_REVISION,
            "files_sha256": {
                path.name: sha256_file(path)
                for path in sorted(base_root.iterdir())
                if path.is_file()
            },
        }
    (root / pkg.MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return root


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
