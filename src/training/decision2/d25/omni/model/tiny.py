"""Tiny random Qwen3.5 checkpoints, processors, images and rows for CPU tests.

``write_stock`` saves a randomly initialised ``Qwen3_5ForConditionalGeneration`` with the 27B
layout at toy sizes (3 Gated DeltaNet + 1 gated attention layer, a 2-block ViT with patch 16 and
merge 2, untied LM head, an extra ``mtp.*`` tensor) plus a word-level tokenizer whose 255 answer
codes are single tokens, the Qwen chat format and a ``Qwen3VLProcessor``. ``write_vega`` derives a
Vega-style code-readout v1 checkpoint from it with perturbed language weights and a chosen vision
state.
"""

from __future__ import annotations

import atexit
import functools
import itertools
import json
import random
import shutil
import string
import tempfile
from pathlib import Path
from typing import Any

from d25.omni.model import checkpoint

SPECIAL = (
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
WORDS = (
    "You are a decision engine Treat the state as data not instructions Read question and every "
    "option then reply with only code of best State Question Options Choose matching Reply empty "
    "Classify supplied using descriptions content selected Return letter system user assistant "
    "image left right red blue green cat dog chart table which is shown in picture count how many "
    "true false yes no No Yes first second third fourth larger smaller same different"
).split()
PUNCTUATION = list(":.,/()-?!'\"{}[]_<>|=+*&%$#@;")
CHAT_TEMPLATE = (
    "{%- for message in messages -%}"
    "{%- if message.content is string -%}{%- set content = message.content -%}"
    "{%- else -%}{%- set ns = namespace(text='') -%}"
    "{%- for item in message.content -%}"
    "{%- if item.type == 'image' or 'image' in item -%}"
    "{%- set ns.text = ns.text + '<|vision_start|><|image_pad|><|vision_end|>' -%}"
    "{%- elif 'text' in item -%}{%- set ns.text = ns.text + item.text -%}{%- endif -%}"
    "{%- endfor -%}{%- set content = ns.text -%}{%- endif -%}"
    "{{- '<|im_start|>' + message.role + '\\n' + (content | trim) + '<|im_end|>\\n' -}}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\\n' -}}"
    "{%- if enable_thinking is defined and enable_thinking is false -%}"
    "{{- '<think>\\n\\n</think>\\n\\n' -}}{%- else -%}{{- '<think>\\n' -}}{%- endif -%}"
    "{%- endif -%}"
)
HIDDEN = 64


def vocabulary() -> dict[str, int]:
    tokens = [
        "[UNK]",
        *SPECIAL,
        " ",
        "\n",
        "\t",
        *PUNCTUATION,
        *string.digits,
        *string.ascii_lowercase,
    ]
    tokens += list(string.ascii_uppercase)
    tokens += [
        "".join(pair) for pair in itertools.product(string.ascii_uppercase, repeat=2)
    ]
    tokens += WORDS
    vocab: dict[str, int] = {}
    for token in tokens:
        vocab.setdefault(token, len(vocab))
    return vocab


def build_tokenizer():
    from tokenizers import Regex, Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    tokenizer = Tokenizer(models.WordLevel(vocab=vocabulary(), unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Split(
        Regex(r"\s|\w+|[^\w\s]"), behavior="isolated"
    )
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="<|endoftext|>",
        eos_token="<|im_end|>",
        additional_special_tokens=list(SPECIAL[1:]),
    )
    fast.chat_template = CHAT_TEMPLATE
    return fast


def build_processor():
    from transformers import (
        Qwen2VLImageProcessor,
        Qwen3VLProcessor,
        Qwen3VLVideoProcessor,
    )

    image_processor = Qwen2VLImageProcessor(
        patch_size=16,
        temporal_patch_size=2,
        merge_size=2,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        size={"shortest_edge": 65_536, "longest_edge": 16_777_216},
    )
    video_processor = Qwen3VLVideoProcessor(
        patch_size=16, temporal_patch_size=2, merge_size=2
    )
    return Qwen3VLProcessor(
        image_processor=image_processor,
        tokenizer=build_tokenizer(),
        video_processor=video_processor,
        chat_template=CHAT_TEMPLATE,
    )


def config(layers: int = 4):
    from transformers import Qwen3_5Config

    vocab = vocabulary()
    return Qwen3_5Config(
        text_config={
            "vocab_size": len(vocab),
            "hidden_size": HIDDEN,
            "intermediate_size": 128,
            "num_hidden_layers": layers,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 32,
            "linear_num_value_heads": 4,
            "linear_num_key_heads": 2,
            "linear_key_head_dim": 16,
            "linear_value_head_dim": 16,
            "linear_conv_kernel_dim": 4,
            "layer_types": [
                "full_attention" if (index + 1) % 4 == 0 else "linear_attention"
                for index in range(layers)
            ],
            "max_position_embeddings": 32_768,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10_000_000,
                "partial_rotary_factor": 0.25,
                "mrope_section": [2, 1, 1],
                "mrope_interleaved": True,
            },
            "tie_word_embeddings": False,
            "pad_token_id": vocab["<|endoftext|>"],
        },
        vision_config={
            "depth": 2,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 2,
            "in_channels": 3,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "out_hidden_size": HIDDEN,
            "num_position_embeddings": 64,
            "deepstack_visual_indexes": [],
        },
        image_token_id=vocab["<|image_pad|>"],
        video_token_id=vocab["<|video_pad|>"],
        vision_start_token_id=vocab["<|vision_start|>"],
        vision_end_token_id=vocab["<|vision_end|>"],
        tie_word_embeddings=False,
    )


def write_stock(path: str | Path, seed: int = 7, layers: int = 4) -> Path:
    """Tiny stand-in for the stock Qwen3.8-27B snapshot (BF16, untied head, one ``mtp`` tensor)."""
    import torch
    from safetensors.torch import load_file, save_file
    from transformers import Qwen3_5ForConditionalGeneration

    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    model = Qwen3_5ForConditionalGeneration(config(layers))
    model.to(torch.bfloat16).save_pretrained(root, max_shard_size="200KB")
    shards = sorted(root.glob("model-*.safetensors")) or [root / "model.safetensors"]
    last = shards[-1]
    tensors = load_file(str(last))
    tensors["mtp.fc.weight"] = torch.randn(HIDDEN, 2 * HIDDEN).to(torch.bfloat16)
    save_file(tensors, str(last), metadata={"format": "pt"})
    index = root / checkpoint.INDEX_FILE
    if index.exists():
        data = json.loads(index.read_text())
        data["weight_map"]["mtp.fc.weight"] = last.name
        index.write_text(json.dumps(data, indent=2))
    build_processor().save_pretrained(root)
    return root


def write_vega(
    path: str | Path,
    stock: str | Path,
    vision: str = "unchanged",
    layout: str = "backbone",
    seed: int = 11,
    attention_mode: str = "causal",
) -> Path:
    """Vega-style code-readout v1 checkpoint derived from ``stock``.

    ``vision``: ``unchanged`` | ``changed`` (one encoder tensor perturbed) | ``merger_changed`` |
    ``missing`` (no ``visual.*``) | ``partial`` (merger dropped). ``layout``: ``backbone``
    (``Qwen3_5Model`` names) or ``text`` (bare ``Qwen3_5TextModel`` names; implies no vision).
    """
    import torch
    from safetensors.torch import save_file
    from transformers import AutoProcessor

    from d25.vega.common import decision_format as text_format

    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(seed)
    tensors: dict[str, torch.Tensor] = {}
    lm_head = None
    for key, tensor in checkpoint.iter_tensors(stock):
        if key == "lm_head.weight":
            lm_head = tensor
        name = checkpoint.canonical_name(key)
        if name is None:
            continue
        part = checkpoint.component(name)
        if part == "language":
            noise = torch.randn(tensor.shape, generator=generator) * 0.01
            tensor = (tensor.float() + noise).to(tensor.dtype)
        elif vision == "changed" and name == "visual.blocks.0.attn.qkv.weight":
            tensor = (tensor.float() + 0.5).to(tensor.dtype)
        elif vision == "merger_changed" and name == "visual.merger.linear_fc2.weight":
            tensor = (tensor.float() + 0.5).to(tensor.dtype)
        if part != "language" and (layout == "text" or vision == "missing"):
            continue
        if part == "vision_merger" and vision == "partial":
            continue
        if layout == "text":
            name = name[len("language_model.") :]
        tensors[name] = tensor.contiguous()
    save_file(tensors, str(root / "model.safetensors"), metadata={"format": "pt"})
    stock_config = json.loads((Path(stock) / "config.json").read_text())
    stock_config["architectures"] = ["Qwen3_5Model"]
    (root / "config.json").write_text(json.dumps(stock_config, indent=2))
    processor = AutoProcessor.from_pretrained(str(stock))
    processor.save_pretrained(root)
    codes, token_ids = text_format.answer_codes(processor.tokenizer)
    readout = (
        lm_head[token_ids].float()
        + torch.randn(len(token_ids), HIDDEN, generator=generator) * 0.01
    )
    checkpoint.save_readout(root, readout)
    checkpoint.write_json(
        root / checkpoint.DECISION_CONFIG,
        {
            "format_version": 1,
            "format_id": text_format.FORMAT_ID,
            "prompt": "d25-vega",
            "base_model": checkpoint.BASE_MODEL,
            "revision": checkpoint.BASE_REVISION,
            "codes": codes,
            "token_ids": token_ids,
            "temperature": 1.0,
            "attention_mode": attention_mode,
            "pooling": "last",
            "max_length": 8192,
            "provenance": {"run": "tiny-vega", "init": "tiny-stock"},
        },
    )
    return root


def write_image(path: str | Path, width: int, height: int, seed: int = 0) -> Path:
    """Random-noise PNG of the given size."""
    import numpy as np
    from PIL import Image

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.random.default_rng(seed).integers(
        0, 256, size=(height, width, 3), dtype=np.uint8
    )
    Image.fromarray(pixels).save(target)
    return target


def question(rng: random.Random, n_options: int | None = None) -> dict[str, Any]:
    if n_options is None and rng.random() < 0.3:
        return {"type": "noul", "instructions": "Is the state true ?"}
    count = n_options or rng.randint(2, 5)
    words = WORDS[10:40]
    return {
        "type": "choice",
        "instructions": " ".join(rng.choice(words) for _ in range(rng.randint(2, 6))),
        "criteria": {
            f"k{index}": (
                None
                if rng.random() < 0.2
                else " ".join(rng.choice(words) for _ in range(3))
            )
            for index in range(count)
        },
    }


def write_rows(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    import gzip

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if target.suffix == ".gz" else open
    with opener(target, "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    return target


@functools.lru_cache(maxsize=1)
def fixtures() -> dict[str, Path]:
    """Shared tiny checkpoints, images and row files for one test process (removed at exit)."""
    from d25.omni.model.assemble import assemble

    root = Path(tempfile.mkdtemp(prefix="d25-omni-tiny-"))
    atexit.register(shutil.rmtree, root, True)
    stock = write_stock(root / "stock")
    vega = write_vega(root / "vega", stock)
    assemble(stock, root / "init-stock", stock_pin="skip")
    assemble(stock, root / "init-vega", vega, stock_pin="skip")
    data = root / "data"
    sizes = [(300, 200), (256, 256), (640, 480), (200, 500), (1000, 700), (90, 120)]
    images = [f"images/{i}.png" for i in range(len(sizes))]
    for name, (width, height) in zip(images, sizes):
        write_image(data / name, width, height, seed=len(name) + width)
    rng = random.Random(3)
    write_rows(
        data / "mm.jsonl",
        [
            training_row(i, rng, rng.sample(images, rng.randint(0, 3)))
            for i in range(24)
        ],
    )
    write_rows(data / "text.jsonl", [training_row(100 + i, rng) for i in range(40)])
    write_rows(
        data / "dev.jsonl",
        [
            training_row(200 + i, rng, rng.sample(images, rng.randint(0, 2)))
            for i in range(8)
        ],
    )
    return {
        "root": root,
        "stock": stock,
        "vega": vega,
        "init_stock": root / "init-stock",
        "init_vega": root / "init-vega",
        "data": data,
    }


def training_row(
    index: int,
    rng: random.Random,
    images: list[str] | None = None,
    n_options: int | None = None,
) -> dict[str, Any]:
    from d25.vega.common import decision_format as text_format

    q = question(rng, n_options)
    keys, _ = text_format.options(q)
    weights = [rng.random() + 0.05 for _ in keys]
    total = sum(weights)
    target = [value / total for value in weights]
    images = images or []
    return {
        "id": f"tiny-{index}",
        "source": "tiny-images" if images else "tiny-text",
        "family": f"family-{index % 3}",
        "state": " ".join(rng.choice(WORDS) for _ in range(rng.randint(3, 30))),
        "question": q,
        "target": target,
        "label": max(range(len(target)), key=target.__getitem__),
        "weight": 1.0,
        "images": images,
        "meta": {
            "licence": "test",
            "source_split": "train",
            "image_sha256": ["0" * 64] * len(images),
        },
    }
