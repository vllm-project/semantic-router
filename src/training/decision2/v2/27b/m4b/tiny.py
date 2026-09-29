"""Tiny random Qwen sources and schema-valid rows for the M4b parity check and CPU tests.

``write_source`` saves a randomly initialized official-architecture model plus a
word-level tokenizer so that ``DecisionModel.from_base`` loads it like the real
base. ``qwen3_5`` mirrors the 27B layout (gated-delta linear-attention and
full-attention decoder layers, a vision tower, untied LM head) at toy sizes;
``qwen3`` is the fallback for Transformers builds without Qwen3.5.
"""

from __future__ import annotations

import importlib.util
import json
import random
from pathlib import Path
from typing import Any

from training.model.data import digest

WORDS = (
    "context question options option key description select the single best "
    "supported by and instructions decision task type choice noul score true "
    "false yes no a b c d level low high"
).split()
VOCAB = 64


def has_qwen3_5() -> bool:
    return importlib.util.find_spec("transformers.models.qwen3_5") is not None


def write_tokenizer(path: Path) -> None:
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {"[UNK]": 0, "[PAD]": 1}
    for word in WORDS:
        vocab.setdefault(word, len(vocab))
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]"
    ).save_pretrained(path)


def text_config(arch: str, layers: int) -> dict[str, Any]:
    common = {
        "vocab_size": VOCAB,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": layers,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "max_position_embeddings": 1024,
        "tie_word_embeddings": False,
        "pad_token_id": 1,
    }
    if arch == "qwen3":
        return common
    # FLA and causal-conv1d kernels need power-of-two head sizes of at least 16.
    return {
        **common,
        "linear_num_value_heads": 4,
        "linear_num_key_heads": 2,
        "linear_key_head_dim": 32,
        "linear_value_head_dim": 32,
        "linear_conv_kernel_dim": 4,
        "layer_types": [
            "full_attention" if (index + 1) % 4 == 0 else "linear_attention"
            for index in range(layers)
        ],
    }


def write_source(
    path: Path, *, arch: str = "qwen3_5", layers: int = 4, seed: int = 7
) -> Path:
    import torch

    path.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    if arch == "qwen3_5":
        from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

        config = Qwen3_5Config(
            text_config=text_config(arch, layers),
            vision_config={
                "depth": 1,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_heads": 2,
                "out_hidden_size": 64,
                "num_position_embeddings": 64,
            },
            tie_word_embeddings=False,
        )
        model = Qwen3_5ForConditionalGeneration(config)
    elif arch == "qwen3":
        from transformers import Qwen3Config, Qwen3ForCausalLM

        model = Qwen3ForCausalLM(Qwen3Config(**text_config(arch, layers)))
    else:
        raise ValueError("arch must be qwen3_5 or qwen3")
    model.float().save_pretrained(path, safe_serialization=True)
    write_tokenizer(path)
    return path


def make_row(index: int, split: str, rng: random.Random, prefix: str) -> dict[str, Any]:
    kind = ("choice", "noul", "score")[index % 3]
    if kind == "noul":
        keys = ["false", "true"]
    elif kind == "score":
        keys = [str(level) for level in range(rng.randint(2, 5))]
    else:
        keys = [chr(ord("a") + k) for k in range(rng.randint(2, 4))]

    def words(count: int) -> str:
        return " ".join(rng.choice(WORDS) for _ in range(count))

    row = {
        "id": f"{prefix}{split}-{index}",
        "state": words(rng.randint(3, 40)),
        "instructions": words(rng.randint(2, 8)),
        "options": [
            {"key": key, "description": words(rng.randint(1, 6))} for key in keys
        ],
        "label": rng.randrange(len(keys)),
        "task_type": kind,
        "family": f"family-{index % 4}",
        "group_id": f"{prefix}{split}-group-{index}",
        "language": "en",
        "split": split,
        "source": "tiny",
        "evaluation_role": split,
        "render_template": "tiny",
        "audit_metadata": {},
    }
    row["input_sha256"] = digest(
        {
            field: row[field]
            for field in ("state", "instructions", "options", "task_type")
        }
    )
    return row


def make_rows(
    count: int, split: str, seed: int = 0, prefix: str = ""
) -> list[dict[str, Any]]:
    rng = random.Random(f"{seed}:{split}")
    return [make_row(index, split, rng, prefix) for index in range(count)]


def write_rows(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path
