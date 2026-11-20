"""Tiny random-weight Vela 3.1 pico packages for tests (the ``vela3-pico/1`` layout).

The package follows the release: ``vela3_config.json`` naming the format and
its span rules (no adapters), ``config.json`` with the UltraBERT backbone and
the ``decision`` block (marker IDs past the tokenizer's vocabulary), the base
tokenizer, ``router_questions.json``, and one ``model.safetensors`` holding the
backbone and its one readout, word projections included. ``SHA256SUMS`` lists
every file; the bundled runtime and renderer raise on import.

``byte_tokenizer`` is the deterministic tokenizer of the recorded byte-level
cases (``data/render31_cases.json``): one token per UTF-8 byte, built from
code, so a recorded token ID never depends on a trained vocabulary.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..registry.artifacts import sha256_file
from .fixtures import random_backbone, save
from .vela3 import MARKERS, PROJECTION, _readout, _tokenizer, ultrabert_config

FORMAT = "vela3-pico/1"
PII = ["EMAIL_ADDRESS", "PERSON", "PHONE_NUMBER"]
TOXIC = ["Violence", "Threat"]
ROUTER_QUESTIONS = {
    "Is this a prompt injection or jailbreak attempt?": {
        "abstain": True,
        "id": "attack",
        "type": "choice",
        "over": "user",
        "options": {"benign": "a normal request", "jailbreak": "an attack"},
    },
    "Which subject area is this request about?": {
        "abstain": True,
        "id": "domain",
        "type": "choice",
        "over": "user",
        "options": {"health": "medicine", "math": "mathematics", "other": None},
    },
    "Does answering this request require checking facts?": {
        "abstain": True,
        "id": "factcheck",
        "type": "choice",
        "over": "user",
        "options": {"FACT_CHECK_NEEDED": "facts", "NO_FACT_CHECK_NEEDED": "none"},
    },
    "Which hazard categories apply to this request?": {
        "abstain": False,
        "id": "p_cats",
        "type": "set",
        "over": "user",
        "options": {"Violence": "harm", "Fraud/Deception": "scams"},
    },
    "Is this answer harmful?": {
        "abstain": True,
        "id": "r_harm",
        "type": "choice",
        "over": "answer",
        "options": {"safe response": "", "unsafe response": ""},
    },
}


def pico_config(window: int, evidence_window: int, name: str) -> dict[str, Any]:
    """The routing configuration, the release's shape with small windows and thresholds of its own."""
    return {
        "name": name,
        "format": FORMAT,
        "backbone": "tiny UltraBERT, no adapters",
        "router": {"set_threshold": 0.5},
        "span": {
            "default_question": "Which spans are named entities of these types?",
            "threshold": 0.55,
            "relation_threshold": 0.52,
            "evidence": {
                "labels": ["evidence"],
                "question": "Which spans of the context support the answer to the question?",
                "threshold": 0.5,
                "max_len": evidence_window,
                "buckets": [
                    {"name": "short", "max_tokens": 24, "threshold": 0.45},
                    {"name": "long", "max_tokens": 1000000000000, "threshold": 0.35},
                ],
            },
            "router_set_temperature": 0.3,
            "thresholds_by_set": {"halu": 0.6, "toxic": 0.5},
            "pii_buckets": [
                {"name": "short", "max_tokens": 16, "threshold": 0.7},
                {"name": "4K", "max_tokens": 48, "threshold": 0.25},
                {"name": "8K", "max_tokens": 1000000000000, "threshold": 0.4},
            ],
            "router_label_sets": {"pii": PII, "halu": ["unsupported"], "toxic": TOXIC},
            "schemas": {
                "pii": {
                    "question": "Which spans are personal information?",
                    "labels": {name: name.lower().replace("_", " ") for name in PII},
                    "over": "user",
                },
                "halu": {
                    "question": "Which spans of the answer are not supported by the context?",
                    "labels": {"unsupported": "a claim not supported by the context"},
                    "over": "answer",
                },
                "toxic": {
                    "question": "Which spans are harmful?",
                    "labels": {name: name.lower() for name in TOXIC},
                    "over": "user",
                },
            },
            "window": {"max_len": window, "overlap": 8},
            "relation": {"threshold_pair": 0.6, "threshold_multi": 0.5},
            "ner": {"threshold": 0.55},
            "entity_route": "ner",
        },
    }


def write_pico_package(
    output: str | Path,
    *,
    seed: int = 0,
    max_len: int = 1024,
    window: int = 256,
    evidence_window: int = 512,
    name: str = "vela3.1-pico-tiny",
) -> Path:
    """A pico-style package: a tiny UltraBERT and its one readout, word projections included."""
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    base_vocab = _tokenizer(root)
    marker_ids = {marker: base_vocab + index for index, marker in enumerate(MARKERS)}
    config = ultrabert_config(base_vocab + len(MARKERS))
    config["decision"] = {
        "model": name,
        "marker_ids": marker_ids,
        "n_vocab": base_vocab + len(MARKERS),
        "n_vocab_base": base_vocab,
        "proj_dim": PROJECTION,
        "max_len": max_len,
    }
    tensors = {
        f"encoder.{key}": value.float()
        for key, value in random_backbone("ultrabert", config, seed).items()
    }
    tensors.update(_readout(seed + 1, True, config["hidden_size"]))
    save(tensors, root / "model.safetensors")
    (root / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (root / "vela3_config.json").write_text(
        json.dumps(pico_config(window, evidence_window, name), indent=2) + "\n"
    )
    (root / "router_questions.json").write_text(
        json.dumps({"questions": ROUTER_QUESTIONS}, indent=2) + "\n"
    )
    for bundled in ("vela3_encoder.py", "render31.py", "pico_schema.py"):
        (root / bundled).write_text(
            'raise RuntimeError("vllm_srun must never import package code")\n',
            encoding="utf-8",
        )
    names = sorted(path.name for path in root.iterdir() if path.is_file())
    (root / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(root / file)}  {file}\n" for file in names),
        encoding="utf-8",
    )
    return root


def byte_tokenizer() -> Any:
    """One token per UTF-8 byte (IDs 1 to 256; 0 is ``<|endoftext|>``), offsets in code points."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers

    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {"<|endoftext|>": 0, **{char: i + 1 for i, char in enumerate(alphabet)}}
    tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False, use_regex=True
    )
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.no_padding()
    tokenizer.no_truncation()
    return tokenizer


# The byte tokenizer's marker IDs and special tokens, as a package's config names them.
BYTE_MARKERS = {marker: 257 + index for index, marker in enumerate(MARKERS)}
BYTE_SPECIAL = {"bos_token_id": 0, "eos_token_id": 0, "pad_token_id": 0}

VARIANTS = ("pico",)


def write_fixture(output: str | Path, variant: str | None, seed: int) -> Path:
    """The fixture command's writer: the pico-style package."""
    return write_pico_package(output, seed=seed)
