"""Tiny random-weight Vela 2.0 packages for tests: a 0.3B-style encoder and a decoder (4B-style by default).

Both follow the released layouts: ``config.json`` naming the member,
``calibration.json`` with the temperatures, thresholds, PII rule and schemas
the family reads, the tokenizer, and safetensors weights under the packages'
parameter names. The encoder's tokenizer carries the eight marker tokens the
family must strip; the decoder ships a sharded index, a broad span head,
``SHA256SUMS`` and ``MODEL_MANIFEST.json``. The bundled engine raises on import.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from ..families.vela2.encoder_layout import MARKERS
from ..heads.candidate import CandidateHead
from ..heads.marker import MarkerHead
from ..heads.span import SpanHead
from ..registry.artifacts import sha256_file
from .fixtures import CORPUS, modernbert_config, qwen3_5_config, random_backbone, save

PII_TYPES = [
    "AGE",
    "CREDIT_CARD",
    "DATE_TIME",
    "DOMAIN_NAME",
    "EMAIL_ADDRESS",
    "GPE",
    "IBAN_CODE",
    "IP_ADDRESS",
    "NRP",
    "ORGANIZATION",
    "PERSON",
    "PHONE_NUMBER",
    "STREET_ADDRESS",
    "TITLE",
    "US_DRIVER_LICENSE",
    "US_SSN",
    "ZIP_CODE",
]
TEXT = [
    *CORPUS,
    "Hi, I'm Tom Baker (tom.baker@example.com). What is the maximum daily dose of paracetamol for an adult?",
    "For adults, the maximum dose of paracetamol is 4 grams in 24 hours. Visit https://example.org/dose today.",
    "Which spans are personal information? a person's name an e-mail address a claim not supported by the context",
    'Task type: span Target: user Question: Labels: <label> </label> Text: Spans: <segment role="user"> </segment>',
]
PROJECTION, SLOTS = 32, 8


def calibration(decoder: bool) -> dict[str, Any]:
    value: dict[str, Any] = {
        "temperature": {"choice": 1.3, "score": 1.1, "set": 1.05, "span": 0.4},
        "thresholds": {
            "set:*": 0.3,
            "span:toxic": 0.1,
            "span:halu": 0.55,
            "span:*": 0.5,
        },
        "pii_types": PII_TYPES,
        "pii_length_rule": {
            "anchors": [
                {"n_tokens": 21, "threshold": 0.75},
                {"n_tokens": 126, "threshold": 0.05},
                {"n_tokens": 1022, "threshold": 0.005919},
            ]
        },
        "pii_sparse_gate": {"K": 3, "t_sparse": 0.1, "probe_threshold": 0.5},
        "pii_schema": {
            "text": "Which spans are personal information?",
            "labels": {name: name.lower().replace("_", " ") for name in PII_TYPES},
        },
        "halu_schema": {
            "text": "Which spans of the answer are not supported by the context?",
            "labels": {"unsupported": "a claim not supported by the context"},
        },
        "relevance_schema": {
            "text": "How relevant is this candidate to the query?",
            "levels": {
                "not_relevant": "The candidate does not help with the query.",
                "partially_relevant": "The candidate partly helps.",
                "highly_relevant": "The candidate directly helps with the query.",
            },
            "values": {
                "not_relevant": 0.0,
                "partially_relevant": 0.5,
                "highly_relevant": 1.0,
            },
        },
    }
    if decoder:
        value["broad_head"] = {
            "temperature": 1.0,
            "threshold": 0.55,
            "router_label_sets": [PII_TYPES, ["unsupported"]],
        }
        value["noul_calibration"] = {"T": 2.0, "b": 0.25, "default": False}
    return value


def _tokenizer(root: Path, special: list[str], markers: bool) -> int:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False, use_regex=True
    )
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=384,
        special_tokens=special,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train_from_iterator(TEXT * 4, trainer=trainer)
    if markers:
        tokenizer.add_special_tokens(list(MARKERS))
    tokenizer.save(str(root / "tokenizer.json"))
    return tokenizer.get_vocab_size()


def _random(module: torch.nn.Module, seed: int, prefix: str) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    for parameter in module.parameters():
        torch.nn.init.normal_(parameter, std=0.2)
    return {
        f"{prefix}{name}": value.detach().float()
        for name, value in module.state_dict().items()
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _bundled_code(root: Path) -> None:
    (root / "vela2_inference.py").write_text(
        'raise RuntimeError("vllm_srun must never import package code")\n',
        encoding="utf-8",
    )


def write_encoder_package(
    output: str | Path,
    *,
    seed: int = 0,
    max_length: int = 2048,
    window_overlap: int = 64,
) -> Path:
    """A 0.3B-style package: a tiny ModernBERT with marker tokens and the marker head."""
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    vocab = _tokenizer(root, ["<pad>", "<eos>", "<bos>"], markers=True)
    encoder = modernbert_config(vocab)
    marker_ids = {
        name: vocab - len(MARKERS) + index for index, name in enumerate(MARKERS)
    }
    tensors = {
        f"encoder.{k}": v
        for k, v in random_backbone("modernbert", encoder, seed).items()
    }
    head = MarkerHead(encoder["hidden_size"], PROJECTION, "both")
    tensors.update(_random(head, seed + 1, ""))
    tensors["log_tau"] = torch.tensor(-2.8)
    tensors["log_tau_span"] = torch.tensor(-2.6)
    save(tensors, root / "model.safetensors")
    _write_json(
        root / "config.json",
        {
            "architectures": ["Vela2Model"],
            "model_type": "vela2-unified",
            "format_version": 1,
            "encoder_config": encoder,
            "proj_dim": PROJECTION,
            "readout": "both",
            "markers": list(MARKERS),
            "marker_ids": marker_ids,
            "bos_token_id": 2,
            "eos_token_id": 1,
            "pad_token_id": 0,
            "max_length": max_length,
            "window_overlap": window_overlap,
        },
    )
    _write_json(root / "calibration.json", calibration(decoder=False))
    _bundled_code(root)
    return root


def write_decoder_package(
    output: str | Path,
    *,
    seed: int = 0,
    max_length: int = 4096,
    broad: bool = True,
    repeat_limit: int = 48,
    backbone: dict[str, Any] | None = None,
) -> Path:
    """A decoder package: a tiny Qwen3.5 backbone (4B-shaped unless ``backbone`` overrides its
    config), the candidate and span heads, sharded."""
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    vocab = _tokenizer(root, ["<|endoftext|>"], markers=False)
    config = qwen3_5_config(vocab, **(backbone or {}))
    hidden = config["hidden_size"]
    tensors = {
        f"backbone.{k}": v for k, v in random_backbone("qwen3_5", config, seed).items()
    }
    heads = _random(CandidateHead(hidden, PROJECTION), seed + 1, "head.")
    heads["set_bias"] = torch.tensor(0.3)
    heads.update(_random(SpanHead(hidden, PROJECTION, SLOTS), seed + 2, "span2."))
    save(tensors, root / "model-00001-of-00002.safetensors")
    save(heads, root / "model-00002-of-00002.safetensors")
    weight_map = dict.fromkeys(tensors, "model-00001-of-00002.safetensors")
    weight_map.update(dict.fromkeys(heads, "model-00002-of-00002.safetensors"))
    _write_json(
        root / "model.safetensors.index.json",
        {"metadata": {}, "weight_map": weight_map},
    )
    if broad:
        save(
            _random(SpanHead(hidden, PROJECTION, SLOTS), seed + 3, "span_broad."),
            root / "broad_head.safetensors",
        )
    _write_json(
        root / "config.json",
        {
            "architectures": ["Vela2DecoderModel"],
            "model_type": "vela2-decoder",
            "format_version": 1,
            "model_name": "vllm-sr-fixtures/Vela-2.0-Tiny",
            "backbone_config": config,
            "head_dim": PROJECTION,
            "span_head": {"d": PROJECTION, "slots": SLOTS},
            "pad_token_id": 0,
            "max_length": max_length,
            "span_layout": {
                "hybrid_rmax": repeat_limit,
                "window": repeat_limit - 8,
                "stride": repeat_limit - 16,
                "window_above": repeat_limit,
            },
        },
    )
    _write_json(root / "calibration.json", calibration(decoder=True))
    _bundled_code(root)
    names = sorted(path.name for path in root.iterdir() if path.is_file())
    digests = {name: sha256_file(root / name) for name in names}
    _write_json(
        root / "MODEL_MANIFEST.json",
        {"format": "vela2-decoder-release/1", "files_sha256": digests},
    )
    digests["MODEL_MANIFEST.json"] = sha256_file(root / "MODEL_MANIFEST.json")
    (root / "SHA256SUMS").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(digests.items())),
        encoding="utf-8",
    )
    return root


WRITERS = {"encoder": write_encoder_package, "decoder": write_decoder_package}
VARIANTS = tuple(WRITERS)


def write_fixture(output: str | Path, variant: str | None, seed: int) -> Path:
    """The fixture command's writer: the 0.3B-style encoder or the 4B-style decoder."""
    return WRITERS[variant or VARIANTS[0]](output, seed=seed)
