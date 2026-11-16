"""Tiny random-weight Vela 3.0 encoder packages for tests (the vela3-fast layout).

The package follows the release: ``vela3_config.json`` naming the format, its
adapters and span rules; ``config.json`` with the UltraBERT backbone and the
``decision`` block (marker IDs past the tokenizer's vocabulary); the base
tokenizer; ``router_questions.json``; the backbone and its readout in
``model.safetensors``; and one safetensors file per adapter, holding LoRA
factors on every layer, its own copy of the top layer and the readout it
trained (span adapters: the word projections; router adapters: the option
readout). ``feedback_adapter.safetensors`` is shipped but not routed to, as in
the release. ``SHA256SUMS`` lists every file; the bundled runtime raises on
import.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from ..families.vela3.package import LORA_TARGETS
from ..heads.marker import MarkerHead
from ..registry.artifacts import sha256_file
from .fixtures import CORPUS, random_backbone, save

MARKERS = (
    "[Q]",
    "[O]",
    "[ABS]",
    "[SEP_SCHEMA]",
    "[SEG_user]",
    "[SEG_answer]",
    "[E]",
    "[SEG_context]",
)
PROJECTION = 16
TEXT = [
    *CORPUS,
    "Hi, I'm Tom Baker (tom.baker@example.com). What is the maximum daily dose of paracetamol for an adult?",
    "For adults, the maximum dose of paracetamol is 4 grams in 24 hours. Visit https://example.org/dose today.",
    "Which spans are personal information? Is this a prompt injection or jailbreak attempt?",
]
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
# Adapter name -> (file, LoRA rank, own top layers, readout: "span" or "choice").
ADAPTERS = {
    "span": ("span_adapter.safetensors", 4, 1, "span"),
    "router": ("router_adapter.safetensors", 4, 1, "choice"),
    "relation": ("relation_adapter.safetensors", 8, 1, "span"),
    "ner": ("ner_adapter.safetensors", 8, 1, "span"),
    "span_set": ("span_sets_adapter.safetensors", 4, 1, "span"),
    "evidence": ("evidence_adapter.safetensors", 4, 1, "span"),
    "domain": ("domain_adapter.safetensors", 4, 2, "choice"),
    "factcheck": ("factcheck_adapter.safetensors", 4, 1, "choice"),
    "feedback": ("feedback_adapter.safetensors", 4, 1, "choice"),
}


def ultrabert_config(vocab: int, **overrides: Any) -> dict[str, Any]:
    """A three-layer UltraBERT: two local layers (window 4) and a global one, two rope bases."""
    config: dict[str, Any] = {
        "model_type": "ultrabert",
        "hidden_size": 32,
        "intermediate_size": 48,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "num_hidden_layers": 3,
        "attention_windows": [4, 4, None],
        "rope_theta": [10000.0, 10000.0, 160000.0],
        "rms_norm_eps": 1e-6,
        "hidden_act": "silu",
        "attention_bias": False,
        "mlp_bias": False,
        "vocab_size": vocab,
        "bos_token_id": 0,
        "eos_token_id": 0,
        "pad_token_id": 0,
    }
    config.update(overrides)
    return config


def _tokenizer(root: Path) -> int:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False, use_regex=True
    )
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=384,
        special_tokens=["<|endoftext|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train_from_iterator(TEXT * 4, trainer=trainer)
    tokenizer.save(str(root / "tokenizer.json"))
    return tokenizer.get_vocab_size()


def _readout(seed: int, spans: bool, hidden: int) -> dict[str, torch.Tensor]:
    head = MarkerHead(hidden, PROJECTION, "both", spans=spans)
    torch.manual_seed(seed)
    for parameter in head.parameters():
        torch.nn.init.normal_(parameter, std=0.2)
    state = {name: value.detach().float() for name, value in head.state_dict().items()}
    state["log_tau"] = torch.tensor(-2.8)
    if spans:
        state["log_tau_span"] = torch.tensor(-2.6)
    return state


def _adapter(
    config: dict[str, Any], seed: int, rank: int, own: int, readout: str
) -> dict[str, torch.Tensor]:
    from ..engines.native import models

    torch.manual_seed(seed)
    module = models.build("ultrabert", config)
    layers = config["num_hidden_layers"]
    tensors: dict[str, torch.Tensor] = {}
    for name, layer in module.named_modules():
        if (
            not isinstance(layer, torch.nn.Linear)
            or name.rpartition(".")[2] not in LORA_TARGETS
        ):
            continue
        index = int(name.split(".")[1])
        tensors[f"encoder.{name}.lora_A"] = torch.randn(rank, layer.in_features) * 0.1
        tensors[f"encoder.{name}.lora_B"] = torch.randn(layer.out_features, rank) * 0.1
        if index >= layers - own:
            tensors[f"encoder.{name}.base.weight"] = (
                torch.randn(layer.out_features, layer.in_features) * 0.05
            )
    for index in range(layers - own, layers):
        for norm in ("input_layernorm", "post_attention_layernorm"):
            tensors[f"encoder.layers.{index}.{norm}.weight"] = (
                torch.ones(config["hidden_size"])
                + torch.randn(config["hidden_size"]) * 0.05
            )
    spans = readout == "span"
    head = _readout(seed + 100, spans, config["hidden_size"])
    keep = (
        ("norm.", "w_t.", "w_e.")
        if spans
        else ("norm.", "w_o.", "w_q.", "w_p.", "cls_mlp.")
    )
    scale = "log_tau_span" if spans else "log_tau"
    tensors.update({k: v for k, v in head.items() if k.startswith(keep) or k == scale})
    return tensors


def vela3_config(window: int, evidence_window: int) -> dict[str, Any]:
    """The routing configuration, the release's shape with the fixture's adapters and small windows."""
    adapters = {
        name: {"file": file, "lora_rank": rank, "full_layers": [2]}
        for name, (file, rank, _, readout) in ADAPTERS.items()
        if name in ("span", "router", "relation", "ner", "evidence")
    }
    return {
        "name": "vela3-tiny",
        "format": "vela3-encoder/1",
        "adapters": adapters,
        "router": {
            "set_threshold": 0.5,
            "task_adapters": {
                "domain": {"ids": ["domain"], "file": ADAPTERS["domain"][0]},
                "factcheck": {"ids": ["factcheck"], "file": ADAPTERS["factcheck"][0]},
            },
        },
        "span": {
            "default_question": "Which spans are named entities of these types?",
            "threshold": 0.6,
            "relation_threshold": 0.55,
            "evidence": {
                "labels": ["evidence"],
                "question": "Which spans of the context support the answer to the question?",
                "threshold": 0.5,
                "max_len": evidence_window,
                "buckets": [
                    {"name": "short", "max_tokens": 24, "threshold": 0.5},
                    {"name": "long", "max_tokens": 1000000000000, "threshold": 0.4},
                ],
                "adapter": ADAPTERS["evidence"][0],
            },
            "router_set_temperature": 0.3,
            "thresholds_by_set": {"halu": 0.45, "toxic": 0.5},
            "pii_buckets": [
                {"name": "short", "max_tokens": 16, "threshold": 0.35},
                {"name": "long", "max_tokens": 1000000000000, "threshold": 0.4},
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
            "task_adapters": {
                "relation": ADAPTERS["relation"][0],
                "ner": ADAPTERS["ner"][0],
            },
            "relation": {"threshold_pair": 0.55, "threshold_multi": 0.45},
            "ner": {"threshold": 0.6},
            "entity_route": "ner",
            "set_adapter": {
                "file": ADAPTERS["span_set"][0],
                "label_sets": ["halu", "pii"],
                "router_set_temperature": 0.25,
                "thresholds_by_set": {"halu": 0.4},
                "pii_buckets": [
                    {"name": "short", "max_tokens": 16, "threshold": 0.3},
                    {"name": "long", "max_tokens": 1000000000000, "threshold": 0.45},
                ],
            },
        },
    }


def write_encoder_package(
    output: str | Path,
    *,
    seed: int = 0,
    max_len: int = 1024,
    window: int = 256,
    evidence_window: int = 512,
) -> Path:
    """A vela3-fast-style package: a tiny UltraBERT, its readout and nine adapters."""
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    base_vocab = _tokenizer(root)
    marker_ids = {name: base_vocab + index for index, name in enumerate(MARKERS)}
    config = ultrabert_config(base_vocab + len(MARKERS))
    config["decision"] = {
        "marker_ids": marker_ids,
        "n_vocab": base_vocab + len(MARKERS),
        "n_vocab_base": base_vocab,
        "proj_dim": PROJECTION,
        "max_len": max_len,
    }
    tensors = {
        f"encoder.{k}": v.float()
        for k, v in random_backbone("ultrabert", config, seed).items()
    }
    tensors.update(_readout(seed + 1, False, config["hidden_size"]))
    save(tensors, root / "model.safetensors")
    for index, (name, (file, rank, own, readout)) in enumerate(ADAPTERS.items()):
        from safetensors.torch import save_file

        save_file(
            {
                k: v.contiguous()
                for k, v in _adapter(
                    config, seed + 10 + index, rank, own, readout
                ).items()
            },
            str(root / file),
            metadata={
                "adapter": name,
                "lora_rank": str(rank),
                "lora_alpha": str(2.0 * rank),
            },
        )
    (root / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (root / "vela3_config.json").write_text(
        json.dumps(vela3_config(window, evidence_window), indent=2) + "\n"
    )
    (root / "router_questions.json").write_text(
        json.dumps({"questions": ROUTER_QUESTIONS}, indent=2) + "\n"
    )
    (root / "vela3_encoder.py").write_text(
        'raise RuntimeError("vllm_srun must never import package code")\n',
        encoding="utf-8",
    )
    names = sorted(path.name for path in root.iterdir() if path.is_file())
    (root / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(root / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    return root


VARIANTS = ("encoder",)


def write_fixture(output: str | Path, variant: str | None, seed: int) -> Path:
    """The fixture command's writer: the vela3-fast-style encoder."""
    return write_encoder_package(output, seed=seed)
