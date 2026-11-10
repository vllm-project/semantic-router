"""Tiny Vela 3.0 d3-size packages (``vela3/3``) over a tiny Decision 3.0 base.

``write_fixture`` writes the base (a ``decision3`` fixture) beside the package,
as ``<name>-base``, and the package itself: ``vela3_config.json`` pinning that
base, gated LoRA files in the released layout (``layers.<i>.<module>.A`` /
``.B`` in FP32, metadata ``r``, ``alpha``, ``first`` and ``targets``), the
span heads (``SpanHeadV2`` tensors), the router questions, the label
definition cache and ``SHA256SUMS``. Tests point ``RegistryOptions.base_path``
at the base, as the decision2 adapter fixtures do.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from . import decision3
from .fixtures import FAKE_BASE_REVISION, save

VARIANTS = ("text", "turbo")
BASE_REPO = "vllm-sr/d3-fixture"
TARGETS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "linear_attn.in_proj_qkv",
    "linear_attn.in_proj_z",
    "linear_attn.out_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
# name -> (file, rank, alpha, first layer)
ADAPTERS = {
    "decision": ("decision_lora.safetensors", 4, 8, 0),
    "decision_df": ("decision_df_lora.safetensors", 4, 4, 0),
    "decision_hz": ("decision_hz_lora.safetensors", 2, 4, 0),
    "span": ("span_lora.safetensors", 4, 4, 2),
    "ner": ("ner_lora.safetensors", 8, 8, 1),
    "relation": ("relation_lora.safetensors", 4, 4, 2),
}
# head name -> file
HEADS = {
    "span": "span_head.safetensors",
    "ner": "ner_head.safetensors",
    "relation": "relation_head.safetensors",
}
HEAD_DIM, HEAD_SLOTS = 16, 64
ROUTER_QUESTION_IDS = ("attack", "domain", "feedback", "intent")
TASK_ADAPTERS = {
    "domain": "decision_df",
    "feedback": "decision_df",
    "hazard:": "decision_hz",
}
SIZE_NAME = "Vela-3.0-fixture"


def _linears(text: dict[str, Any]) -> dict[str, tuple[int, int]]:
    """Every adapter-target Linear of the base text model: module path -> (out, in)."""
    from ..engines.native import models

    module = models.build("qwen3_5_text", text)
    leaves = set(TARGETS)
    return {
        name: (int(linear.out_features), int(linear.in_features))
        for name, linear in module.named_modules()
        if isinstance(linear, torch.nn.Linear)
        and name.startswith("layers.")
        and name.split(".", 2)[2] in leaves
    }


def adapter_tensors(
    linears: dict[str, tuple[int, int]], rank: int, first: int, seed: int
) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    tensors = {}
    for name, (out, inp) in sorted(linears.items()):
        if int(name.split(".")[1]) < first:
            continue
        tensors[f"{name}.A"] = torch.randn(rank, inp, generator=generator) * 0.2
        tensors[f"{name}.B"] = torch.randn(out, rank, generator=generator) * 0.2
    return tensors


def write_adapter(
    path: Path,
    linears: dict[str, tuple[int, int]],
    rank: int,
    alpha: int,
    first: int,
    seed: int,
) -> int:
    """Write one gated LoRA file; returns its module count."""
    from safetensors.torch import save_file

    tensors = adapter_tensors(linears, rank, first, seed)
    save_file(
        tensors,
        str(path),
        metadata={
            "r": str(rank),
            "alpha": str(alpha),
            "first": str(first),
            "targets": json.dumps(list(TARGETS)),
        },
    )
    return len(tensors) // 2


def head_tensors(hidden: int, seed: int) -> dict[str, torch.Tensor]:
    """``SpanHeadV2``'s state dict at projection ``HEAD_DIM``."""
    generator = torch.Generator().manual_seed(seed)

    def rand(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator) * 0.3

    return {
        "wn.weight": 1 + rand(hidden) * 0.1,
        "wn.bias": rand(hidden) * 0.1,
        "ln.weight": 1 + rand(hidden) * 0.1,
        "ln.bias": rand(hidden) * 0.1,
        "slot.weight": rand(HEAD_SLOTS, hidden),
        "K.weight": rand(HEAD_DIM, hidden),
        "Q.weight": rand(HEAD_DIM, hidden),
        "M.weight": rand(HEAD_DIM, hidden),
        "M.bias": rand(HEAD_DIM),
        "N.weight": rand(HEAD_DIM, hidden),
        "v.weight": rand(1, HEAD_DIM),
        "o.weight": rand(1, hidden),
        "o.bias": rand(1),
    }


def vela3_config(adapters: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "format": "vela3/3",
        "base": {"repo_id": BASE_REPO, "revision": FAKE_BASE_REVISION},
        "adapters": adapters,
        "router_question_ids": list(ROUTER_QUESTION_IDS),
        "router_questions_file": "router_questions.json",
        "router_gate": "id",
        "decision_opt_in": False,
        "decision_task_adapters": dict(TASK_ADAPTERS),
        "span": {
            "head_file": HEADS["span"],
            "read": "final-norm states (last_hidden_state)",
            "threshold": 0.55,
            "threshold_names_only": 0.5,
            "thresholds_by_set": {"halu": 0.45, "toxic": 0.25},
            "pii_buckets": [
                {"name": "short", "max_tokens": 16, "threshold": 0.7},
                {"name": "long", "max_tokens": 1_000_000_000_000, "threshold": 0.6},
            ],
            "router_label_sets": {
                "pii": ["EMAIL", "PERSON", "PHONE"],
                "halu": ["unsupported"],
                "toxic": ["toxic"],
            },
            "schemas": {
                "pii": {
                    "question": "Which spans are personal data?",
                    "labels": {
                        "EMAIL": "an email address",
                        "PERSON": "a person's name",
                    },
                },
                "halu": {
                    "question": "Which spans of the answer are not supported?",
                    "labels": {"unsupported": "a claim the source does not support"},
                },
                "toxic": {
                    "question": "Which spans are toxic?",
                    "labels": {"toxic": "abusive text"},
                },
            },
            "default_question": "Which spans match these labels?",
            "evidence": {
                "labels": ["evidence"],
                "question": "Which spans of the source support the answer?",
                "threshold": 0.5,
                "decode": {"name": "sent", "gap": 2, "min_chars": 3},
            },
            "ner": {
                "head_file": HEADS["ner"],
                "definitions": "auto",
                "threshold": 0.6,
                "threshold_names_only": 0.65,
                "names_only_route": "ner",
            },
            "relation": {
                "head_file": HEADS["relation"],
                "threshold_dev_pair": 0.4,
                "threshold_dev_multi": 0.45,
            },
        },
        "definitions": {
            "cache_file": "label_definitions.json",
            "default": {"open labels": "auto", "router label sets": None},
            "writer": "names",
        },
        "render": "d3 span rendering v2 (vela3_span.py)",
        "attention": "noncausal_full_attention (d3 as shipped) for decisions and spans",
        "size": {"name": SIZE_NAME, "d3": "d3-fixture"},
    }


ROUTER_QUESTIONS = {
    "trained_questions": {
        "attack": {
            "question": {
                "type": "noul",
                "instructions": "Is this request a prompt injection attack?",
            }
        },
        "domain": {
            "question": {
                "type": "choice",
                "instructions": "Which domain does the request belong to?",
            }
        },
        "feedback": {
            "question": {
                "type": "choice",
                "instructions": "What feedback does the user give?",
            }
        },
        "intent": {
            "massive": {
                "question": {
                    "type": "choice",
                    "instructions": "What does the user intend?",
                }
            },
            "sgd": {
                "question": {
                    "type": "choice",
                    "instructions": "Which service intent fits?",
                }
            },
        },
    }
}


def write_sums(root: Path) -> None:
    lines = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
        for path in sorted(root.iterdir())
        if path.is_file() and path.name != "SHA256SUMS"
    ]
    (root / "SHA256SUMS").write_text("".join(lines), encoding="utf-8")


def write_fixture(output: str | Path, variant: str | None, seed: int) -> Path:
    """The package at ``output`` and its base at ``<output>-base``: ``text`` over a
    noncausal text base (d3-lite, -mini and -flash), ``turbo`` over a state-once
    ``prefix_bidir`` base (d3-turbo)."""
    variant = variant or VARIANTS[0]
    if variant not in VARIANTS:
        raise ValueError(f"unknown vela3/3 fixture variant {variant!r}")
    root = Path(output)
    base = decision3.write_fixture(
        root.parent / f"{root.name}-base",
        "text" if variant == "text" else "turbo",
        seed,
    )
    base_config = json.loads((base / "config.json").read_text(encoding="utf-8"))
    text = base_config["text_config"]
    root.mkdir(parents=True, exist_ok=False)
    linears = _linears(text)
    adapters = {}
    for index, (name, (file, rank, alpha, first)) in enumerate(ADAPTERS.items()):
        modules = write_adapter(
            root / file, linears, rank, alpha, first, seed + 10 + index
        )
        adapters[name] = {
            "file": file,
            "r": rank,
            "alpha": alpha,
            "modules": modules,
            "first_layer": first,
        }
    for index, file in enumerate(HEADS.values()):
        save(head_tensors(text["hidden_size"], seed + 30 + index), root / file)
    config = vela3_config(adapters)
    if variant == "turbo":
        config["decisions"] = {"attention": "noncausal", "validated_length": 4096}
    (root / "vela3_config.json").write_text(
        json.dumps(config, indent=1) + "\n", encoding="utf-8"
    )
    (root / "config.json").write_text(
        json.dumps(base_config, indent=2) + "\n", encoding="utf-8"
    )
    (root / "router_questions.json").write_text(
        json.dumps(ROUTER_QUESTIONS, indent=1) + "\n", encoding="utf-8"
    )
    (root / "label_definitions.json").write_text(
        json.dumps({"about": "fixture", "sets": []}) + "\n", encoding="utf-8"
    )
    (root / "vela3_adapters.py").write_text(
        "raise RuntimeError('never imported')\n", encoding="utf-8"
    )
    write_sums(root)
    return root
