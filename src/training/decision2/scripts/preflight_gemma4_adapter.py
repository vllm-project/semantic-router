"""Gold-free Gemma 4 text LoRA/source identity admission.

``meta`` validates exact target coverage without reading weight bodies or a
GPU. ``source-32`` and ``adapter-32`` are *prospective* one-GPU zero-step cells
requiring a separately approved preregistration. They compare selected hidden
vectors on 32 synthetic, unlabeled prompts; they never score a decision head,
read benchmark labels, or update a parameter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.preflight_gemma4_native import (
    SOURCE_ID,
    SOURCE_REVISION,
    inspect,
    private_output,
    sha256_file,
    verify_loaded_state,
)
from training.model.data import canonical
from training.model.decision_model import CandidateHead
from training.model.gemma4 import (
    GEMMA_PROMPT_VERSION,
    assert_zero_lora_b,
    attach_text_lora,
    encode_gemma,
    lora_plan,
)
from training.model.infer import question_to_row

PROBE_VERSION = "gemma4-text-lora-identity-32-v1"
LOCK_VERSION = "gemma4-text-lora-identity-prereg-v1"
LIMIT = 8192
MAX_HIDDEN_DRIFT = 1e-4
GPU_ORDINAL = "3"


def code_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    files = {
        "adapter_probe": Path(__file__),
        "source_probe": root / "scripts/preflight_gemma4_native.py",
        "gemma_adapter": root / "training/model/gemma4.py",
        "shared_prompt": root / "training/model/decision_model.py",
        "prompt_normalizer": root / "training/model/infer.py",
        "data_canonicalizer": root / "training/model/data.py",
    }
    return {name: sha256_file(path) for name, path in files.items()}


def read_private_lock(path: Path) -> dict[str, Any]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError("Gemma identity lock must be a private absolute file")
    parent = path.parent.resolve(strict=True)
    if (
        stat.S_IMODE(parent.stat().st_mode) != 0o700
        or parent.stat().st_uid != os.getuid()
        or stat.S_IMODE(path.stat().st_mode) != 0o600
        or path.stat().st_uid != os.getuid()
    ):
        raise ValueError("Gemma identity lock must be owner-held mode 0600")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Gemma identity lock must be a JSON object")
    return value


def verify_lock(
    lock: dict[str, Any],
    source: dict[str, Any],
    rows: list[dict[str, Any]],
    runner_sha256: str,
) -> None:
    plan = {
        "rank": 8,
        "alpha": 16,
        "dropout": 0.05,
        "adapter_parameters": 3_645_440,
        "decision_head_parameters": 2_895_360,
    }
    if set(lock) != {
        "schema_version",
        "source_id",
        "source_revision",
        "config_sha256",
        "tokenizer_json_sha256",
        "shard_sha256",
        "code_sha256",
        "runner_sha256",
        "roster_sha256",
        "cells",
        "gpu_ordinal",
        "max_length",
        "prompt_count",
        "optimizer_updates",
        "formal_labels_read",
        "adapter_plan",
        "max_wall_seconds",
    }:
        raise ValueError("Gemma identity lock fields changed")
    if (
        lock["schema_version"] != LOCK_VERSION
        or lock["source_id"] != SOURCE_ID
        or lock["source_revision"] != SOURCE_REVISION
        or lock["config_sha256"] != source.get("config_sha256")
        or lock["tokenizer_json_sha256"] != source.get("tokenizer_json_sha256")
        or lock["shard_sha256"] != source.get("weights", {}).get("shard_sha256")
        or lock["code_sha256"] != code_hashes()
        or lock["runner_sha256"] != runner_sha256
        or lock["roster_sha256"] != roster_sha256(rows)
        or lock["cells"] != ["source-32", "adapter-32"]
        or lock["gpu_ordinal"] != GPU_ORDINAL
        or lock["max_length"] != LIMIT
        or lock["prompt_count"] != 32
        or lock["optimizer_updates"] != 0
        or lock["formal_labels_read"] is not False
        or lock["adapter_plan"] != plan
        or lock["max_wall_seconds"] != 1200
    ):
        raise ValueError("Gemma identity run differs from its frozen lock")


def synthetic_prompts32() -> list[dict[str, Any]]:
    """Fixed schema and context probes, with no answer or benchmark lineage."""
    settings = (
        "A blue token grants room access after 09:00.",
        "The main rule applies unless the status field says suspended.",
        "The ledger changes from pending to approved after signature.",
        "Document A records a request; document B records the final decision.",
        "Only the later dated notice overrides the previous schedule.",
        "中文规则：蓝色凭证须在九点后使用。",
        "La regla local exige un permiso vigente después de las nueve.",
        "The available notes do not identify the responsible approver.",
    )
    result = []
    for index in range(32):
        kind = ("choice", "noul", "score")[index % 3]
        question: dict[str, Any] = {
            "type": kind,
            "instructions": f"Inspect record {index:02d} under the stated rule.",
        }
        if kind == "choice":
            question["criteria"] = {
                "proceed": "The request may proceed",
                "pause": "The request must pause",
                "unknown": "The record does not determine the outcome",
            }
        elif kind == "noul":
            question["criteria"] = {
                "false": "The condition is not established",
                "true": "The condition is established",
            }
        else:
            question["criteria"] = [
                "No support",
                "Partial support",
                "Strong support",
            ]
        result.append(
            {
                "id": f"schema-only-{index:02d}",
                "state": f"{settings[index % len(settings)]} Record marker {index:02d}.",
                "questions": {"q": question},
            }
        )
    return result


def encode_roster(tokenizer: Any) -> list[dict[str, Any]]:
    if getattr(tokenizer, "pad_token_id", None) is None:
        raise ValueError("Gemma tokenizer has no pad token")
    rows = []
    for item in synthetic_prompts32():
        row = question_to_row(item, "q", item["questions"]["q"])
        encoded = encode_gemma(row, tokenizer, LIMIT)
        if len(encoded["ids"]) > LIMIT:
            raise ValueError("Synthetic Gemma prompt exceeds the fixed input limit")
        rows.append(
            {
                "id": encoded["id"],
                "task_type": encoded["task_type"],
                "ids": encoded["ids"],
                "candidate_positions": encoded["candidate_positions"],
                "query_position": encoded["query_position"],
                "token_ids_sha256": encoded["token_ids_sha256"],
                "prompt_sha256": encoded["prompt_sha256"],
            }
        )
    if len(rows) != 32 or len({row["token_ids_sha256"] for row in rows}) != 32:
        raise ValueError("The fixed identity roster lacks 32 distinct prompts")
    if Counter(row["task_type"] for row in rows) != {
        "choice": 11,
        "noul": 11,
        "score": 10,
    }:
        raise ValueError("Identity roster type coverage changed")
    return rows


def roster_sha256(rows: list[dict[str, Any]]) -> str:
    identity = [
        {
            key: row[key]
            for key in ("id", "task_type", "token_ids_sha256", "prompt_sha256")
        }
        for row in rows
    ]
    return hashlib.sha256(canonical(identity).encode()).hexdigest()


def meta(snapshot: Path) -> dict[str, Any]:
    """Check official text topology, PEFT target cardinality and head count."""
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel

    source = inspect(snapshot, require_weights=False)
    config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    rows = encode_roster(tokenizer)
    with init_empty_weights():
        text = Gemma4TextModel(config.text_config)
        head = CandidateHead(config.text_config.hidden_size)
    plan = lora_plan(text)
    if sum(p.numel() for p in head.parameters()) != plan["decision_head_parameters"]:
        raise RuntimeError("Candidate head count differs from Gemma plan")
    _, actual = attach_text_lora(text)
    if actual != {**plan, "alpha": 16, "dropout": 0.05}:
        raise RuntimeError("Meta PEFT targets differ from Gemma plan")
    return {
        "probe_version": PROBE_VERSION,
        "source": source,
        "prompt_version": GEMMA_PROMPT_VERSION,
        "roster_sha256": roster_sha256(rows),
        "prompt_count": len(rows),
        "max_input_tokens": max(len(row["ids"]) for row in rows),
        "task_type_counts": dict(Counter(row["task_type"] for row in rows)),
        "lora_plan": actual,
        "code_sha256": code_hashes(),
        "gpu_used": False,
        "training_steps": 0,
        "formal_labels_read": False,
        "model_quality_evaluated": False,
    }


def zero_step(
    snapshot: Path, *, mode: str, device: str, lock_path: Path
) -> dict[str, Any]:
    """Collect only source or fresh-zero-LoRA selected hidden states."""
    import torch
    from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

    if mode not in ("source-32", "adapter-32"):
        raise ValueError("Unknown Gemma identity cell")
    lock = read_private_lock(lock_path)
    if (
        os.getenv("ROCR_VISIBLE_DEVICES") != GPU_ORDINAL
        or os.getenv("HIP_VISIBLE_DEVICES") is not None
    ):
        raise ValueError("Gemma GPU isolation differs from the frozen lock")
    if (
        device != "cuda:0"
        or not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
    ):
        raise ValueError("Gemma identity requires one isolated CUDA/ROCm GPU")
    source = inspect(snapshot)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    rows = encode_roster(tokenizer)
    verify_lock(lock, source, rows, sha256_file(snapshot / "run-identity32.sh"))
    full, info = Gemma4ForConditionalGeneration.from_pretrained(
        snapshot,
        dtype=torch.bfloat16,
        local_files_only=True,
        attn_implementation="sdpa",
        device_map={"": device},
        output_loading_info=True,
        low_cpu_mem_usage=True,
    )
    if any(info.get(key) for key in ("missing_keys", "mismatched_keys", "error_msgs")):
        raise RuntimeError("Pinned Gemma source loaded with missing or changed weights")
    loaded = verify_loaded_state(full, snapshot, source["weights"])
    full.requires_grad_(False)
    full.eval()
    text = full.model.language_model
    text.config.use_cache = False
    plan = lora_plan(text)
    if mode == "adapter-32":
        text, plan = attach_text_lora(text)
        assert_zero_lora_b(text, len(plan["target_modules"]))
        text.eval()
    if any(p.requires_grad for p in full.model.vision_tower.parameters()) or any(
        p.requires_grad for p in full.model.embed_vision.parameters()
    ):
        raise RuntimeError("Gemma vision parameters became trainable")
    readings = []
    with torch.inference_mode():
        for row in rows:
            ids = torch.tensor([row["ids"]], dtype=torch.long, device=device)
            mask = torch.ones_like(ids)
            hidden = text(
                input_ids=ids, attention_mask=mask, use_cache=False
            ).last_hidden_state
            positions = [*row["candidate_positions"], row["query_position"]]
            chosen = hidden[0, positions, :].float()
            if not torch.isfinite(chosen).all():
                raise RuntimeError(
                    "Gemma source identity produced nonfinite hidden states"
                )
            readings.append(
                {
                    "id": row["id"],
                    "task_type": row["task_type"],
                    "input_tokens": len(row["ids"]),
                    "token_ids_sha256": row["token_ids_sha256"],
                    "selected_hidden": chosen.cpu().tolist(),
                }
            )
    torch.cuda.synchronize()
    return {
        "probe_version": PROBE_VERSION,
        "mode": mode,
        "source": source,
        "loaded": loaded,
        "prompt_version": GEMMA_PROMPT_VERSION,
        "roster_sha256": roster_sha256(rows),
        "prompt_count": len(rows),
        "lora_plan": plan,
        "code_sha256": code_hashes(),
        "lock_sha256": sha256_file(lock_path),
        "readings": readings,
        "gpu_peak_bytes": torch.cuda.max_memory_allocated(),
        "training_steps": 0,
        "formal_labels_read": False,
        "model_quality_evaluated": False,
    }


def compare(source: dict[str, Any], adapter: dict[str, Any]) -> dict[str, Any]:
    """Require identical 32 inputs and selected hidden states within 1e-4."""
    common = (
        "probe_version",
        "source",
        "loaded",
        "prompt_version",
        "roster_sha256",
        "prompt_count",
        "code_sha256",
        "lock_sha256",
    )
    if (
        source.get("mode") != "source-32"
        or adapter.get("mode") != "adapter-32"
        or any(source.get(key) != adapter.get(key) for key in common)
        or source.get("probe_version") != PROBE_VERSION
        or source.get("prompt_count") != 32
        or source.get("source", {}).get("source_id") != SOURCE_ID
        or source.get("source", {}).get("source_revision") != SOURCE_REVISION
    ):
        raise ValueError("Gemma source and adapter identity receipts are incompatible")
    for value in (source, adapter):
        if (
            value.get("training_steps") != 0
            or value.get("formal_labels_read") is not False
            or value.get("model_quality_evaluated") is not False
        ):
            raise ValueError("Identity receipt includes training or quality evaluation")
    source_plan = source.get("lora_plan")
    adapter_plan = adapter.get("lora_plan")
    if (
        not isinstance(source_plan, dict)
        or not isinstance(adapter_plan, dict)
        or adapter_plan != {**source_plan, "alpha": 16, "dropout": 0.05}
        or source_plan.get("adapter_parameters") != 3_645_440
        or len(source_plan.get("target_modules", [])) != 60
    ):
        raise ValueError("Gemma fresh adapter differs from the locked target plan")
    left, right = source.get("readings"), adapter.get("readings")
    if (
        not isinstance(left, list)
        or not isinstance(right, list)
        or len(left) != 32
        or len(right) != 32
    ):
        raise ValueError("Gemma identity receipts need all 32 prompt readings")
    maximum = 0.0
    for a, b in zip(left, right):
        for key in ("id", "task_type", "input_tokens", "token_ids_sha256"):
            if a.get(key) != b.get(key):
                raise ValueError("Gemma source and adapter prompt identities changed")
        av, bv = a.get("selected_hidden"), b.get("selected_hidden")
        if (
            not isinstance(av, list)
            or not isinstance(bv, list)
            or len(av) != len(bv)
            or not 3 <= len(av) <= 256
        ):
            raise ValueError("Gemma selected hidden vector count changed")
        for x, y in zip(av, bv):
            if (
                not isinstance(x, list)
                or not isinstance(y, list)
                or len(x) != 2816
                or len(y) != len(x)
            ):
                raise ValueError("Gemma hidden vector width changed")
            if any(
                type(v) not in (int, float) or not math.isfinite(v) for v in [*x, *y]
            ):
                raise ValueError("Gemma identity receipt has nonfinite hidden states")
            maximum = max(maximum, *(abs(v1 - v2) for v1, v2 in zip(x, y)))
    if maximum > MAX_HIDDEN_DRIFT:
        raise ValueError("Gemma fresh LoRA changes source hidden states")
    return {
        "probe_version": PROBE_VERSION,
        "source_id": SOURCE_ID,
        "source_revision": SOURCE_REVISION,
        "roster_sha256": source["roster_sha256"],
        "prompt_count": 32,
        "max_hidden_abs_drift": maximum,
        "max_allowed_hidden_abs_drift": MAX_HIDDEN_DRIFT,
        "training_steps": 0,
        "formal_labels_read": False,
        "model_quality_evaluated": False,
        "status": "source_to_fresh_lora_identity_passed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("meta", "source-32", "adapter-32", "compare"))
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--first", type=Path)
    parser.add_argument("--second", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.mode == "compare":
        if (
            args.snapshot is not None
            or args.first is None
            or args.second is None
            or args.lock is not None
        ):
            parser.error("compare requires both receipts and no snapshot")
        result = compare(
            json.loads(args.first.read_text(encoding="utf-8")),
            json.loads(args.second.read_text(encoding="utf-8")),
        )
    else:
        if args.snapshot is None or args.first is not None or args.second is not None:
            parser.error("meta/source-32/adapter-32 require only a snapshot")
        if (args.mode == "meta") != (args.lock is None):
            parser.error("GPU identity cells require --lock; meta forbids it")
        snapshot = args.snapshot.resolve(strict=True)
        result = (
            meta(snapshot)
            if args.mode == "meta"
            else zero_step(
                snapshot, mode=args.mode, device=args.device, lock_path=args.lock
            )
        )
    private_output(args.output, result)
    print(
        json.dumps(
            {
                "mode": args.mode,
                "source_revision": SOURCE_REVISION,
                "receipt_sha256": sha256_file(args.output),
                "model_quality_evaluated": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
