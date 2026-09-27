"""Private, TRAIN-only one-update Gemma 4 numerical and reload admission.

The single pinned training row is used to exercise gradients and serialization,
not to estimate model quality. SELECT, CAL and all benchmark data are excluded.
GPU execution requires a separately reviewed private preregistration lock.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import stat
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
from training.model.decision_model import CandidateHead
from training.model.gemma4 import (
    GEMMA_PROMPT_VERSION,
    assert_zero_lora_b,
    attach_text_lora,
    encode_gemma,
)

PROBE_VERSION = "gemma4-train-row-one-step-reload-v1"
LOCK_VERSION = "gemma4-train-row-one-step-prereg-v1"
TRAIN_SHA256 = "61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755"
MAX_LENGTH = 4096
GPU_ORDINAL = "3"
SEED = 20260927
LORA_LR = 1e-4
HEAD_LR = 1e-3
GRAD_CLIP = 1.0
MAX_GRAD_NORM = 1e6
MAX_RELOAD_LOGIT_DRIFT = 1e-3
MAX_WALL_SECONDS = 1200


def code_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    files = {
        "one_step_probe": Path(__file__),
        "source_probe": root / "scripts/preflight_gemma4_native.py",
        "gemma_adapter": root / "training/model/gemma4.py",
        "shared_prompt": root / "training/model/decision_model.py",
        "data_canonicalizer": root / "training/model/data.py",
    }
    return {name: sha256_file(path) for name, path in files.items()}


def pinned_train_row(
    path: Path, tokenizer: Any, row_spec: dict[str, Any]
) -> tuple[dict[str, Any], str]:
    """Verify the complete TRAIN file and return only one exact admitted row."""
    if sha256_file(path) != TRAIN_SHA256:
        raise ValueError("TRAIN file differs from the frozen rights-clean v2 input")
    fields = {
        "id",
        "group_id",
        "row_sha256",
        "prompt_sha256",
        "token_ids_sha256",
        "token_count",
        "task_type",
        "family",
        "language",
        "label_index",
        "option_count",
    }
    if set(row_spec) != fields or any(
        not isinstance(row_spec[key], str) or not row_spec[key]
        for key in (
            "id",
            "group_id",
            "row_sha256",
            "prompt_sha256",
            "token_ids_sha256",
            "task_type",
            "family",
            "language",
        )
    ):
        raise ValueError("Private TRAIN row specification is incomplete")
    chosen = []
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        if row.get("id") == row_spec["id"]:
            chosen.append((row, hashlib.sha256(line.encode()).hexdigest()))
    if len(chosen) != 1:
        raise ValueError("Pinned TRAIN row missing or duplicated")
    row, row_sha256 = chosen[0]
    if (
        row_sha256 != row_spec["row_sha256"]
        or row.get("split") != "train"
        or row.get("evaluation_role") != "train"
        or row.get("group_id") != row_spec["group_id"]
        or row.get("task_type") != row_spec["task_type"]
        or row.get("family") != row_spec["family"]
        or row.get("language") != row_spec["language"]
        or row.get("label") != row_spec["label_index"]
        or len(row.get("options", [])) != row_spec["option_count"]
    ):
        raise ValueError("Pinned TRAIN row metadata or bytes changed")
    encoded = encode_gemma(row, tokenizer, MAX_LENGTH)
    if (
        encoded["task_type"] != row_spec["task_type"]
        or encoded["label"] != row_spec["label_index"]
        or len(encoded["keys"]) != row_spec["option_count"]
        or len(encoded["ids"]) != row_spec["token_count"]
        or encoded["prompt_sha256"] != row_spec["prompt_sha256"]
        or encoded["token_ids_sha256"] != row_spec["token_ids_sha256"]
    ):
        raise ValueError("Pinned TRAIN row tokenization or prompt changed")
    return encoded, row_sha256


def read_private_lock(path: Path) -> dict[str, Any]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError("One-step lock must be a private absolute file")
    parent = path.parent.resolve(strict=True)
    if (
        stat.S_IMODE(parent.stat().st_mode) != 0o700
        or parent.stat().st_uid != os.getuid()
        or stat.S_IMODE(path.stat().st_mode) != 0o600
        or path.stat().st_uid != os.getuid()
    ):
        raise ValueError("One-step lock must be owner-held mode 0600")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("One-step lock must be a JSON object")
    return value


def verify_lock(
    lock: dict[str, Any],
    source: dict[str, Any],
    encoded: dict[str, Any],
    row_sha256: str,
    runner_sha256: str,
) -> None:
    fields = {
        "schema_version",
        "source_id",
        "source_revision",
        "config_sha256",
        "tokenizer_json_sha256",
        "shard_sha256",
        "code_sha256",
        "runner_sha256",
        "train_file_sha256",
        "train_row",
        "gpu_ordinal",
        "max_length",
        "seed",
        "rank",
        "alpha",
        "dropout",
        "head_dim",
        "lora_lr",
        "head_lr",
        "weight_decay",
        "grad_clip",
        "max_grad_norm",
        "max_reload_logit_drift",
        "optimizer_updates",
        "max_wall_seconds",
        "select_cal_formal_labels_read",
    }
    expected = {
        "schema_version": LOCK_VERSION,
        "source_id": SOURCE_ID,
        "source_revision": SOURCE_REVISION,
        "config_sha256": source.get("config_sha256"),
        "tokenizer_json_sha256": source.get("tokenizer_json_sha256"),
        "shard_sha256": source.get("weights", {}).get("shard_sha256"),
        "code_sha256": code_hashes(),
        "runner_sha256": runner_sha256,
        "train_file_sha256": TRAIN_SHA256,
        "train_row": {
            "id": encoded["id"],
            "group_id": lock["train_row"]["group_id"],
            "row_sha256": row_sha256,
            "prompt_sha256": encoded["prompt_sha256"],
            "token_ids_sha256": encoded["token_ids_sha256"],
            "token_count": len(encoded["ids"]),
            "task_type": encoded["task_type"],
            "family": lock["train_row"]["family"],
            "language": lock["train_row"]["language"],
            "label_index": encoded["label"],
            "option_count": len(encoded["keys"]),
        },
        "gpu_ordinal": GPU_ORDINAL,
        "max_length": MAX_LENGTH,
        "seed": SEED,
        "rank": 8,
        "alpha": 16,
        "dropout": 0.05,
        "head_dim": 256,
        "lora_lr": LORA_LR,
        "head_lr": HEAD_LR,
        "weight_decay": 0.0,
        "grad_clip": GRAD_CLIP,
        "max_grad_norm": MAX_GRAD_NORM,
        "max_reload_logit_drift": MAX_RELOAD_LOGIT_DRIFT,
        "optimizer_updates": 1,
        "max_wall_seconds": MAX_WALL_SECONDS,
        "select_cal_formal_labels_read": False,
    }
    if set(lock) != fields or lock != expected:
        raise ValueError("One-step run differs from its frozen private lock")


def meta(snapshot: Path, train_file: Path, lock_path: Path) -> dict[str, Any]:
    """No-GPU dataset/tokenizer/source admission and future lock inputs."""
    from transformers import AutoTokenizer

    source = inspect(snapshot, require_weights=False)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    lock = read_private_lock(lock_path)
    encoded, row_hash = pinned_train_row(train_file, tokenizer, lock["train_row"])
    return {
        "probe_version": PROBE_VERSION,
        "source": source,
        "train_file_sha256": TRAIN_SHA256,
        "train_row_id": lock["train_row"]["id"],
        "train_row_sha256": row_hash,
        "train_prompt_sha256": encoded["prompt_sha256"],
        "train_token_ids_sha256": encoded["token_ids_sha256"],
        "train_token_count": len(encoded["ids"]),
        "train_task_type": encoded["task_type"],
        "train_label_index": encoded["label"],
        "prompt_version": GEMMA_PROMPT_VERSION,
        "code_sha256": code_hashes(),
        "gpu_used": False,
        "optimizer_updates": 0,
        "select_cal_formal_labels_read": False,
        "model_quality_evaluated": False,
    }


def decision_logits(backbone: Any, head: Any, encoded: dict[str, Any]) -> Any:
    import torch

    device = next(head.parameters()).device
    ids = torch.tensor([encoded["ids"]], dtype=torch.long, device=device)
    mask = torch.ones_like(ids)
    hidden = backbone(
        input_ids=ids, attention_mask=mask, use_cache=False
    ).last_hidden_state
    choices = hidden[:, encoded["candidate_positions"], :]
    query = hidden[:, encoded["query_position"], :]
    logits = head(choices, query)
    if tuple(logits.shape) != (1, 3) or not torch.isfinite(logits).all():
        raise RuntimeError("Gemma one-step Decision logits are invalid")
    return logits


def one_step(
    snapshot: Path,
    train_file: Path,
    lock_path: Path,
    package: Path,
    *,
    device: str,
) -> dict[str, Any]:
    """One optimizer update, adapter/head save and independent source reload."""
    import torch
    import torch.nn.functional as F
    from peft import PeftModel
    from safetensors.torch import load_file, save_file
    from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

    lock = read_private_lock(lock_path)
    if not package.is_absolute() or package.exists() or package.is_symlink():
        raise ValueError("One-step package must be a new private absolute path")
    parent = package.parent.resolve(strict=True)
    if (
        stat.S_IMODE(parent.stat().st_mode) != 0o700
        or parent.stat().st_uid != os.getuid()
    ):
        raise ValueError("One-step package directory must be owner-held mode 0700")
    if (
        os.getenv("ROCR_VISIBLE_DEVICES") != GPU_ORDINAL
        or os.getenv("HIP_VISIBLE_DEVICES") is not None
        or device != "cuda:0"
        or not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
        or not torch.cuda.is_bf16_supported()
    ):
        raise ValueError("One-step run needs exactly the locked BF16 ROCm GPU")
    source = inspect(snapshot)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    encoded, row_hash = pinned_train_row(train_file, tokenizer, lock["train_row"])
    verify_lock(
        lock, source, encoded, row_hash, sha256_file(snapshot / "run-one-step.sh")
    )

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
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
        raise RuntimeError("Gemma one-step source load omitted or changed weights")
    loaded = verify_loaded_state(full, snapshot, source["weights"])
    full.requires_grad_(False)
    if any(p.requires_grad for p in full.model.vision_tower.parameters()) or any(
        p.requires_grad for p in full.model.embed_vision.parameters()
    ):
        raise RuntimeError("Vision became trainable before adapter attachment")
    base = full.model.language_model
    base.config.use_cache = False
    adapter, plan = attach_text_lora(base, rank=8, alpha=16, dropout=0.05)
    assert_zero_lora_b(adapter, 60)
    head = CandidateHead(2816, 256).to(device)
    if sum(p.numel() for p in head.parameters()) != 2_895_360:
        raise RuntimeError("Gemma Decision head parameter count changed")
    lora_named = {name: p for name, p in adapter.named_parameters() if p.requires_grad}
    head_named = dict(head.named_parameters())
    if (
        len(lora_named) != 120
        or sum(p.numel() for p in lora_named.values()) != 3_645_440
        or any(not p.requires_grad for p in head_named.values())
    ):
        raise RuntimeError("Gemma one-step trainables differ from the locked plan")
    before_b = {
        name: p.detach().float().cpu().clone()
        for name, p in lora_named.items()
        if ".lora_B." in name
    }
    before_head = {name: p.detach().cpu().clone() for name, p in head_named.items()}
    optimizer = torch.optim.AdamW(
        [
            {"params": list(lora_named.values()), "lr": LORA_LR},
            {"params": list(head_named.values()), "lr": HEAD_LR},
        ],
        weight_decay=0.0,
    )
    optimizer_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    if optimizer_ids != {id(p) for p in [*lora_named.values(), *head_named.values()]}:
        raise RuntimeError("Optimizer includes a frozen or missing parameter")
    adapter.train()
    head.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = decision_logits(adapter, head, encoded)
        target = torch.tensor([encoded["label"]], dtype=torch.long, device=device)
        loss = F.cross_entropy(logits, target)
    if not torch.isfinite(loss) or not 0 < loss.item() < 1e6:
        raise RuntimeError("One-step TRAIN loss is nonfinite or out of bounds")
    train_loss = float(loss.item())
    loss.backward()
    gradients = [p.grad for p in [*lora_named.values(), *head_named.values()]]
    if any(g is not None and not torch.isfinite(g).all() for g in gradients):
        raise RuntimeError("One-step TRAIN gradients are nonfinite")
    b_nonzero_grad = sum(
        p.grad is not None and torch.count_nonzero(p.grad).item() > 0
        for name, p in lora_named.items()
        if ".lora_B." in name
    )
    if b_nonzero_grad != 60 or not any(
        p.grad is not None and torch.count_nonzero(p.grad).item() > 0
        for p in head_named.values()
    ):
        raise RuntimeError("One-step TRAIN omitted an adapter or head gradient")
    norm = torch.nn.utils.clip_grad_norm_(
        [*lora_named.values(), *head_named.values()], GRAD_CLIP
    )
    if not torch.isfinite(norm) or not 0 < norm.item() < MAX_GRAD_NORM:
        raise RuntimeError("One-step TRAIN gradient norm failed its bound")
    grad_norm_value = float(norm.item())
    optimizer.step()
    torch.cuda.synchronize()
    changed_b = sum(
        not torch.equal(p.detach().float().cpu(), before_b[name])
        for name, p in lora_named.items()
        if name in before_b
    )
    changed_head = sum(
        not torch.equal(p.detach().cpu(), before_head[name])
        for name, p in head_named.items()
    )
    if changed_b != 60 or changed_head < 1:
        raise RuntimeError("One-step TRAIN did not update LoRA B and head")

    adapter.eval()
    head.eval()
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16
    ):
        saved_logits = decision_logits(adapter, head, encoded).float().cpu()
    adapter_state = {name: p.detach().cpu().clone() for name, p in lora_named.items()}
    head_state = {
        name: p.detach().cpu().clone() for name, p in head.state_dict().items()
    }
    package.mkdir(mode=0o700, parents=False, exist_ok=False)
    adapter.save_pretrained(package / "adapter", safe_serialization=True)
    save_file(
        {name: value.float().contiguous() for name, value in head_state.items()},
        str(package / "decision_head.safetensors"),
    )
    adapter_file = package / "adapter/adapter_model.safetensors"
    if not adapter_file.is_file():
        raise RuntimeError("Gemma one-step package is missing its adapter")
    file_hashes = {
        "adapter_safetensors_sha256": sha256_file(adapter_file),
        "adapter_config_sha256": sha256_file(package / "adapter/adapter_config.json"),
        "decision_head_sha256": sha256_file(package / "decision_head.safetensors"),
    }
    del optimizer, adapter, head, full, base, logits, loss, target
    del lora_named, head_named, gradients, before_b, before_head, norm
    gc.collect()
    torch.cuda.empty_cache()

    second_source = inspect(snapshot)
    if second_source != source:
        raise RuntimeError("Official source changed between one-step loads")
    second, second_info = Gemma4ForConditionalGeneration.from_pretrained(
        snapshot,
        dtype=torch.bfloat16,
        local_files_only=True,
        attn_implementation="sdpa",
        device_map={"": device},
        output_loading_info=True,
        low_cpu_mem_usage=True,
    )
    if any(
        second_info.get(key)
        for key in ("missing_keys", "mismatched_keys", "error_msgs")
    ):
        raise RuntimeError("Reloaded Gemma source omitted or changed weights")
    reloaded = verify_loaded_state(second, snapshot, source["weights"])
    if reloaded != loaded:
        raise RuntimeError("Gemma source parameter identity changed on reload")
    second.requires_grad_(False)
    second.model.language_model.config.use_cache = False
    reloaded_adapter = PeftModel.from_pretrained(
        second.model.language_model, package / "adapter", is_trainable=False
    )
    named_reload = dict(reloaded_adapter.named_parameters())
    reloaded_lora = {
        name for name in named_reload if ".lora_A." in name or ".lora_B." in name
    }
    if reloaded_lora != set(adapter_state) or any(
        not torch.equal(value, named_reload[name].detach().cpu())
        for name, value in adapter_state.items()
    ):
        raise RuntimeError("Gemma LoRA package tensors changed across reload")
    reloaded_head = CandidateHead(2816, 256).to(device)
    reloaded_head.load_state_dict(
        load_file(str(package / "decision_head.safetensors")), strict=True
    )
    if any(
        not torch.equal(value.float(), reloaded_head.state_dict()[name].detach().cpu())
        for name, value in head_state.items()
    ):
        raise RuntimeError("Gemma Decision head tensors changed across reload")
    reloaded_adapter.eval()
    reloaded_head.eval()
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16
    ):
        replay_logits = decision_logits(reloaded_adapter, reloaded_head, encoded)
        replay_logits = replay_logits.float().cpu()
    drift = (saved_logits - replay_logits).abs().max().item()
    if not math.isfinite(drift) or drift > MAX_RELOAD_LOGIT_DRIFT:
        raise RuntimeError("Gemma one-step native logits changed across reload")
    torch.cuda.synchronize()
    return {
        "probe_version": PROBE_VERSION,
        "status": "one_step_train_only_reload_passed",
        "source_id": SOURCE_ID,
        "source_revision": SOURCE_REVISION,
        "source": source,
        "loaded": loaded,
        "prompt_version": GEMMA_PROMPT_VERSION,
        "train_file_sha256": TRAIN_SHA256,
        "train_row_id": lock["train_row"]["id"],
        "train_row_sha256": row_hash,
        "train_token_ids_sha256": encoded["token_ids_sha256"],
        "train_token_count": len(encoded["ids"]),
        "train_task_type": encoded["task_type"],
        "pre_update_train_loss": train_loss,
        "post_update_same_train_row_loss": float(
            F.cross_entropy(replay_logits, torch.tensor([encoded["label"]])).item()
        ),
        "unclipped_gradient_norm": grad_norm_value,
        "lora_b_nonzero_gradient_count": b_nonzero_grad,
        "lora_b_changed_count": changed_b,
        "head_changed_tensor_count": changed_head,
        "reload_max_logit_abs_drift": drift,
        "reload_limit": MAX_RELOAD_LOGIT_DRIFT,
        "adapter_plan": plan,
        "package_sha256": file_hashes,
        "code_sha256": code_hashes(),
        "lock_sha256": sha256_file(lock_path),
        "gpu_peak_bytes": torch.cuda.max_memory_allocated(),
        "optimizer_updates": 1,
        "select_cal_formal_labels_read": False,
        "model_quality_evaluated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("meta", "one-step"))
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--package", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.lock is None or (args.mode == "meta") != (args.package is None):
        parser.error("both modes require lock; only one-step requires package")
    os.umask(0o077)
    snapshot = args.snapshot.resolve(strict=True)
    train_file = args.train_file.resolve(strict=True)
    result = (
        meta(snapshot, train_file, args.lock)
        if args.mode == "meta"
        else one_step(
            snapshot,
            train_file,
            args.lock,
            args.package,
            device=args.device,
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
