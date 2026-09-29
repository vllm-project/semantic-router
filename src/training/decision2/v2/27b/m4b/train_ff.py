"""FSDP2 full-parameter trainer for Milestone 4b (Qwen3.8-27B on node B GPU0-2).

The model, fresh head, objective, optimizer, schedule and SELECT rule are those
of ``training.model.train`` / ``v2.dec.train_dec`` (see the M4b preregistration,
section "Trainer"); only the execution is sharded:

* ``fully_shard`` per text decoder layer, the head as its own unit, then the
  root, with FP32 sharded parameters, gradients and AdamW states. ``--precision
  autocast`` computes the backbone under BF16 autocast with its weight cache
  off, ``mp`` uses FSDP's BF16 parameter policy (head FP32); gradients are
  reduced in FP32 as plain sums.
* Every update holds ``--update-rows`` global rows in ``v2.dec.batching`` order.
  Its rows are split across ranks by padded tokens and packed into micro-batches
  within ``--max-batch-tokens`` / ``--max-batch-rows``; ranks with fewer
  micro-batches run zero-weight copies of the shortest TRAIN row so that every
  collective lines up. Each row's loss is divided by the update's global row
  count, so the summed gradient equals the single-process gradient of the mean.
* SELECT is sharded round-robin across ranks through ``training.model.train``'s
  ``evaluate`` and reassembled in file order on rank 0.
* Checkpoints are full FP32 ``DecisionModel`` layouts written by rank 0 from a
  gathered CPU state dict, only on a new BEST and at the final update.

Run from ``src/training/decision2``::

    python3 -m torch.distributed.run --nnodes 1 --master-addr 127.0.0.1 --master-port 29500 \\
        --nproc-per-node 3 \\
        -m v2.27b.m4b.train_ff --model-path ... --revision ... --train ... \\
        --select ... --output ... --arm A1 --seed 20260926
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import shutil
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import timedelta
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist

from training.model.data import check_partition_isolation, file_sha256, load_partition
from training.model.decision_model import PROMPT_VERSION, DecisionModel, collate, encode
from training.model.loss import LOSS_VERSION, per_example_loss
from training.model.source import source_fingerprint
from training.model.train import (
    atomic_json,
    atomic_jsonl,
    evaluate,
    fsync_tree,
    learning_factor,
    metric_summary,
    utc_now,
)
from v2.dec.batching import padded, row_windows, token_batches
from v2.dec.train_dec import attach_teacher_probs, load_teacher, selection_key

TRAINER_VERSION = "m4b-fsdp2-full-trainer/1"
PRECISIONS = ("autocast", "mp", "fp32")
SELECTION = "SELECT family-macro accuracy desc, normalized Brier asc, earliest step"
SELECT32_ROWS = 32
PROBE_LIMIT_GB = 245
SHARED_FILES = (
    "training/model/data.py",
    "training/model/decision_model.py",
    "training/model/loss.py",
    "training/model/source.py",
    "training/model/train.py",
    "training/model/infer.py",
    "v2/dec/batching.py",
    "v2/dec/train_dec.py",
    "v2/dec/runtime_check.py",
    "v2/27b/m4b/train_ff.py",
)


def code_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[3]
    return {name: file_sha256(root / name) for name in SHARED_FILES}


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


# Planning: pure functions of the row lengths, identical on every rank.


def update_plan(
    lengths: list[int], *, seed: int, max_tokens: int, max_rows: int, update_rows: int
) -> list[list[int]]:
    """Row indices of every update: the ``token_batches`` order cut into ``update_rows`` windows."""
    order = [
        index
        for batch in token_batches(
            lengths, seed=seed, epoch=0, max_tokens=max_tokens, max_rows=max_rows
        )
        for index in batch
    ]
    return [
        [index for batch in window for index in batch]
        for window in row_windows([[index] for index in order], update_rows)
    ]


def split_rows(rows: list[int], lengths: list[int], world: int) -> list[list[int]]:
    """Longest-first assignment to the rank with the fewest padded tokens (then rows, then index)."""
    order = sorted(range(len(rows)), key=lambda j: (-padded(lengths[rows[j]]), j))
    loads, counts = [0] * world, [0] * world
    parts: list[list[int]] = [[] for _ in range(world)]
    for j in order:
        rank = min(range(world), key=lambda r: (loads[r], counts[r], r))
        parts[rank].append(j)
        loads[rank] += padded(lengths[rows[j]])
        counts[rank] += 1
    return [[rows[j] for j in sorted(part)] for part in parts]


def pack_rows(
    rows: list[int], lengths: list[int], *, max_tokens: int, max_rows: int
) -> list[list[int]]:
    """``token_batches``' greedy packing of one rank's rows, shortest first."""
    batches: list[list[int]] = []
    current: list[int] = []
    width = 0
    for _, _, index in sorted(
        (lengths[index], j, index) for j, index in enumerate(rows)
    ):
        grown = max(width, padded(lengths[index]))
        if current and (
            grown * (len(current) + 1) > max_tokens or len(current) >= max_rows
        ):
            batches.append(current)
            current, grown = [], padded(lengths[index])
        current.append(index)
        width = grown
    if current:
        batches.append(current)
    return batches


def rank_schedule(
    rows: list[int],
    lengths: list[int],
    world: int,
    *,
    max_tokens: int,
    max_rows: int,
) -> list[list[list[int] | None]]:
    """Per-rank micro-batches of one update, padded with ``None`` (dummy work) to equal counts."""
    packed = [
        pack_rows(part, lengths, max_tokens=max_tokens, max_rows=max_rows)
        for part in split_rows(rows, lengths, world)
    ]
    steps = max(1, max(len(batches) for batches in packed))
    return [batches + [None] * (steps - len(batches)) for batches in packed]


def padded_tokens(batch: list[int], lengths: list[int]) -> int:
    return padded(max(lengths[index] for index in batch)) * len(batch)


def eval_steps(horizon: int) -> set[int]:
    every = math.ceil(horizon / 8)
    return set(range(every, horizon + 1, every)) | {horizon}


def shard_positions(count: int, world: int, rank: int) -> tuple[list[int], int]:
    """Round-robin positions of one rank and the common per-rank width (>= 1)."""
    return list(range(rank, count, world)), max(1, math.ceil(count / world))


def is_best(records: list[tuple[int, dict[str, Any]]], step: int) -> bool:
    best = max(
        records, key=lambda record: selection_key(record[1], record[0], "shared")
    )
    return best[0] == step


# Distributed execution.


@dataclass
class Group:
    rank: int = 0
    world: int = 1
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    sharded: bool = False

    @property
    def main(self) -> bool:
        return self.rank == 0

    def all_reduce(self, tensor: torch.Tensor, op: str = "sum") -> torch.Tensor:
        if self.sharded and self.world > 1:
            dist.all_reduce(
                tensor, op=dist.ReduceOp.MAX if op == "max" else dist.ReduceOp.SUM
            )
        return tensor

    def gather(self, value: Any) -> list[Any]:
        if not (self.sharded and self.world > 1):
            return [value]
        out: list[Any] = [None] * self.world
        dist.all_gather_object(out, value)
        return out

    def barrier(self) -> None:
        if self.sharded and self.world > 1:
            dist.barrier()

    def synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)


def init_group(device_kind: str, timeout_minutes: float) -> Group:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if device_kind == "cuda":
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
    else:
        device = torch.device("cpu")
    if "RANK" in os.environ and not dist.is_initialized():
        dist.init_process_group(
            "nccl" if device_kind == "cuda" else "gloo",
            timeout=timedelta(minutes=timeout_minutes),
        )
    return Group(rank, world, device, sharded=dist.is_initialized())


def build_model(
    model_path: Path, revision: str, head_dim: int, seed: int
) -> tuple[DecisionModel, Any]:
    """Seed, then construct, in ``training.model.train``'s order for a fresh posttrained start."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    return DecisionModel.from_base(
        model_path,
        revision,
        head_dim,
        source_stage="posttrained",
        head_variant="shared",
    )


def state_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def prepare_model(
    model: DecisionModel, group: Group, precision: str, checkpointing: bool
) -> DecisionModel:
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}")
    model = model.float()
    model.backbone.requires_grad_(True)
    model.head.requires_grad_(True)
    if checkpointing:
        model.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    model.backbone.config.use_cache = False
    if not group.sharded:
        return model.to(group.device)
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard

    mesh = init_device_mesh(group.device.type, (group.world,))
    compute = (
        MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        if precision == "mp"
        else MixedPrecisionPolicy(reduce_dtype=torch.float32)
    )
    head = MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32)
    for layer in model.backbone.layers:
        fully_shard(layer, mesh=mesh, mp_policy=compute)
    fully_shard(model.head, mesh=mesh, mp_policy=head)
    fully_shard(model, mesh=mesh, mp_policy=compute)
    for module in model.modules():
        if isinstance(module, FSDPModule):
            module.set_gradient_divide_factor(1.0)
            module.set_force_sum_reduction_for_comms(True)
    return model


def make_optimizer(
    model: DecisionModel, backbone_lr: float, head_lr: float, weight_decay: float
) -> torch.optim.AdamW:
    """``training.model.train``'s full-mode groups; build after sharding (DTensor params)."""
    return torch.optim.AdamW(
        [
            {
                "params": list(model.backbone.parameters()),
                "lr": backbone_lr,
                "peak_lr": backbone_lr,
                "name": "backbone",
            },
            {
                "params": list(model.head.parameters()),
                "lr": head_lr,
                "peak_lr": head_lr,
                "name": "head",
            },
        ],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=weight_decay,
        foreach=True,
    )


def compute_context(precision: str, device: torch.device):
    if precision == "autocast":
        return torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, cache_enabled=False
        )
    return contextlib.nullcontext()


def to_device(items: list[dict[str, Any]], pad_id: int, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in collate(items, pad_id).items()
    }


def batch_terms(
    model: torch.nn.Module,
    items: list[dict[str, Any]],
    *,
    pad_id: int,
    device: torch.device,
    precision: str,
    brier_weight: float,
    kl_weight: float,
) -> tuple[torch.Tensor, dict[str, Any], dict[str, torch.Tensor]]:
    batch = to_device(items, pad_id, device)
    with compute_context(precision, device):
        logits = model(**batch)
        terms = per_example_loss(
            logits,
            batch["labels"],
            batch["candidate_mask"],
            objective="ce_brier",
            brier_weight=brier_weight,
            teacher_probs=batch["teacher_probs"] if kl_weight else None,
            replay_mask=batch["replay_mask"] if kl_weight else None,
            replay_kl_weight=kl_weight,
        )
    return logits, batch, terms


def scalar(value: torch.Tensor) -> float:
    full = getattr(value, "full_tensor", None)
    return float((full() if full is not None else value).item())


def accumulate_update(
    model: torch.nn.Module,
    group: Group,
    *,
    micro_batches: list[list[int] | None],
    rows: int,
    items: list[dict[str, Any]],
    dummy: dict[str, Any],
    pad_id: int,
    precision: str,
    brier_weight: float,
    kl_weight: float,
) -> list[float]:
    """Fresh gradients of one update and its global loss sums (total, ce, brier, kl, correct).

    This rank's ``micro_batches`` hold row indices or None (dummy); when sharded,
    the parameters' ``grad`` are FSDP's summed shards.
    """
    model.zero_grad(set_to_none=True)
    sums = torch.zeros(5, dtype=torch.float64, device=group.device)
    for batch_ids in micro_batches:
        subset = [items[index] for index in batch_ids] if batch_ids else [dummy]
        logits, batch, terms = batch_terms(
            model,
            subset,
            pad_id=pad_id,
            device=group.device,
            precision=precision,
            brier_weight=brier_weight,
            kl_weight=kl_weight,
        )
        (terms["total"].sum() * (1.0 / rows if batch_ids else 0.0)).backward()
        if batch_ids:
            sums += torch.stack(
                [
                    *(
                        terms[name].detach().double().sum()
                        for name in ("total", "ce", "brier", "replay_kl")
                    ),
                    (logits.detach().argmax(-1) == batch["labels"]).sum().double(),
                ]
            )
    group.all_reduce(sums)
    return sums.tolist()


def step_update(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    totals: list[float],
    rows: int,
    step: int,
    horizon: int,
    warmup_ratio: float,
    clip: float,
) -> dict[str, Any]:
    """Clip, check and apply the accumulated gradients of ``accumulate_update``."""
    factor = learning_factor(step, horizon, warmup_ratio)
    for param_group in optimizer.param_groups:
        param_group["lr"] = param_group["peak_lr"] * factor
    norm = scalar(torch.nn.utils.clip_grad_norm_(model.parameters(), clip))
    if not all(math.isfinite(value) for value in totals) or not math.isfinite(norm):
        raise RuntimeError(f"Nonfinite loss or gradient norm at update {step + 1}")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    lrs = {g["name"]: g["lr"] for g in optimizer.param_groups}
    return {
        "loss": totals[0] / rows,
        "ce": totals[1] / rows,
        "brier": totals[2] / rows,
        "kl": totals[3] / rows,
        "accuracy": totals[4] / rows,
        "grad_norm_preclip": norm,
        "lr_backbone": lrs["backbone"],
        "lr_head": lrs["head"],
    }


def run_update(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    group: Group,
    *,
    micro_batches: list[list[int] | None],
    rows: int,
    items: list[dict[str, Any]],
    dummy: dict[str, Any],
    pad_id: int,
    precision: str,
    brier_weight: float,
    kl_weight: float,
    step: int,
    horizon: int,
    warmup_ratio: float,
    clip: float,
) -> dict[str, Any]:
    """One optimizer update; this rank's ``micro_batches`` hold row indices or None (dummy)."""
    totals = accumulate_update(
        model,
        group,
        micro_batches=micro_batches,
        rows=rows,
        items=items,
        dummy=dummy,
        pad_id=pad_id,
        precision=precision,
        brier_weight=brier_weight,
        kl_weight=kl_weight,
    )
    return step_update(
        model,
        optimizer,
        totals=totals,
        rows=rows,
        step=step,
        horizon=horizon,
        warmup_ratio=warmup_ratio,
        clip=clip,
    )


def peak_memory(group: Group) -> dict[str, float]:
    if group.device.type == "cuda":
        values = [
            torch.cuda.max_memory_allocated(group.device) / 1e9,
            torch.cuda.max_memory_reserved(group.device) / 1e9,
        ]
    else:
        values = [0.0, 0.0]
    peaks = group.all_reduce(
        torch.tensor(values, dtype=torch.float64, device=group.device), op="max"
    ).tolist()
    return {"peak_mem_gb": peaks[0], "peak_reserved_gb": peaks[1]}


def softmax(values: list[float]) -> list[float]:
    top = max(values)
    exponentials = [math.exp(value - top) for value in values]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def select_probabilities(
    model: torch.nn.Module,
    items: list[dict[str, Any]],
    *,
    pad_id: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    """One row per call, as ``training.model.infer`` scores a single-question prompt."""
    was_training = model.training
    model.eval()
    out = []
    with torch.inference_mode():
        for item in items:
            batch = to_device([item], pad_id, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(**batch)
            values = logits[0, : len(item["keys"])].float().cpu().tolist()
            out.append(
                {
                    "id": item["id"],
                    "keys": item["keys"],
                    "logits": values,
                    "probabilities": softmax(values),
                }
            )
    model.train(was_training)
    return out


def sharded_rows(
    group: Group, items: list[dict[str, Any]]
) -> tuple[list[int], list[dict[str, Any]]]:
    own, width = shard_positions(len(items), group.world, group.rank)
    filler = items[own[0] if own else 0]
    return own, [items[i] for i in own] + [filler] * (width - len(own))


def sharded_probabilities(
    model: torch.nn.Module, items: list[dict[str, Any]], group: Group, pad_id: int
) -> list[dict[str, Any]]:
    own, shard = sharded_rows(group, items)
    records = select_probabilities(model, shard, pad_id=pad_id, device=group.device)
    gathered = group.gather(list(zip(own, records[: len(own)])))
    pairs = sorted(
        (pair for part in gathered for pair in part), key=lambda pair: pair[0]
    )
    return [record for _, record in pairs]


def sharded_evaluate(
    model: torch.nn.Module,
    items: list[dict[str, Any]],
    group: Group,
    *,
    pad_id: int,
    batch_size: int,
    output: Path,
    tag: str,
) -> dict[str, Any]:
    """``evaluate`` on this rank's rows; rank 0 writes the reassembled records and summary."""
    started = time.perf_counter()
    own, shard = sharded_rows(group, items)
    scratch = output / ".select-shards" / f"rank{group.rank}"
    scratch.mkdir(parents=True, exist_ok=True)
    shard_summary = evaluate(
        model,
        shard,
        pad_id=pad_id,
        batch_size=batch_size,
        device=group.device,
        output=scratch,
        tag=tag,
    )
    path = scratch / f"{tag}-predictions.jsonl"
    with path.open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream][: len(own)]
    path.unlink()
    (scratch / f"{tag}-metrics.json").unlink()
    scratch.rmdir()
    gathered = group.gather(list(zip(own, records)))
    if group.main:
        scratch.parent.rmdir()
    pairs = sorted(
        (pair for part in gathered for pair in part), key=lambda pair: pair[0]
    )
    ordered = [record for _, record in pairs]
    if [record["id"] for record in ordered] != [item["id"] for item in items]:
        raise RuntimeError("Sharded SELECT lost or duplicated rows")
    summary = metric_summary(ordered)
    summary.update(
        {
            "tag": tag,
            "seconds": time.perf_counter() - started,
            "precision": shard_summary["precision"],
            "ranks": group.world,
        }
    )
    if group.main:
        atomic_jsonl(output / f"{tag}-predictions.jsonl", ordered)
        atomic_json(output / f"{tag}-metrics.json", summary)
    return summary


class _StateBackbone:
    """``save_pretrained`` of the live backbone class and config with a gathered state dict."""

    def __init__(self, module: torch.nn.Module, state: dict[str, torch.Tensor]):
        self.module, self.state = module, state

    def save_pretrained(self, path: Path, **kwargs: Any) -> None:
        self.module.save_pretrained(path, state_dict=self.state, **kwargs)


class _StateHead:
    def __init__(self, state: dict[str, torch.Tensor]):
        self.state = state

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.state


def gather_state(
    model: torch.nn.Module, group: Group
) -> dict[str, torch.Tensor] | None:
    """Full CPU state dict on rank 0 (None elsewhere); a collective when sharded."""
    if not group.sharded:
        return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
    )

    state = get_model_state_dict(
        model, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
    )
    return state if group.main else None


def save_full(
    model: DecisionModel, tokenizer: Any, state: dict[str, torch.Tensor], path: Path
) -> None:
    """Write ``state`` through ``DecisionModel.save`` (backbone shards, head file, config)."""
    parts: dict[str, dict[str, torch.Tensor]] = {"backbone": {}, "head": {}}
    for name, tensor in state.items():
        prefix, _, rest = name.partition(".")
        if prefix not in parts:
            raise ValueError(f"Unexpected model state key {name}")
        parts[prefix][rest] = tensor
    proxy = SimpleNamespace(
        metadata=model.metadata,
        backbone=_StateBackbone(model.backbone, parts["backbone"]),
        head=_StateHead(parts["head"]),
    )
    DecisionModel.save(proxy, path, tokenizer)


def save_checkpoint(
    model: DecisionModel,
    tokenizer: Any,
    group: Group,
    *,
    output: Path,
    step: int,
    metrics: dict[str, Any] | None,
) -> str:
    name = f"checkpoint-{step:07d}"
    group.barrier()
    state = gather_state(model, group)
    if group.main:
        destination = output / name
        pending = output / f"{name}.pending"
        if destination.exists():
            raise ValueError(f"Refusing to overwrite {name}")
        if pending.exists():
            shutil.rmtree(pending)
        save_full(model, tokenizer, state, pending)
        atomic_json(
            pending / "checkpoint.json",
            {
                "step": step,
                "dev_metrics": metrics,
                "complete": True,
                "saved_utc": utc_now(),
            },
        )
        fsync_tree(pending)
        os.replace(pending, destination)
        descriptor = os.open(output, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        atomic_json(output / "LATEST.json", {"checkpoint": name, "step": step})
    del state
    group.barrier()
    return name


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--revision", required=True, help="Immutable base revision")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--select", type=Path, required=True)
    parser.add_argument("--cal", type=Path, help="Hashed for isolation only")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", required=True, help="Registered arm name")
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--teacher", type=Path)
    parser.add_argument("--teacher-partial", action="store_true")
    parser.add_argument("--teacher-kl-weight", type=float, default=0.0)
    parser.add_argument("--precision", choices=PRECISIONS, default="autocast")
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--brier-weight", type=float, default=0.5)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--max-batch-tokens", type=int, default=32768)
    parser.add_argument("--max-batch-rows", type=int, default=64)
    parser.add_argument("--update-rows", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--eval-batch", type=int, default=1)
    parser.add_argument("--gradient-checkpointing", choices=("on", "off"), default="on")
    parser.add_argument(
        "--max-updates",
        type=int,
        help="Probe: this many updates, no SELECT or checkpoint",
    )
    parser.add_argument(
        "--onestep",
        action="store_true",
        help="One update, SELECT, checkpoint and onestep.select32.jsonl",
    )
    parser.add_argument("--dist-timeout-minutes", type=float, default=60)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args(argv)
    if bool(args.teacher) != (args.teacher_kl_weight > 0):
        parser.error("--teacher and a positive --teacher-kl-weight go together")
    if not math.isfinite(args.teacher_kl_weight) or args.teacher_kl_weight < 0:
        parser.error("teacher KL weight must be finite and nonnegative")
    if args.teacher_partial and not args.teacher:
        parser.error("--teacher-partial needs --teacher")
    if args.onestep and args.max_updates is not None:
        parser.error("--onestep and --max-updates are separate preflights")
    if args.max_updates is not None and args.max_updates < 1:
        parser.error("--max-updates must be positive")
    for name in ("backbone_lr", "head_lr", "clip"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"{name} must be finite and positive")
    if not 0 <= args.warmup_ratio < 1 or not 0 <= args.weight_decay < 1:
        parser.error("invalid warmup ratio or weight decay")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    group = init_group(args.device, args.dist_timeout_minutes)
    runtime = None
    if group.device.type == "cuda":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("A ROCm/CUDA BF16 device is required")
        from v2.dec.runtime_check import require_runtime

        runtime = require_runtime()
    output = args.output
    mode = "onestep" if args.onestep else "probe" if args.max_updates else "full"
    if group.main:
        if output.exists() and any(output.iterdir()):
            raise ValueError("Fresh run requires a new or empty output directory")
        output.mkdir(parents=True, exist_ok=True)
    group.barrier()
    load_started = time.perf_counter()
    train_rows = load_partition(args.train, "train")
    select_rows = load_partition(args.select, "select")
    partitions = {"train": train_rows, "select": select_rows}
    if args.cal:
        partitions["cal"] = load_partition(args.cal, "cal")
    check_partition_isolation(partitions)
    teacher = None
    teacher_lines = 0
    if args.teacher:
        teacher = load_teacher(args.teacher, train_rows, partial=args.teacher_partial)
        with args.teacher.open(encoding="utf-8") as stream:
            teacher_lines = sum(1 for line in stream if line.strip())
        if len(teacher) != teacher_lines:
            raise ValueError(
                f"{teacher_lines - len(teacher)} teacher records do not join TRAIN by id and input hash"
            )

    hasher = ThreadPoolExecutor(max_workers=1) if group.main else None
    fingerprint = (
        hasher.submit(source_fingerprint, args.model_path)
        if hasher is not None
        else None
    )
    model, tokenizer = build_model(
        args.model_path, args.revision, args.head_dim, args.seed
    )
    head_init_sha256 = state_sha256(model.head)
    if len(set(group.gather(head_init_sha256))) != 1:
        raise RuntimeError("Ranks built different initial heads")
    parameters = {
        "backbone": sum(p.numel() for p in model.backbone.parameters()),
        "head": sum(p.numel() for p in model.head.parameters()),
    }
    model = prepare_model(
        model, group, args.precision, args.gradient_checkpointing == "on"
    )
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    if pad_id is None:
        raise ValueError("Tokenizer needs a pad or EOS token")
    train_items = [encode(row, tokenizer, args.max_length) for row in train_rows]
    if teacher is not None:
        for row, item in zip(train_rows, train_items):
            if row["id"] in teacher:
                attach_teacher_probs(item, teacher[row["id"]])
    select_items = [encode(row, tokenizer, args.max_length) for row in select_rows]
    lengths = [len(item["ids"]) for item in train_items]
    plan = update_plan(
        lengths,
        seed=args.seed,
        max_tokens=args.max_batch_tokens,
        max_rows=args.max_batch_rows,
        update_rows=args.update_rows,
    )
    horizon = len(plan)
    stop = 1 if args.onestep else min(args.max_updates or horizon, horizon)
    select_at = (
        {1} if args.onestep else set() if mode == "probe" else eval_steps(horizon)
    )
    dummy = train_items[min(range(len(lengths)), key=lambda i: (lengths[i], i))]
    optimizer = make_optimizer(model, args.backbone_lr, args.head_lr, args.weight_decay)
    load_seconds = time.perf_counter() - load_started
    source = fingerprint.result() if fingerprint is not None else None
    if hasher is not None:
        hasher.shutdown()
    model.metadata.update(
        {
            "checkpoint_format": "full",
            "full_training_source": {
                "kind": "posttrained",
                "revision": args.revision,
                "source_fingerprint": source,
            },
            "training_mode": "full",
            "loss_version": LOSS_VERSION,
            "fsdp_training": {
                "trainer_version": TRAINER_VERSION,
                "arm": args.arm,
                "seed": args.seed,
                "world_size": group.world,
                "precision": args.precision,
            },
        }
    )
    contract = {
        "trainer_version": TRAINER_VERSION,
        "arm": args.arm,
        "mode": mode,
        "prompt_version": PROMPT_VERSION,
        "loss_version": LOSS_VERSION,
        "objective": "ce_brier",
        "brier_weight": args.brier_weight,
        "teacher_kl_weight": args.teacher_kl_weight,
        "teacher_partial": args.teacher_partial,
        "teacher_rows": len(teacher) if teacher is not None else 0,
        "init": {
            "kind": "posttrained",
            "revision": args.revision,
            "head_variant": "shared",
            "head_dim": args.head_dim,
            "head_init_sha256": head_init_sha256,
        },
        "optimizer": {
            "name": "AdamW",
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": args.weight_decay,
            "backbone_lr": args.backbone_lr,
            "head_lr": args.head_lr,
            "warmup_ratio": args.warmup_ratio,
            "schedule": "training.model.train.learning_factor (linear warmup, cosine to 0.1x)",
            "clip": args.clip,
        },
        "batching": {
            "order": "v2.dec.batching.token_batches epoch 0, cut into update_rows windows",
            "max_batch_tokens": args.max_batch_tokens,
            "max_batch_rows": args.max_batch_rows,
            "update_rows": args.update_rows,
            "rank_split": "longest-first by padded tokens; zero-weight dummy micro-batches",
            "loss_normalization": "sum over the update's rows / global row count",
        },
        "max_length": args.max_length,
        "eval_batch": args.eval_batch,
        "gradient_checkpointing": args.gradient_checkpointing,
        "precision": args.precision,
        "world_size": group.world,
        "seed": args.seed,
        "horizon_updates": horizon,
        "stop_update": stop,
        "select_steps": sorted(select_at),
        "selection": SELECTION,
        "train_count": len(train_items),
    }
    if group.main:
        data_sha = {
            "train": file_sha256(args.train),
            "select": file_sha256(args.select),
        }
        if args.cal:
            data_sha["cal"] = file_sha256(args.cal)
        if args.teacher:
            data_sha["teacher"] = file_sha256(args.teacher)
        atomic_json(
            output / "provenance.json",
            {
                "created_utc": utc_now(),
                "args": {
                    k: str(v) if isinstance(v, Path) else v
                    for k, v in vars(args).items()
                },
                "contract": contract,
                "data_sha256": data_sha,
                "code_sha256": code_hashes(),
                "model_source": source,
                "mirror": {
                    key: value
                    for key, value in sorted(os.environ.items())
                    if key.startswith("DEC_")
                },
                "world_size": group.world,
                "device": (
                    torch.cuda.get_device_name(group.device)
                    if group.device.type == "cuda"
                    else "cpu"
                ),
                "precision": {
                    "autocast": "FP32 sharded params/grads/AdamW; BF16 autocast (cache off) backbone; FP32 head/loss; FP32 sum reduction",
                    "mp": "FP32 sharded params/grads/AdamW; FSDP BF16 param policy backbone; FP32 head/loss; FP32 sum reduction",
                    "fp32": "FP32 everywhere (parity and CPU tests)",
                }[args.precision],
                "parameters": parameters,
                "train_examples": len(train_items),
                "train_type_counts": dict(
                    Counter(row["task_type"] for row in train_rows)
                ),
                "train_tokens": sum(lengths),
                "train_max_tokens": max(lengths),
                "teacher_file_records": teacher_lines,
                "select_examples": len(select_items),
                "versions": {
                    "torch": torch.__version__,
                    "hip": torch.version.hip,
                    "transformers": package_version("transformers"),
                },
                "runtime": runtime,
                "load_seconds": load_seconds,
            },
        )
    metrics_file = (
        (output / "train-metrics.jsonl").open("a", encoding="utf-8")
        if group.main
        else None
    )

    def log(event: dict[str, Any]) -> None:
        if metrics_file is None:
            return
        text = json.dumps(event, ensure_ascii=False, allow_nan=False)
        metrics_file.write(text + "\n")
        metrics_file.flush()
        print(text, flush=True)

    if group.device.type == "cuda":
        # Loading and sharding peak far above training and leave cached blocks behind.
        torch.cuda.empty_cache()
        log({"event": "setup", **peak_memory(group)})
        torch.cuda.reset_peak_memory_stats(group.device)

    records: list[tuple[int, dict[str, Any]]] = []
    saved: set[int] = set()
    timings: list[float] = []
    native_total = padded_total = 0
    started_utc = utc_now()
    started = time.perf_counter()
    model.train()
    for step in range(stop):
        rows = plan[step]
        schedule = rank_schedule(
            rows,
            lengths,
            group.world,
            max_tokens=args.max_batch_tokens,
            max_rows=args.max_batch_rows,
        )
        update_started = time.perf_counter()
        result = run_update(
            model,
            optimizer,
            group,
            micro_batches=schedule[group.rank],
            rows=len(rows),
            items=train_items,
            dummy=dummy,
            pad_id=pad_id,
            precision=args.precision,
            brier_weight=args.brier_weight,
            kl_weight=args.teacher_kl_weight,
            step=step,
            horizon=horizon,
            warmup_ratio=args.warmup_ratio,
            clip=args.clip,
        )
        group.synchronize()
        seconds = time.perf_counter() - update_started
        timings.append(seconds)
        native = [sum(lengths[i] for b in part if b for i in b) for part in schedule]
        pads = [sum(padded_tokens(b, lengths) for b in part if b) for part in schedule]
        native_total += sum(native)
        padded_total += sum(pads)
        log(
            {
                "event": "train",
                "index": step + 1,
                "rows": len(rows),
                "teacher_rows": sum(
                    train_items[i]["teacher_probs"] is not None for i in rows
                ),
                "native_tokens": native,
                "padded_tokens": pads,
                "micro_batches": [sum(1 for b in part if b) for part in schedule],
                "dummy_micro_batches": [
                    sum(1 for b in part if not b) for part in schedule
                ],
                **result,
                "seconds": seconds,
                **peak_memory(group),
            }
        )
        index = step + 1
        if index in select_at:
            metrics = sharded_evaluate(
                model,
                select_items,
                group,
                pad_id=pad_id,
                batch_size=args.eval_batch,
                output=output,
                tag=f"select-step-{index:07d}",
            )
            records.append((index, metrics))
            log(
                {
                    "event": "select",
                    "step": index,
                    "metrics": {k: v for k, v in metrics.items() if k != "by_family"},
                }
            )
            if args.onestep:
                probabilities = sharded_probabilities(
                    model, select_items[:SELECT32_ROWS], group, pad_id
                )
                if group.main:
                    atomic_jsonl(output / "onestep.select32.jsonl", probabilities)
            if is_best(records, index):
                name = save_checkpoint(
                    model, tokenizer, group, output=output, step=index, metrics=metrics
                )
                saved.add(index)
                if group.main:
                    atomic_json(
                        output / "BEST.json",
                        {
                            "checkpoint": name,
                            "step": index,
                            "selection": SELECTION,
                            "family_macro_accuracy": metrics["family_macro_accuracy"],
                            "family_macro_brier": metrics["family_macro_brier"],
                        },
                    )
                log(
                    {
                        "event": "checkpoint",
                        "step": index,
                        "checkpoint": name,
                        "best": name,
                    }
                )
            elif index == stop:
                name = save_checkpoint(
                    model, tokenizer, group, output=output, step=index, metrics=metrics
                )
                saved.add(index)
                log(
                    {
                        "event": "checkpoint",
                        "step": index,
                        "checkpoint": name,
                        "best": None,
                    }
                )
    wall = time.perf_counter() - started
    memory = peak_memory(group)
    per_rank = group.gather(
        {
            "rank": group.rank,
            "peak_allocated_gb": (
                torch.cuda.max_memory_allocated(group.device) / 1e9
                if group.device.type == "cuda"
                else 0.0
            ),
            "peak_reserved_gb": (
                torch.cuda.max_memory_reserved(group.device) / 1e9
                if group.device.type == "cuda"
                else 0.0
            ),
        }
    )
    if group.main:
        steady = timings[1:] or timings
        if mode == "probe":
            atomic_json(
                output / "probe.json",
                {
                    "updates": stop,
                    "horizon_updates": horizon,
                    "world_size": group.world,
                    "precision": args.precision,
                    "max_batch_tokens": args.max_batch_tokens,
                    "load_seconds": load_seconds,
                    "seconds_per_update": sum(timings) / len(timings),
                    "seconds_per_update_after_first": sum(steady) / len(steady),
                    "native_tokens_per_second": native_total / sum(timings),
                    "padded_tokens_per_second": padded_total / sum(timings),
                    "projected_full_hours": sum(steady) / len(steady) * horizon / 3600,
                    **memory,
                    "per_rank_memory": per_rank,
                    "limit_gb": PROBE_LIMIT_GB,
                    "within_limit": memory["peak_mem_gb"] <= PROBE_LIMIT_GB,
                },
            )
        if select_at and stop not in saved:
            raise RuntimeError("Final update has no durable checkpoint")
        atomic_json(
            output / "COMPLETE.json",
            {
                "status": "complete",
                "mode": mode,
                "step": stop,
                "horizon_updates": horizon,
                "best": (
                    json.loads((output / "BEST.json").read_text())["checkpoint"]
                    if select_at
                    else None
                ),
                "started_utc": started_utc,
                "completed_utc": utc_now(),
                "wall_seconds": wall,
                **memory,
                "calibration_status": "untouched",
            },
        )
        metrics_file.close()
    group.barrier()
    if group.sharded:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
