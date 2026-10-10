"""FSDP2 full-weight trainer for Decision 2.5 Vega (code readout, one question per row).

Recipe defaults follow Perplexity's released trainer (perplexity-ai/pplx-decider-v1.1-27b,
Apache-2.0): AdamW lr 2e-6, weight decay 0.01 on every parameter, 15% linear warmup then cosine to
0.1x, gradient clipping 1.0, 256 rows per update, soft-target cross-entropy averaged over rows,
max length 8192, option-order shuffling of choice questions, seed 20260920, one epoch.

Execution: ``fully_shard`` per decoder layer, the readout (FP32 compute) and the root; FP32 sharded
master weights and AdamW state; BF16 parameters for compute; FP32 gradient reduce-scatter. Rows
are packed (no padding) into micro-batches of at most ``--token-budget`` tokens; every rank runs
the same number of micro-batches per update (token-balanced plan derived from the update index).

Run from ``src/training/decision2`` through ``d25.vega.train.launch`` (8 processes, SIGTERM-safe).
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import queue
import random
import shutil
import signal
import socket
import sys
import threading
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from d25.vega.common import decision_format as fmt
from d25.vega.train import data as D
from d25.vega.train import model as M

TRAINER_VERSION = "d25-vega-fsdp2-trainer/1"
STOP = {"flag": False, "signal": None}
EXIT_PAUSED = 10  # stopped after an intermediate export (DCP saved); resume continues
EXIT_SANITY = 3  # loss not decreasing by --sanity-step
EXIT_NONFINITE = 4  # NaN/inf loss or gradient
EXIT_PREEMPTED = 143  # SIGTERM: DCP saved at the current update
DCP_THREADS = {"n": 16}


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    io = p.add_argument_group("inputs and outputs")
    io.add_argument("--run", required=True, help="Run name (checkpoint dir name)")
    io.add_argument(
        "--train", required=True, help="Training JSONL(.gz) file or directory of them"
    )
    io.add_argument(
        "--dev", help="Held-out JSONL(.gz) file for periodic dev loss/accuracy"
    )
    io.add_argument(
        "--output", required=True, help="Run directory, e.g. /data/d25/vega/ckpt/<run>"
    )
    io.add_argument("--cache-dir", default="/data/d25/vega/train/cache")
    io.add_argument(
        "--init",
        required=True,
        help="Base, LoRA-merged, or exported checkpoint directory",
    )
    io.add_argument("--init-kind", choices=("base", "warm", "export"), default="base")
    io.add_argument("--tokenizer", help="Tokenizer directory (default: --init)")
    r = p.add_argument_group("recipe")
    r.add_argument("--attention-mode", choices=M.ATTENTION_MODES, default="causal")
    r.add_argument("--lr", type=float, default=2e-6)
    r.add_argument(
        "--readout-lr", type=float, help="Readout learning rate (default: --lr)"
    )
    r.add_argument("--weight-decay", type=float, default=0.01)
    r.add_argument("--warmup-ratio", type=float, default=0.15)
    r.add_argument(
        "--schedule", choices=("cosine", "linear", "constant"), default="cosine"
    )
    r.add_argument(
        "--min-lr-ratio",
        type=float,
        default=0.1,
        help="Decay floor as a fraction of the peak",
    )
    r.add_argument(
        "--epochs",
        type=float,
        default=1,
        help="Passes over the data; fractions train on the first part of the epoch-0 order with the full schedule",
    )
    r.add_argument("--seed", type=int, default=20260920)
    r.add_argument(
        "--rows-per-update",
        type=int,
        default=256,
        help="Effective batch (rows per optimizer update)",
    )
    r.add_argument(
        "--max-length",
        type=int,
        default=8192,
        help="Longer rows are dropped (never truncated)",
    )
    r.add_argument("--brier-weight", type=float, default=0.0)
    r.add_argument("--teacher", help="Name under meta.teachers to mix into the target")
    r.add_argument(
        "--teacher-weight",
        type=float,
        default=0.0,
        help="target = (1-w) gold + w teacher",
    )
    r.add_argument("--shuffle-options", choices=("on", "off"), default="on")
    r.add_argument("--clip", type=float, default=1.0)
    r.add_argument("--betas", default="0.9,0.999")
    r.add_argument("--eps", type=float, default=1e-8)
    s = p.add_argument_group("system")
    s.add_argument(
        "--token-budget", type=int, default=16384, help="Packed tokens per micro-batch"
    )
    s.add_argument("--max-rows-per-micro", type=int, default=128)
    s.add_argument(
        "--ac",
        default="full",
        help="Activation checkpointing: full | none | every:N | first:N",
    )
    s.add_argument(
        "--prefetch",
        type=int,
        default=1,
        help="FSDP forward/backward prefetch depth (0 = implicit)",
    )
    s.add_argument("--reshard-after-forward", choices=("on", "off"), default="on")
    s.add_argument(
        "--reduce-dtype",
        choices=("fp32", "bf16"),
        default="fp32",
        help="Gradient reduce-scatter dtype",
    )
    s.add_argument("--attn-backend", choices=("flash", "sdpa"), default="flash")
    s.add_argument("--conv-backend", choices=("fla", "torch"), default="fla")
    s.add_argument("--fused-adam", choices=("on", "off"), default="on")
    s.add_argument("--dist-timeout-min", type=float, default=120)
    c = p.add_argument_group("cadence")
    c.add_argument(
        "--save-every", type=int, default=1000, help="DCP checkpoint every N updates"
    )
    c.add_argument(
        "--dcp-threads",
        type=int,
        default=16,
        help="Writer/reader threads per rank for DCP files",
    )
    c.add_argument("--keep-dcp", type=int, default=2)
    c.add_argument("--final-dcp", choices=("on", "off"), default="on")
    c.add_argument(
        "--export-steps", default="", help="Comma-separated update indices to export"
    )
    c.add_argument(
        "--export-fractions",
        default="",
        help="Comma-separated fractions of the horizon to export",
    )
    c.add_argument("--export-every", type=int, default=0)
    c.add_argument(
        "--pause-after-export",
        action="store_true",
        help="After a non-final export, save a DCP checkpoint and exit 10 (the runner evaluates, then resumes)",
    )
    c.add_argument(
        "--sanity-step",
        type=int,
        default=200,
        help="Exit 3 if the mean loss of the 10 updates before this one is not below that of updates 6-15",
    )
    c.add_argument("--dev-every", type=int, default=100)
    c.add_argument("--dev-at-start", choices=("on", "off"), default="off")
    c.add_argument(
        "--dev-rows", type=int, default=0, help="Use the first N dev rows (0 = all)"
    )
    c.add_argument("--parity-rows", type=int, default=32)
    c.add_argument(
        "--max-steps",
        type=int,
        help="Stop after this update (probe); the schedule keeps the full horizon",
    )
    c.add_argument("--no-final-export", action="store_true")
    c.add_argument(
        "--profile-steps", default="", help="e.g. 3,4: torch.profiler trace on rank 0"
    )
    c.add_argument(
        "--stop-after-save",
        type=int,
        help="Test hook: exit 143 right after the DCP save at this update",
    )
    args = p.parse_args(argv)
    if args.readout_lr is None:
        args.readout_lr = args.lr
    if not 0 <= args.warmup_ratio < 1 or not 0 <= args.min_lr_ratio <= 1:
        p.error("invalid warmup ratio or min lr ratio")
    if not 0 <= args.teacher_weight <= 1 or (args.teacher_weight > 0) != bool(
        args.teacher
    ):
        p.error("--teacher and --teacher-weight in (0, 1] go together")
    if args.token_budget < args.max_length:
        p.error("--token-budget must be >= --max-length")
    return args


def lr_factor(step: int, total: int, args: argparse.Namespace) -> float:
    """Factor for update ``step`` (1-based); Perplexity's warmup + cosine-to-0.1x by default."""
    warmup = max(1, int(args.warmup_ratio * total)) if args.warmup_ratio > 0 else 0
    if warmup and step <= warmup:
        return step / warmup
    floor = args.min_lr_ratio
    progress = (step - warmup) / max(1, total - warmup)
    if args.schedule == "constant":
        return 1.0
    if args.schedule == "linear":
        return floor + (1 - floor) * (1 - progress)
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))


def code_sha256() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    files = sorted((root / "train").glob("*.py")) + [
        root / "common" / "decision_format.py"
    ]
    return {str(path.relative_to(root)): D.sha256_file(path) for path in files}


class Runtime:
    def __init__(self, timeout_min: float):
        self.rank = int(os.environ.get("RANK", "0"))
        self.world = int(os.environ.get("WORLD_SIZE", "1"))
        self.local = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(self.local)
        self.device = torch.device("cuda", self.local)
        dist.init_process_group(
            "nccl", timeout=timedelta(minutes=timeout_min), device_id=self.device
        )
        from torch.distributed.device_mesh import init_device_mesh

        self.mesh = init_device_mesh("cuda", (self.world,))

    @property
    def main(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        dist.barrier(device_ids=[self.local])

    def reduce(self, values: list[float], op=dist.ReduceOp.SUM) -> list[float]:
        tensor = torch.tensor(values, dtype=torch.float64, device=self.device)
        dist.all_reduce(tensor, op=op)
        return tensor.tolist()

    def broadcast_object(self, value: Any) -> Any:
        holder = [value]
        dist.broadcast_object_list(holder, src=0)
        return holder[0]


class Logger:
    def __init__(self, directory: Path, enabled: bool):
        self.enabled = enabled
        self.directory = directory
        if enabled:
            directory.mkdir(parents=True, exist_ok=True)

    def write(self, name: str, record: dict[str, Any], echo: bool = True) -> None:
        if not self.enabled:
            return
        record = {"time": utc(), **record}
        line = json.dumps(record, ensure_ascii=False, allow_nan=True)
        with (self.directory / f"{name}.jsonl").open("a") as handle:
            handle.write(line + "\n")
        if echo:
            print(line, flush=True)


# ---------------------------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------------------------


def configure_ac(text_model: torch.nn.Module, policy: str) -> list[int]:
    layers = list(text_model.layers)
    if policy == "none":
        chosen: list[int] = []
    elif policy == "full":
        chosen = list(range(len(layers)))
    elif policy.startswith("every:"):
        n = int(policy.split(":", 1)[1])
        chosen = [i for i in range(len(layers)) if i % n == 0]
    elif policy.startswith("first:"):
        n = int(policy.split(":", 1)[1])
        chosen = list(range(min(n, len(layers))))
    else:
        raise ValueError(f"unknown --ac policy {policy!r}")
    if chosen:
        text_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    for i, layer in enumerate(layers):
        layer.gradient_checkpointing = i in chosen
    return chosen


def shard_model(
    model: M.DecisionReadout, rt: Runtime, args: argparse.Namespace
) -> None:
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    reduce = torch.float32 if args.reduce_dtype == "fp32" else torch.bfloat16
    bf16 = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=reduce)
    fp32 = MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32)
    layers = list(model.text.layers)
    reshard = args.reshard_after_forward == "on"
    for i, layer in enumerate(layers):
        fully_shard(
            layer,
            mesh=rt.mesh,
            mp_policy=bf16,
            reshard_after_forward=reshard and i < len(layers) - 1,
        )
    fully_shard(model.readout, mesh=rt.mesh, mp_policy=fp32)
    fully_shard(model, mesh=rt.mesh, mp_policy=bf16)
    if args.prefetch > 0:
        for i, layer in enumerate(layers):
            layer.set_modules_to_forward_prefetch(layers[i + 1 : i + 1 + args.prefetch])
            layer.set_modules_to_backward_prefetch(
                list(reversed(layers[max(0, i - args.prefetch) : i]))
            )


def local_slice(param, full: torch.Tensor) -> torch.Tensor:
    from torch.distributed.tensor import DTensor

    if not isinstance(param, DTensor):
        return full
    local = param.to_local()
    rank = param.device_mesh.get_local_rank()
    world = param.device_mesh.size()
    chunks = torch.chunk(full, world, dim=0)
    if rank < len(chunks):
        part = chunks[rank]
    else:
        part = full[:0]
    if tuple(part.shape) != tuple(local.shape):
        raise RuntimeError(
            f"shard shape {tuple(local.shape)} != expected {tuple(part.shape)}"
        )
    return part


@torch.no_grad()
def load_initial_weights(
    model: M.DecisionReadout, rt: Runtime, state: dict[str, torch.Tensor] | None
) -> None:
    """Rank 0 holds the full CPU state (model FQNs); broadcast each tensor, keep the local shard."""
    names = [name for name, _ in model.named_parameters()]
    meta = None
    if rt.main:
        missing = [n for n in names if n not in state]
        extra = [n for n in state if n not in names]
        if missing or extra:
            raise RuntimeError(
                f"initial weights mismatch: missing {missing[:5]} extra {extra[:5]}"
            )
        meta = {
            n: (str(state[n].dtype).replace("torch.", ""), list(state[n].shape))
            for n in names
        }
    meta = rt.broadcast_object(meta)
    for name, param in model.named_parameters():
        dtype_name, shape = meta[name]
        if tuple(shape) != tuple(param.shape):
            raise RuntimeError(
                f"{name}: checkpoint shape {shape} != model {tuple(param.shape)}"
            )
        dtype = getattr(torch, dtype_name)
        if rt.main:
            full = (
                state[name]
                .to(device=rt.device, dtype=dtype, non_blocking=False)
                .contiguous()
            )
        else:
            full = torch.empty(shape, dtype=dtype, device=rt.device)
        dist.broadcast(full, src=0)
        target = param.to_local() if hasattr(param, "to_local") else param
        target.copy_(local_slice(param, full).to(target.dtype))
        del full


def gather_full_state(
    model: M.DecisionReadout, rt: Runtime
) -> dict[str, torch.Tensor] | None:
    """Full parameters on rank 0 (CPU; BF16 text, FP32 readout); a collective on every rank."""
    from torch.distributed.tensor import DTensor

    out: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        full = param.full_tensor() if isinstance(param, DTensor) else param.detach()
        if rt.main:
            dtype = torch.float32 if name.startswith("readout.") else torch.bfloat16
            out[name] = full.detach().to("cpu", dtype).contiguous()
        del full
    return out if rt.main else None


def build_optimizer(
    model: M.DecisionReadout, args: argparse.Namespace
) -> torch.optim.Optimizer:
    betas = tuple(float(x) for x in args.betas.split(","))
    groups = [
        {
            "params": [
                p for n, p in model.named_parameters() if not n.startswith("readout.")
            ],
            "lr": args.lr,
            "peak_lr": args.lr,
            "name": "backbone",
        },
        {
            "params": list(model.readout.parameters()),
            "lr": args.readout_lr,
            "peak_lr": args.readout_lr,
            "name": "readout",
        },
    ]
    kwargs = dict(betas=betas, eps=args.eps, weight_decay=args.weight_decay)
    if args.fused_adam == "on":
        try:
            return torch.optim.AdamW(groups, fused=True, **kwargs)
        except Exception as exc:  # noqa: BLE001
            print(f"fused AdamW unavailable ({exc}); using foreach", flush=True)
    return torch.optim.AdamW(groups, foreach=True, **kwargs)


# ---------------------------------------------------------------------------------------------
# Data plan
# ---------------------------------------------------------------------------------------------


class StepPlan:
    def __init__(
        self, step: int, epoch: int, micro: list[M.PackedBatch], stats: dict[str, Any]
    ):
        self.step, self.epoch, self.micro, self.stats = step, epoch, micro, stats


def make_plan(
    step: int,
    schedule: D.Schedule,
    store: D.RowStore,
    encoder: D.Encoder,
    rt: Runtime,
    args: argparse.Namespace,
) -> StepPlan:
    epoch, indices = schedule.rows(step)
    encoded: list[D.Encoded] = []
    dropped = {"too_long": 0, "invalid": 0}
    for index in indices.tolist():
        row = store.get(index)
        if row is None:
            dropped["invalid"] += 1
            continue
        reason = D.validate(row)
        if reason:
            dropped["invalid"] += 1
            continue
        item = encoder.encode(row, epoch=epoch, index=index)
        if len(item.ids) > args.max_length:
            dropped["too_long"] += 1
            continue
        encoded.append(item)
    if not encoded:
        raise RuntimeError(f"update {step + 1} has no trainable rows")
    lengths = [len(item.ids) for item in encoded]
    plan = D.plan_bins(lengths, rt.world, args.token_budget, args.max_rows_per_micro)
    dummy = min(range(len(encoded)), key=lambda i: (lengths[i], i))
    micro = []
    for positions in plan[rt.rank]:
        if positions:
            items = [encoded[i] for i in positions]
            weights = [item.weight for item in items]
        else:
            items = [encoded[dummy]]
            weights = [0.0]
        batch = M.PackedBatch(
            [it.ids for it in items],
            [it.count for it in items],
            [it.target for it in items],
            weights,
        )
        micro.append(batch.pin())
    loads = [sum(lengths[i] for mb in plan[r] for i in mb) for r in range(rt.world)]
    stats = {
        "rows": len(encoded),
        "tokens": sum(lengths),
        "weight": float(sum(item.weight for item in encoded)),
        "micro_batches": len(plan[0]),
        "empty_micro": sum(1 for r in range(rt.world) for mb in plan[r] if not mb),
        "rank_tokens_max": max(loads),
        "rank_tokens_min": min(loads),
        "max_row_tokens": max(lengths),
        **{f"dropped_{k}": v for k, v in dropped.items()},
    }
    return StepPlan(step, epoch, micro, stats)


class Prefetcher:
    """Builds the next updates' plans in a background thread (tokenizer releases the GIL)."""

    def __init__(self, build, start: int, stop: int, depth: int = 2):
        self.build = build
        self.queue: queue.Queue = queue.Queue(maxsize=depth)
        self.error: BaseException | None = None
        self.thread = threading.Thread(
            target=self._run, args=(start, stop), daemon=True
        )
        self.thread.start()

    def _run(self, start: int, stop: int) -> None:
        try:
            for step in range(start, stop):
                self.queue.put(self.build(step))
        except BaseException as exc:  # noqa: BLE001
            self.error = exc
            traceback.print_exc()
            self.queue.put(None)

    def get(self) -> StepPlan:
        item = self.queue.get()
        if item is None:
            raise RuntimeError("data prefetch failed") from self.error
        return item


# ---------------------------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------------------------


@torch.no_grad()
def evaluate(
    model, rt: Runtime, dev: list[D.Encoded], args: argparse.Namespace
) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    lengths = [len(item.ids) for item in dev]
    plan = D.plan_bins(lengths, rt.world, args.token_budget, args.max_rows_per_micro)
    results = []
    for positions in plan[rt.rank]:
        items = [dev[i] for i in positions] if positions else [dev[0]]
        inputs, target, _ = M.PackedBatch(
            [it.ids for it in items],
            [it.count for it in items],
            [it.target for it in items],
            [it.weight for it in items],
        ).to(rt.device)
        logits = model(inputs)
        terms = M.row_losses(logits, target)
        if positions:
            pred = logits.argmax(-1).tolist()
            for j, item in enumerate(items):
                results.append(
                    (
                        item.row_id,
                        item.family,
                        item.kind,
                        float(terms["ce"][j]),
                        float(terms["brier"][j]),
                        float(terms["correct"][j]),
                        pred[j],
                    )
                )
    gathered: list[Any] = [None] * rt.world
    dist.all_gather_object(gathered, results)
    model.train(was_training)
    rows = [r for part in gathered for r in part]
    out: dict[str, Any] = {"rows": len(rows)}
    if not rows:
        return out
    out["ce"] = float(np.mean([r[3] for r in rows]))
    out["brier"] = float(np.mean([r[4] for r in rows]))
    out["accuracy"] = float(np.mean([r[5] for r in rows]))
    for kind in sorted({r[2] for r in rows}):
        part = [r for r in rows if r[2] == kind]
        out[f"accuracy_{kind}"] = float(np.mean([r[5] for r in part]))
        out[f"ce_{kind}"] = float(np.mean([r[3] for r in part]))
    families: dict[str, list[float]] = {}
    for r in rows:
        families.setdefault(r[1], []).append(r[5])
    out["family_macro_accuracy"] = float(
        np.mean([np.mean(v) for v in families.values()])
    )
    out["families"] = len(families)
    return out


@torch.no_grad()
def parity_probabilities(
    model, rt: Runtime, dev: list[D.Encoded], n: int, budget: int
) -> list[dict[str, Any]]:
    """Probabilities of the in-memory model on the first ``n`` dev rows (every rank runs the same batch)."""
    items = dev[:n]
    if not items:
        return []
    was_training = model.training
    model.eval()
    out = []
    budget_rows: list[list[D.Encoded]] = [[]]
    used = 0
    for item in items:
        if used + len(item.ids) > budget and budget_rows[-1]:
            budget_rows.append([])
            used = 0
        budget_rows[-1].append(item)
        used += len(item.ids)
    for chunk in budget_rows:
        inputs, _, _ = M.PackedBatch(
            [it.ids for it in chunk],
            [it.count for it in chunk],
            [it.target for it in chunk],
            [1.0] * len(chunk),
        ).to(rt.device)
        probs = torch.softmax(model(inputs).float(), dim=-1).cpu()
        for j, item in enumerate(chunk):
            out.append({"id": item.row_id, "probs": probs[j, : item.count].tolist()})
    model.train(was_training)
    return out


# ---------------------------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------------------------


def dcp_dirs(root: Path) -> list[tuple[int, Path]]:
    out = []
    if root.exists():
        for path in root.iterdir():
            if (
                path.is_dir()
                and path.name.startswith("step-")
                and (path / "trainer_state.json").exists()
            ):
                out.append((int(path.name.split("-")[1]), path))
    return sorted(out)


def save_dcp(
    model,
    optimizer,
    rt: Runtime,
    root: Path,
    step: int,
    trainer_state: dict[str, Any],
    keep: int,
    log: Logger,
) -> float:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict

    began = time.time()
    final = root / f"step-{step:06d}"
    tmp = root / f"step-{step:06d}.tmp"
    if rt.main:
        root.mkdir(parents=True, exist_ok=True)
        if tmp.exists():
            shutil.rmtree(tmp)
        if final.exists():
            shutil.rmtree(final)
    rt.barrier()
    msd, osd = get_state_dict(model, optimizer)
    writer = dcp.FileSystemWriter(
        str(tmp), thread_count=DCP_THREADS["n"], sync_files=True
    )
    dcp.save({"model": msd, "optim": osd}, storage_writer=writer)
    torch.save(
        {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(),
            "python": random.getstate(),
            "numpy": np.random.get_state(),
        },
        tmp / f"rng-rank{rt.rank}.pt",
    )
    rt.barrier()
    if rt.main:
        (tmp / "trainer_state.json").write_text(json.dumps(trainer_state, indent=2))
        os.replace(tmp, final)
        for old_step, path in dcp_dirs(root)[:-keep] if keep > 0 else []:
            shutil.rmtree(path, ignore_errors=True)
            log.write("events", {"event": "dcp_pruned", "step": old_step})
    rt.barrier()
    seconds = time.time() - began
    log.write(
        "events",
        {"event": "dcp_saved", "step": step, "path": str(final), "seconds": seconds},
    )
    return seconds


def load_dcp(model, optimizer, rt: Runtime, path: Path) -> dict[str, Any]:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

    import inspect

    msd, osd = get_state_dict(model, optimizer)
    state = {"model": msd, "optim": osd}
    reader_kwargs = (
        {"thread_count": DCP_THREADS["n"]}
        if "thread_count" in inspect.signature(dcp.FileSystemReader.__init__).parameters
        else {}
    )
    dcp.load(state, storage_reader=dcp.FileSystemReader(str(path), **reader_kwargs))
    set_state_dict(
        model,
        optimizer,
        model_state_dict=state["model"],
        optim_state_dict=state["optim"],
    )
    rng = path / f"rng-rank{rt.rank}.pt"
    if rng.exists():
        saved = torch.load(rng, weights_only=False)
        torch.set_rng_state(saved["torch"])
        torch.cuda.set_rng_state(saved["cuda"])
        random.setstate(saved["python"])
        np.random.set_state(saved["numpy"])
    return json.loads((path / "trainer_state.json").read_text())


def export_dir(output: str | Path, step: int) -> Path:
    return Path(output) / f"step-{step:06d}"


def export(
    model,
    rt: Runtime,
    args,
    ctx: dict[str, Any],
    step: int,
    dev: list[D.Encoded],
    log: Logger,
) -> None:
    began = time.time()
    destination = export_dir(args.output, step)
    if rt.main and destination.exists():
        log.write("events", {"event": "export_skipped_exists", "step": step})
    exists = rt.broadcast_object(destination.exists() if rt.main else None)
    if exists:
        return
    parity = parity_probabilities(model, rt, dev, args.parity_rows, args.token_budget)
    state = gather_full_state(model, rt)
    if rt.main:
        text = {k[len("text.") :]: v for k, v in state.items() if k.startswith("text.")}
        readout = state["readout.weight"]
        provenance = {
            "run": args.run,
            "step": step,
            "total_steps": ctx["total_steps"],
            "rows_seen": ctx["rows_seen"],
            "trainer": TRAINER_VERSION,
            "init": {
                "kind": args.init_kind,
                "path": str(Path(args.init).resolve()),
                "readout": ctx["readout_source"],
            },
            "data_sha256": ctx["data_sha256"],
            "dev_sha256": ctx.get("dev_sha256"),
            "code_sha256": ctx["code_sha256"],
            "recipe": {
                k: getattr(args, k)
                for k in (
                    "lr",
                    "readout_lr",
                    "weight_decay",
                    "warmup_ratio",
                    "schedule",
                    "min_lr_ratio",
                    "epochs",
                    "seed",
                    "rows_per_update",
                    "max_length",
                    "brier_weight",
                    "teacher",
                    "teacher_weight",
                    "shuffle_options",
                    "clip",
                )
            },
            "exported_utc": utc(),
        }
        config = M.decision_config(
            codes=ctx["codes"],
            token_ids=ctx["token_ids"],
            attention_mode=args.attention_mode,
            max_length=args.max_length,
            provenance=provenance,
        )
        M.export_checkpoint(
            destination,
            config=ctx["config"],
            text_state=text,
            visual_state=ctx["visual"],
            readout=readout,
            tokenizer_source=ctx["tokenizer_dir"],
            decision_config=config,
        )
        with (destination / "parity_rows.jsonl").open("w") as handle:
            for record in parity:
                handle.write(json.dumps(record) + "\n")
        del state, text
    rt.barrier()
    log.write(
        "events",
        {
            "event": "exported",
            "step": step,
            "path": str(destination),
            "seconds": time.time() - began,
        },
    )


# ---------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------


def handle_signal(signum, _frame) -> None:
    STOP["flag"] = True
    STOP["signal"] = signum


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    rt = Runtime(args.dist_timeout_min)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    M.ATTENTION_BACKEND["name"] = args.attn_backend
    M.CONV_BACKEND["name"] = args.conv_backend
    DCP_THREADS["n"] = args.dcp_threads
    out_dir = Path(args.output)
    log = Logger(out_dir / "logs", rt.main)
    started = time.time()

    tokenizer_dir = Path(args.tokenizer or args.init)
    tokenizer = M.load_tokenizer(tokenizer_dir)
    codes, token_ids = fmt.answer_codes(tokenizer)
    M.check_codes(args.init, codes, token_ids)

    cache = Path(args.cache_dir)
    store = D.RowStore(args.train, cache, build=rt.main) if rt.main else None
    rt.barrier()
    if not rt.main:
        store = D.RowStore(args.train, cache, build=False)
    schedule = D.Schedule(len(store), args.rows_per_update, args.epochs, args.seed)
    encoder = D.Encoder(
        tokenizer,
        codes,
        teacher=args.teacher,
        teacher_weight=args.teacher_weight,
        shuffle_options=args.shuffle_options == "on",
        seed=args.seed,
        max_length=args.max_length,
    )
    dev: list[D.Encoded] = []
    dev_sha = None
    if args.dev:
        dev_store = D.RowStore(args.dev, cache, build=rt.main) if rt.main else None
        rt.barrier()
        if not rt.main:
            dev_store = D.RowStore(args.dev, cache, build=False)
        dev_sha = dev_store.combined_sha256()
        for i in range(len(dev_store)):
            row = dev_store.get(i)
            if row is None or D.validate(row):
                continue
            item = encoder.encode(row, epoch=None, index=i)
            if len(item.ids) <= args.max_length:
                dev.append(item)
            if args.dev_rows and len(dev) >= args.dev_rows:
                break

    config = M.load_config(args.init)
    model = M.build_skeleton(config, args.attention_mode, device="meta")
    ac_layers = configure_ac(model.text, args.ac)
    shard_model(model, rt, args)
    model.to_empty(device=rt.device)
    M.reset_rotary(model, rt.device)
    optimizer = build_optimizer(model, args)

    dcp_root = out_dir / "dcp"
    existing = dcp_dirs(dcp_root) if rt.main else None
    existing = rt.broadcast_object(
        [(s, str(p)) for s, p in existing] if rt.main else None
    )
    ctx: dict[str, Any] = {
        "codes": codes,
        "token_ids": token_ids,
        "config": config,
        "tokenizer_dir": str(tokenizer_dir),
        "data_sha256": store.sha256,
        "dev_sha256": dev_sha,
        "code_sha256": code_sha256(),
        "total_steps": schedule.total,
    }
    run_config = {
        "trainer": TRAINER_VERSION,
        "args": vars(args),
        "world_size": rt.world,
        "train_rows": len(store),
        "dev_rows": len(dev),
        "total_steps": schedule.total,
        "data_sha256": store.sha256,
        "dev_sha256": dev_sha,
        "code_sha256": ctx["code_sha256"],
        "codes_sha256": hashlib.sha256(
            json.dumps([codes, token_ids]).encode()
        ).hexdigest(),
        "ac_layers": len(ac_layers),
        "host": socket.gethostname(),
        "versions": {
            "torch": torch.__version__,
            "hip": getattr(torch.version, "hip", None),
        },
    }
    critical = (
        "train",
        "init",
        "attention_mode",
        "lr",
        "readout_lr",
        "weight_decay",
        "warmup_ratio",
        "schedule",
        "min_lr_ratio",
        "epochs",
        "seed",
        "rows_per_update",
        "max_length",
        "brier_weight",
        "teacher",
        "teacher_weight",
        "shuffle_options",
        "clip",
        "betas",
        "eps",
    )
    step = 0
    rows_seen = 0
    if existing:
        last_step, last_path = existing[-1]
        saved_cfg = json.loads((out_dir / "run_config.json").read_text())
        changed = [k for k in critical if saved_cfg["args"].get(k) != vars(args).get(k)]
        if changed or saved_cfg["data_sha256"] != store.sha256:
            raise RuntimeError(
                f"resume refused: run settings changed {changed} or data changed"
            )
        state = load_dcp(model, optimizer, rt, Path(last_path))
        step, rows_seen = state["step"], state["rows_seen"]
        ctx["readout_source"] = state.get("readout_source", "resumed")
        visual = (
            M.read_checkpoint(args.init, token_ids, parts=("visual",))["visual"]
            if rt.main
            else None
        )
        log.write("events", {"event": "resumed", "step": step, "path": last_path})
    else:
        full = None
        visual = None
        if rt.main:
            ckpt = M.read_checkpoint(args.init, token_ids)
            full = {f"text.{k}": v for k, v in ckpt["text"].items()}
            full["readout.weight"] = ckpt["readout"]
            visual = ckpt["visual"]
            ctx["readout_source"] = ckpt["readout_source"]
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "run_config.json").write_text(
                json.dumps(run_config, indent=2, default=str)
            )
        load_initial_weights(model, rt, full)
        del full
        ctx["readout_source"] = rt.broadcast_object(ctx.get("readout_source"))
        log.write(
            "events",
            {
                "event": "initialised",
                "init": args.init,
                "kind": args.init_kind,
                "readout": ctx["readout_source"],
                "seconds": time.time() - started,
            },
        )
    ctx["visual"] = visual
    ctx["rows_seen"] = rows_seen
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    total = schedule.total
    stop = min(total, args.max_steps) if args.max_steps else total
    export_steps = {int(x) for x in args.export_steps.split(",") if x.strip()}
    export_steps |= {
        max(1, min(total, round(float(x) * total)))
        for x in args.export_fractions.split(",")
        if x.strip()
    }
    if rt.main:
        (out_dir / "plan.json").write_text(
            json.dumps(
                {
                    "total_steps": total,
                    "export_steps": sorted(export_steps),
                    "train_rows": len(store),
                },
                indent=2,
            )
        )
    profile_steps = {int(x) for x in args.profile_steps.split(",") if x.strip()}
    flops_tok = M.flops_per_token(config, 1024.0)
    if rt.main:
        log.write(
            "events",
            {
                "event": "start",
                "step": step,
                "total_steps": total,
                "stop": stop,
                "world": rt.world,
                "train_rows": len(store),
                "dev_rows": len(dev),
                "ac_layers": len(ac_layers),
                "setup_seconds": time.time() - started,
            },
        )
    if args.dev_at_start == "on" and dev and step == 0:
        log.write("dev", {"step": 0, **evaluate(model, rt, dev, args)})

    prefetch = Prefetcher(
        lambda s: make_plan(s, schedule, store, encoder, rt, args), step, stop
    )
    params = [p for p in model.parameters() if p.requires_grad]
    model.train()
    last_save = step
    losses: dict[int, float] = {}
    train_log = out_dir / "logs" / "train.jsonl"
    if rt.main and train_log.exists():
        for line in train_log.read_text().splitlines():
            with contextlib.suppress(Exception):
                item = json.loads(line)
                losses[int(item["step"])] = float(item["loss"])
    sanity_checked = step >= args.sanity_step
    while step < stop:
        t0 = time.time()
        plan = prefetch.get()
        t_data = time.time() - t0
        if plan.step != step:
            raise RuntimeError(f"plan for update {plan.step} arrived at update {step}")
        prof = None
        if rt.main and (step + 1) in profile_steps:
            prof = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=False,
            )
            prof.__enter__()
        optimizer.zero_grad(set_to_none=True)
        scale = rt.world / plan.stats["weight"]
        sums = torch.zeros(4, dtype=torch.float64, device=rt.device)
        for micro in plan.micro:
            inputs, target, w = micro.to(rt.device)
            logits = model(inputs)
            terms = M.row_losses(logits, target)
            per_row = (
                terms["ce"] + args.brier_weight * terms["brier"]
                if args.brier_weight
                else terms["ce"]
            )
            loss = (per_row * w).sum() * scale
            loss.backward()
            sums += torch.stack(
                [
                    (terms["ce"] * w).sum(),
                    (terms["brier"] * w).sum(),
                    (terms["correct"] * w).sum(),
                    w.sum(),
                ]
            ).double()
            del inputs, target, w, logits, terms, per_row, loss
        norm = torch.nn.utils.clip_grad_norm_(params, args.clip, foreach=True)
        norm = float(norm.full_tensor() if hasattr(norm, "full_tensor") else norm)
        factor = lr_factor(step + 1, total, args)
        for group in optimizer.param_groups:
            group["lr"] = group["peak_lr"] * factor
        if math.isfinite(norm):
            optimizer.step()
        dist.all_reduce(sums)
        stop_now = torch.tensor([1.0 if STOP["flag"] else 0.0], device=rt.device)
        dist.all_reduce(stop_now, op=dist.ReduceOp.MAX)
        step += 1
        rows_seen += plan.stats["rows"]
        if prof is not None:
            torch.cuda.synchronize()
            prof.__exit__(None, None, None)
            table = prof.key_averages().table(
                sort_by="self_device_time_total", row_limit=60, max_name_column_width=90
            )
            (out_dir / "logs" / f"profile-step{step}.txt").write_text(table)
            prof.export_chrome_trace(str(out_dir / "logs" / f"trace-step{step}.json"))
        torch.cuda.synchronize()
        seconds = time.time() - t0
        mem = rt.reduce(
            [
                torch.cuda.max_memory_allocated() / 2**30,
                torch.cuda.max_memory_reserved() / 2**30,
            ],
            op=dist.ReduceOp.MAX,
        )
        values = sums.tolist()
        wsum = max(values[3], 1e-12)
        record = {
            "step": step,
            "epoch": plan.epoch,
            "lr": args.lr * factor,
            "loss": (values[0] + args.brier_weight * values[1]) / wsum,
            "ce": values[0] / wsum,
            "brier": values[1] / wsum,
            "accuracy": values[2] / wsum,
            "grad_norm": norm,
            "seconds": seconds,
            "data_wait": t_data,
            "tokens_per_s": plan.stats["tokens"] / seconds,
            "mfu_est": plan.stats["tokens"]
            * flops_tok
            / seconds
            / rt.world
            / 1.3074e15,
            "peak_alloc_gib": mem[0],
            "peak_reserved_gib": mem[1],
            "rows_seen": rows_seen,
            **plan.stats,
        }
        log.write("train", record, echo=(step <= 20 or step % 10 == 0 or step == stop))
        if not math.isfinite(norm) or not math.isfinite(record["loss"]):
            log.write(
                "events",
                {
                    "event": "nonfinite",
                    "step": step,
                    "grad_norm": norm,
                    "loss": record["loss"],
                },
            )
            return EXIT_NONFINITE
        losses[step] = record["loss"]
        if not sanity_checked and step >= args.sanity_step and args.sanity_step >= 20:
            sanity_checked = True
            verdict = None
            if rt.main:
                early = [losses[s] for s in range(6, 16) if s in losses]
                late = [losses[s] for s in range(step - 9, step + 1) if s in losses]
                ok = (
                    not early
                    or not late
                    or sum(late) / len(late) < sum(early) / len(early)
                )
                verdict = {
                    "ok": ok,
                    "early_mean": sum(early) / max(1, len(early)),
                    "late_mean": sum(late) / max(1, len(late)),
                }
                log.write("events", {"event": "sanity", "step": step, **verdict})
            verdict = rt.broadcast_object(verdict)
            if not verdict["ok"]:
                return EXIT_SANITY
        ctx["rows_seen"] = rows_seen
        trainer_state = {
            "step": step,
            "rows_seen": rows_seen,
            "total_steps": total,
            "readout_source": ctx["readout_source"],
            "saved_utc": utc(),
            "trainer": TRAINER_VERSION,
        }
        preempted = stop_now.item() > 0
        if (
            (args.save_every and step % args.save_every == 0)
            or preempted
            or (step == stop and args.final_dcp == "on")
        ):
            save_dcp(
                model, optimizer, rt, dcp_root, step, trainer_state, args.keep_dcp, log
            )
            last_save = step
            if args.stop_after_save and step == args.stop_after_save:
                log.write("events", {"event": "test_stop_after_save", "step": step})
                return EXIT_PREEMPTED
        if preempted:
            log.write(
                "events", {"event": "preempted", "step": step, "signal": STOP["signal"]}
            )
            return EXIT_PREEMPTED
        if dev and args.dev_every and step % args.dev_every == 0 and step != stop:
            began = time.time()
            log.write(
                "dev",
                {
                    "step": step,
                    **evaluate(model, rt, dev, args),
                    "seconds": time.time() - began,
                },
            )
        if step in export_steps or (
            args.export_every and step % args.export_every == 0
        ):
            export(model, rt, args, ctx, step, dev, log)
            if args.pause_after_export and step < stop:
                if last_save != step:
                    save_dcp(
                        model,
                        optimizer,
                        rt,
                        dcp_root,
                        step,
                        trainer_state,
                        args.keep_dcp,
                        log,
                    )
                log.write("events", {"event": "paused_for_eval", "step": step})
                return EXIT_PAUSED
    del last_save
    if dev:
        began = time.time()
        log.write(
            "dev",
            {
                "step": step,
                **evaluate(model, rt, dev, args),
                "seconds": time.time() - began,
            },
        )
    if not args.no_final_export and step == total:
        export(model, rt, args, ctx, step, dev, log)
    log.write(
        "events",
        {
            "event": "finished" if step == total else "stopped",
            "step": step,
            "wall_seconds": time.time() - started,
        },
    )
    rt.barrier()
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(
        code
    )  # skip interpreter teardown (RCCL/NCCL destructors, daemon prefetch thread)
