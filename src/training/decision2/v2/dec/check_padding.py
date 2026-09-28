"""Padded versus unpadded micro-batch equivalence on the real decoder backbone.

The model is built exactly as ``train_dec`` builds it (Decision 1.0 or official
Qwen3.5 start, LoRA, gradient checkpointing, FP32 parameters), except that LoRA
dropout is 0 and every LoRA B tensor is perturbed from a fixed seed so that all
adapter tensors receive gradient. The same rows are then run through four
batchings, and three are compared with the first:

* ``exact``: each row alone with no pad token at all (reference);
* ``single``: each row alone through ``collate`` (the trainer's microbatch-1
  path, right-padded to a multiple of 8);
* ``grouped``: rows right-padded together in micro-batches of ``--group``
  (the multi-row training path), each group mixing short and long rows;
* ``extra``: each row alone with ``--extra-pad`` more pad tokens.

Per row: candidate logits and CE + Brier loss; per trainable group (LoRA,
head): the gradient of the summed loss. Both in FP32 (autocast off) and in the
trainer's BF16 autocast. Inference batching (1 vs 2 vs 8 rows, BF16) is checked
on the same rows. Tolerances are fixed below, before any run.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from training.model.data import file_sha256, load_partition
from training.model.decision_model import DecisionModel, collate, encode
from training.model.lora import attach_lora
from training.model.loss import per_example_loss
from training.model.source import source_fingerprint
from training.model.train import atomic_json

from .dec_model import DecModel
from .runtime_check import require_runtime

SCHEMA = "dec-padding-equivalence/1"
TOLERANCES = {
    "fp32": {
        "max_logit_delta": 1e-3,
        "max_loss_delta": 1e-3,
        "min_grad_cosine": 0.9999,
        "max_grad_rel_l2": 1e-2,
        "argmax_margin": 0.0,
    },
    "bf16": {
        "max_logit_delta": 0.25,
        "max_loss_delta": 0.05,
        "min_grad_cosine": 0.99,
        "max_grad_rel_l2": 0.15,
        "argmax_margin": 0.1,
    },
    "inference_bf16": {"max_probability_delta": 0.02, "argmax_margin": 0.1},
}
MODES = ("exact", "single", "grouped", "extra")


def pick_rows(
    rows: list[dict[str, Any]], lengths: list[int], per_type: int
) -> list[int]:
    """Evenly spaced length quantiles within each native type."""
    chosen: list[int] = []
    for kind in ("choice", "noul", "score"):
        members = sorted(
            (i for i, row in enumerate(rows) if row["task_type"] == kind),
            key=lambda i: (lengths[i], rows[i]["id"]),
        )
        if len(members) < per_type:
            raise ValueError(f"Need {per_type} {kind} rows, found {len(members)}")
        step = (len(members) - 1) / max(1, per_type - 1)
        chosen.extend(members[round(k * step)] for k in range(per_type))
    return chosen


def mixed_groups(indices: list[int], lengths: list[int], size: int) -> list[list[int]]:
    """Groups that pair the shortest remaining rows with the longest ones."""
    ordered = sorted(indices, key=lambda i: lengths[i])
    groups: list[list[int]] = []
    while ordered:
        group: list[int] = []
        while ordered and len(group) < size:
            group.append(ordered.pop(-1 if len(group) % 2 == 0 else 0))
        groups.append(group)
    return groups


def batch_for(
    items: list[dict[str, Any]], pad_id: int, mode: str, extra_pad: int
) -> dict[str, Any]:
    batch = collate(items, pad_id)
    if mode == "exact":
        length = len(items[0]["ids"])
        batch["input_ids"] = batch["input_ids"][:, :length]
        batch["attention_mask"] = batch["attention_mask"][:, :length]
    elif mode == "extra":
        length = len(items[0]["ids"])
        pads = torch.full((1, extra_pad), pad_id, dtype=torch.long)
        batch["input_ids"] = torch.cat([batch["input_ids"][:, :length], pads], 1)
        batch["attention_mask"] = torch.cat(
            [batch["attention_mask"][:, :length], torch.zeros_like(pads)], 1
        )
    return batch


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def run_mode(
    model: Any,
    items: list[dict[str, Any]],
    groups: list[list[int]],
    mode: str,
    precision: str,
    pad_id: int,
    extra_pad: int,
    trainable: dict[str, list[torch.nn.Parameter]],
    device: torch.device,
) -> dict[str, Any]:
    model.zero_grad(set_to_none=True)
    batches = groups if mode == "grouped" else [[i] for i in range(len(items))]
    logits_by_row: dict[int, list[float]] = {}
    loss_by_row: dict[int, float] = {}
    padded_tokens = 0
    for positions in batches:
        subset = [items[i] for i in positions]
        batch = to_device(batch_for(subset, pad_id, mode, extra_pad), device)
        padded_tokens += int((batch["attention_mask"] == 0).sum().item())
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=precision == "bf16"
        ):
            logits = model(**batch)
            terms = per_example_loss(
                logits,
                batch["labels"],
                batch["candidate_mask"],
                objective="ce_brier",
                brier_weight=0.5,
            )
        terms["total"].sum().backward()
        for position, row_logits, item, total in zip(
            positions, logits.detach().float().cpu(), subset, terms["total"].detach()
        ):
            logits_by_row[position] = row_logits[: len(item["keys"])].tolist()
            loss_by_row[position] = float(total)
    gradients = {
        name: torch.cat(
            [
                (p.grad if p.grad is not None else torch.zeros_like(p))
                .detach()
                .double()
                .flatten()
                .cpu()
                for p in params
            ]
        )
        for name, params in trainable.items()
    }
    return {
        "logits": [logits_by_row[i] for i in range(len(items))],
        "loss": [loss_by_row[i] for i in range(len(items))],
        "gradients": gradients,
        "pad_tokens": padded_tokens,
    }


def margin(values: list[float]) -> float:
    top = sorted(values, reverse=True)
    return top[0] - top[1]


def argmax(values: list[float]) -> int:
    return max(range(len(values)), key=values.__getitem__)


def compare(
    reference: dict[str, Any], other: dict[str, Any], tolerance: dict[str, float]
) -> dict[str, Any]:
    logit_delta = max(
        abs(a - b)
        for x, y in zip(reference["logits"], other["logits"])
        for a, b in zip(x, y)
    )
    loss_delta = max(abs(a - b) for a, b in zip(reference["loss"], other["loss"]))
    decisive = [
        i
        for i, x in enumerate(reference["logits"])
        if margin(x) >= tolerance["argmax_margin"]
    ]
    flips = [
        i
        for i in decisive
        if argmax(reference["logits"][i]) != argmax(other["logits"][i])
    ]
    grads = {}
    for name, ref in reference["gradients"].items():
        cur = other["gradients"][name]
        norm = ref.norm().item()
        grads[name] = {
            "reference_norm": norm,
            "cosine": (
                (ref @ cur).item() / (norm * cur.norm().item())
                if norm > 0 and cur.norm().item() > 0
                else float("nan")
            ),
            "rel_l2": (cur - ref).norm().item() / norm if norm > 0 else float("nan"),
        }
    passed = (
        logit_delta <= tolerance["max_logit_delta"]
        and loss_delta <= tolerance["max_loss_delta"]
        and not flips
        and all(
            math.isfinite(g["cosine"])
            and g["cosine"] >= tolerance["min_grad_cosine"]
            and g["rel_l2"] <= tolerance["max_grad_rel_l2"]
            for g in grads.values()
        )
    )
    return {
        "max_logit_delta": logit_delta,
        "max_loss_delta": loss_delta,
        "decisive_rows": len(decisive),
        "argmax_flips": flips,
        "gradients": grads,
        "pad_tokens": other["pad_tokens"],
        "pass": passed,
    }


def inference_probabilities(
    model: Any,
    items: list[dict[str, Any]],
    batch_size: int,
    pad_id: int,
    device: torch.device,
) -> list[list[float]]:
    output: list[list[float]] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(items), batch_size):
            subset = items[start : start + batch_size]
            batch = to_device(collate(subset, pad_id), device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(**batch).float()
            for item, row in zip(subset, logits.softmax(-1).cpu().tolist()):
                output.append(row[: len(item["keys"])])
    model.train()
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--init", choices=("decision1", "base"), default="decision1")
    parser.add_argument("--revision", help="Immutable revision for --init base")
    parser.add_argument("--rows", type=Path, required=True, help="TRAIN partition")
    parser.add_argument("--per-type", type=int, default=6)
    parser.add_argument("--group", type=int, default=4)
    parser.add_argument("--extra-pad", type=int, default=61)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.init == "base" and not args.revision:
        parser.error("--init base needs --revision")
    runtime = require_runtime()
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0")

    rows = load_partition(args.rows, "train")
    if args.init == "decision1":
        base, tokenizer = DecisionModel.from_decision1(args.model_path, args.head_dim)
        source_kind = "decision1"
    else:
        base, tokenizer = DecisionModel.from_base(
            args.model_path, args.revision, args.head_dim
        )
        source_kind = "base"
    lengths = [len(encode(row, tokenizer, args.max_length)["ids"]) for row in rows]
    chosen = pick_rows(rows, lengths, args.per_type)
    items = [encode(rows[i], tokenizer, args.max_length) for i in chosen]
    item_lengths = [len(item["ids"]) for item in items]
    groups = mixed_groups(list(range(len(items))), item_lengths, args.group)

    model = DecModel.wrap(base)
    attach_lora(
        model,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=0.0,
        source_kind=source_kind,
        source_fingerprint=source_fingerprint(args.model_path),
    )
    generator = torch.Generator().manual_seed(args.seed)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.copy_(
                    torch.randn(parameter.shape, generator=generator) * 1e-3
                )
    model = model.float().to(device)
    model.backbone.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.backbone.config.use_cache = False
    model.train()
    trainable = defaultdict(list)
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            trainable["head" if name.startswith("head.") else "lora"].append(parameter)
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )

    results: dict[str, Any] = {}
    reference_by_precision: dict[str, dict[str, Any]] = {}
    for precision in ("fp32", "bf16"):
        outputs = {}
        for mode in MODES:
            try:
                outputs[mode] = run_mode(
                    model,
                    items,
                    groups,
                    mode,
                    precision,
                    pad_id,
                    args.extra_pad,
                    trainable,
                    device,
                )
            except Exception as exc:  # recorded, then the gate fails
                results[f"{precision}_{mode}_error"] = repr(exc)
        if "exact" not in outputs:
            continue
        reference_by_precision[precision] = outputs["exact"]
        for mode in MODES[1:]:
            if mode in outputs:
                results[f"{precision}_{mode}_vs_exact"] = compare(
                    outputs["exact"], outputs[mode], TOLERANCES[precision]
                )
        if precision == "bf16":
            repeat = run_mode(
                model,
                items,
                groups,
                "exact",
                "bf16",
                pad_id,
                args.extra_pad,
                trainable,
                device,
            )
            results["bf16_exact_repeat"] = compare(
                outputs["exact"], repeat, TOLERANCES["bf16"]
            )
    if set(reference_by_precision) == {"fp32", "bf16"}:
        results["reference_bf16_vs_fp32_exact"] = compare(
            reference_by_precision["fp32"],
            reference_by_precision["bf16"],
            TOLERANCES["bf16"],
        )

    one = inference_probabilities(model, items, 1, pad_id, device)
    inference = {}
    for size in (2, 8):
        other = inference_probabilities(model, items, size, pad_id, device)
        tol = TOLERANCES["inference_bf16"]
        logit_margin = [
            math.log(max(sorted(p)[-1], 1e-30)) - math.log(max(sorted(p)[-2], 1e-30))
            for p in one
        ]
        flips = [
            i
            for i, (a, b) in enumerate(zip(one, other))
            if logit_margin[i] >= tol["argmax_margin"] and argmax(a) != argmax(b)
        ]
        drift = max(abs(x - y) for a, b in zip(one, other) for x, y in zip(a, b))
        inference[f"batch{size}_vs_batch1"] = {
            "max_probability_delta": drift,
            "argmax_flips": flips,
            "pass": drift <= tol["max_probability_delta"] and not flips,
        }
    results["inference"] = inference

    gated = [
        key
        for key in results
        if key.endswith("_vs_exact") and key.split("_")[0] in ("fp32", "bf16")
    ]
    errors = [key for key in results if key.endswith("_error")]
    status = (
        "PASS"
        if gated
        and not errors
        and all(results[k]["pass"] for k in gated)
        and all(v["pass"] for v in inference.values())
        and "bf16_single_vs_exact" in results
        and "bf16_grouped_vs_exact" in results
        else "FAIL"
    )
    receipt = {
        "schema_version": SCHEMA,
        "status": status,
        "gated": sorted(gated) + [f"inference.{k}" for k in inference],
        "tolerances": TOLERANCES,
        "model_path": str(args.model_path),
        "init": args.init,
        "revision": args.revision,
        "rows_sha256": file_sha256(args.rows),
        "rows": [
            {
                "id": rows[i]["id"],
                "task_type": rows[i]["task_type"],
                "tokens": lengths[i],
                "options": len(rows[i]["options"]),
            }
            for i in chosen
        ],
        "groups": groups,
        "trainable_parameters": {
            k: sum(p.numel() for p in v) for k, v in trainable.items()
        },
        "runtime": runtime,
        "device_name": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, receipt)
    print(json.dumps({"status": status, "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
