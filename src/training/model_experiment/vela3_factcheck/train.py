"""Adapt Vela3 FactCheck from the shared Router weights on one GPU."""

import argparse
import copy
import gc
import json
import math
import random
import time
from dataclasses import replace
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch.nn.functional import cross_entropy, softplus
from vllm_srun.config import ModelConfig, ServeConfig
from vllm_srun.engines.native.adapters import load_adapter_network, module_name
from vllm_srun.families.vela2.encoder_layout import batch_indices
from vllm_srun.families.vela2.members import INDEX_INPUTS
from vllm_srun.runtime import Runtime

from .data import NEGATIVE, POSITIVE, read_rows
from .package import ADAPTER, release_files, sha256

FACTCHECK = "router:factcheck"
QUESTION = "fact_check_classifier:factcheck"
INSTRUCTIONS = "Does answering this request require checking facts?"
MEDIUM_SEQUENCE = 256
LONG_SEQUENCE = 512
PARITY_TOLERANCE = 0.005


def questions(model):
    result = {"type": "choice", "instructions": INSTRUCTIONS}
    schema = model.router.router_question(INSTRUCTIONS)
    result["criteria"] = {
        label: schema.options[label] for label in (NEGATIVE, POSITIVE)
    }
    if model.info.limits.get("max_scan_tokens", 0):
        result["overflow"] = "truncate"
    return {QUESTION: result}


def make_batches(examples, rng):
    order = examples.copy()
    rng.shuffle(order)
    batches = []
    for offset in range(0, len(order), 64):
        block = sorted(order[offset : offset + 64], key=lambda row: len(row[0].ids))
        while block:
            width = len(block[-1][0].ids)
            size = 1 if width > LONG_SEQUENCE else 2 if width > MEDIUM_SEQUENCE else 4
            batches.append(block[-size:])
            del block[-size:]
    rng.shuffle(batches)
    return batches


def fit(args, model, placement, rows, pairs, rng):
    request = questions(model)
    device = model.engine_model.device

    def sequence(text):
        items = model.plan(text, request).items
        if len(items) != 1 or items[0].call.network != FACTCHECK:
            raise ValueError("expected one FactCheck adapter input")
        return items[0].sequence

    examples = [
        (
            sequence(row["text"]),
            (NEGATIVE, POSITIVE).index(row["label"]),
            row.get("weight", 1.0),
        )
        for row in rows
    ]
    pair_sequences = [
        (sequence(row["neg_text"]), sequence(row["pos_text"])) for row in pairs
    ]
    engine = model.engine_model
    spec = engine.spec.backbone
    source = spec.adapters["router"].weight_files[0]
    branch = load_adapter_network(engine.backbone, spec, "router").to(device).eval()
    branch.kernels = engine.kernels
    head = copy.deepcopy(model.networks.heads["router"])
    reference_plan = model.plan(rows[0]["text"], request)
    reference_item = reference_plan.items[0]
    reference_item.call = replace(reference_item.call, network="router")
    reference = placement.accelerator.execute(
        device, lambda: model.networks.run(reference_plan.items)
    )[0][0][0]
    engine.branches.clear()
    model.networks.heads.clear()
    gc.collect()
    torch.cuda.empty_cache()

    branch.requires_grad_(False)
    with safe_open(source, framework="pt") as reader:
        names = reader.keys()
        factors = {
            name: branch.get_parameter(
                module_name(name.removeprefix("encoder."))
            ).requires_grad_(True)
            for name in names
            if name.endswith((".lora_A", ".lora_B"))
        }
    trainable = {**factors, **dict(head.named_parameters())}
    anchors = {
        name: value.detach()
        .to(torch.bfloat16 if name in factors else torch.float32)
        .clone()
        for name, value in trainable.items()
    }

    def forward(sequences):
        width = max(len(s.ids) for s in sequences)
        ids = torch.full(
            (len(sequences), width), model.networks.pad, device=device, dtype=torch.long
        )
        mask = torch.zeros_like(ids)
        for row, seq in enumerate(sequences):
            ids[row, : len(seq.ids)] = torch.tensor(seq.ids, device=device)
            mask[row, : len(seq.ids)] = 1
        indices = batch_indices(list(sequences))
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            hidden = branch(ids, mask)
        logits, _ = head(
            hidden, *(torch.from_numpy(indices[key]).to(device) for key in INDEX_INPUTS)
        )
        return logits.reshape(len(sequences), -1)[:, :2]

    with torch.no_grad():
        local = forward([examples[0][0]]).softmax(-1)[0, 1].item()
        expected = torch.as_tensor(reference[:2]).softmax(-1)[1].item()
        parity_error = abs(local - expected)
    if parity_error > PARITY_TOLERANCE:
        raise ValueError(
            f"training and runtime probability differ by {parity_error:.5f}"
        )

    batches = [
        batch for _ in range(args.epochs) for batch in make_batches(examples, rng)
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": list(factors.values()), "lr": 3e-5},
            {"params": list(head.parameters()), "lr": 1e-4},
        ]
    )
    total_steps = math.ceil(len(batches) / 4)
    warmup = max(1, int(total_steps * 0.08))

    def lrs(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.1 + 0.45 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lrs)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for offset in range(0, len(batches), 4):
        optimizer.zero_grad(set_to_none=True)
        for j, batch in enumerate(batches[offset : offset + 4]):
            sequences, targets, weights = zip(*batch, strict=True)
            logits = forward(sequences)
            target = torch.tensor(targets, device=device)
            gap = logits[:, 1] - logits[:, 0]
            ce = cross_entropy(logits, target, reduction="none", label_smoothing=0.05)
            loss = (
                ce * torch.tensor(weights, device=device)
            ).mean() + 0.005 * gap.square().mean()
            if j == 0 and pair_sequences and args.pair_weight:
                paired = forward(pair_sequences[(offset // 4) % len(pair_sequences)])
                margins = paired[:, 1] - paired[:, 0]
                pair_loss = softplus(2.5 - (margins[1] - margins[0]))
                pair_loss += cross_entropy(
                    paired, target.new_tensor([0, 1]), label_smoothing=0.05
                )
                loss = loss + args.pair_weight * pair_loss
            (loss / 4).backward()
        with torch.no_grad():
            for name, value in trainable.items():
                if value.grad is not None:
                    value.grad.add_(
                        value.detach() - anchors[name].to(value.dtype), alpha=0.02
                    )
        torch.nn.utils.clip_grad_norm_(
            list(trainable.values()), 1.0, error_if_nonfinite=True
        )
        optimizer.step()
        scheduler.step()
        if offset % 400 == 0:
            print(
                f"step {offset // 4 + 1}/{total_steps}, loss {loss.item():.4f}",
                flush=True,
            )

    torch.cuda.synchronize(device)
    with safe_open(source, framework="pt") as reader:
        metadata = {**reader.metadata(), "adapter": "factcheck"}
        tensors = {}
        names = reader.keys()
        for name in names:
            original = reader.get_tensor(name)
            value = trainable.get(name)
            tensors[name] = (
                original
                if value is None
                else value.detach().cpu().to(original.dtype).contiguous()
            )
    output = args.out / ADAPTER
    save_file(tensors, output, metadata=metadata)
    return {
        "seed": args.seed,
        "epochs": args.epochs,
        "pair_weight": args.pair_weight,
        "rows": len(rows),
        "pairs": len(pairs),
        "train_sha256": sha256(args.train),
        "augmentation_sha256": sha256(args.augmentation),
        "initial_adapter_sha256": sha256(source),
        "adapter_sha256": sha256(output),
        "runtime_parity_error": parity_error,
        "steps": total_steps,
        "seconds": time.perf_counter() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "torch_version": str(torch.__version__),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--base-release", default="vllm-sr/vela3-fast")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--augmentation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--pair-weight", type=float, default=0.5)
    args = parser.parse_args()
    if args.epochs < 1 or not math.isfinite(args.pair_weight) or args.pair_weight < 0:
        parser.error("epochs must be positive and pair-weight finite and nonnegative")
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    rows = read_rows(args.train)
    pairs = []
    for row in read_rows(args.augmentation):
        (pairs if row.get("kind") == "pair" else rows).append(row)
    rng.shuffle(rows)
    release = release_files(args.model, args.base_release)
    args.out.mkdir(parents=True)
    runtime = Runtime(
        ServeConfig(
            models=(ModelConfig(model=str(args.model.resolve()), device=args.device),)
        )
    )
    try:
        runtime.load()
        served = runtime.lookup(None)
        receipt = fit(args, served.model, served.placement, rows, pairs, rng)
        receipt["base_repo"] = release.repo_id
        receipt["base_revision"] = release.revision
        receipt["base_model_sha256"] = release.model_sha256
        (args.out / "run.json").write_text(
            json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(receipt))
    finally:
        runtime.stop()


if __name__ == "__main__":
    main()
