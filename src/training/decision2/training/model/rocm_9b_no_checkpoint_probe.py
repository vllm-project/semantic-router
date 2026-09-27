"""Bounded Qwen decision-model backward probe with checkpointing disabled.

Loads an existing exact checkpoint in trainable-adapter mode, then repeatedly
computes gradients from synthetic token IDs. No optimizer is constructed and
no parameter or checkpoint is modified. This is the single-variable counterpart
to the already failed 20-iteration gradient-checkpointed probe. Run only on a
reserved accelerator after the prospective protocol is approved.
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import math
import time
from pathlib import Path

import torch

from training.model.decision_model import DecisionModel
from training.model.loss import per_example_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument(
        "--lengths", type=int, nargs="+", default=[512, 1024, 2048, 4096, 6144]
    )
    return parser.parse_args()


def main() -> None:
    faulthandler.enable()
    args = parse_args()
    if not 1 <= args.iterations <= 100:
        raise ValueError("iterations must be 1..100")
    if not args.lengths or any(length < 16 or length > 8192 for length in args.lengths):
        raise ValueError("all synthetic lengths must be 16..8192")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0")
    model, tokenizer = DecisionModel.from_checkpoint(
        args.checkpoint,
        source_path=args.source,
        trainable_adapter=True,
    )
    model = model.float().to(device)
    if any(
        getattr(module, "gradient_checkpointing", False)
        for module in model.backbone.modules()
    ):
        raise RuntimeError("Gradient checkpointing must be disabled for this cell")
    model.backbone.config.use_cache = False
    model.train()
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable:
        raise RuntimeError("checkpoint has no trainable adapter or head parameters")
    original = [parameter.detach().clone() for parameter in trainable]
    motif = tokenizer.encode(
        "Synthetic runtime probe; choose a numbered option.", add_special_tokens=False
    )
    if not motif:
        raise RuntimeError("synthetic motif tokenization is empty")
    print(
        json.dumps(
            {
                "event": "start",
                "iterations": args.iterations,
                "seed": args.seed,
                "lengths": args.lengths,
                "trainable_parameters": sum(
                    parameter.numel() for parameter in trainable
                ),
                "optimizer": None,
                "gradient_checkpointing": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    started = time.perf_counter()
    for iteration in range(args.iterations):
        length = args.lengths[iteration % len(args.lengths)]
        ids = (motif * math.ceil(length / len(motif)))[:length]
        batch = {
            "input_ids": torch.tensor([ids], dtype=torch.long, device=device),
            "attention_mask": torch.ones((1, length), dtype=torch.long, device=device),
            "candidate_positions": torch.tensor(
                [[length // 4, length // 2, (3 * length) // 4]],
                dtype=torch.long,
                device=device,
            ),
            "candidate_mask": torch.ones((1, 3), dtype=torch.bool, device=device),
            "query_positions": torch.tensor(
                [length - 1], dtype=torch.long, device=device
            ),
        }
        labels = torch.tensor([iteration % 3], dtype=torch.long, device=device)
        model.zero_grad(set_to_none=True)
        step_started = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(**batch)
            terms = per_example_loss(
                logits,
                labels,
                batch["candidate_mask"],
                objective="ce_brier",
                brier_weight=0.5,
            )
            loss = terms["total"].mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite synthetic loss at iteration {iteration + 1}")
        loss.backward()
        torch.cuda.synchronize(device)
        if any(
            parameter.grad is not None and not torch.isfinite(parameter.grad).all()
            for parameter in trainable
        ):
            raise RuntimeError(
                f"nonfinite synthetic gradient at iteration {iteration + 1}"
            )
        print(
            json.dumps(
                {
                    "event": "iteration",
                    "iteration": iteration + 1,
                    "length": length,
                    "seconds": round(time.perf_counter() - step_started, 3),
                    "peak_allocated_gib": round(
                        torch.cuda.max_memory_allocated(device) / 2**30, 3
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    model.zero_grad(set_to_none=True)
    if not all(
        torch.equal(parameter.detach(), value)
        for parameter, value in zip(trainable, original)
    ):
        raise RuntimeError("a trainable weight changed despite no optimizer")
    print(
        json.dumps(
            {
                "event": "complete",
                "iterations": args.iterations,
                "seconds": round(time.perf_counter() - started, 3),
                "weights_unchanged": True,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
