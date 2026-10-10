"""Multimodal decision fine-tuning for Decision 2.5 Omni on one node (FSDP2).

    torchrun --standalone --nproc_per_node 8 -m d25.omni.train.train \\
        --init INIT --arm O-graft-frozen --rows MM_ROWS... --replay-rows TEXT_ROWS... \\
        --replay-ratio 0.5 --out OUT

``INIT`` is an assembled code-readout v1 checkpoint (``d25.omni.model.assemble``); its init kind must
match the arm (``vega`` for the graft arms, ``stock`` for O-fresh). Each optimizer step takes
``--effective-batch-size`` rows (main corpus plus the replay share), split over the ranks into
token-budget microbatches; the loss is the weight-normalised soft-target cross-entropy of the step.
Choice options are shuffled per step and row (targets follow), as in the Perplexity recipe.

Outputs in ``OUT``: ``config.json`` (arguments, data and code sha256, schedule digest),
``training.jsonl``, ``evaluations.jsonl`` (``--dev-rows``), ``checkpoints/step-XXXXX/`` (code-readout
v1, BF16 backbone including the vision tower, FP32 readout), ``resume/step-XXXXX/`` (latest only)
and ``summary.json``. A restart with the same arguments and code resumes from the latest resume
point. ``--dry-run`` stops after planning (rows, token counts, schedule, microbatches) and writes
``plan.json``; it needs no accelerator.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import torch

from d25.omni.common import vision_format
from d25.omni.model import checkpoint, inputs
from d25.omni.model.attention import ATTENTION_MODES
from d25.omni.train import arms, fsdp, schedule
from d25.omni.train.collator import MultimodalCollator, shuffle_options
from d25.omni.train.loss import mask_logits, row_losses, step_loss
from d25.omni.train.model import init_vision_digests, save_checkpoint
from d25.omni.train.rows import TrainRow, measure, read_rows, unique_ids

CODE_FILES = (
    "train.py",
    "collator.py",
    "schedule.py",
    "loss.py",
    "model.py",
    "arms.py",
    "fsdp.py",
    "rows.py",
)
SHARED_CODE = (
    "model/inputs.py",
    "model/attention.py",
    "model/checkpoint.py",
    "common/vision_format.py",
)
RESUME_INSENSITIVE = {
    "stop_after",
    "save_every",
    "resume_every",
    "eval_every",
    "log_every",
    "dry_run",
    "code_tag",
    "plan_world",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--init", required=True)
    parser.add_argument("--arm", choices=sorted(arms.ARMS), required=True)
    parser.add_argument("--encoder-lr-scale", type=float)
    parser.add_argument("--merger-lr-scale", type=float)
    parser.add_argument("--allow-init-mismatch", action="store_true")
    parser.add_argument(
        "--rows", nargs="*", default=[], help="main (multimodal) corpus files"
    )
    parser.add_argument(
        "--replay-rows", nargs="*", default=[], help="text replay corpus files"
    )
    parser.add_argument(
        "--replay-ratio", type=float, default=0.5, help="share of replay rows per step"
    )
    parser.add_argument("--dev-rows", nargs="*", default=[])
    parser.add_argument("--out", required=True)
    parser.add_argument("--attention-mode", choices=ATTENTION_MODES)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--lr-floor", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--no-decay-1d",
        action="store_true",
        help="exclude norms and biases from weight decay",
    )
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--brier-weight", type=float, default=0.0)
    parser.add_argument("--effective-batch-size", type=int, default=256)
    parser.add_argument(
        "--token-budget",
        type=int,
        default=32_768,
        help="padded tokens per microbatch per GPU",
    )
    parser.add_argument("--max-rows-per-microbatch", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=16_384)
    parser.add_argument("--max-pixels", type=int, default=vision_format.MAX_PIXELS)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--no-shuffle-options", action="store_true")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--resume-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument(
        "--stop-after",
        type=int,
        help="pilot: stop after this many steps (LR schedule unchanged)",
    )
    parser.add_argument(
        "--code-tag", default="", help="source tag of the shipped code copy"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--plan-world",
        type=int,
        default=8,
        help="ranks assumed by --dry-run on one process",
    )
    args = parser.parse_args(argv)
    if not 0 <= args.replay_ratio <= 1 or not 0 < args.warmup_ratio < 1:
        parser.error("--replay-ratio must be in [0, 1] and --warmup-ratio in (0, 1)")
    return args


def digest_files(paths: list[str]) -> dict[str, str]:
    return {name: checkpoint.file_sha256(name) for name in paths}


def code_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    files = {name: checkpoint.file_sha256(root / name) for name in CODE_FILES}
    files.update(
        {name: checkpoint.file_sha256(root.parent / name) for name in SHARED_CODE}
    )
    return files


def run_digest(config: dict[str, Any]) -> str:
    stable = {k: v for k, v in config.items() if k != "args"}
    stable["args"] = {
        k: v for k, v in config["args"].items() if k not in RESUME_INSENSITIVE
    }
    return hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()


def measured(
    rows: list[TrainRow], processor, codes, prompt, max_length, distributed
) -> collections.Counter:
    """Set ``row.tokens`` (computed in parallel over ranks; 0 for dropped rows); returns drop reasons."""
    local = measure(
        rows[distributed.rank :: distributed.world],
        processor,
        codes,
        prompt,
        max_length,
    )
    parts = distributed.gather_objects(local)
    reasons: collections.Counter = collections.Counter()
    for index, row in enumerate(rows):
        tokens, reason = parts[index % distributed.world][index // distributed.world]
        row.tokens = 0 if reason else tokens
        if reason:
            reasons[reason] += 1
    return reasons


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def run_microbatches(model, collator, rows, plan, dummy, device, transform=None):
    """``(batch, logits)`` for each microbatch of one rank; empty slots run ``dummy`` at weight 0."""
    for indices in plan:
        batch_rows = [
            transform(rows[i]) if transform else rows[i] for i in indices
        ] or [dummy]
        batch = collator(batch_rows, dummy=not indices).to(device)
        yield batch, model(**batch.inputs)


@torch.no_grad()
def evaluate(model, collator, rows, args, distributed) -> dict[str, float]:
    tokens = [row.tokens for row in rows]
    plan = schedule.plan_step(
        range(len(rows)),
        tokens,
        distributed.world,
        args.token_budget,
        args.max_rows_per_microbatch,
    )
    dummy = min(rows, key=lambda row: row.tokens)
    model.eval()
    totals = [0.0, 0.0, 0.0]
    for batch, logits in run_microbatches(
        model, collator, rows, plan[distributed.rank], dummy, distributed.device
    ):
        losses = row_losses(logits, batch.targets, batch.counts)
        correct = (
            mask_logits(logits, batch.counts).argmax(-1) == batch.targets.argmax(-1)
        ).float()
        totals[0] += float((losses.cross_entropy * batch.weights).sum())
        totals[1] += float((correct * batch.weights).sum())
        totals[2] += float(batch.weights.sum())
    loss_sum, correct_sum, weight_sum = distributed.sum(totals)
    model.train()
    return {
        "loss": loss_sum / weight_sum,
        "accuracy": correct_sum / weight_sum,
        "rows": len(rows),
    }


def archive_stale(out: Path, start: int) -> None:
    """Move checkpoints newer than the resume point out of the way (an interrupted tail)."""
    stale = [
        p
        for p in (out / "checkpoints").glob("step-*")
        if int(p.name.split("-")[1].split(".")[0]) > start
    ]
    if stale:
        target = out / "checkpoints" / f"interrupted-{time.time_ns()}"
        target.mkdir(parents=True)
        for path in stale:
            path.rename(target / path.name)


def plan_report(config, plan, rows, tokens, args) -> dict[str, Any]:
    sample = plan.steps[: min(len(plan.steps), 50)]
    layouts = [
        schedule.plan_step(
            group,
            tokens,
            args.plan_world,
            args.token_budget,
            args.max_rows_per_microbatch,
        )
        for group in sample
    ]
    step_tokens = [sum(tokens[i] for i in group) for group in plan.steps]
    return {
        **config,
        "tokens": {
            "per_step_mean": sum(step_tokens) / max(1, len(step_tokens)),
            "per_step_max": max(step_tokens, default=0),
            "max_row": max(tokens, default=0),
            "image_row_share": sum(1 for row in rows if row.images) / max(1, len(rows)),
            "image_token_share": sum(row.tokens for row in rows if row.images)
            / max(1, sum(tokens)),
        },
        "plan_world": args.plan_world,
        "microbatches_per_step_mean": sum(len(layout[0]) for layout in layouts)
        / max(1, len(layouts)),
        "padding_efficiency": sum(
            schedule.padding_efficiency(layout, tokens) for layout in layouts
        )
        / max(1, len(layouts)),
    }


def main(argv: list[str] | None = None) -> None:
    import transformers
    from transformers import AutoProcessor

    args = parse_args(argv)
    distributed = fsdp.initialize()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    init, out = Path(args.init), Path(args.out)
    decision = checkpoint.read_decision_config(init)
    arm = arms.ARMS[args.arm]
    scales = arm.scales()
    if args.encoder_lr_scale is not None:
        scales["encoder"] = args.encoder_lr_scale
    if args.merger_lr_scale is not None:
        scales["merger"] = args.merger_lr_scale
    init_provenance = decision.get("provenance") or {}
    init_kind = (init_provenance.get("init") or {}).get("kind")
    if init_kind != arm.init and not args.allow_init_mismatch:
        raise SystemExit(f"{args.arm} needs a {arm.init} init; {init} is {init_kind!r}")
    attention_mode = args.attention_mode or decision.get("attention_mode", "causal")
    prompt = decision.get("prompt", "d25-vega")
    processor = inputs.setup_processor(
        AutoProcessor.from_pretrained(str(init)), args.max_pixels
    )
    codes = list(decision["codes"])
    inputs.check_codes(processor.tokenizer, codes, decision["token_ids"])

    main_rows, main_invalid = read_rows(args.rows, "main")
    replay_rows, replay_invalid = read_rows(args.replay_rows, "replay")
    dev_rows, dev_invalid = read_rows(args.dev_rows, "dev")
    unique_ids(main_rows + replay_rows + dev_rows)
    drops = measured(
        main_rows + replay_rows + dev_rows,
        processor,
        codes,
        prompt,
        args.max_length,
        distributed,
    )
    rows = [row for row in main_rows + replay_rows if row.tokens > 0]
    dev_rows = [row for row in dev_rows if row.tokens > 0]
    plan = schedule.build_schedule(
        [i for i, row in enumerate(rows) if row.corpus == "main"],
        [i for i, row in enumerate(rows) if row.corpus == "replay"],
        batch_size=args.effective_batch_size,
        replay_ratio=args.replay_ratio,
        epochs=args.epochs,
        seed=args.seed,
    )
    total = len(plan.steps)
    stop = schedule.total_steps(plan, args.stop_after)
    tokens = [row.tokens for row in rows]
    config = {
        "args": vars(args),
        "init_decision_config_sha256": checkpoint.file_sha256(
            init / checkpoint.DECISION_CONFIG
        ),
        "init_kind": init_kind,
        "scales": scales,
        "attention_mode": attention_mode,
        "data_sha256": {
            "rows": digest_files(args.rows),
            "replay_rows": digest_files(args.replay_rows),
            "dev_rows": digest_files(args.dev_rows),
        },
        "code_sha256": code_hashes(),
        "rows": {
            "main": plan.main_rows,
            "replay": plan.replay_rows,
            "dev": len(dev_rows),
        },
        "dropped": {
            "invalid": dict(main_invalid + replay_invalid + dev_invalid),
            "measure": dict(drops),
        },
        "schedule": {
            "steps": total,
            "sha256": plan.digest(),
            "replay_passes": plan.replay_passes,
        },
        "world_size": distributed.world,
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
    }
    digest = run_digest(config)
    config["digest"] = digest
    if distributed.primary:
        out.mkdir(parents=True, exist_ok=True)
    distributed.barrier()
    if args.dry_run:
        if distributed.primary:
            checkpoint.write_json(
                out / "plan.json", plan_report(config, plan, rows, tokens, args)
            )
        distributed.close()
        return

    summary_path = out / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        if summary.get("digest") == digest and summary.get("steps", 0) >= stop:
            if distributed.primary:
                print(json.dumps({"event": "already complete", **summary}), flush=True)
            distributed.close()
            return
    resume_root = out / "resume"
    latest = fsdp.latest_resume(resume_root)
    if distributed.primary:
        if (out / "config.json").exists():
            if json.loads((out / "config.json").read_text()).get("digest") != digest:
                raise SystemExit(f"{out} holds a different run; choose a new --out")
        else:
            checkpoint.write_json(out / "config.json", config)
    distributed.barrier()

    if distributed.device.type == "cuda":
        torch.backends.cuda.enable_cudnn_sdp(False)
    model, trainable_counts = fsdp.build(
        init,
        distributed,
        attention_mode,
        scales,
        gradient_checkpointing=not args.no_gradient_checkpointing,
    )
    groups = arms.param_groups(
        model, scales, args.weight_decay, decay_1d=not args.no_decay_1d
    )
    optimizer = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    start = 0
    if latest is not None:
        state = fsdp.load_resume(model, optimizer, latest)
        if state["digest"] != digest:
            raise SystemExit("resume state belongs to a different run configuration")
        start = state["step"]
    if distributed.primary:
        archive_stale(out, start)
    distributed.barrier()
    collator = MultimodalCollator(processor, codes, prompt)
    trainable = [p for p in model.parameters() if p.requires_grad]
    vision_trainable = [
        p
        for n, p in model.named_parameters()
        if p.requires_grad and n.startswith("backbone.visual.")
    ]
    init_digests = init_vision_digests(init) if distributed.primary else {}
    started = time.time()
    if distributed.primary:
        print(
            json.dumps(
                {
                    "event": "start",
                    "step": start,
                    "stop": stop,
                    "total_steps": total,
                    "trainable": trainable_counts,
                }
            ),
            flush=True,
        )

    def save_model(number: int) -> None:
        state = fsdp.full_state(model, distributed)
        if distributed.primary:
            provenance = {
                "run": out.name,
                "step": number,
                "total_steps": total,
                "arm": args.arm,
                "scales": scales,
                "lr": args.lr,
                "replay_ratio": args.replay_ratio,
                "effective_batch_size": args.effective_batch_size,
                "epochs": args.epochs,
                "seed": args.seed,
                "run_digest": digest,
                "data_sha256": config["data_sha256"],
                "schedule_sha256": config["schedule"]["sha256"],
                "code_sha256": config["code_sha256"],
                "code_tag": args.code_tag,
                "init": {
                    "path": str(init),
                    "kind": init_kind,
                    "decision_config_sha256": config["init_decision_config_sha256"],
                    "shards_sha256": init_provenance.get("shards_sha256"),
                    "vision": init_provenance.get("vision"),
                },
                "versions": config["versions"],
                "world_size": distributed.world,
            }
            _, digests = save_checkpoint(
                out / "checkpoints" / f"step-{number:05d}",
                state,
                init,
                {"attention_mode": attention_mode},
                provenance,
            )
            for part, scale in (
                ("vision_encoder", scales["encoder"]),
                ("vision_merger", scales["merger"]),
            ):
                names = [
                    name for name in init_digests if checkpoint.component(name) == part
                ]
                if scale == 0 and any(
                    digests.get(name) != init_digests[name] for name in names
                ):
                    raise RuntimeError(f"frozen {part} changed during training")
        del state
        distributed.barrier()

    for step in range(start, stop):
        group = plan.steps[step]
        layout = schedule.plan_step(
            group,
            tokens,
            distributed.world,
            args.token_budget,
            args.max_rows_per_microbatch,
        )
        total_weight = sum(rows[i].weight for i in group)
        dummy = rows[min(group, key=lambda i: tokens[i])]

        def augment(row: TrainRow, step: int = step) -> TrainRow:
            if args.no_shuffle_options:
                return row
            return shuffle_options(row, random.Random(f"{args.seed}:{step}:{row.id}"))

        began = time.time()
        model.train()
        sums = [0.0] * 6
        for batch, logits in run_microbatches(
            model,
            collator,
            rows,
            layout[distributed.rank],
            dummy,
            distributed.device,
            augment,
        ):
            losses = row_losses(logits, batch.targets, batch.counts)
            loss = step_loss(
                losses,
                batch.weights,
                total_weight,
                distributed.world,
                args.brier_weight,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step + 1}")
            loss.backward()
            real = float(batch.weights.sum()) > 0
            sums[0] += float((losses.cross_entropy.detach() * batch.weights).sum())
            sums[1] += float((losses.brier.detach() * batch.weights).sum())
            sums[2] += float(batch.weights.sum())
            sums[3] += batch.tokens if real else 0
            sums[4] += batch.images if real else 0
            sums[5] += len(batch.ids) if real else 0
        ce, brier, weight, step_tokens, images, count = distributed.sum(sums)
        if images == 0:
            for parameter in vision_trainable:
                parameter.grad = None
        norm = torch.nn.utils.clip_grad_norm_(trainable, args.clip)
        norm = float(norm.full_tensor() if hasattr(norm, "full_tensor") else norm)
        if not math.isfinite(norm):
            raise RuntimeError(f"non-finite gradient norm at step {step + 1}")
        factor = schedule.learning_rate_factor(
            step + 1, total, args.warmup_ratio, args.lr_floor
        )
        for group_state in optimizer.param_groups:
            group_state["lr"] = args.lr * factor * group_state["lr_scale"]
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        number = step + 1
        if distributed.primary and (number % args.log_every == 0 or number == stop):
            record = {
                "step": number,
                "loss": ce / weight,
                "brier": brier / weight,
                "lr": args.lr * factor,
                "grad_norm": norm,
                "rows": int(count),
                "replay_rows": sum(1 for i in group if rows[i].corpus == "replay"),
                "images": int(images),
                "tokens": int(step_tokens),
                "microbatches": len(layout[0]),
                "step_seconds": round(time.time() - began, 3),
                "elapsed_seconds": round(time.time() - started, 1),
            }
            if distributed.device.type == "cuda":
                record["gpu_peak_gb"] = round(
                    torch.cuda.max_memory_allocated() / 1e9, 2
                )
            append_jsonl(out / "training.jsonl", record)
            print(json.dumps(record), flush=True)
        if dev_rows and (number % args.eval_every == 0 or number == stop):
            metrics = evaluate(model, collator, dev_rows, args, distributed)
            if distributed.primary:
                append_jsonl(out / "evaluations.jsonl", {"step": number, **metrics})
        if number % args.save_every == 0 or number == stop:
            save_model(number)
        if number % args.resume_every == 0 and number < stop:
            fsdp.save_resume(
                model,
                optimizer,
                resume_root / f"step-{number:05d}",
                {"step": number, "digest": digest},
                distributed,
            )
    if distributed.primary:
        checkpoint.write_json(
            out / "summary.json",
            {
                "run": out.name,
                "steps": stop,
                "planned_steps": total,
                "complete": stop == total,
                "digest": digest,
                "elapsed_seconds": round(time.time() - started, 1),
                "checkpoints": sorted(
                    p.name for p in (out / "checkpoints").glob("step-*") if p.is_dir()
                ),
            },
        )
    distributed.barrier()
    distributed.close()


if __name__ == "__main__":
    main()
