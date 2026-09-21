"""Train full-encoder retrieval tasks from explicit hashed configuration.

Source admission, experiment approval and release gates belong to the caller's
recipe. This module enforces mathematical, input and checkpoint consistency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from .newbase_batches import ObjectiveConfig, embedding_step, reranker_step
from .newbase_data import FrozenCorpus, file_digest
from .newbase_evaluation import select_candidate
from .newbase_model import ExitSpec, NewBaseTask, state_digest, verify_files
from .newbase_scoring import score_corpus
from .newbase_stream import load_stream
from .newbase_teacher import TeacherCache, validate_teacher_config

_ADAM_MOMENTS = 2


def optimizer_groups(
    model, encoder_lr: float, head_lr: float, weight_decay: float = 0.01
) -> tuple[list[dict], list[dict]]:
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or parameter.dtype != torch.float32:
            raise ValueError("Training may not silently freeze or round a parameter")
        is_head = name.startswith("layer_heads.")
        decay = parameter.ndim > 1 and "norm" not in name.lower()
        key = (is_head, decay)
        group = groups.setdefault(
            key,
            {
                "params": [],
                "names": [],
                "lr": head_lr if is_head else encoder_lr,
                "weight_decay": weight_decay if decay else 0.0,
            },
        )
        group["params"].append(parameter)
        group["names"].append(name)
    receipts = [
        {
            "names": group["names"],
            "lr": group["lr"],
            "weight_decay": group["weight_decay"],
            "parameters": sum(parameter.numel() for parameter in group["params"]),
        }
        for group in groups.values()
    ]
    return [
        {key: value for key, value in group.items() if key != "names"}
        for group in groups.values()
    ], receipts


def make_optimizer(model, *, steps: int, warmup: int, config: dict | None = None):
    if type(warmup) is not int or not 0 <= warmup < steps:
        raise ValueError("Warmup must finish before the training budget")
    config = config or {}
    encoder_lr, head_lr = config.get("encoder_lr", 2e-5), config.get("head_lr", 1e-4)
    weight_decay, epsilon = config.get("weight_decay", 0.01), config.get("eps", 1e-8)
    minimum_ratio, betas = (
        config.get("minimum_lr_ratio", 0.1),
        tuple(config.get("betas", (0.9, 0.999))),
    )
    if (
        any(
            not math.isfinite(value) or value <= 0
            for value in (encoder_lr, head_lr, epsilon)
        )
        or not math.isfinite(weight_decay)
        or weight_decay < 0
        or not 0 <= minimum_ratio <= 1
        or len(betas) != _ADAM_MOMENTS
        or any(not 0 <= value < 1 for value in betas)
    ):
        raise ValueError("Invalid finite AdamW/schedule configuration")
    groups, receipt = optimizer_groups(model, encoder_lr, head_lr, weight_decay)
    optimizer = torch.optim.AdamW(groups, betas=betas, eps=epsilon)

    def multiplier(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / (steps - warmup))
        return minimum_ratio + (1 - minimum_ratio) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier), receipt


def save_training_state(
    model,
    tokenizer,
    optimizer,
    scheduler,
    directory: Path,
    *,
    identity: dict,
    step: int,
    elapsed: float,
    evaluations: list | None = None,
    existing_model: bool = False,
) -> None:
    if existing_model:
        receipt = json.loads((directory / "newbase_checkpoint.json").read_bytes())
        verify_files(directory, receipt["files"])
        if state_digest(model.state_dict()) != receipt["state_sha256"]:
            raise ValueError("Model changed during separate development evaluation")
    else:
        model.save(directory, tokenizer)
    state = {
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "python_rng": random.getstate(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "step": step,
        "elapsed_seconds": elapsed,
        "identity": identity,
        "evaluations": evaluations or [],
    }
    torch.save(state, directory / "training_state.pt")
    (directory / "training_state.json").write_text(
        json.dumps(
            {
                "step": step,
                "identity": identity,
                "training_state_sha256": file_digest(directory / "training_state.pt"),
            },
            indent=2,
        )
        + "\n"
    )


def restore_training_state(
    directory: Path, optimizer, scheduler, *, identity: dict
) -> dict:
    receipt = json.loads((directory / "training_state.json").read_bytes())
    if (
        receipt["identity"] != identity
        or file_digest(directory / "training_state.pt")
        != receipt["training_state_sha256"]
    ):
        raise ValueError("Optimizer/RNG checkpoint belongs to a different locked run")
    state = torch.load(
        directory / "training_state.pt", map_location="cpu", weights_only=True
    )
    if state["identity"] != identity or state["step"] != receipt["step"]:
        raise ValueError("Training state and receipt disagree")
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    random.setstate(state["python_rng"])
    torch.set_rng_state(state["torch_rng"])
    if state["cuda_rng"]:
        if len(state["cuda_rng"]) != torch.cuda.device_count():
            raise ValueError("Resume requires the same visible accelerator count")
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return state


def _read_locked(path: Path, expected: str) -> dict:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"Locked file changed: {path.name}")
    return json.loads(raw)


def _verify_code(plan: dict):
    lock = _read_locked(
        Path(plan["execution_lock_path"]), plan["execution_lock_sha256"]
    )
    root = Path(plan["code_root"]).resolve()
    verify_files(root, lock["files"])
    package = __package__ + "."
    for name, module in list(sys.modules.items()):
        if not name.startswith(package) or not getattr(module, "__file__", None):
            continue
        path = Path(module.__file__).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Actual task import escaped the frozen code directory")
        if str(path.relative_to(root)) not in lock["files"]:
            raise ValueError("Actual task module is missing from the execution lock")
    return lock


def validate_config(config: dict) -> None:
    """Validate a mathematical run description, without interpreting approval."""
    if (
        config.get("task") not in {"embedding", "reranker"}
        or not isinstance(config.get("run_id"), str)
        or not config["run_id"]
    ):
        raise ValueError("Explicit task and nonempty run_id are required")
    steps = config["steps"]
    if type(steps) is not int or steps <= 0:
        raise ValueError("Training steps must be a positive integer")
    evaluations = config["eval_steps"]
    if (
        not isinstance(evaluations, list)
        or not evaluations
        or evaluations != sorted(set(evaluations))
        or any(type(step) is not int or not 0 <= step <= steps for step in evaluations)
        or evaluations[0] != 0
    ):
        raise ValueError(
            "Evaluation steps must start at zero and increase within budget"
        )
    seconds = config.get("maximum_seconds")
    if seconds is not None and (
        isinstance(seconds, bool) or not math.isfinite(seconds) or seconds <= 0
    ):
        raise ValueError("Optional time budget must be positive and finite")
    if config["training_precision"] not in {"float32", "bfloat16"}:
        raise ValueError(
            "Training uses FP32 masters with explicit FP32 or BF16 forward"
        )
    for objective in config["objectives"].values():
        ObjectiveConfig(**objective).validate(config["task"])
    if not config["objectives"]:
        raise ValueError("At least one explicit source objective is required")
    if type(config["seed"]) is not int:
        raise ValueError("An integer seed is required")
    if (
        not math.isfinite(config["gradient_clip_norm"])
        or config["gradient_clip_norm"] <= 0
    ):
        raise ValueError("Gradient clipping requires a positive finite norm")
    for key in ("training_token_budget", "evaluation_token_budget"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError("Token budgets must be positive integers")
    ExitSpec.from_dict(config["exits"])
    if config.get("teacher") is not None:
        validate_teacher_config(config["teacher"], config["task"])
    if config.get("anchor") is not None:
        validate_teacher_config(config["anchor"], config["task"], anchor=True)
        layers = config["anchor"].get("target_layers")
        if layers is not None and set(layers) != set(config["exits"]["layers"]):
            raise ValueError(
                "Anchor target_layers must cover every selected student depth"
            )
    if config.get("initialization", "fresh_base") not in {
        "fresh_base",
        "continued_task",
    }:
        raise ValueError(
            "Initialization must distinguish fresh Base from continued task"
        )


def run(
    plan: dict,
    plan_sha: str,
    output: Path,
    *,
    device,
    resume: Path | None = None,
):
    validate_config(plan)
    _verify_code(plan)
    if device.type == "cuda" and torch.cuda.device_count() != 1:
        raise ValueError("Expose exactly one assigned accelerator before training")
    task = plan["task"]
    run_id = plan["run_id"]
    corpus = FrozenCorpus.load(
        Path(plan["train_directory"]), plan.get("train_split", "train")
    )
    development = FrozenCorpus.load(
        Path(plan["development_directory"]), plan.get("development_split", "validation")
    )
    if (
        corpus.manifest_sha256 != plan["train_manifest_sha256"]
        or development.manifest_sha256 != plan["development_manifest_sha256"]
    ):
        raise ValueError("Dataset manifests differ from the configured run")
    draws, stream = load_stream(Path(plan["draws_directory"]), corpus)
    if stream["draws_sha256"] != plan["draws_sha256"] or len(draws) != plan["steps"]:
        raise ValueError("Training stream differs from the configured run")
    teacher = plan.get("teacher") or {}
    teacher_cache = (
        TeacherCache.load(
            Path(teacher["directory"]),
            teacher["manifest_sha256"],
            task=task,
            corpus=corpus,
            draws=draws,
            draws_sha256=stream["draws_sha256"],
        )
        if teacher.get("weight", 0)
        else None
    )
    anchor = plan.get("anchor") or {}
    anchor_cache = (
        TeacherCache.load(
            Path(anchor["directory"]),
            anchor["manifest_sha256"],
            task=task,
            corpus=corpus,
            draws=draws,
            draws_sha256=stream["draws_sha256"],
            target_layers=anchor.get("target_layers"),
        )
        if anchor.get("weight", 0)
        else None
    )
    baseline = (
        _read_locked(Path(plan["baseline_path"]), plan["baseline_sha256"])
        if plan.get("selection")
        else None
    )
    seed = plan["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    base = Path(plan["base_directory"])
    initialize = (
        NewBaseTask.from_task
        if plan.get("initialization") == "continued_task"
        else NewBaseTask.from_base
    )
    model = (
        NewBaseTask.resume(resume)
        if resume
        else initialize(
            base,
            task,
            ExitSpec.from_dict(plan["exits"]),
            expected_files=plan["base_files"],
            provenance=plan.get("provenance"),
        )
    )
    if model.task != task:
        raise ValueError("Checkpoint task differs from the configured task")
    if (
        not resume
        and plan.get("expected_initial_state_sha256") is not None
        and state_digest(model.state_dict()) != plan["expected_initial_state_sha256"]
    ):
        raise ValueError(
            "Fresh initialization differs from its configured state identity"
        )
    if plan["gradient_checkpointing"]:
        model.encoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    tokenizer = AutoTokenizer.from_pretrained(
        base, local_files_only=True, trust_remote_code=False
    )
    if teacher_cache is not None:
        teacher_cache.validate_tokens(
            tokenizer, corpus.components, max(plan["source_maximum_tokens"].values())
        )
    if anchor_cache is not None:
        anchor_cache.validate_tokens(
            tokenizer, corpus.components, max(plan["source_maximum_tokens"].values())
        )
    model.to(device)
    frozen_buffers = state_digest(dict(model.encoder.named_buffers()))
    optimizer, scheduler, parameters = make_optimizer(
        model,
        steps=plan["steps"],
        warmup=plan["optimizer"]["warmup_steps"],
        config=plan["optimizer"],
    )
    identity = {
        "plan_sha256": plan_sha,
        "execution_lock_sha256": plan["execution_lock_sha256"],
        "run_id": run_id,
        "train_manifest_sha256": corpus.manifest_sha256,
        "draws_sha256": stream["draws_sha256"],
        "expected_initial_state_sha256": plan.get("expected_initial_state_sha256"),
    }
    if teacher_cache is not None:
        identity["teacher_manifest_sha256"] = teacher_cache.manifest_sha256
    if anchor_cache is not None:
        identity["anchor_manifest_sha256"] = anchor_cache.manifest_sha256
    completed, elapsed_before = 0, 0.0
    evaluations = []
    if resume:
        state = restore_training_state(resume, optimizer, scheduler, identity=identity)
        completed, elapsed_before = state["step"], state["elapsed_seconds"]
        evaluations = state["evaluations"]
    output.mkdir(parents=True, exist_ok=False)
    (output / "run.json").write_text(
        json.dumps(
            {
                **identity,
                "optimizer_groups": parameters,
                "resumed_from": str(resume) if resume else None,
            },
            indent=2,
        )
        + "\n"
    )
    by_id = {row["id"]: row for row in corpus.records}
    start, error, attempted = time.monotonic(), None, completed

    def elapsed():
        return elapsed_before + time.monotonic() - start

    def checkpoint(step):
        directory = output / f"step-{step}"
        model.save(directory, tokenizer)
        devices = (
            [device.index if device.index is not None else torch.cuda.current_device()]
            if device.type == "cuda"
            else []
        )
        with torch.random.fork_rng(devices=devices):
            evaluation_model = NewBaseTask.resume(directory).to(device).eval()
            report, _ = score_corpus(
                evaluation_model,
                tokenizer,
                development,
                device=device,
                token_budget=plan["evaluation_token_budget"],
                evaluation_config=plan.get("evaluation"),
            )
            del evaluation_model
        report.update(
            {
                "run_id": run_id,
                "step": step,
                "model_state_sha256": state_digest(model.state_dict()),
            }
        )
        evaluations.append(report)
        model.train()
        if step == 0:
            random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        save_training_state(
            model,
            tokenizer,
            optimizer,
            scheduler,
            directory,
            identity=identity,
            step=step,
            elapsed=elapsed(),
            evaluations=evaluations,
            existing_model=True,
        )
        (directory / "development.json").write_text(json.dumps(report, indent=2) + "\n")

    try:
        if not resume:
            checkpoint(completed)
        model.train()
        with (output / "steps.jsonl").open("w", encoding="utf-8") as log:
            for draw in draws[completed:]:
                if (
                    plan.get("maximum_seconds") is not None
                    and elapsed() >= plan["maximum_seconds"]
                ):
                    break
                attempted = draw["step"]
                optimizer.zero_grad(set_to_none=True)
                selected = [by_id[key] for key in draw["record_ids"]]
                function = embedding_step if task == "embedding" else reranker_step
                task_options = (
                    {
                        "anchor_cache": anchor_cache,
                        "anchor_weight": anchor.get("weight", 0.0),
                        "anchor_target_layers": anchor.get("target_layers"),
                    }
                    if task == "embedding"
                    else {
                        "teacher_exit_supervision": teacher.get(
                            "exit_supervision", "full"
                        )
                    }
                )
                loss, metrics = function(
                    model,
                    tokenizer,
                    selected,
                    corpus.components,
                    device=device,
                    token_budget=plan["training_token_budget"],
                    maximum=plan["source_maximum_tokens"][draw["source"]],
                    objective=ObjectiveConfig(**plan["objectives"][draw["source"]]),
                    amp=plan["training_precision"] == "bfloat16",
                    teacher_cache=teacher_cache,
                    teacher_weight=teacher.get("weight", 0.0),
                    teacher_temperature=teacher.get("temperature", 2.0),
                    **task_options,
                )
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite logical objective")
                loss.backward()
                if any(
                    parameter.grad is None or not torch.isfinite(parameter.grad).all()
                    for parameter in model.parameters()
                ):
                    raise ValueError(
                        "A full-encoder/head gradient is missing or nonfinite"
                    )
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    plan["gradient_clip_norm"],
                    error_if_nonfinite=True,
                )
                optimizer.step()
                if any(
                    not torch.isfinite(parameter).all()
                    for parameter in model.parameters()
                ):
                    raise ValueError("Optimizer produced a nonfinite parameter")
                if any(
                    not torch.isfinite(value).all()
                    for state in optimizer.state.values()
                    for value in state.values()
                    if isinstance(value, torch.Tensor)
                ):
                    raise ValueError("Optimizer produced a nonfinite moment or step")
                scheduler.step()
                if state_digest(dict(model.encoder.named_buffers())) != frozen_buffers:
                    raise ValueError("Training mutated a frozen encoder/RoPE buffer")
                completed = draw["step"]
                log.write(
                    json.dumps(
                        {
                            "step": completed,
                            "source": draw["source"],
                            "stratum": draw["stratum"],
                            "record_ids": draw["record_ids"],
                            "loss": loss.detach().item(),
                            "gradient_norm_before_clip": float(norm),
                            "elapsed_seconds": elapsed(),
                            **metrics,
                        }
                    )
                    + "\n"
                )
                log.flush()
                if completed in plan["eval_steps"]:
                    checkpoint(completed)
        if completed not in plan["eval_steps"]:
            save_training_state(
                model,
                tokenizer,
                optimizer,
                scheduler,
                output / "last-incomplete",
                identity=identity,
                step=completed,
                elapsed=elapsed(),
                evaluations=evaluations,
            )
    except BaseException as exception:
        error = type(exception).__name__
        raise
    finally:
        selection = (
            select_candidate(evaluations, baseline, plan["selection"])
            if evaluations and plan.get("selection")
            else None
        )
        (output / "completion.json").write_text(
            json.dumps(
                {
                    **identity,
                    "completed_steps": completed,
                    "attempted_steps": attempted,
                    "expected_steps": len(draws),
                    "all_draws_complete": completed == len(draws),
                    "error_type": error,
                    "elapsed_seconds": elapsed(),
                    "selection": selection,
                    "current_step_or_evaluation_may_overrun": True,
                },
                indent=2,
            )
            + "\n"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    plan = _read_locked(args.config, args.config_sha256)
    run(
        plan,
        args.config_sha256,
        args.output,
        device=torch.device(args.device),
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
