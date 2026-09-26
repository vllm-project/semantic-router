"""One fixed-budget LoRA arm and gold-free native reranker collection.

Run only after the pilot manifest and preregistered gate are recorded. The
script never downloads weights and never reads release-panel gold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

from inference.run import digest, load_prompts

from training.model.data import file_sha256, load_partition

from . import pilot

SEED = 20260927
UPDATES = 64
ACCUMULATION = 8
LORA_RANK = 8
LORA_ALPHA = 16
LORA_LR = 2e-5
NEGATIVE_CAP = 8
TRAIN_CONTRACT = "qwen3-rerank06-sampled8-checkpointed-v2"
TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def _libraries() -> tuple[Any, Any, Any, Any, Any]:
    import torch
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    return (
        torch,
        (LoraConfig, PeftModel, TaskType, get_peft_model),
        AutoModelForCausalLM,
        AutoTokenizer,
        None,
    )


def verify_adapter(adapter: Path, source_receipt: dict[str, Any]) -> dict[str, Any]:
    receipt_path = adapter / "decision2_pilot_receipt.json"
    weights_path = adapter / "adapter_model.safetensors"
    if not receipt_path.is_file() or not weights_path.is_file():
        raise ValueError("Incomplete trained adapter")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("contract") != pilot.ADAPTER
        or receipt.get("source") != source_receipt
        or receipt.get("source_train_sha256")
        != "32a1226931967fd7a0f53cb2a4fd51189d5b643c56eeb1b88cea94ebf4312398"
        or receipt.get("optimizer_updates") != UPDATES
        or receipt.get("train_contract") != TRAIN_CONTRACT
        or receipt.get("adapter_sha256") != file_sha256(weights_path)
    ):
        raise ValueError("Trained adapter receipt mismatch")
    return receipt


def load(
    source: Path, adapter: Path | None, *, training: bool = False
) -> tuple[Any, Any, Any, int, int]:
    source = source.resolve(strict=True)
    source_receipt = pilot.verify_source(source)
    torch, peft, AutoModelForCausalLM, AutoTokenizer, _ = _libraries()
    LoraConfig, PeftModel, TaskType, get_peft_model = peft
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA-compatible BF16 GPU unavailable")
    torch.manual_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(source, padding_side="left")
    yes_ids = tokenizer("yes", add_special_tokens=False).input_ids
    no_ids = tokenizer("no", add_special_tokens=False).input_ids
    if len(yes_ids) != 1 or len(no_ids) != 1 or yes_ids == no_ids:
        raise ValueError("Native yes/no tokenization differs")
    model = AutoModelForCausalLM.from_pretrained(source, dtype=torch.bfloat16).to(
        "cuda"
    )
    if sum(parameter.numel() for parameter in model.parameters()) != 595_776_512:
        raise ValueError("Reranker parameter count differs")
    if training:
        config = LoraConfig(
            r=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            lora_dropout=0,
            target_modules=TARGET_MODULES,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, config)
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model.train()
    elif adapter is not None:
        verify_adapter(adapter, source_receipt)
        model = PeftModel.from_pretrained(model, adapter.resolve(strict=True)).eval()
    else:
        model.eval()
    return torch, tokenizer, model, no_ids[0], yes_ids[0]


def margins(
    model: Any,
    torch: Any,
    tokenizer: Any,
    state: Any,
    question: dict[str, Any],
    no_id: int,
    yes_id: int,
    candidate_indices: list[int] | None = None,
) -> tuple[Any, list[str], int]:
    sequences, keys = pilot.encode(tokenizer, state, question)
    if candidate_indices is not None:
        sequences = [sequences[index] for index in candidate_indices]
        keys = [keys[index] for index in candidate_indices]
    batch = tokenizer.pad(
        [{"input_ids": ids} for ids in sequences],
        padding=True,
        pad_to_multiple_of=8,
        return_tensors="pt",
    ).to("cuda")
    output = model(**batch, logits_to_keep=1).logits[:, -1, :]
    values = output[:, yes_id].float() - output[:, no_id].float()
    return values, keys, max(map(len, sequences))


def selected_indices(row_id: str, keys: list[str], target: str) -> list[int]:
    if target not in keys:
        raise ValueError("Gold target absent from candidate set")
    if len(keys) <= NEGATIVE_CAP:
        return list(range(len(keys)))
    positive = keys.index(target)
    negatives = sorted(
        (index for index in range(len(keys)) if index != positive),
        key=lambda index: hashlib.sha256(
            f"{SEED}:{row_id}:{keys[index]}".encode()
        ).hexdigest(),
    )
    return sorted([positive, *negatives[: NEGATIVE_CAP - 1]])


def predict_row(
    model: Any,
    torch: Any,
    tokenizer: Any,
    state: Any,
    question: dict[str, Any],
    no_id: int,
    yes_id: int,
) -> tuple[dict[str, Any], int]:
    try:
        with torch.inference_mode():
            values, keys, length = margins(
                model, torch, tokenizer, state, question, no_id, yes_id
            )
            answer = pilot.project(question, keys, values.cpu().tolist())
        return answer, length
    except ValueError as exc:
        if str(exc) != "context_overflow":
            raise
        return {
            "type": question["type"],
            "error": "context_overflow",
            "native_max_positions": pilot.MAX_TOKENS,
        }, 0


def run_train(
    source: Path, data: Path, data_manifest: Path, select: Path, output: Path
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    source_receipt = pilot.verify_source(source)
    expected = json.loads(data_manifest.read_text(encoding="utf-8"))
    if (
        file_sha256(data) != expected["output_sha256"]
        or expected["source"] != source_receipt
        or expected["train_sha256"] != pilot.TRAIN_SHA
        or file_sha256(select) != pilot.SELECT_SHA
    ):
        raise ValueError("Pilot source/data manifest mismatch")
    rows = load_partition(data, "train")
    if len(rows) != UPDATES * ACCUMULATION:
        raise ValueError("Fixed pilot train size differs")
    torch, tokenizer, model, no_id, yes_id = load(source, None, training=True)
    from torch.nn import functional as F

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=LORA_LR,
        weight_decay=0.01,
    )
    ordered = sorted(rows, key=lambda row: row["id"])
    random.Random(SEED).shuffle(ordered)
    start = time.monotonic()
    update_losses = []
    max_tokens_seen = 0
    optimizer.zero_grad(set_to_none=True)
    for index, row in enumerate(ordered):
        question = pilot.row_question(row)
        target_key = pilot.target_key(row)
        all_keys = [key for key, _ in pilot.option_items(question)]
        chosen_indices = selected_indices(row["id"], all_keys, target_key)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            values, keys, length = margins(
                model,
                torch,
                tokenizer,
                row["state"],
                question,
                no_id,
                yes_id,
                candidate_indices=chosen_indices,
            )
            target = torch.tensor([keys.index(target_key)], device="cuda")
            loss = F.cross_entropy(values[None, :], target)
        if not torch.isfinite(loss):
            raise ValueError("Non-finite loss")
        (loss / ACCUMULATION).backward()
        max_tokens_seen = max(max_tokens_seen, length)
        update_losses.append(float(loss.detach().cpu()))
        if (index + 1) % ACCUMULATION == 0:
            torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad), 1.0
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step = (index + 1) // ACCUMULATION
            if step % 8 == 0:
                print(
                    json.dumps(
                        {
                            "step": step,
                            "mean_last_8_loss": sum(update_losses[-8:]) / 8,
                            "max_tokens_seen": max_tokens_seen,
                        }
                    ),
                    flush=True,
                )
    output.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(output)
    receipt = {
        "contract": pilot.ADAPTER,
        "train_contract": TRAIN_CONTRACT,
        "source": source_receipt,
        "source_train_sha256": file_sha256(data),
        "optimizer_updates": UPDATES,
        "logical_batch": ACCUMULATION,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "lora_lr": LORA_LR,
        "negative_cap": NEGATIVE_CAP,
        "gradient_checkpointing": True,
        "target_modules": TARGET_MODULES,
        "max_tokens": pilot.MAX_TOKENS,
        "max_tokens_seen": max_tokens_seen,
        "loss_mean": sum(update_losses) / len(update_losses),
        "duration_sec": time.monotonic() - start,
        "adapter_sha256": file_sha256(output / "adapter_model.safetensors"),
        "pilot_code_sha256": file_sha256(Path(pilot.__file__)),
        "trainer_code_sha256": file_sha256(Path(__file__)),
    }
    (output / "decision2_pilot_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    return receipt


def run_select(
    source: Path, adapter: Path | None, select: Path, output: Path
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    if file_sha256(select) != pilot.SELECT_SHA:
        raise ValueError("SELECT file differs")
    rows = load_partition(select, "select")
    torch, tokenizer, model, no_id, yes_id = load(source, adapter)
    counts = Counter()
    with output.open("x", encoding="utf-8") as stream:
        for index, row in enumerate(rows):
            answer, length = predict_row(
                model,
                torch,
                tokenizer,
                row["state"],
                pilot.row_question(row),
                no_id,
                yes_id,
            )
            counts["invalid" if "error" in answer else "valid"] += 1
            counts["over_512"] += int(length > 512)
            stream.write(
                json.dumps({"id": row["id"], "answer": answer}, sort_keys=True) + "\n"
            )
            if (index + 1) % 100 == 0:
                print(
                    json.dumps({"completed": index + 1, "total": len(rows)}), flush=True
                )
    return {
        "prediction_sha256": file_sha256(output),
        "counts": dict(counts),
        "source": pilot.verify_source(source),
        "adapter_sha256": (
            verify_adapter(adapter, pilot.verify_source(source))["adapter_sha256"]
            if adapter
            else None
        ),
        "adapter_version": pilot.ADAPTER,
        "pilot_code_sha256": file_sha256(Path(pilot.__file__)),
        "trainer_code_sha256": file_sha256(Path(__file__)),
    }


def run_panel(
    source: Path, adapter: Path | None, prompts: Path, output: Path, limit: int | None
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    rows = load_prompts(prompts)
    if limit is not None:
        rows = rows[:limit]
    torch, tokenizer, model, no_id, yes_id = load(source, adapter)
    identity = {
        "backend": "qwen3-reranker-06b",
        "model_id": pilot.MODEL_ID,
        "model_revision": pilot.MODEL_REVISION,
        "adapter_version": pilot.ADAPTER,
        "model_weights_sha256": pilot.MODEL_FILES["model.safetensors"],
        "adapter_sha256": (
            file_sha256(adapter / "adapter_model.safetensors") if adapter else None
        ),
    }
    counts = Counter()
    with output.open("x", encoding="utf-8") as stream:
        for index, row in enumerate(rows):
            started = time.perf_counter()
            answers = {}
            for name, question in row["questions"].items():
                answer, length = predict_row(
                    model, torch, tokenizer, row["state"], question, no_id, yes_id
                )
                answers[name] = answer
                counts["invalid" if "error" in answer else "valid"] += 1
                counts["over_512"] += int(length > 512)
            item = {
                "id": row["id"],
                "answers": answers,
                "latency_ms": (time.perf_counter() - started) * 1000,
                "usage": None,
                "source_input_sha256": digest(
                    {"state": row["state"], "questions": row["questions"]}
                ),
                **identity,
            }
            stream.write(json.dumps(item, sort_keys=True) + "\n")
            if (index + 1) % 100 == 0:
                print(
                    json.dumps({"completed": index + 1, "total": len(rows)}), flush=True
                )
    return {
        "prediction_sha256": file_sha256(output),
        "counts": dict(counts),
        "prompts_sha256": file_sha256(prompts),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for action in ("train", "select", "panel"):
        command = sub.add_parser(action)
        command.add_argument("--source", type=Path, required=True)
        command.add_argument("--adapter", type=Path)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--report", type=Path, required=True)
        if action == "train":
            command.add_argument("--data", type=Path, required=True)
            command.add_argument("--data-manifest", type=Path, required=True)
            command.add_argument("--select", type=Path, required=True)
        elif action == "select":
            command.add_argument("--select", type=Path, required=True)
        else:
            command.add_argument("--prompts", type=Path, required=True)
            command.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.command == "train":
        result = run_train(
            args.source, args.data, args.data_manifest, args.select, args.output
        )
    elif args.command == "select":
        result = run_select(args.source, args.adapter, args.select, args.output)
    else:
        result = run_panel(
            args.source, args.adapter, args.prompts, args.output, args.limit
        )
    args.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
