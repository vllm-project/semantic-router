"""Prospective 0.6B option-order LoRA screen; private data and weights only.

The completed sampled-eight control remains immutable. This module starts the
treatment from the same pinned source, changes only the Choice training loss,
and uses the original gold-free native inference adapter.
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

from training.model.data import (
    check_partition_isolation,
    file_sha256,
    load_partition,
)

from . import pilot, train

CONTRACT = "qwen3-rerank06-paired-choice-order-v1"
CONTROL_SHA = "d6fb8ac11f9e9cb95fc082b759c5ec5c716ad200c6efb735fe0bf41cc5e3a05a"
TRAIN512_SHA = "32a1226931967fd7a0f53cb2a4fd51189d5b643c56eeb1b88cea94ebf4312398"
RIGHTS_SHA = "61aa883052759830c4ecf897b36c1062ad816c935a12db824c80abd1f80e9ee8"
CAL_SHA = "3e34f6cb5a32c9f14d0fee0897ee3f2318e59d66fe1ff0a95e2ea5eb2497f60a"
SMOKE_ROWS = 32
KL_WEIGHT = 0.2


def reverse_choice(question: dict[str, Any]) -> dict[str, Any]:
    """Reorder Choice entries while retaining semantic keys and descriptions."""
    if question.get("type") != "choice":
        raise ValueError("Only Choice admits option-order intervention")
    criteria = question.get("criteria")
    if not isinstance(criteria, dict) or len(criteria) < 2:
        raise ValueError("Malformed Choice criteria")
    reversed_criteria = dict(reversed(list(criteria.items())))
    if list(criteria) == list(reversed_criteria):
        raise ValueError("Choice option order did not change")
    return {**question, "criteria": reversed_criteria}


def aligned_indices(
    original_keys: list[str], reordered_keys: list[str], chosen_indices: list[int]
) -> list[int]:
    """Index reordered candidates in original selected-key order for KL."""
    if len(set(original_keys)) != len(original_keys) or set(original_keys) != set(
        reordered_keys
    ):
        raise ValueError("Option keys differ across paired views")
    chosen = [original_keys[index] for index in chosen_indices]
    return [reordered_keys.index(key) for key in chosen]


def _inputs(
    source: Path,
    control: Path,
    train512: Path,
    train_manifest: Path,
    full_train: Path,
    select: Path,
    cal: Path,
    rights_manifest: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    source_identity = pilot.verify_source(source)
    if file_sha256(control / "adapter_model.safetensors") != CONTROL_SHA:
        raise ValueError("Completed control adapter differs")
    train.verify_adapter(control, source_identity)
    expected = {
        train512: TRAIN512_SHA,
        full_train: pilot.TRAIN_SHA,
        select: pilot.SELECT_SHA,
        cal: CAL_SHA,
        rights_manifest: RIGHTS_SHA,
    }
    for path, sha in expected.items():
        if file_sha256(path) != sha:
            raise ValueError(f"Pinned input differs: {path.name}")
    manifest = json.loads(train_manifest.read_text(encoding="utf-8"))
    if (
        manifest.get("output_sha256") != TRAIN512_SHA
        or manifest.get("source") != source_identity
        or manifest.get("train_sha256") != pilot.TRAIN_SHA
        or manifest.get("select_sha256") != pilot.SELECT_SHA
    ):
        raise ValueError("TRAIN512 derivation receipt differs")
    rights = json.loads(rights_manifest.read_text(encoding="utf-8"))
    if rights.get("publication_eligible") is not True:
        raise ValueError("Rights-clean source is not eligible")
    for name, result in rights.get("overlap_audits", {}).items():
        if not (name.startswith("train_vs_") or name.startswith("select_vs_")):
            continue
        for field in (
            "id_rows",
            "group_id_rows",
            "input_sha256_rows",
            "raw_context_rows",
            "normalized_context_rows",
        ):
            if result.get(field) != 0:
                raise ValueError(f"Rights source overlap audit failed: {name}")
        if result.get("near_context", {}).get("count") != 0:
            raise ValueError(f"Rights source near-overlap audit failed: {name}")
    rows = load_partition(train512, "train")
    selected = load_partition(select, "select")
    calibrated = load_partition(cal, "cal")
    if (len(rows), len(selected), len(calibrated)) != (512, 700, 700):
        raise ValueError("Frozen partition counts differ")
    check_partition_isolation({"train": rows, "select": selected, "cal": calibrated})
    parent = {row["id"]: row for row in load_partition(full_train, "train")}
    if any(parent.get(row["id"]) != row for row in rows):
        raise ValueError("TRAIN512 row is not byte-equivalent to parent TRAIN")
    if Counter(row["task_type"] for row in rows) != {
        "choice": 192,
        "noul": 192,
        "score": 128,
    }:
        raise ValueError("Frozen TRAIN512 type composition differs")
    return rows, selected, source_identity


def _self_check_view(rows: list[dict[str, Any]], tokenizer: Any) -> dict[str, int]:
    choice = 0
    longest = 0
    for row in rows:
        question = pilot.row_question(row)
        original, keys = pilot.encode(tokenizer, row["state"], question)
        longest = max(longest, *(len(item) for item in original))
        if question["type"] != "choice":
            continue
        reversed_question = reverse_choice(question)
        reordered, reverse_keys = pilot.encode(
            tokenizer, row["state"], reversed_question
        )
        longest = max(longest, *(len(item) for item in reordered))
        if keys != list(reversed(reverse_keys)):
            raise ValueError("Choice reverse view did not preserve keys")
        target = pilot.target_key(row)
        indices = train.selected_indices(row["id"], keys, target)
        other = aligned_indices(keys, reverse_keys, indices)
        if [keys[index] for index in indices] != [
            reverse_keys[index] for index in other
        ]:
            raise ValueError("Paired selected-key alignment failed")
        choice += 1
    if choice != 192 or longest > pilot.MAX_TOKENS:
        raise ValueError("Choice count or paired native context differs")
    return {"choice_pairs": choice, "longest_native_tokens": longest}


def paired_loss(
    model: Any,
    torch: Any,
    tokenizer: Any,
    row: dict[str, Any],
    no_id: int,
    yes_id: int,
) -> tuple[Any, int, int]:
    """Original CE for all types; key-aligned paired CE + KL for Choice."""
    from torch.nn import functional as F

    question = pilot.row_question(row)
    target_key = pilot.target_key(row)
    original_keys = [key for key, _ in pilot.option_items(question)]
    chosen = train.selected_indices(row["id"], original_keys, target_key)
    original, keys, length = train.margins(
        model,
        torch,
        tokenizer,
        row["state"],
        question,
        no_id,
        yes_id,
        candidate_indices=chosen,
    )
    target = torch.tensor([keys.index(target_key)], device="cuda")
    ce_original = F.cross_entropy(original[None, :], target)
    if question["type"] != "choice":
        return ce_original, length, 1
    reverse_question = reverse_choice(question)
    reverse_keys = [key for key, _ in pilot.option_items(reverse_question)]
    other = aligned_indices(original_keys, reverse_keys, chosen)
    reversed_scores, reversed_selected_keys, reverse_length = train.margins(
        model,
        torch,
        tokenizer,
        row["state"],
        reverse_question,
        no_id,
        yes_id,
        candidate_indices=other,
    )
    if keys != reversed_selected_keys:
        raise ValueError("Paired model margins are not key-aligned")
    ce_reverse = F.cross_entropy(reversed_scores[None, :], target)
    logp, logq = F.log_softmax(original, dim=0), F.log_softmax(reversed_scores, dim=0)
    p, q = logp.exp(), logq.exp()
    symmetric_kl = 0.5 * ((p * (logp - logq)).sum() + (q * (logq - logp)).sum())
    loss = 0.5 * (ce_original + ce_reverse) + KL_WEIGHT * symmetric_kl
    return loss, max(length, reverse_length), 2


def _load_candidate(source: Path, adapter: Path) -> tuple[Any, Any, Any, int, int]:
    receipt = json.loads((adapter / "paired_order_receipt.json").read_text())
    if (
        receipt.get("contract") != CONTRACT
        or receipt.get("source") != pilot.verify_source(source)
        or receipt.get("optimizer_updates") != train.UPDATES
        or receipt.get("adapter_sha256")
        != file_sha256(adapter / "adapter_model.safetensors")
    ):
        raise ValueError("Paired-order adapter receipt differs")
    torch, tokenizer, model, no_id, yes_id = train.load(source, None)
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, adapter).eval()
    return torch, tokenizer, model, no_id, yes_id


def preflight(
    source: Path,
    control: Path,
    train512: Path,
    train_manifest: Path,
    full_train: Path,
    select: Path,
    cal: Path,
    rights_manifest: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    rows, selected, identity = _inputs(
        source,
        control,
        train512,
        train_manifest,
        full_train,
        select,
        cal,
        rights_manifest,
    )
    torch, tokenizer, model, no_id, yes_id = train.load(source, None, training=True)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one visible ROCm GPU required")
    view = _self_check_view(rows, tokenizer)
    smoke = sorted(
        selected,
        key=lambda row: hashlib.sha256(
            f"{CONTRACT}:smoke:{row['id']}".encode()
        ).hexdigest(),
    )[:SMOKE_ROWS]
    model.eval()
    zero = []
    for row in smoke:
        answer, _ = train.predict_row(
            model,
            torch,
            tokenizer,
            row["state"],
            pilot.row_question(row),
            no_id,
            yes_id,
        )
        zero.append(answer)
    del model
    torch.cuda.empty_cache()
    _, _, source_model, _, _ = train.load(source, None, training=False)
    source_answers = []
    for row in smoke:
        answer, _ = train.predict_row(
            source_model,
            torch,
            tokenizer,
            row["state"],
            pilot.row_question(row),
            no_id,
            yes_id,
        )
        source_answers.append(answer)
    del source_model
    torch.cuda.empty_cache()
    if [
        a.get("choice", a.get("native_level", a.get("noul", 0) > 0.5)) for a in zero
    ] != [
        a.get("choice", a.get("native_level", a.get("noul", 0) > 0.5))
        for a in source_answers
    ]:
        raise ValueError("Zero-update categorical parity failed")
    _, _, model, _, _ = train.load(source, None, training=True)
    model.train()
    row = next(row for row in rows if row["task_type"] == "choice")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss, _, views = paired_loss(model, torch, tokenizer, row, no_id, yes_id)
    if not torch.isfinite(loss) or views != 2:
        raise ValueError("Nonfinite paired preflight loss")
    loss.backward()
    trainable_grad = [
        p.grad for p in model.parameters() if p.requires_grad and p.grad is not None
    ]
    if not trainable_grad or not all(torch.isfinite(g).all() for g in trainable_grad):
        raise ValueError("Missing or nonfinite LoRA gradients")
    if not any(float(g.abs().sum()) > 0 for g in trainable_grad):
        raise ValueError("Zero LoRA gradients")
    if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
        raise ValueError("Frozen base accumulated gradients")
    result = {
        "contract": CONTRACT,
        "status": "pass",
        "source": identity,
        "control_sha256": CONTROL_SHA,
        "train512_sha256": TRAIN512_SHA,
        "select_sha256": pilot.SELECT_SHA,
        "cal_sha256": CAL_SHA,
        "rights_manifest_sha256": RIGHTS_SHA,
        "source_disjoint": True,
        "near_overlap": "source_manifest_zero; approximate 8-band SimHash audit",
        "smoke_rows": SMOKE_ROWS,
        "zero_update_categorical_matches": SMOKE_ROWS,
        "finite_nonzero_lora_gradient": True,
        "native_view": view,
        "treatment_code_sha256": file_sha256(Path(__file__)),
        "pilot_code_sha256": file_sha256(Path(pilot.__file__)),
        "control_trainer_code_sha256": file_sha256(Path(train.__file__)),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "gpu_count_visible": torch.cuda.device_count(),
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def run_train(
    source: Path,
    control: Path,
    train512: Path,
    train_manifest: Path,
    full_train: Path,
    select: Path,
    cal: Path,
    rights_manifest: Path,
    preflight_receipt: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    receipt = json.loads(preflight_receipt.read_text())
    if (
        receipt.get("contract") != CONTRACT
        or receipt.get("status") != "pass"
        or receipt.get("treatment_code_sha256") != file_sha256(Path(__file__))
    ):
        raise ValueError("Prospective preflight is missing or source changed")
    rows, _, identity = _inputs(
        source,
        control,
        train512,
        train_manifest,
        full_train,
        select,
        cal,
        rights_manifest,
    )
    torch, tokenizer, model, no_id, yes_id = train.load(source, None, training=True)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one visible ROCm GPU required")
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=train.LORA_LR,
        weight_decay=0.01,
    )
    ordered = sorted(rows, key=lambda row: row["id"])
    random.Random(train.SEED).shuffle(ordered)
    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()
    max_tokens = 0
    sum_loss = 0.0
    views = Counter()
    for index, row in enumerate(ordered):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss, length, view_count = paired_loss(
                model, torch, tokenizer, row, no_id, yes_id
            )
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite paired training loss")
        (loss / train.ACCUMULATION).backward()
        max_tokens = max(max_tokens, length)
        sum_loss += float(loss.detach())
        views[view_count] += 1
        if (index + 1) % train.ACCUMULATION == 0:
            norm = torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad), 1.0
            )
            if not torch.isfinite(norm):
                raise ValueError("Nonfinite paired training gradient norm")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step = (index + 1) // train.ACCUMULATION
            if step % 8 == 0:
                print(json.dumps({"step": step, "max_tokens": max_tokens}), flush=True)
    if len(ordered) != 512 or step != train.UPDATES or views != {1: 320, 2: 192}:
        raise ValueError("Treatment updates or views differ")
    output.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(output)
    result = {
        "contract": CONTRACT,
        "research_only": True,
        "release_qualified": False,
        "source": identity,
        "control_sha256": CONTROL_SHA,
        "source_train_sha256": TRAIN512_SHA,
        "optimizer_updates": step,
        "accumulation": train.ACCUMULATION,
        "lora_rank": train.LORA_RANK,
        "lora_alpha": train.LORA_ALPHA,
        "lora_lr": train.LORA_LR,
        "negative_cap": train.NEGATIVE_CAP,
        "paired_kl_weight": KL_WEIGHT,
        "paired_choice_rows": views[2],
        "single_view_noul_score_rows": views[1],
        "max_native_tokens": max_tokens,
        "mean_loss": sum_loss / len(ordered),
        "duration_sec": time.monotonic() - started,
        "adapter_sha256": file_sha256(output / "adapter_model.safetensors"),
        "preflight_sha256": file_sha256(preflight_receipt),
        "treatment_code_sha256": file_sha256(Path(__file__)),
        "pilot_code_sha256": file_sha256(Path(pilot.__file__)),
        "control_trainer_code_sha256": file_sha256(Path(train.__file__)),
    }
    (output / "paired_order_receipt.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def predict(
    source: Path,
    adapter: Path,
    prompts: Path,
    output: Path,
    *,
    partition: str,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    if partition == "select":
        if file_sha256(prompts) != pilot.SELECT_SHA:
            raise ValueError("SELECT prompts differ")
        rows = load_partition(prompts, "select")
    else:
        rows = load_prompts(prompts)
    torch, tokenizer, model, no_id, yes_id = _load_candidate(source, adapter)
    counts = Counter()
    with output.open("x", encoding="utf-8") as stream:
        for index, row in enumerate(rows):
            if partition == "select":
                answer, _ = train.predict_row(
                    model,
                    torch,
                    tokenizer,
                    row["state"],
                    pilot.row_question(row),
                    no_id,
                    yes_id,
                )
                record = {"id": row["id"], "answer": answer}
                counts["invalid" if "error" in answer else "valid"] += 1
            else:
                answers = {}
                for name, question in row["questions"].items():
                    answer, _ = train.predict_row(
                        model,
                        torch,
                        tokenizer,
                        row["state"],
                        question,
                        no_id,
                        yes_id,
                    )
                    answers[name] = answer
                    counts["invalid" if "error" in answer else "valid"] += 1
                record = {
                    "id": row["id"],
                    "answers": answers,
                    "source_input_sha256": digest(
                        {"state": row["state"], "questions": row["questions"]}
                    ),
                    "backend": "qwen3-reranker-06b",
                    "model_id": pilot.MODEL_ID,
                    "model_revision": pilot.MODEL_REVISION,
                    "adapter_version": pilot.ADAPTER,
                    "adapter_sha256": file_sha256(
                        adapter / "adapter_model.safetensors"
                    ),
                }
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            if (index + 1) % 100 == 0:
                print(json.dumps({"completed": index + 1}), flush=True)
    return {
        "contract": CONTRACT,
        "partition": partition,
        "prompts_sha256": file_sha256(prompts),
        "prediction_sha256": file_sha256(output),
        "counts": dict(counts),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "train", "select", "panel"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--control", type=Path)
    parser.add_argument("--train512", type=Path)
    parser.add_argument("--train-manifest", type=Path)
    parser.add_argument("--full-train", type=Path)
    parser.add_argument("--select", type=Path)
    parser.add_argument("--cal", type=Path)
    parser.add_argument("--rights-manifest", type=Path)
    parser.add_argument("--preflight-receipt", type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--prompts", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    inputs = dict(
        source=args.source,
        control=args.control,
        train512=args.train512,
        train_manifest=args.train_manifest,
        full_train=args.full_train,
        select=args.select,
        cal=args.cal,
        rights_manifest=args.rights_manifest,
    )
    if args.action == "preflight":
        result = preflight(**inputs, output=args.output)
    elif args.action == "train":
        result = run_train(
            **inputs, preflight_receipt=args.preflight_receipt, output=args.output
        )
    else:
        result = predict(
            args.source,
            args.adapter,
            args.select if args.action == "select" else args.prompts,
            args.output,
            partition=args.action,
        )
    args.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
