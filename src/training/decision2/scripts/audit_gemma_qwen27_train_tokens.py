"""CPU-only, aggregate-only TRAIN token admission audit for two official models.

This script never reads SELECT, CAL or benchmark data. Its JSON output includes
counts and hashes of row rosters, never row text, labels or identifiers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import median

import torch
from training.model.data import canonical, file_sha256, load_partition
from training.model.decision_model import CandidateHead, collate, encode
from training.model.gemma4 import encode_gemma
from training.model.loss import per_example_loss
from transformers import AutoTokenizer


def roster_hash(ids: list[str]) -> str:
    return hashlib.sha256(canonical(ids).encode("utf-8")).hexdigest()


def describe(lengths: list[int]) -> dict[str, int]:
    if not lengths:
        raise ValueError("Cannot describe an empty token roster")
    ordered = sorted(lengths)
    return {
        "count": len(ordered),
        "sum": sum(ordered),
        "min": ordered[0],
        "median": int(median(ordered)),
        "p95": ordered[(95 * (len(ordered) - 1)) // 100],
        "max": ordered[-1],
    }


def audit(
    rows: list[dict], qwen_tokenizer: object, gemma_tokenizer: object, max_length: int
) -> dict:
    if max_length < 2:
        raise ValueError("max_length must exceed 1")
    lengths: dict[str, list[int]] = {"qwen": [], "gemma": []}
    admitted: dict[str, list[str]] = {"qwen": [], "gemma": [], "common": []}
    admitted_types: dict[str, Counter] = {
        "qwen": Counter(),
        "gemma": Counter(),
        "common": Counter(),
    }
    admitted_languages: dict[str, Counter] = {
        "qwen": Counter(),
        "gemma": Counter(),
        "common": Counter(),
    }
    admitted_lengths: dict[str, list[int]] = {"qwen": [], "gemma": []}
    common_lengths: dict[str, list[int]] = {"qwen": [], "gemma": []}
    common_lengths_by_type: dict[str, dict[str, list[int]]] = {
        name: {kind: [] for kind in ("choice", "noul", "score")}
        for name in ("qwen", "gemma")
    }
    types = Counter()
    languages = Counter()
    for row in rows:
        types[row["task_type"]] += 1
        languages[row["language"]] += 1
        qwen_ids = encode(row, qwen_tokenizer, 10**8)["ids"]
        gemma_ids = encode_gemma(row, gemma_tokenizer, 10**8)["ids"]
        measured = {"qwen": len(qwen_ids), "gemma": len(gemma_ids)}
        for name in ("qwen", "gemma"):
            lengths[name].append(measured[name])
            if measured[name] <= max_length:
                admitted[name].append(row["id"])
                admitted_lengths[name].append(measured[name])
                admitted_types[name][row["task_type"]] += 1
                admitted_languages[name][row["language"]] += 1
        if all(measured[name] <= max_length for name in ("qwen", "gemma")):
            admitted["common"].append(row["id"])
            admitted_types["common"][row["task_type"]] += 1
            admitted_languages["common"][row["language"]] += 1
            for name in ("qwen", "gemma"):
                common_lengths[name].append(measured[name])
                common_lengths_by_type[name][row["task_type"]].append(measured[name])
    if sum(types.values()) != len(rows):
        raise AssertionError("Type accounting failed")
    return {
        "max_length": max_length,
        "train_rows": len(rows),
        "all_train_ids_sha256": roster_hash([row["id"] for row in rows]),
        "all_type_counts": dict(sorted(types.items())),
        "all_language_counts": dict(sorted(languages.items())),
        "models": {
            name: {
                "all_token_lengths": describe(lengths[name]),
                "admitted_token_lengths": describe(admitted_lengths[name]),
                "admitted_count": len(admitted[name]),
                "admitted_ids_sha256": roster_hash(admitted[name]),
                "admitted_type_counts": dict(sorted(admitted_types[name].items())),
                "admitted_language_counts": dict(
                    sorted(admitted_languages[name].items())
                ),
                "rejected_over_length_count": len(rows) - len(admitted[name]),
            }
            for name in ("qwen", "gemma")
        },
        "common": {
            "admitted_count": len(admitted["common"]),
            "admitted_ids_sha256": roster_hash(admitted["common"]),
            "admitted_type_counts": dict(sorted(admitted_types["common"].items())),
            "admitted_language_counts": dict(
                sorted(admitted_languages["common"].items())
            ),
            "token_lengths": {
                name: describe(common_lengths[name]) for name in ("qwen", "gemma")
            },
            "token_lengths_by_type": {
                name: {
                    kind: describe(common_lengths_by_type[name][kind])
                    for kind in ("choice", "noul", "score")
                }
                for name in ("qwen", "gemma")
            },
        },
    }


def smoke_multitype_head_loss(
    rows: list[dict], gemma_tokenizer: object, max_length: int
) -> dict[str, dict[str, int | bool | list[str]]]:
    """Prove the shared head/loss path on one real TRAIN row per task type.

    Random hidden vectors replace the frozen text model: this is a CPU schema
    and backward check, not proof of long-input Gemma backward viability.
    """
    result: dict[str, dict[str, int | bool | list[str]]] = {}
    torch.manual_seed(20260927)
    for row in rows:
        task_type = row["task_type"]
        if task_type in result:
            continue
        encoded = encode_gemma(row, gemma_tokenizer, 10**8)
        if len(encoded["ids"]) > max_length:
            continue
        batch = collate([encoded], gemma_tokenizer.pad_token_id)
        head = CandidateHead(hidden_size=2816, head_dim=256)
        candidates = torch.randn(1, len(encoded["keys"]), 2816)
        query = torch.randn(1, 2816)
        for objective in ("ce", "ce_brier"):
            head.zero_grad(set_to_none=True)
            logits = head(candidates, query)
            loss = per_example_loss(
                logits,
                batch["labels"],
                batch["candidate_mask"],
                objective=objective,
                brier_weight=0.5,
            )["total"].mean()
            loss.backward()
            if not torch.isfinite(loss) or not all(
                parameter.grad is not None and torch.isfinite(parameter.grad).all()
                for parameter in head.parameters()
            ):
                raise RuntimeError(
                    f"CPU Decision head backward failed for {task_type}/{objective}"
                )
        result[task_type] = {
            "candidate_count": len(encoded["keys"]),
            "token_count": len(encoded["ids"]),
            "objectives": ["ce", "ce_brier"],
            "finite_backward": True,
        }
        if len(result) == 3:
            break
    if set(result) != {"choice", "noul", "score"}:
        raise RuntimeError("TRAIN cannot supply all three admitted task types")
    return dict(sorted(result.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--qwen-tokenizer", type=Path, required=True)
    parser.add_argument("--gemma-tokenizer", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = load_partition(args.train, "train")
    qwen = AutoTokenizer.from_pretrained(args.qwen_tokenizer, local_files_only=True)
    gemma = AutoTokenizer.from_pretrained(args.gemma_tokenizer, local_files_only=True)
    report = audit(rows, qwen, gemma, args.max_length)
    report["cpu_multitype_head_loss_smoke"] = smoke_multitype_head_loss(
        rows, gemma, args.max_length
    )
    report["train_file_sha256"] = file_sha256(args.train)
    report["qwen_tokenizer_sha256"] = file_sha256(
        args.qwen_tokenizer / "tokenizer.json"
    )
    report["gemma_tokenizer_sha256"] = file_sha256(
        args.gemma_tokenizer / "tokenizer.json"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
