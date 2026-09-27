"""Complete rights-clean TRAIN Choice/Noul source screen for pinned AutoJev.

This invokes only the released typed decision readout, never a chat API or an
evaluation partition. Its optional probability artifact is private research.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from inference.autojev27 import (
    MODEL_ID,
    MODEL_REVISION,
    SOURCE_REVISION,
    _admission_reason,
    _native_answer,
    verify_release,
)
from research.eikos_teacher_train_pilot import probabilities, question, write_once
from training.model.data import canonical, file_sha256, load_partition

TRAIN_SHA256 = "61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755"
RIGHTS_MANIFEST_SHA256 = (
    "61aa883052759830c4ecf897b36c1062ad816c935a12db824c80abd1f80e9ee8"
)
EXPECTED = {"choice": 3908, "noul": 3031}
CONTEXT_LIMIT = 8192
ALLOWED_SOURCES = {
    "google_goemotions_official_train",
    "legacy:stage4-general-composition-v2",
    "legacy:stage3_replay",
    "legacy:cosmos_qa",
    "legacy:snli",
    "decision2_programmatic_original_v1",
    "css_flute_official_train",
    "decision2_targeted_programmatic_v1",
    "legacy:squad2_answerability",
}
Decision = Callable[[Any, dict[str, Any]], tuple[dict[str, Any], int]]


def train_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep every Choice/Noul record in file order, with no label filter."""
    selected = [row for row in rows if row["task_type"] in EXPECTED]
    counts = collections.Counter(row["task_type"] for row in selected)
    if dict(counts) != EXPECTED:
        raise ValueError("Frozen TRAIN Choice/Noul counts differ")
    if len({row["id"] for row in selected}) != len(selected) or len(
        {row["input_sha256"] for row in selected}
    ) != len(selected):
        raise ValueError("Duplicate TRAIN identity")
    for row in selected:
        options = row["options"]
        keys = [option["key"] for option in options]
        if (
            row["source"] not in ALLOWED_SOURCES
            or not 0 <= row["label"] < len(options)
            or len(keys) != len(set(keys))
            or len(keys) > 255
            or (row["task_type"] == "choice" and len(keys) < 1)
            or (row["task_type"] == "noul" and set(keys) != {"false", "true"})
        ):
            raise ValueError("TRAIN source or native option contract differs")
    return selected


def roster_sha256(rows: list[dict[str, Any]]) -> str:
    identities = [
        {
            "id": row["id"],
            "input_sha256": row["input_sha256"],
            "group_id": row["group_id"],
            "task_type": row["task_type"],
            "source": row["source"],
            "family": row["family"],
            "language": row["language"],
            "option_count": len(row["options"]),
        }
        for row in rows
    ]
    return hashlib.sha256(canonical(identities).encode()).hexdigest()


def _counts(
    rows: list[dict[str, Any]], key: Callable[[dict[str, Any]], str]
) -> dict[str, int]:
    return dict(sorted(collections.Counter(key(row) for row in rows).items()))


def gold_bucket(row: dict[str, Any]) -> str:
    if row["task_type"] == "noul":
        return row["options"][row["label"]]["key"]
    if row["label"] == 0:
        return "first"
    if row["label"] == len(row["options"]) - 1:
        return "last"
    return "interior"


def profile(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "groups": len({row["group_id"] for row in rows}),
        "roster_sha256": roster_sha256(rows),
        "by_type": _counts(rows, lambda row: row["task_type"]),
        "by_source": _counts(rows, lambda row: row["source"]),
        "by_family": _counts(rows, lambda row: row["family"]),
        "by_language": _counts(rows, lambda row: row["language"]),
        "by_type_option_count": _counts(
            rows, lambda row: f"{row['task_type']}:{len(row['options'])}"
        ),
        "by_type_gold_bucket": _counts(
            rows, lambda row: f"{row['task_type']}:{gold_bucket(row)}"
        ),
    }


def token_profile(
    rows: list[dict[str, Any]], model_path: Path, source_path: Path
) -> dict[str, Any]:
    """CPU-only exact published processor and question renderer, no weights."""
    sys.path.insert(0, str(source_path / "src"))
    from autojev.model import decision_messages
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True)
    codes = json.loads((model_path / "decision_config.json").read_text())["codes"]
    lengths = []
    for row in rows:
        prompt = processor.apply_chat_template(
            decision_messages(
                {"state": row["state"], "question": question(row)}, codes
            ),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        encoded = processor(
            text=[prompt], images=None, padding=True, return_tensors="pt"
        )
        lengths.append(int(encoded["attention_mask"].sum()))
    ordered = sorted(lengths)
    token_identity = [
        {"id": row["id"], "tokens": length} for row, length in zip(rows, lengths)
    ]
    return {
        "min": ordered[0],
        "median": ordered[len(ordered) // 2],
        "p95": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "max": ordered[-1],
        "over_8192": sum(length > CONTEXT_LIMIT for length in lengths),
        "token_roster_sha256": hashlib.sha256(
            canonical(token_identity).encode()
        ).hexdigest(),
    }


def _fresh_stats() -> dict[str, float | int]:
    return {
        "n": 0,
        "valid": 0,
        "invalid": 0,
        "context_overflow": 0,
        "candidate_limit": 0,
        "ties": 0,
        "correct": 0,
        "gold_probability_sum": 0.0,
        "brier_sum": 0.0,
    }


def aggregate(
    rows: list[dict[str, Any]], decide: Decision
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    facets: dict[str, dict[str, dict[str, float | int]]] = {
        name: {}
        for name in (
            "by_type",
            "by_source",
            "by_family",
            "by_language",
            "by_type_option_count",
            "by_type_gold_bucket",
        )
    }
    distributions = []
    overall = _fresh_stats()
    for row in rows:
        gold = row["options"][row["label"]]["key"]
        categories = {
            "by_type": row["task_type"],
            "by_source": row["source"],
            "by_family": row["family"],
            "by_language": row["language"],
            "by_type_option_count": f"{row['task_type']}:{len(row['options'])}",
            "by_type_gold_bucket": f"{row['task_type']}:{gold_bucket(row)}",
        }
        bins = [overall] + [
            facets[name].setdefault(key, _fresh_stats())
            for name, key in categories.items()
        ]
        for stats in bins:
            stats["n"] += 1
        try:
            answer, _ = decide(row["state"], question(row))
            probs = probabilities(row, answer)
        except ValueError as error:
            reason = _admission_reason(error)
            if reason is None:
                raise
            for stats in bins:
                stats["invalid"] += 1
                stats[reason] += 1
            continue
        maximum = max(probs.values())
        winners = [key for key, value in probs.items() if abs(value - maximum) <= 1e-8]
        tie = int(len(winners) != 1)
        correct = int(not tie and winners[0] == gold)
        brier = sum((value - int(key == gold)) ** 2 for key, value in probs.items()) / 2
        for stats in bins:
            stats["valid"] += 1
            stats["ties"] += tie
            stats["correct"] += correct
            stats["gold_probability_sum"] += probs[gold]
            stats["brier_sum"] += brier
        distributions.append(
            {
                "id": row["id"],
                "input_sha256": row["input_sha256"],
                "group_id": row["group_id"],
                "task_type": row["task_type"],
                "source": row["source"],
                "family": row["family"],
                "language": row["language"],
                "option_count": len(row["options"]),
                "probabilities": probs,
            }
        )
    return {"overall": overall, **facets}, distributions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--train-sha256", required=True)
    parser.add_argument("--rights-manifest", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--source-path", required=True, type=Path)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--roster-sha256")
    parser.add_argument("--aggregate", type=Path)
    parser.add_argument("--distributions", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if (
        args.train_sha256 != TRAIN_SHA256
        or file_sha256(args.train) != TRAIN_SHA256
        or file_sha256(args.rights_manifest) != RIGHTS_MANIFEST_SHA256
    ):
        raise ValueError("Frozen TRAIN or rights manifest bytes differ")
    rows = train_rows(load_partition(args.train, "train"))
    preflight = profile(rows)
    if args.dry_run:
        preflight["native_token_profile"] = token_profile(
            rows, args.model_path, args.source_path
        )
        print(json.dumps(preflight, sort_keys=True))
        return
    if (
        args.revision != MODEL_REVISION
        or args.roster_sha256 != preflight["roster_sha256"]
        or args.aggregate is None
        or args.distributions is None
        or args.aggregate == args.distributions
        or args.aggregate.exists()
        or args.distributions.exists()
    ):
        raise ValueError("Frozen native source, roster and private outputs required")
    release = verify_release(args.model_path, args.source_path, args.revision)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path.insert(0, str(args.source_path / "src"))
    from autojev.model import DecisionModel

    model = DecisionModel(checkpoint=args.model_path, device="cuda:0", train=False)
    if sum(p.numel() for p in model.parameters()) != release["loaded_parameters"]:
        raise ValueError("Loaded parameters differ from frozen AutoJev package")
    stats, vectors = aggregate(
        rows, lambda state, item: _native_answer(model, state, item)
    )
    artifact_sha = None
    if stats["overall"]["valid"] == len(rows):
        artifact = {
            "schema": "decision2-autojev-choice-noul-train-distributions/1",
            "source": f"{MODEL_ID}@{MODEL_REVISION}",
            "source_revision": SOURCE_REVISION,
            "train_sha256": TRAIN_SHA256,
            "rights_manifest_sha256": RIGHTS_MANIFEST_SHA256,
            "roster_sha256": preflight["roster_sha256"],
            "script_sha256": file_sha256(__file__),
            **release,
            "rows": vectors,
        }
        write_once(args.distributions, artifact)
        artifact_sha = file_sha256(args.distributions)
    receipt = {
        "schema": "decision2-autojev-choice-noul-train-audit/1",
        "source": f"{MODEL_ID}@{MODEL_REVISION}",
        "source_revision": SOURCE_REVISION,
        "train_sha256": TRAIN_SHA256,
        "rights_manifest_sha256": RIGHTS_MANIFEST_SHA256,
        "roster_sha256": preflight["roster_sha256"],
        "script_sha256": file_sha256(__file__),
        **release,
        "profile": preflight,
        "metrics": stats,
        "private_distribution_sha256": artifact_sha,
    }
    write_once(args.aggregate, receipt)
    print(
        json.dumps({k: v for k, v in receipt.items() if k != "metrics"}, sort_keys=True)
    )


if __name__ == "__main__":
    main()
