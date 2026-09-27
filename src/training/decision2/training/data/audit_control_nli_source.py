"""Aggregate-only screen of the publisher's ConTRoL TRAIN for 4B research.

This does not create a student corpus or read the publisher dev/test labels.
Raw text, IDs, protected prompts and matching rosters stay private. An output
summary is source feasibility evidence, never a score or training admission.
"""

from __future__ import annotations

import argparse
import collections
import json
import subprocess
from pathlib import Path
from typing import Any

from training.data.audit_nli_evidence_score import (
    Pair,
    normalize,
    overlap_screen,
    read_protected,
    sha_file,
    shortcut_screen,
    source_summary,
)

PUBLISHER_COMMIT = "d7acc335bef6c716f2830e1413d0d90c133ad6e9"
TRAIN_SHA256 = "e51b63fa1da381a27fb5244e6f3c8f317eed51921e20fc05334023db3e2e834f"
LABELS = {"c": "contradiction", "n": "neutral", "e": "entailment"}


def read_train(repo: Path) -> tuple[list[Pair], dict[str, Any]]:
    if (
        subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
        != PUBLISHER_COMMIT
    ):
        raise ValueError("ConTRoL publisher revision differs")
    if subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain"], text=True
    ).strip():
        raise ValueError("ConTRoL source checkout has changes")
    readme = repo / "readme.md"
    train_file = repo / "data/train.jsonl"
    if sha_file(train_file) != TRAIN_SHA256:
        raise ValueError("ConTRoL publisher TRAIN bytes differ")
    if "Attribution-NonCommercial-ShareAlike 4.0" not in readme.read_text():
        raise ValueError("ConTRoL publisher license statement differs")

    source_ids: set[str] = set()
    pairs: list[Pair] = []
    by_group: dict[str, set[str]] = collections.defaultdict(set)
    with train_file.open(encoding="utf-8") as stream:
        for position, line in enumerate(stream):
            row = json.loads(line)
            if set(row) != {"uid", "premise", "hypothesis", "label"}:
                raise ValueError("ConTRoL TRAIN schema differs")
            if row["label"] not in LABELS:
                raise ValueError("ConTRoL TRAIN NLI label differs")
            if not isinstance(row["premise"], str) or not row["premise"].strip():
                raise ValueError("ConTRoL premise missing")
            if not isinstance(row["hypothesis"], str) or not row["hypothesis"].strip():
                raise ValueError("ConTRoL hypothesis missing")
            source_id = str(row["uid"])
            if source_id in source_ids:
                raise ValueError("ConTRoL publisher IDs repeat")
            source_ids.add(source_id)
            group = normalize(row["premise"])
            by_group[group].add(source_id)
            pairs.append(
                Pair(
                    premise=row["premise"],
                    hypothesis=row["hypothesis"],
                    label=LABELS[row["label"]],
                    group=group,
                    genre="ConTRoL publisher TRAIN",
                    position=position,
                )
            )
    return pairs, {
        "publisher_revision": PUBLISHER_COMMIT,
        "publisher_train_sha256": TRAIN_SHA256,
        "publisher_readme_sha256": sha_file(readme),
        "license": "CC BY-NC-SA 4.0; noncommercial research, attribution and adapted-data terms",
        "source_ids": len(source_ids),
        "premise_groups": len(by_group),
    }


def audit(
    repo: Path,
    *,
    tokenizer_path: Path | None,
    protected_inventory: Path | None,
) -> dict[str, Any]:
    pairs, provenance = read_train(repo)
    tokenizer = None
    tokenizer_files = None
    if tokenizer_path:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, local_files_only=True, trust_remote_code=False
        )
        tokenizer_files = {
            name: sha_file(tokenizer_path / name)
            for name in ("tokenizer.json", "tokenizer_config.json")
        }
    summary = source_summary(pairs, tokenizer)
    shortcut = shortcut_screen(pairs, cap_train=6_000, cap_test=2_000)
    by_group = collections.Counter(row.group for row in pairs)
    answer = {
        "status": "source_screen_only_no_train_admission",
        "source_role": "publisher_train_only",
        "publisher_dev_test_labels_opened": False,
        "provenance": provenance,
        "summary": summary,
        "premise_group_sizes": {
            "singletons": sum(n == 1 for n in by_group.values()),
            "multiple_rows": sum(n > 1 for n in by_group.values()),
        },
        "shortcut": shortcut,
        "tokenizer_files": tokenizer_files,
    }
    if protected_inventory:
        protected, audit_receipt = read_protected(protected_inventory, [])
        answer["protected_role_inventory"] = audit_receipt
        answer["protected_overlap"] = overlap_screen(pairs, protected)
    else:
        answer["protected_overlap"] = "NOT_RUN; source cannot enter TRAIN"
    return answer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publisher-repo", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--protected-inventory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit(
        args.publisher_repo,
        tokenizer_path=args.tokenizer_path,
        protected_inventory=args.protected_inventory,
    )
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
