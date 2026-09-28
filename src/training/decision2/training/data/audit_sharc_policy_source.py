"""Aggregate-only, TRAIN-only ShARC policy-pair feasibility inventory.

This inventory does not create/admit training data or inspect any model score.
Pass the publisher archive as an input path on an authorized remote CPU.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import statistics
import zipfile
from pathlib import Path
from typing import Any


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _visible(row: dict[str, Any]) -> str:
    history = row.get("history")
    if not isinstance(history, list):
        raise ValueError("history is not a list")
    parts = [row["snippet"], row["question"], row.get("scenario", "")]
    if any(not isinstance(part, str) for part in parts):
        raise ValueError("visible input has a non-string field")
    for item in history:
        if not isinstance(item, dict):
            raise ValueError("history entry is not an object")
        question, answer = item.get("follow_up_question"), item.get("follow_up_answer")
        if not isinstance(question, str) or not isinstance(answer, str):
            raise ValueError("history entry is malformed")
        parts.extend((question, answer))
    return "\n".join(parts)


def inventory(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_tree: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    malformed = collections.Counter()
    labels = collections.Counter()
    seen_ids: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            malformed["non_object"] += 1
            continue
        required = ("utterance_id", "tree_id", "source_url", "snippet", "question")
        if any(not isinstance(row.get(key), str) or not row[key] for key in required):
            malformed["missing_identity_or_rule"] += 1
            continue
        try:
            _visible(row)
        except ValueError:
            malformed["invalid_visible_input"] += 1
            continue
        item_id = row["utterance_id"]
        if item_id in seen_ids:
            malformed["repeated_utterance_id"] += 1
            continue
        seen_ids.add(item_id)
        label = row.get("answer")
        labels[label if isinstance(label, str) else "<non-string>"] += 1
        by_tree[row["tree_id"]].append(row)

    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    inconsistent_trees = 0
    for members in by_tree.values():
        rule_keys = {
            (row["source_url"], row["snippet"], row["question"]) for row in members
        }
        if len(rule_keys) != 1:
            inconsistent_trees += 1
            continue
        yes = [row for row in members if row.get("answer") == "Yes"]
        no = [row for row in members if row.get("answer") == "No"]
        candidates = [
            (left, right)
            for left in yes
            for right in no
            if _visible(left) != _visible(right)
        ]
        if candidates:
            pair = min(
                candidates,
                key=lambda value: _sha256(
                    (
                        value[0]["utterance_id"] + "\0" + value[1]["utterance_id"]
                    ).encode()
                ),
            )
            pairs.append(pair)
    lengths = sorted(len(_visible(row)) for pair in pairs for row in pair)
    urls = {pair[0]["source_url"] for pair in pairs}
    snippets = {_sha256(pair[0]["snippet"].encode()) for pair in pairs}
    feasibility = len(pairs) >= 100 and len(urls) >= 50 and len(snippets) >= 100
    return {
        "schema": "decision2-sharc-source-inventory/1",
        "train_rows": len(rows),
        "valid_unique_rows": len(seen_ids),
        "malformed": dict(sorted(malformed.items())),
        "answer_classes": dict(sorted(labels.items())),
        "distinct_trees": len(by_tree),
        "inconsistent_rule_trees": inconsistent_trees,
        "candidate_choice_tree_pairs": len(pairs),
        "pair_source_urls": len(urls),
        "pair_exact_snippets": len(snippets),
        "visible_chars_min": lengths[0] if lengths else None,
        "visible_chars_median": statistics.median(lengths) if lengths else None,
        "visible_chars_max": lengths[-1] if lengths else None,
        "feasibility_pass": feasibility,
        "training_admitted": False,
        "model_score": None,
    }


def read_train(archive: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    archive_bytes = archive.read_bytes()
    with zipfile.ZipFile(archive) as package:
        members = sorted(name for name in package.namelist() if not name.endswith("/"))
        train_names = [
            name for name in members if name.rsplit("/", 1)[-1] == "sharc_train.json"
        ]
        if len(train_names) != 1:
            raise ValueError("Expected exactly one publisher TRAIN JSON member")
        train_bytes = package.read(train_names[0])
    rows = json.loads(train_bytes)
    if not isinstance(rows, list):
        raise ValueError("Publisher TRAIN JSON is not an array")
    return rows, {
        "archive_sha256": _sha256(archive_bytes),
        "train_member": train_names[0],
        "train_member_sha256": _sha256(train_bytes),
        "archive_members": members,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows, identity = read_train(args.archive)
    report = {**identity, **inventory(rows)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(
        json.dumps(
            {
                "report_sha256": _sha256(args.output.read_bytes()),
                "feasibility_pass": report["feasibility_pass"],
                "candidate_choice_tree_pairs": report["candidate_choice_tree_pairs"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
