"""Narrow normalized-input ShARC v2 feasibility audit, with no raw output."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from .audit_sharc_policy_source import _visible, read_train

ARCHIVE_SHA256 = "72dca3f4f3ba73b1d796b40e952a80d53cd2011ef90b2168b8bcaa818f5edd1e"


def normalized(text: str) -> str:
    result = unicodedata.normalize("NFKC", text).casefold()
    result = re.sub(r"\s+", " ", result).strip()
    return result.rstrip(".?! ")


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
        labels[label if label in ("Yes", "No") else "Other"] += 1
        by_tree[row["tree_id"]].append(row)

    field_variation = collections.Counter()
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for members in by_tree.values():
        for field in ("source_url", "snippet", "question"):
            field_variation[field + "_exact_single"] += (
                len({row[field] for row in members}) == 1
            )
            field_variation[field + "_normalized_single"] += (
                len(
                    {
                        (
                            row[field].strip()
                            if field == "source_url"
                            else normalized(row[field])
                        )
                        for row in members
                    }
                )
                == 1
            )
        by_rule: dict[tuple[str, str, str], list[dict[str, Any]]] = (
            collections.defaultdict(list)
        )
        for row in members:
            key = (
                row["source_url"].strip(),
                normalized(row["snippet"]),
                normalized(row["question"]),
            )
            by_rule[key].append(row)
        candidate_pairs = []
        for rule_rows in by_rule.values():
            yes = [row for row in rule_rows if row.get("answer") == "Yes"]
            no = [row for row in rule_rows if row.get("answer") == "No"]
            candidate_pairs.extend(
                (left, right)
                for left in yes
                for right in no
                if _visible(left) != _visible(right)
            )
        if candidate_pairs:
            pairs.append(
                min(
                    candidate_pairs,
                    key=lambda value: hashlib.sha256(
                        (
                            value[0]["utterance_id"] + "\0" + value[1]["utterance_id"]
                        ).encode()
                    ).hexdigest(),
                )
            )
    urls = {row["source_url"].strip() for pair in pairs for row in pair}
    snippets = {normalized(pair[0]["snippet"]) for pair in pairs}
    return {
        "schema": "decision2-sharc-source-inventory/2",
        "train_rows": len(rows),
        "valid_unique_rows": len(seen_ids),
        "malformed": dict(sorted(malformed.items())),
        "answer_classes": dict(sorted(labels.items())),
        "distinct_trees": len(by_tree),
        "field_variation_tree_counts": dict(sorted(field_variation.items())),
        "candidate_choice_tree_pairs": len(pairs),
        "pair_source_urls": len(urls),
        "pair_normalized_snippets": len(snippets),
        "feasibility_pass": len(pairs) >= 100
        and len(urls) >= 50
        and len(snippets) >= 100,
        "training_admitted": False,
        "model_score": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows, identity = read_train(args.archive)
    if identity["archive_sha256"] != ARCHIVE_SHA256:
        raise ValueError("Publisher archive SHA-256 changed")
    report = {**identity, **inventory(rows)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(
        json.dumps(
            {
                "feasibility_pass": report["feasibility_pass"],
                "candidate_choice_tree_pairs": report["candidate_choice_tree_pairs"],
                "report_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
