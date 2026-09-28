"""Build a private answer-blind ShARC TRAIN source-review packet.

This creates no training rows. It never opens publisher DEV/TEST members and
never writes publisher answer/evidence to either output file.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import statistics
import zipfile
from pathlib import Path
from typing import Any

from .audit_sharc_policy_source import _visible, read_train
from .audit_sharc_policy_source_v2 import ARCHIVE_SHA256, normalized

SALT = "decision2-sharc-blind-v3"
TRAIN_SHA256 = "d37d349758a69644fbf49827b1a0893f6099752b0ccc9c7e29bd315e5228429c"
NEGATIVE_MEMBERS = {
    "question": (
        "sharc_negative_question_utterance_ids.txt",
        "4e185fb8fc82bc750597812b0458e0dc4288b37f54966ae5c00af9b27553b5e1",
    ),
    "scenario": (
        "sharc_negative_scenario_utterance_ids.txt",
        "177b73465fbd241dd233a676e989cef97d5971d96bc1d5df3cc0cdda815e5220",
    ),
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def rank(*parts: str) -> str:
    return sha256("\0".join((SALT, *parts)).encode("utf-8"))


def _state(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "scenario": row["scenario"],
        "history": [
            {
                "follow_up_question": item["follow_up_question"],
                "follow_up_answer": item["follow_up_answer"],
            }
            for item in row["history"]
        ],
    }


def _state_key(row: dict[str, Any]) -> str:
    return json.dumps(_state(row), ensure_ascii=False, sort_keys=True)


def read_exclusions(archive: Path) -> tuple[set[str], dict[str, int]]:
    excluded: set[str] = set()
    counts: dict[str, int] = {}
    with zipfile.ZipFile(archive) as package:
        for kind, (basename, expected_sha) in NEGATIVE_MEMBERS.items():
            names = [
                name
                for name in package.namelist()
                if name.rsplit("/", 1)[-1] == basename
            ]
            if len(names) != 1:
                raise ValueError(f"Expected one publisher {kind} exclusion member")
            data = package.read(names[0])
            if sha256(data) != expected_sha:
                raise ValueError(f"Publisher {kind} exclusion SHA-256 changed")
            values = [line.strip() for line in data.decode("ascii").splitlines()]
            if not values or any(not re.fullmatch(r"[0-9a-f]{40}", x) for x in values):
                raise ValueError(f"Malformed publisher {kind} exclusion IDs")
            if len(values) != len(set(values)):
                raise ValueError(f"Duplicate publisher {kind} exclusion ID")
            counts[kind] = len(values)
            excluded.update(values)
    return excluded, counts


def select_pairs(
    rows: list[dict[str, Any]], excluded: set[str], count: int = 24
) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], dict[str, int]]:
    """Select opposite-label pairs, one per tree and distinct source URL."""
    if count < 1 or count % 2:
        raise ValueError("Blind packet count must be a positive even number")
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = (
        collections.defaultdict(list)
    )
    seen_ids: set[str] = set()
    excluded_train = 0
    for row in rows:
        required = ("utterance_id", "tree_id", "source_url", "snippet", "question")
        if not isinstance(row, dict) or any(
            not isinstance(row.get(key), str) or not row[key].strip()
            for key in required
        ):
            raise ValueError("Publisher TRAIN row has invalid identity or rule")
        item_id = row["utterance_id"]
        if item_id in seen_ids:
            raise ValueError("Publisher TRAIN utterance ID is duplicated")
        seen_ids.add(item_id)
        _visible(row)  # Validate the complete original input schema.
        if not isinstance(row.get("scenario"), str):
            raise ValueError("Publisher TRAIN scenario is missing or malformed")
        if item_id in excluded:
            excluded_train += 1
            continue
        if row.get("answer") not in ("Yes", "No"):
            continue
        key = (
            row["tree_id"],
            row["source_url"].strip(),
            normalized(row["snippet"]),
            normalized(row["question"]),
        )
        if not key[2] or not key[3]:
            raise ValueError("Publisher TRAIN has empty normalized rule/question")
        grouped[key].append(row)

    per_tree: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for (tree_id, _, _, _), members in grouped.items():
        yes = [row for row in members if row["answer"] == "Yes"]
        no = [row for row in members if row["answer"] == "No"]
        candidates = (
            (left, right)
            for left in yes
            for right in no
            if left["utterance_id"] != right["utterance_id"]
            and _state_key(left) != _state_key(right)
        )
        best = min(
            candidates,
            key=lambda pair: rank(
                "pair", tree_id, pair[0]["utterance_id"], pair[1]["utterance_id"]
            ),
            default=None,
        )
        if best is None:
            continue
        old = per_tree.get(tree_id)
        if old is None or rank(
            "tree-pair", tree_id, best[0]["utterance_id"], best[1]["utterance_id"]
        ) < rank("tree-pair", tree_id, old[0]["utterance_id"], old[1]["utterance_id"]):
            per_tree[tree_id] = best

    per_url: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for pair in per_tree.values():
        url = pair[0]["source_url"].strip()
        old = per_url.get(url)
        if old is None or rank("url-pair", pair[0]["tree_id"]) < rank(
            "url-pair", old[0]["tree_id"]
        ):
            per_url[url] = pair
    if len(per_url) < count:
        raise ValueError(
            f"Only {len(per_url)} distinct eligible source URLs; need {count}"
        )
    urls = sorted(per_url, key=lambda url: (rank("url", url), url))[:count]
    selected = [per_url[url] for url in urls]
    if len({pair[0]["tree_id"] for pair in selected}) != count:
        raise AssertionError("Selected source URLs do not have distinct trees")
    return selected, {
        "train_rows": len(rows),
        "excluded_train_rows": excluded_train,
        "eligible_trees": len(per_tree),
        "eligible_source_urls": len(per_url),
        "selected_pairs": count,
        "selected_source_urls": len(urls),
        "selected_trees": len({pair[0]["tree_id"] for pair in selected}),
        "selected_normalized_snippets": len(
            {normalized(pair[0]["snippet"]) for pair in selected}
        ),
    }


def make_packet(
    selected: list[tuple[dict[str, Any], dict[str, Any]]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    blind_pairs: list[dict[str, Any]] = []
    mapping: list[dict[str, Any]] = []
    for index, (yes, no) in enumerate(selected):
        # The reviewer never sees the source labels. Both possible orders occur
        # equally often and do not depend on a model prediction.
        first, second = (yes, no) if index % 2 == 0 else (no, yes)
        blind_id = f"P{index + 1:02d}"
        blind_pairs.append(
            {
                "blind_id": blind_id,
                "rule_snippet": first["snippet"],
                "question": first["question"],
                "states": [
                    {"state_id": "A", **_state(first)},
                    {"state_id": "B", **_state(second)},
                ],
            }
        )
        mapping.append(
            {
                "blind_id": blind_id,
                "source_url": first["source_url"],
                "tree_id": first["tree_id"],
                "exact_snippet_sha256": sha256(first["snippet"].encode("utf-8")),
                "normalized_snippet_sha256": sha256(
                    normalized(first["snippet"]).encode("utf-8")
                ),
                "utterance_ids": {
                    "A": first["utterance_id"],
                    "B": second["utterance_id"],
                },
            }
        )
    return (
        {"schema": "decision2-sharc-blind-review/3", "pairs": blind_pairs},
        {"schema": "decision2-sharc-blind-mapping/3", "pairs": mapping},
    )


def _write_private(path: Path, value: dict[str, Any]) -> str:
    data = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "wb") as output:
        output.write(data)
    return sha256(data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    rows, identity = read_train(args.archive)
    if identity["archive_sha256"] != ARCHIVE_SHA256:
        raise ValueError("Publisher archive SHA-256 changed")
    if identity["train_member_sha256"] != TRAIN_SHA256:
        raise ValueError("Publisher TRAIN member SHA-256 changed")
    excluded, negative_counts = read_exclusions(args.archive)
    selected, aggregate = select_pairs(rows, excluded)
    packet, mapping = make_packet(selected)
    args.out_dir.mkdir(mode=0o700, parents=False, exist_ok=False)
    packet_hash = _write_private(args.out_dir / "blind_review.json", packet)
    mapping_hash = _write_private(args.out_dir / "private_mapping.json", mapping)
    lengths = [len(_visible(row)) for pair in selected for row in pair]
    receipt = {
        "schema": "decision2-sharc-blind-receipt/3",
        "archive_sha256": identity["archive_sha256"],
        "train_member_sha256": identity["train_member_sha256"],
        "negative_list_counts": negative_counts,
        "blind_packet_sha256": packet_hash,
        "private_mapping_sha256": mapping_hash,
        "visible_chars_min": min(lengths),
        "visible_chars_median": statistics.median(lengths),
        "visible_chars_max": max(lengths),
        "training_admitted": False,
        "gpu_hours": 0,
        **aggregate,
    }
    receipt_hash = _write_private(args.out_dir / "aggregate_receipt.json", receipt)
    print(
        json.dumps(
            {
                "blind_packet_sha256": packet_hash,
                "private_mapping_sha256": mapping_hash,
                "aggregate_receipt_sha256": receipt_hash,
                "selected_pairs": aggregate["selected_pairs"],
                "eligible_source_urls": aggregate["eligible_source_urls"],
                "training_admitted": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
