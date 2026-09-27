"""Prepare private, label-blinded ConTRoL projection review packets.

This script reads only publisher TRAIN and gold-free protected prompts. Its
outputs contain restricted text and must stay in a private experiment directory.
No output is a student TRAIN file or a release evaluation result.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import re
import unicodedata
from pathlib import Path
from typing import Any

SOURCE_TRAIN_SHA256 = "e51b63fa1da381a27fb5244e6f3c8f317eed51921e20fc05334023db3e2e834f"
SOURCE_LABELS = {"c": "contradiction", "n": "neutral", "e": "entailment"}
RELATIONS = ("contradiction", "neutral", "entailment")
TASKS = ("choice", "support_noul", "contradiction_noul")
SEED = "control-native-blind-review-v1-20260928"


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def compact(text: str) -> str:
    return "".join(char for char in normalize(text) if char.isalnum())


def grams(text: str) -> set[str]:
    text = compact(text)
    return {text[index : index + 6] for index in range(max(0, len(text) - 5))}


def text_leaves(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [leaf for child in value.values() for leaf in text_leaves(child)]
    if isinstance(value, list):
        return [leaf for child in value for leaf in text_leaves(child)]
    return []


def read_source(path: Path) -> list[dict[str, Any]]:
    if sha_file(path) != SOURCE_TRAIN_SHA256:
        raise ValueError("Publisher TRAIN bytes differ from the frozen source")
    pairs = []
    for position, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        row = json.loads(line)
        if set(row) != {"uid", "premise", "hypothesis", "label"}:
            raise ValueError("Publisher TRAIN schema differs")
        if row["label"] not in SOURCE_LABELS:
            raise ValueError("Publisher TRAIN label differs")
        if not row["premise"] or not row["hypothesis"]:
            raise ValueError("Publisher TRAIN text missing")
        pairs.append(
            {
                "position": position,
                "premise": row["premise"],
                "hypothesis": row["hypothesis"],
                "label": SOURCE_LABELS[row["label"]],
                "group": normalize(row["premise"]),
            }
        )
    return pairs


def read_protected(path: Path) -> tuple[list[dict[str, str]], str]:
    inventory_hash = sha_file(path)
    entries = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(entries, list) or not entries:
        raise ValueError("Protected inventory missing")
    protected = []
    roles = set()
    for entry in entries:
        role = entry["role"]
        source = Path(entry["path"])
        if role in roles or not source.is_file() or sha_file(source) != entry["sha256"]:
            raise ValueError("Protected role missing, changed, or repeated")
        roles.add(role)
        for line in source.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if not entry.get("extra") and not set(row) <= {
                "id",
                "group_id",
                "review_id",
                "family",
                "operation",
                "language",
                "state",
                "questions",
                "instructions",
                "options",
            }:
                raise ValueError("Protected prompt contains a non-prompt field")
            for field in ("state", "instructions", "options"):
                for leaf in text_leaves(row.get(field)):
                    text = normalize(leaf)
                    if len(compact(text)) >= 20:
                        protected.append({"role": role, "text": text})
    required = {
        "typed_dev",
        "css_pilot",
        "typed_final_goldfree",
        "css15_goldfree",
        "jevbench_public231",
        "rights_clean_train",
        "rights_clean_select",
        "rights_clean_cal",
    }
    if not required <= roles:
        raise ValueError("Core protected roles missing")
    return protected, inventory_hash


def quarantine_groups(
    pairs: list[dict[str, Any]], protected: list[dict[str, str]]
) -> set[str]:
    """Match the frozen source scanner's exact and <=800-char near rule."""
    exact = collections.defaultdict(set)
    snippets = collections.defaultdict(set)
    for item in protected:
        exact[item["text"]].add(item["role"])
        if len(item["text"]) <= 800:
            snippets[item["text"]].add(item["role"])
    references = list(snippets)
    ref_grams = [grams(text) for text in references]
    index = collections.defaultdict(list)
    for ref_id, gram_set in enumerate(ref_grams):
        for gram in sorted(gram_set)[::2]:
            index[gram].append(ref_id)
    source_text_groups = collections.defaultdict(set)
    for pair in pairs:
        for text in (pair["premise"], pair["hypothesis"]):
            source_text_groups[normalize(text)].add(pair["group"])
    blocked = set()
    for source_text, groups in source_text_groups.items():
        if source_text in exact:
            blocked.update(groups)
        source_grams = grams(source_text)
        if not source_grams:
            continue
        rare = sorted(
            (gram for gram in source_grams if gram in index),
            key=lambda gram: (len(index[gram]), gram),
        )[:6]
        candidates = {
            ref_id for gram in rare if len(index[gram]) <= 128 for ref_id in index[gram]
        }
        for ref_id in candidates:
            other = ref_grams[ref_id]
            if not other:
                continue
            common = len(source_grams & other)
            if (
                common / len(source_grams | other) >= 0.72
                or common / min(len(source_grams), len(other)) >= 0.90
            ):
                blocked.update(groups)
    return blocked


def retained_pairs(
    pairs: list[dict[str, Any]], blocked: set[str]
) -> list[dict[str, Any]]:
    seen_pairs = {}
    duplicates = 0
    conflicts = set()
    candidate = []
    for pair in pairs:
        if pair["group"] in blocked:
            continue
        key = normalize(pair["premise"]), normalize(pair["hypothesis"])
        previous = seen_pairs.get(key)
        if previous is not None:
            duplicates += 1
            if previous != pair["label"]:
                conflicts.add(pair["group"])
            continue
        seen_pairs[key] = pair["label"]
        candidate.append(pair)
    candidate = [pair for pair in candidate if pair["group"] not in conflicts]
    if (len(blocked), duplicates, len(conflicts), len(candidate)) != (5, 26, 13, 6618):
        raise ValueError("Candidate pool differs from the audited projection")
    if len({pair["group"] for pair in candidate}) != 1509:
        raise ValueError("Candidate premise-group count differs")
    return candidate


def stable_rank(*parts: object) -> str:
    return sha_bytes("\0".join(str(part) for part in (SEED, *parts)).encode())


def sample_pairs(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chosen = []
    groups = set()
    for label in RELATIONS:
        pool = [pair for pair in pairs if pair["label"] == label]
        lengths = sorted(
            len(pair["premise"]) + len(pair["hypothesis"]) for pair in pool
        )
        cut1, cut2 = lengths[len(lengths) // 3], lengths[2 * len(lengths) // 3]
        for bucket, want in ((0, 3), (1, 4), (2, 3)):
            eligible = [
                pair
                for pair in pool
                if (
                    0
                    if len(pair["premise"]) + len(pair["hypothesis"]) < cut1
                    else (
                        1
                        if len(pair["premise"]) + len(pair["hypothesis"]) < cut2
                        else 2
                    )
                )
                == bucket
            ]
            eligible.sort(key=lambda pair: stable_rank(pair["group"], pair["position"]))
            count = 0
            for pair in eligible:
                if pair["group"] in groups:
                    continue
                copy = dict(pair)
                copy["length_bin"] = ("short", "medium", "long")[bucket]
                chosen.append(copy)
                groups.add(pair["group"])
                count += 1
                if count == want:
                    break
            if count != want:
                raise ValueError("Insufficient independent groups for stratum")
    if len(chosen) != 30 or len(groups) != 30:
        raise ValueError("Sample must have 30 independent groups")
    chosen.sort(key=lambda pair: stable_rank("presentation", pair["position"]))
    return chosen


def make_packets(pairs: list[dict[str, Any]]) -> tuple[list[dict], list[dict]]:
    blind = []
    reveal = []
    relation_slot = collections.Counter()
    for pair in pairs:
        label = pair["label"]
        slot = relation_slot[label]
        relation_slot[label] += 1
        task = TASKS[(slot + RELATIONS.index(label)) % len(TASKS)]
        sample_id = stable_rank("sample", pair["position"], pair["group"])[:16]
        native_id = sha_bytes(
            f"{pair['position']}\0{pair['group']}\0"
            f"{'choice' if task == 'choice' else 'supports' if task == 'support_noul' else 'contradicts'}".encode()
        )
        if task == "choice":
            options = ["supports", "contradicts", "undetermined"]
            shift = int(native_id[:8], 16) % 3
            options = options[shift:] + options[:shift]
            expected = {
                "entailment": "supports",
                "contradiction": "contradicts",
                "neutral": "undetermined",
            }[label]
            question = "Which relation between the context and claim is established?"
        else:
            options = ["false", "true"]
            if int(native_id[:8], 16) % 2:
                options.reverse()
            target = "entailment" if task == "support_noul" else "contradiction"
            expected = "true" if label == target else "false"
            question = (
                "Does the context establish the claim?"
                if task == "support_noul"
                else "Does the context establish that the claim is false?"
            )
        blind.append(
            {
                "sample_id": sample_id,
                "task": task,
                "length_bin": pair["length_bin"],
                "state": pair["premise"],
                "claim": pair["hypothesis"],
                "question": question,
                "options": options,
            }
        )
        reveal.append(
            {
                "sample_id": sample_id,
                "source_position": pair["position"],
                "source_label": label,
                "expected_key": expected,
                "expected_option_position": options.index(expected),
            }
        )
    if collections.Counter(item["task"] for item in blind) != dict.fromkeys(TASKS, 10):
        raise ValueError("Sample task counts differ")
    return blind, reveal


def long_leaf_neighbors(
    pairs: list[dict[str, Any]], protected: list[dict[str, str]]
) -> list[dict[str, Any]]:
    """Target 10 >800-char leaves with rare-word source neighbors for review."""
    words = lambda text: set(re.findall(r"[a-z]{5,}", text.casefold()))
    source_docs = [words(pair["premise"] + " " + pair["hypothesis"]) for pair in pairs]
    frequency = collections.Counter(
        word for document in source_docs for word in document
    )
    index = collections.defaultdict(list)
    for source_id, document in enumerate(source_docs):
        for word in document:
            if 2 <= frequency[word] <= 40:
                index[word].append(source_id)
    ranked = []
    seen_leaves = set()
    for item in protected:
        leaf = item["text"]
        if len(leaf) <= 800 or leaf in seen_leaves:
            continue
        seen_leaves.add(leaf)
        leaf_words = words(leaf)
        candidates = collections.Counter()
        for word in leaf_words:
            if word in index:
                weight = math.log1p(len(pairs) / (1 + frequency[word]))
                for source_id in index[word]:
                    candidates[source_id] += weight
        if not candidates:
            continue
        source_id, score = max(
            candidates.items(), key=lambda value: (value[1], -value[0])
        )
        source = pairs[source_id]
        ranked.append(
            {
                "role": item["role"],
                "protected_leaf_sha256": sha_bytes(leaf.encode()),
                "leaf_chars": len(leaf),
                "source_position": source["position"],
                "source_group_sha256": sha_bytes(source["group"].encode()),
                "rare_word_score": round(score, 4),
                "protected_leaf": leaf,
                "source_state": source["premise"],
                "source_claim": source["hypothesis"],
            }
        )
    ranked.sort(
        key=lambda item: (
            -item["rare_word_score"],
            item["role"],
            item["protected_leaf_sha256"],
        )
    )
    selected = []
    role_counts = collections.Counter()
    for item in ranked:
        if role_counts[item["role"]] >= 3:
            continue
        selected.append(item)
        role_counts[item["role"]] += 1
        if len(selected) == 10:
            break
    if len(selected) != 10:
        raise ValueError("Fewer than 10 long protected leaf neighbors")
    return selected


def write_private_json(path: Path, value: Any) -> str:
    if path.exists():
        raise FileExistsError(path)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return sha_file(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publisher-train", type=Path, required=True)
    parser.add_argument("--protected-inventory", type=Path, required=True)
    parser.add_argument("--private-output-dir", type=Path, required=True)
    args = parser.parse_args()
    destination = args.private_output_dir
    if not destination.is_dir() or destination.stat().st_mode & 0o077:
        raise ValueError("Output directory must exist and be owner-only")
    source = read_source(args.publisher_train)
    protected, inventory_hash = read_protected(args.protected_inventory)
    blocked = quarantine_groups(source, protected)
    pairs = retained_pairs(source, blocked)
    blind, reveal = make_packets(sample_pairs(pairs))
    long_neighbors = long_leaf_neighbors(pairs, protected)
    hashes = {
        "blind": write_private_json(destination / "blind.json", blind),
        "reveal": write_private_json(destination / "reveal.json", reveal),
        "long_neighbors": write_private_json(
            destination / "long_neighbors.json", long_neighbors
        ),
    }
    print(
        json.dumps(
            {
                "seed": SEED,
                "publisher_train_sha256": SOURCE_TRAIN_SHA256,
                "protected_inventory_sha256": inventory_hash,
                "blocked_groups": len(blocked),
                "retained_pairs": len(pairs),
                "sample_groups": 30,
                "long_leaves": len(long_neighbors),
                "private_packet_sha256": hashes,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
