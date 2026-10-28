"""Pinned multilingual extension with AEGIS translation and source-group isolation.

CultureGuard is synthetic translation/adaptation of AEGIS, not twelve independent
natural benchmarks. Its jailbreak tag is never used as a PromptGuard label.
"""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from ..sequence_repair.data import normalized_text
from .vela_data import LABELS, SOURCE_SHA256, hazard_target

SPLITS = {"test": "test", "valid": "validation", "train": "train"}
MIN_TRAIN_CHARACTERS = 10


def fingerprint(text):
    return hashlib.sha256(normalized_text(text).encode()).hexdigest()


def source_groups(aegis):
    """Map all response variants of a source ID to the original prompt group."""
    mapping, texts = {}, set()
    for split in ["test", "validation", "train"]:
        payload = (Path(aegis) / f"{split}.json").read_bytes()
        if hashlib.sha256(payload).hexdigest() != SOURCE_SHA256[split]:
            raise ValueError("AEGIS reference differs from pinned source")
        for row in json.loads(payload):
            key = fingerprint(row["prompt"])
            value = (split, f"aegis:{key}")
            if str(row["id"]) in mapping and mapping[str(row["id"])][1] != value[1]:
                raise ValueError("One original ID maps to different prompts")
            mapping.setdefault(str(row["id"]), value)
            texts.add(key)
    return mapping, texts


def group_for(row, split, original):
    source_id = str(row["id"])
    if source_id in original:
        source_split, group = original[source_id]
        if source_split != split:
            raise ValueError("Translation moved across the original AEGIS split")
        return group
    return f"cultureguard:{source_id}"


def capped(rows, maximum):
    if not maximum:
        return rows
    count, result = Counter(), []
    for row in sorted(rows, key=lambda item: fingerprint(item["id"])):
        key = row["language"], row["label"]
        if count[key] >= maximum:
            continue
        count[key] += 1
        result.append(row)
    return result


def merge_annotations(first, other):
    """A refusal response variant can make prompt-category attribution eligible."""
    if first["hazard_exclusion"]:
        return other if not other["hazard_exclusion"] else first
    if other["hazard_exclusion"]:
        return first
    mask = [
        int(a == b and observed_a and observed_b)
        for a, b, observed_a, observed_b in zip(
            first["targets"],
            other["targets"],
            first["label_mask"],
            other["label_mask"],
            strict=True,
        )
    ]
    target = [
        value if observed else 0
        for value, observed in zip(first["targets"], mask, strict=True)
    ]
    return {
        **first,
        "targets": target,
        "label_mask": mask,
        "hazard_exclusion": (
            None
            if any(mask) and (first["label"] == "safe" or any(target))
            else "duplicate category annotations disagree"
        ),
    }


def build(source, aegis, train_per_label=2000, dev_per_label=150):
    contract = json.loads(
        (Path(__file__).parent / "configs/cultureguard-source-v1.json").read_text()
    )
    original, original_text = source_groups(aegis)
    groups, all_rows, audit = {}, defaultdict(list), Counter()
    languages = sorted({name.split("/")[0] for name in contract["files"]})
    for source_split, split in SPLITS.items():
        for language in languages:
            name = f"{language}/{source_split}.jsonl"
            payload = (Path(source) / name).read_bytes()
            if hashlib.sha256(payload).hexdigest() != contract["files"][name]:
                raise ValueError(
                    f"CultureGuard source differs from pinned revision: {name}"
                )
            for line in payload.decode().split("\n"):
                if not line.strip():
                    continue
                row = json.loads(line)
                group = group_for(row, split, original)
                if group in groups and groups[group] != split:
                    # Different original IDs can share the same normalized prompt.
                    # Process test, validation, then train so the entire family
                    # remains in its most restrictive source partition.
                    audit[f"{split}:group_reserved_for_{groups[group]}"] += 1
                    continue
                groups[group] = split
                text = (row.get("prompt") or "").strip()
                if not text or text == "REDACTED":
                    audit[f"{split}:empty_or_redacted"] += 1
                    continue
                key = fingerprint(text)
                if key in original_text:
                    audit[f"{split}:original_AEGIS_text_already_covered"] += 1
                    continue
                if split == "train" and len(text) < MIN_TRAIN_CHARACTERS:
                    audit["train:short_ambiguous_prompt"] += 1
                    continue
                if row["prompt_label"] not in {"safe", "unsafe"}:
                    raise ValueError("Unknown source prompt label")
                targets, mask, reason = hazard_target(
                    {**row, "violated_categories": row.get("violated_categories") or ""}
                )
                item = {
                    "id": f"cultureguard:{language}:{row['id']}:{key}",
                    "group_id": group,
                    "text": text,
                    "label": row["prompt_label"],
                    "source": "cultureguard",
                    "source_revision": contract["revision"],
                    "source_split": split,
                    "source_tag": row.get("tag"),
                    "annotation_provenance": "synthetic translation/adaptation of source labels",
                    "language": language,
                    "length_bucket": "natural_short",
                    "targets": targets,
                    "label_mask": mask,
                    "hazard_exclusion": reason,
                }
                all_rows[split].append(item)
    # Inconsistent normalized text is excluded globally, including across languages.
    labels, conflicts = {}, set()
    for rows in all_rows.values():
        for row in rows:
            key = fingerprint(row["text"])
            if key in labels and labels[key] != row["label"]:
                conflicts.add(key)
            labels.setdefault(key, row["label"])
    audit["contradictory_normalized_texts"] = len(conflicts)
    seen, result = set(), {"safety": {}, "hazard": {}}
    for split in ["test", "validation", "train"]:
        by_text = {}
        for row in all_rows[split]:
            key = fingerprint(row["text"])
            if key in seen or key in conflicts:
                audit[f"{split}:normalized_duplicate_or_conflict"] += 1
                continue
            by_text[key] = (
                merge_annotations(by_text[key], row) if key in by_text else row
            )
        seen.update(by_text)
        unique = list(by_text.values())
        cap = (
            train_per_label
            if split == "train"
            else dev_per_label if split == "validation" else 0
        )
        result["safety"][split] = capped(unique, cap)
        eligible = []
        for row in unique:
            if row["hazard_exclusion"]:
                audit[f"{split}:hazard:{row['hazard_exclusion']}"] += 1
            else:
                eligible.append(row)
        result["hazard"][split] = capped(eligible, cap)
    return result, dict(audit), contract


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--aegis", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-per-language-label", type=int, default=2000)
    parser.add_argument("--dev-per-language-label", type=int, default=150)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing overwrite of an immutable corpus")
    result, audit, source = build(
        args.source,
        args.aegis,
        args.train_per_language_label,
        args.dev_per_language_label,
    )
    manifest = {
        "source": source,
        "audit": audit,
        "selection": {
            "train_per_language_label": args.train_per_language_label,
            "dev_per_language_label": args.dev_per_language_label,
        },
        "files": {},
        "test_used_for_selection": False,
    }
    for task, splits in result.items():
        directory = args.output / task
        directory.mkdir(parents=True)
        for split, rows in splits.items():
            path = directory / f"{split}.jsonl"
            path.write_text(
                "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows)
            )
            info = {
                "rows": len(rows),
                "groups": len({row["group_id"] for row in rows}),
                "labels": dict(Counter(row["label"] for row in rows)),
                "languages": dict(Counter(row["language"] for row in rows)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            if task == "hazard":
                info["supervision"] = {
                    label: {
                        "positive": sum(row["targets"][i] for row in rows),
                        "negative": sum(
                            row["label_mask"][i] and not row["targets"][i]
                            for row in rows
                        ),
                        "unknown": sum(not row["label_mask"][i] for row in rows),
                    }
                    for i, label in enumerate(LABELS)
                }
            manifest["files"][f"{task}/{split}"] = info
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps({"files": manifest["files"], "audit": audit}, indent=2), flush=True
    )


if __name__ == "__main__":
    main()
