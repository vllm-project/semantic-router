"""Materialize a 9B Milestone 3 TRAIN partition and its teacher file from a frozen spec.

Recipe rows are joined by id from hash-verified pool files (A0s from the
positional-key pk1 file). Every recipe row must receive exactly one teacher
distribution bound to its input hash and option keys. Retention replay rows
come from an A7 view: excluded families are dropped, rows whose input (current
or pre-renumbering hash) repeats A0 or a recipe row are dropped, and whole
groups are sampled to a native-token budget, stratified by source x task type x
language in a seed-keyed hash order. Replay rows keep their gold labels and get
no teacher term. The run refuses rows whose source matches a sealed C1
candidate or an ``mlx-diag`` source, rows over ``max_length`` tokens, and any
id / group / input shared with the isolation partitions.

    python3 -m lux9b.m3_data --spec SPEC --root hf=/hf --root repo=/code \\
        --tokenizer /model --output-dir OUT

Writes ``train.jsonl``, ``teacher.jsonl`` and ``manifest.json`` (inputs, counts,
tokens, dedupe and guard results, output hashes) into a new directory.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from training.model.data import (
    canonical,
    check_partition_isolation,
    file_sha256,
    load_partition,
)

SCHEMA = "decision2-9b-m3-data/1"
C1_STOPWORDS = frozenset(
    {
        "annotated",
        "annotations",
        "arguments",
        "benchmark",
        "claim",
        "corporate",
        "dataset",
        "detection",
        "human",
        "legal",
        "mining",
        "political",
        "preferences",
        "preview",
        "reddit",
        "review",
        "sample",
        "safety",
        "stance",
        "taiwan",
        "world",
    }
)
_TOKENIZER: Any = None


def resolve(reference: str, roots: dict[str, Path]) -> Path:
    name, _, relative = reference.partition(":")
    if name not in roots or not relative:
        raise ValueError(f"{reference}: unknown root or empty path")
    return roots[name] / relative


def verified(entry: dict[str, str], roots: dict[str, Path], inputs: dict) -> Path:
    path = resolve(entry["file"], roots)
    actual = file_sha256(path)
    if actual != entry["sha256"]:
        raise ValueError(f"{entry['file']}: sha256 {actual} != {entry['sha256']}")
    inputs[entry["file"]] = actual
    return path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def c1_keys(registry: dict[str, Any]) -> set[str]:
    """Distinctive name keys of every sealed C1 candidate (grades A/B)."""
    keys: set[str] = set()
    for source in registry["sources"]:
        if str(source.get("grade", "")).startswith("C"):
            continue
        name = source["dataset_id"].split("/")[-1].lower()
        keys.add(re.sub(r"[^a-z0-9]", "", name))
        keys.update(
            token
            for token in re.split(r"[^a-z0-9]+", name)
            if len(token) >= 5 and token not in C1_STOPWORDS
        )
    return keys


def denied_hits(source: str, keys: set[str], extra: list[str]) -> list[str]:
    lowered = source.lower()
    flat = re.sub(r"[^a-z0-9]", "", lowered)
    tokens = set(re.split(r"[^a-z0-9]+", lowered))
    hits = sorted(k for k in keys if (len(k) >= 6 and k in flat) or k in tokens)
    return hits + sorted(k for k in extra if k in flat)


def _init_tokenizer(path: str) -> None:
    global _TOKENIZER
    from transformers import AutoTokenizer

    _TOKENIZER = AutoTokenizer.from_pretrained(path, local_files_only=True)


def _length(row: dict[str, Any]) -> int:
    from training.model.decision_model import encode

    return len(encode(row, _TOKENIZER, 1 << 30)["ids"])


def token_lengths(
    rows: list[dict[str, Any]], tokenizer: Path, workers: int
) -> list[int]:
    if workers <= 1:
        _init_tokenizer(str(tokenizer))
        return [_length(row) for row in rows]
    with ProcessPoolExecutor(
        workers, initializer=_init_tokenizer, initargs=(str(tokenizer),)
    ) as pool:
        return list(pool.map(_length, rows, chunksize=128))


def check_teacher(row: dict[str, Any], record: dict[str, Any]) -> None:
    if record["input_sha256"] != row["input_sha256"]:
        raise ValueError(f"{row['id']}: teacher input hash differs from the row")
    probs = record["teacher_probs"]
    keys = [option["key"] for option in row["options"]]
    if not isinstance(probs, dict) or set(probs) != set(keys):
        raise ValueError(f"{row['id']}: teacher keys differ from the options")
    values = [probs[key] for key in keys]
    if any(type(v) not in (int, float) or v < 0 for v in values) or (
        abs(sum(values) - 1.0) > 1e-6
    ):
        raise ValueError(f"{row['id']}: invalid teacher distribution")


def stratified_groups(
    rows: list[dict[str, Any]], lengths: dict[str, int], budget: int, seed: str
) -> set[str]:
    from v2.dec.build_template_s import group_rows, select_groups

    groups = group_rows(rows)
    tokens = {
        g: sum(lengths[r["id"]] for r in members) for g, members in groups.items()
    }
    return set(select_groups(groups, tokens, budget, seed))


def extra_component(
    component: dict[str, Any],
    seed: str,
    max_length: int,
    roots: dict[str, Path],
    inputs: dict[str, str],
    blocked: set[str],
    taken: set[str],
    tokenizer: Path,
    workers: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Gold-label rows of one extra component after exclusion, dedupe and budget."""
    members = None
    if "view" in component:
        view = json.loads(verified(component["view"], roots, inputs).read_text())
        members = {m["id"] for m in view["members"] if m["part"] == "train"}
    excluded = set(component.get("exclude_families", []))
    counts: Counter = Counter()
    pool: list[dict[str, Any]] = []
    for entry in component["files"]:
        for row in load_partition(verified(entry, roots, inputs), "train"):
            if members is not None and row["id"] not in members:
                continue
            counts["in_view"] += 1
            if row["family"] in excluded:
                counts["excluded_family"] += 1
                continue
            audit = row["audit_metadata"]
            originals = {
                (audit.get("a7") or {}).get("original_input_sha256"),
                (audit.get("option_key_renumbering") or {}).get(
                    "original_input_sha256"
                ),
            } - {None}
            if row["input_sha256"] in blocked or originals & blocked:
                counts["duplicate_of_a0_or_recipe"] += 1
                continue
            if row["id"] in taken:
                raise ValueError(f"{row['id']}: extra row id repeats an earlier id")
            pool.append(row)
    lengths = dict(
        zip((r["id"] for r in pool), token_lengths(pool, tokenizer, workers))
    )
    over = {r["group_id"] for r in pool if lengths[r["id"]] > max_length}
    counts["over_length_rows"] = sum(r["group_id"] in over for r in pool)
    pool = [r for r in pool if r["group_id"] not in over]
    pool_tokens = sum(lengths[r["id"]] for r in pool)
    budget = component.get("budget_tokens")
    chosen = (
        stratified_groups(pool, lengths, budget, seed)
        if budget is not None and budget < pool_tokens
        else {r["group_id"] for r in pool}
    )
    rows = [r for r in pool if r["group_id"] in chosen]
    return rows, {
        **dict(counts),
        "view_members": len(members) if members is not None else None,
        "pool_rows": len(pool),
        "pool_tokens": pool_tokens,
        "budget_tokens": budget,
        "rows": len(rows),
        "tokens": sum(lengths[r["id"]] for r in rows),
        "rows_by_type": dict(Counter(r["task_type"] for r in rows)),
        "rows_by_family": dict(Counter(r["family"] for r in rows).most_common()),
        "rows_by_language": dict(Counter(r["language"] for r in rows)),
        "max_tokens": max((lengths[r["id"]] for r in rows), default=0),
    }


def build(spec: dict[str, Any], roots: dict[str, Path], tokenizer: Path, workers: int):
    inputs: dict[str, str] = {}
    recipe = read_jsonl(verified(spec["recipe"], roots, inputs))
    by_pool: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in recipe:
        by_pool[entry["pool"]].append(entry)
    if len({e["id"] for e in recipe}) != len(recipe):
        raise ValueError("Recipe lists an id twice")
    rows: list[dict[str, Any]] = []
    stats: dict[str, Any] = {"recipe_rows": len(recipe), "pools": {}}
    for pool, entries in sorted(by_pool.items()):
        available: dict[str, dict[str, Any]] = {}
        for entry in spec["pools"][pool]:
            for row in load_partition(verified(entry, roots, inputs), "train"):
                available.setdefault(row["id"], row)
        missing = [e["id"] for e in entries if e["id"] not in available]
        if missing:
            raise ValueError(
                f"{pool}: {len(missing)} recipe ids missing, e.g. {missing[:3]}"
            )
        wrong_source = [
            e["id"] for e in entries if available[e["id"]]["source"] != e["source"]
        ]
        if wrong_source:
            raise ValueError(f"{pool}: source differs for {wrong_source[:3]}")
        rows.extend(available[e["id"]] for e in entries)
        stats["pools"][pool] = len(entries)
    recipe_excluded = set(spec.get("recipe_exclude_families", []))
    if recipe_excluded:
        stats["recipe_excluded_family"] = sum(
            r["family"] in recipe_excluded for r in rows
        )
        rows = [r for r in rows if r["family"] not in recipe_excluded]
    recipe_ids = {row["id"] for row in rows}

    teacher: dict[str, dict[str, Any]] = {}
    teacher_by_file: dict[str, int] = {}
    for entry in spec["teachers"]:
        used = 0
        for record in read_jsonl(verified(entry, roots, inputs)):
            if record["id"] not in recipe_ids:
                continue
            previous = teacher.get(record["id"])
            if previous is not None and previous != record:
                raise ValueError(f"{record['id']}: conflicting teacher records")
            teacher[record["id"]] = record
            used += 1
        teacher_by_file[entry["file"]] = used
    uncovered = recipe_ids - set(teacher)
    if uncovered:
        raise ValueError(f"{len(uncovered)} recipe rows lack a teacher target")
    for row in rows:
        check_teacher(row, teacher[row["id"]])

    blocked: set[str] = {row["input_sha256"] for row in rows}
    for row in load_partition(verified(spec["a0_for_dedupe"], roots, inputs), "train"):
        blocked.add(row["input_sha256"])
        renumbering = row["audit_metadata"].get("option_key_renumbering") or {}
        if renumbering.get("original_input_sha256"):
            blocked.add(renumbering["original_input_sha256"])

    components = (
        [{"name": "replay", **spec["replay"]}] if "replay" in spec else []
    ) + [dict(c) for c in spec.get("extras", [])]
    replay: list[dict[str, Any]] = []
    extra_stats: dict[str, dict[str, Any]] = {}
    taken = set(recipe_ids)
    for component in components:
        seed = (
            spec["seed"]
            if component["name"] == "replay"
            else f"{spec['seed']}:{component['name']}"
        )
        chosen_rows, extra_stats[component["name"]] = extra_component(
            component,
            seed,
            spec["max_length"],
            roots,
            inputs,
            blocked,
            taken,
            tokenizer,
            workers,
        )
        replay.extend(chosen_rows)
        blocked.update(r["input_sha256"] for r in chosen_rows)
        taken.update(r["id"] for r in chosen_rows)
    replay_stats = extra_stats.get("replay", {})

    recipe_lengths = token_lengths(rows, tokenizer, workers)
    if max(recipe_lengths) > spec["max_length"]:
        raise ValueError("A recipe row exceeds max_length")
    native = {e["id"]: e["native"] for e in recipe}
    stats["native_token_mismatches"] = sum(
        native[row["id"]] != length for row, length in zip(rows, recipe_lengths)
    )
    stats["recipe_tokens"] = sum(recipe_lengths)
    stats["recipe_rows_by_type"] = dict(Counter(r["task_type"] for r in rows))
    stats["recipe_tokens_by_type"] = dict(
        sum(
            (Counter({r["task_type"]: n}) for r, n in zip(rows, recipe_lengths)),
            Counter(),
        )
    )

    train = sorted(rows + replay, key=lambda r: r["id"])
    if len({r["id"] for r in train}) != len(train):
        raise ValueError("Duplicate id in the materialized TRAIN")
    partitions = {"train": train}
    for entry in spec["isolation"]:
        partitions[entry["role"]] = load_partition(
            verified(entry, roots, inputs), entry["role"]
        )
    check_partition_isolation(partitions)

    registry = json.loads(
        resolve(spec["denied_sources"]["c1_registry"], roots).read_text()
    )
    keys = c1_keys(registry)
    sources = Counter(r["source"] for r in train)
    denied = {
        s: hits
        for s in sources
        if (hits := denied_hits(s, keys, spec["denied_sources"].get("extra", [])))
    }
    if denied:
        raise ValueError(f"Denied sources in TRAIN: {denied}")
    guard = {
        "c1_registry_sha256": file_sha256(
            resolve(spec["denied_sources"]["c1_registry"], roots)
        ),
        "c1_keys": len(keys),
        "extra": spec["denied_sources"].get("extra", []),
        "denied": 0,
        "sources": dict(sorted(sources.items())),
    }
    return (
        train,
        [teacher[i] for i in sorted(recipe_ids)],
        {
            "schema": SCHEMA,
            "name": spec["name"],
            "inputs_sha256": inputs,
            "recipe": stats,
            "teacher": {"rows": len(recipe_ids), "by_file": teacher_by_file},
            "replay": replay_stats,
            "extras": extra_stats,
            "train_rows": len(train),
            "train_rows_by_type": dict(Counter(r["task_type"] for r in train)),
            "train_rows_by_language": dict(Counter(r["language"] for r in train)),
            "isolation": sorted(p for p in partitions if p != "train"),
            "source_guard": guard,
        },
    )


def write_lines(path: Path, records: list[dict[str, Any]]) -> str:
    with path.open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(canonical(record) + "\n")
    return file_sha256(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--root", action="append", required=True, help="name=<path>")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=min(48, os.cpu_count() or 1))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"{args.output_dir} is not empty")
    roots = {}
    for item in args.root:
        name, _, path = item.partition("=")
        roots[name] = Path(path)
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    train, teacher, manifest = build(spec, roots, args.tokenizer, args.workers)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest["spec_sha256"] = file_sha256(args.spec)
    manifest["train_sha256"] = write_lines(args.output_dir / "train.jsonl", train)
    manifest["teacher_sha256"] = write_lines(args.output_dir / "teacher.jsonl", teacher)
    load_partition(args.output_dir / "train.jsonl", "train")
    manifest["tokenizer_files_sha256"] = {
        name: file_sha256(args.tokenizer / name)
        for name in ("tokenizer.json", "tokenizer_config.json")
        if (args.tokenizer / name).is_file()
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                k: manifest[k]
                for k in (
                    "train_rows",
                    "train_rows_by_type",
                    "train_sha256",
                    "teacher_sha256",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
