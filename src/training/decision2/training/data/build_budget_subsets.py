"""Make nested, source-balanced training budgets from the frozen combined 6k arm.

The input is TRAIN only. The external SELECT/CAL files are read for isolation
checks and copied byte-for-byte. The CSS panel is read only for ID/hash bans.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

from transfer import build as transfer

from training.data import build_pilot as pilot
from training.data.build_css_flute import read_panel_exclusions
from training.model.data import check_partition_isolation, load_partition

FROZEN_TRAIN_SHA256 = "4e82651181fd4f9b11e82718275a7370b82cbe810f20bbe87db9786cfdf888ad"
FROZEN_TRAIN_ROWS = 6000
SIZES = (1024, 2048)
SOURCE_COUNTS = {"legacy": 5000, "programmatic": 300, "flute": 700}


def bucket(row: dict[str, Any]) -> str:
    source = row["source"]
    if source.startswith("legacy:"):
        return "legacy"
    if source == "decision2_programmatic_original_v1":
        return "programmatic"
    if source == "css_flute_official_train":
        return "flute"
    raise ValueError(f"Unexpected frozen combined source: {source}")


def hash_rank(seed: str, value: str) -> str:
    return hashlib.sha256(f"{seed}\0{value}".encode()).hexdigest()


def stratum(row: dict[str, Any]) -> tuple[str, ...]:
    category = bucket(row)
    if category == "flute":
        return (row["options"][row["label"]]["key"],)
    if category == "programmatic":
        return (row["family"], row["task_type"], row["language"])
    return (row["family"], row["task_type"], row["language"], row["source"])


def balanced_group_order(
    rows: list[dict[str, Any]], seed: str
) -> list[list[dict[str, Any]]]:
    """Interleave source groups by their row mass across exact strata."""
    by_id: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        by_id[row["group_id"]].append(row)
    by_stratum: dict[tuple[str, ...], list[list[dict[str, Any]]]] = (
        collections.defaultdict(list)
    )
    for group_id, group in by_id.items():
        strata = {stratum(row) for row in group}
        if len(strata) != 1:
            raise ValueError(f"Frozen source group crosses strata: {group_id}")
        by_stratum[next(iter(strata))].append(sorted(group, key=lambda row: row["id"]))
    ranked: list[tuple[float, tuple[str, ...], str, list[dict[str, Any]]]] = []
    for key, groups in by_stratum.items():
        groups.sort(
            key=lambda group: (
                hash_rank(seed, group[0]["group_id"]),
                group[0]["group_id"],
            )
        )
        total = sum(len(group) for group in groups)
        cumulative = 0
        for group in groups:
            midpoint = (cumulative + len(group) / 2) / total
            ranked.append((midpoint, key, group[0]["group_id"], group))
            cumulative += len(group)
    ranked.sort(key=lambda item: (item[0], item[1], item[2]))
    return [group for _, _, _, group in ranked]


def extend_whole_groups(
    order: list[list[dict[str, Any]]], selected: list[list[dict[str, Any]]], target: int
) -> list[list[dict[str, Any]]]:
    chosen = list(selected)
    chosen_ids = {group[0]["group_id"] for group in chosen}
    count = sum(len(group) for group in chosen)
    if count > target:
        raise ValueError("A nested budget cannot remove a selected source group")
    for group in order:
        if count == target:
            break
        if group[0]["group_id"] not in chosen_ids and count + len(group) <= target:
            chosen.append(group)
            chosen_ids.add(group[0]["group_id"])
            count += len(group)
    if count != target:
        raise ValueError(f"Insufficient whole groups to reach budget {target}: {count}")
    return chosen


def largest_remainder(total: int, weights: dict[str, int]) -> dict[str, int]:
    denominator = sum(weights.values())
    quotas = {name: total * value // denominator for name, value in weights.items()}
    remainder = total - sum(quotas.values())
    order = sorted(
        weights, key=lambda name: (-(total * weights[name] % denominator), name)
    )
    for name in order[:remainder]:
        quotas[name] += 1
    return quotas


def nested_subsets(
    rows: list[dict[str, Any]], seed: str, sizes: tuple[int, ...] = SIZES
) -> tuple[dict[int, list[dict[str, Any]]], dict[int, dict[str, int]]]:
    categories: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        categories[bucket(row)].append(row)
    actual = {name: len(categories[name]) for name in SOURCE_COUNTS}
    if actual != SOURCE_COUNTS:
        raise ValueError(f"Frozen source mix changed: {actual}")
    if not sizes or tuple(sorted(set(sizes))) != sizes or sizes[-1] >= len(rows):
        raise ValueError(
            "Budgets must be ascending, unique and smaller than the frozen train arm"
        )
    orders = {
        name: balanced_group_order(categories[name], f"{seed}/{name}")
        for name in SOURCE_COUNTS
    }
    selected: dict[str, list[list[dict[str, Any]]]] = {
        name: [] for name in SOURCE_COUNTS
    }
    output: dict[int, list[dict[str, Any]]] = {}
    quotas_by_size = {}
    for size in sizes:
        quotas = largest_remainder(size, SOURCE_COUNTS)
        current = []
        for name in SOURCE_COUNTS:
            selected[name] = extend_whole_groups(
                orders[name], selected[name], quotas[name]
            )
            current.extend(row for group in selected[name] for row in group)
        if len(current) != size or len({row["id"] for row in current}) != size:
            raise AssertionError("Budget cardinality or IDs differ")
        current.sort(
            key=lambda row: (hash_rank(f"{seed}/output/{size}", row["id"]), row["id"])
        )
        output[size] = current
        quotas_by_size[size] = quotas
    smaller, larger = (set(row["id"] for row in output[size]) for size in sizes)
    if not smaller < larger:
        raise AssertionError("Budget subsets are not strictly nested")
    return output, quotas_by_size


def category_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    return {
        field: dict(sorted(collections.Counter(row[field] for row in rows).items()))
        for field in ("source", "family", "task_type", "language")
    }


def panel_overlap(
    rows: list[dict[str, Any]], excluded: dict[str, set[Any]]
) -> dict[str, int]:
    counts: collections.Counter[str] = collections.Counter()
    for row in rows:
        state = row["state"]
        text = state if isinstance(state, str) else pilot.canonical(state)
        if pilot.sha_bytes(text.encode("utf-8")) in excluded["raw_context_sha256"]:
            counts["raw_context"] += 1
        if (
            transfer.normalized_context_sha256(text)
            in excluded["normalized_context_sha256"]
        ):
            counts["normalized_context"] += 1
        if row["input_sha256"] in excluded["panel_input_sha256"]:
            counts["input_sha256"] += 1
    if counts:
        raise ValueError(
            f"Frozen source unexpectedly overlaps CSS panel: {dict(counts)}"
        )
    return {"raw_context": 0, "normalized_context": 0, "input_sha256": 0}


def build(args: argparse.Namespace) -> dict[str, Any]:
    source = args.source.resolve()
    select_file = args.select_file.resolve()
    cal_file = args.cal_file.resolve()
    output_dir = args.output_dir.resolve()
    paths = (source, select_file, cal_file)
    if len(set(paths)) != 3 or output_dir.exists():
        raise ValueError(
            "Use three distinct source partitions and a fresh output directory"
        )
    if pilot.sha_file(source) != FROZEN_TRAIN_SHA256:
        raise ValueError("Input differs from the frozen combined 6k arm")
    train = load_partition(source, "train")
    select = load_partition(select_file, "select")
    cal = load_partition(cal_file, "cal")
    if (
        len(train) != FROZEN_TRAIN_ROWS
        or len({row["group_id"] for row in train}) < 5000
    ):
        raise ValueError("Frozen arm row or group count changed")
    excluded, panel_receipt = read_panel_exclusions(args.panel_dir.resolve())
    subsets, quotas = nested_subsets(train, args.seed)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer.resolve()), local_files_only=True, trust_remote_code=False
    )
    payloads: dict[str, bytes] = {
        "select.jsonl": select_file.read_bytes(),
        "cal.jsonl": cal_file.read_bytes(),
    }
    receipts = {}
    for size, rows in subsets.items():
        check_partition_isolation({"train": rows, "select": select, "cal": cal})
        cross = {
            "train_select": pilot.overlap_audit(rows, select),
            "train_cal": pilot.overlap_audit(rows, cal),
        }
        if any(
            pilot.audit_has_exact_overlap(audit) or audit["near_duplicate"]["count"]
            for audit in cross.values()
        ):
            raise ValueError(f"Budget {size} overlaps an independent holdout")
        if pilot.train_consistency_audit(rows)["conflicting_gold_groups"]:
            raise ValueError(f"Budget {size} contains conflicting gold")
        panel = panel_overlap(rows, excluded)
        lengths = [pilot.count_tokens(row, tokenizer) for row in rows]
        if max(lengths) > args.max_row_tokens:
            raise ValueError(f"Budget {size} has row over token cap")
        name = f"combined_{size}.train.jsonl"
        payloads[name] = pilot.jsonl_bytes(rows)
        receipts[str(size)] = {
            "file": name,
            "sha256": pilot.sha_bytes(payloads[name]),
            "rows": len(rows),
            "source_quotas": quotas[size],
            "counts": category_counts(rows),
            "group_count": len({row["group_id"] for row in rows}),
            "group_integrity": all(
                sum(row["group_id"] == group for row in rows)
                == sum(row["group_id"] == group for row in train)
                for group in {row["group_id"] for row in rows}
            ),
            "selected_id_sha256": pilot.sha_bytes(
                pilot.canonical(sorted(row["id"] for row in rows)).encode()
            ),
            "token_audit": {
                "total": sum(lengths),
                "minimum": min(lengths),
                "maximum": max(lengths),
            },
            "cross_partition_audit": cross,
            "css_panel_overlap": panel,
        }
        if not receipts[str(size)]["group_integrity"]:
            raise AssertionError("Budget split a frozen source group")
    if not set(row["id"] for row in subsets[SIZES[0]]) < set(
        row["id"] for row in subsets[SIZES[1]]
    ):
        raise AssertionError("Expected strict budget nesting")
    output_dir.mkdir(parents=True, mode=0o700)
    for name, payload in payloads.items():
        pilot._atomic_write(output_dir / name, payload)
    manifest = {
        "schema_version": "decision2-combined-budget-subsets/1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "generator_code_sha256": pilot.sha_file(Path(__file__)),
        "frozen_train": {"sha256": FROZEN_TRAIN_SHA256, "rows": FROZEN_TRAIN_ROWS},
        "select_source_sha256": pilot.sha_file(select_file),
        "cal_source_sha256": pilot.sha_file(cal_file),
        "css_panel_exclusions": panel_receipt,
        "seed": args.seed,
        "sampling": "nested source quotas; stratum row-mass quantile; complete group selection",
        "tokenizer_revision": args.tokenizer_revision,
        "max_row_tokens": args.max_row_tokens,
        "budgets": receipts,
        "outputs": {
            name: {
                "sha256": pilot.sha_bytes(payload),
                "bytes": len(payload),
                "rows": len(payload.splitlines()),
            }
            for name, payload in payloads.items()
        },
        "interpretation": "These are nested train budgets from one frozen 6k arm, not new data mixes or independent evaluation partitions.",
        "limitations": [
            "Source balance is approximate within fine strata because complete legacy source groups are preserved.",
            "FLUTE is supervised same-task for the CSS FLUTE evaluation.",
            "The independent SELECT/CAL are monitor splits; CSS evaluation labels are never read by this builder.",
        ],
    }
    pilot._atomic_write(
        output_dir / "budget_subsets.manifest.json",
        (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode(),
    )
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--select-file", type=Path, required=True)
    parser.add_argument("--cal-file", type=Path, required=True)
    parser.add_argument("--panel-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--max-row-tokens", type=int, default=8192)
    parser.add_argument("--seed", default="decision2-combined-budget-v1")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = build(args)
    print(
        pilot.canonical(
            {
                size: {"sha256": info["sha256"], "tokens": info["token_audit"]["total"]}
                for size, info in result["budgets"].items()
            }
        )
    )


if __name__ == "__main__":
    main()
