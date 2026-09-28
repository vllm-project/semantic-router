"""Template-S training mixtures for the ~27B Milestone 2 contrasts (CPU, deterministic).

Every row is admitted under the pinned BEST368 segmented encoder at the training
limit; a row over the limit is excluded, never truncated. Token counts are the
trainer's own (``len(encode(row)["ids"])``). For an arm X,
``rho = min(tokens(base) // 2, tokens(X))``. Samples are whole groups taken as a
prefix of a stratified, seeded order (source x family x task types x language,
interleaved by within-stratum token position) until the token target is
reached. ``resample`` duplicates base groups; each duplicated row keeps every
field except ``id``, which gains a ``#r1`` suffix. Mixture rows go to the
private output directory only; the manifest holds counts and hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

RESAMPLE_SUFFIX = "#r1"
SCHEMA = "decision2-27b-m2-mixtures/1"


def canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def admit(
    rows: list[dict[str, Any]], length_of: Callable[[dict[str, Any]], int], limit: int
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, Any]]:
    """Keep rows whose native length fits ``limit``; return kept rows and lengths."""
    kept, lengths, over = [], {}, Counter()
    for row in rows:
        length = length_of(row)
        if length > limit:
            over[row["task_type"]] += 1
            continue
        kept.append(row)
        lengths[row["id"]] = length
    report = {
        "rows_in": len(rows),
        "rows_admitted": len(kept),
        "over_limit": sum(over.values()),
        "over_limit_by_type": dict(sorted(over.items())),
        "tokens_admitted": sum(lengths.values()),
    }
    return kept, lengths, report


def stratum(rows: list[dict[str, Any]]) -> tuple[str, str, str, str]:
    first = min(rows, key=lambda row: row["id"])
    types = "+".join(sorted({row["task_type"] for row in rows}))
    return (first["source"], first["family"], types, first["language"])


def group_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    return dict(groups)


def stratified_order(
    groups: dict[str, list[dict[str, Any]]], tokens: dict[str, int], seed: str
) -> list[str]:
    """Seeded within-stratum shuffles, interleaved by relative token position."""
    by_stratum: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for group_id in sorted(groups):
        by_stratum[stratum(groups[group_id])].append(group_id)
    keyed = []
    for key in sorted(by_stratum):
        members = list(by_stratum[key])
        random.Random(f"{seed}|{canonical(list(key))}").shuffle(members)
        total = sum(tokens[g] for g in members)
        position = 0
        for group_id in members:
            size = tokens[group_id]
            keyed.append(
                ((position + size / 2) / total, canonical(list(key)), group_id)
            )
            position += size
    keyed.sort()
    return [group_id for _, _, group_id in keyed]


def take_prefix(order: list[str], tokens: dict[str, int], target: int) -> list[str]:
    """Longest prefix at or below ``target``, plus the next group if it lands closer."""
    chosen, total = [], 0
    for group_id in order:
        size = tokens[group_id]
        if total + size <= target:
            chosen.append(group_id)
            total += size
            if total == target:
                break
            continue
        if abs(total + size - target) < abs(total - target):
            chosen.append(group_id)
        break
    return chosen


def sample_groups(
    rows: list[dict[str, Any]], lengths: dict[str, int], target: int, seed: str
) -> tuple[list[dict[str, Any]], list[str]]:
    groups = group_rows(rows)
    tokens = {
        g: sum(lengths[row["id"]] for row in members) for g, members in groups.items()
    }
    chosen = take_prefix(stratified_order(groups, tokens, seed), tokens, target)
    selected = [row for g in chosen for row in groups[g]]
    return selected, sorted(chosen)


def resampled(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{**row, "id": row["id"] + RESAMPLE_SUFFIX} for row in rows]


def parse_mixture(spec: str) -> tuple[str, list[tuple[str, str]]]:
    name, _, parts = spec.partition("=")
    items = []
    for part in parts.split(","):
        arm, _, size = part.partition(":")
        if not name or not arm or size not in ("full", "rho"):
            raise ValueError(f"bad mixture spec {spec!r}; use NAME=ARM:full|rho,...")
        items.append((arm, size))
    if items[0] != ("base", "full"):
        raise ValueError("every mixture starts with base:full")
    return name, items


def summarize(rows: list[dict[str, Any]], lengths: dict[str, int]) -> dict[str, Any]:
    by_type: dict[str, dict[str, int]] = defaultdict(lambda: {"rows": 0, "tokens": 0})
    for row in rows:
        by_type[row["task_type"]]["rows"] += 1
        by_type[row["task_type"]]["tokens"] += lengths[row["id"]]
    updates = math.ceil(len(rows) / 16)
    return {
        "rows": len(rows),
        "tokens": sum(lengths[row["id"]] for row in rows),
        "by_task_type": {k: dict(v) for k, v in sorted(by_type.items())},
        "by_language": dict(sorted(Counter(row["language"] for row in rows).items())),
        "by_source": dict(sorted(Counter(row["source"] for row in rows).items())),
        "groups": len({row["group_id"] for row in rows}),
        "updates_at_batch_16": updates,
        "save_every_for_8_checkpoints": math.ceil(updates / 8),
    }


def build(
    base: list[dict[str, Any]],
    arms: dict[str, list[dict[str, Any]]],
    lengths: dict[str, int],
    mixtures: list[str],
    seed: str,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Pure mixture construction over already admitted rows and native lengths."""
    base_tokens = sum(lengths[row["id"]] for row in base)
    pools = {"base": base, "resample": base, **arms}
    outputs, report = {}, {}
    for spec in mixtures:
        name, items = parse_mixture(spec)
        rows = list(base)
        parts = {"base": {"rows": len(base), "tokens": base_tokens, "size": "full"}}
        for arm, size in items[1:]:
            if (
                arm == "base"
                or arm not in pools
                or (arm == "resample" and size != "rho")
            ):
                raise ValueError(f"{name}: unsupported part {arm}:{size}")
            pool = pools[arm]
            pool_tokens = sum(lengths[row["id"]] for row in pool)
            if size == "full":
                picked, groups = list(pool), sorted({row["group_id"] for row in pool})
                target = pool_tokens
            else:
                target = min(base_tokens // 2, pool_tokens)
                picked, groups = sample_groups(pool, lengths, target, f"{seed}|{arm}")
            part_lengths = {row["id"]: lengths[row["id"]] for row in picked}
            if arm == "resample":
                picked = resampled(picked)
                part_lengths = {
                    row["id"]: lengths[row["id"][: -len(RESAMPLE_SUFFIX)]]
                    for row in picked
                }
            lengths.update(part_lengths)
            rows.extend(picked)
            parts[arm] = {
                "size": size,
                "target_tokens": target,
                "rows": len(picked),
                "tokens": sum(part_lengths.values()),
                "groups": len(groups),
                "group_ids_sha256": sha_bytes(canonical(groups).encode("utf-8")),
            }
        rows.sort(key=lambda row: row["id"])
        if len({row["id"] for row in rows}) != len(rows):
            raise ValueError(f"{name}: duplicate row ids")
        outputs[name] = rows
        report[name] = {"spec": spec, "parts": parts, **summarize(rows, lengths)}
    return outputs, report


def write_rows(path: Path, rows: list[dict[str, Any]]) -> str:
    data = "".join(canonical(row) + "\n" for row in rows).encode("utf-8")
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
    return sha_bytes(data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, required=True, help="Base snapshot (tokenizer)"
    )
    parser.add_argument("--limit", type=int, default=4096)
    parser.add_argument("--base", required=True, help="NAME=PATH")
    parser.add_argument(
        "--arm", action="append", default=[], help="NAME=PATH[,PATH...]"
    )
    parser.add_argument("--aho", action="append", default=[], help="NAME=PATH")
    parser.add_argument("--select", type=Path, required=True)
    parser.add_argument("--cal", type=Path, required=True)
    parser.add_argument("--expect", action="append", default=[], help="PATH=SHA256")
    parser.add_argument("--mixture", action="append", required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    from training.model.data import (
        check_partition_isolation,
        load_partition,
        validate_row,
    )
    from training.model.decision_model import encode

    expected = dict(item.rsplit("=", 1) for item in args.expect)
    inputs = [args.base.split("=", 1)[1], str(args.select), str(args.cal)]
    inputs += [p for spec in args.arm for p in spec.split("=", 1)[1].split(",")]
    inputs += [spec.split("=", 1)[1] for spec in args.aho]
    hashes = {}
    for path in inputs:
        hashes[path] = sha_file(Path(path))
        if path in expected and expected[path] != hashes[path]:
            raise SystemExit(
                f"{path}: {hashes[path]} differs from the frozen {expected[path]}"
            )
    missing = sorted(set(inputs) - set(expected))
    if missing:
        raise SystemExit(f"no frozen hash given for {missing}")

    tokenizer = AutoTokenizer.from_pretrained(args.source, local_files_only=True)

    def length_of(row: dict[str, Any]) -> int:
        return len(encode(row, tokenizer, 1 << 30)["ids"])

    admission, lengths = {}, {}
    base_name, base_path = args.base.split("=", 1)
    base, base_lengths, admission[base_name] = admit(
        read_rows(Path(base_path)), length_of, args.limit
    )
    lengths.update(base_lengths)
    arms = {}
    for spec in args.arm:
        name, paths = spec.split("=", 1)
        rows = [row for path in paths.split(",") for row in read_rows(Path(path))]
        arms[name], arm_lengths, admission[name] = admit(rows, length_of, args.limit)
        lengths.update(arm_lengths)
    outputs, report = build(base, arms, lengths, args.mixture, args.seed)

    select = load_partition(args.select, "select")
    cal = load_partition(args.cal, "cal")
    aho_rows = {}
    for spec in args.aho:
        name, path = spec.split("=", 1)
        rows = load_partition(path, "select")
        kept, _, admission[f"aho-{name}"] = admit(rows, length_of, args.limit)
        aho_rows[name] = kept
    args.output_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    files = {}
    for name, rows in outputs.items():
        for row in rows:
            validate_row(row, "train")
        check_partition_isolation({"train": rows, "select": select, "cal": cal})
        for aho_name, kept in aho_rows.items():
            check_partition_isolation({"train": rows, f"aho-{aho_name}": kept})
        file_name = f"{name}.train.jsonl"
        files[file_name] = write_rows(args.output_dir / file_name, rows)
    for name, kept in aho_rows.items():
        file_name = f"aho-{name}.jsonl"
        files[file_name] = write_rows(
            args.output_dir / file_name, sorted(kept, key=lambda r: r["id"])
        )
    manifest = {
        "schema": SCHEMA,
        "seed": args.seed,
        "limit": args.limit,
        "tokenizer_json_sha256": sha_file(args.source / "tokenizer.json"),
        "encoder": "pinned BEST368 training.model.decision_model.encode",
        "inputs_sha256": hashes,
        "admission": admission,
        "mixtures": report,
        "files_sha256": files,
        "isolation": "zero id/group_id/input_sha256 intersections vs SELECT, CAL and every AHO slice",
    }
    text = json.dumps(manifest, indent=1, sort_keys=True) + "\n"
    fd = os.open(
        args.output_dir / "MIXTURES.json", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644
    )
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(text)
    print(
        json.dumps(
            {
                name: {
                    k: v
                    for k, v in r.items()
                    if k in ("rows", "tokens", "updates_at_batch_16")
                }
                for name, r in report.items()
            }
        )
    )
    print(json.dumps(files))


if __name__ == "__main__":
    main()
