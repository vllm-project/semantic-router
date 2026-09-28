"""Training mixtures for the ~27B Milestone 2 and 3 contrasts (CPU, deterministic).

Every row is admitted under the pinned BEST368 segmented encoder at the training
limit; a row over the limit is excluded, never truncated. Token counts are the
trainer's own (``len(encode(row)["ids"])``). For an arm X,
``rho = min(tokens(base) // 2, tokens(X))``. Samples are whole groups taken as a
prefix of a stratified, seeded order (source x family x task types x language,
interleaved by within-stratum token position) until the token target is
reached. ``resample`` duplicates base groups; each duplicated row keeps every
field except ``id``, which gains a ``#r1`` suffix. Mixture rows go to the
private output directory only; the manifest holds counts and hashes.

Milestone 3 terms (see ``GRAMMAR``). ``ARM:tokens=N:family-equal`` first drops
every arm group sharing a ``group_id`` or any row's ``input_sha256`` with the
mixture built so far, then water-fills N tokens over ``family``: a family whose
tokens fit the equal share gives all its groups and the rest is re-split
equally among the others (integer shares; the remainder goes one token each to
the first families by name). Each family's groups are shuffled with a seed per
family and its share is filled by the M2 prefix rule: the longest prefix at or
below the share, plus the next group if that lands closer. ``repeat:match=M``
keeps the rows built so far as pass 1 and appends passes ``#r2``, ``#r3``, ...
(pass K>1 has suffix ``#r{K-1}``): every whole pass while it fits under M's
total, then a stratified seeded whole-group prefix by the same rule; the total
must land within 0.5% of M's. The trainer shuffles the whole file, so a full
pass is every pass-1 row and only the last pass depends on its seed.
``--aho-sample`` draws at most COUNT admitted rows by equal per-family row
shares (water-filled), whole groups in seeded order, never above a share.
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
SCHEMA = "decision2-27b-mixtures/2"
GRAMMAR = (
    "NAME=base:full[,TERM...]; TERM = ARM:full | ARM:rho | resample:rho"
    " | ARM:tokens=N:family-equal | repeat:match=EARLIER_NAME (last term only)"
)
REPEAT_TOLERANCE = 0.005


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


def waterfill(totals: dict[str, int], target: int) -> dict[str, int]:
    """Equal integer shares of ``target``; keys whose total fits a share take it all."""
    quotas, active, remaining = {}, sorted(totals), target
    while active:
        small = [k for k in active if totals[k] * len(active) <= remaining]
        if not small:
            break
        for key in small:
            quotas[key] = totals[key]
            remaining -= totals[key]
        active = [k for k in active if k not in small]
    if active:
        share, extra = divmod(remaining, len(active))
        for index, key in enumerate(active):
            quotas[key] = share + (index < extra)
    return quotas


def take_at_most(order: list[str], sizes: dict[str, int], target: int) -> list[str]:
    chosen, total = [], 0
    for key in order:
        if total + sizes[key] > target:
            break
        chosen.append(key)
        total += sizes[key]
    return chosen


def group_family(group_id: str, members: list[dict[str, Any]]) -> str:
    families = {row["family"] for row in members}
    if len(families) != 1:
        raise ValueError(f"group {group_id} spans families {sorted(families)}")
    return families.pop()


def family_order(
    groups: dict[str, list[dict[str, Any]]], seed: str
) -> dict[str, list[str]]:
    by_family: dict[str, list[str]] = defaultdict(list)
    for group_id in sorted(groups):
        by_family[group_family(group_id, groups[group_id])].append(group_id)
    for family, members in by_family.items():
        random.Random(f"{seed}|{family}").shuffle(members)
    return dict(sorted(by_family.items()))


def family_equal(
    pool: list[dict[str, Any]],
    present: list[dict[str, Any]],
    lengths: dict[str, int],
    target: int,
    seed: str,
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    seen_inputs = {row["input_sha256"] for row in present}
    seen_groups = {row["group_id"] for row in present}
    groups = group_rows(pool)
    dropped = sorted(
        g
        for g, members in groups.items()
        if g in seen_groups or any(r["input_sha256"] in seen_inputs for r in members)
    )
    dropped_rows = sum(len(groups[g]) for g in dropped)
    for group_id in dropped:
        del groups[group_id]
    inputs = [row["input_sha256"] for members in groups.values() for row in members]
    if len(inputs) != len(set(inputs)):
        raise ValueError("arm pool repeats an input_sha256 across its own rows")
    tokens = {
        g: sum(lengths[row["id"]] for row in members) for g, members in groups.items()
    }
    orders = family_order(groups, seed)
    totals = {f: sum(tokens[g] for g in order) for f, order in orders.items()}
    quotas = waterfill(totals, target)
    chosen, families = [], {}
    for family, order in orders.items():
        picked = take_prefix(order, tokens, quotas[family])
        chosen.extend(picked)
        families[family] = {
            "available_tokens": totals[family],
            "quota": quotas[family],
            "tokens": sum(tokens[g] for g in picked),
            "groups": len(picked),
            "rows": sum(len(groups[g]) for g in picked),
            "exhausted": len(picked) == len(order),
        }
    selected = [row for g in chosen for row in groups[g]]
    extra = {
        "dropped_duplicate_groups": len(dropped),
        "dropped_duplicate_rows": dropped_rows,
        "families": families,
    }
    return selected, sorted(chosen), extra


def repeat_passes(
    rows: list[dict[str, Any]], lengths: dict[str, int], target: int, seed: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Passes 2.. of ``rows`` (ids ``#r1``, ``#r2``, ...) up to ``target`` tokens."""
    size = sum(lengths[row["id"]] for row in rows)
    if not 0 < size <= target:
        raise ValueError(f"repeat: pass 1 has {size} tokens for a {target} target")
    groups = group_rows(rows)
    tokens = {
        g: sum(lengths[row["id"]] for row in members) for g, members in groups.items()
    }
    added, passes, total, index = [], [], size, 0
    while total < target:
        index += 1
        suffix = f"#r{index}"
        if target - total >= size:
            picked, kind = list(rows), "full"
        else:
            order = stratified_order(groups, tokens, f"{seed}|r{index}")
            chosen = take_prefix(order, tokens, target - total)
            picked, kind = [row for g in chosen for row in groups[g]], "prefix"
        if not picked:
            break
        pass_tokens = sum(lengths[row["id"]] for row in picked)
        for row in picked:
            lengths[row["id"] + suffix] = lengths[row["id"]]
        added.extend({**row, "id": row["id"] + suffix} for row in picked)
        passes.append(
            {"suffix": suffix, "kind": kind, "rows": len(picked), "tokens": pass_tokens}
        )
        total += pass_tokens
        if kind == "prefix":
            break
    if abs(total - target) > REPEAT_TOLERANCE * target:
        raise ValueError(f"repeat: {total} tokens is not within 0.5% of {target}")
    return added, {"pass1_tokens": size, "passes": passes}


def parse_size(size: str) -> bool:
    if size in ("full", "rho"):
        return True
    kind, _, rest = size.partition("=")
    if kind == "match":
        return bool(rest)
    count, _, rule = rest.partition(":")
    return kind == "tokens" and count.isdigit() and rule == "family-equal"


def parse_mixture(spec: str) -> tuple[str, list[tuple[str, str]]]:
    name, _, parts = spec.partition("=")
    items = []
    for part in parts.split(","):
        arm, _, size = part.partition(":")
        if not name or not arm or not parse_size(size):
            raise ValueError(f"bad mixture spec {spec!r}; use {GRAMMAR}")
        if (arm == "repeat") != size.startswith("match="):
            raise ValueError(f"bad mixture spec {spec!r}; use {GRAMMAR}")
        items.append((arm, size))
    if items[0] != ("base", "full"):
        raise ValueError("every mixture starts with base:full")
    if any(arm == "repeat" for arm, _ in items[:-1]):
        raise ValueError("repeat:match is the last term")
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


def breakdown(rows: list[dict[str, Any]], lengths: dict[str, int]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for field in ("source", "family", "language", "task_type"):
        cells: dict[str, dict[str, int]] = defaultdict(lambda: {"rows": 0, "tokens": 0})
        for row in rows:
            cells[row[field]]["rows"] += 1
            cells[row[field]]["tokens"] += lengths[row["id"]]
        out[field] = {k: dict(v) for k, v in sorted(cells.items())}
    return out


def sample_aho(
    rows: list[dict[str, Any]], count: int, seed: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups = group_rows(rows)
    sizes = {g: len(members) for g, members in groups.items()}
    orders = family_order(groups, seed)
    totals = {f: sum(sizes[g] for g in order) for f, order in orders.items()}
    quotas = waterfill(totals, count)
    chosen = {f: take_at_most(order, sizes, quotas[f]) for f, order in orders.items()}
    kept = sorted(
        (row for picked in chosen.values() for g in picked for row in groups[g]),
        key=lambda row: row["id"],
    )
    report = {
        "count": count,
        "rows": len(kept),
        "by_family": {f: sum(sizes[g] for g in picked) for f, picked in chosen.items()},
        "row_ids_sha256": sha_bytes(
            canonical([row["id"] for row in kept]).encode("utf-8")
        ),
    }
    return kept, report


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
            if arm == "repeat":
                match = size.partition("=")[2]
                if match not in report:
                    raise ValueError(f"{name}: repeat needs an earlier mixture {match}")
                target = report[match]["tokens"]
                added, extra = repeat_passes(
                    rows, lengths, target, f"{seed}|repeat|{name}"
                )
                rows.extend(added)
                parts[arm] = {
                    "size": size,
                    "target_tokens": target,
                    "rows": len(added),
                    "tokens": sum(lengths[row["id"]] for row in added),
                    **extra,
                }
                continue
            if size.startswith("tokens="):
                if arm in ("base", "resample") or arm not in pools:
                    raise ValueError(f"{name}: unsupported part {arm}:{size}")
                target = int(size.partition("=")[2].partition(":")[0])
                picked, groups, extra = family_equal(
                    pools[arm], rows, lengths, target, f"{seed}|{arm}"
                )
                rows.extend(picked)
                parts[arm] = {
                    "size": size,
                    "target_tokens": target,
                    "rows": len(picked),
                    "tokens": sum(lengths[row["id"]] for row in picked),
                    "groups": len(groups),
                    "group_ids_sha256": sha_bytes(canonical(groups).encode("utf-8")),
                    **extra,
                }
                continue
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
        report[name] = {
            "spec": spec,
            "parts": parts,
            **summarize(rows, lengths),
            "breakdown": breakdown(rows, lengths),
        }
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
    parser.add_argument(
        "--aho-sample", action="append", default=[], help="NAME=PATH[,PATH...]:COUNT"
    )
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
    samples = {}
    for spec in args.aho_sample:
        name, _, rest = spec.partition("=")
        paths, _, count = rest.rpartition(":")
        if not name or not paths or not count.isdigit():
            raise SystemExit(
                f"bad --aho-sample {spec!r}; use NAME=PATH[,PATH...]:COUNT"
            )
        samples[name] = (paths.split(","), int(count))
        inputs += samples[name][0]
    if set(samples) & {spec.split("=", 1)[0] for spec in args.aho}:
        raise SystemExit("an --aho-sample name repeats an --aho name")
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
    aho_samples = {}
    for name, (paths, count) in samples.items():
        rows = [row for path in paths for row in load_partition(path, "select")]
        kept, _, admission[f"aho-{name}"] = admit(rows, length_of, args.limit)
        aho_rows[name], aho_samples[name] = sample_aho(
            kept, count, f"{args.seed}|aho|{name}"
        )
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
        "grammar": GRAMMAR,
        "rules": {
            "family_equal": "drop arm groups sharing group_id or input_sha256 with the"
            " mixture so far; water-filled equal integer per-family token shares;"
            " per family the longest seeded prefix at or below the share plus the"
            " next group if that lands closer",
            "repeat": "pass 1 = rows so far; whole passes #r1.. while they fit under"
            " the matched total, then a stratified seeded whole-group prefix by the"
            " same closest rule; |total - matched| <= 0.5%",
            "aho_sample": "water-filled equal per-family row shares, whole groups in"
            " seeded order, never above a share",
        },
        "mixture_specs": args.mixture,
        "aho_samples": aho_samples,
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
