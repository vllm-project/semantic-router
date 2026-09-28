"""Template-S training mixtures over hash-pinned data arms.

TRAIN is every row of the base arms plus a token-matched slice of the
treatment arms (experiment matrix v1.1, template S): rho = floor(fraction x
base tokens), capped at the treatment tokens, split across the treatment arms
in proportion to their tokens and filled with whole groups in a fixed hash
order. Tokens are Qwen3-0.6B-Base native tokens of the segmented option prompt
(the data registry's unit), so every testbed trains on the same rows whatever
its own tokenizer.

Template S2 (Milestone 4) adds base-row filters (excluded families, positional
renumbering of construction-order option keys), components with a fixed token
budget, a view-then-equal-shares policy for large corpora (every admissible
member of a named view, then equal token shares per sub-arm, water-filled,
whole groups in hash order), and a matched-token control that resamples the
base (whole copies, then a stratified partial copy), and a published recipe
manifest joined to its arm files by id (`id_manifest`). Rows repeating any base
input (before or after renumbering) or an earlier row, and groups with a row
over `max_row_tokens`, are dropped and counted. S2 mixtures are built once
(`python3 -m v2.06b.mixture`) and trained from the hash-pinned materialized file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .common import digest, file_sha256, write_json

UNIT = "qwen3-0.6b-base-native-segmented"


def arm_rows(entry: dict[str, Any]) -> list[dict[str, Any]]:
    from training.model.data import load_partition

    path = Path(entry["path"])
    if file_sha256(path) != entry["sha256"]:
        raise ValueError(f"{entry['arm']}: arm file differs from its frozen hash")
    rows = load_partition(path, "train")
    if len(rows) != entry["rows"]:
        raise ValueError(
            f"{entry['arm']}: expected {entry['rows']} rows, found {len(rows)}"
        )
    return rows


def token_counter(tokenizer_path: str) -> Any:
    from transformers import AutoTokenizer

    from training.model.decision_model import encode

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    return lambda row: len(encode(row, tokenizer, 1 << 30)["ids"])


def select_groups(
    rows: list[dict[str, Any]], tokens: list[int], share: int, seed: str, arm: str
) -> list[int]:
    """Whole groups in sha256(seed, arm, group) order, skipping any that would overflow."""
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault(row["group_id"], []).append(index)
    order = sorted(
        groups,
        key=lambda g: (hashlib.sha256(f"{seed}\0{arm}\0{g}".encode()).hexdigest(), g),
    )
    chosen, used = [], 0
    for group in order:
        size = sum(tokens[i] for i in groups[group])
        if used + size <= share:
            chosen.extend(groups[group])
            used += size
    return sorted(chosen)


def resample_groups(
    rows: list[dict[str, Any]], tokens: list[int], rho: int, seed: str
) -> list[int]:
    """Template-S control: whole groups of the base, stratified by source x type x language."""
    strata: dict[tuple[str, str, str], list[int]] = {}
    for index, row in enumerate(rows):
        strata.setdefault(
            (row["source"], row["task_type"], row["language"]), []
        ).append(index)
    total = sum(tokens)
    chosen: list[int] = []
    for key in sorted(strata):
        members = strata[key]
        share = rho * sum(tokens[i] for i in members) // total
        picked = select_groups(
            [rows[i] for i in members],
            [tokens[i] for i in members],
            share,
            seed,
            "|".join(key),
        )
        chosen.extend(members[i] for i in picked)
    return sorted(chosen)


def duplicate(row: dict[str, Any], copy: int | None = None) -> dict[str, Any]:
    """A resampled copy: new id and group, same content; teacher targets follow the original."""
    suffix = "#resample" if copy is None else f"#resample{copy}"
    return {
        **row,
        "id": row["id"] + suffix,
        "group_id": row["group_id"] + suffix,
        "teacher_source_id": row["id"],
    }


def build(spec: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if spec["template"] != "S" or spec["unit"] != UNIT:
        raise ValueError("Only template S in the Qwen3 native unit is implemented")
    count = token_counter(spec["tokenizer"])
    report: dict[str, Any] = {
        "template": "S",
        "unit": UNIT,
        "seed": spec["seed"],
        "arms": {},
    }
    train: list[dict[str, Any]] = []
    base_counts: list[int] = []
    for entry in spec["base"]:
        rows = arm_rows(entry)
        counts = [count(row) for row in rows]
        base_counts.extend(counts)
        train.extend(rows)
        report["arms"][entry["arm"]] = {
            "role": "base",
            "rows": len(rows),
            "tokens": sum(counts),
        }
    base_rows = list(train)
    base_tokens = sum(base_counts)
    treatment = []
    for entry in spec["treatment"]:
        if entry.get("resample_of") == "base":
            treatment.append((entry["arm"], base_rows, base_counts, True))
            continue
        rows = arm_rows(entry)
        treatment.append((entry["arm"], rows, [count(row) for row in rows], False))
    available = sum(sum(tokens) for _, _, tokens, _ in treatment)
    rho = min(int(spec["fraction_of_base"] * base_tokens), available)
    for arm, rows, tokens, resample in treatment:
        share = rho * sum(tokens) // available
        if resample:
            chosen = resample_groups(rows, tokens, share, f"{spec['seed']}\0{arm}")
            picked = [duplicate(rows[i]) for i in chosen]
        else:
            chosen = select_groups(rows, tokens, share, spec["seed"], arm)
            picked = [rows[i] for i in chosen]
        train.extend(picked)
        levels: dict[str, int] = {}
        for row in picked:
            if row["task_type"] == "score":
                key = str(len(row["options"]))
                levels[key] = levels.get(key, 0) + 1
        report["arms"][arm] = {
            "role": "resample" if resample else "treatment",
            "rows_available": len(rows),
            "tokens_available": sum(tokens),
            "share_tokens": share,
            "rows": len(picked),
            "tokens": sum(tokens[i] for i in chosen),
            "groups": len({row["group_id"] for row in picked}),
            "score_level_counts": dict(
                sorted(levels.items(), key=lambda kv: int(kv[0]))
            ),
        }
    ids = [row["id"] for row in train]
    if len(set(ids)) != len(ids):
        raise ValueError("Mixture rows repeat an id across arms")
    report.update(
        {
            "base_tokens": base_tokens,
            "rho_tokens": rho,
            "train_rows": len(train),
            "train_tokens": base_tokens
            + sum(a["tokens"] for a in report["arms"].values() if a["role"] != "base"),
            "train_ids_sha256": digest(sorted(ids)),
        }
    )
    for key, value in spec.get("expected", {}).items():
        if report[key] != value:
            raise ValueError(f"Mixture {key} is {report[key]}, frozen as {value}")
    return train, report


def profile(rows: list[dict[str, Any]], tokens: list[int]) -> dict[str, Any]:
    """Counts only: rows, tokens, groups, types, languages, Score levels, top families."""
    levels = Counter(str(len(r["options"])) for r in rows if r["task_type"] == "score")
    return {
        "rows": len(rows),
        "tokens": sum(tokens),
        "max_row_tokens": max(tokens) if tokens else 0,
        "groups": len({r["group_id"] for r in rows}),
        "task_types": dict(sorted(Counter(r["task_type"] for r in rows).items())),
        "languages": dict(sorted(Counter(r["language"] for r in rows).items())),
        "score_level_counts": dict(sorted(levels.items(), key=lambda kv: int(kv[0]))),
        "families_top12": dict(Counter(r["family"] for r in rows).most_common(12)),
    }


def filtered_base(
    entry: dict[str, Any], count: Any, cap: int
) -> tuple[list[dict[str, Any]], list[int], set[str], dict[str, Any]]:
    """Base rows minus excluded families, optionally with positional option keys."""
    from v2.common.option_keys import renumber_rows

    rows = arm_rows(entry)
    seen = {row["input_sha256"] for row in rows}
    excluded = set(entry.get("exclude_families", []))
    kept = [row for row in rows if row["family"] not in excluded]
    renumbered = 0
    if entry.get("renumber_option_keys"):
        kept, summary = renumber_rows(kept)
        renumbered = summary["renumbered"]
        seen |= {row["input_sha256"] for row in kept}
    tokens = [count(row) for row in kept]
    if any(t > cap for t in tokens):
        raise ValueError(f"{entry['arm']}: a base row exceeds {cap} tokens")
    return (
        kept,
        tokens,
        seen,
        {
            "role": "base",
            "rows_in_file": len(rows),
            "excluded_family_rows": len(rows) - len(kept),
            "renumbered_rows": renumbered,
            **profile(kept, tokens),
        },
    )


def budget_component(
    comp: dict[str, Any], count: Any, cap: int, seen: set[str]
) -> tuple[list[dict[str, Any]], list[int], dict[str, Any]]:
    """A fixed token budget split over the arms by their tokens (template S slices)."""
    counted = []
    for entry in comp["arms"]:
        rows = arm_rows(entry)
        counted.append((entry["arm"], rows, [count(row) for row in rows]))
    available = sum(sum(tokens) for _, _, tokens in counted)
    budget = min(comp["tokens"], available)
    picked: list[dict[str, Any]] = []
    picked_tokens: list[int] = []
    arms = {}
    for arm, rows, tokens in counted:
        share = budget * sum(tokens) // available
        chosen = select_groups(rows, tokens, share, comp["seed"], arm)
        if any(rows[i]["input_sha256"] in seen for i in chosen):
            raise ValueError(f"{arm}: budget slice repeats a base input")
        if any(tokens[i] > cap for i in chosen):
            raise ValueError(f"{arm}: budget slice has a row over {cap} tokens")
        picked.extend(rows[i] for i in chosen)
        picked_tokens.extend(tokens[i] for i in chosen)
        arms[arm] = {
            "rows_available": len(rows),
            "tokens_available": sum(tokens),
            "share_tokens": share,
            "rows": len(chosen),
            "tokens": sum(tokens[i] for i in chosen),
        }
    return (
        picked,
        picked_tokens,
        {
            "role": "treatment",
            "policy": "token_budget",
            "budget_tokens": budget,
            "arms": arms,
            **profile(picked, picked_tokens),
        },
    )


def view_component(
    comp: dict[str, Any], count: Any, cap: int, seen: set[str], target: int, seed: str
) -> tuple[list[dict[str, Any]], list[int], dict[str, Any]]:
    """Every admissible view member, then equal token shares per arm (water-filled)."""
    excluded = set(comp.get("exclude_families", []))
    drops: Counter[str] = Counter()
    pools: dict[str, list[dict[str, Any]]] = {}
    for entry in comp["arms"]:
        rows = arm_rows(entry)
        pools[entry["arm"]] = [row for row in rows if row["family"] not in excluded]
        drops[f"{entry['arm']}:excluded_family"] += len(rows) - len(pools[entry["arm"]])
    view_path = Path(comp["view"]["path"])
    if file_sha256(view_path) != comp["view"]["sha256"]:
        raise ValueError("View file differs from its frozen hash")
    members = [
        m["id"]
        for m in json.loads(view_path.read_text())["members"]
        if m["part"] == "train"
    ]
    by_id = {row["id"]: (arm, row) for arm, rows in pools.items() for row in rows}
    used = set(seen)
    taken: set[str] = set()
    picked: list[dict[str, Any]] = []
    picked_tokens: list[int] = []
    per_arm: Counter[str] = Counter()
    per_arm_tokens: Counter[str] = Counter()

    cache: dict[str, int] = {}

    def tokens_of(row: dict[str, Any]) -> int:
        if row["id"] not in cache:
            cache[row["id"]] = count(row)
        return cache[row["id"]]

    def admit(arm: str, row: dict[str, Any]) -> int | None:
        if row["input_sha256"] in used:
            drops[f"{arm}:repeats_earlier_input"] += 1
            return None
        tokens = tokens_of(row)
        if tokens > cap:
            drops[f"{arm}:over_cap"] += 1
            return None
        return tokens

    def take(arm: str, row: dict[str, Any], tokens: int, phase: str) -> None:
        used.add(row["input_sha256"])
        taken.add(row["id"])
        picked.append(row)
        picked_tokens.append(tokens)
        per_arm[f"{arm}:{phase}"] += 1
        per_arm_tokens[f"{arm}:{phase}"] += tokens

    resolved: set[str] = set()
    for member in members:
        if member not in by_id:
            drops["view:excluded_or_absent"] += 1
            continue
        resolved.add(member)
        arm, row = by_id[member]
        tokens = admit(arm, row)
        if tokens is not None:
            take(arm, row, tokens, "view")
    view_tokens = sum(picked_tokens)
    orders: dict[str, list[list[dict[str, Any]]]] = {}
    for arm, rows in pools.items():
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            if row["id"] not in resolved:
                groups.setdefault(row["group_id"], []).append(row)
        orders[arm] = [
            groups[g]
            for g in sorted(
                groups,
                key=lambda g: (
                    hashlib.sha256(f"{seed}\0{arm}\0{g}".encode()).hexdigest(),
                    g,
                ),
            )
        ]
    pointer = dict.fromkeys(orders, 0)
    remaining = max(0, target - view_tokens)
    active = [arm for arm in pools if orders[arm]]
    rounds = 0
    while remaining > 0 and active:
        share = remaining // len(active)
        if share == 0:
            break
        rounds += 1
        progress = False
        for arm in list(active):
            budget = share
            order = orders[arm]
            while pointer[arm] < len(order):
                group = order[pointer[arm]]
                fresh: list[dict[str, Any]] = []
                inputs: set[str] = set()
                for row in group:
                    if (
                        row["input_sha256"] not in used
                        and row["input_sha256"] not in inputs
                    ):
                        inputs.add(row["input_sha256"])
                        fresh.append(row)
                sizes = [tokens_of(row) for row in fresh]
                if not fresh or max(sizes) > cap:
                    drops[f"{arm}:repeats_earlier_input"] += len(group) - len(fresh)
                    drops[f"{arm}:over_cap_group_rows"] += len(fresh)
                    pointer[arm] += 1
                    continue
                if sum(sizes) > budget:
                    break
                drops[f"{arm}:repeats_earlier_input"] += len(group) - len(fresh)
                for row, tokens in zip(fresh, sizes):
                    take(arm, row, tokens, "fill")
                budget -= sum(sizes)
                pointer[arm] += 1
                progress = True
            remaining -= share - budget
            if pointer[arm] >= len(order):
                active.remove(arm)
        if not progress:
            break
    return (
        picked,
        picked_tokens,
        {
            "role": "treatment",
            "policy": "view_then_equal_shares",
            "target_tokens": target,
            "view_tokens": view_tokens,
            "fill_rounds": rounds,
            "rows_by_arm_phase": dict(sorted(per_arm.items())),
            "tokens_by_arm_phase": dict(sorted(per_arm_tokens.items())),
            "dropped": {k: v for k, v in sorted(drops.items()) if v},
            **profile(picked, picked_tokens),
        },
    )


def manifest_component(
    comp: dict[str, Any], count: Any, cap: int, seen: set[str]
) -> tuple[list[dict[str, Any]], list[int], dict[str, Any]]:
    """Rows of a published recipe manifest (`{id, pool, ...}` per line), joined by id.

    Pools in `skip_pools` come from the base instead. Whole groups with a row
    over the cap are dropped; rows repeating an earlier input are dropped.
    """
    path = Path(comp["manifest"]["path"])
    if file_sha256(path) != comp["manifest"]["sha256"]:
        raise ValueError("Recipe manifest differs from its frozen hash")
    entries = [json.loads(line) for line in path.read_text().splitlines()]
    if len(entries) != comp["manifest"]["rows"]:
        raise ValueError("Recipe manifest row count differs")
    skip = set(comp.get("skip_pools", []))
    by_pool: dict[str, dict[str, dict[str, Any]]] = {}
    for pool, arm_entries in comp["pools"].items():
        by_pool[pool] = {row["id"]: row for e in arm_entries for row in arm_rows(e)}
    wanted = [e for e in entries if e["pool"] not in skip]
    missing = [e["id"] for e in wanted if e["id"] not in by_pool.get(e["pool"], {})]
    if missing:
        raise ValueError(f"{len(missing)} manifest ids are absent from their pools")
    groups: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for e in wanted:
        row = by_pool[e["pool"]][e["id"]]
        groups.setdefault(row["group_id"], []).append((e["pool"], row))
    drops: Counter[str] = Counter()
    used = set(seen)
    picked: list[dict[str, Any]] = []
    picked_tokens: list[int] = []
    per_pool: Counter[str] = Counter()
    per_pool_tokens: Counter[str] = Counter()
    for members in groups.values():
        sizes = [count(row) for _, row in members]
        if max(sizes) > cap:
            for pool, _ in members:
                drops[f"{pool}:over_cap_group_rows"] += 1
            continue
        for (pool, row), size in zip(members, sizes):
            if row["input_sha256"] in used:
                drops[f"{pool}:repeats_earlier_input"] += 1
                continue
            used.add(row["input_sha256"])
            picked.append(row)
            picked_tokens.append(size)
            per_pool[pool] += 1
            per_pool_tokens[pool] += size
    return (
        picked,
        picked_tokens,
        {
            "role": "treatment",
            "policy": "id_manifest",
            "manifest_rows": len(entries),
            "skipped_pool_rows": dict(
                sorted(Counter(e["pool"] for e in entries if e["pool"] in skip).items())
            ),
            "rows_by_pool": dict(sorted(per_pool.items())),
            "tokens_by_pool": dict(sorted(per_pool_tokens.items())),
            "dropped": {k: v for k, v in sorted(drops.items()) if v},
            **profile(picked, picked_tokens),
        },
    )


def resample_component(
    comp: dict[str, Any],
    base_rows: list[dict[str, Any]],
    base_tokens: list[int],
    seed: str,
) -> tuple[list[dict[str, Any]], list[int], dict[str, Any]]:
    """Matched-token control: whole copies of the base, then a stratified partial copy."""
    total = sum(base_tokens)
    copies = comp["tokens"] // total
    rows: list[dict[str, Any]] = []
    tokens: list[int] = []
    for copy in range(1, copies + 1):
        rows.extend(duplicate(row, copy) for row in base_rows)
        tokens.extend(base_tokens)
    rest = comp["tokens"] - copies * total
    chosen = resample_groups(
        base_rows, base_tokens, rest, f"{seed}\0resample{copies + 1}"
    )
    rows.extend(duplicate(base_rows[i], copies + 1) for i in chosen)
    tokens.extend(base_tokens[i] for i in chosen)
    return (
        rows,
        tokens,
        {
            "role": "resample",
            "policy": "resample_base",
            "target_tokens": comp["tokens"],
            "whole_copies": copies,
            "partial_copy_rows": len(chosen),
            **profile(rows, tokens),
        },
    )


def build_s2(
    spec: dict[str, Any], count: Any = None
) -> tuple[list[dict[str, Any]], list[int], dict[str, Any]]:
    if spec["template"] != "S2" or spec["unit"] != UNIT:
        raise ValueError("Template S2 in the Qwen3 native unit expected")
    count = count or token_counter(spec["tokenizer"])
    cap = spec["max_row_tokens"]
    report: dict[str, Any] = {
        "template": "S2",
        "unit": UNIT,
        "seed": spec["seed"],
        "max_row_tokens": cap,
        "components": {},
    }
    train: list[dict[str, Any]] = []
    tokens: list[int] = []
    seen: set[str] = set()
    base_rows: list[dict[str, Any]] = []
    base_tokens: list[int] = []
    for entry in spec["base"]:
        rows, counts, inputs, part = filtered_base(entry, count, cap)
        base_rows.extend(rows)
        base_tokens.extend(counts)
        seen |= inputs
        report["components"][entry["arm"]] = part
    train.extend(base_rows)
    tokens.extend(base_tokens)
    for comp in spec["components"]:
        if comp["policy"] == "token_budget":
            rows, counts, part = budget_component(comp, count, cap, seen)
        elif comp["policy"] == "view_then_equal_shares":
            target = comp.get(
                "tokens", comp.get("fill_to_total_tokens", 0) - sum(tokens)
            )
            rows, counts, part = view_component(
                comp, count, cap, seen, target, spec["seed"]
            )
        elif comp["policy"] == "id_manifest":
            rows, counts, part = manifest_component(comp, count, cap, seen)
        elif comp["policy"] == "resample_base":
            rows, counts, part = resample_component(
                comp, base_rows, base_tokens, spec["seed"]
            )
        else:
            raise ValueError(f"Unknown component policy {comp['policy']}")
        seen |= {row["input_sha256"] for row in rows}
        train.extend(rows)
        tokens.extend(counts)
        report["components"][comp["name"]] = part
    ids = [row["id"] for row in train]
    if len(set(ids)) != len(ids):
        raise ValueError("Mixture rows repeat an id")
    report.update(
        {
            "train_rows": len(train),
            "train_tokens": sum(tokens),
            "train_ids_sha256": digest(sorted(ids)),
            "profile": profile(train, tokens),
        }
    )
    for key, value in spec.get("expected", {}).items():
        if report[key] != value:
            raise ValueError(f"Mixture {key} is {report[key]}, frozen as {value}")
    return train, tokens, report


def load_materialized(
    entry: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rows of a hash-pinned materialized S2 mixture, checked against its frozen identity."""
    from training.model.data import load_partition

    path = Path(entry["path"])
    if file_sha256(path) != entry["sha256"]:
        raise ValueError("Materialized mixture differs from its frozen hash")
    rows = load_partition(path, "train")
    report = {
        "template": "materialized",
        "sha256": entry["sha256"],
        "build_spec": entry["build_spec"],
        "build_report_sha256": entry["build_report_sha256"],
        "train_rows": len(rows),
        "train_ids_sha256": digest(sorted(row["id"] for row in rows)),
    }
    for key, value in entry.get("expected", {}).items():
        if report[key] != value:
            raise ValueError(f"Mixture {key} is {report[key]}, frozen as {value}")
    return rows, report


def main() -> None:
    from training.model.data import canonical

    parser = argparse.ArgumentParser(description="Materialize one template-S2 mixture")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise FileExistsError("refusing to overwrite a materialized mixture")
    spec = json.loads(args.spec.read_text())
    rows, _, report = build_s2(spec)
    pending = args.output.with_name(args.output.name + ".pending")
    with pending.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(canonical(row) + "\n")
    pending.replace(args.output)
    report["spec_sha256"] = file_sha256(args.spec)
    report["rows_file_sha256"] = file_sha256(args.output)
    write_json(args.report, report, exclusive=True)
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "train_rows",
                    "train_tokens",
                    "train_ids_sha256",
                    "rows_file_sha256",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
