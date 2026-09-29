"""Materialize a 9B Milestone 3 TRAIN partition and its teacher file from a frozen spec.

Recipe rows are joined by id from hash-verified pool files (A0s from the
positional-key pk1 file). Every recipe row must receive exactly one teacher
distribution bound to its input hash and option keys. The recipe may drop whole
pools (``recipe_exclude_pools``) and be cut to ``recipe_budget_tokens`` native
tokens in whole groups, stratified by pool x source x task type x language in
the ``<seed>:recipe`` hash order. Retention replay rows
come from an A7 view: excluded families are dropped, rows whose input (current
or pre-renumbering hash) repeats A0 or a recipe row are dropped, and whole
groups are sampled to a native-token budget, stratified by source x task type x
language in a seed-keyed hash order. Replay rows keep their gold labels and get
no teacher term. The run refuses rows whose source matches a sealed C1
candidate or an ``mlx-diag`` source, rows over ``max_length`` tokens, and any
id / group / input shared with the isolation partitions.

An optional ``dose`` adds teacher-covered rows from named pool files (listed in
pinned ``members`` id files) to a native-token budget, equal per ``family``:
groups sharing a group id, input (current or pre-renumbering hash) or id with
the rows so far or with the ``dedupe`` files are dropped, as are groups with a
row lacking a teacher target or over ``max_tokens``. The budget is water-filled
over families (a family below its equal share gives everything, the rest is
re-split); each family takes the longest prefix of its groups in
``sha256("<seed>:<name>:" + id)`` order (a group keyed by its smallest member
key and counted toward its first family by name) at or below its share, plus
the next group if that lands closer. Dose rows carry their teacher targets into
``teacher.jsonl`` unless ``emit_teacher`` is false.

    python3 -m lux9b.m3_data --spec SPEC --root hf=/hf --root repo=/code \\
        --tokenizer /model --output-dir OUT

Writes ``train.jsonl``, ``teacher.jsonl`` and ``manifest.json`` (inputs, counts,
tokens, dedupe and guard results, output hashes) into a new directory.
"""

from __future__ import annotations

import argparse
import hashlib
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


def recipe_budget(
    rows: list[dict[str, Any]],
    native: dict[str, int],
    pool_of: dict[str, str],
    budget: int,
    seed: str,
    tolerance: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Whole recipe groups to a native-token budget, stratified by pool x source x
    task type x language (a group's stratum is that of its first recipe row)."""
    from v2.dec.build_template_s import group_rows, select_groups

    groups = group_rows(rows)
    keyed = {
        g: [
            {
                "source": f"{pool_of[m[0]['id']]}|{m[0]['source']}",
                "task_type": m[0]["task_type"],
                "language": m[0]["language"],
            }
        ]
        for g, m in groups.items()
    }
    tokens = {g: sum(native[r["id"]] for r in m) for g, m in groups.items()}
    total = sum(tokens.values())
    if budget >= total:
        raise ValueError(f"recipe budget {budget} >= available {total} tokens")
    chosen = set(select_groups(keyed, tokens, budget, seed))
    kept = [r for r in rows if r["group_id"] in chosen]
    realized = sum(tokens[g] for g in chosen)
    if realized > budget * (1 + tolerance):
        raise ValueError(
            f"recipe budget overshoot {realized} > {budget} (+{tolerance})"
        )
    return kept, {
        "budget_tokens": budget,
        "tolerance": tolerance,
        "seed": seed,
        "available_rows": len(rows),
        "available_groups": len(groups),
        "available_native_tokens": total,
        "strata": len({tuple(k[0].values()) for k in keyed.values()}),
        "groups": len(chosen),
        "rows": len(kept),
        "native_tokens": realized,
        "overshoot_tokens": realized - budget,
    }


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


def waterfill(totals: dict[str, int], target: int) -> dict[str, int]:
    """Equal integer shares of ``target``; keys whose total fits a share take it all
    (the remainder goes one token each to the first keys by name)."""
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


def argmax_key(probs: dict[str, float], keys: list[str]) -> str:
    return max(keys, key=lambda k: (probs[k], -keys.index(k)))


def dose_component(
    dose: dict[str, Any],
    seed: str,
    teacher_entries: list[dict[str, str]],
    roots: dict[str, Path],
    inputs: dict[str, str],
    blocked: set[str],
    taken: set[str],
    taken_groups: set[str],
    tokenizer: Path,
    workers: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    """Teacher-covered rows of named pools, family-equal to a native-token budget."""
    members: dict[str, str] = {}
    for entry in dose["members"]:
        for record in read_jsonl(verified(entry, roots, inputs)):
            members.setdefault(record["id"], record["pool"])
    blocked = set(blocked)
    for entry in dose.get("dedupe", []):
        for row in load_partition(verified(entry, roots, inputs), "train"):
            blocked.add(row["input_sha256"])
            renumbering = row["audit_metadata"].get("option_key_renumbering") or {}
            if renumbering.get("original_input_sha256"):
                blocked.add(renumbering["original_input_sha256"])
    counts: Counter = Counter(
        dict.fromkeys(
            (
                "file_rows",
                "not_member",
                "id_in_train",
                "duplicate_input",
                "group_in_train_rows",
                "no_teacher_rows",
                "over_length_rows",
            ),
            0,
        )
    )
    pool_of: dict[str, str] = {}
    candidates: list[dict[str, Any]] = []
    for entry in dose["files"]:
        for row in load_partition(verified(entry, roots, inputs), "train"):
            counts["file_rows"] += 1
            if row["id"] in pool_of:
                raise ValueError(f"{row['id']}: dose files repeat an id")
            pool_of[row["id"]] = entry["pool"]
            if row["id"] not in members:
                counts["not_member"] += 1
                continue
            if members[row["id"]] != entry["pool"]:
                raise ValueError(f"{row['id']}: member pool differs from its file")
            if row["id"] in taken:
                counts["id_in_train"] += 1
                continue
            audit = row["audit_metadata"]
            originals = {
                (audit.get("a7") or {}).get("original_input_sha256"),
                (audit.get("option_key_renumbering") or {}).get(
                    "original_input_sha256"
                ),
            } - {None}
            if row["input_sha256"] in blocked or originals & blocked:
                counts["duplicate_input"] += 1
                continue
            candidates.append(row)
    wanted = {r["id"] for r in candidates}
    teacher: dict[str, dict[str, Any]] = {}
    by_file: dict[str, int] = {}
    for entry in teacher_entries:
        used = 0
        for record in read_jsonl(verified(entry, roots, inputs)):
            if record["id"] not in wanted:
                continue
            previous = teacher.get(record["id"])
            if previous is not None and previous != record:
                raise ValueError(f"{record['id']}: conflicting teacher records")
            if previous is None:
                used += 1
            teacher[record["id"]] = record
        by_file[entry["file"]] = used
    for row in candidates:
        if row["id"] in teacher:
            check_teacher(row, teacher[row["id"]])
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        groups[row["group_id"]].append(row)
    for label, test in (
        ("group_in_train", lambda g, m: g in taken_groups),
        ("no_teacher", lambda g, m: any(r["id"] not in teacher for r in m)),
    ):
        for group_id in [g for g, m in groups.items() if test(g, m)]:
            counts[f"{label}_rows"] += len(groups.pop(group_id))
    pool = [r for m in groups.values() for r in m]
    lengths = dict(
        zip((r["id"] for r in pool), token_lengths(pool, tokenizer, workers))
    )
    for group_id in [
        g
        for g, m in groups.items()
        if any(lengths[r["id"]] > dose["max_tokens"] for r in m)
    ]:
        counts["over_length_rows"] += len(groups.pop(group_id))
    pool = [r for m in groups.values() for r in m]
    if len({r["input_sha256"] for r in pool}) != len(pool):
        raise ValueError("dose pool repeats an input_sha256 across its own rows")

    def key(row_id: str) -> str:
        return hashlib.sha256(f"{seed}:{row_id}".encode()).hexdigest()

    tokens = {g: sum(lengths[r["id"]] for r in m) for g, m in groups.items()}
    orders: dict[str, list[str]] = defaultdict(list)
    for group_id in sorted(groups, key=lambda g: min(key(r["id"]) for r in groups[g])):
        orders[min(r["family"] for r in groups[group_id])].append(group_id)
    totals = {f: sum(tokens[g] for g in order) for f, order in orders.items()}
    budget = dose["budget_tokens"]
    if budget >= sum(totals.values()):
        raise ValueError(f"dose budget {budget} >= available {sum(totals.values())}")
    quotas = waterfill(totals, budget)
    chosen: list[str] = []
    families: dict[str, Any] = {}
    for family in sorted(orders):
        picked = take_prefix(orders[family], tokens, quotas[family])
        chosen.extend(picked)
        families[family] = {
            "available_tokens": totals[family],
            "quota": quotas[family],
            "tokens": sum(tokens[g] for g in picked),
            "groups": len(picked),
            "rows": sum(len(groups[g]) for g in picked),
            "exhausted": len(picked) == len(orders[family]),
        }
    rows = sorted((r for g in chosen for r in groups[g]), key=lambda r: r["id"])
    realized = sum(tokens[g] for g in chosen)
    tolerance = dose.get("tolerance", 0.01)
    if abs(realized - budget) > budget * tolerance:
        raise ValueError(f"dose {realized} tokens outside {budget} +-{tolerance}")

    def tally(field) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"rows": 0, "tokens": 0, "teacher_argmax_gold": 0}
        )
        for r in rows:
            cell = out[field(r)]
            cell["rows"] += 1
            cell["tokens"] += lengths[r["id"]]
            keys = [o["key"] for o in r["options"]]
            probs = teacher[r["id"]]["teacher_probs"]
            cell["teacher_argmax_gold"] += argmax_key(probs, keys) == keys[r["label"]]
        for cell in out.values():
            cell["teacher_agreement"] = round(
                cell["teacher_argmax_gold"] / cell["rows"], 4
            )
        return dict(sorted(out.items()))

    stats = {
        **dict(counts),
        "seed": seed,
        "members": len(members),
        "teacher_by_file": by_file,
        "eligible_rows": len(pool),
        "eligible_groups": len(groups),
        "eligible_tokens": sum(totals.values()),
        "budget_tokens": budget,
        "tolerance": tolerance,
        "max_tokens_limit": dose["max_tokens"],
        "families": families,
        "rows": len(rows),
        "groups": len(chosen),
        "tokens": realized,
        "overshoot_tokens": realized - budget,
        "max_tokens": max((lengths[r["id"]] for r in rows), default=0),
        "by_pool": tally(lambda r: pool_of[r["id"]]),
        "by_row_family": tally(lambda r: r["family"]),
        "by_type": tally(lambda r: r["task_type"]),
        "by_language": tally(lambda r: r["language"]),
        "teacher_argmax_gold": sum(
            c["teacher_argmax_gold"] for c in tally(lambda r: "all").values()
        ),
    }
    return rows, {r["id"]: teacher[r["id"]] for r in rows}, stats


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
    excluded_pools = spec.get("recipe_exclude_pools", [])
    if excluded_pools:
        unknown = sorted(set(excluded_pools) - set(by_pool))
        if unknown:
            raise ValueError(f"recipe_exclude_pools not in the recipe: {unknown}")
        stats["recipe_excluded_pools"] = {
            p: {
                "rows": len(by_pool[p]),
                "native_tokens": sum(e["native"] for e in by_pool[p]),
            }
            for p in sorted(excluded_pools)
        }
        for p in excluded_pools:
            del by_pool[p]
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
    native = {e["id"]: e["native"] for e in recipe}
    pool_of = {e["id"]: e["pool"] for e in recipe}
    if "recipe_budget_tokens" in spec:
        rows, stats["recipe_budget"] = recipe_budget(
            rows,
            native,
            pool_of,
            spec["recipe_budget_tokens"],
            f"{spec['seed']}:recipe",
            spec.get("recipe_budget_tolerance", 0.01),
        )
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
    stats["duplicate_input_rows"] = len(rows) - len(blocked)
    a0_inputs: set[str] = set()
    if "a0_for_dedupe" in spec:
        a0_file = verified(spec["a0_for_dedupe"], roots, inputs)
        for row in load_partition(a0_file, "train"):
            a0_inputs.add(row["input_sha256"])
            renumbering = row["audit_metadata"].get("option_key_renumbering") or {}
            if renumbering.get("original_input_sha256"):
                a0_inputs.add(renumbering["original_input_sha256"])
        stats["rows_sharing_a0_input"] = sum(
            r["input_sha256"] in a0_inputs for r in rows
        )
    elif "replay" in spec or spec.get("extras"):
        raise ValueError("Replay / extras need a0_for_dedupe")
    blocked |= a0_inputs

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
    dose_rows: list[dict[str, Any]] = []
    dose_teacher: dict[str, dict[str, Any]] = {}
    if "dose" in spec:
        dose = spec["dose"]
        dose_rows, dose_teacher, dose_stats = dose_component(
            dose,
            f"{spec['seed']}:{dose['name']}",
            spec["teachers"] + dose.get("teachers", []),
            roots,
            inputs,
            blocked,
            taken,
            {r["group_id"] for r in rows + replay},
            tokenizer,
            workers,
        )
        if not dose.get("emit_teacher", True):
            dose_teacher = {}

    recipe_lengths = token_lengths(rows, tokenizer, workers)
    if max(recipe_lengths) > spec["max_length"]:
        raise ValueError("A recipe row exceeds max_length")
    stats["native_token_mismatches"] = sum(
        native[row["id"]] != length for row, length in zip(rows, recipe_lengths)
    )
    stats["recipe_tokens"] = sum(recipe_lengths)
    stats["recipe_native_tokens"] = sum(native[r["id"]] for r in rows)
    stats["recipe_rows_by_type"] = dict(Counter(r["task_type"] for r in rows))
    by_type: Counter = Counter()
    by_pool_tokens: Counter = Counter()
    by_pool_native: Counter = Counter()
    for r, n in zip(rows, recipe_lengths):
        by_type[r["task_type"]] += n
        by_pool_tokens[pool_of[r["id"]]] += n
        by_pool_native[pool_of[r["id"]]] += native[r["id"]]
    stats["recipe_tokens_by_type"] = dict(by_type)
    stats["recipe_token_share_by_type"] = {
        t: round(n / stats["recipe_tokens"], 4) for t, n in sorted(by_type.items())
    }
    selected_by_pool = Counter(pool_of[r["id"]] for r in rows)
    stats["recipe_selected_by_pool"] = {
        p: {
            "rows": selected_by_pool[p],
            "native_tokens": by_pool_native[p],
            "tokens": by_pool_tokens[p],
        }
        for p in sorted(selected_by_pool)
    }
    stats["recipe_languages"] = len({r["language"] for r in rows})

    train = sorted(rows + replay + dose_rows, key=lambda r: r["id"])
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
    manifest = {
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
    }
    if "dose" in spec:
        teacher.update(dose_teacher)
        manifest["dose"] = dose_stats
        manifest["teacher"].update(
            recipe_rows=len(recipe_ids),
            dose_rows=len(dose_teacher),
            rows=len(recipe_ids) + len(dose_teacher),
        )
        by_type = Counter(stats["recipe_tokens_by_type"])
        by_type.update({t: c["tokens"] for t, c in dose_stats["by_type"].items()})
        total = sum(by_type.values())
        manifest["train_tokens"] = total
        manifest["train_token_share_by_type"] = {
            t: round(n / total, 4) for t, n in sorted(by_type.items())
        }
        manifest["train_languages"] = len({r["language"] for r in train})
    return train, [teacher[i] for i in sorted(teacher)], manifest


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
