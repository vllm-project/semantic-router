"""Build HT-DEV v1: convert snapshots to a pool, export scan inputs, select the panel.

    python3 -m v2.eval.htdev.build pool --sources-dir <dir> --output-dir <private dir> [--source KEY ...]
    python3 -m v2.eval.htdev.build export-scan --pool <pool.jsonl> --output-dir <private dir>
    python3 -m v2.eval.htdev.build select --pool <pool.jsonl> --admission <ADMISSION.json> \
        --exclusions <ids file> --output-dir <private dir> [--config config.json]

`pool` runs every converter over its pinned snapshot (file sha256s checked against
pins.json) and writes the full admissible pool (`pool.jsonl`: id = `task|source item id`,
state, question, gold, split, group, provenance) plus a count-only `POOL.json`.

`export-scan` writes the pool's states in the shapes the isolation scans read:
`overlap-protected.jsonl` (`v2.eval.sealed.overlap scan --protected`; hit ids equal pool
ids), `overlap-corpus.jsonl` (id + state rows, for `--corpus` when the pool is scanned as
a corpus), `embed-candidates.jsonl` (`v2/data/embed_scan.py --candidates`) and
`embed-inventory.json` (the pool as a `--protected-inventory` entry).

`select` needs an admission file (`{"admitted": [source keys]}`; a missing file is
refused) and an exclusion file (pool ids dropped by item-level scans, one per line or
`{"id": ...}` JSON lines; may be empty). Per task, in `sha256("ht-dev/1:<task>:<item id>")`
order after the split preference (test, validation; train only when the floor is not
reached otherwise): drop excluded ids and repeated states, pick the C1 balancing
stratum (`gold_length_rank` for per-item options whose gold is the longest too often,
`length` when the C1 length baseline fails, else the gold), fill the strata as evenly
as the pool allows under the group cap, up to the cap. A Choice task whose length-only
baseline macro-F1 exceeds chance + 0.10 is re-drawn once with length-stratified balance
(recorded). A primary below the floor or not admitted is replaced by its backups in
order; with fewer than `min_tasks` tasks the build stops (STOPPED.json, no panel).
Writes ht-dev.prompts.jsonl (gold-free), ht-dev.gold.jsonl and MANIFEST.json, never
overwriting.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from v2.eval.htdev import fetch
from v2.eval.sealed.build import (
    BINS,
    FOLDS,
    OPTION_LENGTH_GAIN,
    bin_of,
    fold,
    gold_record,
    length_baseline,
    quintiles,
    write_new,
)
from v2.eval.sealed.schema import LONG_INPUT_CHARS, input_chars, normalized, validate
from v2.eval.sealed.score import macro_f1

SCHEMA = "dev2-htdev-build/1"
CONFIG = Path(__file__).with_name("config.json")
PANEL = "ht-dev"
SPLIT_RANK = {"test": 0, "all": 0, "validation": 1, "train": 2}
REFILL_PASSES = 5


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")


def private_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, 0o700)


def code_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def load_source(key: str):
    module = importlib.import_module(f"v2.eval.htdev.sources.{key}")
    if module.SPEC.key != key:
        raise ValueError(f"{key}: SPEC.key is {module.SPEC.key!r}")
    return module


def verify_snapshot(key: str, root: Path, pins: dict[str, Any]) -> dict[str, str]:
    spec = pins["sources"][key]
    observed = {}
    for relative, expected in sorted(spec["files"].items()):
        digest = fetch.sha_file(root / relative)
        if digest != expected:
            raise ValueError(f"{key}/{relative}: sha256 differs from pins.json")
        observed[relative] = digest
    return observed


def pool_row(candidate) -> dict[str, Any]:
    return {
        "id": f"{candidate.task}|{candidate.source_item_id}",
        "task": candidate.task,
        "source": candidate.source,
        "split": candidate.split,
        "source_item_id": candidate.source_item_id,
        "group_id": candidate.group_id,
        "language": candidate.language,
        "balance_label": candidate.balance_label,
        "state": candidate.state,
        "question": candidate.question,
        "gold": candidate.gold,
        "overlap_texts": candidate.overlap_texts,
        "option_texts": candidate.option_texts,
        "provenance": candidate.provenance,
    }


def pool(args: argparse.Namespace) -> int:
    pins = fetch.load_pins(args.pins)
    keys = args.source or sorted(
        k for k, v in pins["sources"].items() if not v.get("unavailable")
    )
    rows, sources, seen = [], {}, set()
    for key in keys:
        root = args.sources_dir / key
        if not root.is_dir():
            sources[key] = {"status": "missing"}
            continue
        files = verify_snapshot(key, root, pins)
        module = load_source(key)
        invalid: Counter = Counter()
        counts: dict[str, Counter] = defaultdict(Counter)
        for candidate in module.candidates(root):
            problems = validate(candidate, module.SPEC)
            row = pool_row(candidate)
            if row["id"] in seen:
                problems.append("duplicate id")
            if problems:
                invalid.update(problems)
                continue
            seen.add(row["id"])
            rows.append(row)
            counts[row["task"]][f"{row['split']}|{row['balance_label']}"] += 1
        sources[key] = {
            "status": "ok",
            "revision": pins["sources"][key]["revision"],
            "files_sha256": files,
            "invalid": dict(invalid),
            "tasks": {t: dict(sorted(c.items())) for t, c in counts.items()},
        }
    private_dir(args.output_dir)
    data = jsonl_bytes(rows)
    write_new(args.output_dir / "pool.jsonl", data)
    receipt = {
        "schema": SCHEMA + ":pool",
        "pool_sha256": sha_bytes(data),
        "items": len(rows),
        "sources": sources,
        "code_commit": code_commit(),
    }
    write_new(
        args.output_dir / "POOL.json",
        (json.dumps(receipt, indent=1, sort_keys=True) + "\n").encode(),
    )
    print(json.dumps({"items": len(rows), "pool_sha256": receipt["pool_sha256"]}))
    return 0


def export_scan(args: argparse.Namespace) -> int:
    rows = read_jsonl(args.pool)
    protected = [
        {
            "source": row["source"],
            "task": row["task"],
            "source_item_id": row["source_item_id"],
            "state": row["state"],
            "overlap_texts": row["overlap_texts"],
        }
        for row in rows
    ]
    corpus = [{"id": row["id"], "state": row["state"]} for row in rows]
    embed = [
        {"id": row["id"], "group_id": row["group_id"], "state": row["state"]}
        for row in rows
    ]
    private_dir(args.output_dir)
    digests = {}
    for name, data in (
        ("overlap-protected.jsonl", jsonl_bytes(protected)),
        ("overlap-corpus.jsonl", jsonl_bytes(corpus)),
        ("embed-candidates.jsonl", jsonl_bytes(embed)),
    ):
        write_new(args.output_dir / name, data)
        digests[name] = sha_bytes(data)
    inventory = [
        {
            "role": "ht-dev-pool",
            "path": str(args.output_dir / "embed-candidates.jsonl"),
            "sha256": digests["embed-candidates.jsonl"],
        }
    ]
    write_new(
        args.output_dir / "embed-inventory.json",
        (json.dumps(inventory, indent=1) + "\n").encode(),
    )
    print(json.dumps({"items": len(rows), **digests}))
    return 0


def order_key(prefix: str, row: dict[str, Any]) -> tuple[int, str]:
    digest = hashlib.sha256(
        f"{prefix}:{row['task']}:{row['source_item_id']}".encode()
    ).hexdigest()
    return SPLIT_RANK.get(row["split"], 2), digest


def gold_key(row: dict[str, Any]) -> str:
    return json.dumps(row["gold"])


def option_rank(row: dict[str, Any]) -> int | None:
    """Rank of the gold option's text length (0 = longest) for per-item options."""
    texts = row.get("option_texts")
    if not texts or row["question"]["type"] != "choice":
        return None
    keys = list(row["question"]["criteria"])
    gold = len(texts[keys.index(row["gold"])])
    return sum(len(text) > gold for text in texts)


def option_gain(rows: list[dict[str, Any]]) -> float:
    """Rate the gold is the unique longest per-item option, minus chance."""
    hits = expected = 0.0
    counted = 0
    for row in rows:
        texts = row.get("option_texts")
        if not texts:
            continue
        counted += 1
        lengths = [len(text) for text in texts]
        if lengths.count(max(lengths)) != 1:
            continue
        expected += 1.0 / len(lengths)
        hits += option_rank(row) == 0
    return (hits - expected) / counted if counted else 0.0


def length_predictions(rows: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """Length-only predictions: majority gold per input-length quintile (5-fold CV by
    group) and, for per-item options, the longest option."""
    out: dict[str, list[Any]] = {}
    if len(rows) >= 2 * FOLDS:
        lengths = [row["_chars"] for row in rows]
        edges = quintiles(lengths)
        golds = [gold_key(row) for row in rows]
        folds = [fold(row["group_id"]) for row in rows]
        guesses: list[Any] = [None] * len(rows)
        for k in range(FOLDS):
            train = [i for i, f in enumerate(folds) if f != k]
            if not train:
                continue
            overall = Counter(golds[i] for i in train).most_common(1)[0][0]
            cells: dict[int, Counter] = defaultdict(Counter)
            for i in train:
                cells[bin_of(lengths[i], edges)][golds[i]] += 1
            for i, f in enumerate(folds):
                if f == k:
                    cell = cells.get(bin_of(lengths[i], edges))
                    guesses[i] = json.loads(
                        cell.most_common(1)[0][0] if cell else overall
                    )
        out["length_quintile"] = guesses
    if rows and all(row.get("option_texts") for row in rows):
        longest = []
        for row in rows:
            keys = list(row["question"]["criteria"])
            lengths = [len(text) for text in row["option_texts"]]
            longest.append(keys[lengths.index(max(lengths))])
        out["longest_option"] = longest
    return out


def length_only(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores = {
        name: macro_f1([(row["gold"], p) for row, p in zip(rows, preds)])
        for name, preds in length_predictions(rows).items()
    }
    return {"macro_f1": max(scores.values()) if scores else None, "by_rule": scores}


def option_count(row: dict[str, Any]) -> int:
    question = row["question"]
    return 2 if question["type"] == "noul" else len(question["criteria"])


def waterfill(available: dict[Any, int], total: int) -> dict[Any, int]:
    """Equal shares per stratum, capped by availability, leftovers to the others."""
    quota = {key: 0 for key in available}
    remaining = total
    open_keys = sorted((k for k in available if available[k] > 0), key=str)
    while remaining > 0 and open_keys:
        share = max(1, remaining // len(open_keys))
        progressed = False
        for key in list(open_keys):
            if remaining <= 0:
                break
            add = min(share, available[key] - quota[key], remaining)
            if add > 0:
                quota[key] += add
                remaining -= add
                progressed = True
            if quota[key] >= available[key]:
                open_keys.remove(key)
        if not progressed:
            break
    return quota


def strata(
    rows: list[dict[str, Any]], balance: str
) -> tuple[Callable[[dict[str, Any]], Any], Callable[[dict[Any, int], int], dict]]:
    if balance == "length":
        edges = quintiles([row["_chars"] for row in rows])
        golds = sorted({gold_key(row) for row in rows})

        def of(row):
            return bin_of(row["_chars"], edges), gold_key(row)

        def quotas(available, cap):
            per_cell = max(1, cap // (BINS * len(golds)))
            bins = sorted({b for b, _ in available})
            quota = {}
            for b in bins:
                smallest = min(available.get((b, g), 0) for g in golds)
                for g in golds:
                    quota[(b, g)] = min(per_cell, smallest)
            return quota

        return of, quotas
    if balance == "gold_length_rank":

        def quotas(available, cap):
            smallest = min(available.values())
            return {k: min(max(1, cap // len(available)), smallest) for k in available}

        return option_rank, quotas
    return gold_key, waterfill


def take(
    rows: list[dict[str, Any]], cap: int, group_cap: int, balance: str
) -> list[dict[str, Any]]:
    """rows are in selection order; fill strata evenly under the group cap."""
    of, quotas = strata(rows, balance)
    taken: list[dict[str, Any]] = []
    chosen: set[int] = set()
    per_group: Counter = Counter()
    per_stratum: Counter = Counter()
    for _ in range(REFILL_PASSES):
        open_rows = [
            (i, row)
            for i, row in enumerate(rows)
            if i not in chosen and per_group[row["group_id"]] < group_cap
        ]
        available = Counter(per_stratum)
        available.update(of(row) for _, row in open_rows)
        if not available:
            break
        quota = quotas(dict(available), cap)
        before = len(taken)
        for i, row in open_rows:
            if len(taken) >= cap:
                break
            key = of(row)
            if per_stratum[key] >= quota.get(key, 0):
                continue
            if per_group[row["group_id"]] >= group_cap:
                continue
            taken.append(row)
            chosen.add(i)
            per_stratum[key] += 1
            per_group[row["group_id"]] += 1
        if len(taken) == before or len(taken) >= cap or balance != "gold":
            break
    order = {id(row): i for i, row in enumerate(rows)}
    return sorted(taken, key=lambda row: order[id(row)])


def choose_balance(rows: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    gain = option_gain(rows)
    c1 = length_baseline(rows)
    checks = {"option_length_gain": gain, "c1_length_gain": c1["gain"]}
    if gain >= OPTION_LENGTH_GAIN:
        return "gold_length_rank", checks
    if c1["fails"]:
        return "length", checks
    return "gold", checks


def select_task(
    rows: list[dict[str, Any]], entry: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    cap, floor = config["cap"], config["floor"]
    group_cap = entry.get("group_cap", 1)
    early = [row for row in rows if SPLIT_RANK.get(row["split"], 2) <= 1]
    stages = [("test+validation", early)]
    if config.get("train_only_below_floor", True) and len(early) < len(rows):
        stages.append(("test+validation+train", rows))
    result: dict[str, Any] = {}
    for stage, pool_rows in stages:
        if not pool_rows:
            continue
        qtype = pool_rows[0]["question"]["type"]
        if qtype == "choice":
            balance, checks = choose_balance(pool_rows)
        else:
            balance, checks = "gold", {}
        chosen = take(pool_rows, cap, group_cap, balance)
        gate = []
        baseline = length_only(chosen)
        chance = 1.0 / option_count(pool_rows[0])
        limit = chance + config["choice_length_margin"]
        if (
            qtype == "choice"
            and baseline["macro_f1"] is not None
            and baseline["macro_f1"] > limit
        ):
            first = baseline
            redraw = (
                "gold_length_rank"
                if option_rank(pool_rows[0]) is not None
                and (
                    first["by_rule"].get("longest_option", 0)
                    >= first["by_rule"].get("length_quintile", 0)
                )
                else "length"
            )
            chosen = take(pool_rows, cap, group_cap, redraw)
            baseline = length_only(chosen)
            gate.append(
                {
                    "rule": "length-only macro-F1 > chance + 0.10",
                    "before": first,
                    "redraw_balance": redraw,
                    "after": baseline,
                    "passes_after": baseline["macro_f1"] is None
                    or baseline["macro_f1"] <= limit,
                }
            )
            balance = redraw
        result = {
            "stage": stage,
            "balance": balance,
            "pool_checks": checks,
            "chosen": chosen,
            "length_only": baseline,
            "chance": chance,
            "gate": gate,
        }
        if len(chosen) >= floor:
            break
    return result


def counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups = Counter(row["group_id"] for row in rows)
    return {
        "items": len(rows),
        "by_split": dict(sorted(Counter(row["split"] for row in rows).items())),
        "by_label": dict(sorted(Counter(gold_key(row) for row in rows).items())),
        "by_split_label": dict(
            sorted(Counter(f"{r['split']}|{gold_key(r)}" for r in rows).items())
        ),
        "groups": len(groups),
        "max_per_group": max(groups.values()) if groups else 0,
    }


def read_exclusions(path: Path) -> set[str]:
    ids = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        ids.add(json.loads(line)["id"] if line.startswith("{") else line)
    return ids


def panel_id(prefix: str, row: dict[str, Any]) -> str:
    digest = hashlib.sha256(
        f"{prefix}:id:{row['task']}:{row['source_item_id']}".encode()
    ).hexdigest()
    return f"htdev-{digest[:16]}"


def select(args: argparse.Namespace) -> int:
    if not args.admission.is_file():
        raise SystemExit(f"refusing to select: no admission file at {args.admission}")
    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes)
    admission_bytes = args.admission.read_bytes()
    admitted = set(json.loads(admission_bytes)["admitted"])
    exclusions = read_exclusions(args.exclusions)
    pool_bytes = args.pool.read_bytes()
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    dropped: dict[str, Counter] = defaultdict(Counter)
    for line in pool_bytes.decode("utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["id"] in exclusions:
            dropped[row["task"]]["excluded"] += 1
            continue
        row["_chars"] = input_chars(row["state"], row["question"])
        by_task[row["task"]].append(row)
    prefix = config["seed_prefix"]
    tasks: dict[str, Any] = {}
    slots = []
    selected: list[dict[str, Any]] = []
    for slot in config["slots"]:
        record = {"kind": slot["kind"], "tried": [], "task": None}
        for role, entry in [("primary", slot["primary"])] + [
            (f"backup{i + 1}", b) for i, b in enumerate(slot["backups"])
        ]:
            task, source = entry["task"], entry["source"]
            if source not in admitted:
                record["tried"].append({"task": task, "result": "not admitted"})
                continue
            rows = sorted(
                (r for r in by_task.get(task, []) if r["source"] == source),
                key=lambda r: order_key(prefix, r),
            )
            seen, unique = set(), []
            for row in rows:
                key = normalized(
                    json.dumps(row["state"], ensure_ascii=False, sort_keys=True)
                )
                if key in seen:
                    dropped[task]["duplicate_state"] += 1
                    continue
                seen.add(key)
                unique.append(row)
            outcome = select_task(unique, entry, config) if unique else {}
            chosen = outcome.get("chosen", [])
            classes = len({gold_key(row) for row in chosen})
            if len(chosen) < config["floor"] or classes < 2:
                record["tried"].append(
                    {"task": task, "result": "below floor", "selected": len(chosen)}
                )
                tasks[task] = {
                    "role": role,
                    "source": source,
                    "status": "below floor",
                    "eligible": counts(unique),
                    "selected_if_used": counts(chosen),
                    "stage": outcome.get("stage"),
                    "exclusions": dict(dropped.get(task, {})),
                }
                continue
            record["tried"].append({"task": task, "result": "selected"})
            record["task"] = task
            for row in chosen:
                row["_id"] = panel_id(prefix, row)
            tasks[task] = {
                "role": role,
                "source": source,
                "status": "selected",
                "type": chosen[0]["question"]["type"],
                "group_cap": entry.get("group_cap", 1),
                "stage": outcome["stage"],
                "balance": outcome["balance"],
                "pool_checks": outcome["pool_checks"],
                "length_only": outcome["length_only"],
                "chance": outcome["chance"],
                "length_gate": outcome["gate"],
                "eligible": counts(unique),
                "selected": counts(chosen),
                "long_input": sum(r["_chars"] >= LONG_INPUT_CHARS for r in chosen),
                "exclusions": dict(dropped.get(task, {})),
            }
            selected.extend(chosen)
            break
        slots.append(record)
    admitted_tasks = [s["task"] for s in slots if s["task"]]
    private_dir(args.output_dir)
    base = {
        "schema": SCHEMA,
        "panel": PANEL,
        "config_sha256": sha_bytes(config_bytes),
        "pool_sha256": sha_bytes(pool_bytes),
        "admission_sha256": sha_bytes(admission_bytes),
        "admitted_sources": sorted(admitted),
        "exclusions_sha256": fetch.sha_file(args.exclusions),
        "exclusion_ids": len(exclusions),
        "code_commit": code_commit(),
        "slots": slots,
        "tasks": tasks,
    }
    if len(admitted_tasks) < config["min_tasks"]:
        base["status"] = (
            f"stopped: {len(admitted_tasks)} tasks < {config['min_tasks']} required"
        )
        write_new(
            args.output_dir / "STOPPED.json",
            (json.dumps(base, indent=1, sort_keys=True) + "\n").encode(),
        )
        print(json.dumps({"status": base["status"]}))
        return 3
    selected.sort(key=lambda row: row["_id"])
    if len({row["_id"] for row in selected}) != len(selected):
        raise ValueError("panel id collision")
    prompts = [
        {"id": r["_id"], "state": r["state"], "questions": {"decision": r["question"]}}
        for r in selected
    ]
    gold = [
        {
            "id": r["_id"],
            "task": r["task"],
            "source": r["source"],
            "split": r["split"],
            "source_item_id": r["source_item_id"],
            "group_id": r["group_id"],
            "language": r["language"],
            "input_chars": r["_chars"],
            "long": r["_chars"] >= LONG_INPUT_CHARS,
            "provenance": r["provenance"],
            "questions": {"decision": r["question"]},
            "gold": {"decision": gold_record(r)},
        }
        for r in selected
    ]
    digests = {}
    for name, rows in (
        (f"{PANEL}.prompts.jsonl", prompts),
        (f"{PANEL}.gold.jsonl", gold),
    ):
        data = jsonl_bytes(rows)
        write_new(args.output_dir / name, data)
        digests[name] = sha_bytes(data)
    manifest = {
        **base,
        "status": "built",
        "config": config,
        "files_sha256": digests,
        "items": len(selected),
        "task_count": len(admitted_tasks),
        "by_type": dict(Counter(r["question"]["type"] for r in selected)),
        "tasks_by_type": dict(Counter(tasks[t]["type"] for t in admitted_tasks)),
    }
    data = (json.dumps(manifest, indent=1, sort_keys=True) + "\n").encode()
    write_new(args.output_dir / "MANIFEST.json", data)
    print(
        json.dumps(
            {
                "items": len(selected),
                "tasks": len(admitted_tasks),
                "manifest_sha256": sha_bytes(data),
                **digests,
            }
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("pool")
    one.add_argument("--sources-dir", type=Path, required=True)
    one.add_argument("--output-dir", type=Path, required=True)
    one.add_argument("--source", action="append")
    one.add_argument("--pins", type=Path, default=fetch.PINS)
    two = commands.add_parser("export-scan")
    two.add_argument("--pool", type=Path, required=True)
    two.add_argument("--output-dir", type=Path, required=True)
    three = commands.add_parser("select")
    three.add_argument("--pool", type=Path, required=True)
    three.add_argument("--admission", type=Path, required=True)
    three.add_argument("--exclusions", type=Path, required=True)
    three.add_argument("--output-dir", type=Path, required=True)
    three.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args(argv)
    return {"pool": pool, "export-scan": export_scan, "select": select}[args.command](
        args
    )


if __name__ == "__main__":
    raise SystemExit(main())
