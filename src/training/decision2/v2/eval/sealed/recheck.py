"""JevArena-C1 content recheck: new training files against the C1 v1.2 items (custodian only).

    python3 -m v2.eval.sealed.recheck verify --spec SPEC --output VERIFY.json
    python3 -m v2.eval.sealed.recheck items --prompts PROMPTS --candidates-dir DIR \
        --build-manifest MANIFEST --retired RETIRED --expect-scored N \
        --output ITEMS --protected-output PROTECTED --receipt ITEMS.json
    python3 -m v2.eval.sealed.recheck scan --spec SPEC --items ITEMS --workers N \
        --output SCAN.json --private SCAN-PRIVATE.json
    python3 -m v2.eval.sealed.recheck judge --spec SPEC --items ITEMS --scan SCAN.json \
        --scan-private SCAN-PRIVATE.json --registered-hits HITS --registered-receipt R.json \
        [--protected-hits HITS --protected-receipt R.json] --output VERDICT.json \
        --quarantine-dir DIR

`verify` checks every training file of the spec against its pinned SHA-256 and row count.

`items` identifies the selected items without the gold or the salt. Every decrypted prompt is
matched to its build-5 candidate by its state and question (canonical JSON); the candidate files
must have the SHA-256 that the build-5 manifest records, and the gold field is dropped. The v1.2
retired list (pinned) marks the items that are not scored; the scored count must equal
--expect-scored. It writes the items (private) and the same rows in the overlap scanner's
protected shape (state and overlap_texts, as the build screened them).

`scan` applies the rules of the data track's Index-row guard (`v2.data.ib1.index_guard`, G0)
to the string leaves of each item's state, overlap texts and question (instructions and option or
level texts), and of each training row's state, instructions and option descriptions:

    E   raw equality of a leaf of >= 20 characters;
    N1  equality of a whole normalized leaf of >= 6 tokens;
    N2  equality of a normalized sentence or whole leaf of >= 8 tokens;
    G   any shared word 13-gram of normalized tokens (within one leaf).

Boilerplate, reported apart and never quarantining:

    B1  a training row's instructions or option description that occurs verbatim in at least
        50 rows of its family in the same file (a fixed builder template);
    B2  an item unit (any rule) that occurs in at least 50 items of the item's own task (the
        task's fixed instructions and option or level texts).

Planted controls: for 200 seeded scored items, the longest state leaf with a non-template
13-gram is copied into a synthetic training row exactly and perturbed (case swapped, commas and
semicolons dropped, whitespace doubled, a prefix added); every copy must hit its own item outside
the boilerplate (the perturbed one through N2 or G). Controls are built in memory only.

`judge` decides each training file: an item set exposure is a scored item that a non-boilerplate
G0 hit or a non-CLEAN verdict of the registered scanner (`v2.eval.sealed.overlap`, run over the
protected-shape items) links to the file. PASS when there is none; otherwise QUARANTINE, and the
groups of every linked row are listed (privately). An arm's exposure is the union over the files
it trained on, minus the families it left out. The protected-row scan (all splits of the C1
sources) is disclosed by verdict counts only.

Public outputs (VERIFY, the items receipt, SCAN, VERDICT) hold counts, family and task names and
hashes only, never text or ids. Private outputs (items, hits, quarantine lists) stay in the
custodial directory.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import multiprocessing
import os
import random
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from v2.eval.sealed.build import read_jsonl, write_new
from v2.eval.sealed.overlap import string_leaves
from v2.eval.sealed.score import (
    POSTKEY_ITEM_SET,
    POSTKEY_RETIRED_SHA256,
    RETIRED_SCHEMA,
)
from v2.eval.sealed.score import sha_file as sha256_file

SPEC_SCHEMA = "dev2-c1-recheck-spec/1"
SCHEMAS = {
    "verify": "dev2-c1-recheck-verify/1",
    "items": "dev2-c1-recheck-items/1",
    "scan": "dev2-c1-recheck-scan/1",
    "private": "dev2-c1-recheck-scan-private/1",
    "verdict": "dev2-c1-recheck-verdict/1",
}
PROMPTS_SHA256 = "0b29686f60c980f3fbc8a03b88537fc0bf90ee967afa67d4fe0c958b1bfde16a"
BUILD_MANIFEST_SHA256 = (
    "71297ecb5458a182c68e88f88b52eb37f8f45770d02e82aa93f39d04dcbedd86"
)
PROTECTED_SHA256 = "36797f509bd96c3cb703df37cc48114c9bbdf0d2241802e56262139f4bef0a1a"
ITEM_SET = POSTKEY_ITEM_SET
SCORED_ITEMS = 2840
RULES = ("E", "N1", "N2", "G")
TEMPLATE_ROWS = 50
TEMPLATE_ITEMS = 50
CONTROLS = 200
CONTROL_SEED = "c1-recheck-controls-v1"
CONTROL_PREFIX = "Note: this is a control item. "
SHA = re.compile(r"^[0-9a-f]{64}$")
KEY = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_STATE: dict[str, Any] = {}


def sha256_text(lines: Iterable[str]) -> str:
    return hashlib.sha256("".join(f"{line}\n" for line in lines).encode()).hexdigest()


def write_json(path: Path, value: Any) -> str:
    data = (
        json.dumps(value, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode()
    write_new(path, data)
    return hashlib.sha256(data).hexdigest()


def tally(values: Iterable[Any]) -> dict[str, int]:
    return dict(sorted(Counter(str(v) for v in values).items()))


# --------------------------------------------------------------------------- spec


def load_spec(path: Path) -> dict[str, Any]:
    spec = json.loads(path.read_text(encoding="utf-8"))
    problems = []
    if spec.get("schema") != SPEC_SCHEMA:
        problems.append(f"schema is not {SPEC_SCHEMA}")
    if not KEY.match(str(spec.get("name", ""))):
        problems.append("name must be lower-case letters, digits and dashes")
    if spec.get("item_set") != ITEM_SET:
        problems.append(f"item_set must be {ITEM_SET}")
    keys = set()
    for entry in spec.get("datasets") or []:
        key = str(entry.get("key", ""))
        if not KEY.match(key) or key in keys:
            problems.append(f"dataset key {key!r} is malformed or repeated")
        keys.add(key)
        if not str(entry.get("path", "")).startswith("/"):
            problems.append(f"{key}: path must be absolute")
        if not SHA.match(str(entry.get("sha256", ""))):
            problems.append(f"{key}: sha256 must be 64 hex digits")
        if not isinstance(entry.get("rows"), int) or entry["rows"] <= 0:
            problems.append(f"{key}: rows must be a positive integer")
        if not entry.get("label") or not entry.get("source"):
            problems.append(f"{key}: label and source are required")
    if not keys:
        problems.append("no datasets")
    for arm in spec.get("arms") or []:
        name = f"{arm.get('track')} {arm.get('arm')}"
        if not arm.get("track") or not arm.get("arm") or not arm.get("uses"):
            problems.append(f"arm {name}: track, arm and uses are required")
        for use in arm.get("uses") or []:
            if use.get("dataset") not in keys:
                problems.append(f"arm {name}: unknown dataset {use.get('dataset')!r}")
            if not isinstance(use.get("exclude_families", []), list):
                problems.append(f"arm {name}: exclude_families must be a list")
    if problems:
        raise ValueError("; ".join(problems))
    return spec


def verify(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec)
    out: dict[str, Any] = {
        "schema": SCHEMAS["verify"],
        "spec": spec["name"],
        "spec_sha256": sha256_file(args.spec),
        "datasets": {},
    }
    failed = []
    for entry in spec["datasets"]:
        path, cell = Path(entry["path"]), {"path": entry["path"]}
        if not path.is_file():
            cell["problem"] = "missing"
            failed.append(entry["key"])
        else:
            cell["sha256"] = sha256_file(path)
            rows, groups, families, broken = 0, set(), Counter(), 0
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    rows += 1
                    try:
                        row = json.loads(line)
                        groups.add(row["group_id"])
                        families[row.get("family")] += 1
                        broken += "id" not in row or "state" not in row
                    except (ValueError, KeyError, TypeError):
                        broken += 1
            cell.update(
                rows=rows, groups=len(groups), families=tally(families.elements())
            )
            if broken:
                cell["problem"] = f"{broken} rows lack id, group_id or state"
            elif cell["sha256"] != entry["sha256"]:
                cell["problem"] = "SHA-256 differs from the spec"
            elif rows != entry["rows"]:
                cell["problem"] = "row count differs from the spec"
            if "problem" in cell:
                failed.append(entry["key"])
        out["datasets"][entry["key"]] = cell
    out["ok"] = not failed
    write_json(args.output, out)
    print(json.dumps({"ok": out["ok"], "failed": failed}))
    return 0 if out["ok"] else 1


# --------------------------------------------------------------------------- items


def candidate_key(state: Any, question: Any) -> str:
    return json.dumps(
        {"question": question, "state": state},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def item_id(row: dict[str, Any]) -> str:
    return f"{row['task']}|{row['source_item_id']}"


def items(args: argparse.Namespace) -> int:
    pinned = {
        args.prompts: PROMPTS_SHA256,
        args.build_manifest: BUILD_MANIFEST_SHA256,
        args.retired: POSTKEY_RETIRED_SHA256,
    }
    for path, expected in pinned.items():
        if sha256_file(path) != expected:
            raise ValueError(f"{path}: SHA-256 differs from {expected[:12]}")
    manifest = json.loads(args.build_manifest.read_text(encoding="utf-8"))
    pool: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source, expected in sorted(manifest["candidates_sha256"].items()):
        path = args.candidates_dir / f"{source}.jsonl"
        if sha256_file(path) != expected:
            raise ValueError(f"{path}: SHA-256 differs from the build-5 manifest")
        for row in read_jsonl(path):
            pool[candidate_key(row["state"], row["question"])].append(row)
    retired_list = json.loads(args.retired.read_text(encoding="utf-8"))
    if retired_list.get("schema") != RETIRED_SCHEMA:
        raise ValueError(f"the retired list is not {RETIRED_SCHEMA}")
    retired = set(retired_list["candidates"])
    out, missing, ambiguous = [], 0, 0
    for prompt in read_jsonl(args.prompts):
        matches = pool.get(
            candidate_key(prompt["state"], prompt["questions"]["decision"])
        )
        if not matches:
            missing += 1
            continue
        if len({row["task"] for row in matches}) > 1:
            raise ValueError("a prompt matches candidates of two tasks")
        ambiguous += len({item_id(row) for row in matches}) > 1
        first = matches[0]
        out.append(
            {
                "id": prompt["id"],
                "task": first["task"],
                "source": first["source"],
                "source_item_id": first["source_item_id"],
                "group_id": first["group_id"],
                "language": first["language"],
                "scored": any(item_id(row) not in retired for row in matches),
                "state": prompt["state"],
                "question": prompt["questions"]["decision"],
                "overlap_texts": first.get("overlap_texts") or [],
            }
        )
    if missing:
        raise ValueError(f"{missing} prompts match no build-5 candidate")
    if len({item_id(row) for row in out}) < len(out):
        raise ValueError("two prompts map to the same candidate")
    scored = sum(row["scored"] for row in out)
    if scored != args.expect_scored:
        raise ValueError(f"{scored} scored items, expected {args.expect_scored}")
    lines = [json.dumps(row, ensure_ascii=False, sort_keys=True) for row in out]
    write_new(args.output, "".join(f"{line}\n" for line in lines).encode())
    shaped = [
        json.dumps(
            {
                key: row[key]
                for key in (
                    "task",
                    "source_item_id",
                    "source",
                    "state",
                    "overlap_texts",
                )
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        for row in out
    ]
    write_new(args.protected_output, "".join(f"{line}\n" for line in shaped).encode())
    by_task: dict[str, Counter] = defaultdict(Counter)
    groups: dict[str, set[str]] = defaultdict(set)
    for row in out:
        cell = by_task[row["task"]]
        cell["items"] += 1
        cell["scored" if row["scored"] else "retired"] += 1
        groups[row["task"]].add(row["group_id"])
    receipt = {
        "schema": SCHEMAS["items"],
        "item_set": ITEM_SET,
        "prompts_sha256": PROMPTS_SHA256,
        "build_manifest_sha256": BUILD_MANIFEST_SHA256,
        "candidates_sha256": dict(sorted(manifest["candidates_sha256"].items())),
        "retired_sha256": POSTKEY_RETIRED_SHA256,
        "items": len(out),
        "scored": scored,
        "retired": len(out) - scored,
        "ambiguous": ambiguous,
        "by_task": {
            task: {**dict(cell), "groups": len(groups[task])}
            for task, cell in sorted(by_task.items())
        },
        "items_sha256": sha256_text(lines),
        "protected_shape_sha256": sha256_text(shaped),
    }
    write_json(args.receipt, receipt)
    print(
        json.dumps({k: receipt[k] for k in ("items", "scored", "retired", "ambiguous")})
    )
    return 0


# --------------------------------------------------------------------------- scan


def _unit_hashes() -> Any:
    from v2.data.ib1.index_guard import unit_hashes

    return unit_hashes


def leaves_of(value: Any) -> list[str]:
    return [text for text in string_leaves(value) if text.strip()]


def item_leaves(item: dict[str, Any]) -> list[str]:
    question = item["question"]
    criteria = question.get("criteria")
    texts = [
        *leaves_of(item["state"]),
        *leaves_of(item.get("overlap_texts") or []),
        *leaves_of(question.get("instructions")),
        *leaves_of(list(criteria.values()) if isinstance(criteria, dict) else criteria),
    ]
    return list(dict.fromkeys(texts))


def template_eligible(row: dict[str, Any]) -> list[str]:
    """The texts of a training row that a builder may share across rows."""
    texts = leaves_of(row.get("instructions"))
    for option in row.get("options") or []:
        if isinstance(option, dict):
            texts.extend(leaves_of(option.get("description")))
    for question in (row.get("questions") or {}).values():
        if isinstance(question, dict):
            texts.extend(leaves_of(question.get("instructions")))
            criteria = question.get("criteria")
            texts.extend(
                leaves_of(
                    list(criteria.values()) if isinstance(criteria, dict) else criteria
                )
            )
    return list(dict.fromkeys(texts))


def row_leaves(
    row: dict[str, Any], templates: set[tuple[str, str]]
) -> list[tuple[str, bool]]:
    """(leaf, is a B1 template) for the state, instructions and option texts of a row."""
    family = str(row.get("family"))
    out = [(text, False) for text in dict.fromkeys(leaves_of(row.get("state")))]
    out.extend((text, (family, text) in templates) for text in template_eligible(row))
    return out


def build_index(rows: list[dict[str, Any]]) -> dict[str, Any]:
    unit_hashes = _unit_hashes()
    index: dict[str, dict[int, Any]] = {rule: {} for rule in RULES}
    for number, item in enumerate(rows):
        units: dict[str, set[int]] = {rule: set() for rule in RULES}
        for leaf in item_leaves(item):
            for rule, values in unit_hashes(leaf).items():
                units[rule] |= values
        for rule in RULES:
            for value in units[rule]:
                index[rule].setdefault(value, []).append(number)
    tasks = [item["task"] for item in rows]
    template: dict[str, dict[int, frozenset[str]]] = {rule: {} for rule in RULES}
    for rule in RULES:
        for value, members in index[rule].items():
            if len(members) >= TEMPLATE_ITEMS:
                counts = Counter(tasks[m] for m in members)
                own = frozenset(t for t, n in counts.items() if n >= TEMPLATE_ITEMS)
                if own:
                    template[rule][value] = own
            index[rule][value] = members[0] if len(members) == 1 else tuple(members)
    return {
        "index": index,
        "template": template,
        "tasks": tasks,
        "units": {rule: len(index[rule]) for rule in RULES},
        "template_units": {rule: len(template[rule]) for rule in RULES},
    }


def match(leaves: list[tuple[str, bool]]) -> dict[int, dict[str, set[str]]]:
    """Per matched item: rules hit by data leaves outside B2, by B1 leaves and by B2 units."""
    unit_hashes = _STATE["unit_hashes"]
    index, template, tasks = _STATE["index"], _STATE["template"], _STATE["tasks"]
    hits: dict[int, dict[str, set[str]]] = {}
    for text, b1 in leaves:
        for rule, values in unit_hashes(text).items():
            table, marks = index[rule], template[rule]
            for value in values:
                members = table.get(value)
                if members is None:
                    continue
                own = marks.get(value, ())
                for member in (members,) if type(members) is int else members:
                    cell = hits.setdefault(
                        member, {"data": set(), "b1": set(), "b2": set()}
                    )
                    if b1:
                        cell["b1"].add(rule)
                    elif tasks[member] in own:
                        cell["b2"].add(rule)
                    else:
                        cell["data"].add(rule)
    return hits


def _scan_chunk(task: tuple[str, int, int]) -> list[dict[str, Any]]:
    key, start, stop = task
    templates, lines = _STATE["b1"][key], _STATE["lines"][key]
    out = []
    for number in range(start, stop):
        if not lines[number].strip():
            continue
        row = json.loads(lines[number])
        hits = match(row_leaves(row, templates))
        if hits:
            out.append(
                {
                    "row": number,
                    "id": row["id"],
                    "group_id": row["group_id"],
                    "family": row.get("family"),
                    "hits": {
                        str(member): {k: sorted(v) for k, v in cell.items() if v}
                        for member, cell in sorted(hits.items())
                    },
                }
            )
    return out


def perturb(text: str) -> str:
    """The G0 control perturbation of one leaf (index_guard.controls)."""
    out = " ".join(text.split()).swapcase().replace(",", " ").replace(";", " ")
    return CONTROL_PREFIX + re.sub(r"\s", "  ", out)


def control_text(number: int, item: dict[str, Any]) -> str | None:
    """The item's longest state leaf with a 13-gram outside the boilerplate, if any."""
    from v2.data.ib1.index_guard import GRAM, tokens

    unit_hashes = _STATE["unit_hashes"]
    marks, task = _STATE["template"]["G"], item["task"]
    for text in sorted(leaves_of(item["state"]), key=lambda t: (-len(t), t)):
        if len(tokens(text)) < GRAM:
            continue
        if any(task not in marks.get(v, ()) for v in unit_hashes(text)["G"]):
            return text
    return None


def controls(rows: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = []
    for number, item in enumerate(rows):
        if item["scored"]:
            text = control_text(number, item)
            if text is not None:
                eligible.append(
                    (
                        hashlib.sha256(
                            f"{CONTROL_SEED}|{item['id']}".encode()
                        ).hexdigest(),
                        number,
                        text,
                    )
                )
    picked = sorted(eligible)[:CONTROLS]
    exact = perturbed = 0
    for _, number, text in picked:
        exact += bool(match([(text, False)]).get(number, {}).get("data"))
        found = match([(perturb(text), False)]).get(number, {}).get("data", set())
        perturbed += bool(found & {"N2", "G"})
    return {
        "eligible": len(eligible),
        "planted": len(picked),
        "exact_flagged": exact,
        "perturbed_flagged": perturbed,
        "pass": len(picked) == CONTROLS and exact == perturbed == CONTROLS,
    }


def b1_templates(path: Path) -> tuple[set[tuple[str, str]], dict[str, Any], list[str]]:
    """B1 template strings, counts, and every physical line (rows are 0-based line numbers,
    as the overlap scanner reports them)."""
    counts: Counter = Counter()
    rows, groups, families = 0, set(), Counter()
    with path.open(encoding="utf-8", newline="") as stream:
        lines = stream.read().split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    for line in lines:
        if line.strip():
            row = json.loads(line)
            rows += 1
            groups.add(row["group_id"])
            family = str(row.get("family"))
            families[family] += 1
            counts.update((family, text) for text in template_eligible(row))
    templates = {key for key, count in counts.items() if count >= TEMPLATE_ROWS}
    stats = {
        "rows": rows,
        "groups": len(groups),
        "families": dict(sorted(families.items())),
        "b1_template_strings": len(templates),
    }
    return templates, stats, lines


def summarize(
    found: list[dict[str, Any]], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Count-only summary of one training file's matched rows."""
    scored = {str(n) for n, item in enumerate(rows) if item["scored"]}
    data_rows, data_groups, b_rows = set(), set(), Counter()
    by_rule, by_family = Counter(), Counter()
    items_hit, retired_hit, boiler_items = set(), set(), set()
    for record in found:
        data_scored = {
            m for m, cell in record["hits"].items() if cell.get("data") and m in scored
        }
        data_retired = {
            m
            for m, cell in record["hits"].items()
            if cell.get("data") and m not in scored
        }
        retired_hit |= data_retired
        if data_scored:
            data_rows.add(record["row"])
            data_groups.add(record["group_id"])
            by_family[str(record["family"])] += 1
            items_hit |= data_scored
            for rule in {r for m in data_scored for r in record["hits"][m]["data"]}:
                by_rule[rule] += 1
            continue
        kinds = {k for m, cell in record["hits"].items() if m in scored for k in cell}
        if kinds:
            boiler_items |= {m for m in record["hits"] if m in scored}
            b_rows[
                "b1" if kinds == {"b1"} else "b2" if kinds == {"b2"} else "b1_and_b2"
            ] += 1
        elif data_retired:
            b_rows["retired_items_only"] += 1
    hit_rows = [rows[int(m)] for m in items_hit]
    return {
        "rows_matched": len(found),
        "exposed": {
            "rows": len(data_rows),
            "groups": len(data_groups),
            "rows_by_rule": dict(sorted(by_rule.items())),
            "rows_by_family": dict(sorted(by_family.items())),
            "scored_items": len(items_hit),
            "scored_items_by_source": tally(item["source"] for item in hit_rows),
            "scored_items_by_task": tally(item["task"] for item in hit_rows),
            "c1_groups": len({(item["source"], item["group_id"]) for item in hit_rows}),
            "c1_groups_by_source": tally(
                source
                for source, _ in {
                    (item["source"], item["group_id"]) for item in hit_rows
                }
            ),
        },
        "boilerplate_only": {
            "rows": sum(v for k, v in b_rows.items() if k != "retired_items_only"),
            "rows_by_kind": {
                k: v for k, v in sorted(b_rows.items()) if k != "retired_items_only"
            },
            "scored_items": len(boiler_items),
        },
        "retired_items": {
            "items": len(retired_hit),
            "rows_only_on_retired": b_rows["retired_items_only"],
        },
    }


def scan(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec)
    rows = read_jsonl(args.items)
    built = build_index(rows)
    _STATE.update(built, unit_hashes=_unit_hashes(), b1={}, lines={})
    control = controls(rows)
    files: dict[str, Any] = {}
    tasks: list[tuple[str, int, int]] = []
    step = max(1, args.chunk_rows)
    for entry in spec["datasets"]:
        path = Path(entry["path"])
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"{path}: SHA-256 differs from the spec")
        templates, stats, lines = b1_templates(path)
        if stats["rows"] != entry["rows"]:
            raise ValueError(
                f"{path}: {stats['rows']} rows, the spec says {entry['rows']}"
            )
        _STATE["b1"][entry["key"]] = templates
        _STATE["lines"][entry["key"]] = lines
        files[entry["key"]] = {"sha256": entry["sha256"], **stats}
        tasks.extend(
            (entry["key"], start, min(start + step, len(lines)))
            for start in range(0, len(lines), step)
        )
    found: dict[str, list[dict[str, Any]]] = {key: [] for key in files}
    gc.collect()
    gc.freeze()
    try:
        with multiprocessing.get_context("fork").Pool(max(1, args.workers)) as pool:
            for (key, _, _), part in zip(tasks, pool.imap(_scan_chunk, tasks)):
                found[key].extend(part)
    finally:
        gc.unfreeze()
    public = {
        "schema": SCHEMAS["scan"],
        "spec": spec["name"],
        "spec_sha256": sha256_file(args.spec),
        "items_sha256": sha256_file(args.items),
        "rules": {
            "E": "raw equality of a leaf of >= 20 characters",
            "N1": "equality of a whole normalized leaf of >= 6 tokens",
            "N2": "equality of a normalized sentence or whole leaf of >= 8 tokens",
            "G": "a shared word 13-gram of normalized tokens within one leaf",
            "normalization": "NFKC, casefold, \\w+ tokens (v2.data.ib1.index_guard)",
        },
        "boilerplate": {
            "B1": f"a training instructions or option text in >= {TEMPLATE_ROWS} rows of its family in the file",
            "B2": f"an item unit in >= {TEMPLATE_ITEMS} items of the item's own task",
        },
        "items": {
            "items": len(rows),
            "scored": sum(item["scored"] for item in rows),
            "units": built["units"],
            "template_units": built["template_units"],
        },
        "controls": control,
        "datasets": {
            key: {**files[key], **summarize(found[key], rows)} for key in files
        },
    }
    private = {
        "schema": SCHEMAS["private"],
        "spec": spec["name"],
        "items_sha256": public["items_sha256"],
        "datasets": {key: found[key] for key in files},
    }
    write_json(args.private, private)
    write_json(args.output, public)
    print(
        json.dumps(
            {
                "controls_pass": control["pass"],
                "exposed_items": {
                    k: v["exposed"]["scored_items"]
                    for k, v in public["datasets"].items()
                },
            }
        )
    )
    return 0 if control["pass"] else 1


# --------------------------------------------------------------------------- judge


def registered_hits(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def line_rows(path: Path, wanted: set[int]) -> dict[int, dict[str, Any]]:
    """The rows at these 0-based physical line numbers."""
    out = {}
    if not wanted:
        return out
    with path.open(encoding="utf-8", newline="") as stream:
        for number, line in enumerate(stream.read().split("\n")):
            if number in wanted:
                row = json.loads(line)
                out[number] = {"group_id": row["group_id"], "family": row.get("family")}
    if len(out) < len(wanted):
        raise ValueError(f"{path}: a scanner row is not a training row")
    return out


def reasons(entry: dict[str, Any]) -> list[str]:
    """Why the registered scanner flagged an item against one file (overlap.verdict's terms)."""
    out = []
    if entry.get("exact") is not None:
        out.append(
            "exact_8plus_tokens"
            if entry["exact_tokens"] >= 8
            else "exact_under_8_tokens"
        )
    if entry["containment"] >= 0.5:
        out.append("containment_0.5plus")
    elif entry["containment"] >= 0.2:
        out.append("containment_0.2_to_0.5")
    return out


def protected_disclosure(
    hits: Path, receipt: Path, labels: list[str]
) -> dict[str, Any]:
    raw = json.loads(receipt.read_text(encoding="utf-8"))
    if raw["protected"]["sha256"] != PROTECTED_SHA256:
        raise ValueError("the protected-row scan did not use the pinned protected rows")
    per = {label: defaultdict(Counter) for label in labels}
    for hit in registered_hits(hits):
        for label, entry in (hit.get("labels") or {}).items():
            if label in per and entry["verdict"] != "CLEAN":
                per[label][hit["source"]][entry["verdict"]] += 1
    return {
        "protected_sha256": PROTECTED_SHA256,
        "protected_rows": raw["protected"]["candidates"],
        "receipt_sha256": sha256_file(receipt),
        "hits_sha256": sha256_file(hits),
        "by_dataset": {
            label: {
                "non_clean_rows": sum(sum(c.values()) for c in per[label].values()),
                "by_source": {
                    s: dict(sorted(c.items())) for s, c in sorted(per[label].items())
                },
            }
            for label in labels
        },
    }


def judge(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec)
    rows = read_jsonl(args.items)
    by_key = {item_id(item): n for n, item in enumerate(rows)}
    scan_public = json.loads(args.scan.read_text(encoding="utf-8"))
    scan_private = json.loads(args.scan_private.read_text(encoding="utf-8"))
    if not scan_public["controls"]["pass"]:
        raise ValueError("the planted controls did not pass")
    items_sha = sha256_file(args.items)
    if (
        scan_public["items_sha256"] != items_sha
        or scan_private["items_sha256"] != items_sha
    ):
        raise ValueError("the scan was not run on these items")
    receipt = json.loads(args.registered_receipt.read_text(encoding="utf-8"))
    if receipt["protected"]["sha256"] != sha256_file(args.protected_items):
        raise ValueError("the registered scan did not use these items")
    labels = [entry["key"] for entry in spec["datasets"]]
    flagged: dict[str, dict[int, set[int]]] = {key: defaultdict(set) for key in labels}
    verdicts: dict[str, Counter] = {key: Counter() for key in labels}
    why: dict[str, Counter] = {key: Counter() for key in labels}
    for hit in registered_hits(args.registered_hits):
        number = by_key[hit["id"]]
        if not rows[number]["scored"]:
            continue
        for label, entry in (hit.get("labels") or {}).items():
            if label not in flagged or entry["verdict"] == "CLEAN":
                continue
            verdicts[label][entry["verdict"]] += 1
            why[label].update(reasons(entry))
            for field in ("row", "exact_row"):
                if entry.get(field) is not None:
                    flagged[label][number].add(entry[field])
    datasets, private, arm_rows = {}, {}, {}
    for entry in spec["datasets"]:
        key = entry["key"]
        named = line_rows(
            Path(entry["path"]), {r for rs in flagged[key].values() for r in rs}
        )
        exposure: dict[int, set[tuple[str, str]]] = defaultdict(set)
        for record in scan_private["datasets"][key]:
            for member, cell in record["hits"].items():
                if cell.get("data") and rows[int(member)]["scored"]:
                    exposure[int(member)].add(
                        (record["group_id"], str(record["family"]))
                    )
        by_g0 = set(exposure)
        for number, lines in flagged[key].items():
            for line in lines:
                exposure[number].add(
                    (named[line]["group_id"], str(named[line]["family"]))
                )
        groups = sorted({group for cells in exposure.values() for group, _ in cells})
        arm_rows[key] = exposure
        hit_items = [rows[n] for n in exposure]
        by_scanner = set(flagged[key])
        datasets[key] = {
            "sha256": entry["sha256"],
            "rows": entry["rows"],
            "verdict": "PASS" if not exposure else "QUARANTINE",
            "exposed_scored_items": len(exposure),
            "exposed_by": {
                "g0_only": len(by_g0 - by_scanner),
                "registered_only": len(by_scanner - by_g0),
                "both": len(by_g0 & by_scanner),
            },
            "exposed_scored_items_by_source": tally(
                item["source"] for item in hit_items
            ),
            "exposed_c1_groups": len(
                {(item["source"], item["group_id"]) for item in hit_items}
            ),
            "quarantine_groups": len(groups),
            "quarantine_sha256": sha256_text(groups) if groups else None,
            "registered_scanner": {
                "scored_items_non_clean": dict(sorted(verdicts[key].items())),
                "scored_item_reasons": dict(sorted(why[key].items())),
                "label_verdicts": receipt["corpora"][key]["verdicts"],
            },
            "g0": {
                "exposed_rows": scan_public["datasets"][key]["exposed"]["rows"],
                "exposed_groups": scan_public["datasets"][key]["exposed"]["groups"],
                "boilerplate_only_rows": scan_public["datasets"][key][
                    "boilerplate_only"
                ]["rows"],
            },
        }
        if groups:
            private[key] = groups
    arms = []
    for arm in spec.get("arms") or []:
        exposed: set[int] = set()
        for use in arm["uses"]:
            left_out = set(use.get("exclude_families") or [])
            for number, cells in arm_rows[use["dataset"]].items():
                if any(family not in left_out for _, family in cells):
                    exposed.add(number)
        arms.append(
            {
                "track": arm["track"],
                "arm": arm["arm"],
                "datasets": [use["dataset"] for use in arm["uses"]],
                "exposed_scored_items": len(exposed),
                "item8": "valid" if not exposed else "not valid (exposure > 0)",
            }
        )
    args.quarantine_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    for key, groups in private.items():
        write_new(
            args.quarantine_dir / f"QUARANTINE-{key}.txt",
            "".join(f"{g}\n" for g in groups).encode(),
        )
    verdict = {
        "schema": SCHEMAS["verdict"],
        "spec": spec["name"],
        "spec_sha256": sha256_file(args.spec),
        "item_set": {
            "version": ITEM_SET,
            "retired_sha256": POSTKEY_RETIRED_SHA256,
            "items": len(rows),
            "scored": sum(item["scored"] for item in rows),
        },
        "items_sha256": items_sha,
        "scan_sha256": sha256_file(args.scan),
        "registered_receipt_sha256": sha256_file(args.registered_receipt),
        "registered_hits_sha256": sha256_file(args.registered_hits),
        "controls": scan_public["controls"],
        "datasets": datasets,
        "arms": arms,
        "verdict": (
            "PASS"
            if all(d["verdict"] == "PASS" for d in datasets.values())
            else "QUARANTINE"
        ),
    }
    if args.protected_hits and args.protected_receipt:
        verdict["disclosure_protected_rows"] = protected_disclosure(
            args.protected_hits, args.protected_receipt, labels
        )
    write_json(args.output, verdict)
    print(
        json.dumps(
            {
                "verdict": verdict["verdict"],
                **{k: v["verdict"] for k, v in datasets.items()},
            }
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("verify")
    one.add_argument("--spec", type=Path, required=True)
    one.add_argument("--output", type=Path, required=True)
    two = sub.add_parser("items")
    two.add_argument("--prompts", type=Path, required=True)
    two.add_argument("--candidates-dir", type=Path, required=True)
    two.add_argument("--build-manifest", type=Path, required=True)
    two.add_argument("--retired", type=Path, required=True)
    two.add_argument("--expect-scored", type=int, default=SCORED_ITEMS)
    two.add_argument("--output", type=Path, required=True)
    two.add_argument("--protected-output", type=Path, required=True)
    two.add_argument("--receipt", type=Path, required=True)
    three = sub.add_parser("scan")
    three.add_argument("--spec", type=Path, required=True)
    three.add_argument("--items", type=Path, required=True)
    three.add_argument("--workers", type=int, default=32)
    three.add_argument("--chunk-rows", type=int, default=500)
    three.add_argument("--output", type=Path, required=True)
    three.add_argument("--private", type=Path, required=True)
    four = sub.add_parser("judge")
    four.add_argument("--spec", type=Path, required=True)
    four.add_argument("--items", type=Path, required=True)
    four.add_argument("--protected-items", type=Path, required=True)
    four.add_argument("--scan", type=Path, required=True)
    four.add_argument("--scan-private", type=Path, required=True)
    four.add_argument("--registered-hits", type=Path, required=True)
    four.add_argument("--registered-receipt", type=Path, required=True)
    four.add_argument("--protected-hits", type=Path)
    four.add_argument("--protected-receipt", type=Path)
    four.add_argument("--output", type=Path, required=True)
    four.add_argument("--quarantine-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    return {"verify": verify, "items": items, "scan": scan, "judge": judge}[
        args.command
    ](args)


if __name__ == "__main__":
    raise SystemExit(main())
