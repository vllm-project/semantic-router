"""Score5-typed-DEV v1: a 5-level typed Score development panel (never a release score).

    python3 -m v2.eval.score5t scan --manifest <training-corpora.json> --output <scan.json> [--workers N]
    python3 -m v2.eval.score5t build --output-dir <dir> --panels-root <root> \
        --select <path> --cal <path> --cal698 <path> [--{select,cal,cal698}-sha256 <sha>] \
        --training-scan <scan.json>
    python3 -m v2.eval.score5t validate --panels-root <root> --run NAME=RUN_DIR ... --output <json>

`build` draws `resource_ledger` groups (base, counterfactual, order and label variants) from
the typed FINAL generator (`benchmark.generate`, asserted to be FINAL's file) with two fresh
public seeds, walks instance indices in order and keeps whole groups until each half (fit,
check) has 100. A group is dropped when one of its items repeats a row of typed FINAL, typed
DEV, SELECT700, CAL700, CAL698 or a training-scan hit exactly (K0, the generator payload
digest) or up to event ids and row order (K1). `scan` is the fixed-string scan of the
training-corpora manifest that `build` reads (its output lists private paths).

`summary` / `blocks` (for `v2.eval.dev_readout`) give the level histogram, top share,
accuracy against always-majority and the gate-equivalent COLLAPSE / WARN / NO-GAIN flags;
`validate` checks them against the known typed FINAL collapses (V1-V3).
Rules: `v2/eval/records/score5t-dev-prereg-2026-09-29.md`.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import gzip
import hashlib
import json
import math
import mmap
import multiprocessing
import os
import random
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import benchmark.generate as generator
from benchmark.generate import digest, prompt_record, replay_ledger
from v2.eval import panels
from v2.eval.gates import TOP_SHARE, wilson
from v2.eval.htdev.build import code_commit, jsonl_bytes, private_dir, sha_bytes
from v2.eval.proxy_calibration import kendall_tau_b, pearson, ranks
from v2.eval.same_panel import read_jsonl, sha_file, utc_now, write_json
from v2.eval.sealed.build import gold_record, write_new
from v2.eval.sealed.schema import LONG_INPUT_CHARS, MAX_INPUT_CHARS, input_chars
from v2.eval.sealed.score import macro_f1, outcomes, quadratic_kappa

SCHEMA = "dev2-score5t-build/1"
SCAN_SCHEMA = "dev2-score5t-scan/1"
VALIDATE_SCHEMA = "dev2-score5t-validate/1"
PANEL = "score5t-dev"
FAMILY = "resource_ledger"
TASK = f"score5t/{FAMILY}"
SOURCE = "benchmark-generate-resource-ledger"
SCOPE = "development readout only; never a release score, v3, chart or card"
SEED_NAMES = {
    "fit": "decision2-score5t-dev-v1/fit",
    "check": "decision2-score5t-dev-v1/check",
}
GROUPS_PER_HALF = 100
MAX_INDICES = 1000
VARIANTS = ("base", "counterfactual", "order", "label")
LEVELS = 5
FINAL_GENERATOR_SHA256 = (
    "c6569d4c86c3b9ea2ebdda564d8ba4d3428cc807d41d39984379821cfbb9422c"
)
LEDGER_INSTRUCTIONS = (
    "Replay posted events by increasing tick from the initial amount. Add or remove "
    "units, clamping to zero and capacity after every event. Ignore unposted events "
    "and the input order of event rows. Rate the final amount."
)
LEDGER_CRITERIA = [
    "Final amount is 0 units.",
    "Final amount is 1 unit.",
    "Final amount is 2 units.",
    "Final amount is 3 units.",
    "Final amount is 4 units.",
]
LEDGER_QUESTIONS = {
    "decision": {
        "type": "score",
        "instructions": LEDGER_INSTRUCTIONS,
        "criteria": LEDGER_CRITERIA,
    }
}
SIGNATURES = {
    "instructions": LEDGER_INSTRUCTIONS.encode(),
    "criterion": LEDGER_CRITERIA[0].encode(),
}
LOOSE_TOKENS = (b"posted", b"capacity")
LEDGER_FIELDS = ("initial", "capacity", "events")
EVENT_FIELDS = ("tick", "kind", "units", "posted")
SKIPPED_COMPRESSED = (".zst", ".zstd", ".xz", ".bz2", ".lz4", ".zip")
SEALED_ROOT = "/data/dev2/private/sealed/"
CAL698_SHA256 = "19cc1a8c4ebe6fd13031079f6b7d131046ce503f435f894e3ea672f91c2ed41f"
REASONS = ("final-k0", "final-k1", "dev", "select", "cal", "cal698", "training")
ANALYTIC_GOLD_OF_90 = (10, 17, 18, 18, 27)
ADAPTER_SPEC = "v2/06b/records/adapters/dev2-06b-causal-8k.json"
REPLICATES = 5000
BOOT_SEED = 20260929
RARE_SHARE = 0.02
CHANCE = 0.20
BLOCKS = ("full", "fit", "check")
SEAL = "SEAL-SCORE5T.json"
FINAL_TOP_SHARE = {
    "m7-mx-soup": 0.985,
    "m6-mxcx-soup": 0.9475,
    "m7-mxcx-soup": 0.8275,
    "m6-mxcxa-soup": 0.7875,
    "m4-t-a7-soup": 0.6425,
    "kai1": 0.5575,
}
V1_COLLAPSE = ("m6-mxcx-soup", "m7-mx-soup")
V2_NOT_COLLAPSE = ("m4-t-a7-soup", "m6-mxcxa-soup")
V3_ORDER = (
    "m7-mx-soup",
    "m6-mxcx-soup",
    "m7-mxcx-soup",
    "m6-mxcxa-soup",
    "m4-t-a7-soup",
)


def seed_bytes(name: str) -> bytes:
    return hashlib.sha256(name.encode()).digest()


def panel_id(item_id: str) -> str:
    key = f"score5t-dev-v1:id:{item_id}".encode()
    return "score5t-" + hashlib.sha256(key).hexdigest()[:16]


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=1, sort_keys=True) + "\n").encode()


# ------------------------------------------------------------------ keys


def integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def ledger_state(value: Any) -> tuple | None:
    """K1's state part (initial, capacity, sorted (tick, kind, units, posted) events) when
    `value` is a ledger state; other keys are ignored."""
    if not isinstance(value, dict) or not all(k in value for k in LEDGER_FIELDS):
        return None
    initial, capacity, events = (value[k] for k in LEDGER_FIELDS)
    if not (integer(initial) and integer(capacity) and isinstance(events, list)):
        return None
    rows = []
    for event in events:
        if not isinstance(event, dict) or not all(k in event for k in EVENT_FIELDS):
            return None
        tick, kind, units, posted = (event[k] for k in EVENT_FIELDS)
        if not (
            integer(tick)
            and isinstance(kind, str)
            and integer(units)
            and isinstance(posted, bool)
        ):
            return None
        rows.append((tick, kind, units, posted))
    return initial, capacity, tuple(sorted(rows))


def ledger_key(state: Any, questions: Any = None) -> str | None:
    """K1, the ledger problem up to event ids and row order: sha256 of the state part plus
    the decision question's instructions and criteria when `questions` holds one.

    Exclusion compares `ledger_key(state)`, the state part alone: a row holding the same
    ledger under any question, or none, is a near-duplicate."""
    part = ledger_state(state)
    if part is None:
        return None
    question = questions.get("decision") if isinstance(questions, dict) else None
    if isinstance(question, dict):
        part += (question.get("instructions"), question.get("criteria"))
    return digest(part)


def payload_key(state: Any, questions: Any) -> str:
    """K0: the generator's `payload_sha256` of {state, questions}."""
    return digest({"state": state, "questions": questions})


def structure_key(state: Any) -> tuple | None:
    """K2 (disclosure only): initial, the tick-10 add's units, the tick-20 removal's units
    and posted flag."""
    part = ledger_state(state)
    if part is None:
        return None
    events = {(tick, kind): (units, posted) for tick, kind, units, posted in part[2]}
    add, removal = events.get((10, "add")), events.get((20, "remove"))
    if add is None or removal is None:
        return None
    return part[0], add[0], *removal


def ledgers(value: Any, questions: Any = None) -> Iterator[tuple[dict[str, Any], Any]]:
    """Every ledger state nested in `value` (dicts, lists, JSON objects inside strings),
    with the `questions` dict beside it when it is the `state` of a {state, questions}.
    """
    if isinstance(value, dict):
        if ledger_state(value) is not None:
            yield value, questions
        sibling = value.get("questions")
        sibling = sibling if isinstance(sibling, dict) else None
        for key, child in value.items():
            yield from ledgers(child, sibling if key == "state" else None)
    elif isinstance(value, list):
        for child in value:
            yield from ledgers(child)
    elif isinstance(value, str) and all(word in value for word in LEDGER_FIELDS):
        decoder = json.JSONDecoder()
        start = value.find("{")
        while start != -1:
            try:
                child, end = decoder.raw_decode(value, start)
            except ValueError:
                end = start + 1
            else:
                yield from ledgers(child)
            start = value.find("{", end)


def row_keys(row: Any) -> tuple[set[str], set[str], int]:
    """K0 keys (the row's own and those of nested {state, questions} ledgers), K1 state-part
    keys and the number of ledger states of one row."""
    k0: set[str] = set()
    k1: set[str] = set()
    found = 0
    if (
        isinstance(row, dict)
        and "state" in row
        and isinstance(row.get("questions"), dict)
    ):
        k0.add(payload_key(row["state"], row["questions"]))
    for state, questions in ledgers(row):
        found += 1
        k1.add(ledger_key(state))
        if questions is not None:
            k0.add(payload_key(state, questions))
    return k0, k1, found


# ------------------------------------------------------------------ scan


@contextlib.contextmanager
def contents(path: Path) -> Iterator[Any]:
    """The file's bytes: gzip decompressed in memory, anything else memory-mapped."""
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as stream:
            yield stream.read()
        return
    with path.open("rb") as stream:
        if os.fstat(stream.fileno()).st_size == 0:
            yield b""
            return
        with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
            yield data


def line_at(data: Any, position: int) -> tuple[int, int]:
    end = data.find(b"\n", position)
    return data.rfind(b"\n", 0, position) + 1, len(data) if end == -1 else end


def signature_rows(data: Any) -> dict[tuple[int, int], set[str]]:
    rows: dict[tuple[int, int], set[str]] = defaultdict(set)
    for name, needle in SIGNATURES.items():
        position = data.find(needle)
        while position != -1:
            line = line_at(data, position)
            rows[line].add(name)
            position = data.find(needle, line[1])
    return rows


def loose_rows(data: Any) -> int:
    posted, capacity = LOOSE_TOKENS
    count, position = 0, data.find(posted)
    while position != -1:
        start, end = line_at(data, position)
        count += data.find(capacity, start, end) != -1
        position = data.find(posted, end)
    return count


def hit_keys(data: Any, rows: list[tuple[int, int]], document: bool) -> dict[str, Any]:
    """K0 / K1 keys of the ledger states in the signature rows (a non-JSONL file is read as
    one JSON document when it parses); `unparsed` counts rows (documents) without one.
    """
    units: list[Any] = []
    if document:
        with contextlib.suppress(ValueError, RecursionError):
            units.append(json.loads(data[:]))
    if not units:
        for start, end in rows:
            text = data[start:end].decode("utf-8", "replace")
            try:
                units.append(json.loads(text))
            except (ValueError, RecursionError):
                units.append(text)
    k0: set[str] = set()
    k1: set[str] = set()
    states = unparsed = 0
    for unit in units:
        try:
            row_k0, row_k1, found = row_keys(unit)
        except (ValueError, RecursionError):
            row_k0, row_k1, found = set(), set(), 0
        k0 |= row_k0
        k1 |= row_k1
        states += found
        unparsed += not found
    return {
        "ledger_states": states,
        "unparsed": unparsed,
        "k0": sorted(k0),
        "k1": sorted(k1),
    }


def scan_file(path: str) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        return {"path": path, "status": "missing"}
    if source.suffix in SKIPPED_COMPRESSED:
        return {"path": path, "status": "skipped-compressed"}
    try:
        with contents(source) as data:
            out: dict[str, Any] = {
                "path": path,
                "status": "ok",
                "size": source.stat().st_size,
                "bytes": len(data),
                "signature_rows": 0,
                "loose_rows": 0,
            }
            rows = signature_rows(data)
            if rows:
                out["signature_rows"] = len(rows)
                out["by_signature"] = {
                    name: sum(name in names for names in rows.values())
                    for name in SIGNATURES
                }
                document = ".jsonl" not in source.suffixes
                out.update(hit_keys(data, sorted(rows), document))
            if all(data.find(token) != -1 for token in LOOSE_TOKENS):
                out["loose_rows"] = loose_rows(data)
            return out
    except (OSError, EOFError) as error:
        return {
            "path": path,
            "status": "error",
            "error": f"{type(error).__name__}: {error}",
        }


def scan(args: argparse.Namespace) -> int:
    started = time.monotonic()
    raw = args.manifest.read_bytes()
    manifest = json.loads(raw)
    labels: dict[str, list[str]] = {}
    sizes: dict[str, int | None] = {}
    entries = 0
    for label, group in manifest["labels"].items():
        for entry in group["files"]:
            entries += 1
            names = labels.setdefault(entry["path"], [])
            if label not in names:
                names.append(label)
            sizes.setdefault(entry["path"], entry.get("bytes"))
    sealed = [p for p in labels if os.path.normpath(p).startswith(SEALED_ROOT)]
    if sealed:
        raise SystemExit(f"{len(sealed)} manifest paths are under {SEALED_ROOT}")
    paths = sorted(labels, key=lambda p: -(sizes[p] or 0))
    if args.workers > 1:
        with multiprocessing.Pool(args.workers) as pool:
            results = list(pool.imap_unordered(scan_file, paths))
    else:
        results = [scan_file(path) for path in paths]
    results.sort(key=lambda r: r["path"])
    ok = [r for r in results if r["status"] == "ok"]
    hits = [r for r in ok if r["signature_rows"]]
    loose = sorted(
        (r for r in ok if r["loose_rows"]), key=lambda r: (-r["loose_rows"], r["path"])
    )
    by_label: dict[str, Counter] = defaultdict(Counter)
    for r in results:
        for label in labels[r["path"]]:
            by_label[label]["files"] += 1
            for field in ("bytes", "signature_rows", "loose_rows"):
                by_label[label][field] += r.get(field, 0)
    report = {
        "schema": SCAN_SCHEMA,
        "code_commit": code_commit(),
        "manifest": {
            "path": str(args.manifest),
            "sha256": sha_bytes(raw),
            "schema": manifest.get("schema"),
            "labels": len(manifest["labels"]),
            "entries": entries,
            "files": len(paths),
        },
        "signatures": {name: needle.decode() for name, needle in SIGNATURES.items()},
        "loose_tokens": [token.decode() for token in LOOSE_TOKENS],
        "rows": "lines; a non-JSONL file's signature rows are parsed as one JSON document",
        "workers": args.workers,
        "files_scanned": len(ok),
        "bytes_scanned": sum(r["bytes"] for r in ok),
        "missing": [r["path"] for r in results if r["status"] == "missing"],
        "skipped_compressed": [
            r["path"] for r in results if r["status"] == "skipped-compressed"
        ],
        "errors": [
            {"path": r["path"], "error": r["error"]}
            for r in results
            if r["status"] == "error"
        ],
        "size_mismatch": [
            r["path"]
            for r in ok
            if sizes[r["path"]] is not None and r["size"] != sizes[r["path"]]
        ],
        "signature_rows": sum(r["signature_rows"] for r in ok),
        "signature_rows_by": {
            name: sum(r["by_signature"][name] for r in hits) for name in SIGNATURES
        },
        "signature_files": len(hits),
        "ledger_states": sum(r["ledger_states"] for r in hits),
        "unparsed": sum(r["unparsed"] for r in hits),
        "loose_rows": sum(r["loose_rows"] for r in ok),
        "loose_files": len(loose),
        "loose_top_files": [
            {"path": r["path"], "labels": labels[r["path"]], "rows": r["loose_rows"]}
            for r in loose[:20]
        ],
        "by_label": {label: dict(cell) for label, cell in sorted(by_label.items())},
        "hits": [{**r, "labels": labels[r["path"]]} for r in hits],
        "k0": sorted({key for r in hits for key in r["k0"]}),
        "k1": sorted({key for r in hits for key in r["k1"]}),
    }
    report["runtime_seconds"] = round(time.monotonic() - started, 3)
    data = json_bytes(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_new(args.output, data)
    counted = ("missing", "skipped_compressed", "errors", "size_mismatch")
    totals = ("signature_rows", "signature_files", "ledger_states", "unparsed")
    print(
        json.dumps(
            {
                "files_scanned": report["files_scanned"],
                "bytes_scanned": report["bytes_scanned"],
                **{key: len(report[key]) for key in counted},
                **{key: report[key] for key in totals},
                "loose_rows": report["loose_rows"],
                "runtime_seconds": report["runtime_seconds"],
                "scan_sha256": sha_bytes(data),
            }
        )
    )
    return 0


# ------------------------------------------------------------------ build


def verified_rows(path: Path, sha256: str) -> tuple[bytes, list[Any]]:
    data = path.read_bytes()
    observed = sha_bytes(data)
    if observed != sha256:
        raise SystemExit(f"{path}: sha256 {observed} != expected {sha256}")
    return data, [json.loads(line) for line in data.splitlines() if line.strip()]


def source_counts(
    data: bytes, rows: list[Any]
) -> tuple[dict[str, Any], set[str], set[str]]:
    k0: set[str] = set()
    k1: set[str] = set()
    ledger_rows = 0
    for row in rows:
        row_k0, row_k1, found = row_keys(row)
        k0 |= row_k0
        k1 |= row_k1
        ledger_rows += found > 0
    lines = data.splitlines()
    counts = {
        "rows": len(rows),
        "k0": len(k0),
        "ledger_rows": ledger_rows,
        "k1": len(k1),
        **{
            f"{name}_rows": sum(needle in line for line in lines)
            for name, needle in SIGNATURES.items()
        },
    }
    return counts, k0, k1


def exclusions(
    args: argparse.Namespace,
) -> tuple[dict[str, dict[str, set[str]]], dict[str, Any], dict[str, set]]:
    """Exclusion keys ("k0" / "k1" state part -> reasons), per-source counts, and the typed
    FINAL / DEV item and group ids plus FINAL's K2 values (in memory only)."""
    keys: dict[str, dict[str, set[str]]] = {
        "k0": defaultdict(set),
        "k1": defaultdict(set),
    }

    def add(k0: Any, k1: Any, k0_reason: str, k1_reason: str) -> None:
        for key in k0:
            keys["k0"][key].add(k0_reason)
        for key in k1:
            keys["k1"][key].add(k1_reason)

    report: dict[str, Any] = {}
    known: dict[str, set] = {"ids": set(), "group_ids": set()}
    for name, label in (("typed-final", "final"), ("typed-dev", "dev")):
        spec = panels.ALL[name]
        data, rows = verified_rows(
            panels.path(args.panels_root, name, "prompts"), spec["prompts_sha256"]
        )
        counts, k0, k1 = source_counts(data, rows)
        if label == "final":
            add(k0, k1, "final-k0", "final-k1")
            ledger_rows = [r for r in rows if ledger_state(r["state"]) is not None]
            final_k2 = {structure_key(r["state"]) for r in ledger_rows}
            known["final_k2"] = final_k2 - {None}
            counts["ledger_rows_with_panel_question"] = sum(
                r["questions"] == LEDGER_QUESTIONS for r in ledger_rows
            )
            counts["k2"] = len(known["final_k2"])
        else:
            add(k0, k1, label, label)
        _, gold = verified_rows(
            panels.path(args.panels_root, name, "gold"), spec["gold_sha256"]
        )
        known["ids"] |= {r["id"] for r in rows} | {r["id"] for r in gold}
        known["group_ids"] |= {r["group_id"] for r in gold}
        report[label] = {
            "file": spec["prompts"],
            "sha256": spec["prompts_sha256"],
            "gold_sha256": spec["gold_sha256"],
            **counts,
        }
    for label in ("select", "cal", "cal698"):
        path, sha256 = getattr(args, label), getattr(args, f"{label}_sha256")
        data, rows = verified_rows(path, sha256)
        counts, k0, k1 = source_counts(data, rows)
        add(k0, k1, label, label)
        report[label] = {"file": str(path), "sha256": sha256, **counts}
    raw = args.training_scan.read_bytes()
    found = json.loads(raw)
    signatures = {name: needle.decode() for name, needle in SIGNATURES.items()}
    if found.get("schema") != SCAN_SCHEMA or found.get("signatures") != signatures:
        raise SystemExit(f"{args.training_scan}: not a {SCAN_SCHEMA} ledger scan")
    add(found["k0"], found["k1"], "training", "training")
    report["training"] = {
        "file": str(args.training_scan),
        "sha256": sha_bytes(raw),
        "manifest_sha256": found["manifest"]["sha256"],
        "manifest_files": found["manifest"]["files"],
        **{
            key: found[key]
            for key in (
                "files_scanned",
                "bytes_scanned",
                "signature_rows",
                "signature_rows_by",
                "signature_files",
                "ledger_states",
                "unparsed",
                "loose_rows",
                "loose_files",
            )
        },
        **{key: len(found[key]) for key in ("missing", "skipped_compressed", "errors")},
        "k0": len(found["k0"]),
        "k1": len(found["k1"]),
    }
    return keys, report, known


def ledger_groups(seed: bytes, indices: int) -> dict[int, list[dict[str, Any]]]:
    """The generator's `resource_ledger` items of instances 0..indices-1, by instance."""
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in generator.generate("final", seed, indices):
        if item["family"] != FAMILY:
            continue
        if item["questions"] != LEDGER_QUESTIONS:
            raise ValueError(f"{item['id']}: not the ledger question")
        if item["provenance"]["payload_sha256"] != payload_key(
            item["state"], item["questions"]
        ):
            raise ValueError(f"{item['id']}: K0 differs from the payload digest")
        groups[item["provenance"]["instance_index"]].append(item)
    for index, group in groups.items():
        if tuple(item["provenance"]["variant"] for item in group) != VARIANTS:
            raise ValueError(f"instance {index}: unexpected variants")
    return dict(groups)


def select_groups(
    groups: dict[int, list[dict[str, Any]]],
    keys: dict[str, dict[str, set[str]]],
    wanted: int,
) -> tuple[list[list[dict[str, Any]]], list[dict[str, Any]]]:
    """Whole groups in instance order until `wanted` pass; a group is dropped when any of
    its items' K0 or K1 state part is an exclusion key."""
    accepted: list[list[dict[str, Any]]] = []
    log: list[dict[str, Any]] = []
    for index in sorted(groups):
        if len(accepted) == wanted:
            break
        group = groups[index]
        reasons: set[str] = set()
        for item in group:
            reasons |= keys["k0"].get(
                payload_key(item["state"], item["questions"]), set()
            )
            reasons |= keys["k1"].get(ledger_key(item["state"]), set())
        log.append(
            {
                "instance_index": index,
                "group_id": group[0]["group_id"],
                "accepted": not reasons,
                "reasons": sorted(reasons, key=REASONS.index),
                "k1": sorted({ledger_key(i["state"], i["questions"]) for i in group}),
                "k2": sorted({digest(structure_key(i["state"])) for i in group}),
            }
        )
        if not reasons:
            accepted.append(group)
    if len(accepted) < wanted:
        raise ValueError(
            f"only {len(accepted)} of {wanted} groups pass in {len(groups)} indices"
        )
    return accepted, log


def check_panel(items: list[tuple[str, dict[str, Any]]], known: dict[str, set]) -> None:
    if len(items) != len(SEED_NAMES) * GROUPS_PER_HALF * len(VARIANTS):
        raise ValueError(f"{len(items)} panel items")
    for half, item in items:
        provenance = item["provenance"]
        if (
            provenance["seed_commitment_sha256"] != digest(seed_bytes(SEED_NAMES[half]))
            or provenance["generator_code_sha256"] != FINAL_GENERATOR_SHA256
        ):
            raise ValueError(f"{item['id']}: unexpected seed or generator provenance")
    ids = {item["id"] for _, item in items}
    panel_ids = {panel_id(i) for i in ids}
    k0 = {payload_key(item["state"], item["questions"]) for _, item in items}
    if not len(k0) == len(ids) == len(panel_ids) == len(items):
        raise ValueError("K0, an item id or a panel id repeats within the panel")
    ours = ids | panel_ids | {item["group_id"] for _, item in items}
    if ours & (known["ids"] | known["group_ids"]):
        raise ValueError("a panel item or group id equals a typed FINAL / DEV id")


def native_inputs() -> Callable[[dict[str, Any]], list[tuple]]:
    """The native 0.6B adapter's model text of a prompt line, per question: the
    (prefix, options, suffix) segments the tokenizer encodes, from
    `training.model.infer.question_to_row` and `training.model.decision_model.segments`.
    `decision_model` imports torch at load, so its two torch-free text functions are
    compiled from its source."""
    from training.model import infer
    from training.model.data import canonical

    source = Path(infer.__file__).with_name("decision_model.py")
    tree = ast.parse(source.read_text(encoding="utf-8"))
    body = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in ("_payload", "segments")
    ]
    if len(body) != 2:
        raise ValueError(f"{source}: _payload / segments not found")
    scope: dict[str, Any] = {"Any": Any, "canonical": canonical}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(source), "exec"), scope)

    def text(prompt: dict[str, Any]) -> list[tuple]:
        out = []
        for key, question in prompt["questions"].items():
            row = infer.question_to_row(prompt, key, question)
            prefix, options, suffix = scope["segments"](row)
            out.append((prefix, tuple(options), suffix))
        return out

    return text


def render(item: dict[str, Any], half: str) -> tuple[dict[str, Any], dict[str, Any]]:
    item_id = panel_id(item["id"])
    question = item["questions"]["decision"]
    decision = item["gold"]["decision"]
    if (
        set(item["gold"]) != {"decision"}
        or decision != gold_record({"question": question, "gold": decision["value"]})
        or decision["value"] != replay_ledger(item["state"])
    ):
        raise ValueError(f"{item['id']}: gold differs from gold_record / the replay")
    chars = input_chars(item["state"], question)
    if chars > MAX_INPUT_CHARS:
        raise ValueError(f"{item['id']}: input over {MAX_INPUT_CHARS} characters")
    provenance = item["provenance"]
    prompt = {"id": item_id, "state": item["state"], "questions": item["questions"]}
    gold = {
        "id": item_id,
        "task": TASK,
        "source": SOURCE,
        "half": half,
        "split": half,
        "source_item_id": item["id"],
        "group_id": item["group_id"],
        "cluster_id": item["group_id"],
        "language": "en",
        "input_chars": chars,
        "long": chars >= LONG_INPUT_CHARS,
        "provenance": {
            "seed_name": SEED_NAMES[half],
            "seed_commitment_sha256": provenance["seed_commitment_sha256"],
            "generator_split": item["split"],
            "instance_index": provenance["instance_index"],
            "variant": provenance["variant"],
            "generator_item_id": item["id"],
            "payload_sha256": provenance["payload_sha256"],
            "generator_code_sha256": provenance["generator_code_sha256"],
        },
        "questions": item["questions"],
        "gold": {"decision": decision},
    }
    return prompt, gold


def drop_counts(log: list[dict[str, Any]]) -> dict[str, Any]:
    dropped = [entry for entry in log if not entry["accepted"]]
    reasons = Counter(reason for entry in dropped for reason in entry["reasons"])
    return {
        "indices_walked": len(log),
        "groups_dropped": len(dropped),
        "dropped_by_reason": {reason: reasons[reason] for reason in REASONS},
        "dropped_by_reason_set": dict(
            sorted(Counter("+".join(entry["reasons"]) for entry in dropped).items())
        ),
    }


def half_stats(
    groups: list[list[dict[str, Any]]], log: list[dict[str, Any]]
) -> dict[str, Any]:
    items = [item for group in groups for item in group]
    levels = Counter(item["gold"]["decision"]["value"] for item in items)
    return {
        "groups_accepted": len(groups),
        **drop_counts(log),
        "items": len(items),
        "gold_levels": {str(level): levels[level] for level in range(LEVELS)},
        "variants": dict(Counter(item["provenance"]["variant"] for item in items)),
    }


def overlap(
    halves: dict[str, list[list[dict[str, Any]]]], final_k2: set
) -> dict[str, Any]:
    def k1(item: dict[str, Any]) -> str:
        return ledger_key(item["state"], item["questions"])

    def k2(item: dict[str, Any]) -> tuple | None:
        return structure_key(item["state"])

    out: dict[str, Any] = {"final_k2_distinct": len(final_k2)}
    sets = {}
    for half, groups in halves.items():
        items = [item for group in groups for item in group]
        states = {tuple(sorted({k1(item) for item in group})) for group in groups}
        sets[half] = ({k1(i) for i in items}, states, {k2(i) for i in items})
        out[half] = {
            "k1_distinct": len(sets[half][0]),
            "k1_group_states_distinct": len(states),
            "k2_distinct": len(sets[half][2]),
            "groups_with_k2_in_final": sum(
                any(k2(i) in final_k2 for i in group) for group in groups
            ),
            "groups_with_all_k2_in_final": sum(
                all(k2(i) in final_k2 for i in group) for group in groups
            ),
            "items_with_k2_in_final": sum(k2(i) in final_k2 for i in items),
        }
    fit, check = (sets[half] for half in SEED_NAMES)
    out["fit_and_check"] = {
        "k1_shared": len(fit[0] & check[0]),
        "k1_group_states_shared": len(fit[1] & check[1]),
        "k2_shared": len(fit[2] & check[2]),
    }
    return out


def build(args: argparse.Namespace) -> int:
    from training.model import infer

    code_sha = sha_bytes(Path(generator.__file__).read_bytes())
    if code_sha != FINAL_GENERATOR_SHA256:
        raise SystemExit(
            f"benchmark/generate.py sha256 {code_sha} is not typed FINAL's"
        )
    keys, report, known = exclusions(args)
    halves: dict[str, list[list[dict[str, Any]]]] = {}
    logs: dict[str, list[dict[str, Any]]] = {}
    for half, name in SEED_NAMES.items():
        groups = ledger_groups(seed_bytes(name), MAX_INDICES)
        halves[half], logs[half] = select_groups(groups, keys, GROUPS_PER_HALF)
    items = [
        (half, item)
        for half, groups in halves.items()
        for group in groups
        for item in group
    ]
    check_panel(items, known)
    pairs = [render(item, half) for half, item in items]
    files = {
        f"{PANEL}.prompts.jsonl": jsonl_bytes([p for p, _ in pairs]),
        f"{PANEL}.gold.jsonl": jsonl_bytes([g for _, g in pairs]),
    }
    written = [
        json.loads(line) for line in files[f"{PANEL}.prompts.jsonl"].splitlines()
    ]
    text = native_inputs()
    for (_, item), prompt in zip(items, written, strict=True):
        source = prompt_record(item)
        digests = {infer.prompt_input_sha256(p) for p in (source, prompt)}
        if text(source) != text(prompt) or len(digests) != 1:
            raise ValueError(f"{item['id']}: model input differs under the panel id")
    for half in SEED_NAMES:
        files[f"{PANEL}.{half}.prompts.jsonl"] = jsonl_bytes(
            [p for p, g in pairs if g["half"] == half]
        )
        files[f"{PANEL}.{half}.gold.jsonl"] = jsonl_bytes(
            [g for _, g in pairs if g["half"] == half]
        )
    files["groups.jsonl"] = jsonl_bytes(
        [{"half": half, **entry} for half, log in logs.items() for entry in log]
    )
    files["exclusion-report.json"] = json_bytes(
        {
            "schema": SCHEMA,
            "panel": PANEL,
            "sources": report,
            "drops": {half: drop_counts(log) for half, log in logs.items()},
        }
    )
    private_dir(args.output_dir)
    for name, data in files.items():
        write_new(args.output_dir / name, data)
    model_files = ("infer.py", "decision_model.py", "data.py")
    manifest = {
        "schema": SCHEMA,
        "panel": PANEL,
        "scope": SCOPE,
        "code_commit": code_commit(),
        "generator": {
            "file": "benchmark/generate.py",
            "sha256": FINAL_GENERATOR_SHA256,
            "call": f'generate("final", seed, {MAX_INDICES})',
            "family": FAMILY,
        },
        "seeds": {
            half: {
                "name": name,
                "bytes": "sha256(name), 32 raw bytes",
                "commitment_sha256": digest(seed_bytes(name)),
            }
            for half, name in SEED_NAMES.items()
        },
        "keys": {
            "k0": "generator payload_sha256 of {state, questions}",
            "k1": "sha256 of (initial, capacity, sorted (tick, kind, units, posted), "
            "instructions, criteria)",
            "k1_match": "on the state part, whatever the other row's question (none, "
            "or typed FINAL's ledger question)",
            "k2": "(initial, tick-10 add units, tick-20 removal units, removal posted); "
            "disclosure only",
        },
        "items": len(pairs),
        "long_inputs": sum(g["long"] for _, g in pairs),
        "halves": {half: half_stats(halves[half], logs[half]) for half in SEED_NAMES},
        "overlap": overlap(halves, known["final_k2"]),
        "context": {
            "analytic_gold_share": {
                str(level): f"{n}/90" for level, n in enumerate(ANALYTIC_GOLD_OF_90)
            },
        },
        "exclusion_sources": {
            label: {"file": entry["file"], "sha256": entry["sha256"]}
            for label, entry in report.items()
        },
        "training_scan": {
            key: report["training"][key]
            for key in ("manifest_sha256", "signature_rows", "unparsed", "missing")
        },
        "adapter_input_check": {
            "adapter_spec": ADAPTER_SPEC,
            "text": "training.model.infer.question_to_row, then "
            "training.model.decision_model.segments (the segments the tokenizer encodes)",
            "note": "pure text construction: decision_model imports torch at load, so "
            "_payload and segments are compiled from its source; the item id only "
            "reaches the row id",
            "sources_sha256": {
                f"training/model/{name}": sha_file(Path(infer.__file__).with_name(name))
                for name in model_files
            },
            "items_identical": len(pairs),
        },
        "files_sha256": {name: sha_bytes(data) for name, data in files.items()},
    }
    data = json_bytes(manifest)
    write_new(args.output_dir / "MANIFEST.json", data)
    shown = ("indices_walked", "groups_dropped", "dropped_by_reason")
    print(
        json.dumps(
            {
                "items": len(pairs),
                **{
                    half: {key: manifest["halves"][half][key] for key in shown}
                    for half in SEED_NAMES
                },
                "training_scan": manifest["training_scan"],
                "manifest_sha256": sha_bytes(data),
                **manifest["files_sha256"],
            }
        )
    )
    return 0


# ------------------------------------------------------------------ summary


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def bootstrap(
    correct: list[int], always: list[int], replicates: int
) -> tuple[list[float], list[float]]:
    """Item bootstrap of accuracy and of accuracy - always-majority on the same resampled
    items, as `m7_scorebias.bootstrap` (random.Random(BOOT_SEED), n `randrange` draws per
    replicate, interpolated 2.5 / 97.5 percentiles)."""
    rng = random.Random(BOOT_SEED)
    n = len(correct)
    accuracy, gain = [], []
    for _ in range(replicates):
        hits = majority = 0
        for _ in range(n):
            index = rng.randrange(n)
            hits += correct[index]
            majority += always[index]
        accuracy.append(hits / n)
        gain.append((hits - majority) / n)
    return (
        [percentile(accuracy, 2.5), percentile(accuracy, 97.5)],
        [percentile(gain, 2.5), percentile(gain, 97.5)],
    )


def majority_level(gold: list[int]) -> int:
    counts = Counter(gold)
    best = max(counts.values())
    return min(k for k, v in counts.items() if v == best)


def flags(
    top_share: float, top_upper: float, accuracy_lower: float, gain_lower: float
) -> list[str]:
    out = []
    if top_share >= TOP_SHARE or accuracy_lower <= CHANCE:
        out.append("COLLAPSE")
    elif top_upper >= TOP_SHARE:
        out.append("WARN")
    if gain_lower <= 0:
        out.append("NO-GAIN")
    return out


def summary(
    gold: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    replicates: int = REPLICATES,
) -> dict[str, Any]:
    unknown = set(predictions) - {row["id"] for row in gold}
    if unknown:
        raise ValueError(f"{len(unknown)} prediction ids are not in {PANEL}")
    if not gold:
        raise ValueError("no gold rows")
    pairs = [(row[5], row[6]) for row in outcomes(gold, predictions)]
    n = len(pairs)
    truth = [g for g, _ in pairs]
    predicted = [p for _, p in pairs]
    histogram = {str(level): predicted.count(level) for level in range(LEVELS)}
    histogram["invalid"] = predicted.count(None)
    top_category = max(histogram, key=histogram.__getitem__)
    top = histogram[top_category]
    answered = [histogram[str(level)] for level in range(LEVELS)]
    correct = [int(g == p) for g, p in pairs]
    majority = majority_level(truth)
    always = [int(g == majority) for g in truth]
    accuracy_ci, gain_ci = bootstrap(correct, always, replicates)
    top_wilson = list(wilson(top, n))
    accuracy_wilson = list(wilson(sum(correct), n))
    gold_histogram = {str(level): truth.count(level) for level in range(LEVELS)}
    return {
        "scope": f"{SCOPE} ({PANEL})",
        "n": n,
        "histogram": histogram,
        "top_category": top_category,
        "top_share": top / n,
        "top_share_wilson95": top_wilson,
        "modal_level": answered.index(max(answered)) if any(answered) else None,
        "rare_levels": [
            level for level in range(LEVELS) if answered[level] < RARE_SHARE * n
        ],
        "invalid_or_missing": histogram["invalid"],
        "correct": sum(correct),
        "accuracy": sum(correct) / n,
        "accuracy_wilson95": accuracy_wilson,
        "accuracy_boot95": accuracy_ci,
        "bootstrap": {"replicates": replicates, "seed": BOOT_SEED},
        "gold_histogram": gold_histogram,
        "always_majority_level": majority,
        "always_majority_accuracy": sum(always) / n,
        "acc_minus_majority": (sum(correct) - sum(always)) / n,
        "acc_minus_majority_boot95": gain_ci,
        "macro_f1": macro_f1(pairs),
        "qwk_answered": quadratic_kappa(pairs),
        "l4_share": histogram["4"] / n,
        "gold_l4_share": gold_histogram["4"] / n,
        "flags": flags(top / n, top_wilson[1], accuracy_wilson[0], gain_ci[0]),
    }


def blocks(
    gold: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    replicates: int = REPLICATES,
) -> dict[str, dict[str, Any]]:
    out = {"full": summary(gold, predictions, replicates)}
    for half in SEED_NAMES:
        rows = [row for row in gold if row["half"] == half]
        ids = {row["id"] for row in rows}
        subset = {key: value for key, value in predictions.items() if key in ids}
        out[half] = summary(rows, subset, replicates)
    return out


# ------------------------------------------------------------------ validate


def finite(value: float) -> float | None:
    return value if math.isfinite(value) else None


def criteria(block: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """V1-V3 on one block ({model: summary}) of the typed FINAL models."""
    collapse = {name: "COLLAPSE" in entry["flags"] for name, entry in block.items()}
    shares = [block[name]["top_share"] for name in V3_ORDER if name in block]
    v1 = all(collapse.get(name) is True for name in V1_COLLAPSE)
    v2 = all(collapse.get(name) is False for name in V2_NOT_COLLAPSE)
    v3 = len(shares) == len(V3_ORDER) and all(a > b for a, b in zip(shares, shares[1:]))
    return {
        "V1": v1,
        "V2": v2,
        "V3": v3,
        "pass": v1 and v2 and v3,
        "missing": sorted({*V1_COLLAPSE, *V2_NOT_COLLAPSE, *V3_ORDER} - set(block)),
        "collapse": {name: collapse.get(name) for name in V3_ORDER},
        "top_share": {
            name: block[name]["top_share"] for name in V3_ORDER if name in block
        },
    }


def agreement(block: dict[str, dict[str, Any]]) -> dict[str, Any]:
    names = [name for name in FINAL_TOP_SHARE if name in block]
    ours = [block[name]["top_share"] for name in names]
    final = [FINAL_TOP_SHARE[name] for name in names]
    paired = len(names) >= 2
    return {
        "models": names,
        "kendall_tau_b": finite(kendall_tau_b(ours, final)) if paired else None,
        "spearman_rho": finite(pearson(ranks(ours), ranks(final))) if paired else None,
        "panel_minus_final_top_share": {
            name: block[name]["top_share"] - FINAL_TOP_SHARE[name] for name in names
        },
    }


def kai1_position(block: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    if "kai1" not in block:
        return None
    share = block["kai1"]["top_share"]
    others = [name for name in FINAL_TOP_SHARE if name in block and name != "kai1"]
    return {
        "top_share": share,
        "rank": 1 + sum(block[name]["top_share"] > share for name in others),
        "models": len(others) + 1,
        "lowest": all(block[name]["top_share"] > share for name in others),
    }


def table(result: dict[str, Any]) -> str:
    columns = f"{'full top (cat)':>22}{'fit':>8}{'check':>8}{'acc':>8}{'maj':>8}"
    lines = [f"{'model':<22}{columns}  flags"]
    for name, run in result["runs"].items():
        full, fit, check = (run["blocks"][block] for block in BLOCKS)
        lines.append(
            f"{name:<22}{full['top_share']:>11.4f} ({full['top_category']:>7})"
            f"{fit['top_share']:>8.4f}{check['top_share']:>8.4f}"
            f"{full['accuracy']:>8.4f}{full['always_majority_accuracy']:>8.4f}"
            f"  {','.join(full['flags']) or '-'}"
        )
    checks = result["criteria"]
    missing = f" (missing: {', '.join(checks['missing'])})" if checks["missing"] else ""
    lines.append(
        f"V1 {checks['V1']}  V2 {checks['V2']}  V3 {checks['V3']}  -> "
        f"{result['verdict']}{missing}"
    )
    return "\n".join(lines)


def validate(args: argparse.Namespace) -> int:
    if PANEL not in panels.ALL:
        raise SystemExit(f"{PANEL} is not registered in v2/eval/panels.py")
    verified = panels.verify(args.panels_root, [PANEL])
    spec = panels.ALL[PANEL]
    gold = read_jsonl(panels.path(args.panels_root, PANEL, "gold"))
    runs: dict[str, dict[str, Any]] = {}
    for entry in args.run:
        name, _, value = entry.partition("=")
        if not name or not value or name in runs:
            raise SystemExit(f"--run {entry!r}: expected a new NAME=RUN_DIR")
        run = Path(value)
        seal = json.loads((run / SEAL).read_text(encoding="utf-8"))
        path = run / "output" / f"{PANEL}.predictions.jsonl"
        if seal.get("prompts_sha256") != spec["prompts_sha256"]:
            raise ValueError(f"{name}: {SEAL} seals other prompts")
        if sha_file(path) != seal.get("predictions_sha256"):
            raise ValueError(f"{name}: {path.name} changed after {SEAL}")
        predictions = {row["id"]: row for row in read_jsonl(path)}
        runs[name] = {
            "run": str(run),
            "seal_sha256": sha_file(run / SEAL),
            "predictions_sha256": seal["predictions_sha256"],
            "seal_missing": seal.get("missing"),
            "blocks": blocks(gold, predictions, args.replicates),
        }
    models = {
        block: {
            name: run["blocks"][block]
            for name, run in runs.items()
            if name in FINAL_TOP_SHARE
        }
        for block in BLOCKS
    }
    checks = {block: criteria(models[block]) for block in BLOCKS}

    def flagged(flag: str) -> dict[str, list[str]]:
        return {
            block: [
                name
                for name, run in runs.items()
                if flag in run["blocks"][block]["flags"]
            ]
            for block in BLOCKS
        }

    result = {
        "schema": VALIDATE_SCHEMA,
        "panel": PANEL,
        "scope": SCOPE,
        "rule": (
            f"COLLAPSE: top share >= {TOP_SHARE} or the accuracy Wilson 95% lower bound "
            f"<= {CHANCE}; WARN: not COLLAPSE and the top-share Wilson 95% upper bound "
            f">= {TOP_SHARE}; NO-GAIN: the paired-bootstrap 95% lower bound of accuracy - "
            "always-majority <= 0. PASS iff V1, V2 and V3 hold on the full panel."
        ),
        "code_commit": code_commit(),
        "created_utc": utc_now(),
        "panel_sha256": verified,
        "final_top_share": FINAL_TOP_SHARE,
        "verdict": "PASS" if checks["full"]["pass"] else "FAIL",
        "criteria": checks["full"],
        "secondary": {
            "per_half": {half: checks[half] for half in SEED_NAMES},
            "kai1": kai1_position(models["full"]),
            "agreement": {block: agreement(models[block]) for block in BLOCKS},
            "warn": flagged("WARN"),
            "no_gain": flagged("NO-GAIN"),
            "extra_runs": sorted(set(runs) - set(FINAL_TOP_SHARE)),
        },
        "runs": runs,
    }
    write_json(args.output, result)
    print(table(result))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("scan")
    one.add_argument("--manifest", type=Path, required=True)
    one.add_argument("--output", type=Path, required=True)
    one.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    two = commands.add_parser("build")
    two.add_argument("--output-dir", type=Path, required=True)
    two.add_argument("--panels-root", type=Path, default=panels.DEFAULT_ROOT)
    for name, sha256 in (
        ("select", panels.DEVELOPMENT["select"]["gold_sha256"]),
        ("cal", panels.DEVELOPMENT["cal"]["gold_sha256"]),
        ("cal698", CAL698_SHA256),
    ):
        two.add_argument(f"--{name}", type=Path, required=True)
        two.add_argument(f"--{name}-sha256", default=sha256)
    two.add_argument("--training-scan", type=Path, required=True)
    three = commands.add_parser("validate")
    three.add_argument("--panels-root", type=Path, default=panels.DEFAULT_ROOT)
    three.add_argument("--run", action="append", required=True, metavar="NAME=RUN_DIR")
    three.add_argument("--output", type=Path, required=True)
    three.add_argument("--replicates", type=int, default=REPLICATES)
    args = parser.parse_args(argv)
    return {"scan": scan, "build": build, "validate": validate}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
