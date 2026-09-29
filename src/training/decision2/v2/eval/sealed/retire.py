"""Retire the C1 candidates that share a source passage with flagged protected rows.

    python3 -m v2.eval.sealed.retire build --hits <hits.jsonl | extract.json> ... \
        --protected <protected.jsonl> --sources-dir <snapshots dir> \
        --candidates-dir <dir> --build-manifest <manifest.json> \
        --build-hits <build hits.jsonl> [--baseline <hits.jsonl> ...] \
        [--previous <RETIRED.json>] --version LABEL \
        --output <private RETIRED.json> --receipt <receipt.json>

Protected rows are the raw rows of the C1 source snapshots as `independence rows` writes
them (id `KEY/file|row index`). A row is flagged when any --hits input (an overlap scan
hits jsonl, or a scan extract: a .json object whose `non_clean` list holds hit rows,
refused if its `protected_sha256` is not that of --protected) gives it a verdict other
than CLEAN; --baseline hits only split the flagged rows into recurring and new. The
candidates files and the build hits must have the SHA-256 that the build manifest
records. Flagged rows and candidates get the shingles of `overlap.load_protected`. A
candidate of any source is linked to a flagged row by text at the scanner's OVERLAP
standard: they share a shingle and the shared shingles are at least half (0.5) of
either side's set, so a short flagged sentence inside a long candidate still links,
while generic wording common to otherwise unrelated texts does not. A candidate of the
row's own source is linked by id when a top-level `id` or `*_id` field of the raw row
(re-read from the snapshot and checked against the protected row) has a non-null value
equal to the candidate's group id, its source item id, or the part of that before the
first ':'. Every candidate in the source group of a
linked candidate is retired, since the group's items share a source passage.

RETIRED.json (private, mode 600) lists the retired candidate ids and the flagged
protected ids, united with --previous. C1 items are a salted selection of candidates, so
the receipt bounds the retired items of each task by min(selected, the sum over retired
groups of min(group_cap, retired candidates the build kept)). It reports the
standard-error inflation this allows per task and for C1, the mean of per-task
macro-F1: at worst (the largest task factor; null when a task may lose every item) and
at equal item variance. The receipt holds counts and hashes only, never ids or text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from v2.eval.sealed.build import excluded, read_jsonl, write_new
from v2.eval.sealed.independence import MIN_CHARS, data_rows, strings
from v2.eval.sealed.overlap import VERDICTS, Params, shingles, string_leaves, tokens
from v2.eval.sealed.schema import normalized

SCHEMA = "dev2-c1-retire/1"
RETIRED_SCHEMA = "dev2-c1-retired/1"
KINDS = ("text", "id", "both")
PARAMS = Params()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def item_id(row: dict[str, Any]) -> str:
    return f"{row['task']}|{row['source_item_id']}"


def tally(values: Iterable[Any]) -> dict[Any, int]:
    return dict(sorted(Counter(values).items()))


def features(row: dict[str, Any]) -> set[int]:
    """The shingles of a row, built the way `overlap.load_protected` builds them."""
    leaves = [
        *string_leaves(row.get("overlap_texts") or []),
        *string_leaves(row.get("state")),
    ]
    texts = dict.fromkeys(normalized(text) for text in leaves)
    texts.pop("", None)
    own: set[int] = set()
    for norm in texts:
        own |= shingles(tokens(norm, PARAMS), PARAMS)
    return own


def text_linked(shared: int, size: int, other: int) -> bool:
    """`shared` shingles of two sets of `size` and `other` cover at least the OVERLAP
    share of either set."""
    return shared > 0 and (
        shared / size >= PARAMS.overlap or shared / other >= PARAMS.overlap
    )


def id_values(raw: Any) -> set[str]:
    """Stripped non-null values of the top-level `id` and `*_id` fields of a raw row."""
    if not isinstance(raw, dict):
        return set()
    return {
        str(value).strip()
        for name, value in raw.items()
        if isinstance(name, str)
        and (name.lower() == "id" or name.lower().endswith("_id"))
        and value is not None
        and str(value).strip()
    }


def hit_rows(path: Path, protected_sha: str) -> list[dict[str, Any]]:
    """The rows of a hits jsonl, or the `non_clean` rows of a scan extract (.json),
    which must not name other protected rows than `protected_sha`."""
    if path.name.endswith(".json"):
        try:
            document = json.loads(path.read_bytes())
        except ValueError:
            document = None
        if isinstance(document, dict) and "non_clean" in document:
            if document.get("protected_sha256", protected_sha) != protected_sha:
                raise ValueError(f"{path}: an extract of other protected rows")
            return document["non_clean"]
    return read_jsonl(path)


def non_clean(paths: Iterable[Path], protected_sha: str) -> dict[str, str]:
    """The most severe verdict other than CLEAN of each id over the hits inputs."""
    worst: dict[str, str] = {}
    for path in paths:
        for row in hit_rows(path, protected_sha):
            verdict = row["verdict"]
            if verdict not in VERDICTS:
                raise ValueError(f"{path}: unknown verdict {verdict!r}")
            if VERDICTS.index(verdict) < VERDICTS.index(worst.get(row["id"], "CLEAN")):
                worst[row["id"]] = verdict
    return worst


def load_candidates(
    directory: Path, expected: dict[str, str]
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Every candidate of the build's sources; each file must match the manifest."""
    rows: list[dict[str, Any]] = []
    digests: dict[str, str] = {}
    for source, digest in sorted(expected.items()):
        path = directory / f"{source}.jsonl"
        if not path.is_file():
            raise ValueError(f"{path}: no such candidates file")
        digests[source] = sha256(path)
        if digests[source] != digest:
            raise ValueError(f"{path}: SHA-256 differs from the build manifest")
        rows.extend(read_jsonl(path))
    if len({item_id(row) for row in rows}) < len(rows):
        raise ValueError("duplicate candidate ids")
    return rows, digests


def protected_rows(path: Path, wanted: set[str]) -> dict[str, dict[str, Any]]:
    """The protected row of each flagged id, which must occur exactly once."""
    found: dict[str, dict[str, Any]] = {}
    repeated = 0
    with path.open(encoding="utf-8") as stream:
        for row in map(json.loads, filter(str.strip, stream)):
            key = item_id(row)
            if key in wanted:
                repeated += key in found
                found[key] = row
    if repeated:
        raise ValueError(f"{repeated} flagged ids occur twice in the protected rows")
    if len(found) < len(wanted):
        missing = len(wanted) - len(found)
        raise ValueError(f"{missing} flagged ids are not in the protected rows")
    return found


def raw_rows(
    sources_dir: Path, rows: dict[str, dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, list[str]]]:
    """The snapshot row behind each flagged row (row `index` of `data_rows(file)`,
    checked against the protected row's texts) and the SHA-256 of each file read."""
    wanted: dict[tuple[str, str], dict[int, str]] = defaultdict(dict)
    for key, row in rows.items():
        source, _, relative = row["task"].partition("/")
        wanted[(source, relative)][int(row["source_item_id"])] = key
    raws: dict[str, Any] = {}
    digests: dict[str, list[str]] = defaultdict(list)
    for (source, relative), keys in sorted(wanted.items()):
        path = sources_dir / source / relative
        if not path.is_file():
            raise ValueError(f"{path}: no such snapshot file")
        count = 0
        for index, value in enumerate(data_rows(path)):
            count += 1
            if index in keys:
                raws[keys[index]] = value
        if any(not 0 <= index < count for index in keys):
            raise ValueError(f"{path}: row index out of range ({count} rows)")
        for key in keys.values():
            texts = {t for t in strings(raws[key]) if len(t.strip()) >= MIN_CHARS}
            if sorted(texts) != rows[key].get("overlap_texts"):
                raise ValueError(f"{path}: a raw row differs from its protected row")
        digests[source].append(sha256(path))
    return raws, {source: sorted(values) for source, values in digests.items()}


def shingle_index(
    candidates: list[dict[str, Any]], wanted: set[int]
) -> tuple[dict[int, list[int]], dict[int, int]]:
    """Candidates by shingle for the `wanted` shingles, and the shingle-set size of
    those candidates."""
    postings: dict[int, list[int]] = defaultdict(list)
    sizes: dict[int, int] = {}
    for number, candidate in enumerate(candidates):
        own = features(candidate)
        common = own & wanted
        if common:
            sizes[number] = len(own)
            for key in common:
                postings[key].append(number)
    return postings, sizes


def id_index(candidates: list[dict[str, Any]]) -> dict[tuple[str, str], set[int]]:
    """Candidates by (source, group id | source item id | its part before ':')."""
    index: dict[tuple[str, str], set[int]] = defaultdict(set)
    for number, candidate in enumerate(candidates):
        item = str(candidate["source_item_id"])
        for value in (str(candidate["group_id"]), item, item.split(":", 1)[0]):
            index[(candidate["source"], value)].add(number)
    return index


def links(
    candidates: list[dict[str, Any]],
    rows: dict[str, dict[str, Any]],
    raws: dict[str, Any],
) -> tuple[dict[int, set[str]], int]:
    """The link kinds of each linked candidate, and the flagged rows linking none."""
    flagged = {key: features(row) for key, row in rows.items()}
    wanted: set[int] = set().union(*flagged.values())
    postings, sizes = shingle_index(candidates, wanted)
    by_id = id_index(candidates)
    kinds: dict[int, set[str]] = defaultdict(set)
    unlinked = 0
    for key, own in sorted(flagged.items()):
        shared = Counter(number for k in own for number in postings.get(k, ()))
        text = {
            n for n, count in shared.items() if text_linked(count, len(own), sizes[n])
        }
        source = rows[key]["task"].partition("/")[0]
        ids = {n for v in id_values(raws[key]) for n in by_id.get((source, v), ())}
        for number in text:
            kinds[number].add("text")
        for number in ids:
            kinds[number].add("id")
        unlinked += not text and not ids
    return dict(kinds), unlinked


def closure(candidates: list[dict[str, Any]], linked: Iterable[int]) -> set[str]:
    """Ids of every candidate in the source group of a linked candidate."""
    groups = {(candidates[n]["source"], candidates[n]["group_id"]) for n in linked}
    return {
        item_id(row) for row in candidates if (row["source"], row["group_id"]) in groups
    }


def read_previous(path: Path | None) -> dict[str, list[str]]:
    if path is None:
        return {"candidates": [], "protected_rows": []}
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("schema") != RETIRED_SCHEMA:
        raise ValueError(f"{path}: not a {RETIRED_SCHEMA} file")
    return {name: list(document[name]) for name in ("candidates", "protected_rows")}


def bounds(
    candidates: list[dict[str, Any]],
    retired: set[str],
    build_hits: dict[str, dict[str, Any]],
    manifest: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Per task: C1 items, the most of them that can be retired whatever the salt,
    and the standard-error inflation that allows."""
    per_group: dict[str, Counter] = defaultdict(Counter)
    for row in candidates:
        key = item_id(row)
        if key in retired and excluded(build_hits.get(key)) is None:
            per_group[row["task"]][row["group_id"]] += 1
    config = manifest["config"]["tasks"]
    tasks = {}
    for task, entry in sorted(manifest["tasks"].items()):
        items, cap = entry["selected"], config.get(task, {}).get("group_cap", 1)
        bound = min(items, sum(min(cap, n) for n in per_group[task].values()))
        tasks[task] = {
            "items": items,
            "retired_upper_bound": bound,
            "se_inflation_upper_bound": (
                round(math.sqrt(items / (items - bound)), 4) if bound < items else None
            ),
        }
    return tasks


def power(tasks: dict[str, dict[str, Any]], items: int) -> dict[str, Any]:
    """Totals, and the standard-error inflation of C1 (the mean of per-task macro-F1):
    the largest task factor, and the factor at equal item variance."""
    scored = [cell for cell in tasks.values() if cell["items"]]
    factors = [cell["se_inflation_upper_bound"] for cell in scored]
    left = [
        (cell["items"], cell["items"] - cell["retired_upper_bound"])
        for cell in scored
        if cell["se_inflation_upper_bound"] is not None
    ]
    equal = (
        math.sqrt(sum(1 / m for _, m in left) / sum(1 / n for n, _ in left))
        if left
        else None
    )
    return {
        "tasks": tasks,
        "totals": {
            "items_v_prev": sum(cell["items"] for cell in tasks.values()),
            "build_manifest_items": items,
            "retired_items_upper_bound": sum(
                cell["retired_upper_bound"] for cell in tasks.values()
            ),
            "items_lower_bound": sum(
                cell["items"] - cell["retired_upper_bound"] for cell in tasks.values()
            ),
        },
        "c1_se_inflation_upper_bound": (
            max(factors) if factors and None not in factors else None
        ),
        "c1_se_inflation_equal_variance": None if equal is None else round(equal, 4),
    }


def run(args: argparse.Namespace) -> tuple[bytes, dict[str, Any]]:
    """The RETIRED.json bytes and the count-only receipt."""
    raw = args.build_manifest.read_bytes()
    manifest = json.loads(raw)
    candidates, candidates_sha = load_candidates(
        args.candidates_dir, manifest["candidates_sha256"]
    )
    build_sha = sha256(args.build_hits)
    if build_sha != manifest["overlap_hits_sha256"]:
        raise ValueError(f"{args.build_hits}: SHA-256 differs from the build manifest")
    build_hits = {hit["id"]: hit for hit in read_jsonl(args.build_hits)}
    protected_sha = sha256(args.protected)
    verdicts = non_clean(args.hits, protected_sha)
    before = set(non_clean(args.baseline, protected_sha)) if args.baseline else None
    earlier = read_previous(args.previous)
    rows = protected_rows(args.protected, set(verdicts))
    raws, snapshot_sha = raw_rows(args.sources_dir, rows)
    kinds, unlinked = links(candidates, rows, raws)
    retired = closure(candidates, kinds) | set(earlier["candidates"])
    document = {
        "schema": RETIRED_SCHEMA,
        "version": args.version,
        "candidates": sorted(retired),
        "protected_rows": sorted(set(rows) | set(earlier["protected_rows"])),
    }
    data = (json.dumps(document, sort_keys=True, indent=1) + "\n").encode("utf-8")
    known = {item_id(row): row for row in candidates}
    kept = [known[key] for key in document["candidates"] if key in known]
    kind = Counter("both" if len(k) > 1 else next(iter(k)) for k in kinds.values())
    receipt = {
        "schema": SCHEMA,
        "version": args.version,
        "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "params": {
            "shingle": PARAMS.shingle,
            "min_tokens": PARAMS.min_tokens,
            "review": PARAMS.review,
            "overlap": PARAMS.overlap,
            "cjk_split": PARAMS.cjk_split,
        },
        "inputs_sha256": {
            "hits": [sha256(path) for path in args.hits],
            "baselines": [sha256(path) for path in args.baseline],
            "protected": protected_sha,
            "candidates": candidates_sha,
            "build_manifest": hashlib.sha256(raw).hexdigest(),
            "build_hits": build_sha,
            "previous": sha256(args.previous) if args.previous else None,
            "snapshot_files": snapshot_sha,
        },
        "flagged": {
            "rows": len(verdicts),
            "by_verdict": tally(verdicts.values()),
            "by_source": tally(key.partition("/")[0] for key in verdicts),
            "recurring": None if before is None else len(verdicts.keys() & before),
            "new": None if before is None else len(verdicts.keys() - before),
            "without_link": unlinked,
        },
        "linked": {
            "candidates": len(kinds),
            "by_kind": {name: kind[name] for name in KINDS},
            "by_source": tally(candidates[number]["source"] for number in kinds),
        },
        "retired": {
            "candidates": len(document["candidates"]),
            "protected_rows": len(document["protected_rows"]),
            "sha256": hashlib.sha256(data).hexdigest(),
            "groups_by_source": tally(
                source for source, _ in {(r["source"], r["group_id"]) for r in kept}
            ),
            "candidates_by_task": tally(row["task"] for row in kept),
            "previous": (
                None
                if args.previous is None
                else {
                    "candidates": len(set(earlier["candidates"])),
                    "protected_rows": len(set(earlier["protected_rows"])),
                    "candidates_not_in_build": len(
                        set(earlier["candidates"]) - known.keys()
                    ),
                }
            ),
        },
        "power": power(
            bounds(candidates, retired, build_hits, manifest), manifest["items"]
        ),
    }
    return data, receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("build", help="write RETIRED.json and its receipt")
    command.add_argument("--hits", type=Path, action="append", required=True)
    command.add_argument("--protected", type=Path, required=True)
    command.add_argument("--sources-dir", type=Path, required=True)
    command.add_argument("--candidates-dir", type=Path, required=True)
    command.add_argument("--build-manifest", type=Path, required=True)
    command.add_argument("--build-hits", type=Path, required=True)
    command.add_argument("--baseline", type=Path, action="append", default=[])
    command.add_argument("--previous", type=Path)
    command.add_argument("--version", required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.resolve() == args.receipt.resolve():
        parser.error("output and receipt must be different files")
    for path in (args.output, args.receipt):
        if path.exists():
            parser.error(f"refusing to overwrite {path}")
    try:
        data, receipt = run(args)
    except ValueError as error:
        raise SystemExit(f"retire: refused: {error}") from error
    text = (json.dumps(receipt, indent=1, sort_keys=True) + "\n").encode("utf-8")
    for path, content in ((args.output, data), (args.receipt, text)):
        path.parent.mkdir(parents=True, exist_ok=True)
        write_new(path, content)
    summary = {
        "flagged": receipt["flagged"]["rows"],
        "linked": receipt["linked"]["candidates"],
        "retired_candidates": receipt["retired"]["candidates"],
        "retired_protected_rows": receipt["retired"]["protected_rows"],
        "retired_items_upper_bound": receipt["power"]["totals"][
            "retired_items_upper_bound"
        ],
        "c1_se_inflation_upper_bound": receipt["power"]["c1_se_inflation_upper_bound"],
        "retired_sha256": receipt["retired"]["sha256"],
        "receipt_sha256": hashlib.sha256(text).hexdigest(),
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
