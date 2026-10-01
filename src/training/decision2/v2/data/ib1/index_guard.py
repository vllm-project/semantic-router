"""G0: row-level exclusion of IB1 candidates against every Decision Index suite row (prereg §3).

    python3 -m v2.data.ib1.index_guard reference --suite ROWS.jsonl.gz [--suite ...] --out-dir REF
    python3 -m v2.data.ib1.index_guard scan --reference REF --candidates T --candidates D --out-dir OUT
    python3 -m v2.data.ib1.index_guard controls --suite ... --reference REF --out RECEIPT

Matching follows the external Index report's audit: NFKC + casefold + ``\\w+`` tokens. Index side: every string
leaf of a suite row's ``state`` and ``questions`` (instructions and option texts). IB1 side: every string leaf of
``state``, plus instructions and option descriptions unless they are fixed template strings (reported apart).

E   raw equality of a leaf of >= 20 characters;
N1  equality of a whole normalized leaf of >= 6 tokens;
N2  equality of a normalized sentence or whole leaf of >= 8 tokens (either side);
G   any shared word 13-gram of normalized tokens (within one leaf).

Any non-template hit drops the candidate's whole group. Hashes are 64-bit blake2b values; the reference keeps,
per hash, the Index benchmark of its first occurrence, so private receipts can name the benchmark of a hit. Public
receipts hold counts per IB1 family and rule only.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import multiprocessing
import os
import re
import unicodedata
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from v2.data.ib1.families import TEMPLATE_STRINGS
from v2.data.sources.common import sha

RULES = ("E", "N1", "N2", "G")
MIN_RAW_CHARS = 20
MIN_LEAF_TOKENS = 6
MIN_SENTENCE_TOKENS = 8
GRAM = 13
TOKEN_RE = re.compile(r"\w+")
SENTENCE_RE = re.compile(r"(?<=[.!?\u3002\uff01\uff1f])\s+|\n+")
PERSON = {rule: f"ib1-g0-{rule}".encode() for rule in RULES}
SKIP_QUESTION_KEYS = frozenset({"type"})
CONTROL_SALT = "ib1-g0-control-v1"
CONTROLS = 2000
CONTROL_PREFIX = "Note: this is a control item. "
_WORK: dict[str, Any] = {}


def h64(text: str, rule: str) -> int:
    return int.from_bytes(
        hashlib.blake2b(
            text.encode("utf-8"), digest_size=8, person=PERSON[rule]
        ).digest(),
        "big",
    )


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(unicodedata.normalize("NFKC", text).casefold())


def leaves(value: Any, skip: frozenset[str] = frozenset()) -> Iterator[str]:
    if isinstance(value, str):
        if value.strip():
            yield value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            if key not in skip:
                yield from leaves(item, skip)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from leaves(item, skip)


def unit_hashes(leaf: str) -> dict[str, set[int]]:
    """Hashes of one text leaf under every rule."""
    out: dict[str, set[int]] = {rule: set() for rule in RULES}
    if len(leaf) >= MIN_RAW_CHARS:
        out["E"].add(h64(leaf, "E"))
    words = tokens(leaf)
    if len(words) >= MIN_LEAF_TOKENS:
        out["N1"].add(h64(" ".join(words), "N1"))
    if len(words) >= MIN_SENTENCE_TOKENS:
        out["N2"].add(h64(" ".join(words), "N2"))
    for sentence in SENTENCE_RE.split(leaf):
        part = tokens(sentence)
        if len(part) >= MIN_SENTENCE_TOKENS:
            out["N2"].add(h64(" ".join(part), "N2"))
    for start in range(0, len(words) - GRAM + 1):
        out["G"].add(h64(" ".join(words[start : start + GRAM]), "G"))
    return out


def suite_leaves(row: Mapping[str, Any]) -> list[str]:
    return list(leaves(row.get("state"))) + list(
        leaves(row.get("questions"), SKIP_QUESTION_KEYS)
    )


def suite_name(row: Mapping[str, Any]) -> str:
    return str(
        row.get("benchmark")
        or row.get("family")
        or str(row.get("id", "")).split(":")[0]
    )


def candidate_leaves(row: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    """(data leaves, template leaves) of an IB1 row."""
    data = list(leaves(row["state"]))
    template = []
    extra = list(leaves(row["instructions"])) + [
        text for option in row["options"] for text in leaves(option.get("description"))
    ]
    for text in extra:
        (template if text in TEMPLATE_STRINGS else data).append(text)
    return data, template


# --------------------------------------------------------------------------- reference


def read_suite(paths: Sequence[Path]) -> Iterator[str]:
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    yield line


def _reference_chunk(
    lines: list[str],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[str], int]:
    names: dict[str, int] = {}
    hashes: dict[str, list[int]] = {rule: [] for rule in RULES}
    codes: dict[str, list[int]] = {rule: [] for rule in RULES}
    detectable = 0
    for line in lines:
        row = json.loads(line)
        code = names.setdefault(suite_name(row), len(names))
        row_has_gram = False
        for leaf in suite_leaves(row):
            for rule, values in unit_hashes(leaf).items():
                hashes[rule].extend(values)
                codes[rule].extend([code] * len(values))
                row_has_gram |= rule == "G" and bool(values)
        detectable += row_has_gram
    order = sorted(names, key=names.get)
    return (
        {r: np.array(v, dtype=np.uint64) for r, v in hashes.items()},
        {r: np.array(v, dtype=np.int32) for r, v in codes.items()},
        order,
        detectable,
    )


def chunks(items: list[Any], size: int) -> list[list[Any]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def build_reference(
    paths: Sequence[Path], workers: int
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[str], dict[str, Any]]:
    lines = list(read_suite(paths))
    with multiprocessing.get_context("fork").Pool(workers) as pool:
        parts = pool.map(_reference_chunk, chunks(lines, 500), chunksize=1)
    names: list[str] = []
    index: dict[str, int] = {}
    all_hashes: dict[str, list[np.ndarray]] = {rule: [] for rule in RULES}
    all_codes: dict[str, list[np.ndarray]] = {rule: [] for rule in RULES}
    detectable = 0
    for hashes, codes, local, count in parts:
        remap = np.array(
            [index.setdefault(name, len(index)) for name in local] or [0],
            dtype=np.int32,
        )
        for rule in RULES:
            all_hashes[rule].append(hashes[rule])
            all_codes[rule].append(
                remap[codes[rule]] if len(codes[rule]) else codes[rule]
            )
        detectable += count
    names = sorted(index, key=index.get)
    ref_hashes, ref_codes = {}, {}
    for rule in RULES:
        values = (
            np.concatenate(all_hashes[rule])
            if all_hashes[rule]
            else np.array([], dtype=np.uint64)
        )
        code = (
            np.concatenate(all_codes[rule])
            if all_codes[rule]
            else np.array([], dtype=np.int32)
        )
        unique, first = np.unique(values, return_index=True)
        ref_hashes[rule], ref_codes[rule] = unique, code[first].astype(np.int32)
    stats = {
        "rows": len(lines),
        "rows_with_a_13gram_leaf": detectable,
        "units": {rule: int(len(ref_hashes[rule])) for rule in RULES},
    }
    return ref_hashes, ref_codes, names, stats


def save_reference(
    out: Path,
    hashes: Mapping[str, np.ndarray],
    codes: Mapping[str, np.ndarray],
    names: list[str],
    meta: Mapping[str, Any],
) -> None:
    out.mkdir(mode=0o700)
    for rule in RULES:
        np.save(out / f"{rule}.hash.npy", hashes[rule])
        np.save(out / f"{rule}.code.npy", codes[rule])
    fd = os.open(out / "reference.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump({**meta, "benchmarks": names}, stream, indent=1, sort_keys=True)


def load_reference(
    path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    hashes = {rule: np.load(path / f"{rule}.hash.npy") for rule in RULES}
    codes = {rule: np.load(path / f"{rule}.code.npy") for rule in RULES}
    return (
        hashes,
        codes,
        json.loads((path / "reference.json").read_text(encoding="utf-8")),
    )


# --------------------------------------------------------------------------- matching


def hits(units: Iterable[str]) -> dict[str, list[int]]:
    """Rule -> Index benchmark codes of the matched hashes for these text leaves."""
    ref_hashes, ref_codes = _WORK["hashes"], _WORK["codes"]
    wanted: dict[str, set[int]] = {rule: set() for rule in RULES}
    for leaf in units:
        for rule, values in unit_hashes(leaf).items():
            wanted[rule] |= values
    found: dict[str, list[int]] = {}
    for rule in RULES:
        if not wanted[rule] or not len(ref_hashes[rule]):
            continue
        query = np.fromiter(wanted[rule], dtype=np.uint64, count=len(wanted[rule]))
        at = np.searchsorted(ref_hashes[rule], query)
        at = np.minimum(at, len(ref_hashes[rule]) - 1)
        match = ref_hashes[rule][at] == query
        if match.any():
            found[rule] = sorted({int(c) for c in ref_codes[rule][at[match]]})
    return found


def _scan_chunk(lines: list[str]) -> list[dict[str, Any]]:
    out = []
    for line in lines:
        row = json.loads(line)
        data, template = candidate_leaves(row)
        found, found_template = hits(data), hits(template)
        if found or found_template:
            out.append(
                {
                    "id": row["id"],
                    "group_id": row["group_id"],
                    "family": row["family"],
                    "split": row["split"],
                    "rules": sorted(found),
                    "codes": sorted({c for v in found.values() for c in v}),
                    "template_rules": sorted(found_template),
                    "template_codes": sorted(
                        {c for v in found_template.values() for c in v}
                    ),
                }
            )
    return out


def scan(
    candidate_paths: Sequence[Path], reference: Path, workers: int
) -> tuple[list[dict[str, Any]], collections.Counter, dict[str, Any]]:
    hashes, codes, meta = load_reference(reference)
    _WORK.update(hashes=hashes, codes=codes)
    lines = [
        line
        for path in candidate_paths
        for line in path.read_bytes().decode("utf-8").split("\n")
        if line.strip()
    ]
    population: collections.Counter = collections.Counter()
    for line in lines:
        row = json.loads(line)
        population[(row["family"], row["split"])] += 1
    with multiprocessing.get_context("fork").Pool(workers) as pool:
        parts = pool.map(_scan_chunk, chunks(lines, 400), chunksize=1)
    return [item for part in parts for item in part], population, meta


def controls(paths: Sequence[Path], reference: Path) -> dict[str, Any]:
    """Suite rows rendered as candidate rows (exact and perturbed copies) must all be flagged."""
    hashes, codes, _ = load_reference(reference)
    _WORK.update(hashes=hashes, codes=codes)
    eligible = []
    for line in read_suite(paths):
        row = json.loads(line)
        units = suite_leaves(row)
        if any(len(tokens(u)) >= GRAM for u in units):
            eligible.append((sha(f"{CONTROL_SALT}:{row.get('id')}"), units))
    eligible.sort()
    picked = [units for _, units in eligible[:CONTROLS]]
    exact = sum(bool(hits(units)) for units in picked)

    def perturb(units: list[str]) -> list[str]:
        out = []
        for i, unit in enumerate(units):
            text = " ".join(unit.split()).swapcase().replace(",", " ").replace(";", " ")
            text = re.sub(r"\s", "  ", text)
            out.append((CONTROL_PREFIX if i == 0 else "") + text)
        return out

    perturbed = 0
    for units in picked:
        found = hits(perturb(units))
        perturbed += bool(set(found) & {"N1", "N2", "G"})
    return {
        "controls": len(picked),
        "exact_copies_flagged": exact,
        "perturbed_copies_flagged": perturbed,
        "pass": exact == len(picked)
        and perturbed == len(picked)
        and len(picked) == CONTROLS,
    }


# --------------------------------------------------------------------------- receipts


def write_new(path: Path, text: str) -> str:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(text)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def receipts(
    flagged: Sequence[Mapping[str, Any]],
    population: Mapping[tuple[str, str], int],
    meta: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    names = meta["benchmarks"]
    by_family: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter
    )
    groups: dict[str, set[str]] = collections.defaultdict(set)
    drop: set[str] = set()
    for item in flagged:
        family = item["family"]
        if item["rules"]:
            drop.add(item["group_id"])
            groups[family].add(item["group_id"])
            by_family[family]["rows"] += 1
            for rule in item["rules"]:
                by_family[family][f"rows_{rule}"] += 1
        if item["template_rules"]:
            by_family[family][
                "template_only_rows" if not item["rules"] else "template_and_data_rows"
            ] += 1
    families = sorted({family for family, _ in population})
    public = {
        "schema": "decision2.ib1.index-guard.v1",
        "rules": {
            "E": f"raw leaf equality, leaves >= {MIN_RAW_CHARS} characters",
            "N1": f"normalized whole-leaf equality, >= {MIN_LEAF_TOKENS} tokens",
            "N2": f"normalized sentence or leaf equality, >= {MIN_SENTENCE_TOKENS} tokens",
            "G": f"shared word {GRAM}-gram",
        },
        "reference": {
            k: meta[k]
            for k in ("rows", "rows_with_a_13gram_leaf", "units", "inputs")
            if k in meta
        },
        "candidates": {
            f: {s: population.get((f, s), 0) for s in ("train", "select")}
            for f in families
        },
        "flagged": {
            f: {**dict(sorted(by_family[f].items())), "groups_dropped": len(groups[f])}
            for f in families
        },
        "groups_dropped": len(drop),
    }
    private = {
        "schema": "decision2.ib1.index-guard-private.v1",
        "benchmarks": names,
        "flagged": [
            {
                **item,
                "benchmarks": [names[c] for c in item["codes"]],
                "template_benchmarks": [names[c] for c in item["template_codes"]],
            }
            for item in flagged
        ],
        "hits_by_benchmark": dict(
            sorted(
                collections.Counter(
                    names[c] for item in flagged for c in item["codes"]
                ).items()
            )
        ),
    }
    return public, private, sorted(drop)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("reference")
    one.add_argument("--suite", type=Path, action="append", required=True)
    one.add_argument("--out-dir", type=Path, required=True)
    one.add_argument("--workers", type=int, default=32)
    two = sub.add_parser("scan")
    two.add_argument("--reference", type=Path, required=True)
    two.add_argument("--candidates", type=Path, action="append", required=True)
    two.add_argument("--out-dir", type=Path, required=True)
    two.add_argument("--workers", type=int, default=32)
    three = sub.add_parser("controls")
    three.add_argument("--suite", type=Path, action="append", required=True)
    three.add_argument("--reference", type=Path, required=True)
    three.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "reference":
        hashes, codes, names, stats = build_reference(args.suite, args.workers)
        inputs = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.suite
        }
        save_reference(args.out_dir, hashes, codes, names, {**stats, "inputs": inputs})
        print(json.dumps(stats))
        return 0
    if args.command == "scan":
        flagged, population, meta = scan(args.candidates, args.reference, args.workers)
        public, private, drop = receipts(flagged, population, meta)
        public["candidate_inputs"] = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.candidates
        }
        args.out_dir.mkdir(mode=0o700)
        public["drop_groups_sha256"] = write_new(
            args.out_dir / "drop-groups.txt", "".join(g + "\n" for g in drop)
        )
        write_new(
            args.out_dir / "index-guard.public.json",
            json.dumps(public, indent=1, sort_keys=True) + "\n",
        )
        write_new(
            args.out_dir / "index-guard.private.json",
            json.dumps(private, indent=1, sort_keys=True) + "\n",
        )
        print(json.dumps({"groups_dropped": public["groups_dropped"]}))
        return 0
    result = controls(args.suite, args.reference)
    write_new(args.out, json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(json.dumps(result))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
