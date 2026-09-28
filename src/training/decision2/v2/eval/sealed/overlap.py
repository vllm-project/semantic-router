"""Screen C1 candidates for text overlap with training data, eval panels and peer sets.

    python3 -m v2.eval.sealed.overlap scan --protected <candidates.jsonl> \
        [--manifest <corpora manifest.json> [--labels GLOB,...]] \
        [--corpus LABEL=PATH_OR_GLOB ...] --workers N \
        --output <receipt.json> --hits <private hits.jsonl>

A candidate (id `task|source_item_id`) is scanned through its texts: `overlap_texts`
plus every string leaf of `state`, deduplicated after normalisation (NFKC, casefold,
collapsed whitespace). On both sides a string holding a JSON object or array is
decoded into its string leaves first. Tokens are `\\w+` runs, except that every CJK
ideograph or kana is a token of its own (unspaced Chinese or Japanese would otherwise
be a handful of tokens). A text of >= 8 tokens yields every 8-token window, one of
5-7 tokens a single shingle of all its tokens, a shorter one none; a shingle is the
8-byte blake2b of its space-joined tokens. Only candidate shingles are held in memory.

Every string leaf of every corpus row is shingled the same way (strings under 5
characters are skipped). Per candidate and corpus label the scan keeps the row with
the largest containment (the share of the candidate's distinct shingles found in that
one row) and the longest exact match: a candidate text of >= 20 normalised characters
(and at least one token) equal to a corpus string, or one of >= 12 tokens contained
in a corpus string. The 20-character minimum applies to exact matching on both sides;
shingles need none, so a short text (e.g. a few CJK characters) still matches the same
text in a corpus. An exact match records the matched text's token count and how many
candidates share that text, so template-like matches can be told apart.

    OVERLAP  an exact match, or containment >= 0.5
    REVIEW   0.2 <= containment < 0.5 (or, with --exact-min-tokens N, an exact match
             of a text shorter than N tokens only)
    CLEAN    otherwise (also candidates without a screenable text; see the receipt)

Rows: jsonl and jsonl.gz lines (row = 0-based line number; a line that is not JSON is
scanned as text), the items of a .json list (of a dict: the items of its list values
and its other values; a dict without containers is one row; a .json file that is not
one document is read as JSON lines), parquet rows (text columns only), csv/tsv records
(the header is record 0). A directory stands for its supported files outside dot
directories. The receipt holds counts only; the private hits file has one line per
candidate with its verdict and best locations (label, file, row), never text.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import fnmatch
import gc
import glob
import gzip
import hashlib
import json
import multiprocessing
import os
import re
import resource
import sys
import threading
import time
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any, NamedTuple

from v2.eval.sealed.schema import normalized

SCHEMA = "dev2-sealed-c1-overlap/1"
VERDICTS = ("OVERLAP", "REVIEW", "CLEAN")
EXACT_KINDS = ("equal", "contains")
TOKEN_BINS = ((4, "0-4"), (7, "5-7"), (11, "8-11"), (19, "12-19"))
SUFFIXES = {
    ".jsonl.gz": "jsonl.gz",
    ".jsonl": "jsonl",
    ".json": "json",
    ".parquet": "parquet",
    ".csv": "csv",
    ".tsv": "tsv",
}
_CJK = "\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0003134f"
_WORDS = re.compile(r"\w+")
_WORDS_CJK = re.compile(rf"(?=\w)[{_CJK}]|[^\W{_CJK}]+")
_STATE: dict[str, Any] = {}


@dataclasses.dataclass(frozen=True)
class Params:
    shingle: int = 8
    min_tokens: int = 5
    contains_tokens: int = 12
    min_chars: int = 20
    overlap: float = 0.5
    review: float = 0.2
    exact_min_tokens: int = 0
    cjk_split: bool = True
    chunk_bytes: int = 16 << 20

    def __post_init__(self) -> None:
        if not 1 <= self.min_tokens <= self.shingle:
            raise ValueError("need 1 <= min_tokens <= shingle")
        if self.contains_tokens < self.shingle + 2:
            raise ValueError("contains_tokens needs an interior window")
        if not 0 < self.review <= self.overlap <= 1:
            raise ValueError("need 0 < review <= overlap <= 1")
        if self.min_chars < 1 or self.chunk_bytes < 1 or self.exact_min_tokens < 0:
            raise ValueError("min_chars, chunk_bytes, exact_min_tokens out of range")


def tokens(norm: str, params: Params = Params()) -> list[str]:
    return (_WORDS_CJK if params.cjk_split else _WORDS).findall(norm)


def shingle_key(window: Sequence[str]) -> int:
    data = " ".join(window).encode("utf-8", "surrogatepass")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def shingles(words: Sequence[str], params: Params = Params()) -> set[int]:
    size = params.shingle
    if len(words) >= size:
        return {shingle_key(words[i : i + size]) for i in range(len(words) - size + 1)}
    if len(words) >= params.min_tokens:
        return {shingle_key(words)}
    return set()


def string_leaves(value: Any) -> Iterator[str]:
    """String values at any depth (not dict keys); JSON-container strings are decoded."""
    if isinstance(value, str):
        if value[:1] in ("{", "[") and value[-1:] in ("}", "]"):
            try:
                decoded = json.loads(value)
            except (ValueError, RecursionError):
                decoded = None
            if isinstance(decoded, (dict, list)):
                yield from string_leaves(decoded)
                return
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from string_leaves(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from string_leaves(child)


def verdict(
    containment: float, exact_tokens: int | None = None, params: Params = Params()
) -> str:
    """`exact_tokens`: tokens of the longest exact match, None without one; an exact
    match shorter than `exact_min_tokens` only makes a candidate REVIEW."""
    exact = exact_tokens is not None
    if containment >= params.overlap or (
        exact and exact_tokens >= params.exact_min_tokens
    ):
        return "OVERLAP"
    return "REVIEW" if exact or containment >= params.review else "CLEAN"


def _members(value: int | tuple[int, ...]) -> tuple[int, ...]:
    return (value,) if type(value) is int else value


@dataclasses.dataclass
class Protected:
    """Candidate shingle index: `index` and `exact` map to one candidate or a tuple."""

    ids: list[str]
    sources: list[str]
    tasks: list[str]
    sizes: list[int]
    index: dict[int, Any]
    exact: dict[str, Any]
    anchors: dict[int, tuple[tuple[int, str, int], ...]]
    stats: dict[str, int]
    sha256: str


def load_protected(path: Path, params: Params = Params()) -> Protected:
    raw = path.read_bytes()
    ids: list[str] = []
    sources: list[str] = []
    tasks: list[str] = []
    sizes: list[int] = []
    postings: dict[int, Any] = {}
    exact: dict[str, Any] = {}
    long: list[tuple[int, str]] = []
    seen: set[str] = set()
    stats: Counter = Counter()
    for number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            raise ValueError(f"{path}:{number}: blank line")
        row = json.loads(line)
        task = row.get("task") if isinstance(row, dict) else None
        item = row.get("source_item_id") if isinstance(row, dict) else None
        if not (isinstance(task, str) and task and isinstance(item, str) and item):
            raise ValueError(f"{path}:{number}: needs string task and source_item_id")
        candidate_id = f"{task}|{item}"
        if candidate_id in seen:
            raise ValueError(f"{path}:{number}: duplicate candidate id")
        seen.add(candidate_id)
        candidate = len(ids)
        ids.append(candidate_id)
        sources.append(str(row.get("source", "")))
        tasks.append(task)
        leaves = [
            *string_leaves(row.get("overlap_texts") or []),
            *string_leaves(row.get("state")),
        ]
        texts = dict.fromkeys(normalized(text) for text in leaves)
        texts.pop("", None)
        own: set[int] = set()
        screenable = False
        for norm in texts:
            words = tokens(norm, params)
            own |= shingles(words, params)
            if len(norm) >= params.min_chars and words:
                screenable = True
                exact.setdefault(norm, []).append(candidate)
                stats["exact_texts"] += 1
                if len(words) >= params.contains_tokens:
                    long.append((candidate, norm))
        for key in own:
            postings.setdefault(key, []).append(candidate)
        sizes.append(len(own))
        stats["texts"] += len(texts)
        stats["shingles"] += len(own)
        stats["without_shingles"] += not own
        stats["unscreenable"] += not own and not screenable
    size = params.shingle
    anchors: dict[int, list[tuple[int, str, int]]] = {}
    for candidate, norm in long:
        words = tokens(norm, params)
        interior = [
            shingle_key(words[i : i + size]) for i in range(1, len(words) - size)
        ]
        key = min(interior, key=lambda k: (len(postings[k]), k))
        anchors.setdefault(key, []).append((candidate, norm, len(words)))
    for mapping in (postings, exact):
        for key, members in mapping.items():
            mapping[key] = members[0] if len(members) == 1 else tuple(members)
    stats.update(
        candidates=len(ids),
        sources=len(set(sources)),
        tasks=len(set(tasks)),
        contains_texts=len(long),
        distinct_shingles=len(postings),
        shared_shingles=sum(type(v) is tuple for v in postings.values()),
        shared_exact_texts=sum(type(v) is tuple for v in exact.values()),
    )
    return Protected(
        ids,
        sources,
        tasks,
        sizes,
        postings,
        exact,
        {key: tuple(value) for key, value in anchors.items()},
        dict(sorted(stats.items())),
        hashlib.sha256(raw).hexdigest(),
    )


def file_kind(name: str) -> str | None:
    lower = name.lower()
    for suffix, kind in SUFFIXES.items():
        if lower.endswith(suffix):
            return kind
    return None


def expand(pattern: str) -> list[str]:
    """Files of one corpus path, directory or glob (`**` recursive), sorted."""
    magic = any(char in pattern for char in "*?[")
    matches = sorted(glob.glob(pattern, recursive=True)) if magic else [pattern]
    files: list[str] = []
    for match in matches:
        path = Path(match)
        if path.is_dir():
            files.extend(
                str(child)
                for child in sorted(path.rglob("*"))
                if child.is_file()
                and file_kind(child.name)
                and not any(p.startswith(".") for p in child.relative_to(path).parts)
            )
        elif path.is_file() and (not magic or file_kind(path.name)):
            files.append(str(path))
    return list(dict.fromkeys(files))


def parse_corpora(specs: Iterable[str]) -> list[tuple[str, list[str]]]:
    merged: dict[str, list[str]] = {}
    for spec in specs:
        label, separator, pattern = spec.partition("=")
        if not separator or not label or not pattern:
            raise ValueError(f"corpus must be LABEL=PATH_OR_GLOB, got {spec!r}")
        files = expand(pattern)
        if not files:
            raise ValueError(f"corpus {label}: nothing matches {pattern!r}")
        merged.setdefault(label, []).extend(files)
    return [(label, list(dict.fromkeys(files))) for label, files in merged.items()]


def manifest_corpora(
    path: Path, patterns: Sequence[str] = ()
) -> tuple[list[tuple[str, list[str]]], list[tuple[str, str, int]], str]:
    """(label, files) pairs of a corpus manifest, (path, sha256, bytes) checks, sha."""
    raw = path.read_bytes()
    corpora, checks = [], []
    for label, entry in json.loads(raw)["labels"].items():
        if patterns and not any(fnmatch.fnmatchcase(label, p) for p in patterns):
            continue
        files = entry["files"]
        corpora.append((label, [item["path"] for item in files]))
        checks.extend((item["path"], item["sha256"], item["bytes"]) for item in files)
    if not corpora:
        raise ValueError("no manifest label selected")
    return corpora, checks, hashlib.sha256(raw).hexdigest()


class _Task(NamedTuple):
    label: int
    file: int
    path: str
    kind: str
    start: int = 0
    stop: int = 0
    group: int = -1
    base: int = 0
    cost: int = 0
    part: int = 0
    parts: int = 1


def _parquet() -> Any:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise ImportError("reading .parquet corpora needs pyarrow") from error
    return parquet


def _parquet_layout(path: str) -> list[tuple[int, int]]:
    """(rows, uncompressed bytes) per row group."""
    metadata = _parquet().ParquetFile(path).metadata
    return [
        (metadata.row_group(i).num_rows, metadata.row_group(i).total_byte_size)
        for i in range(metadata.num_row_groups)
    ]


def plan(
    corpora: Sequence[tuple[str, Sequence[str]]],
    params: Params,
    mapper: Any = map,
) -> tuple[list[_Task], list[str], list[int]]:
    """Tasks in canonical order (label, file, chunk), plus file paths and sizes."""
    files = [path for _, paths in corpora for path in paths]
    kinds = [file_kind(path) for path in files]
    unsupported = [path for path, kind in zip(files, kinds) if kind is None]
    if unsupported:
        raise ValueError(f"unsupported corpus file {unsupported[0]}")
    parquet = [path for path, kind in zip(files, kinds) if kind == "parquet"]
    layouts = dict(zip(parquet, mapper(_parquet_layout, parquet)))
    labels = [label for label, (_, paths) in enumerate(corpora) for _ in paths]
    sizes = [os.path.getsize(path) for path in files]
    tasks: list[_Task] = []
    for file, (label, path, kind) in enumerate(zip(labels, files, kinds)):
        size = sizes[file]
        task = _Task(label, file, path, kind, 0, size, cost=size)
        if kind == "jsonl":
            for start in range(0, max(size, 1), params.chunk_bytes):
                stop = min(start + params.chunk_bytes, size)
                tasks.append(task._replace(start=start, stop=stop, cost=stop - start))
        elif kind == "parquet":
            tasks.extend(_parquet_tasks(task, layouts[path], params.chunk_bytes))
        elif kind == "jsonl.gz":
            parts = max(1, -(-size * 3 // params.chunk_bytes))
            tasks.extend(
                task._replace(cost=size * 6 // parts, part=part, parts=parts)
                for part in range(parts)
            )
        else:
            tasks.append(task)
    return tasks, files, sizes


def _parquet_tasks(
    task: _Task, layout: list[tuple[int, int]], chunk: int
) -> Iterator[_Task]:
    """Row ranges of about `chunk` uncompressed bytes within each row group."""
    base = 0
    for group, (rows, total) in enumerate(layout):
        step = max(1, rows * chunk // max(total, 1))
        for start in range(0, rows, step):
            stop = min(start + step, rows)
            cost = total * (stop - start) // max(rows, 1)
            yield task._replace(
                start=start, stop=stop, group=group, base=base, cost=cost
            )
        base += rows


def _decode_line(raw: bytes, counts: Counter) -> Any:
    try:
        return json.loads(raw)
    except (ValueError, RecursionError):
        counts["unparsed"] += 1
        return raw.decode("utf-8", "replace")


def _line_rows(
    stream: Any,
    start: int,
    stop: int | None,
    counts: Counter,
    part: int = 0,
    parts: int = 1,
) -> Iterator[tuple[int, Any]]:
    """Lines starting in [start, stop) as (line number from start, value); of those
    only the ones whose number is `part` modulo `parts`."""
    position = start
    if start:
        stream.seek(start - 1)
        if stream.read(1) != b"\n":
            position += len(stream.readline())
    line = 0
    while stop is None or position < stop:
        raw = stream.readline()
        if not raw:
            break
        position += len(raw)
        line += 1
        if (line - 1) % parts == part and raw.strip():
            yield line - 1, _decode_line(raw, counts)
    counts["lines"] = line


def _json_rows(path: str, counts: Counter) -> Iterator[tuple[int, Any]]:
    with open(path, "rb") as stream:
        data = stream.read()
    try:
        document = json.loads(data)
    except (ValueError, RecursionError):
        with open(path, "rb") as stream:
            yield from _line_rows(stream, 0, None, counts)
        return
    if isinstance(document, list):
        yield from enumerate(document)
    elif isinstance(document, dict) and any(
        isinstance(value, (dict, list)) for value in document.values()
    ):
        items = (
            item
            for value in document.values()
            for item in (value if isinstance(value, list) else [value])
        )
        yield from enumerate(items)
    else:
        yield 0, document


def _has_text(kind: Any) -> bool:
    import pyarrow as pa

    types = pa.types
    if types.is_string(kind) or types.is_large_string(kind):
        return True
    if getattr(types, "is_string_view", lambda _: False)(kind):
        return True
    if types.is_dictionary(kind):
        return _has_text(kind.value_type)
    if types.is_struct(kind):
        return any(_has_text(kind.field(i).type) for i in range(kind.num_fields))
    if types.is_map(kind):
        return _has_text(kind.key_type) or _has_text(kind.item_type)
    if hasattr(kind, "value_type"):
        return _has_text(kind.value_type)
    return False


def _parquet_rows(task: _Task) -> Iterator[tuple[int, Any]]:
    handle = _parquet().ParquetFile(task.path)
    columns = [field.name for field in handle.schema_arrow if _has_text(field.type)]
    if not columns:
        return
    offset = 0
    batches = handle.iter_batches(
        batch_size=1024, row_groups=[task.group], columns=columns, use_threads=False
    )
    for batch in batches:
        low = max(task.start - offset, 0)
        high = min(task.stop - offset, batch.num_rows)
        if low < high:
            rows = batch.slice(low, high - low).to_pylist()
            for i, row in enumerate(rows):
                yield task.base + offset + low + i, row
        offset += batch.num_rows
        if offset >= task.stop:
            break


def _csv_rows(path: str, delimiter: str) -> Iterator[tuple[int, Any]]:
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    with open(path, newline="", encoding="utf-8", errors="replace") as stream:
        yield from enumerate(csv.reader(stream, delimiter=delimiter))


def _task_rows(task: _Task, counts: Counter) -> Iterator[tuple[int, Any]]:
    if task.kind == "jsonl":
        with open(task.path, "rb") as stream:
            yield from _line_rows(stream, task.start, task.stop, counts)
    elif task.kind == "jsonl.gz":
        with gzip.open(task.path, "rb") as stream:
            yield from _line_rows(stream, 0, None, counts, task.part, task.parts)
    elif task.kind == "json":
        yield from _json_rows(task.path, counts)
    elif task.kind == "parquet":
        yield from _parquet_rows(task)
    else:
        yield from _csv_rows(task.path, "\t" if task.kind == "tsv" else ",")


def _indexed_keys(words: list[str], index: dict[int, Any], params: Params) -> set[int]:
    """The shingle keys of one corpus string that are candidate shingles."""
    size = params.shingle
    if len(words) < size:
        if len(words) < params.min_tokens:
            return set()
        size = len(words)
    blake, join, from_bytes = hashlib.blake2b, " ".join, int.from_bytes
    found = set()
    for i in range(len(words) - size + 1):
        data = join(words[i : i + size]).encode("utf-8", "surrogatepass")
        key = from_bytes(blake(data, digest_size=8).digest(), "big")
        if key in index:
            found.add(key)
    return found


def _scan_task(item: tuple[int, _Task]) -> tuple[Any, ...]:
    """Per candidate: (largest shingle count in one row, row) and the longest exact
    match (row, kind, tokens of the matched text, candidates sharing that text)."""
    number, task = item
    started = time.perf_counter()
    protected: Protected = _STATE["protected"]
    params: Params = _STATE["params"]
    index, exact, anchors = protected.index, protected.exact, protected.anchors
    split = (_WORDS_CJK if params.cjk_split else _WORDS).findall
    counts: Counter = Counter()
    best: dict[int, tuple[int, int]] = {}
    first: dict[int, tuple[int, int, int, int]] = {}
    for row, value in _task_rows(task, counts):
        counts["rows"] += 1
        found: set[int] = set()
        for text in string_leaves(value):
            if len(text) < params.min_tokens:
                continue
            norm = normalized(text)
            words = split(norm)
            long = len(norm) >= params.min_chars
            if not long and len(words) < params.min_tokens:
                continue
            counts["strings"] += 1
            counts["tokens"] += len(words)
            equal = _members(exact.get(norm, ())) if long else ()
            for candidate in equal:
                if len(words) > first.get(candidate, (0, 0, -1))[2]:
                    first[candidate] = (row, 0, len(words), len(equal))
            here = _indexed_keys(words, index, params)
            for candidate, needle, length in (
                anchor for key in here for anchor in anchors.get(key, ())
            ):
                if length > first.get(candidate, (0, 0, -1))[2] and needle in norm:
                    shared = len(_members(exact[needle]))
                    first[candidate] = (row, 1, length, shared)
            found |= here
        matched = Counter(c for key in found for c in _members(index[key]))
        for candidate, count in matched.items():
            if count > best.get(candidate, (0,))[0]:
                best[candidate] = (count, row)
    seconds = time.perf_counter() - started
    return number, dict(counts), seconds, best, first


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class _Memory:
    """Best-effort peak of summed PSS and RSS over this process and its children."""

    def __init__(self, interval: float = 1.0) -> None:
        self.interval = interval
        self.peak_pss_kb = self.peak_rss_kb = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def _proc(path: str) -> str:
        try:
            with open(path) as stream:
                return stream.read()
        except OSError:
            return ""

    def sample(self) -> None:
        own = os.getpid()
        pids, threads = [own], f"/proc/{own}/task"
        for tid in os.listdir(threads) if os.path.isdir(threads) else ():
            pids.extend(map(int, self._proc(f"{threads}/{tid}/children").split()))
        fields: Counter = Counter()
        for pid in pids:
            for line in self._proc(f"/proc/{pid}/smaps_rollup").splitlines():
                name, _, value = line.partition(":")
                if name in ("Pss", "Rss"):
                    fields[name] += int(value.split()[0])
        self.peak_pss_kb = max(self.peak_pss_kb, fields["Pss"])
        self.peak_rss_kb = max(self.peak_rss_kb, fields["Rss"])

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self.sample()

    def start(self) -> None:
        self.sample()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self.sample()


def _log(message: str) -> None:
    print(f"[overlap] {message}", file=sys.stderr, flush=True)


def scan(
    protected_path: Path,
    corpora: Sequence[tuple[str, Sequence[str]]],
    *,
    params: Params = Params(),
    workers: int = 1,
    checks: Sequence[tuple[str, str, int]] = (),
    log: Any = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """(count-only receipt, one private hit record per candidate)."""
    started = time.time()
    labels = [label for label, _ in corpora]
    if len(set(labels)) != len(labels):
        raise ValueError("corpus labels must be unique")
    protected = load_protected(protected_path, params)
    if log:
        log(f"{protected.stats['candidates']} candidates indexed")
    memory = _Memory()
    _STATE.update(protected=protected, params=params)
    pool = None
    try:
        mapper: Any = map
        if workers > 1:
            gc.collect()
            gc.freeze()
            pool = multiprocessing.get_context("fork").Pool(workers)
            mapper = pool.imap
        memory.start()
        digests = mapper(_sha256, [path for path, _, _ in checks])
        for (path, sha256, size), digest in zip(checks, digests):
            if digest != sha256 or os.path.getsize(path) != size:
                raise ValueError(f"corpus file changed since the manifest: {path}")
        tasks, files, sizes = plan(corpora, params, mapper)
        order = sorted(range(len(tasks)), key=lambda i: (-tasks[i].cost, i))
        items = [(i, tasks[i]) for i in order]
        results = (
            pool.imap_unordered(_scan_task, items)
            if pool is not None
            else map(_scan_task, items)
        )
        merged = _merge(results, tasks, len(labels), log)
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()
        memory.stop()
        _STATE.clear()
        if workers > 1:
            gc.unfreeze()
    receipt, hits = _report(protected, corpora, tasks, files, sizes, merged, params)
    usage = resource.getrusage
    receipt["resources"] = {
        "workers": workers,
        "tasks": len(tasks),
        "wall_seconds": round(time.time() - started, 1),
        "peak_pss_mb": round(memory.peak_pss_kb / 1024),
        "peak_rss_sum_mb": round(memory.peak_rss_kb / 1024),
        "max_rss_parent_mb": round(usage(resource.RUSAGE_SELF).ru_maxrss / 1024),
        "max_rss_worker_mb": round(usage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024),
        "manifest_files_verified": len(checks),
    }
    return receipt, hits


def _merge(
    results: Iterable[tuple[Any, ...]], tasks: list[_Task], labels: int, log: Any
) -> dict[str, Any]:
    best: list[dict[int, tuple[int, int, int]]] = [{} for _ in range(labels)]
    first: list[dict[int, tuple[int, ...]]] = [{} for _ in range(labels)]
    stats: list[Counter] = [Counter() for _ in range(labels)]
    lines = [0] * len(tasks)
    done, last = 0, time.time()
    for number, counts, seconds, task_best, task_first in results:
        label = tasks[number].label
        lines[number] = counts.pop("lines", 0)
        stats[label].update(counts)
        stats[label]["seconds"] += seconds
        table = best[label]
        for candidate, (count, row) in task_best.items():
            old = table.get(candidate)
            if old is None or (-count, number, row) < (-old[0], old[1], old[2]):
                table[candidate] = (count, number, row)
        table = first[label]
        for candidate, (row, kind, length, shared) in task_first.items():
            old = table.get(candidate)
            if old is None or (-length, number, row) < (-old[3], old[0], old[1]):
                table[candidate] = (number, row, kind, length, shared)
        done += 1
        if log and (time.time() - last > 30 or done == len(tasks)):
            last = time.time()
            log(f"{done}/{len(tasks)} tasks")
    bases = [0] * len(tasks)
    running: dict[int, int] = {}
    for number, task in enumerate(tasks):
        if task.kind == "jsonl":
            bases[number] = running.get(task.file, 0)
            running[task.file] = bases[number] + lines[number]
    return {"best": best, "first": first, "stats": stats, "bases": bases}


def _share(matched: int, size: int) -> float:
    return matched / size if size else 0.0


def _hit(
    candidate: int,
    protected: Protected,
    labels: list[str],
    merged: dict[str, Any],
    where: Any,
    params: Params,
) -> dict[str, Any]:
    """The private record of one candidate: overall and per-label verdicts, no text."""
    size = protected.sizes[candidate]
    per_label, top, exact_at = {}, None, None
    for number, label in enumerate(labels):
        found = merged["best"][number].get(candidate)
        exact = merged["first"][number].get(candidate)
        if found is None and exact is None:
            continue
        count = found[0] if found else 0
        entry: dict[str, Any] = {
            "verdict": verdict(_share(count, size), exact and exact[3], params),
            "containment": round(_share(count, size), 4),
            "matched": count,
            "file": None,
            "row": None,
            "exact": None,
        }
        if found:
            entry["file"], entry["row"] = where(found[1], found[2])
            if top is None or count > top[1]:
                top = (label, count, entry["file"], entry["row"])
        if exact:
            entry["exact"] = EXACT_KINDS[exact[2]]
            entry["exact_file"], entry["exact_row"] = where(exact[0], exact[1])
            entry["exact_tokens"], entry["exact_shared"] = exact[3], exact[4]
            if exact_at is None or exact[3] > exact_at[4]:
                exact_at = (
                    label,
                    entry["exact"],
                    entry["exact_file"],
                    entry["exact_row"],
                    exact[3],
                    exact[4],
                )
        per_label[label] = entry
    count = top[1] if top else 0
    location = top or (exact_at and (exact_at[0], 0, exact_at[2], exact_at[3]))
    return {
        "id": protected.ids[candidate],
        "source": protected.sources[candidate],
        "task": protected.tasks[candidate],
        "verdict": verdict(_share(count, size), exact_at and exact_at[4], params),
        "containment": round(_share(count, size), 4),
        "matched": count,
        "shingles": size,
        "label": location[0] if location else None,
        "file": location[2] if location else None,
        "row": location[3] if location else None,
        "exact": (
            dict(zip(("label", "kind", "file", "row", "tokens", "shared"), exact_at))
            if exact_at
            else None
        ),
        "labels": per_label,
    }


def _report(
    protected: Protected,
    corpora: Sequence[tuple[str, Sequence[str]]],
    tasks: list[_Task],
    files: list[str],
    sizes: list[int],
    merged: dict[str, Any],
    params: Params,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    labels = [label for label, _ in corpora]
    bases = merged["bases"]

    def where(number: int, row: int) -> tuple[str, int]:
        return files[tasks[number].file], bases[number] + row

    hits = [
        _hit(candidate, protected, labels, merged, where, params)
        for candidate in range(len(protected.ids))
    ]
    empty = dict.fromkeys(VERDICTS, 0)
    reasons: Counter = Counter()
    exact_only = dict.fromkeys([name for _, name in TOKEN_BINS] + ["20+"], 0)
    by_label = {label: Counter() for label in labels}
    by_source: dict[str, dict[str, Any]] = {}
    by_task: dict[str, dict[str, int]] = {}
    for hit in hits:
        result, exact = hit["verdict"], hit["exact"]
        share = _share(hit["matched"], hit["shingles"])
        by_exact = exact is not None and exact["tokens"] >= params.exact_min_tokens
        if result == "OVERLAP" and by_exact and share >= params.overlap:
            reasons["both"] += 1
        elif result == "OVERLAP" and by_exact:
            reasons["exact_only"] += 1
            reasons["exact_only_shared_text"] += exact["shared"] > 1
            length = exact["tokens"]
            exact_only[next((n for t, n in TOKEN_BINS if length <= t), "20+")] += 1
        elif result == "OVERLAP":
            reasons["containment_only"] += 1
        elif exact is not None and share < params.review:
            reasons["review_short_exact"] += 1
        source = by_source.setdefault(
            hit["source"], {"candidates": 0, **empty, "labels": {}}
        )
        task = by_task.setdefault(hit["task"], {"candidates": 0, **empty})
        for cell in (source, task):
            cell["candidates"] += 1
            cell[result] += 1
        for label, entry in hit["labels"].items():
            by_label[label]["candidates_hit"] += 1
            by_label[label][entry["verdict"]] += 1
            by_label[label]["exact"] += entry["exact"] is not None
            if entry["verdict"] != "CLEAN":
                cell = source["labels"].setdefault(label, {"OVERLAP": 0, "REVIEW": 0})
                cell[entry["verdict"]] += 1
    summary, first_file = {}, 0
    for number, (label, paths) in enumerate(corpora):
        stats, found = merged["stats"][number], by_label[label]
        summary[label] = {
            "files": len(paths),
            "bytes": sum(sizes[first_file : first_file + len(paths)]),
            "rows": stats["rows"],
            "strings": stats["strings"],
            "tokens": stats["tokens"],
            "unparsed_lines": stats["unparsed"],
            "seconds": round(stats["seconds"], 1),
            "candidates_hit": found["candidates_hit"],
            "verdicts": {v: found[v] for v in VERDICTS[:2]},
            "exact": found["exact"],
        }
        first_file += len(paths)
    receipt = {
        "schema": SCHEMA,
        "params": dataclasses.asdict(params),
        "protected": {"sha256": protected.sha256, **protected.stats},
        "corpora": summary,
        "verdicts": {**empty, **Counter(hit["verdict"] for hit in hits)},
        "overlap_reasons": {
            **{
                key: reasons[key]
                for key in (
                    "exact_only",
                    "containment_only",
                    "both",
                    "exact_only_shared_text",
                    "review_short_exact",
                )
            },
            "exact_only_tokens": exact_only,
        },
        "by_source": dict(sorted(by_source.items())),
        "by_task": dict(sorted(by_task.items())),
    }
    return receipt, hits


def _write_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("scan", help="screen a candidates file")
    command.add_argument("--protected", type=Path, required=True)
    command.add_argument(
        "--corpus", action="append", default=[], metavar="LABEL=PATH_OR_GLOB"
    )
    command.add_argument("--manifest", type=Path, help="c1-corpora manifest.json")
    command.add_argument("--labels", default="", help="comma-separated label globs")
    command.add_argument("--no-verify", action="store_true")
    command.add_argument("--workers", type=int, default=1)
    command.add_argument("--chunk-mb", type=int, default=16)
    command.add_argument("--no-cjk-split", action="store_true")
    command.add_argument(
        "--exact-min-tokens",
        type=int,
        default=0,
        help="exact matches of shorter texts give REVIEW instead of OVERLAP",
    )
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--hits", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.corpus and args.manifest is None:
        parser.error("give --manifest and/or --corpus")
    if args.output.resolve() == args.hits.resolve():
        parser.error("receipt and hits must be different files")
    for path in (args.output, args.hits):
        if path.exists():
            parser.error(f"refusing to overwrite {path}")
    params = Params(
        exact_min_tokens=args.exact_min_tokens,
        cjk_split=not args.no_cjk_split,
        chunk_bytes=args.chunk_mb << 20,
    )
    corpora: list[tuple[str, list[str]]] = []
    checks: list[tuple[str, str, int]] = []
    manifest_sha256 = None
    if args.manifest is not None:
        patterns = [p for p in args.labels.split(",") if p]
        corpora, checks, manifest_sha256 = manifest_corpora(args.manifest, patterns)
    for label, files in parse_corpora(args.corpus):
        if label in dict(corpora):
            parser.error(f"label {label} is already in the manifest")
        corpora.append((label, files))
    receipt, hits = scan(
        args.protected,
        corpora,
        params=params,
        workers=args.workers,
        checks=[] if args.no_verify else checks,
        log=_log,
    )
    data = "".join(
        json.dumps(hit, ensure_ascii=False, sort_keys=True) + "\n" for hit in hits
    ).encode("utf-8")
    receipt["manifest_sha256"] = manifest_sha256
    receipt["hits_sha256"] = hashlib.sha256(data).hexdigest()
    _write_new(args.hits, data)
    text = json.dumps(receipt, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
    _write_new(args.output, text.encode("utf-8"))
    summary = {
        "candidates": receipt["protected"]["candidates"],
        "verdicts": receipt["verdicts"],
        "wall_seconds": receipt["resources"]["wall_seconds"],
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
