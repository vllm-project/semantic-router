"""Deterministic CPU overlap screen for Decision 2.0 v2 candidate data.

`scan` compares candidate training rows with a pinned, input-only protected
inventory; `self_scan` compares candidate groups with each other or with a
second candidate set. Text leaves are NFKC/casefold/whitespace normalized and
need >= 20 compact characters.

E  exact: equal normalized leaves, and equal normalized canonical states.
S  the audit_vitaminc_score short-leaf rule against protected leaves of at
   most 800 normalized characters, reproduced exactly.
L  the same rule against 400/200 windows (last one end-aligned) of longer
   protected leaves. Candidate units are 200/100 windows (the whole leaf when
   shorter), so every copied candidate window lies inside one protected
   window; 400/200 windows on both sides reach 0.90 containment only when the
   two window grids happen to align within ~40 characters.
N  word 13-grams and compact-character 30-grams whose blake2b-64 value is
   0 mod 4, a deterministic 1/4 sample applied identically to both sides.

A matched unit is boilerplate (reported separately, never quarantining) when
it occurs in >= 5 distinct candidate groups; an N gram is also boilerplate in
>= 5 distinct protected rows. E/S/L units have no protected-row condition:
one protected passage is often shared by several protected rows. Receipts
never contain text; only the private receipt names groups and row IDs.
"""

from __future__ import annotations

import argparse
import bisect
import collections
import dataclasses
import functools
import hashlib
import heapq
import json
import multiprocessing
import os
import sys
from array import array
from collections.abc import Callable, Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

from training.model.data import canonical
from v2.data.textnorm import (
    compact,
    decode_canonical_json,
    is_cjk_heavy,
    normalize,
    text_leaves,
    word_tokens,
)

PRIVATE_SCHEMA = "decision2.v2.overlap.private.v1"
PUBLIC_SCHEMA = "decision2.v2.overlap.public.v1"
SELF_PRIVATE_SCHEMA = "decision2.v2.overlap.self.private.v1"
SELF_PUBLIC_SCHEMA = "decision2.v2.overlap.self.public.v1"
ANSWER_KEYS = frozenset(
    {
        "gold",
        "label",
        "labels",
        "answer",
        "answers",
        "target",
        "targets",
        "teacher_probs",
        "target_probs",
        "correct",
        "solution",
    }
)
PROTECTED_FIELDS = ("state", "instructions", "questions", "options")
CANDIDATE_FIELDS = ("state", "instructions", "options")
METHODS = ("E", "S", "L", "N")
KINDS = ("E_leaf", "E_state", "S", "L", "N_word", "N_char")
KIND_METHOD = {
    "E_leaf": "E",
    "E_state": "E",
    "S": "S",
    "L": "L",
    "N_word": "N",
    "N_char": "N",
}
_GRAM_PERSON = b"v2-overlap-gram"
_WORD_PERSON = b"v2-overlap-word"
_CHAR_PERSON = b"v2-overlap-char"
_MASK64 = (1 << 64) - 1
_MIX64 = 0x9E3779B97F4A7C15
_BUCKET_BITS = 6
_DIRECTORY_BITS = 16
_SLICE_CACHE_MAX = 1024
_WORK: dict[str, Any] = {}


@dataclasses.dataclass(frozen=True)
class Params:
    min_compact: int = 20
    short_max_chars: int = 800
    gram: int = 6
    index_every: int = 2
    rare_grams: int = 6
    max_posting: int = 128
    jaccard: float = 0.72
    containment: float = 0.90
    window: int = 400
    window_stride: int = 200
    candidate_window: int = 200
    candidate_window_stride: int = 100
    word_ngram: int = 13
    char_ngram: int = 30
    char_sample_mod: int = 4
    boilerplate_rows: int = 5
    boilerplate_groups: int = 5
    ngram_min_hits: int = 1

    def __post_init__(self) -> None:
        if any(
            type(value) is int and value < 1
            for value in dataclasses.asdict(self).values()
        ):
            raise ValueError("Overlap parameters must be positive")
        if self.candidate_window > self.window - self.window_stride:
            raise ValueError("candidate_window must be <= window - window_stride")


@dataclasses.dataclass(frozen=True)
class ProtectedLeaves:
    """Input-only reference text; `refs[i]` is (role, row id or group id)."""

    refs: tuple[tuple[str, str], ...]
    leaves: tuple[tuple[int, str], ...]
    states: tuple[tuple[int, Any], ...]
    files: tuple[dict[str, Any], ...] = ()
    inventory_sha256: str | None = None


def method_descriptions(params: Params) -> dict[str, str]:
    return {
        "E": (
            f"normalized leaf equality (compact length >= {params.min_compact}) and "
            "equality of normalized canonical-JSON states (canonical-JSON strings "
            "decoded first)"
        ),
        "S": (
            "audit_vitaminc_score rule against protected leaves <= "
            f"{params.short_max_chars} normalized chars: compact {params.gram}-gram "
            f"sets, every {params.index_every}nd sorted protected gram indexed, the "  # codespell:ignore nd
            f"{params.rare_grams} rarest candidate grams by (posting length, gram), "
            f"postings <= {params.max_posting}, Jaccard >= {params.jaccard} or "
            f"containment >= {params.containment}"
        ),
        "L": (
            f"S rule against {params.window}/{params.window_stride} windows (last "
            "window end-aligned) of the compact text of protected leaves > "
            f"{params.short_max_chars} chars; candidate units are "
            f"{params.candidate_window}/{params.candidate_window_stride} windows of "
            f"leaves over {params.candidate_window} compact chars, else the leaf"
        ),
        "N": (
            f"word {params.word_ngram}-grams over word_tokens and compact "
            f"{params.char_ngram}-grams kept iff "
            "int.from_bytes(blake2b(utf8, digest_size=8, person), 'big') % "
            f"{params.char_sample_mod} == 0; only 8-byte hashes are stored"
        ),
        "boilerplate": (
            f"a matched unit in >= {params.boilerplate_groups} distinct candidate "
            f"groups, or an N gram in >= {params.boilerplate_rows} distinct protected "
            "rows; reported separately and never quarantining"
        ),
    }


def _answer_key(key: Any) -> bool:
    return isinstance(key, str) and key.strip().casefold() in ANSWER_KEYS


def answer_keys(row: Mapping[str, Any]) -> list[str]:
    """Answer-bearing keys at top level and inside native question objects."""
    found = {key for key in row if _answer_key(key)}
    questions = row.get("questions")
    if isinstance(questions, dict):
        for question in questions.values():
            if isinstance(question, dict):
                found.update(key for key in question if _answer_key(key))
    return sorted(found)


def protected_from_rows(
    named_rows: Mapping[str, Iterable[Mapping[str, Any]]],
    *,
    ref_key: str = "id",
    fields: tuple[str, ...] = PROTECTED_FIELDS,
    input_only: bool = True,
) -> ProtectedLeaves:
    collected: list[tuple[str, str, Mapping[str, Any]]] = []
    for role in sorted(named_rows):
        for row in named_rows[role]:
            if not isinstance(row, Mapping):
                raise ValueError(f"{role}: every row must be a JSON object")
            ref = row.get(ref_key)
            if not isinstance(ref, str) or not ref:
                raise ValueError(f"{role}: every row needs a nonempty string {ref_key}")
            if input_only and (keys := answer_keys(row)):
                raise ValueError(
                    f"{role}: row {ref} carries answer-bearing keys {keys}"
                )
            collected.append((role, ref, row))
    refs = sorted({(role, ref) for role, ref, _ in collected})
    if ref_key == "id" and len(refs) != len(collected):
        raise ValueError("Duplicate row id within a protected role")
    position = {ref: index for index, ref in enumerate(refs)}
    leaves: list[tuple[int, str]] = []
    states: list[tuple[int, Any]] = []
    for role, ref, row in collected:
        index = position[(role, ref)]
        for field in fields:
            if field in row:
                leaves.extend(
                    (index, leaf) for leaf in text_leaves(row[field], decode_json=True)
                )
        if "state" in row:
            states.append((index, row["state"]))
    return ProtectedLeaves(tuple(refs), tuple(leaves), tuple(states))


def _parse_lines(data: bytes, label: str) -> Iterator[dict[str, Any]]:
    lines = data.split(b"\n")
    if lines and not lines[-1]:
        lines.pop()
    for number, raw in enumerate(lines, 1):
        if not raw.strip():
            raise ValueError(f"{label}:{number}: blank line")
        row = json.loads(raw)
        if not isinstance(row, dict):
            raise ValueError(f"{label}:{number}: every line must be a JSON object")
        yield row


def load_protected_inventory(path: str | Path) -> ProtectedLeaves:
    """Verify every pinned file's SHA-256, then collect input-only leaves.

    Relative paths resolve against the inventory's directory.
    """
    inventory = Path(path)
    raw = inventory.read_bytes()
    entries = json.loads(raw)
    if not isinstance(entries, list) or not entries:
        raise ValueError("Protected inventory must be a nonempty JSON list")
    named: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    files: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for entry in entries:
        if not isinstance(entry, dict) or not all(
            isinstance(entry.get(field), str) and entry[field]
            for field in ("role", "path", "sha256")
        ):
            raise ValueError("Every inventory entry needs string role, path and sha256")
        role = entry["role"]
        location = Path(entry["path"])
        if not location.is_absolute():
            location = inventory.parent / location
        if (role, str(location)) in seen:
            raise ValueError(f"Duplicate inventory entry for role {role}")
        seen.add((role, str(location)))
        if not location.is_file():
            raise ValueError(f"Protected file for role {role} is missing")
        data = location.read_bytes()
        if hashlib.sha256(data).hexdigest() != entry["sha256"].lower():
            raise ValueError(f"Protected file SHA-256 mismatch for role {role}")
        rows = list(_parse_lines(data, role))
        named[role].extend(rows)
        files.append(
            {
                "role": role,
                "path": str(location),
                "sha256": entry["sha256"].lower(),
                "rows": len(rows),
            }
        )
    protected = protected_from_rows(named)
    return dataclasses.replace(
        protected,
        files=tuple(sorted(files, key=lambda item: (item["role"], item["path"]))),
        inventory_sha256=hashlib.sha256(raw).hexdigest(),
    )


def window_starts(length: int, size: int, stride: int) -> list[int]:
    if length <= size:
        return [0]
    starts = list(range(0, length - size + 1, stride))
    if starts[-1] != length - size:
        starts.append(length - size)
    return starts


@functools.cache
def _slices(length: int, size: int) -> tuple[slice, ...]:
    return tuple(map(slice, range(length - size + 1), range(size, length + 1)))


def _gram_stream(text: str, size: int) -> Iterator[str]:
    if len(text) <= _SLICE_CACHE_MAX:
        return map(text.__getitem__, _slices(len(text), size))
    return (text[i : i + size] for i in range(len(text) - size + 1))


def grams(text: str, params: Params) -> set[str]:
    """audit_vitaminc_score `sixgrams` of an already compact string."""
    if len(text) < params.min_compact:
        return set()
    return set(_gram_stream(text, params.gram))


def gram_key(gram: str) -> int:
    """Exact (bijective) for grams of <= 8 UTF-8 bytes, else blake2b-64."""
    raw = gram.encode("utf-8")
    if len(raw) <= 8:
        return (int.from_bytes(raw, "big") * _MIX64) & _MASK64
    digest = hashlib.blake2b(raw, digest_size=8, person=_GRAM_PERSON).digest()
    return int.from_bytes(digest, "big")


def word_ngram_keys(normalized: str, params: Params) -> set[int]:
    tokens = word_tokens(normalized)
    size = params.word_ngram
    blake = hashlib.blake2b
    return {
        int.from_bytes(
            blake(
                " ".join(tokens[i : i + size]).encode("utf-8"),
                digest_size=8,
                person=_WORD_PERSON,
            ).digest(),
            "big",
        )
        for i in range(len(tokens) - size + 1)
    }


def char_ngram_keys(text: str, params: Params) -> set[int]:
    size, modulus = params.char_ngram, params.char_sample_mod
    blake = hashlib.blake2b
    keys = set()
    for i in range(len(text) - size + 1):
        key = int.from_bytes(
            blake(
                text[i : i + size].encode("utf-8"), digest_size=8, person=_CHAR_PERSON
            ).digest(),
            "big",
        )
        if key % modulus == 0:
            keys.add(key)
    return keys


def _digest(text: str) -> bytes:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()


def state_key(state: Any, params: Params) -> bytes | None:
    value = state
    if isinstance(state, str):
        decoded = decode_canonical_json(state)
        if decoded is not None:
            value = decoded
    text = normalize(canonical(value))
    return _digest(text) if len(compact(text)) >= params.min_compact else None


class _Pairs:
    """(key, value) pairs bucketed by the key's top bits, in flat arrays."""

    def __init__(self) -> None:
        self.keys = [array("Q") for _ in range(1 << _BUCKET_BITS)]
        self.values = [array("I") for _ in range(1 << _BUCKET_BITS)]

    def add(self, key: int, value: int) -> None:
        bucket = key >> (64 - _BUCKET_BITS)
        self.keys[bucket].append(key)
        self.values[bucket].append(value)


class Postings:
    """Sorted key -> distinct values in flat arrays (fork-shared, no copies).

    A key with more than `cap` distinct values keeps an empty span.
    """

    def __init__(self, parts: list[_Pairs], cap: int) -> None:
        keys, offsets, values = array("Q"), array("I", [0]), array("I")
        self.pairs = 0
        for bucket in range(1 << _BUCKET_BITS):
            packed: list[int] = []
            for part in parts:
                packed.extend(
                    (key << 32) | value
                    for key, value in zip(part.keys[bucket], part.values[bucket])
                )
                part.keys[bucket], part.values[bucket] = array("Q"), array("I")
            packed.sort()
            current, run = -1, array("I")
            for item in packed:
                key, value = item >> 32, item & 0xFFFFFFFF
                if key != current:
                    if current >= 0:
                        self._emit(keys, offsets, values, current, run, cap)
                    current, run = key, array("I", [value])
                elif value != run[-1]:
                    run.append(value)
            if current >= 0:
                self._emit(keys, offsets, values, current, run, cap)
        self.keys, self.offsets, self.values = keys, offsets, values
        self.shift = 64 - _DIRECTORY_BITS
        self.directory = array(
            "I",
            (
                bisect.bisect_left(keys, bucket << self.shift)
                for bucket in range(1 << _DIRECTORY_BITS)
            ),
        )
        self.directory.append(len(keys))

    def _emit(
        self,
        keys: array,
        offsets: array,
        values: array,
        key: int,
        run: array,
        cap: int,
    ) -> None:
        keys.append(key)
        self.pairs += len(run)
        if len(run) <= cap:
            values.extend(run)
        offsets.append(len(values))

    def __len__(self) -> int:
        return len(self.keys)

    def span(self, key: int) -> tuple[int, int] | None:
        bucket = key >> self.shift
        low, high = self.directory[bucket], self.directory[bucket + 1]
        index = bisect.bisect_left(self.keys, key, low, high)
        if index < high and self.keys[index] == key:
            return self.offsets[index], self.offsets[index + 1]
        return None


@dataclasses.dataclass
class _Reference:
    refs: tuple[tuple[str, str], ...]
    unit_refs: list[tuple[int, ...]]
    blob: str
    offsets: array
    exact: dict[bytes, int]
    states: dict[bytes, tuple[int, ...]]
    unit_grams: array
    window_unit: array
    window_start: array
    window_grams: array
    window: int
    short: Postings
    windows: Postings
    words: Postings
    chars: Postings
    stats: dict[str, Any]

    def unit_text(self, unit: int) -> tuple[str, int]:
        """Compact text of a short unit and its distinct gram count."""
        return (
            self.blob[self.offsets[unit] : self.offsets[unit + 1]],
            self.unit_grams[unit],
        )

    def window_text(self, window: int) -> tuple[str, int]:
        unit = self.window_unit[window]
        start = self.offsets[unit] + self.window_start[window]
        end = min(start + self.window, self.offsets[unit + 1])
        return self.blob[start:end], self.window_grams[window]


def _run(function: Callable[[Any], Any], tasks: list[Any], workers: int) -> list[Any]:
    if workers <= 1 or len(tasks) <= 1:
        return [function(task) for task in tasks]
    context = multiprocessing.get_context("fork")
    with context.Pool(min(workers, len(tasks))) as pool:
        return pool.map(function, tasks, chunksize=1)


def _chunks(costs: list[int], workers: int) -> list[tuple[int, int]]:
    if not costs:
        return []
    parts = 1 if workers <= 1 else 4 * workers
    target = max(1, -(-sum(costs) // parts))
    bounds, start, running = [], 0, 0
    for index, cost in enumerate(costs):
        running += cost
        if running >= target:
            bounds.append((start, index + 1))
            start, running = index + 1, 0
    if start < len(costs):
        bounds.append((start, len(costs)))
    return bounds


def _index_chunk(bounds: tuple[int, int]) -> tuple[Any, ...]:
    """Postings pairs plus distinct gram counts of the chunk's units/windows."""
    work = _WORK
    params: Params = work["params"]
    texts, comps, unit_refs = work["texts"], work["comps"], work["unit_refs"]
    long_units, window_base = work["long"], work["window_base"]
    short, windows, words, chars = _Pairs(), _Pairs(), _Pairs(), _Pairs()
    unit_grams, window_grams = array("H"), array("H")
    every = params.index_every
    for unit in range(*bounds):
        text = comps[unit]
        if long_units[unit]:
            unit_grams.append(0)
            starts = window_starts(len(text), params.window, params.window_stride)
            for offset, start in enumerate(starts):
                piece = grams(text[start : start + params.window], params)
                window_grams.append(len(piece))
                for gram in sorted(piece)[::every]:
                    windows.add(gram_key(gram), window_base[unit] + offset)
        else:
            piece = grams(text, params)
            unit_grams.append(len(piece))
            for gram in sorted(piece)[::every]:
                short.add(gram_key(gram), unit)
        refs = unit_refs[unit][: params.boilerplate_rows]
        for key in word_ngram_keys(texts[unit], params):
            for ref in refs:
                words.add(key, ref)
        for key in char_ngram_keys(text, params):
            for ref in refs:
                chars.add(key, ref)
    return short, windows, words, chars, unit_grams, window_grams


def _build_reference(
    protected: ProtectedLeaves, params: Params, workers: int
) -> _Reference:
    table: dict[str, tuple[set[int], collections.Counter[str]]] = {}
    for ref, leaf in protected.leaves:
        normalized = normalize(leaf)
        if not normalized:
            continue
        refs, roles = table.setdefault(normalized, (set(), collections.Counter()))
        refs.add(ref)
        roles[protected.refs[ref][0]] += 1
    by_role: dict[str, collections.Counter[str]] = collections.defaultdict(
        collections.Counter
    )
    texts, comps = [], []
    for normalized in sorted(table):
        text = compact(normalized)
        long = len(normalized) > params.short_max_chars
        bucket = (
            "below_min_compact"
            if len(text) < params.min_compact
            else ("gt_800" if long else "le_800")
        )
        for role, count in table[normalized][1].items():
            by_role[role][bucket] += count
        if len(text) >= params.min_compact:
            texts.append(normalized)
            comps.append(text)
    unit_refs = [tuple(sorted(table[text][0])) for text in texts]
    del table
    long_units = bytearray(len(text) > params.short_max_chars for text in texts)
    offsets = array("Q", [0])
    for text in comps:
        offsets.append(offsets[-1] + len(text))
    window_unit, window_start, window_base = array("I"), array("I"), array("I")
    for unit, text in enumerate(comps):
        window_base.append(len(window_unit))
        if long_units[unit]:
            for start in window_starts(len(text), params.window, params.window_stride):
                window_unit.append(unit)
                window_start.append(start)
    states: dict[bytes, set[int]] = collections.defaultdict(set)
    for ref, state in protected.states:
        key = state_key(state, params)
        if key is not None:
            states[key].add(ref)
    _WORK.clear()
    _WORK.update(
        params=params,
        texts=texts,
        comps=comps,
        unit_refs=unit_refs,
        long=long_units,
        window_base=window_base,
    )
    try:
        costs = [
            len(text) * (2 if flag else 1) for text, flag in zip(comps, long_units)
        ]
        parts = _run(_index_chunk, _chunks(costs, workers), workers)
    finally:
        _WORK.clear()
    unit_grams, window_grams = array("H"), array("H")
    for part in parts:
        unit_grams.extend(part[4])
        window_grams.extend(part[5])
    short = Postings([part[0] for part in parts], params.max_posting)
    windows = Postings([part[1] for part in parts], params.max_posting)
    words = Postings([part[2] for part in parts], params.boilerplate_rows - 1)
    chars = Postings([part[3] for part in parts], params.boilerplate_rows - 1)
    del parts
    windowed = set(window_unit)
    roles = collections.Counter(role for role, _ in protected.refs)
    stats = {
        "refs_by_role": dict(sorted(roles.items())),
        "leaves_by_role": {
            role: {
                bucket: counts[bucket]
                for bucket in ("le_800", "gt_800", "below_min_compact")
            }
            for role, counts in sorted(by_role.items())
        },
        "distinct_leaves": {
            "le_800": len(texts) - sum(long_units),
            "gt_800": sum(long_units),
        },
        "long_leaf_windows": len(window_unit),
        "long_leaves_not_windowed": sum(
            1 for unit, flag in enumerate(long_units) if flag and unit not in windowed
        ),
        "cjk_heavy_distinct_leaves": sum(1 for text in comps if is_cjk_heavy(text)),
        "states_indexed": len(states),
        "index_keys": {
            "S": len(short),
            "L": len(windows),
            "N_word": len(words),
            "N_char": len(chars),
        },
    }
    return _Reference(
        refs=protected.refs,
        unit_refs=unit_refs,
        blob="".join(comps),
        offsets=offsets,
        exact={_digest(text): unit for unit, text in enumerate(texts)},
        states={key: tuple(sorted(refs)) for key, refs in states.items()},
        unit_grams=unit_grams,
        window_unit=window_unit,
        window_start=window_start,
        window_grams=window_grams,
        window=params.window,
        short=short,
        windows=windows,
        words=words,
        chars=chars,
        stats=stats,
    )


@dataclasses.dataclass
class _Candidates:
    row_ids: list[str]
    row_group: array
    groups: list[str]
    group_rows: array
    leaves: list[str]
    comps: list[str]
    leaf_rows: list[array]
    state_rows: dict[bytes, list[int]]


def _candidates(rows: Iterable[Mapping[str, Any]], params: Params) -> _Candidates:
    row_ids: list[str] = []
    row_groups: list[str] = []
    seen: set[str] = set()
    leaf_map: dict[str, list[int]] = {}
    state_rows: dict[bytes, list[int]] = collections.defaultdict(list)
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("Every candidate row must be a JSON object")
        row_id, group = row.get("id"), row.get("group_id")
        if not (
            isinstance(row_id, str) and row_id and isinstance(group, str) and group
        ):
            raise ValueError(
                "Every candidate row needs nonempty string id and group_id"
            )
        if row_id in seen:
            raise ValueError(f"Duplicate candidate id {row_id}")
        missing = [field for field in CANDIDATE_FIELDS if field not in row]
        if missing:
            raise ValueError(f"{row_id}: missing {missing}")
        seen.add(row_id)
        index = len(row_ids)
        row_ids.append(row_id)
        row_groups.append(group)
        for field in CANDIDATE_FIELDS:
            for leaf in text_leaves(row[field], decode_json=True):
                normalized = normalize(leaf)
                if normalized:
                    leaf_map.setdefault(normalized, []).append(index)
        key = state_key(row["state"], params)
        if key is not None:
            state_rows[key].append(index)
    groups = sorted(set(row_groups))
    position = {group: index for index, group in enumerate(groups)}
    row_group = array("I", (position[group] for group in row_groups))
    group_rows = array("I", bytes(4 * len(groups)))
    for group in row_group:
        group_rows[group] += 1
    leaves, comps, leaf_rows = [], [], []
    for normalized in sorted(leaf_map):
        text = compact(normalized)
        if len(text) >= params.min_compact:
            leaves.append(normalized)
            comps.append(text)
            leaf_rows.append(array("I", sorted(set(leaf_map[normalized]))))
    return _Candidates(
        row_ids,
        row_group,
        groups,
        group_rows,
        leaves,
        comps,
        leaf_rows,
        dict(state_rows),
    )


def _near(
    units: set[str],
    lookup: Callable[[str], tuple[int, int] | None],
    values: array,
    reference_text: Callable[[int], tuple[str, int]],
    params: Params,
) -> list[int]:
    """The audit_vitaminc_score candidate selection and Jaccard/containment test.

    Postings over the cap have empty spans and sort after every capped posting,
    which selects the same capped postings as sorting by true length.
    """
    present = []
    over_cap = params.max_posting + 1
    for gram in units:
        span = lookup(gram)
        if span is not None:
            low, high = span
            present.append((high - low if high > low else over_cap, gram, low, high))
    candidates: set[int] = set()
    for length, _, low, high in heapq.nsmallest(params.rare_grams, present):
        if length <= params.max_posting:
            candidates.update(values[low:high])
    size = len(units)
    hits = []
    for candidate in sorted(candidates):
        text, other = reference_text(candidate)
        common = len(units.intersection(_gram_stream(text, params.gram)))
        if (
            common / (size + other - common) >= params.jaccard
            or common / min(size, other) >= params.containment
        ):
            hits.append(candidate)
    return hits


def _ngram_hits(
    keys: set[int], postings: Postings
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    hits = []
    for key in sorted(keys):
        span = postings.span(key)
        if span is not None:
            hits.append((key, tuple(postings.values[span[0] : span[1]])))
    return tuple(hits)


def _scan_leaf(
    normalized: str, text: str, reference: _Reference, params: Params
) -> tuple[Any, ...] | None:
    units = grams(text, params)
    keys = {gram: gram_key(gram) for gram in units}
    short: list[int] = []
    if len(reference.short):
        short = _near(
            units,
            lambda gram: reference.short.span(keys[gram]),
            reference.short.values,
            reference.unit_text,
            params,
        )
    long: set[int] = set()
    if len(reference.windows):
        memo: dict[str, tuple[int, int] | None] = {}

        def lookup(gram: str) -> tuple[int, int] | None:
            if gram not in memo:
                memo[gram] = reference.windows.span(keys[gram])
            return memo[gram]

        size, stride = params.candidate_window, params.candidate_window_stride
        pieces = (
            [units]
            if len(text) <= size
            else (
                grams(text[start : start + size], params)
                for start in window_starts(len(text), size, stride)
            )
        )
        for piece in pieces:
            for window in _near(
                piece, lookup, reference.windows.values, reference.window_text, params
            ):
                long.add(reference.window_unit[window])
    words = _ngram_hits(word_ngram_keys(normalized, params), reference.words)
    chars = _ngram_hits(char_ngram_keys(text, params), reference.chars)
    if short or long or words or chars:
        return tuple(short), tuple(sorted(long)), words, chars
    return None


def _foreign(refs: tuple[int, ...], own: frozenset[int]) -> bool:
    """False only when every reference is the leaf's single own group.

    Empty refs mark an over-cap (boilerplate) gram, which is always kept.
    """
    return not refs or len(own) != 1 or not own.issuperset(refs)


def _own_refs(rows: Iterable[int], row_group: array, own_map: array) -> frozenset[int]:
    return frozenset(own_map[row_group[row]] for row in rows)


def _scan_chunk(bounds: tuple[int, int]) -> list[tuple[Any, ...]]:
    work = _WORK
    reference: _Reference = work["reference"]
    own_map = work["own_map"]
    results = []
    for leaf in range(*bounds):
        hit = _scan_leaf(
            work["leaves"][leaf], work["comps"][leaf], reference, work["params"]
        )
        if hit is None:
            continue
        if own_map is not None:
            own = _own_refs(work["leaf_rows"][leaf], work["row_group"], own_map)
            short, long, words, chars = hit
            hit = (
                tuple(u for u in short if _foreign(reference.unit_refs[u], own)),
                tuple(u for u in long if _foreign(reference.unit_refs[u], own)),
                tuple(item for item in words if _foreign(item[1], own)),
                tuple(item for item in chars if _foreign(item[1], own)),
            )
            if not any(hit):
                continue
        results.append((leaf, *hit))
    return results


@dataclasses.dataclass
class _Hits:
    """Per candidate group: kind -> reference -> matched unit ids."""

    kinds: dict[str, dict[int, set[Any]]] = dataclasses.field(
        default_factory=lambda: collections.defaultdict(dict)
    )
    boilerplate: dict[str, set[Any]] = dataclasses.field(
        default_factory=lambda: collections.defaultdict(set)
    )
    rows: set[int] = dataclasses.field(default_factory=set)


@dataclasses.dataclass
class _Match:
    groups: dict[int, _Hits]
    unit_groups: dict[tuple[str, Any], set[int]]
    unit_refs: dict[tuple[str, Any], tuple[int, ...]]
    boilerplate: set[tuple[str, Any]]


def _match(
    candidates: _Candidates,
    reference: _Reference,
    params: Params,
    workers: int,
    own_map: array | None = None,
) -> _Match:
    """Match every candidate leaf and state; `own_map` (candidate group index ->
    reference index) drops matches that only point back to the leaf's groups."""

    def foreign(refs: tuple[int, ...], rows: Iterable[int]) -> bool:
        return own_map is None or _foreign(
            refs, _own_refs(rows, candidates.row_group, own_map)
        )

    exact = {}
    for leaf, text in enumerate(candidates.leaves):
        unit = reference.exact.get(_digest(text))
        if unit is not None:
            exact[leaf] = unit
    _WORK.clear()
    _WORK.update(
        reference=reference,
        params=params,
        leaves=candidates.leaves,
        comps=candidates.comps,
        leaf_rows=candidates.leaf_rows,
        row_group=candidates.row_group,
        own_map=own_map,
    )
    try:
        chunks = _chunks([len(text) for text in candidates.comps], workers)
        scanned = [hit for part in _run(_scan_chunk, chunks, workers) for hit in part]
    finally:
        _WORK.clear()
    matches: dict[int, list[tuple[str, Any, tuple[int, ...]]]] = (
        collections.defaultdict(list)
    )
    for leaf, unit in exact.items():
        if foreign(reference.unit_refs[unit], candidates.leaf_rows[leaf]):
            matches[leaf].append(("E_leaf", unit, reference.unit_refs[unit]))
    for leaf, short, long, words, chars in scanned:
        own = exact.get(leaf)
        for kind, units in (("S", short), ("L", long)):
            matches[leaf].extend(
                (kind, unit, reference.unit_refs[unit]) for unit in units if unit != own
            )
        for kind, found in (("N_word", words), ("N_char", chars)):
            matches[leaf].extend((kind, key, refs) for key, refs in found)
    located: list[tuple[array, list[tuple[str, Any, tuple[int, ...]]]]] = [
        (candidates.leaf_rows[leaf], found)
        for leaf, found in sorted(matches.items())
        if found
    ]
    for key, rows in sorted(candidates.state_rows.items()):
        refs = reference.states.get(key)
        if refs and foreign(refs, rows):
            located.append((array("I", rows), [("E_state", key, refs)]))
    unit_groups: dict[tuple[str, Any], set[int]] = collections.defaultdict(set)
    unit_refs: dict[tuple[str, Any], tuple[int, ...]] = {}
    for rows, found in located:
        groups = {candidates.row_group[row] for row in rows}
        for kind, unit, refs in found:
            unit_groups[(kind, unit)].update(groups)
            unit_refs[(kind, unit)] = refs
    boilerplate = {
        unit
        for unit, groups in unit_groups.items()
        if len(groups) >= params.boilerplate_groups
        or (KIND_METHOD[unit[0]] == "N" and not unit_refs[unit])
    }
    hits: dict[int, _Hits] = collections.defaultdict(_Hits)
    for rows, found in located:
        by_group: dict[int, list[int]] = collections.defaultdict(list)
        for row in rows:
            by_group[candidates.row_group[row]].append(row)
        for kind, unit, refs in found:
            flagged = (kind, unit) not in boilerplate
            for group, members in by_group.items():
                entry = hits[group]
                if not flagged:
                    entry.boilerplate[kind].add(unit)
                    continue
                for ref in refs:
                    entry.kinds[kind].setdefault(ref, set()).add(unit)
                entry.rows.update(members)
    return _Match(dict(hits), unit_groups, unit_refs, boilerplate)


def _methods(counts: Mapping[str, int], params: Params) -> list[str]:
    methods = []
    for method in METHODS:
        count = sum(n for kind, n in counts.items() if KIND_METHOD[kind] == method)
        if count and (method != "N" or count >= params.ngram_min_hits):
            methods.append(method)
    return methods


def _ids_by_role(refs: Iterable[int], reference: _Reference) -> dict[str, list[str]]:
    result: dict[str, list[str]] = collections.defaultdict(list)
    for ref in sorted(set(refs)):
        role, ref_id = reference.refs[ref]
        result[role].append(ref_id)
    return {role: sorted(ids) for role, ids in sorted(result.items())}


def _group_record(
    entry: _Hits, rows: int, reference: _Reference, params: Params
) -> dict[str, Any] | None:
    units = {
        kind: set().union(*by_ref.values())
        for kind, by_ref in entry.kinds.items()
        if by_ref
    }
    counts = {kind: len(units.get(kind, ())) for kind in KINDS}
    methods = _methods(counts, params)
    if not methods:
        return None
    by_method = {}
    for method in methods:
        refs = [
            ref
            for kind, by_ref in entry.kinds.items()
            if KIND_METHOD[kind] == method
            for ref in by_ref
        ]
        by_method[method] = _ids_by_role(refs, reference)
    flagged_refs = [
        ref
        for kind, by_ref in entry.kinds.items()
        if KIND_METHOD[kind] in methods
        for ref in by_ref
    ]
    protected_ids = _ids_by_role(flagged_refs, reference)
    return {
        "rows": rows,
        "hit_rows": len(entry.rows),
        "methods": methods,
        "roles": sorted(protected_ids),
        "protected_ids": protected_ids,
        "by_method": by_method,
        "counts": counts,
        "boilerplate": {kind: len(entry.boilerplate.get(kind, ())) for kind in KINDS},
    }


def _boilerplate_summary(match: _Match) -> dict[str, dict[str, int]]:
    summary = {}
    for kind in KINDS:
        units = [unit for unit in match.boilerplate if unit[0] == kind]
        groups = (
            set().union(*(match.unit_groups[unit] for unit in units))
            if units
            else set()
        )
        summary[kind] = {"units": len(units), "groups": len(groups)}
    return summary


def scan(
    candidate_rows: Iterable[Mapping[str, Any]],
    protected_leaves: ProtectedLeaves,
    *,
    params: Params = Params(),
    workers: int = 1,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (private_receipt, public_receipt) for candidates vs protected."""
    reference = _build_reference(protected_leaves, params, workers)
    candidates = _candidates(candidate_rows, params)
    match = _match(candidates, reference, params, workers)
    records = {}
    for group, entry in sorted(match.groups.items()):
        record = _group_record(entry, candidates.group_rows[group], reference, params)
        if record is not None:
            records[candidates.groups[group]] = record
    boilerplate_units = []
    for unit in sorted(match.boilerplate, key=lambda item: (item[0], str(item[1]))):
        if KIND_METHOD[unit[0]] == "N":
            continue
        refs = match.unit_refs[unit]
        boilerplate_units.append(
            {
                "kind": unit[0],
                "candidate_groups": len(match.unit_groups[unit]),
                "protected_rows": len(refs),
                "roles": sorted({reference.refs[ref][0] for ref in refs}),
                "example_protected_ids": [
                    list(reference.refs[ref]) for ref in refs[:20]
                ],
            }
        )
    boilerplate = _boilerplate_summary(match)
    private = {
        "schema": PRIVATE_SCHEMA,
        "params": dataclasses.asdict(params),
        "protected": {
            "inventory_sha256": protected_leaves.inventory_sha256,
            "files": list(protected_leaves.files),
        },
        "groups": records,
        "boilerplate": boilerplate,
        "boilerplate_units": boilerplate_units,
    }
    public = _public_receipt(candidates, reference, records, boilerplate, params)
    public["protected_inventory_sha256"] = protected_leaves.inventory_sha256
    public["protected_files"] = [
        {"role": item["role"], "sha256": item["sha256"], "rows": item["rows"]}
        for item in protected_leaves.files
    ]
    return private, public


def _public_receipt(
    candidates: _Candidates,
    reference: _Reference,
    records: Mapping[str, Mapping[str, Any]],
    boilerplate: Mapping[str, Any],
    params: Params,
) -> dict[str, Any]:
    empty = lambda: {"groups": 0, "rows": 0}  # noqa: E731
    by_role: dict[str, dict[str, dict[str, int]]] = {
        role: {method: empty() for method in (*METHODS, "any")}
        for role in reference.stats["refs_by_role"]
    }
    by_method = {method: empty() for method in METHODS}
    quarantine = empty()
    for record in records.values():
        quarantine["groups"] += 1
        quarantine["rows"] += record["rows"]
        for method in record["methods"]:
            by_method[method]["groups"] += 1
            by_method[method]["rows"] += record["rows"]
            for role in record["by_method"][method]:
                by_role[role][method]["groups"] += 1
                by_role[role][method]["rows"] += record["rows"]
        for role in record["roles"]:
            by_role[role]["any"]["groups"] += 1
            by_role[role]["any"]["rows"] += record["rows"]
    stats = reference.stats
    leaves_by_length = {
        bucket: sum(counts[bucket] for counts in stats["leaves_by_role"].values())
        for bucket in ("le_800", "gt_800", "below_min_compact")
    }
    return {
        "schema": PUBLIC_SCHEMA,
        "candidates": {
            "rows": len(candidates.row_ids),
            "groups": len(candidates.groups),
            "distinct_leaves_scanned": len(candidates.leaves),
            "states_keyed": sum(len(rows) for rows in candidates.state_rows.values()),
        },
        "protected": {
            "rows_by_role": stats["refs_by_role"],
            "leaves_by_role": stats["leaves_by_role"],
            "leaves_by_length": leaves_by_length,
            "distinct_leaves_by_length": stats["distinct_leaves"],
            "long_leaf_windows_scanned": stats["long_leaf_windows"],
            "cjk_heavy_distinct_leaves": stats["cjk_heavy_distinct_leaves"],
            "states_indexed": stats["states_indexed"],
            "index_keys": stats["index_keys"],
        },
        "flagged": {
            "by_role": by_role,
            "by_method": by_method,
            "quarantine": quarantine,
            "boilerplate": dict(boilerplate),
        },
        "params": dataclasses.asdict(params),
        "methods": method_descriptions(params),
        "protected_leaves_over_800_chars_not_near_scanned": stats[
            "long_leaves_not_windowed"
        ],
    }


def quarantine_groups(private_receipt: Mapping[str, Any]) -> list[str]:
    """Union of groups with any non-boilerplate E, S, L or N match."""
    return sorted(
        group
        for group, record in private_receipt["groups"].items()
        if set(record["methods"]) & set(METHODS)
    )


def self_scan(
    rows: Iterable[Mapping[str, Any]],
    other_rows: Iterable[Mapping[str, Any]] | None = None,
    *,
    params: Params = Params(),
    workers: int = 1,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Pairs of different groups within `rows`, or between `rows` and `other_rows`.

    References are candidate groups, so the protected-row boilerplate condition
    counts distinct reference groups here.
    """
    rows = list(rows)
    within = other_rows is None
    set_name = "a" if within else "b"
    reference = _build_reference(
        protected_from_rows(
            {set_name: rows if within else list(other_rows)},
            ref_key="group_id",
            fields=CANDIDATE_FIELDS,
            input_only=False,
        ),
        params,
        workers,
    )
    candidates = _candidates(rows, params)
    own_map = None
    if within:
        position = {ref: index for index, ref in enumerate(reference.refs)}
        own_map = array("I", (position[(set_name, g)] for g in candidates.groups))
    match = _match(candidates, reference, params, workers, own_map)
    directed: dict[tuple[str, str], list[dict[str, set[Any]]]] = (
        collections.defaultdict(list)
    )
    for group, entry in match.groups.items():
        own = candidates.groups[group]
        by_other: dict[str, dict[str, set[Any]]] = collections.defaultdict(
            lambda: collections.defaultdict(set)
        )
        for kind, by_ref in entry.kinds.items():
            for ref, units in by_ref.items():
                other = reference.refs[ref][1]
                if not (within and other == own):
                    by_other[other][kind].update(units)
        for other, kinds in by_other.items():
            pair = (min(own, other), max(own, other)) if within else (own, other)
            directed[pair].append(kinds)
    pairs = []
    for (first, second), directions in sorted(directed.items()):
        counts = {
            kind: max(len(direction.get(kind, ())) for direction in directions)
            for kind in KINDS
        }
        methods = _methods(counts, params)
        if methods:
            pairs.append(
                {"a": first, "b": second, "methods": methods, "counts": counts}
            )
    boilerplate = _boilerplate_summary(match)
    private = {
        "schema": SELF_PRIVATE_SCHEMA,
        "mode": "within" if within else "across",
        "params": dataclasses.asdict(params),
        "pairs": pairs,
        "boilerplate": boilerplate,
    }
    by_method = {}
    for method in METHODS:
        chosen = [pair for pair in pairs if method in pair["methods"]]
        by_method[method] = {
            "pairs": len(chosen),
            "a_groups": len({pair["a"] for pair in chosen}),
            "b_groups": len({pair["b"] for pair in chosen}),
        }
    involved = {pair["a"] for pair in pairs} | (
        {pair["b"] for pair in pairs} if within else set()
    )
    public = {
        "schema": SELF_PUBLIC_SCHEMA,
        "mode": private["mode"],
        "candidates": {
            "rows": len(candidates.row_ids),
            "groups": len(candidates.groups),
        },
        "reference": {
            "groups": len(reference.refs),
            "distinct_leaves_by_length": reference.stats["distinct_leaves"],
            "long_leaf_windows_scanned": reference.stats["long_leaf_windows"],
        },
        "pairs": len(pairs),
        "by_method": by_method,
        "candidate_groups_in_pairs": len(involved),
        "boilerplate": boilerplate,
        "params": dataclasses.asdict(params),
        "methods": method_descriptions(params),
    }
    return private, public


def _read_candidate_files(
    paths: list[Path], sink: list[dict[str, Any]]
) -> Iterator[dict[str, Any]]:
    for path in paths:
        checksum = hashlib.sha256()
        count = 0
        with path.open("rb") as stream:
            for number, raw in enumerate(stream, 1):
                checksum.update(raw)
                if not raw.strip():
                    raise ValueError(f"{path}:{number}: blank line")
                row = json.loads(raw)
                if not isinstance(row, dict):
                    raise ValueError(
                        f"{path}:{number}: every line must be a JSON object"
                    )
                count += 1
                yield row
        sink.append({"path": str(path), "sha256": checksum.hexdigest(), "rows": count})


def _write_new(path: Path, payload: Mapping[str, Any]) -> None:
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=1) + "\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(data)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--candidates", action="append", required=True, type=Path)
    parser.add_argument("--protected-inventory", type=Path)
    parser.add_argument(
        "--self-scan",
        action="store_true",
        help="compare groups within one --candidates file, or the first file "
        "against the second",
    )
    parser.add_argument("--private-receipt", required=True, type=Path)
    parser.add_argument("--public-receipt", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    if args.self_scan == (args.protected_inventory is not None):
        parser.error("give exactly one of --protected-inventory or --self-scan")
    if args.self_scan and len(args.candidates) > 2:
        parser.error("--self-scan takes one or two --candidates files")
    if args.private_receipt.resolve() == args.public_receipt.resolve():
        parser.error("private and public receipts must be different files")
    for path in (args.private_receipt, args.public_receipt):
        if path.exists():
            parser.error(f"refusing to overwrite {path}")
    files: list[dict[str, Any]] = []
    if args.self_scan:
        first = list(_read_candidate_files(args.candidates[:1], files))
        second = (
            list(_read_candidate_files(args.candidates[1:], files))
            if len(args.candidates) == 2
            else None
        )
        private, public = self_scan(first, second, workers=args.workers)
    else:
        protected = load_protected_inventory(args.protected_inventory)
        private, public = scan(
            _read_candidate_files(args.candidates, files),
            protected,
            workers=args.workers,
        )
    private["candidate_files"] = files
    public["candidate_files"] = [
        {"sha256": item["sha256"], "rows": item["rows"]} for item in files
    ]
    _write_new(args.private_receipt, private)
    _write_new(args.public_receipt, public)
    summary = (
        public["flagged"]["quarantine"] if not args.self_scan else public["by_method"]
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
