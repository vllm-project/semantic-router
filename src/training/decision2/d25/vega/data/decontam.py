"""Decontamination of training rows against public Decision Index suite rows (SPEC "Decontamination").

Normalisation: NFKC + casefold + \\w+ tokens. A training row is dropped when

0. its normalised state + instructions + option contents equal one question of a suite item (``exact_item``,
   catches verbatim copies even when every part alone is generic), or
1. its normalised state equals a suite state (``exact_state``), or
2. its normalised question+options text equals a suite question's (``exact_qo``; instructions plus the
   order-free set of option contents), or its option set alone equals a suite question's option set of
   at least ``MIN_OPTION_TOKENS`` tokens with ``MIN_OPTION_WORDS`` alphabetic tokens (``exact_opts``), or
3. it shares 13-grams with suite items beyond the calibrated rule (``ngram``): ``hits >= min_hits`` or
   the best-matching suite item has at least ``min_coverage`` of its 13-grams in the row.

Any key (state, question+options, option set, 13-gram) that occurs in more than ``BOILERPLATE`` suite
items is boilerplate and ignored. Two refinements keep fixed prompts from acting as item identity: a suite
13-gram with at least ``TEMPLATE_TOKENS`` tokens inside that item's boilerplate spans (a kit instruction
prefix plus the first words of the item) is not indexed, and the question+options check is skipped when
the instructions alone are boilerplate (then only the option-set check applies). The rule is calibrated with planted positive controls (suite content
rendered as training rows: verbatim, embedded in a longer unrelated state, question transplanted onto
an unrelated state, and token-perturbed) and a false-positive rate on a clean (generated) sample.

Index directory layout (all arrays sorted by hash):
    meta.json, rule.json, items.jsonl.gz (suite item id + benchmark),
    state.npz / qo.npz / opts.npz / ngram.npz with ``hash`` (uint64) and ``item`` (int32, representative item),
    item_ngrams.npy (non-boilerplate 13-grams per item).

CLI:
    python -m d25.vega.data.decontam build-index --suite a.jsonl.gz b.jsonl.gz --out DIR
    python -m d25.vega.data.decontam calibrate --index DIR --clean rows.jsonl.gz ... --out DIR/rule.json
    python -m d25.vega.data.decontam check --index DIR --rows rows.jsonl.gz --out flags.jsonl.gz
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np

from d25.vega.common import decision_format as df
from d25.vega.data.util import (
    normalize_tokens,
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)

N = 13
BOILERPLATE = 50
MIN_OPTION_TOKENS = 10
MIN_OPTION_WORDS = 6
TEMPLATE_TOKENS = 7
MIN_QO_TOKENS = 8
PRIME = np.uint64(0x9E3779B97F4A7C15)
CODE_KEY = re.compile(
    r"^([A-Za-z]{1,2}|\d{1,3}|[A-Za-z]\d{1,3}|result_\d+|option_?\d+|opt_?\d+|choice_?\d+)$"
)
_TOKEN_CACHE: dict[str, int] = {}


def token_hash(token: str) -> int:
    value = _TOKEN_CACHE.get(token)
    if value is None:
        value = int.from_bytes(
            hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "little"
        )
        if len(_TOKEN_CACHE) < 4_000_000:
            _TOKEN_CACHE[token] = value
    return value


def text_hash(tokens: Sequence[str]) -> int:
    return int.from_bytes(
        hashlib.blake2b(" ".join(tokens).encode("utf-8"), digest_size=8).digest(),
        "little",
    )


def ngram_hashes(tokens: Sequence[str]) -> np.ndarray:
    count = len(tokens) - N + 1
    if count <= 0:
        return np.zeros(0, dtype=np.uint64)
    values = np.fromiter(
        (token_hash(t) for t in tokens), dtype=np.uint64, count=len(tokens)
    )
    result = np.zeros(count, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for offset in range(N):
            result = result * PRIME + values[offset : offset + count]
    return result


def describe_state(state: Any) -> str:
    if state in (None, ""):
        return ""
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def option_contents(question: dict[str, Any]) -> list[list[str]]:
    """Normalised option contents, independent of code-like keys (A/B/1/result_3) and of key/description split."""
    if question.get("type") != "choice":
        return []
    contents = []
    for key, value in (question.get("criteria") or {}).items():
        desc = (
            ""
            if value is None
            else (
                value
                if isinstance(value, str)
                else json.dumps(value, ensure_ascii=False)
            )
        )
        if desc and CODE_KEY.match(str(key)):
            contents.append(normalize_tokens(desc))
        else:
            contents.append(normalize_tokens(f"{key} {desc}"))
    return contents


def question_keys(
    question: dict[str, Any],
) -> tuple[int | None, int | None, list[str], list[str], list[list[str]], str]:
    """(qo hash, option-set hash or None, instruction tokens, flat option tokens, instruction lines, options key)."""
    text = df.describe(question.get("instructions") or "")
    instructions = normalize_tokens(text)
    lines = [normalize_tokens(line) for line in text.split("\n")]
    contents = option_contents(question)
    flat = [tok for content in contents for tok in content]
    joined = sorted(" ".join(c) for c in contents)
    options_key = " \u241e ".join(joined)
    qo = (
        text_hash(instructions + ["\u241f"] + [options_key])
        if len(instructions) + len(flat) >= MIN_QO_TOKENS
        else None
    )
    # option sets made of numbers (score levels 0..9, numeric answers) are generic, not item identities
    wordy = sum(1 for tok in flat if any(ch.isalpha() for ch in tok))
    opts = (
        text_hash(["\u241e".join(joined)])
        if len(flat) >= MIN_OPTION_TOKENS and wordy >= MIN_OPTION_WORDS
        else None
    )
    return qo, opts, instructions, flat, lines, options_key


def item_key(state_tokens: list[str], key: tuple) -> int:
    """Exact identity of one question of an item: state + instructions + option contents."""
    return text_hash(state_tokens + ["\u241d"] + key[2] + ["\u241f", key[5]])


def segments_for(
    state: Any, questions: Iterable[dict[str, Any]]
) -> tuple[list[str], list[list[str]], list[tuple]]:
    """Token segments for 13-grams: the state, each instruction line (kit prompts put the item after a newline,
    so a fixed prompt never joins the item text in one gram), and each question's option block.
    """
    state_tokens = normalize_tokens(describe_state(state))
    keys = [question_keys(q) for q in questions]
    segments = (
        [state_tokens] + [line for k in keys for line in k[4]] + [k[3] for k in keys]
    )
    return state_tokens, segments, keys


def row_ngrams(segments: Sequence[Sequence[str]]) -> np.ndarray:
    parts = [ngram_hashes(seg) for seg in segments if len(seg) >= N]
    if not parts:
        return np.zeros(0, dtype=np.uint64)
    return np.unique(np.concatenate(parts))


# ----------------------------------------------------------------------------------------------- index


def suite_rows(paths: Sequence[str]):
    for path in paths:
        for row in read_jsonl(path):
            questions = row.get("questions") or {}
            bench = (
                row.get("benchmark")
                or row.get("family")
                or (row.get("_evaluation") or {}).get("dataset")
            )
            if not bench and "proxy" in row:
                bench = f"proxy-{row.get('part') or row.get('proxy')}:{row.get('catalog_id') or row.get('proxy')}"
            yield {
                "id": str(row.get("id")),
                "bench": str(bench or "?"),
                "state": row.get("state"),
                "questions": (
                    list(questions.values())
                    if isinstance(questions, dict)
                    else list(questions)
                ),
            }


def _suite_item(item: dict[str, Any]):
    state_tokens, segments, keys = segments_for(item["state"], item["questions"])
    state_h = text_hash(state_tokens) if state_tokens else None
    qos = sorted({k[0] for k in keys if k[0] is not None})
    opts = sorted({k[1] for k in keys if k[1] is not None})
    instr = sorted({text_hash(k[2]) for k in keys})
    items = sorted({item_key(state_tokens, k) for k in keys})
    return state_h, qos, opts, row_ngrams(segments), instr, items


_BOILER: np.ndarray | None = None


def _init_boiler(values: np.ndarray) -> None:
    global _BOILER
    _BOILER = values


def _kept_ngrams(item: dict[str, Any]) -> np.ndarray:
    """Non-boilerplate 13-grams of a suite item that are not anchored in its templated text.

    A token is templated when a boilerplate 13-gram (in > BOILERPLATE items) covers it; a 13-gram with at
    least TEMPLATE_TOKENS templated tokens mostly repeats a fixed prompt (kit instruction prefixes followed
    by the first words of the item) and is not used for matching.
    """
    assert _BOILER is not None
    _, segments, _ = segments_for(item["state"], item["questions"])
    parts = []
    for seg in segments:
        grams = ngram_hashes(seg)
        if not len(grams):
            continue
        boiler = np.isin(grams, _BOILER)
        covered = np.zeros(len(seg) + 1, dtype=np.int32)
        starts = np.nonzero(boiler)[0]
        np.add.at(covered, starts, 1)
        np.add.at(covered, starts + N, -1)
        templated = (np.cumsum(covered)[: len(seg)] > 0).astype(np.int32)
        window = np.convolve(templated, np.ones(N, dtype=np.int32), mode="valid")
        parts.append(grams[~boiler & (window < TEMPLATE_TOKENS)])
    if not parts:
        return np.zeros(0, dtype=np.uint64)
    return np.unique(np.concatenate(parts))


def _distinct_item_counts(
    hashes: np.ndarray, items: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(hashes) == 0:
        return np.zeros(0, np.uint64), np.zeros(0, np.int64), np.zeros(0, np.int64)
    order = np.lexsort((items, hashes))
    hashes, items = hashes[order], items[order]
    pair_new = np.ones(len(hashes), dtype=bool)
    pair_new[1:] = (hashes[1:] != hashes[:-1]) | (items[1:] != items[:-1])
    return np.unique(hashes[pair_new], return_index=True, return_counts=True)


def _unique_with_items(
    hashes: np.ndarray, items: np.ndarray, limit: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """Count distinct items per hash; keep hashes with count <= limit and a representative item."""
    if len(hashes) == 0:
        return np.zeros(0, np.uint64), np.zeros(0, np.int32), 0
    order = np.lexsort((items, hashes))
    hashes, items = hashes[order], items[order]
    pair_new = np.ones(len(hashes), dtype=bool)
    pair_new[1:] = (hashes[1:] != hashes[:-1]) | (items[1:] != items[:-1])
    hashes, items = hashes[pair_new], items[pair_new]
    uniq, first, counts = np.unique(hashes, return_index=True, return_counts=True)
    keep = counts <= limit
    return uniq[keep], items[first][keep].astype(np.int32), int((~keep).sum())


def build_index(suite_paths: Sequence[str], out: Path, workers: int) -> dict[str, Any]:
    started = time.time()
    items = list(suite_rows(suite_paths))
    print(f"suite items: {len(items)}", flush=True)
    with Pool(workers) as pool:
        results = pool.map(_suite_item, items, chunksize=256)
    (
        state_h,
        state_i,
        qo_h,
        qo_i,
        op_h,
        op_i,
        ng_parts,
        ng_items,
        in_h,
        in_i,
        it_h,
        it_i,
    ) = ([] for _ in range(12))
    for index, (sh, qos, opts, grams, instr, item_keys) in enumerate(results):
        if sh is not None:
            state_h.append(sh)
            state_i.append(index)
        qo_h += qos
        qo_i += [index] * len(qos)
        op_h += opts
        op_i += [index] * len(opts)
        in_h += instr
        in_i += [index] * len(instr)
        it_h += item_keys
        it_i += [index] * len(item_keys)
        ng_parts.append(grams)
        ng_items.append(np.full(len(grams), index, dtype=np.int32))
    out.mkdir(parents=True, exist_ok=True)
    stats: dict[str, Any] = {"items": len(items)}
    for name, hs, its in (
        ("state", state_h, state_i),
        ("qo", qo_h, qo_i),
        ("opts", op_h, op_i),
        ("item", it_h, it_i),
    ):
        h, i, boiler = _unique_with_items(
            np.array(hs, dtype=np.uint64), np.array(its, dtype=np.int32), BOILERPLATE
        )
        np.savez(out / f"{name}.npz", hash=h, item=i)
        stats[name] = {"kept": int(len(h)), "boilerplate": boiler}
    # instructions shared by > BOILERPLATE items are generic: question+options matches then rest on the options
    ih, icount = (
        np.unique(np.array(in_h, dtype=np.uint64), return_counts=True)
        if in_h
        else (np.zeros(0, np.uint64), np.zeros(0))
    )
    np.save(out / "instr_boiler.npy", ih[icount > BOILERPLATE])
    stats["instructions"] = {
        "distinct": int(len(ih)),
        "boilerplate": int((icount > BOILERPLATE).sum()),
    }
    all_grams = np.concatenate(ng_parts) if ng_parts else np.zeros(0, np.uint64)
    all_items = np.concatenate(ng_items) if ng_items else np.zeros(0, np.int32)
    uniq, first, counts = _distinct_item_counts(all_grams, all_items)
    boiler_grams = uniq[counts > BOILERPLATE]
    with Pool(workers, initializer=_init_boiler, initargs=(boiler_grams,)) as pool:
        kept_parts = pool.map(_kept_ngrams, items, chunksize=256)
    ng_parts = kept_parts
    kept_all = np.concatenate(kept_parts) if kept_parts else np.zeros(0, np.uint64)
    kept_items = (
        np.concatenate(
            [np.full(len(g), k, dtype=np.int32) for k, g in enumerate(kept_parts)]
        )
        if kept_parts
        else np.zeros(0, np.int32)
    )
    h, i, _ = _unique_with_items(kept_all, kept_items, BOILERPLATE)
    np.savez(out / "ngram.npz", hash=h, item=i)
    stats["ngram"] = {
        "kept": int(len(h)),
        "boilerplate": int(len(boiler_grams)),
        "total_item_ngrams": int(len(all_grams)),
        "template_anchored_removed": int(int((counts <= BOILERPLATE).sum()) - len(h)),
    }
    # non-boilerplate n-grams per item (for coverage)
    item_counts = np.zeros(len(items), dtype=np.int64)
    for index, grams in enumerate(ng_parts):
        if len(grams):
            pos = np.searchsorted(h, grams)
            pos[pos >= len(h)] = 0
            item_counts[index] = int((h[pos] == grams).sum()) if len(h) else 0
    np.save(out / "item_ngrams.npy", item_counts)
    write_jsonl(
        out / "items.jsonl.gz", ({"id": it["id"], "bench": it["bench"]} for it in items)
    )
    meta = {
        "n": N,
        "boilerplate_items": BOILERPLATE,
        "min_option_tokens": MIN_OPTION_TOKENS,
        "min_option_words": MIN_OPTION_WORDS,
        "template_tokens": TEMPLATE_TOKENS,
        "min_qo_tokens": MIN_QO_TOKENS,
        "normalisation": "NFKC + casefold + \\w+ tokens; blake2b-64 token/text hashes; polynomial 13-gram hash",
        "suite_files": {str(p): sha256_file(p) for p in suite_paths},
        "benchmarks": dict(Counter(it["bench"] for it in items)),
        "stats": stats,
        "seconds": round(time.time() - started, 1),
    }
    write_json(out / "meta.json", meta)
    if not (out / "rule.json").exists():
        write_json(
            out / "rule.json",
            {"min_hits": 1, "min_coverage": None, "calibrated": False},
        )
    return meta


# ----------------------------------------------------------------------------------------------- check


class Index:
    def __init__(self, path: Path):
        self.path = path
        self.meta = json.loads((path / "meta.json").read_text())
        self.rule = json.loads((path / "rule.json").read_text())
        names = ("state", "qo", "opts", "ngram") + (
            ("item",) if (path / "item.npz").exists() else ()
        )
        self.tables = {name: np.load(path / f"{name}.npz") for name in names}
        self.hash = {name: table["hash"] for name, table in self.tables.items()}
        self.item = {name: table["item"] for name, table in self.tables.items()}
        self.item_ngrams = np.load(path / "item_ngrams.npy")
        self.items = [r["bench"] for r in read_jsonl(path / "items.jsonl.gz")]
        boiler = path / "instr_boiler.npy"
        self.instr_boiler = (
            set(int(v) for v in np.load(boiler)) if boiler.exists() else set()
        )

    def lookup(self, name: str, value: int | None) -> int:
        if value is None or name not in self.hash:
            return -1
        hashes = self.hash[name]
        pos = int(np.searchsorted(hashes, np.uint64(value)))
        if pos < len(hashes) and int(hashes[pos]) == value:
            return int(self.item[name][pos])
        return -1

    def ngram_stats(self, grams: np.ndarray) -> tuple[int, int, int]:
        """(hits, best item, hits on best item)."""
        hashes = self.hash["ngram"]
        if len(grams) == 0 or len(hashes) == 0:
            return 0, -1, 0
        pos = np.searchsorted(hashes, grams)
        pos[pos >= len(hashes)] = 0
        found = hashes[pos] == grams
        hits = int(found.sum())
        if not hits:
            return 0, -1, 0
        items = self.item["ngram"][pos[found]]
        values, counts = np.unique(items, return_counts=True)
        best = int(np.argmax(counts))
        return hits, int(values[best]), int(counts[best])


def decide(
    index: Index, row: dict[str, Any], rule: dict[str, Any] | None = None
) -> dict[str, Any]:
    rule = rule or index.rule
    question = row["question"]
    state_tokens, segments, keys = segments_for(row.get("state"), [question])
    reasons = []
    hit_item = -1
    if state_tokens:
        item = index.lookup("state", text_hash(state_tokens))
        if item >= 0:
            reasons.append("exact_state")
            hit_item = item
    qo, opts, instruction_tokens = keys[0][:3]
    item = index.lookup("item", item_key(state_tokens, keys[0]))
    if item >= 0:
        reasons.append("exact_item")
        hit_item = item if hit_item < 0 else hit_item
    item = (
        -1
        if text_hash(instruction_tokens) in index.instr_boiler
        else index.lookup("qo", qo)
    )
    if item >= 0:
        reasons.append("exact_qo")
        hit_item = item if hit_item < 0 else hit_item
    item = index.lookup("opts", opts)
    if item >= 0:
        reasons.append("exact_opts")
        hit_item = item if hit_item < 0 else hit_item
    hits, best, best_hits = index.ngram_stats(row_ngrams(segments))
    coverage = best_hits / max(1, int(index.item_ngrams[best])) if best >= 0 else 0.0
    min_hits = rule.get("min_hits")
    min_cov = rule.get("min_coverage")
    if hits and (
        (min_hits is not None and hits >= min_hits)
        or (min_cov is not None and coverage >= min_cov)
    ):
        reasons.append("ngram")
        hit_item = best if hit_item < 0 else hit_item
    return {
        "id": row["id"],
        "drop": bool(reasons),
        "reasons": reasons,
        "hits": hits,
        "best_hits": best_hits,
        "coverage": round(coverage, 4),
        "bench": (
            index.items[hit_item]
            if hit_item >= 0
            else (index.items[best] if best >= 0 else None)
        ),
    }


_INDEX: Index | None = None


def _init(path: str) -> None:
    global _INDEX
    _INDEX = Index(Path(path))


def _decide_batch(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    assert _INDEX is not None
    return [decide(_INDEX, row) for row in rows]


def batched(rows: Iterable[dict[str, Any]], size: int):
    batch = []
    for row in rows:
        batch.append(row)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def check_rows(
    index_dir: Path, rows: Iterable[dict[str, Any]], workers: int
) -> Iterable[dict[str, Any]]:
    with Pool(workers, initializer=_init, initargs=(str(index_dir),)) as pool:
        for result in pool.imap(_decide_batch, batched(rows, 512), chunksize=1):
            yield from result


def summarize(
    flags: Iterable[dict[str, Any]], source_of: dict[str, str] | None = None
) -> dict[str, Any]:
    per_source: dict[str, Counter] = defaultdict(Counter)
    benches: Counter = Counter()
    for flag in flags:
        source = (source_of or {}).get(flag["id"], "all")
        per_source[source]["rows"] += 1
        if flag["drop"]:
            per_source[source]["dropped"] += 1
            for reason in flag["reasons"]:
                per_source[source][reason] += 1
            benches[flag.get("bench") or "?"] += 1
    return {
        "per_source": {k: dict(v) for k, v in sorted(per_source.items())},
        "by_benchmark": dict(benches.most_common()),
    }


# ----------------------------------------------------------------------------------------------- calibration


def planted_controls(
    index_dir: Path,
    suite_paths: Sequence[str],
    clean: list[dict[str, Any]],
    per_bench: int,
    seed: int,
):
    """Suite content rendered as training rows (never written anywhere; in-memory only)."""
    rng = random.Random(seed)
    by_bench: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in suite_rows(suite_paths):
        if item["questions"]:
            by_bench[item["bench"]].append(item)
    filler_words = [
        w for row in clean[:2000] for w in describe_state(row.get("state")).split()[:50]
    ] or ["filler"]
    controls = []
    for bench, items in sorted(by_bench.items()):
        sample = rng.sample(items, min(per_bench, len(items)))
        for k, item in enumerate(sample):
            question = rng.choice(item["questions"])
            if question.get("type") not in ("choice", "noul"):
                continue
            if question.get("type") == "choice" and not question.get("criteria"):
                continue
            other = clean[rng.randrange(len(clean))]
            text = describe_state(item["state"])
            words = text.split()
            base = {"source": "planted", "family": bench, "target": None, "meta": {}}
            controls.append(
                {
                    **base,
                    "id": f"verbatim:{bench}:{k}",
                    "variant": "verbatim",
                    "state": item["state"],
                    "question": question,
                }
            )
            if len(words) >= 8:
                host = describe_state(other.get("state")).split()
                cut = len(host) // 2
                embedded = " ".join(host[:cut] + words + host[cut:])
                controls.append(
                    {
                        **base,
                        "id": f"embedded:{bench}:{k}",
                        "variant": "embedded",
                        "state": embedded,
                        "question": other["question"],
                    }
                )
                start = rng.randrange(0, max(1, len(words) // 2))
                window = words[start : start + max(8, len(words) // 2)]
                partial = " ".join(host[:cut] + window + host[cut:])
                controls.append(
                    {
                        **base,
                        "id": f"partial:{bench}:{k}",
                        "variant": "partial",
                        "state": partial,
                        "question": other["question"],
                    }
                )
                noisy = [
                    rng.choice(filler_words) if (i % 20 == 19) else w
                    for i, w in enumerate(words)
                ]
                controls.append(
                    {
                        **base,
                        "id": f"perturbed:{bench}:{k}",
                        "variant": "perturbed",
                        "state": " ".join(noisy),
                        "question": question,
                    }
                )
            controls.append(
                {
                    **base,
                    "id": f"transplant:{bench}:{k}",
                    "variant": "transplant",
                    "state": other.get("state"),
                    "question": question,
                }
            )
    return controls


def _raw_stats(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    assert _INDEX is not None
    out = []
    for row in rows:
        result = decide(_INDEX, row, rule={"min_hits": None, "min_coverage": None})
        out.append(
            {
                "id": row["id"],
                "variant": row.get("variant", "clean"),
                "bench": row.get("family"),
                "exact": bool(result["reasons"]),
                "hits": result["hits"],
                "coverage": result["coverage"],
            }
        )
    return out


def calibrate(
    index_dir: Path,
    suite_paths: Sequence[str],
    clean_paths: Sequence[str],
    out: Path,
    workers: int,
    per_bench: int,
    clean_limit: int,
    seed: int,
) -> dict[str, Any]:
    clean: list[dict[str, Any]] = []
    for path in clean_paths:
        for row in read_jsonl(path):
            clean.append(row)
    random.Random(seed).shuffle(clean)
    clean = clean[:clean_limit]
    controls = planted_controls(index_dir, suite_paths, clean, per_bench, seed)
    with Pool(workers, initializer=_init, initargs=(str(index_dir),)) as pool:
        planted = [
            r for batch in pool.imap(_raw_stats, batched(controls, 256)) for r in batch
        ]
        negatives = [
            r for batch in pool.imap(_raw_stats, batched(clean, 256)) for r in batch
        ]
    # A control is catchable by n-grams only if it has any non-boilerplate hit; uncatchable controls
    # (short texts) are reported separately and rely on the exact checks.
    grid_hits = [1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 20, 30, 50]
    grid_cov = [None, 0.05, 0.1, 0.2, 0.3, 0.5]
    results = []
    for min_hits in grid_hits:
        for min_cov in grid_cov:

            def caught(r):
                return r["exact"] or (
                    r["hits"] > 0
                    and (
                        r["hits"] >= min_hits
                        or (min_cov is not None and r["coverage"] >= min_cov)
                    )
                )

            catchable = [r for r in planted if r["exact"] or r["hits"] > 0]
            recall = sum(caught(r) for r in catchable) / max(1, len(catchable))
            fp = sum(caught(r) for r in negatives) / max(1, len(negatives))
            results.append(
                {
                    "min_hits": min_hits,
                    "min_coverage": min_cov,
                    "recall_catchable": recall,
                    "fp_rate": fp,
                }
            )
    # Among rules that catch every catchable planted control, take the lowest clean false-positive rate;
    # ties go to the more conservative rule (smaller min_hits, then smaller coverage).
    perfect = [r for r in results if r["recall_catchable"] >= 1.0]
    best = (
        min(
            perfect,
            key=lambda r: (
                r["fp_rate"],
                r["min_hits"],
                r["min_coverage"] if r["min_coverage"] is not None else 2.0,
            ),
        )
        if perfect
        else results[0]
    )
    by_variant: dict[str, Counter] = defaultdict(Counter)
    for r in planted:
        c = by_variant[r["variant"]]
        c["n"] += 1
        c["exact"] += r["exact"]
        c["ngram_any"] += r["hits"] > 0
        c["uncatchable"] += (not r["exact"]) and r["hits"] == 0
        ok = r["exact"] or (
            r["hits"] > 0
            and (
                r["hits"] >= best["min_hits"]
                or (
                    best["min_coverage"] is not None
                    and r["coverage"] >= best["min_coverage"]
                )
            )
        )
        c["caught"] += ok
    uncatchable = Counter(
        r["bench"] for r in planted if not r["exact"] and r["hits"] == 0
    )
    rule = {
        "min_hits": best["min_hits"],
        "min_coverage": best["min_coverage"],
        "calibrated": True,
        "recall_catchable": best["recall_catchable"],
        "fp_rate_clean": best["fp_rate"],
        "planted": len(planted),
        "clean_rows": len(negatives),
        "per_variant": {k: dict(v) for k, v in by_variant.items()},
        "uncatchable_by_benchmark": dict(uncatchable.most_common()),
        "grid": results,
        "clean_files": list(clean_paths),
        "seed": seed,
        "note": "Uncatchable = planted text with no exact match and no non-boilerplate 13-gram (too short or boilerplate); "
        "only verbatim exact copies of such items can be detected.",
    }
    write_json(out, rule)
    return rule


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build-index")
    b.add_argument("--suite", nargs="+", required=True)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--workers", type=int, default=os.cpu_count() or 8)
    c = sub.add_parser("calibrate")
    c.add_argument("--index", type=Path, required=True)
    c.add_argument("--suite", nargs="+", required=True)
    c.add_argument("--clean", nargs="+", required=True)
    c.add_argument("--out", type=Path)
    c.add_argument("--per-bench", type=int, default=60)
    c.add_argument("--clean-limit", type=int, default=20000)
    c.add_argument("--seed", type=int, default=20261009)
    c.add_argument("--workers", type=int, default=os.cpu_count() or 8)
    k = sub.add_parser("check")
    k.add_argument("--index", type=Path, required=True)
    k.add_argument("--rows", nargs="+", required=True)
    k.add_argument("--out", type=Path, required=True)
    k.add_argument("--workers", type=int, default=os.cpu_count() or 8)
    args = parser.parse_args(argv)
    if args.command == "build-index":
        print(
            json.dumps(
                build_index(args.suite, args.out, args.workers)["stats"], indent=1
            )
        )
    elif args.command == "calibrate":
        rule = calibrate(
            args.index,
            args.suite,
            args.clean,
            args.out or args.index / "rule.json",
            args.workers,
            args.per_bench,
            args.clean_limit,
            args.seed,
        )
        print(json.dumps({k: v for k, v in rule.items() if k != "grid"}, indent=1))
    else:
        rows = (row for path in args.rows for row in read_jsonl(path))
        flags = list(check_rows(args.index, rows, args.workers))
        write_jsonl(args.out, flags)
        print(json.dumps(summarize(flags), indent=1))


if __name__ == "__main__":
    sys.exit(main())
