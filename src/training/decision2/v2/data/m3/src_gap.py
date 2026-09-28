"""Milestone 3b gap sources: long-evidence arm H7 (HoVer, Natural Questions) and
deferred-source arm H8 (TyDi QA page windows, MIRACL, JCommonsenseQA, SentiMix
Spanglish), built with the data-arms v2 framework.

    python3 -m v2.data.m3.src_gap extract-nq --shards DIR --out nq.jsonl \\
        --report nq.report.json [--workers 32]
    python3 -m v2.data.m3.src_gap build --arm h7 --sources ROOT --tokenizer SNAPSHOT \\
        --existing existing.json [--v1-rows v1-rows.json] --out-dir OUT [--workers 16]
    python3 -m v2.data.m3.src_gap protected --base pi-v3/manifest.json \\
        [--base-sha256 HEX] --add ROLE=PATH [--expect ROLE=HEX] [--add ...] \\
        [--report-only ROLE ...] --out-dir PI
    python3 -m v2.data.m3.src_gap quarantining --manifest PI/manifest.json \\
        --receipt PI/receipt.json --out PI/manifest.quarantining.json
    python3 -m v2.data.m3.src_gap budget --rows A.jsonl --tokens A.tokens.jsonl \\
        --out B.jsonl --report B.budget.json
    python3 -m v2.data.m3.src_gap length-baseline --cells DIR --out lengths.json
    python3 -m v2.data.m3.src_gap pair-check --pairs A7k.aho.jsonl [--pairs ...] \\
        --against H6.aho.jsonl [--against ...] --out pairs.json

Rules: records/m3b-gap-sources-2026-09-28.md and m3b-prereg-amendment-2-2026-09-28.md
(PI-v4). Rows are ``m2.common`` rows; group
namespaces are shared where items are: HoVer claims join the v2 ``multihop``
groups through their HotpotQA parent question, TyDi QA and MIRACL use
``tydi-miracl``. Every family returns rows before its whole-group cap; the build
first drops each group whose id, or any of whose input hashes, occurs in an
existing arm (``--existing``, a JSON list of row files), then caps whole groups
in seed-hash order and slices TRAIN / AHO / SHO with ``m2.common.slice_of``.

Long states are sized in native tokens of the state text (``--tokenizer``, the
pinned Qwen3.5 snapshot directory): each item draws a target from
``LONG_TARGETS`` by SHA-256 and states never exceed ``STATE_CAP`` tokens, so
rows stay under the 8,192-token budget that ``budget`` enforces afterwards.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import functools
import gzip
import hashlib
import html
import json
import math
import multiprocessing
import os
import random
import re
import sqlite3
import sys
import unicodedata
import zipfile
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from training.model.data import canonical, file_sha256
from v2.data.m2 import common, constructions, src_qa
from v2.data.m2.common import (
    choice_options,
    make_row,
    noul_options,
    read_jsonl,
    score_options,
    sha,
)
from v2.data.m2.text import usable_answer
from v2.data.sources import ordinal
from v2.data.textnorm import normalize

Rows = list[dict[str, Any]]
Report = dict[str, Any]
Built = tuple[Rows, Report]
Dirs = Mapping[str, Path]
Paragraph = tuple[str, str]

SCHEMA = "decision2-m3b-gap-build/v1"
RULES = "m3b-gap-sources-2026-09-28.md; m3b-prereg-amendment-2-2026-09-28.md"
ARMS = ("h7", "h8")
LONG_TARGETS = (2300, 3200, 4400, 6000)
STATE_CAP = 7400
MIN_WINDOW_TOKENS = 2200
MAX_ANSWER_WORDS = 8
PARAGRAPH_OVERHEAD = 6
V1_ROWS_KEY = "v1_rows"
TOKENIZER_KEY = "tokenizer"

SOURCE_DIRS = {
    "hover": "m3b/hover",
    "hotpotqa": "hotpotqa@1908d6afbbead072334abe2965f91bd2709910ab",
    "hotpotqa_validation": "m3b/hotpotqa-validation",
    "nq": "m3b/nq-extract-64",
    "tydiqa": "tydiqa@da78f23f9119363459acbaf46bf89426ff26c259",
    "miracl": "miracl@5be20db9509754dadad47689368639fcec739c00",
    "miracl_corpus": "m3b/miracl-corpus",
    "jglue": "jglue@6f071c09316baae89c3d083a90985b4b1cb9968c",
    "sentimix": "m3b/sentimix",
}

HOVER = "hover_train_v1.1"
NQ = "natural_questions_train"
TYDI = src_qa.TYDI_SOURCE
MIRACL = "miracl_v1.0_train"
JCQA = "jglue_jcommonsenseqa_v1.3_train"
SENTIMIX = "sentimix_spanglish_train"

HOVER_TWINS = "hover_answerable"
HOVER_COVERAGE = "hover_coverage"
NQ_WINDOW = "nq_window_removal"
HOVER_TWIN_SHARE = 50
HOTPOTQA_TRAIN_FILES = "distractor/train-*.jsonl"

VERIFY_INSTRUCTIONS = (
    "Do these paragraphs contain enough information to verify this claim? Claim: {}"
)
CHECK_INSTRUCTIONS = (
    "How much of the information needed to check this claim do these paragraphs "
    "contain? Claim: {}"
)
WINDOW_INSTRUCTIONS = (
    "Does this page excerpt state the answer to this question? Question: {}"
)
POOL_INSTRUCTIONS = "Does any of these passages answer the question? Question: {}"
CHOICE_INSTRUCTIONS = "Which option best answers the question in the state?"
SENTIMENT_INSTRUCTIONS = "Rate the overall sentiment that the message expresses."
SENTIMENT_LEVELS = (
    "Negative: the message expresses a negative feeling or opinion.",
    "Neutral: the message expresses neither a positive nor a negative feeling or opinion.",
    "Positive: the message expresses a positive feeling or opinion.",
)
SENTIMENT_LABELS = {"negative": 0, "neutral": 1, "positive": 2}


def _clean(text: str) -> str:
    return " ".join(text.split())


def _sorted(counter: Mapping[str, int]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter) if counter[key]}


def _inputs(root: Path, paths: Iterable[Path]) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in sorted(paths, key=lambda item: item.as_posix())
    }


def _pick(seed: str, local_id: str, choices: Sequence[Any]) -> Any:
    return choices[int(sha(f"{seed}:{local_id}"), 16) % len(choices)]


# Token budgets


class TokenCounter:
    """Native tokens of a text under the pinned tokenizer snapshot directory,
    loaded like ``v2.data.freeze.load_tokenizer`` (the ``row_tokens`` path: the
    raw ``tokenizer.json`` splits Thai, Telugu and Bengali marks differently),
    memoized; without a path, ``len(text) // 4`` (tests only)."""

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self.memo: dict[str, int] = {}
        self._encode: Callable[[str], int]
        if path is None:
            self._encode = lambda text: max(1, len(text) // 4)
        else:
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            from v2.data.freeze import load_tokenizer

            tokenizer = load_tokenizer(
                {"name": "native", "path": str(path), "revision": None}
            )
            self._encode = lambda text: len(
                tokenizer.encode(text, add_special_tokens=False)
            )

    def __call__(self, text: str) -> int:
        value = self.memo.get(text)
        if value is None:
            if len(self.memo) > 400_000:
                self.memo.clear()
            value = self.memo[text] = self._encode(text)
        return value


@functools.cache
def _counter(path: str | None) -> TokenCounter:
    return TokenCounter(Path(path) if path else None)


def counter_for(dirs: Dirs) -> TokenCounter:
    path = dirs.get(TOKENIZER_KEY)
    return _counter(str(path) if path else None)


def item_tokens(count: TokenCounter, paragraph: Paragraph) -> int:
    return count(f"{paragraph[0]}: {paragraph[1]}") + PARAGRAPH_OVERHEAD


# Constructions


def fill(
    ordered: Sequence[Paragraph],
    start: int,
    target: int,
    count: TokenCounter,
    cap: int = STATE_CAP,
) -> tuple[list[Paragraph], int]:
    """Paragraphs of ``ordered`` in order, skipping any that would pass ``cap``,
    until the running total (from ``start``) reaches ``target``."""
    chosen, total = [], start
    for paragraph in ordered:
        if total >= target:
            break
        size = item_tokens(count, paragraph)
        if total + size > cap:
            continue
        chosen.append(paragraph)
        total += size
    return chosen, total


def claim_twins(
    record: Mapping[str, Any],
    *,
    source: str,
    family: str,
    namespace: str,
    template: str,
    seed: str,
    count: TokenCounter,
) -> tuple[Rows, str | None]:
    """C3 at page scale: all supporting paragraphs vs one removed. Distractors
    come from the claim's retrieved pool in seeded order until the state reaches
    its token target; the complete twin swaps one of them for an unused one, so
    both twins hold the same number of paragraphs and each differs from the
    shared base by one edit."""
    gold = list(record["gold"])
    if len(gold) < 2:
        return [], "too_few_supporting"
    rng = constructions.rng_for(seed, record["local_id"])
    target = _pick(f"{seed}:target", record["local_id"], LONG_TARGETS)
    pool = rng.sample(list(record["distractors"]), len(record["distractors"]))
    gold_tokens = sum(item_tokens(count, paragraph) for paragraph in gold)
    base, total = fill(pool[:-1], gold_tokens, target, count)
    if len(base) < 2:
        return [], "too_few_distractors"
    removed = rng.randrange(len(gold))
    last, lost = item_tokens(count, base[-1]), item_tokens(count, gold[removed])
    extra = next(
        (
            paragraph
            for paragraph in pool
            if paragraph not in base
            and total - min(last, lost) + item_tokens(count, paragraph) <= STATE_CAP
        ),
        None,
    )
    if extra is None:
        return [], "too_few_distractors"
    complete = gold + base[:-1] + [extra]
    incomplete = [p for i, p in enumerate(gold) if i != removed] + base + [extra]
    rng.shuffle(complete)
    rng.shuffle(incomplete)
    rows = []
    for label, paragraphs, twin in (
        (1, complete, "complete"),
        (0, incomplete, "removed"),
    ):
        rows.append(
            make_row(
                source=source,
                family=family,
                task_type="noul",
                language="en",
                namespace=namespace,
                group_key=record["group_key"],
                local_id=f"{record['local_id']}:{twin}",
                state=constructions.render_paragraphs(paragraphs),
                instructions=VERIFY_INSTRUCTIONS.format(_clean(record["claim"])),
                options=noul_options("en"),
                label=label,
                template=template,
                audit={
                    "twin": twin,
                    "supporting": len(gold),
                    "paragraphs": len(paragraphs),
                    "target_tokens": target,
                    "hops": record.get("hops"),
                },
            )
        )
    return rows, None


def claim_coverage_text(present: int, needed: int) -> str:
    if present == 0:
        return f"The paragraphs state none of the {needed} facts needed to check the claim."
    if present == needed:
        return f"The paragraphs state all {needed} facts needed to check the claim."
    return f"The paragraphs state {present} of the {needed} facts needed to check the claim."


def claim_coverage(
    record: Mapping[str, Any],
    *,
    source: str,
    family: str,
    namespace: str,
    template: str,
    seed: str,
    count: TokenCounter,
) -> tuple[Rows, str | None]:
    """C4 at page scale: one row per grade g = 0..n with exactly g supporting
    paragraphs, padded with retrieved distractors to one paragraph count fixed
    per group (the count that brings the all-supporting row to its target)."""
    gold, pool = list(record["gold"]), list(record["distractors"])
    needed = len(gold)
    if not 2 <= needed <= 9:
        return [], "unsupported_hops"
    rng = constructions.rng_for(seed, record["local_id"])
    target = _pick(f"{seed}:target", record["local_id"], LONG_TARGETS)
    order = rng.sample(pool, len(pool))
    gold_tokens = sum(item_tokens(count, paragraph) for paragraph in gold)
    padding, _ = fill(order[: len(order) - needed], gold_tokens, target, count)
    total = needed + len(padding)
    if len(padding) < 2 or len(pool) < total:
        return [], "too_few_distractors"
    options = score_options([claim_coverage_text(g, needed) for g in range(needed + 1)])
    instructions = CHECK_INSTRUCTIONS.format(_clean(record["claim"]))
    rows = []
    for present in range(needed + 1):
        keep = sorted(rng.sample(range(needed), present))
        paragraphs = [gold[i] for i in keep] + rng.sample(pool, total - present)
        rng.shuffle(paragraphs)
        if sum(item_tokens(count, p) for p in paragraphs) > STATE_CAP:
            return [], "over_state_cap"
        rows.append(
            make_row(
                source=source,
                family=family,
                task_type="score",
                language="en",
                namespace=namespace,
                group_key=record["group_key"],
                local_id=f"{record['local_id']}:g{present}",
                state=constructions.render_paragraphs(paragraphs),
                instructions=instructions,
                options=options,
                label=present,
                template=template,
                audit={
                    "present": present,
                    "needed": needed,
                    "paragraphs": total,
                    "target_tokens": target,
                    "verdict": record.get("verdict"),
                },
            )
        )
    return rows, None


def grow_window(
    size: Callable[[int], int], count: int, gold: int, target: int, rng: random.Random
) -> tuple[list[int], int] | None:
    """Consecutive paragraph indices (of ``count``) around ``gold`` until their
    tokens (``size``) reach ``target`` or the document ends. Each step extends
    the left side with a probability drawn once per item (the right side when the
    left is closed, and vice versa), so the gold paragraph lands anywhere in the
    window; a side closes at the first paragraph that would pass ``STATE_CAP``."""
    if size(gold) > STATE_CAP:
        return None
    low, high, total = gold, gold + 1, size(gold)
    lead = rng.random()
    left_open, right_open = low > 0, high < count
    while total < target and (left_open or right_open):
        left = left_open and (not right_open or rng.random() < lead)
        index = low - 1 if left else high
        if total + size(index) > STATE_CAP:
            if left:
                left_open = False
            else:
                right_open = False
            continue
        total += size(index)
        if left:
            low -= 1
            left_open = low > 0
        else:
            high += 1
            right_open = high < count
    return list(range(low, high)), total


def window_twins(
    record: Mapping[str, Any],
    *,
    source: str,
    family: str,
    namespace: str,
    template: str,
    seed: str,
    count: TokenCounter,
) -> tuple[Rows, str | None]:
    """Page-window answer removal (Noul twins). A window of consecutive
    paragraphs holds the annotated answer paragraph; the complete twin drops the
    other paragraph (never the page's first) whose token length is closest to
    it, the removed twin drops the answer paragraph, so both twins keep the same
    paragraphs in page order but one. Dropped when the answer paragraph is the
    page's first paragraph (whether an excerpt starts with the page lead would
    otherwise carry the label), an answer is long or names the page subject, no
    answer string is located in the answer paragraph, or the title or the
    removed twin still states an answer."""
    language = record["language"]
    paragraphs, gold = list(record["paragraphs"]), record["gold"]
    if gold == 0:
        return [], "answer_in_lead_paragraph"
    answers = [a for a in dict.fromkeys(record["answers"]) if a.strip()]
    if not answers or not all(usable_answer(answer) for answer in answers):
        return [], "unusable_answer"
    if any(len(answer.split()) > MAX_ANSWER_WORDS for answer in answers):
        return [], "answer_too_long"
    if not any(src_qa.states(paragraphs[gold], a, language) for a in answers):
        return [], "answer_not_in_gold_paragraph"
    title = _clean(record.get("title") or "")
    if any(src_qa.states(title, answer, language) for answer in answers):
        return [], "answer_in_title"
    if title and any(src_qa.states(answer, title, language) for answer in answers):
        return [], "answer_names_page_subject"
    rng = constructions.rng_for(seed, record["local_id"])
    target = _pick(f"{seed}:target", record["local_id"], LONG_TARGETS)
    sizes: dict[int, int] = {}

    def size(index: int) -> int:
        if index not in sizes:
            sizes[index] = count(paragraphs[index]) + 2
        return sizes[index]

    grown = grow_window(size, len(paragraphs), gold, target, rng)
    if grown is None:
        return [], "gold_paragraph_over_cap"
    indices, total = grown
    if len(indices) < 4:
        return [], "window_under_four_paragraphs"
    if total < MIN_WINDOW_TOKENS:
        return [], "window_under_min_tokens"
    others = [index for index in indices if index not in (gold, 0)]
    rng.shuffle(others)
    partner = min(others, key=lambda index: abs(size(index) - size(gold)))
    complete = [paragraphs[i] for i in indices if i != partner]
    removed = [paragraphs[i] for i in indices if i != gold]
    removed_text = "\n\n".join(removed)
    if any(src_qa.states(removed_text, answer, language) for answer in answers):
        return [], "answer_left_after_removal"
    rows = []
    for label, kept, twin in ((1, complete, "complete"), (0, removed, "removed")):
        body = "\n\n".join(kept)
        rows.append(
            make_row(
                source=source,
                family=family,
                task_type="noul",
                language=language,
                namespace=namespace,
                group_key=record["group_key"],
                local_id=f"{record['local_id']}:{twin}",
                state=f"{title}\n\n{body}" if title else body,
                instructions=WINDOW_INSTRUCTIONS.format(_clean(record["question"])),
                options=noul_options("en"),
                label=label,
                template=template,
                audit={
                    "twin": twin,
                    "paragraphs": len(kept),
                    "gold_position": indices.index(gold),
                    "window_tokens": total,
                    "target_tokens": target,
                },
            )
        )
    return rows, None


def relevance_twins(
    record: Mapping[str, Any],
    *,
    source: str,
    family: str,
    namespace: str,
    template: str,
    seed: str,
) -> tuple[Rows, str | None]:
    """C2 Noul twins: one judged-relevant and one judged-non-relevant passage."""
    if not record["positives"] or not record["negatives"]:
        return [], "missing_judgment"
    rng = constructions.rng_for(seed, record["local_id"])
    gold, other = rng.choice(record["positives"]), rng.choice(record["negatives"])
    rows = []
    for label, (title, text), twin in (
        (1, gold, "relevant"),
        (0, other, "non_relevant"),
    ):
        rows.append(
            make_row(
                source=source,
                family=family,
                task_type="noul",
                language=record["language"],
                namespace=namespace,
                group_key=record["group_key"],
                local_id=f"{record['local_id']}:{twin}",
                state=f"{_clean(title)}\n\n{_clean(text)}",
                instructions=constructions.RELEVANCE_NOUL_INSTRUCTIONS.format(
                    _clean(record["question"])
                ),
                options=noul_options("en"),
                label=label,
                template=template,
                audit={"twin": twin},
            )
        )
    return rows, None


def pool_twins(
    record: Mapping[str, Any],
    *,
    source: str,
    family: str,
    namespace: str,
    template: str,
    seed: str,
    max_negatives: int = 9,
) -> tuple[Rows, str | None]:
    """Judged-pool twins: one relevant passage among k judged-non-relevant ones
    vs the same pool with the relevant passage swapped for an unused
    non-relevant one (C3 edit pattern, equal passage counts)."""
    negatives = list(record["negatives"])
    if not record["positives"] or len(negatives) < 3:
        return [], "too_few_judgments"
    rng = constructions.rng_for(seed, record["local_id"])
    gold = rng.choice(record["positives"])
    picked = rng.sample(negatives, min(len(negatives), max_negatives + 1))
    base, extra = picked[:-1], picked[-1]
    complete = [gold] + base[:-1] + [extra]
    removed = base + [extra]
    rng.shuffle(complete)
    rng.shuffle(removed)
    rows = []
    for label, passages, twin in ((1, complete, "complete"), (0, removed, "removed")):
        rows.append(
            make_row(
                source=source,
                family=family,
                task_type="noul",
                language=record["language"],
                namespace=namespace,
                group_key=record["group_key"],
                local_id=f"{record['local_id']}:{twin}",
                state=constructions.render_paragraphs(passages),
                instructions=POOL_INSTRUCTIONS.format(_clean(record["question"])),
                options=noul_options("en"),
                label=label,
                template=template,
                audit={"twin": twin, "passages": len(passages)},
            )
        )
    return rows, None


# HoVer


def hover_family(item: Mapping[str, Any]) -> str:
    if item["label"] == "SUPPORTED" and (
        int(sha("hover-alloc:" + item["uid"]), 16) % 100 < HOVER_TWIN_SHARE
    ):
        return HOVER_TWINS
    return HOVER_COVERAGE


class WikiDB:
    """HoVer's ``wiki_wo_links.db`` (DrQA layout: ``documents(id, text)``, ids in
    NFD), opened read-only."""

    def __init__(self, path: Path) -> None:
        self.connection = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)

    def text(self, title: str) -> str | None:
        found = self.connection.execute(
            "select text from documents where id = ?",
            (unicodedata.normalize("NFD", title),),
        ).fetchone()
        return None if found is None else _clean(found[0])


def hotpot_questions(
    train_dir: Path, validation: Path, wanted: set[str]
) -> dict[str, str]:
    """HotpotQA id -> question for ``wanted`` ids (distractor TRAIN JSONL and the
    validation parquet of the pinned revision)."""
    found: dict[str, str] = {}
    for path in sorted(train_dir.glob(HOTPOTQA_TRAIN_FILES)):
        for item in read_jsonl(path):
            if item["id"] in wanted:
                found[item["id"]] = item["question"]
    missing = wanted - set(found)
    if missing and validation.exists():
        import pyarrow.parquet as pq

        table = pq.read_table(validation, columns=["id", "question"])
        for ident, question in zip(
            table.column("id").to_pylist(), table.column("question").to_pylist()
        ):
            if ident in missing:
                found[ident] = question
    return found


def hover_records(dirs: Dirs, family: str) -> tuple[list[tuple[str, Any]], Report]:
    """(family, record or drop reason) per HoVer TRAIN claim; claims allocated
    to another family carry ``None``."""
    root = Path(dirs["hover"])
    claims = json.loads((root / "hover_train_release_v1.1.json").read_text("utf-8"))
    retrieved = {
        item["id"]: item["doc_retrieval_results"][0][0]
        for item in json.loads(
            (root / "train_tfidf_doc_retrieval_results.json").read_text("utf-8")
        )
    }
    validation = (
        Path(dirs["hotpotqa_validation"])
        / "distractor/validation-00000-of-00001.parquet"
    )
    parents = hotpot_questions(
        Path(dirs["hotpotqa"]), validation, {item["hpqa_id"] for item in claims}
    )
    wiki = WikiDB(root / "wiki_wo_links.db")
    out: list[tuple[str, Any]] = []
    for item in claims:
        allocated = hover_family(item)
        if allocated != family:
            out.append((allocated, None))
            continue
        titles = list(dict.fromkeys(title for title, _ in item["supporting_facts"]))
        question = parents.get(item["hpqa_id"])
        if question is None:
            out.append((family, "hotpotqa_parent_missing"))
            continue
        gold = [(title, wiki.text(title)) for title in titles]
        if any(text is None for _, text in gold):
            out.append((family, "supporting_title_missing"))
            continue
        if not all(text for _, text in gold):
            out.append((family, "empty_supporting_paragraph"))
            continue
        support = {normalize(unicodedata.normalize("NFC", t)) for t in titles}
        texts = {normalize(text) for _, text in gold}
        distractors: list[Paragraph] = []
        for title in retrieved.get(item["uid"], []):
            name = unicodedata.normalize("NFC", title)
            if normalize(name) in support:
                continue
            text = wiki.text(title)
            if not text or normalize(text) in texts:
                continue
            texts.add(normalize(text))
            distractors.append((name, text))
        out.append(
            (
                family,
                {
                    "local_id": item["uid"],
                    "group_key": normalize(question),
                    "claim": item["claim"],
                    "verdict": item["label"],
                    "hops": item["num_hops"],
                    "gold": [(unicodedata.normalize("NFC", t), x) for t, x in gold],
                    "distractors": distractors,
                },
            )
        )
    report = {
        "inputs": _inputs(
            root,
            [
                root / "hover_train_release_v1.1.json",
                root / "train_tfidf_doc_retrieval_results.json",
                root / "wiki_wo_links.db",
            ],
        ),
        "claims": len(claims),
        "hotpotqa_parents": {
            "claims_parent_ids": len({item["hpqa_id"] for item in claims}),
            "found": len(parents),
        },
        "validation_input": {
            validation.name: file_sha256(validation) if validation.exists() else None
        },
    }
    return out, report


def build_hover(dirs: Dirs, *, family: str, seed: str) -> Built:
    records, report = hover_records(dirs, family)
    count = counter_for(dirs)
    construct = claim_twins if family == HOVER_TWINS else claim_coverage
    allocation: collections.Counter[str] = collections.Counter()
    drops: collections.Counter[str] = collections.Counter()
    verdicts: collections.Counter[str] = collections.Counter()
    rows: Rows = []
    for label, record in records:
        allocation[label] += 1
        if label != family:
            continue
        if isinstance(record, str):
            drops[record] += 1
            continue
        made, reason = construct(
            record,
            source=HOVER,
            family=family,
            namespace="multihop",
            template=f"m3b/{family}/v1",
            seed=seed,
            count=count,
        )
        if reason is not None:
            drops[reason] += 1
            continue
        verdicts[record["verdict"]] += 1
        rows.extend(made)
    return rows, {
        **report,
        "construction": (
            "page-scale C3 answerability twins (SUPPORTED claims)"
            if family == HOVER_TWINS
            else "page-scale C4 coverage Score (all claims)"
        ),
        "allocation_rule": f"SUPPORTED and sha256('hover-alloc:'+uid)%100 < "
        f"{HOVER_TWIN_SHARE}: {HOVER_TWINS}; otherwise {HOVER_COVERAGE}",
        "distractor_rule": "the claim's TF-IDF top-100 retrieved titles minus "
        "supporting titles, distinct normalized paragraphs, seeded order",
        "group_key": "textnorm.normalize(HotpotQA parent question), namespace multihop",
        "allocation": _sorted(allocation),
        "candidates": allocation[family],
        "drops": _sorted(drops),
        "kept_items": sum(verdicts.values()),
        "kept_by_verdict": _sorted(verdicts),
        "rows": len(rows),
        "targets": list(LONG_TARGETS),
        "state_cap": STATE_CAP,
    }


# Natural Questions (page windows)

_BREAK_TAG = re.compile(r"<(?:br|/p|/li|/td|/th|/tr|/div|/h\d|/dd|/dt)\b[^>]*>", re.I)
_SUP = re.compile(r"<sup\b[^>]*>.*?</sup>", re.I | re.S)
_TAG = re.compile(r"<[^>]*>")
_OPEN_TAG = re.compile(r"\s*<\s*([A-Za-z][A-Za-z0-9]*)")


def html_text(fragment: str) -> str:
    """Visible text of an HTML fragment: reference superscripts removed, block
    ends read as spaces, other tags removed, entities decoded."""
    text = _SUP.sub("", fragment)
    text = _BREAK_TAG.sub(" ", text)
    text = _TAG.sub("", text)
    return _clean(html.unescape(text))


def nq_example(row: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """A kept NQ TRAIN example: every top-level ``<p>`` candidate as text in page
    order, the index of the annotated long answer among them and the short
    answer strings (byte spans of the page HTML)."""
    annotations = row["annotations"]
    if len(annotations["id"]) != 1:
        return None, "not_one_annotation"
    long = annotations["long_answer"][0]
    if long["candidate_index"] < 0:
        return None, "no_long_answer"
    if annotations["yes_no_answer"][0] != -1:
        return None, "yes_no_answer"
    short = annotations["short_answers"][0]
    spans = list(zip(short["start_byte"], short["end_byte"]))
    if not spans:
        return None, "no_short_answer"
    candidates = row["long_answer_candidates"]
    page = row["html"].encode("utf-8")
    gold = long["candidate_index"]
    if not candidates["top_level"][gold]:
        return None, "long_answer_not_top_level"
    paragraphs: list[str] = []
    position = None
    for index, (start, end, top) in enumerate(
        zip(candidates["start_byte"], candidates["end_byte"], candidates["top_level"])
    ):
        if not top:
            continue
        fragment = page[start:end].decode("utf-8", errors="replace")
        tag = _OPEN_TAG.match(fragment)
        if tag is None or tag.group(1).lower() != "p":
            if index == gold:
                return None, "long_answer_not_paragraph"
            continue
        text = html_text(fragment)
        if index == gold:
            if not text:
                return None, "empty_long_answer"
            position = len(paragraphs)
        if text:
            paragraphs.append(text)
    low, high = candidates["start_byte"][gold], candidates["end_byte"][gold]
    if any(not low <= start < end <= high for start, end in spans):
        return None, "short_answer_outside_long_answer"
    answers = [
        html_text(page[start:end].decode("utf-8", errors="replace"))
        for start, end in spans
    ]
    if position is None or not all(answers):
        return None, "empty_short_answer"
    return {
        "id": str(row["id"]),
        "title": _clean(row["title"]),
        "question": _clean(row["text"]),
        "paragraphs": paragraphs,
        "gold": position,
        "answers": answers,
    }, None


NQ_COLUMNS = [
    "id",
    "document.title",
    "document.html",
    "question.text",
    "long_answer_candidates",
    "annotations",
]


def nq_extract_shard(path: str) -> tuple[str, list[dict[str, Any]], dict[str, int]]:
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=NQ_COLUMNS)
    kept, drops = [], collections.Counter()
    for batch in table.to_batches(max_chunksize=64):
        for row in batch.to_pylist():
            example, reason = nq_example(row)
            if example is None:
                drops[str(reason)] += 1
            else:
                kept.append(example)
    return path, kept, dict(drops)


def extract_nq(shards: Path, out: Path, report_path: Path, workers: int) -> Report:
    paths = sorted(str(path) for path in shards.glob("train-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no train-*.parquet under {shards}")
    with multiprocessing.get_context("fork").Pool(min(workers, len(paths))) as pool:
        results = pool.map(nq_extract_shard, paths, chunksize=1)
    drops: collections.Counter[str] = collections.Counter()
    lines, kept = [], 0
    for _, examples, dropped in sorted(results):
        drops.update(dropped)
        kept += len(examples)
        lines.extend(canonical(example) + "\n" for example in examples)
    data = "".join(lines).encode("utf-8")
    digest = common._write_new(out, data)
    report = {
        "schema": "decision2-m3b-nq-extract/v1",
        "inputs": {Path(p).name: file_sha256(Path(p)) for p in paths},
        "examples": kept + sum(drops.values()),
        "kept": kept,
        "drops": _sorted(drops),
        "rule": "one annotation; long answer a top-level <p> candidate; short "
        "answer spans inside it; yes/no NONE; paragraphs = top-level <p> "
        "candidates as visible text (html_text)",
        "output_sha256": digest,
    }
    common._write_new(
        report_path, (json.dumps(report, indent=1, sort_keys=True) + "\n").encode()
    )
    return report


def build_nq(dirs: Dirs, *, family: str, seed: str) -> Built:
    root = Path(dirs["nq"])
    path = root / "nq-train.jsonl"
    count = counter_for(dirs)
    drops: collections.Counter[str] = collections.Counter()
    rows: Rows = []
    seen: set[str] = set()
    records = 0
    for example in read_jsonl(path):
        records += 1
        if example["id"] in seen:
            drops["duplicate_id"] += 1
            continue
        seen.add(example["id"])
        made, reason = window_twins(
            {
                **example,
                "local_id": example["id"],
                "group_key": normalize(example["question"]),
                "language": "en",
            },
            source=NQ,
            family=family,
            namespace="nq",
            template=f"m3b/{family}/v1",
            seed=seed,
            count=count,
        )
        if reason is not None:
            drops[reason] += 1
            continue
        rows.extend(made)
    return rows, {
        "inputs": _inputs(root, [path]),
        "construction": "page-window answer-removal twins (window_twins)",
        "group_key": "textnorm.normalize(question), namespace nq",
        "records": records,
        "candidates": records,
        "drops": _sorted(drops),
        "pairs": len(rows) // 2,
        "targets": list(LONG_TARGETS),
        "min_window_tokens": MIN_WINDOW_TOKENS,
        "state_cap": STATE_CAP,
    }


# TyDi QA primary (page windows, H8)


def tydi_window_records(
    root: Path, code: str
) -> Iterator[tuple[dict[str, Any] | None, str | None]]:
    language = src_qa.TYDI_LANGUAGES[code]
    seen: set[str] = set()
    for path in src_qa._files(root, src_qa.TYDI_FILES):
        for number, record in src_qa.tydi_lines(path, language):
            where = f"{path.name}:{number}"
            question, title, url, document, starts, ends, marks = src_qa._tydi_fields(
                record, where
            )
            key = sha(f"{normalize(question)}\n{url}")
            if key in seen:
                yield None, "duplicate_example"
                continue
            seen.add(key)
            minimal = next(
                (
                    mark
                    for mark in marks
                    if mark[0] >= 0 and 0 <= mark[1] < mark[2] and mark[3] == "NONE"
                ),
                None,
            )
            if minimal is None:
                yield None, "no_minimal_answer"
                continue
            if not src_qa._valid_offsets(starts, ends, len(document), marks):
                yield None, "bad_passage_offsets"
                continue
            index, low, high, _ = minimal
            if not starts[index] <= low < high <= ends[index]:
                yield None, "minimal_outside_passage"
                continue
            try:
                passages = [
                    _clean(document[s:e].decode("utf-8"))
                    for s, e in zip(starts, ends, strict=True)
                ]
                answer = document[low:high].decode("utf-8").strip()
            except UnicodeDecodeError:
                yield None, "bad_byte_offsets"
                continue
            if not passages[index]:
                yield None, "empty_gold_passage"
                continue
            gold = sum(1 for text in passages[:index] if text)
            yield {
                "local_id": f"{path.stem}:{number}",
                "group_key": normalize(question),
                "language": code,
                "title": title,
                "question": question,
                "paragraphs": [text for text in passages if text],
                "gold": gold,
                "answers": [answer],
            }, None


def build_tydi_window(dirs: Dirs, *, family: str, seed: str, code: str) -> Built:
    root = Path(dirs["tydiqa"])
    count = counter_for(dirs)
    drops: collections.Counter[str] = collections.Counter()
    rows: Rows = []
    records = 0
    for record, reason in tydi_window_records(root, code):
        records += 1
        if record is not None:
            made, reason = window_twins(
                record,
                source=TYDI,
                family=family,
                namespace=src_qa.TYDI_NAMESPACE,
                template=f"m3b/tydi_window_removal/v1",
                seed=seed,
                count=count,
            )
            rows.extend(made)
        if reason is not None:
            drops[reason] += 1
    return rows, {
        "inputs": _inputs(root, src_qa._files(root, src_qa.TYDI_FILES)),
        "construction": "page-window answer-removal twins (window_twins) on the "
        "primary task's minimal answers",
        "language": code,
        "group_key": "textnorm.normalize(question_text), namespace tydi-miracl",
        "records": records,
        "candidates": records,
        "drops": _sorted(drops),
        "pairs": len(rows) // 2,
        "targets": list(LONG_TARGETS),
        "min_window_tokens": MIN_WINDOW_TOKENS,
        "state_cap": STATE_CAP,
    }


# MIRACL (es, fa, fr, hi, zh; MIRACL-English is excluded as HAGRID's parent)

MIRACL_LANGUAGES = ("es", "fa", "fr", "hi", "zh")
MIRACL_RELEVANCE = "miracl_relevance_{}"
MIRACL_POOL = "miracl_pool_{}"


def _tsv(path: Path) -> Iterator[list[str]]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            line = line.rstrip("\n")
            if line.strip():
                yield line.split("\t")


def miracl_records(dirs: Dirs, code: str) -> tuple[list[dict[str, Any]], Report]:
    root = Path(dirs["miracl"]) / f"miracl-v1.0-{code}"
    topics_path = root / "topics" / f"topics.miracl-v1.0-{code}-train.tsv"
    qrels_path = root / "qrels" / f"qrels.miracl-v1.0-{code}-train.tsv"
    topics = {fields[0]: fields[1] for fields in _tsv(topics_path)}
    judged: dict[str, dict[str, int]] = collections.defaultdict(dict)
    for fields in _tsv(qrels_path):
        qid, _, docid, relevance = fields
        judged[qid][docid] = int(relevance)
    wanted = {docid for docs in judged.values() for docid in docs}
    corpus_root = Path(dirs["miracl_corpus"]) / f"miracl-corpus-v1.0-{code}"
    corpus_files = sorted(
        corpus_root.glob("docs-*.jsonl.gz"),
        key=lambda path: int(path.name.split("-")[1].split(".")[0]),
    )
    passages: dict[str, Paragraph] = {}
    for path in corpus_files:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                if (
                    line.startswith('{"docid": "')
                    and line[11 : line.index('"', 11)] not in wanted
                ):
                    continue
                doc = json.loads(line)
                if doc["docid"] in wanted:
                    passages[doc["docid"]] = (_clean(doc["title"]), _clean(doc["text"]))
    records = []
    for qid in sorted(judged):
        if qid not in topics:
            continue
        docs = judged[qid]
        positives = [passages[d] for d in sorted(docs) if docs[d] > 0 and d in passages]
        negatives = [
            passages[d] for d in sorted(docs) if docs[d] <= 0 and d in passages
        ]
        records.append(
            {
                "local_id": f"{code}:{qid}",
                "group_key": normalize(topics[qid]),
                "language": code,
                "question": topics[qid],
                "positives": [p for p in dict.fromkeys(positives) if p[1]],
                "negatives": [p for p in dict.fromkeys(negatives) if p[1]],
            }
        )
    report = {
        "inputs": {
            **_inputs(Path(dirs["miracl"]), [topics_path, qrels_path]),
            **_inputs(Path(dirs["miracl_corpus"]), corpus_files),
        },
        "topics": len(topics),
        "judged_topics": len(judged),
        "judged_passages": len(wanted),
        "passages_found": len(passages),
    }
    return records, report


def miracl_kind(local_id: str) -> str:
    return "pool" if int(sha("miracl-alloc:" + local_id), 16) % 2 else "relevance"


def build_miracl(dirs: Dirs, *, family: str, seed: str, code: str, kind: str) -> Built:
    records, report = miracl_records(dirs, code)
    drops: collections.Counter[str] = collections.Counter()
    rows: Rows = []
    allocation: collections.Counter[str] = collections.Counter()
    construct = pool_twins if kind == "pool" else relevance_twins
    for record in records:
        label = miracl_kind(record["local_id"])
        allocation[label] += 1
        if label != kind:
            continue
        made, reason = construct(
            record,
            source=MIRACL,
            family=family,
            namespace=src_qa.TYDI_NAMESPACE,
            template=f"m3b/miracl_{kind}/v1",
            seed=seed,
        )
        if reason is not None:
            drops[reason] += 1
            continue
        rows.extend(made)
    return rows, {
        **report,
        "construction": (
            "judged-pool twins (pool_twins)"
            if kind == "pool"
            else "C2 relevance Noul twins"
        ),
        "language": code,
        "allocation_rule": "sha256('miracl-alloc:'+lang:qid)%2: 1 pool, 0 relevance",
        "allocation": _sorted(allocation),
        "group_key": "textnorm.normalize(query), namespace tydi-miracl",
        "candidates": allocation[kind],
        "drops": _sorted(drops),
        "pairs": len(rows) // 2,
    }


# JCommonsenseQA (C8 Choice, ja)

JCQA_FILE = Path("datasets/jcommonsenseqa-v1.3/train-v1.3.json")


def build_jcqa(dirs: Dirs, *, family: str, seed: str) -> Built:
    root = Path(dirs["jglue"])
    path = root / JCQA_FILE
    v1_ids = v1_local_ids(dirs, JCQA)
    items = list(read_jsonl(path))
    v1_questions = {
        normalize(item["question"])
        for item in items
        if item["q_id"] in v1_ids or str(item["q_id"]) in v1_ids
    }
    drops: collections.Counter[str] = collections.Counter()
    rows: Rows = []
    seen: set[str] = set()
    for item in items:
        local_id = str(item["q_id"])
        question = _clean(item["question"])
        choices = [_clean(item[f"choice{i}"]) for i in range(5)]
        if local_id in v1_ids:
            reason = "v1_item"
        elif normalize(question) in v1_questions:
            reason = "v1_question"
        elif local_id in seen:
            reason = "duplicate_id"
        elif not question or not all(choices):
            reason = "empty_text"
        elif len({normalize(choice) for choice in choices}) != 5:
            reason = "duplicate_choices"
        elif type(item["label"]) is not int or not 0 <= item["label"] < 5:
            reason = "bad_label"
        else:
            reason = None
        if reason is not None:
            drops[reason] += 1
            continue
        seen.add(local_id)
        rows.append(
            make_row(
                source=JCQA,
                family=family,
                task_type="choice",
                language="ja",
                namespace="jcqa",
                group_key=normalize(question),
                local_id=local_id,
                state=question,
                instructions=CHOICE_INSTRUCTIONS,
                options=choice_options(choices),
                label=item["label"],
                template=f"m3b/{family}/v1",
                audit={"original_label": item["label"]},
                rotate_choice=True,
            )
        )
    return rows, {
        "inputs": _inputs(root, [path]),
        "construction": "C8 Choice (5 options, English scaffold, rotation by seed)",
        "group_key": "textnorm.normalize(question), namespace jcqa",
        "items": len(items),
        "candidates": len(items),
        "v1_local_ids": len(v1_ids),
        "drops": _sorted(drops),
        "rows": len(rows),
    }


def v1_local_ids(dirs: Dirs, source: str) -> set[str]:
    manifest = dirs.get(V1_ROWS_KEY)
    if manifest is None:
        return set()
    ids: set[str] = set()
    for path in json.loads(Path(manifest).read_text(encoding="utf-8")):
        for row in read_jsonl(Path(path)):
            if row.get("source") == source:
                ids.add(str(row["audit_metadata"]["source_local_id"]))
    return ids


# SentiMix Spanglish (Score, L = 3)

SENTIMIX_MEMBER = "Semeval_2020_task9_data/Spanglish/Spanglish_train.conll"


def sentimix_items(text: str) -> tuple[list[tuple[str, str, str]], collections.Counter]:
    """(tweet id, text, label) per CoNLL block; a block starts at a
    ``meta<TAB>id<TAB>label`` line at the file start or after a blank line."""
    items, skipped = [], collections.Counter()
    header: list[str] | None = None
    tokens: list[str] = []
    fresh = True

    def close() -> None:
        if header is None:
            return
        if header[2] not in SENTIMENT_LABELS or not tokens:
            skipped["empty_or_unknown_label"] += 1
        else:
            items.append((header[1], " ".join(tokens), header[2]))

    for line in text.splitlines():
        fields = line.split("\t")
        if fresh and len(fields) == 3 and fields[0] == "meta":
            close()
            header, tokens = [field.strip() for field in fields], []
        elif line.strip():
            if header is not None and fields[0]:
                tokens.append(fields[0])
        fresh = not line.strip()
    close()
    return items, skipped


def build_sentimix(dirs: Dirs, *, family: str, seed: str) -> Built:
    root = Path(dirs["sentimix"])
    path = root / "Semeval_2020_task9_data.zip"
    with zipfile.ZipFile(path) as archive:
        data = archive.read(SENTIMIX_MEMBER)
    items, drops = sentimix_items(data.decode("utf-8"))
    labels: dict[str, set[str]] = collections.defaultdict(set)
    for _, text, label in items:
        labels[normalize(text)].add(label)
    candidates, seen = [], set()
    for ident, text, label in items:
        key = normalize(text)
        if len(labels[key]) > 1:
            drops["conflicting_labels"] += 1
        elif key in seen:
            drops["duplicate_text"] += 1
        else:
            seen.add(key)
            candidates.append((ident, text, label))
    kept, balance = ordinal.balance(
        candidates,
        cell=lambda item: family,
        level=lambda item: SENTIMENT_LABELS[item[2]],
        levels=lambda item: 3,
        ident=lambda item: item[0],
        seed=f"{seed}:balance",
    )
    drops["level_balance"] += len(candidates) - len(kept)
    rows = [
        make_row(
            source=SENTIMIX,
            family=family,
            task_type="score",
            language="es-en",
            namespace="sentimix-spanglish",
            group_key=normalize(text),
            local_id=ident,
            state=f"Message: {_clean(text)}",
            instructions=SENTIMENT_INSTRUCTIONS,
            options=score_options(SENTIMENT_LEVELS),
            label=SENTIMENT_LABELS[label],
            template=f"m3b/{family}/v1",
            audit={"upstream_label": label},
        )
        for ident, text, label in kept
    ]
    return rows, {
        "inputs": {path.name: file_sha256(path)},
        "member": {"name": SENTIMIX_MEMBER, "sha256": hashlib.sha256(data).hexdigest()},
        "construction": "Score L3 (negative / neutral / positive), A7s template",
        "group_key": "textnorm.normalize(text), namespace sentimix-spanglish",
        "items": len(items),
        "candidates": len(candidates),
        "drops": _sorted(drops),
        "balance_rule": "keep at most floor(6/5 * rarest level) per level in "
        "sha256(seed:balance:id) order",
        "balance": balance,
        "rows": len(rows),
    }


# Family registry and build


@dataclasses.dataclass(frozen=True)
class GapFamily:
    arm: str
    family: str
    source: str
    build: Callable[[Dirs], Built]
    cap_rows: int
    seed: str

    def __post_init__(self) -> None:
        if self.arm not in ARMS:
            raise ValueError(f"{self.family}: unknown arm {self.arm}")
        if self.cap_rows < 1:
            raise ValueError(f"{self.family}: cap_rows must be positive")


def _families() -> tuple[GapFamily, ...]:
    specs = [
        GapFamily(
            "h7",
            HOVER_TWINS,
            HOVER,
            functools.partial(
                build_hover, family=HOVER_TWINS, seed="h7-hover-twins-v1"
            ),
            4400,
            "h7-hover-twins-v1",
        ),
        GapFamily(
            "h7",
            HOVER_COVERAGE,
            HOVER,
            functools.partial(
                build_hover, family=HOVER_COVERAGE, seed="h7-hover-coverage-v1"
            ),
            3300,
            "h7-hover-coverage-v1",
        ),
        GapFamily(
            "h7",
            NQ_WINDOW,
            NQ,
            functools.partial(build_nq, family=NQ_WINDOW, seed="h7-nq-window-v1"),
            2600,
            "h7-nq-window-v1",
        ),
        GapFamily(
            "h8",
            "jcqa",
            JCQA,
            functools.partial(build_jcqa, family="jcqa", seed="h8-jcqa-v1"),
            9000,
            "h8-jcqa-v1",
        ),
        GapFamily(
            "h8",
            "sentimix_spanglish",
            SENTIMIX,
            functools.partial(
                build_sentimix, family="sentimix_spanglish", seed="h8-sentimix-v1"
            ),
            8000,
            "h8-sentimix-v1",
        ),
    ]
    for code in src_qa.TYDI_LANGUAGES:
        family = f"tydi_window_removal_{code}"
        seed = f"h8-tydi-window-{code}-v1"
        specs.append(
            GapFamily(
                "h8",
                family,
                TYDI,
                functools.partial(
                    build_tydi_window, family=family, seed=seed, code=code
                ),
                200,
                seed,
            )
        )
    for code in MIRACL_LANGUAGES:
        for kind, template, cap in (
            ("relevance", MIRACL_RELEVANCE, 1000),
            ("pool", MIRACL_POOL, 600),
        ):
            family = template.format(code)
            seed = f"h8-miracl-{kind}-{code}-v1"
            specs.append(
                GapFamily(
                    "h8",
                    family,
                    MIRACL,
                    functools.partial(
                        build_miracl, family=family, seed=seed, code=code, kind=kind
                    ),
                    cap,
                    seed,
                )
            )
    names = [spec.family for spec in specs]
    if len(set(names)) != len(names):
        raise ValueError("duplicate family names")
    return tuple(specs)


FAMILIES = _families()
_SELECTED: dict[str, GapFamily] = {}
_DIRS: dict[str, Path] = {}


def resolve(root: Path) -> dict[str, Path]:
    return {name: root / sub for name, sub in SOURCE_DIRS.items()}


def existing_keys(manifest: Path) -> tuple[set[str], set[str], list[dict[str, Any]]]:
    """Group ids and input hashes of every row file listed in ``manifest``."""
    groups: set[str] = set()
    inputs: set[str] = set()
    files = []
    for path in json.loads(manifest.read_text(encoding="utf-8")):
        count = 0
        for row in read_jsonl(Path(path)):
            groups.add(row["group_id"])
            inputs.add(row["input_sha256"])
            count += 1
        files.append(
            {"path": str(path), "sha256": file_sha256(Path(path)), "rows": count}
        )
    return groups, inputs, files


def isolate(
    rows: Rows, groups: set[str], inputs: set[str]
) -> tuple[Rows, dict[str, int]]:
    """Drop every group that shares a group id or any input hash with ``groups``
    / ``inputs``."""
    shared = {
        row["group_id"]
        for row in rows
        if row["group_id"] in groups or row["input_sha256"] in inputs
    }
    kept = [row for row in rows if row["group_id"] not in shared]
    return kept, {
        "groups_dropped": len(shared),
        "rows_dropped": len(rows) - len(kept),
        "by_group_id": len({row["group_id"] for row in rows} & groups),
    }


def _run_family(name: str) -> tuple[str, Rows, Report]:
    item = _SELECTED[name]
    rows, report = item.build(_DIRS)
    for row in rows:
        if row["family"] != item.family or row["source"] != item.source:
            raise ValueError(f"{name}: row {row['id']} has the wrong family/source")
    return name, rows, report


def build(
    arm: str,
    dirs: Dirs,
    *,
    existing: tuple[set[str], set[str]] = (set(), set()),
    workers: int = 1,
    families: Sequence[GapFamily] = FAMILIES,
) -> tuple[Rows, Report]:
    selected = [item for item in families if item.arm == arm]
    if not selected:
        raise ValueError(f"no families registered for arm {arm}")
    _SELECTED.clear()
    _SELECTED.update({item.family: item for item in selected})
    _DIRS.clear()
    _DIRS.update(dirs)
    names = [item.family for item in selected]
    if workers > 1 and len(names) > 1:
        with multiprocessing.get_context("fork").Pool(min(workers, len(names))) as pool:
            results = pool.map(_run_family, names, chunksize=1)
    else:
        results = [_run_family(name) for name in names]
    rows: Rows = []
    reports: dict[str, Any] = {}
    for name, built, report in sorted(results, key=lambda result: result[0]):
        item = _SELECTED[name]
        isolated, dropped = isolate(built, *existing)
        capped = common.cap_groups(isolated, item.cap_rows, item.seed)
        rows.extend(capped)
        reports[name] = {
            **report,
            "source": item.source,
            "candidate_rows": len(built),
            "isolation": dropped,
            "cap_rows": item.cap_rows,
            "cap_seed": item.seed,
            "after_cap": len(capped),
            "groups": len({row["group_id"] for row in capped}),
            "shortfall": max(0, item.cap_rows - len(capped)),
            "label_histogram_after_cap": dict(
                sorted(collections.Counter(str(row["label"]) for row in capped).items())
            ),
        }
    return rows, {
        "schema": SCHEMA,
        "arm": arm,
        "rules": RULES,
        "families": reports,
        "slices": "AHO sha256(group_id)%10==0; SHO sha256('sho-v2:'+group_id)%50==0 "
        "among the rest (m2.common.slice_of)",
        "status": "M3b amendment 2 build",
    }


# Post-build helpers


def project_protected(
    base: Path,
    extra: Sequence[tuple[str, Path]],
    out_dir: Path,
    *,
    base_sha256: str | None = None,
    expected: Mapping[str, str] | None = None,
    report_only: Sequence[str] = (),
) -> Report:
    """PI-v3 manifest entries plus input-only projections of ``extra`` row files
    (``build_protected_inventory.project_row``), written as a new manifest.
    ``base_sha256`` and ``expected`` (role -> SHA-256 of the origin file) are
    checked before anything is written; ``report_only`` roles must be in the
    manifest and are listed in the receipt."""
    from v2.data.build_protected_inventory import project_row

    expected = dict(expected or {})
    unknown = sorted(set(expected) - {role for role, _ in extra})
    if unknown:
        raise ValueError(f"expected hashes for roles not added: {unknown}")
    if base_sha256 is not None and file_sha256(base) != base_sha256:
        raise ValueError(f"{base}: SHA-256 differs from {base_sha256}")
    origins = {}
    for role, path in extra:
        origins[role] = file_sha256(path)
        if role in expected and origins[role] != expected[role]:
            raise ValueError(f"{role}: {path} SHA-256 differs from {expected[role]}")
    entries = [
        dict(entry, path=str((base.parent / entry["path"]).resolve()))
        for entry in json.loads(base.read_text(encoding="utf-8"))
    ]
    roles = {entry["role"] for entry in entries}
    missing = sorted(set(report_only) - roles - set(origins))
    if missing:
        raise ValueError(f"report-only roles not in the manifest: {missing}")
    out_dir.mkdir(parents=True, mode=0o700)
    added = []
    for role, path in extra:
        if role in roles:
            raise ValueError(f"duplicate role {role}")
        roles.add(role)
        rows = [project_row(row) for row in read_jsonl(path)]
        data = "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for row in rows
        ).encode("utf-8")
        target = out_dir / f"{role}.jsonl"
        digest = common._write_new(target, data)
        entries.append({"role": role, "path": str(target), "sha256": digest})
        added.append(
            {
                "role": role,
                "origin": str(path),
                "origin_sha256": origins[role],
                "origin_verified": role in expected,
                "rows": len(rows),
                "sha256": digest,
            }
        )
    manifest = (json.dumps(entries, indent=1, sort_keys=True) + "\n").encode("utf-8")
    report = {
        "schema": "decision2-m3b-protected/v1",
        "base_manifest_sha256": file_sha256(base),
        "base_verified": base_sha256 is not None,
        "added": added,
        "manifest_sha256": common._write_new(out_dir / "manifest.json", manifest),
        "roles": len(entries),
        "report_only_roles": sorted(set(report_only)),
    }
    common._write_new(
        out_dir / "receipt.json",
        (json.dumps(report, indent=1, sort_keys=True) + "\n").encode(),
    )
    return report


def quarantining_manifest(manifest: Path, receipt: Path, out: Path) -> Report:
    """The entries of ``manifest`` whose role is not report-only in its
    ``protected`` receipt, unchanged. A scan against this subset gives the
    quarantine decision without the report-only rows, which otherwise raise
    posting counts and protected-row boilerplate counts and so hide matches
    with quarantining roles."""
    data = json.loads(receipt.read_text(encoding="utf-8"))
    if data["manifest_sha256"] != file_sha256(manifest):
        raise ValueError(f"{receipt} does not describe {manifest}")
    report_only = set(data.get("report_only_roles", []))
    entries = [
        entry
        for entry in json.loads(manifest.read_text(encoding="utf-8"))
        if entry["role"] not in report_only
    ]
    digest = common._write_new(
        out, (json.dumps(entries, indent=1, sort_keys=True) + "\n").encode("utf-8")
    )
    report = {
        "schema": "decision2-m3b-protected-quarantining/v1",
        "from_manifest_sha256": data["manifest_sha256"],
        "dropped_report_only_roles": sorted(report_only),
        "roles": len(entries),
        "manifest_sha256": digest,
    }
    common._write_new(
        out.with_name(out.name.replace(".json", "") + ".receipt.json"),
        (json.dumps(report, indent=1, sort_keys=True) + "\n").encode(),
    )
    return report


def apply_budget(
    rows_path: Path, tokens_path: Path, out: Path, report_path: Path, limit: int = 8192
) -> Report:
    """Drop whole groups holding a row above ``limit`` native tokens; count rows
    above 1,024 Kai tokens (T1a-ineligible, kept)."""
    tokens = {line["id"]: line for line in read_jsonl(tokens_path)}
    rows = list(read_jsonl(rows_path))
    missing = [row["id"] for row in rows if row["id"] not in tokens]
    if missing:
        raise ValueError(f"{len(missing)} rows lack token counts, e.g. {missing[:3]}")
    over = {row["group_id"] for row in rows if tokens[row["id"]]["native"] > limit}
    kept = [row for row in rows if row["group_id"] not in over]
    data = "".join(canonical(row) + "\n" for row in sorted(kept, key=lambda r: r["id"]))
    digest = common._write_new(out, data.encode("utf-8"))
    native = [tokens[row["id"]]["native"] for row in kept]
    long_tokens = sum(value for value in native if value >= 2000)
    report = {
        "input_sha256": file_sha256(rows_path),
        "tokens_sha256": file_sha256(tokens_path),
        "limit": limit,
        "rows_in": len(rows),
        "groups_over_limit": len(over),
        "rows_kept": len(kept),
        "native_tokens": sum(native),
        "native_max": max(native, default=0),
        "long_evidence_tokens": long_tokens,
        "long_evidence_share": round(long_tokens / sum(native), 6) if native else 0.0,
        "kai_over_1024": sum(1 for row in kept if tokens[row["id"]]["kai"] > 1024),
        "output_sha256": digest,
    }
    common._write_new(
        report_path, (json.dumps(report, indent=1, sort_keys=True) + "\n").encode()
    )
    return report


def logistic_1d(
    xs: Sequence[float], ys: Sequence[float], steps: int = 25
) -> tuple[float, float]:
    """(weight, bias) of a one-feature logistic regression by Newton steps
    (a 1e-6 ridge keeps separable data finite)."""
    weight = bias = 0.0
    for _ in range(steps):
        gw = gb = hww = hwb = hbb = 0.0
        for x, y in zip(xs, ys):
            p = 1 / (1 + math.exp(-max(-35.0, min(35.0, weight * x + bias))))
            gw += (p - y) * x
            gb += p - y
            curve = p * (1 - p)
            hww += curve * x * x
            hwb += curve * x
            hbb += curve
        gw += 1e-6 * weight
        hww += 1e-6
        hbb += 1e-6
        det = hww * hbb - hwb * hwb
        if abs(det) < 1e-12:
            break
        weight -= (hbb * gw - hwb * gb) / det
        bias -= (hww * gb - hwb * gw) / det
    return weight, bias


def length_baseline(
    rows: Sequence[Mapping[str, Any]], folds: int = 5
) -> dict[str, Any]:
    """Group-disjoint (``shortcut.fold_of``) logistic regression of the label
    on standardized log state length in characters, per task type; Score and
    Choice use one-vs-rest fits per label. Compared with the cross-validated
    majority label on the same rows."""
    from v2.data.shortcut import fold_of

    def state_text(row: Mapping[str, Any]) -> str:
        state = row["state"]
        return state if isinstance(state, str) else canonical(state)

    out: dict[str, Any] = {}
    by_type: dict[str, list[Mapping[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        by_type[row["task_type"]].append(row)
    for task_type, members in sorted(by_type.items()):
        xs = [math.log(1 + len(state_text(row))) for row in members]
        labels = [row["label"] for row in members]
        fold = [fold_of(row["group_id"]) for row in members]
        levels = sorted(set(labels))
        correct = majority = 0
        for held in range(folds):
            train = [i for i in range(len(members)) if fold[i] != held]
            test = [i for i in range(len(members)) if fold[i] == held]
            if not train or not test:
                continue
            prior = collections.Counter(labels[i] for i in train).most_common(1)[0][0]
            mean = sum(xs[i] for i in train) / len(train)
            spread = (
                math.sqrt(sum((xs[i] - mean) ** 2 for i in train) / len(train)) or 1.0
            )
            zs = [(xs[i] - mean) / spread for i in train]
            fits = {
                level: logistic_1d(zs, [float(labels[i] == level) for i in train])
                for level in levels
            }
            for i in test:
                z = (xs[i] - mean) / spread
                guess = max(levels, key=lambda lv: (fits[lv][0] * z + fits[lv][1], -lv))
                correct += guess == labels[i]
                majority += prior == labels[i]
        out[task_type] = {
            "n": len(members),
            "accuracy": round(correct / len(members), 6),
            "majority": round(majority / len(members), 6),
            "delta": round((correct - majority) / len(members), 6),
        }
    return out


def slice_stats(rows: Sequence[Mapping[str, Any]], tokens: Mapping[str, Any]) -> Report:
    """Counts and native-token totals of one final slice ("long" = rows of at
    least 2,000 native tokens)."""
    native = [tokens[row["id"]]["native"] for row in rows]
    long_tokens = sum(value for value in native if value >= 2000)
    families: dict[str, dict[str, Any]] = {}
    for row, value in zip(rows, native):
        item = families.setdefault(
            row["family"],
            {"rows": 0, "native": 0, "long": 0, "groups": set(), "languages": set()},
        )
        item["rows"] += 1
        item["native"] += value
        item["long"] += value if value >= 2000 else 0
        item["groups"].add(row["group_id"])
        item["languages"].add(row["language"])
        item["type"] = row["task_type"]
    score = collections.Counter(
        (len(row["options"]), row["label"])
        for row in rows
        if row["task_type"] == "score"
    )

    def by(field: str, task_type: str | None = None) -> dict[str, int]:
        return _sorted(
            collections.Counter(
                str(row[field])
                for row in rows
                if task_type is None or row["task_type"] == task_type
            )
        )

    return {
        "rows": len(rows),
        "groups": len({row["group_id"] for row in rows}),
        "native_tokens": sum(native),
        "native_max": max(native, default=0),
        "long_evidence_tokens": long_tokens,
        "long_evidence_share": round(long_tokens / max(1, sum(native)), 4),
        "rows_ge_2000": sum(1 for value in native if value >= 2000),
        "kai_over_1024": sum(1 for row in rows if tokens[row["id"]]["kai"] > 1024),
        "task_type": by("task_type"),
        "language": by("language"),
        "source": by("source"),
        "families": {
            name: {
                "rows": item["rows"],
                "groups": len(item["groups"]),
                "native": item["native"],
                "long_share": round(item["long"] / max(1, item["native"]), 4),
                "type": item["type"],
                "language": sorted(item["languages"]),
            }
            for name, item in sorted(families.items())
        },
        "noul_labels": by("label", "noul"),
        "choice_gold_positions": by("label", "choice"),
        "score_levels": {
            f"L{levels}": {
                str(grade): score[(levels, grade)] for grade in range(levels)
            }
            for levels in sorted({levels for levels, _ in score})
        },
    }


GATE_FIELDS = ("task_type", "view", "n", "accuracy", "majority", "delta", "pass")


def gap_stats(
    arm: str, final: Path, audits: Path, embed_public: Path | None = None
) -> Report:
    """Per-slice statistics and aggregate gate results of a finalized arm; the
    SHO slice is reported by rows, groups and SHA-256 only."""
    tokens = {line["id"]: line for line in read_jsonl(final / f"{arm}.tokens.jsonl")}
    out: dict[str, Any] = {"schema": "decision2-m3b-gap-stats/v1", "arm": arm}
    for part in common.SLICES:
        path = final / f"{arm}.{part}.jsonl"
        rows = list(read_jsonl(path))
        stats = slice_stats(rows, tokens)
        if part == "sho":
            stats = {key: stats[key] for key in ("rows", "groups")}
        out[part] = {**stats, "sha256": file_sha256(path)}
    gates = {}
    for path in sorted(audits.glob("cells/*.jsonl.shortcut.json")):
        receipt = json.loads(path.read_text(encoding="utf-8"))
        gates[path.name.replace(".jsonl.shortcut.json", "")] = {
            "verdict": receipt["verdict"],
            "gates": [
                {key: gate[key] for key in GATE_FIELDS} for gate in receipt["gates"]
            ],
        }
    out["gates"] = gates
    out["length_baseline"] = json.loads(
        (audits / "length-baseline.json").read_text(encoding="utf-8")
    )

    def load(path: Path) -> Any:
        return json.loads(path.read_text(encoding="utf-8"))

    out["budget"] = {
        part: {
            key: value
            for key, value in load(audits / f"{arm}.{part}.budget.json").items()
            if key in ("rows_in", "groups_over_limit", "rows_kept", "native_max")
        }
        for part in common.SLICES
    }
    out["quarantine"] = {
        part: load(audits / f"{arm}.{part}.quarantine.json") for part in common.SLICES
    }
    out["gate_drops"] = {
        part: load(final / f"{arm}.{part}.gates.json") for part in common.SLICES
    }
    out["dedup"] = {
        part: load(final / f"{arm}.{part}.dedup.json") for part in ("aho", "sho")
    }
    isolation = load(final / f"{arm}.isolation.json")
    out["isolation"] = {
        "verdict": isolation["verdict"],
        "failures": len(isolation["failures"]),
        "partitions": len(isolation["partitions"]),
    }
    for key, name in (("overlap", "overlap"), ("overlap_quarantining", "overlap-q")):
        path = audits / f"{name}.public.json"
        if not path.exists():
            continue
        overlap = load(path)
        out[key] = {
            "protected_inventory_sha256": overlap["protected_inventory_sha256"],
            "by_method": overlap["flagged"]["by_method"],
            "quarantine": overlap["flagged"]["quarantine"],
            "by_role_any": {
                role: value["any"]
                for role, value in overlap["flagged"]["by_role"].items()
                if value["any"]["groups"]
            },
        }
    if embed_public is not None:
        embed = load(embed_public)
        out["embedding"] = {
            key: embed[key]
            for key in ("protected_manifest_sha256", "thresholds", "by_file")
        }
    return out


PAIR_PREFIXES = ("Sentence 1: ", "Sentence 2: ")


def pair_texts(row: Mapping[str, Any]) -> tuple[str, str] | None:
    """The two sentences of a sentence-pair row: A6h2 states are
    ``{sentence_1, sentence_2}``, A7k states ``Sentence 1: …`` / ``Sentence 2: …``
    lines; any other row is not a pair."""
    state = row["state"]
    if isinstance(state, Mapping):
        if set(state) == {"sentence_1", "sentence_2"}:
            return str(state["sentence_1"]), str(state["sentence_2"])
        return None
    lines = str(state).split("\n")
    if len(lines) == 2 and all(
        line.startswith(prefix) for line, prefix in zip(lines, PAIR_PREFIXES)
    ):
        return lines[0][len(PAIR_PREFIXES[0]) :], lines[1][len(PAIR_PREFIXES[1]) :]
    return None


def pair_check(pairs: Sequence[Path], against: Sequence[Path]) -> Report:
    """Counts of ``pairs`` rows whose input hash, or normalized sentence pair
    (in either order), occurs in the ``against`` files; rows sharing one
    normalized sentence with an ``against`` pair are counted separately."""
    inputs: set[str] = set()
    known: set[tuple[str, str]] = set()
    sentences: set[str] = set()
    against_files = []
    for path in against:
        rows = list(read_jsonl(path))
        for row in rows:
            inputs.add(row["input_sha256"])
            texts = pair_texts(row)
            if texts is not None:
                first, second = (normalize(text) for text in texts)
                known.add((first, second))
                sentences.update((first, second))
        against_files.append(
            {"name": path.name, "sha256": file_sha256(path), "rows": len(rows)}
        )
    fields = ("rows", "not_a_pair", "input_sha256", "pair", "sentence_shared")
    checked = []
    for path in pairs:
        found: collections.Counter[str] = collections.Counter()
        for row in read_jsonl(path):
            found["rows"] += 1
            found["input_sha256"] += row["input_sha256"] in inputs
            texts = pair_texts(row)
            if texts is None:
                found["not_a_pair"] += 1
                continue
            first, second = (normalize(text) for text in texts)
            found["pair"] += (first, second) in known or (second, first) in known
            found["sentence_shared"] += first in sentences or second in sentences
        checked.append(
            {
                "name": path.name,
                "sha256": file_sha256(path),
                **{field: found[field] for field in fields},
            }
        )
    return {
        "schema": "decision2-m3b-pair-check/v1",
        "against": against_files,
        "against_pairs": len(known),
        "pairs": checked,
        "key": "textnorm.normalize of both sentences, either order",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    nq = commands.add_parser("extract-nq")
    nq.add_argument("--shards", type=Path, required=True)
    nq.add_argument("--out", type=Path, required=True)
    nq.add_argument("--report", type=Path, required=True)
    nq.add_argument("--workers", type=int, default=32)
    run = commands.add_parser("build")
    run.add_argument("--arm", required=True, choices=ARMS)
    run.add_argument("--sources", type=Path, required=True)
    run.add_argument("--tokenizer", type=Path)
    run.add_argument("--existing", type=Path)
    run.add_argument("--v1-rows", type=Path)
    run.add_argument("--out-dir", type=Path, required=True)
    run.add_argument("--workers", type=int, default=16)
    protected = commands.add_parser("protected")
    protected.add_argument("--base", type=Path, required=True)
    protected.add_argument("--base-sha256")
    protected.add_argument("--add", action="append", default=[])
    protected.add_argument("--expect", action="append", default=[])
    protected.add_argument("--report-only", action="append", default=[])
    protected.add_argument("--out-dir", type=Path, required=True)
    quarantining = commands.add_parser("quarantining")
    quarantining.add_argument("--manifest", type=Path, required=True)
    quarantining.add_argument("--receipt", type=Path, required=True)
    quarantining.add_argument("--out", type=Path, required=True)
    pairs = commands.add_parser("pair-check")
    pairs.add_argument("--pairs", type=Path, action="append", required=True)
    pairs.add_argument("--against", type=Path, action="append", required=True)
    pairs.add_argument("--out", type=Path, required=True)
    stats = commands.add_parser("stats")
    stats.add_argument("--arm", required=True, choices=ARMS)
    stats.add_argument("--final", type=Path, required=True)
    stats.add_argument("--audits", type=Path, required=True)
    stats.add_argument("--embed-public", type=Path)
    stats.add_argument("--out", type=Path, required=True)
    budget = commands.add_parser("budget")
    budget.add_argument("--rows", type=Path, required=True)
    budget.add_argument("--tokens", type=Path, required=True)
    budget.add_argument("--out", type=Path, required=True)
    budget.add_argument("--report", type=Path, required=True)
    budget.add_argument("--limit", type=int, default=8192)
    lengths = commands.add_parser("length-baseline")
    lengths.add_argument("--cells", type=Path, required=True)
    lengths.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "extract-nq":
        report = extract_nq(args.shards, args.out, args.report, args.workers)
        print(
            json.dumps(
                {"kept": report["kept"], "drops": report["drops"]}, sort_keys=True
            )
        )
        return 0
    if args.command == "build":
        existing_files = [
            args.out_dir / f"{args.arm}.{part}.jsonl" for part in common.SLICES
        ] + [args.out_dir / f"{args.arm}.build.json"]
        if any(path.exists() for path in existing_files):
            parser.error(f"refusing to overwrite {args.arm} in {args.out_dir}")
        dirs = resolve(args.sources)
        if args.tokenizer is not None:
            dirs[TOKENIZER_KEY] = args.tokenizer
        if args.v1_rows is not None:
            dirs[V1_ROWS_KEY] = args.v1_rows
        groups: set[str] = set()
        inputs: set[str] = set()
        files: list[dict[str, Any]] = []
        if args.existing is not None:
            groups, inputs, files = existing_keys(args.existing)
        rows, report = build(
            args.arm, dirs, existing=(groups, inputs), workers=args.workers
        )
        report["existing"] = {
            "manifest_sha256": file_sha256(args.existing) if args.existing else None,
            "files": files,
            "groups": len(groups),
            "inputs": len(inputs),
        }
        report["tokenizer"] = (
            {
                "path": str(args.tokenizer),
                "files": _inputs(
                    args.tokenizer,
                    [
                        p
                        for p in args.tokenizer.iterdir()
                        if p.name.startswith(("tokenizer", "vocab", "merges"))
                    ],
                ),
            }
            if args.tokenizer
            else None
        )
        report["v1_rows_sha256"] = file_sha256(args.v1_rows) if args.v1_rows else None
        manifest = common.write_arm(rows, args.out_dir, args.arm, report)
        print(
            json.dumps(
                {
                    part: {k: manifest[part][k] for k in ("rows", "groups", "sha256")}
                    for part in common.SLICES
                },
                indent=1,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "protected":
        extra, expected = [], {}
        for item in args.add:
            role, _, path = item.partition("=")
            if not role or not path:
                parser.error("--add expects ROLE=PATH")
            extra.append((role, Path(path)))
        for item in args.expect:
            role, _, digest = item.partition("=")
            if not role or not re.fullmatch(r"[0-9a-f]{64}", digest):
                parser.error("--expect expects ROLE=SHA256")
            expected[role] = digest
        report = project_protected(
            args.base,
            extra,
            args.out_dir,
            base_sha256=args.base_sha256,
            expected=expected,
            report_only=args.report_only,
        )
        print(
            json.dumps(
                {"manifest_sha256": report["manifest_sha256"], "roles": report["roles"]}
            )
        )
        return 0
    if args.command == "quarantining":
        report = quarantining_manifest(args.manifest, args.receipt, args.out)
        print(json.dumps(report, sort_keys=True))
        return 0
    if args.command == "pair-check":
        report = pair_check(args.pairs, args.against)
        common._write_new(
            args.out, (json.dumps(report, indent=1, sort_keys=True) + "\n").encode()
        )
        print(json.dumps(report["pairs"], sort_keys=True))
        return 0
    if args.command == "stats":
        report = gap_stats(args.arm, args.final, args.audits, args.embed_public)
        common._write_new(
            args.out, (json.dumps(report, indent=1, sort_keys=True) + "\n").encode()
        )
        print(
            json.dumps(
                {
                    part: {
                        key: report[part][key]
                        for key in ("rows", "groups", "sha256")
                        + (("native_tokens",) if part != "sho" else ())
                    }
                    for part in common.SLICES
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "budget":
        report = apply_budget(args.rows, args.tokens, args.out, args.report, args.limit)
        print(json.dumps(report, sort_keys=True))
        return 0
    if args.command == "length-baseline":
        result = {}
        for path in sorted(args.cells.glob("*.jsonl")):
            result[path.stem] = length_baseline(list(read_jsonl(path)))
        common._write_new(
            args.out, (json.dumps(result, indent=1, sort_keys=True) + "\n").encode()
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
