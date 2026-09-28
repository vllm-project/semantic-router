"""Multi-hop long-evidence families from HotpotQA (distractor), 2WikiMultihopQA
and MuSiQue-Full TRAIN: C3 answerability twins (H3), C4 evidence-coverage Score
(H6) and C5 abstention Choice (E11) of data-arms-v2-prereg-2026-09-28.md.

Each source item becomes a record whose gold paragraphs follow the
supporting-fact order and whose distractors keep source order, and goes to
exactly one family of its source by type and
``sha256(f"{source}-alloc:" + id) % 100``. Rows of all three sources share the
``multihop`` namespace keyed by the normalized question (prereg section 1.3).
MuSiQue keeps its native answerable/unanswerable twins rendered as v1 A3 and
skips every id used by a v1 row.
"""

from __future__ import annotations

import collections
import functools
import json
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from training.model.data import file_sha256
from v2.data.m2 import constructions
from v2.data.m2.build import v1_local_ids
from v2.data.m2.common import (
    ABSTAIN,
    group_id,
    make_row,
    noul_options,
    read_jsonl,
    sha,
)
from v2.data.m2.spec import FamilySpec
from v2.data.sources import musique
from v2.data.textnorm import normalize

Rows = list[dict[str, Any]]
Record = dict[str, Any]
Paragraph = tuple[str, str]
Outcome = tuple[Record | None, str | None]
Construct = Callable[[Record], tuple[Rows, str | None]]
MusiqueEntry = tuple[str, dict[bool, dict[str, Any]] | None]

NAMESPACE = "multihop"
HOTPOTQA = "hotpotqa_distractor_train"
TWOWIKI = "twowiki_train"
MUSIQUE = musique.SOURCE
HOTPOTQA_FILES = "distractor/train-*.jsonl"
TWOWIKI_FILE = "train.jsonl"
TWIN_SHARE = 55
MUSIQUE_TWIN_SHARE = 70
TWIN_DISTRACTORS = 4
ABSTENTION_DISTRACTORS = 3
TEMPLATES = {
    "twins": "m2/answerability_twins/v1",
    "coverage": "m2/coverage/v1",
    "abstention": "m2/abstention/v1",
    "native_twins": "m2/musique_answerable/v1",
}

HOTPOTQA_TWINS = "hotpotqa_answerable"
HOTPOTQA_COVERAGE = "hotpotqa_coverage"
HOTPOTQA_ABSTENTION = "hotpotqa_abstention"
TWOWIKI_TWINS = "twowiki_answerable"
TWOWIKI_COVERAGE = "twowiki_coverage"
TWOWIKI_ABSTENTION = "twowiki_abstention"
MUSIQUE_TWINS = "musique_answerable_v2"
MUSIQUE_COVERAGE = "musique_coverage"
UNALLOCATED = "unallocated"
DUPLICATE = "duplicate_id"
V1_EXCLUDED = "v1_excluded"


def _clean(text: str) -> str:
    return " ".join(text.split())


def _json(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


def _sorted(counter: Mapping[str, int]) -> dict[str, int]:
    return dict(sorted(counter.items()))


def alloc(source: str, local_id: str) -> int:
    return int(sha(f"{source}-alloc:" + local_id), 16) % 100


def yes_no(answer: str) -> bool:
    return answer.strip().lower() in constructions.YES_NO


def hotpotqa_family(item: Mapping[str, Any]) -> str:
    if item["type"] == "bridge":
        low = alloc(HOTPOTQA, item["id"]) < TWIN_SHARE
        return HOTPOTQA_TWINS if low else HOTPOTQA_COVERAGE
    if item["type"] == "comparison":
        return HOTPOTQA_TWINS if yes_no(item["answer"]) else HOTPOTQA_ABSTENTION
    return UNALLOCATED


def twowiki_family(item: Mapping[str, Any]) -> str:
    if item["type"] in ("compositional", "inference"):
        low = alloc(TWOWIKI, item["_id"]) < TWIN_SHARE
        return TWOWIKI_TWINS if low else TWOWIKI_COVERAGE
    if item["type"] == "bridge_comparison":
        return TWOWIKI_COVERAGE
    if item["type"] == "comparison":
        return TWOWIKI_ABSTENTION
    return UNALLOCATED


def musique_family(key: str, v1_ids: set[str]) -> str:
    if key in v1_ids:
        return V1_EXCLUDED
    low = alloc(MUSIQUE, key) < MUSIQUE_TWIN_SHARE
    return MUSIQUE_TWINS if low else MUSIQUE_COVERAGE


def record(
    *,
    local_id: str,
    question: str,
    answer: str,
    qtype: str,
    paragraphs: Sequence[Paragraph],
    supporting: Sequence[str],
) -> Outcome:
    """Gold = the first paragraph of each distinct supporting title in
    supporting-fact order; distractors = the distinct non-empty paragraphs whose
    title is not a supporting title, in source order."""
    if not _clean(question):
        return None, "empty_question"
    if not _clean(answer):
        return None, "empty_answer"
    titles = list(dict.fromkeys(supporting))
    first: dict[str, str] = {}
    for title, text in paragraphs:
        first.setdefault(title, text)
    if not all(title in first for title in titles):
        return None, "missing_supporting_title"
    gold = [(title, first[title]) for title in titles]
    if not all(text for _, text in gold):
        return None, "empty_supporting_paragraph"
    support = frozenset(titles)
    distractors = list(
        dict.fromkeys(p for p in paragraphs if p[0] not in support and p[1])
    )
    return {
        "local_id": local_id,
        "group_key": normalize(question),
        "question": question,
        "answer": answer,
        "qtype": qtype,
        "gold": gold,
        "distractors": distractors,
    }, None


def hotpotqa_record(item: Mapping[str, Any]) -> Outcome:
    """Sentences carry their own leading spaces: joined with "" then collapsed."""
    context = item["context"]
    return record(
        local_id=item["id"],
        question=item["question"],
        answer=item["answer"],
        qtype=item["type"],
        paragraphs=[
            (title, _clean("".join(sentences)))
            for title, sentences in zip(
                context["title"], context["sentences"], strict=True
            )
        ],
        supporting=item["supporting_facts"]["title"],
    )


def twowiki_record(item: Mapping[str, Any]) -> Outcome:
    """``context`` and ``supporting_facts`` are JSON strings; sentences are
    joined with a space."""
    return record(
        local_id=item["_id"],
        question=item["question"],
        answer=item["answer"],
        qtype=item["type"],
        paragraphs=[
            (title, _clean(" ".join(sentences)))
            for title, sentences in _json(item["context"])
        ],
        supporting=[title for title, _ in _json(item["supporting_facts"])],
    )


def musique_pair(entry: MusiqueEntry) -> Outcome:
    key, pair = entry
    if pair is None:
        return None, "incomplete_twin_pair"
    question = pair[True]["question"]
    if not _clean(question):
        return None, "empty_question"
    if normalize(pair[False]["question"]) != normalize(question):
        return None, "twin_question_mismatch"
    return {
        "local_id": key,
        "group_key": normalize(question),
        "question": question,
        "qtype": key.split("__", 1)[0],
        "twins": pair,
    }, None


def musique_coverage_record(entry: MusiqueEntry) -> Outcome:
    """Answerable twin: gold = supporting paragraphs in paragraph order (one
    per hop), distractors = the other distinct non-empty paragraphs."""
    base, reason = musique_pair(entry)
    if base is None:
        return None, reason
    item = base["twins"][True]
    paragraphs = [
        (p["title"], _clean(p["paragraph_text"]), p["is_supporting"])
        for p in item["paragraphs"]
    ]
    gold = [(title, text) for title, text, supporting in paragraphs if supporting]
    if len(gold) != len(item["question_decomposition"]):
        return None, "supporting_count_not_hops"
    if not all(text for _, text in gold):
        return None, "empty_supporting_paragraph"
    golden = frozenset(gold)
    distractors = list(
        dict.fromkeys(
            (title, text)
            for title, text, supporting in paragraphs
            if not supporting and text and (title, text) not in golden
        )
    )
    return {
        **base,
        "answer": item["answer"],
        "gold": gold,
        "distractors": distractors,
    }, None


def native_twins(record: Record, *, family: str) -> tuple[Rows, str | None]:
    """MuSiQue's own answerable/unanswerable twins over all 20 paragraphs,
    rendered as the v1 A3 Noul rows."""
    rows = []
    for answerable in (True, False):
        item = record["twins"][answerable]
        twin = "answerable" if answerable else "unanswerable"
        rows.append(
            make_row(
                source=MUSIQUE,
                family=family,
                task_type="noul",
                language="en",
                namespace=NAMESPACE,
                group_key=record["group_key"],
                local_id=f"{record['local_id']}:{twin}",
                state=musique.render(item["paragraphs"]),
                instructions=constructions.ANSWERABLE_INSTRUCTIONS.format(
                    _clean(item["question"])
                ),
                options=noul_options("en"),
                label=int(answerable),
                template=TEMPLATES["native_twins"],
                audit={
                    "musique_id": record["local_id"],
                    "answerable": answerable,
                    "qtype": record["qtype"],
                },
            )
        )
    return rows, None


def _abstention(record: Record, **kwargs: Any) -> tuple[Rows, str | None]:
    """C5 needs an answer naming one of the two compared entities; a yes/no
    answer never does."""
    if yes_no(record["answer"]):
        return [], "yes_no_answer"
    return constructions.abstention_twins(record, **kwargs)


def _construct(
    kind: str, source: str, family: str, seed: str, padding: int
) -> Construct:
    if kind == "native_twins":
        return functools.partial(native_twins, family=family)
    shared = {
        "source": source,
        "family": family,
        "namespace": NAMESPACE,
        "template": TEMPLATES[kind],
        "seed": seed,
    }
    if kind == "twins":
        return functools.partial(
            constructions.answerability_twins, distractors=TWIN_DISTRACTORS, **shared
        )
    if kind == "coverage":
        return functools.partial(
            constructions.coverage_group, padding=padding, **shared
        )
    if kind == "abstention":
        return functools.partial(
            _abstention, distractors=ABSTENTION_DISTRACTORS, **shared
        )
    raise ValueError(f"unknown construction {kind}")


def _allocate(
    items: Iterable[Mapping[str, Any]],
    key: str,
    family: Callable[[Mapping[str, Any]], str],
) -> Iterator[tuple[str, Mapping[str, Any]]]:
    """(family, item) per source item; a repeated id keeps its first item."""
    seen: set[str] = set()
    for item in items:
        if item[key] in seen:
            yield DUPLICATE, item
            continue
        seen.add(item[key])
        yield family(item), item


def _musique_entries(
    path: Path, v1_ids: set[str], family: str
) -> tuple[list[tuple[str, MusiqueEntry]], set[str]]:
    """(allocation, (id, twins or None)) per MuSiQue id in id order, plus the
    normalized questions of the v1 ids. Only ``family`` keeps its items."""
    labels: dict[str, str] = {}
    members: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    v1_questions: set[str] = set()
    for item in read_jsonl(path):
        key = item["id"]
        if type(item["answerable"]) is not bool:
            raise ValueError(f"MuSiQue {key}: answerable must be a boolean")
        label = labels.setdefault(key, musique_family(key, v1_ids))
        if label == V1_EXCLUDED:
            v1_questions.add(normalize(item["question"]))
        elif label == family:
            members[key].append(item)
    paired = [
        item
        for key in sorted(members)
        if sorted(member["answerable"] for member in members[key]) == [False, True]
        for item in members[key]
    ]
    pairs = musique.twins(paired)
    entries = [(labels[key], (key, pairs.get(key))) for key in sorted(labels)]
    return entries, v1_questions


def _balance(rows: Rows) -> dict[str, Any]:
    grades: dict[str, collections.Counter[str]] = collections.defaultdict(
        collections.Counter
    )
    gold: collections.Counter[str] = collections.Counter()
    for row in rows:
        if row["task_type"] == "score":
            grades[str(len(row["options"]))][str(row["label"])] += 1
        elif row["task_type"] == "choice":
            text = row["options"][row["label"]]["description"]
            gold["abstain" if text == ABSTAIN else "entity"] += 1
    report: dict[str, Any] = {}
    if grades:
        report["grades_by_level"] = {
            level: _sorted(grades[level]) for level in sorted(grades, key=int)
        }
    if gold:
        report["gold_kind"] = _sorted(gold)
    return report


def _tally(
    family: str,
    allocated: Iterable[tuple[str, Any]],
    to_record: Callable[[Any], Outcome],
    construct: Construct,
) -> tuple[Rows, dict[str, Any]]:
    allocation: collections.Counter[str] = collections.Counter()
    drops: collections.Counter[str] = collections.Counter()
    qtypes: collections.Counter[str] = collections.Counter()
    rows: Rows = []
    for label, item in allocated:
        allocation[label] += 1
        if label != family:
            continue
        normalized, reason = to_record(item)
        if normalized is None:
            drops[str(reason)] += 1
            continue
        made, reason = construct(normalized)
        if reason is not None or not made:
            drops[reason or "no_rows"] += 1
            continue
        qtypes[str(normalized["qtype"])] += len(made)
        rows.extend(made)
    return rows, {
        "allocation": _sorted(allocation),
        "candidates": allocation[family],
        "drops": _sorted(drops),
        "kept_items": allocation[family] - sum(drops.values()),
        "rows": len(rows),
        "qtype_histogram": _sorted(qtypes),
        "label_histogram": _sorted(
            collections.Counter(str(row["label"]) for row in rows)
        ),
        **_balance(rows),
    }


def _inputs(root: Path, paths: Sequence[Path]) -> dict[str, str]:
    return {path.relative_to(root).as_posix(): file_sha256(path) for path in paths}


def build_hotpotqa(
    dirs: Mapping[str, Path], *, family: str, kind: str, seed: str, padding: int
) -> tuple[Rows, dict[str, Any]]:
    root = dirs["hotpotqa"]
    paths = sorted(root.glob(HOTPOTQA_FILES))
    if not paths:
        raise FileNotFoundError(f"no {HOTPOTQA_FILES} under {root}")
    items = (item for path in paths for item in read_jsonl(path))
    rows, report = _tally(
        family,
        _allocate(items, "id", hotpotqa_family),
        hotpotqa_record,
        _construct(kind, HOTPOTQA, family, seed, padding),
    )
    return rows, {"inputs": _inputs(root, paths), **report}


def build_twowiki(
    dirs: Mapping[str, Path], *, family: str, kind: str, seed: str, padding: int
) -> tuple[Rows, dict[str, Any]]:
    root = dirs["twowiki"]
    path = root / TWOWIKI_FILE
    rows, report = _tally(
        family,
        _allocate(read_jsonl(path), "_id", twowiki_family),
        twowiki_record,
        _construct(kind, TWOWIKI, family, seed, padding),
    )
    return rows, {"inputs": _inputs(root, [path]), **report}


def build_musique(
    dirs: Mapping[str, Path], *, family: str, kind: str, seed: str, padding: int
) -> tuple[Rows, dict[str, Any]]:
    root = dirs["musique"]
    path = root / musique.DATA
    v1_ids = {local_id.split(":")[0] for local_id in v1_local_ids(dirs, MUSIQUE)}
    entries, v1_questions = _musique_entries(path, v1_ids, family)
    rows, report = _tally(
        family,
        entries,
        musique_coverage_record if kind == "coverage" else musique_pair,
        _construct(kind, MUSIQUE, family, seed, padding),
    )
    shared = {group_id(NAMESPACE, question) for question in v1_questions}
    return rows, {
        "inputs": _inputs(root, [path]),
        **report,
        "v1_ids": len(v1_ids),
        "v1_shared_question_groups": len(shared & {row["group_id"] for row in rows}),
    }


def _spec(
    arm: str,
    source: str,
    family: str,
    kind: str,
    cap_rows: int,
    seed: str,
    padding: int = 0,
) -> FamilySpec:
    builder = {
        HOTPOTQA: build_hotpotqa,
        TWOWIKI: build_twowiki,
        MUSIQUE: build_musique,
    }[source]
    return FamilySpec(
        arm=arm,
        family=family,
        source=source,
        build=functools.partial(
            builder, family=family, kind=kind, seed=seed, padding=padding
        ),
        cap_rows=cap_rows,
        seed=seed,
    )


FAMILIES: tuple[FamilySpec, ...] = (
    _spec("h3", HOTPOTQA, HOTPOTQA_TWINS, "twins", 16000, "h3-hotpotqa-v1"),
    _spec(
        "h6",
        HOTPOTQA,
        HOTPOTQA_COVERAGE,
        "coverage",
        9000,
        "h6-hotpotqa-coverage-v1",
        padding=2,
    ),
    _spec(
        "e11",
        HOTPOTQA,
        HOTPOTQA_ABSTENTION,
        "abstention",
        4000,
        "e11-hotpotqa-abstention-v1",
    ),
    _spec("h3", TWOWIKI, TWOWIKI_TWINS, "twins", 16000, "h3-twowiki-v1"),
    _spec(
        "h6",
        TWOWIKI,
        TWOWIKI_COVERAGE,
        "coverage",
        9000,
        "h6-twowiki-coverage-v1",
        padding=2,
    ),
    _spec(
        "e11",
        TWOWIKI,
        TWOWIKI_ABSTENTION,
        "abstention",
        4000,
        "e11-twowiki-abstention-v1",
    ),
    _spec("h3", MUSIQUE, MUSIQUE_TWINS, "native_twins", 16000, "h3-musique-v2"),
    _spec(
        "h6",
        MUSIQUE,
        MUSIQUE_COVERAGE,
        "coverage",
        6000,
        "h6-musique-coverage-v1",
        padding=3,
    ),
)
