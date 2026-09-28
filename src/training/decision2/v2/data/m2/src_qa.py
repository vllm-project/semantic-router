"""Extractive-QA sources of the v2 arms: C1 evidence-removal twins (E11, H5)
and TyDi QA primary C2 relevance (H5).

Rules: data-arms-v2-prereg-2026-09-28.md sections 1.3, 1.7 and 3. C1 reads
answerable questions only, passes each distinct (question, passage) once and
groups by the normalized passage (TyDi QA: by the normalized question, in the
namespace shared with MIRACL). A pair is also dropped when its removed twin
still states an answer through the title or through a mention that the
word-boundary matcher of ``text.mentions`` misses (see ``states``).

TyDi QA primary TRAIN is streamed line by line and lines of other languages are
skipped before parsing; byte offsets index the UTF-8 document. Allocation is
disjoint: an example with a minimal answer goes to C1 when
``int(sha256("tydi-alloc:" + normalized question), 16) % 2 == 0`` and to C2
otherwise, like every passage-only answer; examples without a passage answer
are dropped. One pass per language builds all of its families; the last
language is cached in the process.
"""

from __future__ import annotations

import collections
import dataclasses
import functools
import itertools
import json
import re
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from training.model.data import file_sha256
from v2.data.m2 import constructions
from v2.data.m2.common import read_jsonl, sha
from v2.data.m2.spec import FamilySpec
from v2.data.m2.text import mentions, units
from v2.data.textnorm import normalize

Record = dict[str, Any]
Rows = list[dict[str, Any]]
Built = tuple[Rows, dict[str, Any]]
Mark = tuple[int, int, int, str]

REMOVAL_TEMPLATE = "m2/removal_twins/v1"
RELEVANCE_TEMPLATE = "m2/tydi_relevance/v1"
PASSAGE_KEY = "textnorm.normalize(context)"
CONTAINMENT_LANGUAGES = frozenset(
    {"zh", "zh-hant", "ja", "ko", "th", "ar", "bn", "te"}  # codespell:ignore te
)
CANNOTANSWER = "CANNOTANSWER"
JSQUAD_SEPARATOR = " [SEP] "
GERMANQUAD_HEAD = re.compile(r"[^\n]*\n(?:\s*=+[^\n]*=+[ \t]*\n)*")

TYDI = "tydiqa"
TYDI_SOURCE = "tydiqa_primary_train"
TYDI_NAMESPACE = "tydi-miracl"
TYDI_FILES = "primary_task/train-*.jsonl"
TYDI_LANGUAGES = {
    "ar": "arabic",
    "bn": "bengali",
    "en": "english",
    "fi": "finnish",
    "id": "indonesian",
    "ja": "japanese",
    "ko": "korean",
    "ru": "russian",
    "sw": "swahili",
    "te": "telugu",  # codespell:ignore te
    "th": "thai",
}
CONTEXT_UNITS = 6
CONTEXT_CHARS = 3000
GOLD_CHARS = 1500
NEGATIVE_CHARS = (200, 1500)
NEGATIVES = 3
ALLOCATION_RULE = (
    "minimal answer (first annotation with passage index >= 0, start >= 0, "
    "end > start, yes_no_answer NONE): C1 when int(sha256('tydi-alloc:' + "
    "normalized question), 16) % 2 == 0, else C2; passage answer only: C2; "
    "no passage answer: dropped"
)
CONTEXT_RULE = (
    "gold passage plus neighbouring candidates (previous, next, alternating; a "
    f"side stops at its first neighbour that does not fit) joined by newlines "
    f"until >= {CONTEXT_UNITS} text.units or >= {CONTEXT_CHARS} characters, "
    f"adding at most {CONTEXT_CHARS} characters; a gold passage above "
    f"{CONTEXT_CHARS} characters is dropped"
)
RELEVANCE_RULE = (
    "gold = passage of the first annotation with a passage answer (at most "
    f"{GOLD_CHARS} characters after whitespace normalization); negatives = "
    "candidates no annotation selected, "
    f"{NEGATIVE_CHARS[0]}-{NEGATIVE_CHARS[1]} characters after whitespace "
    f"normalization, distinct normalized texts in document order; at least "
    f"{NEGATIVES} required"
)
_LANGUAGE_KEY = b'"language": "'
_TYDI_CACHE: dict[tuple[Any, ...], dict[str, Built]] = {}


def states(text: str, answer: str, language: str) -> bool:
    """Whether ``text`` states ``answer``: ``text.mentions`` with underscores
    read as spaces, and plain normalized containment in CONTAINMENT_LANGUAGES
    (no spaces between words, or particles and clitics written attached)."""
    text, answer = text.replace("_", " "), answer.replace("_", " ")
    if language in CONTAINMENT_LANGUAGES:
        needle = normalize(answer)
        return bool(needle) and needle in normalize(text)
    return mentions(text, answer)


def removal_pair(
    record: Mapping[str, Any], *, source: str, family: str, namespace: str, seed: str
) -> tuple[Rows, str | None]:
    """C1 twins of one record, dropped when the removed twin still states an answer."""
    rows, reason = constructions.removal_twins(
        record,
        source=source,
        family=family,
        namespace=namespace,
        template=REMOVAL_TEMPLATE,
        seed=seed,
    )
    if reason is not None:
        return [], reason
    language = record["language"]
    answers = [answer for answer in dict.fromkeys(record["answers"]) if answer]
    if any(states(record.get("title") or "", a, language) for a in answers):
        return [], "answer_in_title"
    removed = next(row["state"] for row in rows if row["label"] == 0)
    if any(states(removed, answer, language) for answer in answers):
        return [], "answer_left_after_removal"
    return rows, None


def _string(value: Any, what: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{what} must be a string")
    return value


def _title(value: Any) -> str:
    return "" if value is None else _string(value, "title")


def _files(root: Path, pattern: str) -> list[Path]:
    paths = sorted(root.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no {pattern} under {root}")
    return paths


def _inputs(root: Path, paths: Sequence[Path]) -> dict[str, str]:
    return {path.relative_to(root).as_posix(): file_sha256(path) for path in paths}


def qa_record(
    local_id: Any,
    *,
    title: str,
    context: Any,
    question: Any,
    answers: Any,
    starts: Any,
    shift: int = 0,
    drop: str | None = None,
) -> Record:
    """One C1 candidate; ``drop`` names why it is not used."""
    ident = str(local_id) if type(local_id) is int else local_id
    if not isinstance(ident, str) or not ident.strip():
        raise ValueError(f"record id {local_id!r} must be a nonempty string or int")
    context = _string(context, f"{ident}: context")
    question = _string(question, f"{ident}: question")
    if (
        not isinstance(answers, list)
        or not isinstance(starts, list)
        or len(answers) != len(starts)
        or not all(isinstance(answer, str) for answer in answers)
        or not all(type(start) is int for start in starts)
    ):
        raise ValueError(f"{ident}: answers must be strings aligned with int offsets")
    if drop is None and not any(answer.strip() for answer in answers):
        drop = "unanswerable"
    if drop is None and not (context.strip() and question.strip()):
        drop = "empty_text"
    return {
        "local_id": ident,
        "title": title,
        "context": context,
        "question": question,
        "answers": answers,
        "answer_starts": [start - shift for start in starts],
        "drop": drop,
    }


def read_hf(
    path: Path, *, title: Callable[[Mapping[str, Any]], str]
) -> Iterator[Record]:
    """HF SQuAD-style JSONL: id, context, question, answers{text, answer_start}."""
    for record in read_jsonl(path):
        answers = record.get("answers")
        if not isinstance(answers, dict):
            raise ValueError(f"{path}: {record.get('id')!r}: answers must be an object")
        yield qa_record(
            record.get("id"),
            title=title(record),
            context=record.get("context"),
            question=record.get("question"),
            answers=answers.get("text"),
            starts=answers.get("answer_start"),
        )


def read_germanquad(path: Path) -> Iterator[Record]:
    """GermanQuAD JSONL; the leading title line and section header lines move
    unchanged into the title, so a twin never removes them."""
    for record in read_hf(path, title=_no_title):
        match = GERMANQUAD_HEAD.match(record["context"])
        cut = match.end() if match else 0
        yield {
            **record,
            "title": record["context"][:cut].strip(),
            "context": record["context"][cut:],
            "answer_starts": [start - cut for start in record["answer_starts"]],
        }


def read_klue(path: Path) -> Iterator[Record]:
    """KLUE-MRC TRAIN; ``is_impossible`` questions are dropped."""
    for record in read_jsonl(path):
        impossible = record.get("is_impossible")
        answers = record.get("answers")
        if type(impossible) is not bool or not isinstance(answers, dict):
            raise ValueError(f"{path}: {record.get('guid')!r}: malformed record")
        yield qa_record(
            record.get("guid"),
            title=_title(record.get("title")),
            context=record.get("context"),
            question=record.get("question"),
            answers=answers.get("text"),
            starts=answers.get("answer_start"),
            drop="unanswerable" if impossible else None,
        )


def _articles(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("data"), list):
        raise ValueError(f"{path}: expected SQuAD-style JSON with a data list")
    return data["data"]


def read_squad_json(
    path: Path,
    *,
    title: Callable[[Mapping[str, Any]], str],
    title_prefix: bool = False,
) -> Iterator[Record]:
    """SQuAD JSON (data[].paragraphs[].qas[]). ``title_prefix`` strips the
    JSQuAD ``<title> [SEP] `` context prefix and shifts the answer offsets."""
    for article in _articles(path):
        name = title(article)
        for paragraph in article["paragraphs"]:
            context = _string(paragraph.get("context"), f"{path}: context")
            shift = 0
            if title_prefix:
                head, separator, body = context.partition(JSQUAD_SEPARATOR)
                if separator:
                    context, shift = body, len(head) + len(separator)
            for qa in paragraph["qas"]:
                answers = qa.get("answers") or []
                yield qa_record(
                    qa.get("id"),
                    title=name,
                    context=context,
                    question=qa.get("question"),
                    answers=[answer.get("text") for answer in answers],
                    starts=[answer.get("answer_start") for answer in answers],
                    shift=shift,
                    drop="unanswerable" if qa.get("is_impossible") else None,
                )


def read_quac(path: Path) -> Iterator[Record]:
    """The first question of every QuAC section; a CANNOTANSWER first question
    drops the section. The trailing CANNOTANSWER token leaves the context."""
    for article in _articles(path):
        parts = [
            _string(article.get(field), f"{path}: {field}").strip()
            for field in ("title", "section_title")
        ]
        title = " — ".join(part for part in parts if part)
        for paragraph in article["paragraphs"]:
            questions = paragraph.get("qas") or []
            if not questions:
                raise ValueError(f"{path}: {paragraph.get('id')!r}: no questions")
            first = questions[0]
            answer = first.get("orig_answer") or {}
            context = _string(paragraph.get("context"), f"{path}: context")
            if context.endswith(CANNOTANSWER):
                context = context[: -len(CANNOTANSWER)]
            yield qa_record(
                first.get("id"),
                title=title,
                context=context.rstrip(),
                question=first.get("question"),
                answers=[answer.get("text")],
                starts=[answer.get("answer_start")],
                drop="cannotanswer" if answer.get("text") == CANNOTANSWER else None,
            )


def _no_title(_: Mapping[str, Any]) -> str:
    return ""


def _field_title(record: Mapping[str, Any]) -> str:
    return _title(record.get("title"))


def _squad_title(record: Mapping[str, Any]) -> str:
    """SQuAD titles are Wikipedia URL forms (``Frédéric_Chopin``)."""
    return _title(record.get("title")).replace("_", " ")


def _sqac_title(article: Mapping[str, Any]) -> str:
    """SQAC AnCora newswire articles carry file names as titles."""
    title = _title(article.get("title"))
    return "" if title.endswith(".txt") else title


@dataclasses.dataclass(frozen=True)
class RemovalSource:
    arm: str
    family: str
    source: str
    directory: str
    pattern: str
    language: str
    namespace: str
    cap_rows: int
    seed: str
    reader: Callable[[Path], Iterable[Record]]
    title_rule: str
    exclude_v1: bool = False


REMOVAL_SOURCES = (
    RemovalSource(
        "e11",
        "squad2_removal",
        "squad2_train",
        "squad2",
        "squad_v2/train-*.jsonl",
        "en",
        "squad",
        16000,
        "e11-squad2-v1",
        functools.partial(read_hf, title=_squad_title),
        "title with underscores read as spaces",
    ),
    RemovalSource(
        "e11",
        "quac_removal",
        "quac_train_v0.2",
        "quac",
        "train_v0.2.json",
        "en",
        "quac",
        10000,
        "e11-quac-v1",
        read_quac,
        "title — section_title; first question of each section only",
    ),
    RemovalSource(
        "h5",
        "jsquad_removal",
        "jsquad_v1.3_train",
        "jglue",
        "datasets/jsquad-v1.3/train-v1.3.json",
        "ja",
        "jsquad",
        6000,
        "h5-jsquad-v1",
        functools.partial(read_squad_json, title=_field_title, title_prefix=True),
        "article title; the '<title> [SEP] ' context prefix is removed",
    ),
    RemovalSource(
        "h5",
        "klue_mrc_removal",
        "klue_mrc_train",
        "klue",
        "mrc/train.jsonl",
        "ko",
        "klue-mrc",
        6000,
        "h5-klue-mrc-v1",
        read_klue,
        "news title",
        exclude_v1=True,
    ),
    RemovalSource(
        "h5",
        "cmrc2018_removal",
        "cmrc2018_train",
        "cmrc2018",
        "data/train-*.jsonl",
        "zh",
        "cmrc2018",
        6000,
        "h5-cmrc2018-v1",
        functools.partial(read_hf, title=_no_title),
        "none (no title field)",
    ),
    RemovalSource(
        "h5",
        "drcd_removal",
        "drcd_train",
        "drcd",
        "DRCD_training.json",
        "zh-hant",
        "drcd",
        6000,
        "h5-drcd-v1",
        functools.partial(read_squad_json, title=_no_title),
        "none (article titles are simplified script; passages are traditional)",
    ),
    RemovalSource(
        "h5",
        "germanquad_removal",
        "germanquad_train",
        "germanquad",
        "plain_text/train/*.jsonl",
        "de",
        "germanquad",
        6000,
        "h5-germanquad-v1",
        read_germanquad,
        "the context's leading title line and section header lines, unchanged",
    ),
    RemovalSource(
        "h5",
        "piaf_removal",
        "piaf_train",
        "piaf",
        "plain_text/train-*.jsonl",
        "fr",
        "piaf",
        6000,
        "h5-piaf-v1",
        functools.partial(read_hf, title=_field_title),
        "title",
    ),
    RemovalSource(
        "h5",
        "sqac_removal",
        "sqac_train",
        "sqac",
        "train.json",
        "es",
        "sqac",
        6000,
        "h5-sqac-v1",
        functools.partial(read_squad_json, title=_sqac_title),
        "article title unless it is a file name (AnCora)",
    ),
)


def _v1_excluded(
    dirs: Mapping[str, Path], item: RemovalSource, paths: Sequence[Path]
) -> tuple[frozenset[str], frozenset[str]]:
    """Local ids of v1 rows of the same source and every passage they used."""
    if not item.exclude_v1:
        return frozenset(), frozenset()
    from v2.data.m2.build import v1_local_ids

    ids = frozenset(v1_local_ids(dirs, item.source))
    if not ids:
        return ids, frozenset()
    passages = frozenset(
        normalize(record["context"])
        for path in paths
        for record in item.reader(path)
        if record["local_id"] in ids
    )
    return ids, passages


def build_removal(dirs: Mapping[str, Path], *, item: RemovalSource) -> Built:
    root = Path(dirs[item.directory])
    paths = _files(root, item.pattern)
    v1_ids, v1_passages = _v1_excluded(dirs, item, paths)
    drops: collections.Counter[str] = collections.Counter()
    seen_ids: set[str] = set()
    seen_pairs: set[str] = set()
    rows: Rows = []
    records = candidates = 0
    context, passage = None, ""
    for path in paths:
        for record in item.reader(path):
            records += 1
            if record["context"] != context:
                context, passage = record["context"], normalize(record["context"])
            pair = sha(normalize(record["question"]) + "\n" + passage)
            if record["drop"] is not None:
                reason = record["drop"]
            elif record["local_id"] in v1_ids:
                reason = "v1_item"
            elif passage in v1_passages:
                reason = "v1_passage"
            elif pair in seen_pairs:
                reason = "duplicate_question"
            elif record["local_id"] in seen_ids:
                reason = "duplicate_id"
            else:
                seen_pairs.add(pair)
                seen_ids.add(record["local_id"])
                candidates += 1
                made, reason = removal_pair(
                    {**record, "group_key": passage, "language": item.language},
                    source=item.source,
                    family=item.family,
                    namespace=item.namespace,
                    seed=item.seed,
                )
                rows.extend(made)
            if reason is not None:
                drops[reason] += 1
    report: dict[str, Any] = {
        "inputs": _inputs(root, paths),
        "construction": "C1 constructions.removal_twins + states() guard",
        "template": REMOVAL_TEMPLATE,
        "language": item.language,
        "namespace": item.namespace,
        "group_key": PASSAGE_KEY,
        "title_rule": item.title_rule,
        "records": records,
        "candidates": candidates,
        "pairs": len(rows) // 2,
        "drops": dict(sorted(drops.items())),
    }
    if item.exclude_v1:
        report["v1_excluded"] = {"items": len(v1_ids), "passages": len(v1_passages)}
    return rows, report


def tydi_lines(path: Path, language: str) -> Iterator[tuple[int, dict[str, Any]]]:
    """(line number, record) of one TyDi language; a line whose serialized
    ``"language": "<name>"`` names another language is skipped unparsed."""
    wanted = _LANGUAGE_KEY + language.encode("ascii") + b'"'
    with path.open("rb") as stream:
        for number, line in enumerate(stream, 1):
            if line.isspace():
                continue
            at = line.rfind(_LANGUAGE_KEY)
            if at >= 0 and not line.startswith(wanted, at):
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{number}: expected a JSON object")
            if record.get("language") == language:
                yield number, record


def _ints(value: Any, what: str) -> list[int]:
    if not isinstance(value, list) or not all(type(item) is int for item in value):
        raise ValueError(f"{what} must be a list of integers")
    return value


def _tydi_fields(
    record: Mapping[str, Any], where: str
) -> tuple[str, str, str, bytes, list[int], list[int], list[Mark]]:
    question = _string(record.get("question_text"), f"{where}: question_text")
    if not question.strip():
        raise ValueError(f"{where}: empty question_text")
    document = _string(record.get("document_plaintext"), f"{where}: document")
    candidates = record.get("passage_answer_candidates")
    annotations = record.get("annotations")
    if not isinstance(candidates, dict) or not isinstance(annotations, dict):
        raise ValueError(f"{where}: candidates and annotations must be objects")
    starts = _ints(candidates.get("plaintext_start_byte"), f"{where}: starts")
    ends = _ints(candidates.get("plaintext_end_byte"), f"{where}: ends")
    columns = [
        _ints(annotations.get(field), f"{where}: {field}")
        for field in (
            "passage_answer_candidate_index",
            "minimal_answers_start_byte",
            "minimal_answers_end_byte",
        )
    ]
    kinds = annotations.get("yes_no_answer")
    if (
        len(starts) != len(ends)
        or not isinstance(kinds, list)
        or any(len(column) != len(kinds) for column in columns)
    ):
        raise ValueError(f"{where}: misaligned candidates or annotations")
    return (
        question,
        _title(record.get("document_title")),
        _title(record.get("document_url")),
        document.encode("utf-8"),
        starts,
        ends,
        list(zip(*columns, kinds, strict=True)),
    )


def _valid_offsets(
    starts: Sequence[int], ends: Sequence[int], size: int, marks: Sequence[Mark]
) -> bool:
    """Candidates in document order, disjoint and inside the document."""
    bounds = [0, *(x for pair in zip(starts, ends, strict=True) for x in pair), size]
    return all(low <= high for low, high in itertools.pairwise(bounds)) and all(
        -1 <= mark[0] < len(starts) for mark in marks
    )


def tydi_window(texts: Sequence[str], gold: int, language: str) -> list[int]:
    """Candidate indices of the C1 context around ``gold`` (CONTEXT_RULE)."""
    chosen = [gold]
    size, count, added = len(texts[gold]), len(units(texts[gold], language)), 0
    nearest, step, open_sides, side = [gold - 1, gold + 1], (-1, 1), [True, True], 0
    while any(open_sides) and count < CONTEXT_UNITS and size < CONTEXT_CHARS:
        if open_sides[side]:
            index = nearest[side]
            if (
                0 <= index < len(texts)
                and added + len(texts[index]) + 1 <= CONTEXT_CHARS
            ):
                chosen.append(index)
                nearest[side] += step[side]
                added += len(texts[index]) + 1
                size += len(texts[index]) + 1
                count += len(units(texts[index], language))
            else:
                open_sides[side] = False
        side = 1 - side
    return sorted(chosen)


def tydi_removal_record(
    document: bytes,
    passages: Sequence[str],
    starts: Sequence[int],
    ends: Sequence[int],
    mark: Mark,
    *,
    local_id: str,
    group_key: str,
    question: str,
    title: str,
    language: str,
) -> tuple[Record | None, str | None]:
    """C1 record: the minimal answer inside the ``tydi_window`` context."""
    index, low, high, _ = mark
    if not starts[index] <= low < high <= ends[index]:
        return None, "minimal_outside_passage"
    try:
        head = document[starts[index] : low].decode("utf-8")
        span = document[low:high].decode("utf-8")
    except UnicodeDecodeError:
        return None, "bad_byte_offsets"
    answer = span.strip()
    texts = [passage.strip() for passage in passages]
    if not answer:
        return None, "empty_minimal_answer"
    if len(texts[index]) > CONTEXT_CHARS:
        return None, "gold_passage_too_long"
    window = tydi_window(texts, index, language)
    raw = passages[index]
    start = (
        sum(len(texts[i]) + 1 for i in window if i < index)
        + len(head)
        - (len(raw) - len(raw.lstrip()))
        + (len(span) - len(span.lstrip()))
    )
    context = "\n".join(texts[i] for i in window)
    if context[start : start + len(answer)] != answer:
        return None, "answer_offset_mismatch"
    return {
        "local_id": local_id,
        "group_key": group_key,
        "language": language,
        "title": title,
        "context": context,
        "question": question,
        "answers": [answer],
        "answer_starts": [start],
    }, None


def tydi_relevance_record(
    passages: Sequence[str],
    marks: Sequence[Mark],
    index: int,
    *,
    local_id: str,
    group_key: str,
    question: str,
    language: str,
) -> tuple[Record | None, str | None]:
    """C2 record: the annotated passage and never-selected negatives (RELEVANCE_RULE)."""
    cleaned = [" ".join(passage.split()) for passage in passages]
    gold = cleaned[index]
    if not gold:
        return None, "empty_gold_passage"
    if len(gold) > GOLD_CHARS:
        return None, "gold_passage_too_long"
    selected = {mark[0] for mark in marks if mark[0] >= 0}
    seen = {normalize(gold)}
    negatives = []
    for position, text in enumerate(cleaned):
        if (
            position in selected
            or not NEGATIVE_CHARS[0] <= len(text) <= NEGATIVE_CHARS[1]
        ):
            continue
        key = normalize(text)
        if key not in seen:
            seen.add(key)
            negatives.append(text)
    if len(negatives) < NEGATIVES:
        return None, "too_few_negatives"
    return {
        "local_id": local_id,
        "group_key": group_key,
        "language": language,
        "question": question,
        "gold": gold,
        "negatives": negatives,
    }, None


def tydi_families(code: str) -> dict[str, str]:
    names = {"removal": f"tydi_removal_{code}"}
    if code != "en":
        names["choice"] = f"tydi_relevance_choice_{code}"
        names["noul"] = f"tydi_relevance_noul_{code}"
    return names


def tydi_removal_seed(code: str) -> str:
    return f"{'e11' if code == 'en' else 'h5'}-tydi-removal-{code}-v1"


def _tydi_build(root: Path, code: str) -> dict[str, Built]:
    language = TYDI_LANGUAGES[code]
    paths = _files(root, TYDI_FILES)
    names = tydi_families(code)
    removal_seed = tydi_removal_seed(code)
    relevance_seed = f"h5-tydi-relevance-{code}-v1"
    rows: dict[str, Rows] = {family: [] for family in names.values()}
    shared: collections.Counter[str] = collections.Counter()
    allocation: collections.Counter[str] = collections.Counter()
    drops = {"removal": collections.Counter(), "relevance": collections.Counter()}
    seen: set[str] = set()
    records = short_gold = 0
    for path in paths:
        for number, record in tydi_lines(path, language):
            records += 1
            local_id = f"{path.stem}:{number}"
            question, title, url, document, starts, ends, marks = _tydi_fields(
                record, f"{path.name}:{number}"
            )
            group_key = normalize(question)
            key = sha(f"{group_key}\n{url}")
            if key in seen:
                shared["duplicate_example"] += 1
                continue
            seen.add(key)
            passage = next((mark for mark in marks if mark[0] >= 0), None)
            if passage is None:
                shared["no_passage_answer"] += 1
                continue
            if not _valid_offsets(starts, ends, len(document), marks):
                shared["bad_passage_offsets"] += 1
                continue
            try:
                passages = [
                    document[s:e].decode("utf-8")
                    for s, e in zip(starts, ends, strict=True)
                ]
            except UnicodeDecodeError:
                shared["bad_byte_offsets"] += 1
                continue
            minimal = next(
                (
                    mark
                    for mark in marks
                    if mark[0] >= 0 and 0 <= mark[1] < mark[2] and mark[3] == "NONE"
                ),
                None,
            )
            if minimal is not None and int(sha("tydi-alloc:" + group_key), 16) % 2 == 0:
                allocation["removal"] += 1
                item, reason = tydi_removal_record(
                    document,
                    passages,
                    starts,
                    ends,
                    minimal,
                    local_id=local_id,
                    group_key=group_key,
                    question=question,
                    title=title,
                    language=code,
                )
                if item is not None:
                    made, reason = removal_pair(
                        item,
                        source=TYDI_SOURCE,
                        family=names["removal"],
                        namespace=TYDI_NAMESPACE,
                        seed=removal_seed,
                    )
                    rows[names["removal"]].extend(made)
                if reason is not None:
                    drops["removal"][reason] += 1
                continue
            allocation["relevance"] += 1
            if "choice" not in names:
                continue
            item, reason = tydi_relevance_record(
                passages,
                marks,
                passage[0],
                local_id=local_id,
                group_key=group_key,
                question=question,
                language=code,
            )
            if item is not None:
                built, reason = constructions.relevance(
                    item,
                    source=TYDI_SOURCE,
                    choice_family=names["choice"],
                    noul_family=names["noul"],
                    namespace=TYDI_NAMESPACE,
                    template=RELEVANCE_TEMPLATE,
                    seed=relevance_seed,
                )
                for family, made in built.items():
                    rows[family].extend(made)
                if reason is None and len(item["gold"]) < NEGATIVE_CHARS[0]:
                    short_gold += 1
            if reason is not None:
                drops["relevance"][reason] += 1
    base = {
        "inputs": _inputs(root, paths),
        "language": code,
        "tydi_language": language,
        "namespace": TYDI_NAMESPACE,
        "group_key": "textnorm.normalize(question_text)",
        "records": records,
        "allocation": {
            "removal": allocation["removal"],
            "relevance": allocation["relevance"],
            "dropped": dict(sorted(shared.items())),
        },
        "allocation_rule": ALLOCATION_RULE,
    }
    removal = rows[names["removal"]]
    built_rows = {
        names["removal"]: (
            removal,
            {
                **base,
                "construction": "C1 constructions.removal_twins + states() guard",
                "template": REMOVAL_TEMPLATE,
                "seed": removal_seed,
                "context_rule": CONTEXT_RULE,
                "candidates": allocation["removal"],
                "pairs": len(removal) // 2,
                "drops": dict(sorted((shared + drops["removal"]).items())),
            },
        )
    }
    for kind in ("choice", "noul"):
        if kind in names:
            built_rows[names[kind]] = (
                rows[names[kind]],
                {
                    **base,
                    "construction": f"C2 constructions.relevance ({kind})",
                    "template": f"{RELEVANCE_TEMPLATE}/{kind}",
                    "seed": relevance_seed,
                    "relevance_rule": RELEVANCE_RULE,
                    "candidates": allocation["relevance"],
                    "drops": dict(sorted((shared + drops["relevance"]).items())),
                    "gold_under_200_chars": short_gold,
                },
            )
    return built_rows


def build_tydi(dirs: Mapping[str, Path], *, family: str, code: str) -> Built:
    """One TyDi family; every family of ``code`` comes from one cached pass."""
    root = Path(dirs[TYDI])
    stamp = tuple(
        (path.name, path.stat().st_size, path.stat().st_mtime_ns)
        for path in _files(root, TYDI_FILES)
    )
    key = (str(root), code, stamp)
    if key not in _TYDI_CACHE:
        _TYDI_CACHE.clear()
        _TYDI_CACHE[key] = _tydi_build(root, code)
    rows, report = _TYDI_CACHE[key][family]
    return list(rows), dict(report)


def _families() -> tuple[FamilySpec, ...]:
    specs = [
        FamilySpec(
            item.arm,
            item.family,
            item.source,
            functools.partial(build_removal, item=item),
            item.cap_rows,
            item.seed,
        )
        for item in REMOVAL_SOURCES
    ]
    for code in TYDI_LANGUAGES:
        names = tydi_families(code)
        specs.append(
            FamilySpec(
                "e11" if code == "en" else "h5",
                names["removal"],
                TYDI_SOURCE,
                functools.partial(build_tydi, family=names["removal"], code=code),
                4000,
                tydi_removal_seed(code),
            )
        )
        for kind, cap in (("choice", 1500), ("noul", 3000)):
            if kind in names:
                specs.append(
                    FamilySpec(
                        "h5",
                        names[kind],
                        TYDI_SOURCE,
                        functools.partial(build_tydi, family=names[kind], code=code),
                        cap,
                        f"h5-tydi-relevance-{kind}-{code}-v1",
                    )
                )
    return tuple(specs)


FAMILIES = _families()
