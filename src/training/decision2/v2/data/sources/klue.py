"""KLUE (ko) projections: A5 YNAT, NLI and MRC rows and A6h STS Score rows.

Only publisher TRAIN files are read. States never carry labels, scores or
source fields (YNAT url/date, NLI/MRC/STS source, MRC news_category/answers).
KLUE-STS ``source`` is kept in audit metadata as the balance stratum. The NLI
and STS projection helpers are shared with the JGLUE module.
"""

from __future__ import annotations

import collections
import json
import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from training.model.data import file_sha256
from v2.data.sources import ordinal
from v2.data.sources.common import (
    choice_options,
    make_row,
    noul_options,
    rotate,
    score_options,
)
from v2.data.textnorm import normalize

A5 = "a5"
A6H = "a6h"
YNAT_FILE = Path("ynat/train.jsonl")
NLI_FILE = Path("nli/train.jsonl")
MRC_FILE = Path("mrc/train.jsonl")
STS_FILE = Path("sts/train.jsonl")

YNAT_SECTIONS = (
    "IT·과학: 정보통신, 인터넷, 과학기술 관련 기사",
    "경제: 금융, 산업, 기업, 부동산 등 경제 관련 기사",
    "사회: 사건·사고, 교육, 노동, 환경 등 사회 관련 기사",
    "생활·문화: 건강, 여행, 공연·전시, 날씨 등 생활·문화 관련 기사",
    "세계: 해외 각국 소식과 국제 정세 관련 기사",
    "스포츠: 경기 결과와 선수·구단 소식 등 스포츠 관련 기사",
    "정치: 국회·정당, 행정, 북한, 외교·국방 관련 기사",
)
YNAT_INSTRUCTIONS = "다음 뉴스 제목이 어느 분야의 기사인지 고르세요."

NLI_LABELS = ("entailment", "neutral", "contradiction")
NLI_TEXT = {
    "instructions": "전제가 사실이라고 할 때, 전제와 가설의 관계로 가장 알맞은 것을 고르세요.",
    "options": (
        "전제가 사실이면 가설도 반드시 사실이다",
        "전제만으로는 가설이 사실인지 아닌지 알 수 없다",
        "전제가 사실이면 가설은 사실일 수 없다",
    ),
}

MRC_STATE = "제목: {title}\n본문: {context}"
MRC_INSTRUCTIONS = "지문의 내용만으로 다음 질문에 답할 수 있습니까?\n질문: {question}"

STS_LEVELS = (3, 4, 5, 6)
STS_TOP = 5
STS_GUARD = 0.2
STS_TEXT = {
    "state": "문장 1: {first}\n문장 2: {second}",
    "instructions": (
        "두 문장의 의미가 얼마나 비슷한지 평가하세요. 유사도는 0점(의미상 전혀 관련 없음)부터 "
        "5점(의미가 완전히 같음)까지이며, 각 등급의 점수 범위는 여러 평가자가 매긴 점수의 "
        "평균을 기준으로 합니다."
    ),
    "range": "{phrase} ({low}–{high}점)",
    "anchors": (
        "두 문장은 의미상 전혀 관련이 없다",
        "주제는 비슷하지만 내용은 서로 다르다",
        "세부 내용이 일부 겹치지만 핵심 의미는 다르다",
        "핵심 의미는 대체로 같지만 중요한 정보가 다르거나 빠져 있다",
        "의미가 거의 같고 사소한 세부 사항만 다르다",
        "두 문장의 의미가 완전히 같다",
    ),
    "bands": {
        3: (
            "두 문장의 의미가 서로 다르다",
            "두 문장의 의미가 부분적으로 겹친다",
            "두 문장의 의미가 거의 같다",
        ),
        4: (
            "두 문장의 의미가 거의 관련이 없다",
            "주제나 일부 내용은 겹치지만 핵심 의미는 다르다",
            "핵심 의미는 비슷하지만 눈에 띄는 차이가 있다",
            "두 문장의 의미가 거의 같다",
        ),
        5: (
            "두 문장의 의미가 전혀 관련이 없다",
            "주제는 겹치지만 의미는 대부분 다르다",
            "두 문장의 의미가 부분적으로 겹친다",
            "핵심 의미는 대체로 같지만 차이가 있다",
            "두 문장의 의미가 거의 같다",
        ),
    },
}


def read_jsonl(path: Path) -> Iterator[tuple[str, dict[str, Any]]]:
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{number}: expected a JSON object")
            yield f"{path}:{number}", record


def text(record: Mapping[str, Any], field: str, where: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where}: {field} must be a nonempty string")
    return value.strip()


def ident(record: Mapping[str, Any], field: str, where: str) -> str:
    value = record.get(field)
    if type(value) is int:
        return str(value)
    return text(record, field, where)


def index(value: Any, size: int, where: str) -> int:
    if type(value) is not int or not 0 <= value < size:
        raise ValueError(f"{where}: label {value!r} is not an index below {size}")
    return value


def rating(value: Any, low: float, high: float, where: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{where}: rating {value!r} is not a finite number")
    if not low <= value <= high:
        raise ValueError(f"{where}: rating {value} outside [{low}, {high}]")
    return float(value)


def input_receipt(root: Path, relative: Path, records: int) -> dict[str, Any]:
    return {
        "file": relative.as_posix(),
        "sha256": file_sha256(root / relative),
        "records": records,
    }


def choice_row(descriptions: Sequence[str], gold: int, **fields: Any) -> dict[str, Any]:
    """Choice row whose options are rotated by sha256('<arm>-v1:' + row id)."""
    options = choice_options(descriptions)
    probe = make_row(task_type="choice", options=options, label=gold, **fields)
    options, label = rotate(options, gold, f"{fields['arm']}-v1:{probe['id']}")
    return make_row(task_type="choice", options=options, label=label, **fields)


def nli_group_keys(
    pairs: Sequence[tuple[str, str]], keys: Sequence[str] | None = None
) -> list[str]:
    """Group key per pair (default: normalized premise); the keys of mutually
    reversed pairs are merged into the smallest one."""
    normalized = [(normalize(p), normalize(h)) for p, h in pairs]
    keys = list(keys) if keys is not None else [premise for premise, _ in normalized]
    key_of: dict[tuple[str, str], list[str]] = collections.defaultdict(list)
    for pair, key in zip(normalized, keys):
        key_of[pair].append(key)
    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for (premise, hypothesis), own in sorted(key_of.items()):
        for key in own + key_of.get((hypothesis, premise), []):
            left, right = find(own[0]), find(key)
            if left != right:
                parent[max(left, right)] = min(left, right)
    return [find(key) for key in keys]


def nli_rows(
    items: Sequence[tuple[str, str, str, int]],
    *,
    family: str,
    source: str,
    language: str,
    texts: Mapping[str, Any],
    template: str,
    keys: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    keys = nli_group_keys(
        [(premise, hypothesis) for _, premise, hypothesis, _ in items], keys
    )
    return [
        choice_row(
            texts["options"],
            label,
            arm=A5,
            source=source,
            family=family,
            language=language,
            group_key=key,
            local_id=local_id,
            state={"premise": premise, "hypothesis": hypothesis},
            instructions=texts["instructions"],
            render_template=template,
            audit={"original_label": NLI_LABELS[label]},
        )
        for (local_id, premise, hypothesis, label), key in zip(items, keys)
    ]


@dataclass(frozen=True)
class RatedPair:
    local_id: str
    first: str
    second: str
    mean: float
    stratum: str | None = None
    group: str | None = None


def sts_cuts(means: Sequence[float]) -> dict[int, list[float]]:
    cuts = {levels: ordinal.quantile_cuts(means, levels) for levels in STS_LEVELS[:-1]}
    cuts[STS_LEVELS[-1]] = ordinal.anchor_cuts(STS_TOP)
    return cuts


def sts_descriptions(
    cuts: Mapping[int, Sequence[float]], texts: Mapping[str, Any]
) -> dict[int, list[str]]:
    descriptions = {}
    for levels, points in cuts.items():
        phrases = (
            texts["anchors"] if levels == STS_LEVELS[-1] else texts["bands"][levels]
        )
        descriptions[levels] = [
            texts["range"].format(
                phrase=phrase, low=ordinal.number(low), high=ordinal.number(high)
            )
            for phrase, (low, high) in zip(phrases, ordinal.bands(points, 0, STS_TOP))
        ]
    return descriptions


def sts_rows(
    pairs: Sequence[RatedPair],
    *,
    family: str,
    source: str,
    language: str,
    texts: Mapping[str, Any],
    template: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Guard-banded Score rows; cut points are fitted on every TRAIN pair given."""
    cuts = sts_cuts([pair.mean for pair in pairs])
    descriptions = sts_descriptions(cuts, texts)
    drops: collections.Counter[int] = collections.Counter()
    rows = []
    for pair in pairs:
        levels = ordinal.level_count(family, pair.local_id, STS_LEVELS)
        if ordinal.near_cut(pair.mean, cuts[levels], STS_GUARD):
            drops[levels] += 1
            continue
        anchored = levels == STS_LEVELS[-1]
        label = (
            ordinal.anchor_level(pair.mean, STS_TOP)
            if anchored
            else ordinal.bin_index(pair.mean, cuts[levels])
        )
        audit: dict[str, Any] = {
            "mean_rating": pair.mean,
            "levels": levels,
            "binning": "integer_anchor" if anchored else "train_quantile",
        }
        if pair.stratum is not None:
            audit["stratum"] = pair.stratum
        rows.append(
            make_row(
                arm=A6H,
                source=source,
                family=family,
                task_type="score",
                language=language,
                group_key=pair.group or pair.local_id,
                local_id=pair.local_id,
                state=texts["state"].format(first=pair.first, second=pair.second),
                instructions=texts["instructions"],
                options=score_options(descriptions[levels]),
                label=label,
                render_template=f"{template}_{'anchor' if anchored else 'quantile'}_v1",
                audit=audit,
            )
        )
    report = {
        "level_counts": list(STS_LEVELS),
        "level_count_rule": f"{STS_LEVELS}[int(sha256(f'{family}:{{local_id}}'), 16)"
        f" % {len(STS_LEVELS)}]",
        "cut_rule": "L=6: k+0.5 anchor midpoints, level = round(mean) clamped 0..5; "
        "L=3..5: linear-interpolated quantiles k/L of every TRAIN mean rating, "
        "level = number of cuts <= mean",
        "cut_points": {f"L{levels}": cuts[levels] for levels in STS_LEVELS},
        "guard_band": STS_GUARD,
        "guard_band_rule": f"drop if |mean - cut| <= {STS_GUARD} (+{ordinal.TOLERANCE})",
        "guard_band_drops": {f"L{levels}": drops[levels] for levels in STS_LEVELS},
        "candidates": len(pairs),
        "kept_after_guard_band": len(rows),
    }
    return rows, report


def ynat(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, classes = [], collections.Counter()
    for where, record in read_jsonl(root / YNAT_FILE):
        guid = ident(record, "guid", where)
        label = index(record.get("label"), len(YNAT_SECTIONS), where)
        classes[label] += 1
        rows.append(
            choice_row(
                YNAT_SECTIONS,
                label,
                arm=A5,
                source="klue_ynat_train",
                family="klue_ynat",
                language="ko",
                group_key=guid,
                local_id=guid,
                state=text(record, "title", where),
                instructions=YNAT_INSTRUCTIONS,
                render_template="klue_ynat_headline_section_v1",
                audit={"original_label": label},
            )
        )
    return rows, {
        "input": input_receipt(root, YNAT_FILE, len(rows)),
        "train_class_histogram": {
            YNAT_SECTIONS[label].split(":")[0]: classes[label]
            for label in range(len(YNAT_SECTIONS))
        },
    }


def nli(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    items = []
    for where, record in read_jsonl(root / NLI_FILE):
        items.append(
            (
                ident(record, "guid", where),
                text(record, "premise", where),
                text(record, "hypothesis", where),
                index(record.get("label"), len(NLI_LABELS), where),
            )
        )
    rows = nli_rows(
        items,
        family="klue_nli",
        source="klue_nli_train",
        language="ko",
        texts=NLI_TEXT,
        template="klue_nli_premise_hypothesis_json_v1",
    )
    classes = collections.Counter(NLI_LABELS[item[3]] for item in items)
    return rows, {
        "input": input_receipt(root, NLI_FILE, len(items)),
        "train_class_histogram": {name: classes[name] for name in NLI_LABELS},
    }


def mrc(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    for where, record in read_jsonl(root / MRC_FILE):
        guid = ident(record, "guid", where)
        impossible = record.get("is_impossible")
        if type(impossible) is not bool:
            raise ValueError(f"{where}: is_impossible must be a boolean")
        context = text(record, "context", where)
        rows.append(
            make_row(
                arm=A5,
                source="klue_mrc_train",
                family="klue_mrc_answerable",
                task_type="noul",
                language="ko",
                group_key=normalize(context),
                local_id=guid,
                state=MRC_STATE.format(
                    title=text(record, "title", where), context=context
                ),
                instructions=MRC_INSTRUCTIONS.format(
                    question=text(record, "question", where)
                ),
                options=noul_options("ko"),
                label=0 if impossible else 1,
                render_template="klue_mrc_answerable_v1",
                audit={
                    "is_impossible": impossible,
                    "question_type": record.get("question_type"),
                },
            )
        )
    answerable = sum(row["label"] for row in rows)
    return rows, {
        "input": input_receipt(root, MRC_FILE, len(rows)),
        "train_class_histogram": {
            "answerable": answerable,
            "unanswerable": len(rows) - answerable,
        },
    }


def sts(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pairs = []
    for where, record in read_jsonl(root / STS_FILE):
        labels = record.get("labels")
        if not isinstance(labels, dict):
            raise ValueError(f"{where}: labels must be an object")
        pairs.append(
            RatedPair(
                local_id=ident(record, "guid", where),
                first=text(record, "sentence1", where),
                second=text(record, "sentence2", where),
                mean=rating(labels.get("real-label"), 0, STS_TOP, where),
                stratum=text(record, "source", where),
            )
        )
    rows, report = sts_rows(
        pairs,
        family="klue_sts",
        source="klue_sts_train",
        language="ko",
        texts=STS_TEXT,
        template="klue_sts_similarity",
    )
    report["input"] = input_receipt(root, STS_FILE, len(pairs))
    report["stratum_field"] = "source"
    report["strata"] = dict(
        sorted(collections.Counter(pair.stratum for pair in pairs).items())
    )
    return rows, report
