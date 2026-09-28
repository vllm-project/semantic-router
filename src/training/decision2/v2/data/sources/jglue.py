"""JGLUE v1.3 (ja) projections: A5 JCommonsenseQA and JNLI rows, A6h JSTS rows.

Only the v1.3 TRAIN files are read. JNLI and JSTS reuse the KLUE NLI and STS
projections with Japanese text and group by the image id (the ``yjcaptions_id``
prefix before the first '-'), which never reaches a state.
"""

from __future__ import annotations

import collections
from pathlib import Path
from typing import Any

from v2.data.sources import klue
from v2.data.textnorm import normalize

JCQA_FILE = Path("datasets/jcommonsenseqa-v1.3/train-v1.3.json")
JNLI_FILE = Path("datasets/jnli-v1.3/train-v1.3.json")
JSTS_FILE = Path("datasets/jsts-v1.3/train-v1.3.json")

JCQA_CHOICES = 5
JCQA_INSTRUCTIONS = "質問に対する答えとして最も適切なものを選んでください。"

JNLI_TEXT = {
    "instructions": "前提が正しいとしたとき、前提と仮説の関係として最も適切なものを選んでください。",
    "options": (
        "前提が正しければ、仮説も必ず正しい",
        "前提だけでは、仮説が正しいかどうか判断できない",
        "前提が正しければ、仮説が正しいことはありえない",
    ),
}

JSTS_TEXT = {
    "state": "文1：{first}\n文2：{second}",
    "instructions": (
        "2つの文の意味がどの程度似ているかを評価してください。類似度は0点（まったく関係がない）"
        "から5点（意味が完全に同じ）までで、各段階の点数の範囲は、複数の評価者がつけた点数の"
        "平均に基づいています。"
    ),
    "range": "{phrase}（{low}〜{high}点）",
    "anchors": (
        "2つの文の意味はまったく関係がない",
        "話題は似ているが、内容は異なる",
        "細部の一部は共通しているが、中心となる意味は異なる",
        "中心となる意味はおおむね同じだが、重要な情報が異なるか欠けている",
        "意味はほぼ同じで、違いはささいな細部だけである",
        "2つの文の意味は完全に同じである",
    ),
    "bands": {
        3: (
            "2つの文の意味は異なる",
            "2つの文の意味は部分的に重なる",
            "2つの文の意味はほぼ同じである",
        ),
        4: (
            "2つの文の意味はほとんど関係がない",
            "話題や一部の内容は重なるが、中心となる意味は異なる",
            "中心となる意味は近いが、目立つ違いがある",
            "2つの文の意味はほぼ同じである",
        ),
        5: (
            "2つの文の意味はまったく関係がない",
            "話題は重なるが、意味の大部分は異なる",
            "2つの文の意味は部分的に重なる",
            "中心となる意味はおおむね同じだが、違いがある",
            "2つの文の意味はほぼ同じである",
        ),
    },
}


def jcommonsenseqa(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, classes, duplicates, records = [], collections.Counter(), 0, 0
    for where, record in klue.read_jsonl(root / JCQA_FILE):
        records += 1
        q_id = klue.ident(record, "q_id", where)
        choices = [klue.text(record, f"choice{i}", where) for i in range(JCQA_CHOICES)]
        label = klue.index(record.get("label"), JCQA_CHOICES, where)
        if len({normalize(choice) for choice in choices}) != JCQA_CHOICES:
            duplicates += 1
            continue
        classes[label] += 1
        rows.append(
            klue.choice_row(
                choices,
                label,
                arm=klue.A5,
                source="jglue_jcommonsenseqa_v1.3_train",
                family="jglue_jcommonsenseqa",
                language="ja",
                group_key=q_id,
                local_id=q_id,
                state=klue.text(record, "question", where),
                instructions=JCQA_INSTRUCTIONS,
                render_template="jglue_jcommonsenseqa_question_v1",
                audit={"original_label": label},
            )
        )
    return rows, {
        "input": klue.input_receipt(root, JCQA_FILE, records),
        "train_class_histogram": {
            f"choice{i}": classes[i] for i in range(JCQA_CHOICES)
        },
        "dropped": {"duplicate_choices": duplicates},
    }


def image_id(record: dict[str, Any], where: str) -> str:
    image = klue.text(record, "yjcaptions_id", where).split("-", 1)[0]
    if not image:
        raise ValueError(f"{where}: yjcaptions_id has no image id")
    return image


def jnli(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    items, images = [], []
    for where, record in klue.read_jsonl(root / JNLI_FILE):
        label = record.get("label")
        if label not in klue.NLI_LABELS:
            raise ValueError(f"{where}: unknown JNLI label {label!r}")
        images.append(image_id(record, where))
        items.append(
            (
                klue.ident(record, "sentence_pair_id", where),
                klue.text(record, "sentence1", where),
                klue.text(record, "sentence2", where),
                klue.NLI_LABELS.index(label),
            )
        )
    rows = klue.nli_rows(
        items,
        family="jglue_jnli",
        source="jglue_jnli_v1.3_train",
        language="ja",
        texts=JNLI_TEXT,
        template="jglue_jnli_premise_hypothesis_json_v1",
        keys=images,
    )
    classes = collections.Counter(klue.NLI_LABELS[item[3]] for item in items)
    return rows, {
        "input": klue.input_receipt(root, JNLI_FILE, len(items)),
        "train_class_histogram": {name: classes[name] for name in klue.NLI_LABELS},
        "image_ids": len(set(images)),
        "groups_after_reversed_pair_merge": len({row["group_id"] for row in rows}),
    }


def jsts(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pairs = [
        klue.RatedPair(
            local_id=klue.ident(record, "sentence_pair_id", where),
            first=klue.text(record, "sentence1", where),
            second=klue.text(record, "sentence2", where),
            mean=klue.rating(record.get("label"), 0, klue.STS_TOP, where),
            group=image_id(record, where),
        )
        for where, record in klue.read_jsonl(root / JSTS_FILE)
    ]
    rows, report = klue.sts_rows(
        pairs,
        family="jglue_jsts",
        source="jglue_jsts_v1.3_train",
        language="ja",
        texts=JSTS_TEXT,
        template="jglue_jsts_similarity",
    )
    report["input"] = klue.input_receipt(root, JSTS_FILE, len(pairs))
    report["image_ids"] = len({pair.group for pair in pairs})
    return rows, report
