"""Measuring Hate Speech: how hateful is this comment toward the group it mentions?

One row per annotation; one item per comment (its first row). Gold: the per-comment
`hate_speech_score` (a comment-level Rasch estimate) < -1 supportive or counter-speech,
-1..0.5 neutral or ambiguous, > 0.5 hateful (dataset card). Facet ratings, targets and
annotator fields are never read. Single split (`all`); the comment is the group.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from v2.eval.htdev.common import HtCandidate, make, read_parquet, score, spec

TASK = "hate/mhs"
FILE = "measuring-hate-speech.parquet"
QUESTION = score(
    "The state is a social-media comment. How hateful is this comment toward the group "
    "it mentions?",
    ["supportive or counter-speech", "neutral or ambiguous", "hateful"],
)

SPEC = spec(
    key="mhs",
    dataset_id="ucberkeley-dlab/measuring-hate-speech",
    revision="5468f6e118396646b02a2f691e771f6b6d9502ea",
    licence="cc-by-4.0",
    evidence="README.md at the pinned revision: licence cc-by-4.0",
    label_provenance="crowd ratings on 10 survey items aggregated per comment with "
    "faceted Rasch measurement",
    tasks=(TASK,),
)


def hate_level(value: float) -> int:
    if value < -1:
        return 0
    return 1 if value <= 0.5 else 2


def candidates(root: Path) -> Iterator[HtCandidate]:
    seen: set[str] = set()
    rows = read_parquet(root / FILE, ["comment_id", "text", "hate_speech_score"])
    for index, row in enumerate(rows):
        comment = str(row["comment_id"])
        text = str(row["text"] or "").strip()
        if comment in seen or row["hate_speech_score"] is None or not text:
            continue
        seen.add(comment)
        item = make(
            SPEC,
            TASK,
            comment,
            comment,
            "all",
            FILE,
            index,
            {"comment": text},
            dict(QUESTION),
            hate_level(float(row["hate_speech_score"])),
            overlap_texts=[text],
        )
        if item:
            yield item
