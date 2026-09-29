"""XED English (backup for emotion): which emotion does the speaker express?

`en-annotated.tsv`: subtitle line, tab, comma-separated Plutchik labels (1 anger ...
8 trust, README order). Single-label lines only. One released file (`all`); there is no
movie id, so each line is its own group.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from v2.eval.htdev.common import HtCandidate, choice, make, spec

TASK = "emotion/xed"
FILE = "AnnotatedData/en-annotated.tsv"
EMOTIONS = (
    "anger",
    "anticipation",
    "disgust",
    "fear",
    "joy",
    "sadness",
    "surprise",
    "trust",
)
OPTIONS = [(e, e) for e in EMOTIONS]
INSTRUCTIONS = (
    "The state is one line of dialogue from a film. Which emotion does the speaker "
    "mainly express?"
)

SPEC = spec(
    key="xed",
    dataset_id="Helsinki-NLP/XED (English)",
    revision="76666d251f147cf518071bef4f8415dab6b96115",
    licence="cc-by-4.0",
    evidence="README.md at the pinned commit: 'License: Creative Commons Attribution 4.0'",
    label_provenance="human annotators with Plutchik's 8 emotions, multi-label",
    tasks=(TASK,),
)


def candidates(root: Path) -> Iterator[HtCandidate]:
    text = (root / FILE).read_text(encoding="utf-8")
    for index, line in enumerate(text.splitlines()):
        line_text, _, labels = line.rpartition("\t")
        values = [v.strip() for v in labels.split(",") if v.strip()]
        line_text = line_text.strip()
        if len(values) != 1 or values[0] not in {str(i) for i in range(1, 9)}:
            continue
        if not line_text:
            continue
        item_id = f"line{index}"
        item = make(
            SPEC,
            TASK,
            item_id,
            item_id,
            "all",
            FILE,
            index,
            {"text": line_text},
            choice(item_id, INSTRUCTIONS, OPTIONS),
            EMOTIONS[int(values[0]) - 1],
            overlap_texts=[line_text],
        )
        if item:
            yield item
