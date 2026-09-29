"""BRIGHTER (English): which emotion does the speaker mainly express?

Items annotated with exactly one of anger / fear / joy / sadness / surprise, or with none
(-> no clear emotion); multi-emotion items are dropped. English has no disgust label.
The item id is the group.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from v2.eval.htdev.common import HtCandidate, choice, make, read_parquet, spec

TASK = "emotion/brighter"
EMOTIONS = ("anger", "fear", "joy", "sadness", "surprise")
FILES = (
    ("eng/test-00000-of-00001.parquet", "test"),
    ("eng/dev-00000-of-00001.parquet", "validation"),
    ("eng/train-00000-of-00001.parquet", "train"),
)
OPTIONS = [(e, e) for e in EMOTIONS] + [("no_clear_emotion", "no clear emotion")]
INSTRUCTIONS = (
    "The state is a short text. Which emotion does the speaker mainly express?"
)

SPEC = spec(
    key="brighter",
    dataset_id="brighter-dataset/BRIGHTER-emotion-categories (eng)",
    revision="419566ac8f46f951b51584be36c13e5363008144",
    licence="cc-by-4.0",
    evidence="README.md at the pinned revision: licence cc-by-4.0 ('This dataset is "
    "licensed under CC-BY 4.0')",
    label_provenance="at least 5 crowd annotators per item, perceived emotions, "
    "multi-label, aggregated",
    tasks=(TASK,),
)


def candidates(root: Path) -> Iterator[HtCandidate]:
    for relative, split in FILES:
        rows = read_parquet(root / relative, ["id", "text", *EMOTIONS])
        for index, row in enumerate(rows):
            values = [row[e] for e in EMOTIONS]
            text = str(row["text"] or "").strip()
            if any(v not in (0, 1) for v in values) or not text:
                continue
            marked = [e for e, v in zip(EMOTIONS, values) if v == 1]
            if len(marked) > 1:
                continue
            gold = marked[0] if marked else "no_clear_emotion"
            item_id = str(row["id"])
            item = make(
                SPEC,
                TASK,
                item_id,
                item_id,
                split,
                relative,
                index,
                {"text": text},
                choice(item_id, INSTRUCTIONS, OPTIONS),
                gold,
                overlap_texts=[text],
            )
            if item:
                yield item
