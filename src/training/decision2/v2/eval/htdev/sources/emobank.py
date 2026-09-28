"""EmoBank (second backup for emotion): how positive or negative is the sentence's feeling?

`emobank.csv` (the release's own train/dev/test column). Gold: mean valence (1-5)
rounded to the nearest of five levels (cuts 1.5 / 2.5 / 3.5 / 4.5). The source
document (`meta.tsv`) is the group.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from v2.eval.htdev.common import HtCandidate, level, make, read_csv, score, spec

TASK = "emotion/emobank"
FILE = "corpus/emobank.csv"
META = "corpus/meta.tsv"
SPLITS = {"test": "test", "dev": "validation", "train": "train"}
CUTS = (1.5, 2.5, 3.5, 4.5)
QUESTION = score(
    "The state is one sentence. How positive or negative is the feeling it expresses?",
    [
        "very negative",
        "somewhat negative",
        "neutral",
        "somewhat positive",
        "very positive",
    ],
)

SPEC = spec(
    key="emobank",
    dataset_id="JULIELab/EmoBank",
    revision="248ce2a43e165a66d31aeaed83cff9641d6654e0",
    licence="cc-by-sa-4.0",
    evidence="README.md at the pinned commit: 'licensed under CC-BY-SA 4.0'",
    label_provenance="5 crowd raters per sentence, valence 1-5 (averaged)",
    tasks=(TASK,),
)


def candidates(root: Path) -> Iterator[HtCandidate]:
    documents = {row["id"]: row["document"] for row in read_csv(root / META, "\t")}
    for index, row in enumerate(read_csv(root / FILE)):
        split = SPLITS.get(row["split"].strip())
        sentence = row["text"].strip()
        try:
            value = float(row["V"])
        except ValueError:
            continue
        if split is None or not sentence:
            continue
        item = make(
            SPEC,
            TASK,
            row["id"],
            documents.get(row["id"], row["id"]),
            split,
            FILE,
            index,
            {"sentence": sentence},
            dict(QUESTION),
            level(value, CUTS),
            overlap_texts=[sentence],
        )
        if item:
            yield item
