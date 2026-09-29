"""WiC-TSV (English): is the target word used in the given sense in this sentence?

The test labels are not public, so Development (validation) comes first, then Training.
Line-aligned files: examples (word, token index, sentence), definitions, hypernyms,
labels (T/F). The target word is the group.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from v2.eval.htdev.common import HtCandidate, make, noul, spec

TASK = "wordsense/wic_tsv"
PARTS = (("Development", "dev", "validation"), ("Training", "train", "train"))
QUESTION = noul(
    "The state gives a sentence, a target word that occurs in it, and one sense of that "
    "word (a definition and its hypernyms).",
    "In this sentence the target word is used in the given sense.",
    "In this sentence the target word is used in a different sense.",
)

SPEC = spec(
    key="wic_tsv",
    dataset_id="semantic-web-company/wic-tsv (en)",
    revision="7bddf326add99eb40a2b85a331deffafead2a73f",
    licence="cc-by-4.0",
    evidence="data/en/LICENSE at the pinned commit: CC BY 4.0",
    label_provenance="sense-inventory examples and expert annotation (T/F)",
    tasks=(TASK,),
)


def lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def candidates(root: Path) -> Iterator[HtCandidate]:
    for directory, prefix, split in PARTS:
        base = f"data/en/{directory}/{prefix}"
        examples = lines(root / f"{base}_examples.txt")
        definitions = lines(root / f"{base}_definitions.txt")
        hypernyms = lines(root / f"{base}_hypernyms.txt")
        labels = lines(root / f"{base}_labels.txt")
        if not len(examples) == len(definitions) == len(hypernyms) == len(labels):
            raise ValueError(f"{base}: files are not line-aligned")
        for index, example in enumerate(examples):
            parts = example.split("\t")
            label = labels[index].strip()
            if len(parts) != 3 or label not in ("T", "F"):
                continue
            word, _, sentence = (p.strip() for p in parts)
            state = {
                "sentence": sentence,
                "target_word": word,
                "sense_definition": definitions[index].strip(),
                "hypernyms": ", ".join(
                    h.strip().replace("_", " ")
                    for h in hypernyms[index].split("\t")
                    if h.strip()
                ),
            }
            item = make(
                SPEC,
                TASK,
                f"{split}:{index}",
                word.casefold(),
                split,
                f"{base}_examples.txt",
                index,
                state,
                dict(QUESTION),
                label == "T",
                overlap_texts=[sentence, state["sense_definition"]],
            )
            if item:
                yield item
