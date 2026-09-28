"""GAPA (alisa-yingjia-wan/gapa): how likely is someone to say a person has an attribute?

Prolific raters answered "How likely is it for someone to say that a <person term> has
<attribute>?" on a 1-7 Likert scale for the person terms woman, man and nonbinary person;
each rater saw an attribute with one person term only. Only `clean/human.csv` is read: its
50 attributes were written by study participants describing face images. The `llm`
(LLM-generated) and `novel` (extracted from novels by an LLM pipeline) splits and the
`raw` config are not read. Raters who passed fewer than 4 of their 5 attention checks (the
paper's inclusion rule) and attention-check trials are dropped.

One Score item per (attribute, person term); gold is the mean rating rounded half up to
the nearest scale point (level = point - 1). The attribute is the group.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

from v2.eval.sealed.schema import MAX_INPUT_CHARS, Candidate, SourceSpec, input_chars

TASK = "gapa/association"
SPLIT = "clean/human.csv"
POINTS = 7
MIN_CHECKS_PASSED = 4
PERSON_TERMS = ("woman", "man", "nonbinary person")

SPEC = SourceSpec(
    key="gapa",
    dataset_id="alisa-yingjia-wan/gapa",
    revision="cecb0381a965686b3b2fb6eec479127b08901b1b",
    licence="mit",
    licence_flag=None,
    first_release="2026-09-14",
    evidence=(
        "https://huggingface.co/api/datasets/alisa-yingjia-wan/gapa/commits/main: first "
        "commit 2026-09-16, not gated; arXiv 2609.16366v1 2026-09-14 (COLM 2026; Fig. 1 "
        "shows an excerpt of ratings); code repo github.com/Yingjia-Wan/GAPA first commit "
        "2026-09-17."
    ),
    label_provenance=(
        "Prolific raters (US, native English) gave 1-7 Likert ratings, each seeing an "
        "attribute with one person term; 5 attention checks per rater. Human split: "
        "2,300 ratings by 46 raters, 1,900 by the 38 raters kept here."
    ),
    languages=("en",),
    tasks=(TASK,),
    notes=(
        "Gold = floor(mean + 0.5) of the 1-7 ratings per (attribute, person term), "
        "level = point - 1; raters with n_attchecks_passed < 4 dropped. Attributes are "
        "short phrases; 26 of the 48 described images were GPT-4o-generated, the "
        "descriptions are human-written. Subjective task; ratings for nonbinary person "
        "stay mid-scale."
    ),
)

SCALE = "on a scale from 1 (not at all likely) to 7 (extremely likely)."
QUESTION = {
    "type": "score",
    "instructions": (
        "The state gives a person term and a physical attribute. How likely is it for "
        "someone to say that a person described by the person term has the physical "
        "attribute?"
    ),
    "criteria": [f"{point} {SCALE}" for point in range(1, POINTS + 1)],
}


def number(text: str) -> float | None:
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def rounded_point(values: list[int]) -> int:
    """Mean rounded half up to the nearest scale point, in integer arithmetic."""
    return (2 * sum(values) + len(values)) // (2 * len(values))


def candidates(root: Path) -> Iterator[Candidate]:
    ratings: dict[tuple[str, str], list[int]] = defaultdict(list)
    with (root / SPLIT).open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["attention_check"].strip().lower() != "false":
                continue
            passed = number(row["n_attchecks_passed"])
            if passed is None or passed < MIN_CHECKS_PASSED:
                continue
            attribute, term = row["attribute"].strip(), row["person_term"].strip()
            value = number(row["rating"])
            if not attribute or term not in PERSON_TERMS:
                continue
            if value is None or value != int(value) or not 1 <= value <= POINTS:
                continue
            ratings[(attribute, term)].append(int(value))
    for (attribute, term), values in sorted(ratings.items()):
        state = {"person_term": term, "physical_attribute": attribute}
        question = dict(QUESTION)
        if input_chars(state, question) > MAX_INPUT_CHARS:
            continue
        gold = rounded_point(values) - 1
        yield Candidate(
            source=SPEC.key,
            task=TASK,
            source_item_id=f"{attribute}|{term}",
            group_id=attribute,
            balance_label=str(gold),
            language="en",
            state=state,
            question=question,
            gold=gold,
            overlap_texts=[attribute],
        )
