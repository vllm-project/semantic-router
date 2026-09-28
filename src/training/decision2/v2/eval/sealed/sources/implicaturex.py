"""ImplicatureX (McGill-NLP/ImplicatureX): how likely is an implicature, given the utterance?

271 expert-validated implicature items (scalar, discourse, synthetic and naturally-occurring
conversational). Prolific crowdworkers rated the likelihood of each implicature on a
7-point Likert scale ("absolutely impossible" ... "absolutely certain"), once without and
once with the authors' cancelling utterance. Only `implicatureX.csv` and
`prolific_responses.csv` are read; the control variants (`implicatureX_prior`,
`implicatureBot` = explicit negation, `implicaturePlus`, `implicatureApprox`) have no human
ratings, and the expert plausibility checks are not labels for this question.

Two Score items per implicature: without and with the cancelling utterance, which is
appended to the utterance as in the paper (on a new line when it opens a new speaker turn
such as "A:", otherwise after a space, so line breaks do not mark the condition). Gold is
the mean rating of the (item, condition) rounded half up to the nearest scale point
(level = point - 1). The implicature is the group.
"""

from __future__ import annotations

import csv
import itertools
import re
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

from v2.eval.sealed.schema import MAX_INPUT_CHARS, Candidate, SourceSpec, input_chars

TASK = "implicaturex/likelihood"
POINTS = 7
CONDITIONS = {"False": False, "True": True}
SPEAKER_TURN = re.compile(r"^[A-Z][0-9]?\s*:")

SPEC = SourceSpec(
    key="implicaturex",
    dataset_id="McGill-NLP/ImplicatureX",
    revision="959e5e279611c877224d7573fe66fa9748f59aa4",
    licence="mit",
    licence_flag=None,
    first_release="2026-07-23",
    evidence=(
        "https://huggingface.co/api/datasets/McGill-NLP/ImplicatureX/commits/main: first "
        "commit 2026-07-31, not gated; data first pushed to "
        "github.com/cesare-spinoso/ImplicatureX on 2026-07-23 (initial commit 2026-07-02 "
        "without data); arXiv 2607.25094v1 2026-07-27."
    ),
    label_provenance=(
        "Prolific crowdworkers (native English, CA/US/UK, degree, >= 99% approval) rated "
        "implicature likelihood on a 1-7 Likert scale, 5 per batch; the authors dropped "
        "workers failing >= 3 of 10 attention checks: 2,285 ratings by 76 workers, 3-5 "
        "per item and condition. Items validated by two linguists."
    ),
    languages=("en",),
    tasks=(TASK,),
    notes=(
        "Gold = floor(mean + 0.5) of the 1-7 ratings per (item, condition), level = "
        "point - 1. The condition drives the level (every rated follow-up cancels), so "
        "a cancelling sentence is itself a strong cue. Old text: contexts and triggering "
        "utterances come from Switchboard, PDTB/WSJ and earlier implicature datasets "
        "(Degen 2015 scalar ratings, Circa and others), so labels for the "
        "no-cancellation condition may be partly recoverable; cancelling utterances and "
        "ratings are new. Files carry a canary GUID."
    ),
)

SCALE = "on a scale from 1 (absolutely impossible) to 7 (absolutely certain)."
QUESTION = {
    "type": "score",
    "instructions": (
        "The state holds an utterance, preceded by its context when there is one, and a "
        "possible interpretation of what was meant. Considering the context and the full "
        "utterance, how likely is it that the interpretation is true?"
    ),
    "criteria": [f"{point} {SCALE}" for point in range(1, POINTS + 1)],
}


def read_rows(path: Path) -> list[dict[str, str]]:
    """CSV rows after the leading `#` comment line (canary) the files start with."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        lines = iter(stream)
        first = next(lines, "")
        if not first.startswith("#"):
            lines = itertools.chain([first], lines)
        return list(csv.DictReader(lines))


def rounded_point(values: list[int]) -> int:
    """Mean rounded half up to the nearest scale point, in integer arithmetic."""
    return (2 * sum(values) + len(values)) // (2 * len(values))


def candidates(root: Path) -> Iterator[Candidate]:
    ratings: dict[tuple[str, bool], list[int]] = defaultdict(list)
    for row in read_rows(root / "prolific_responses.csv"):
        condition = CONDITIONS.get(row["contains_cancellation"].strip())
        value = row["likelihood"].strip()
        if condition is None or value not in {str(p) for p in range(1, POINTS + 1)}:
            continue
        ratings[(row["item_id"].strip(), condition)].append(int(value))
    for item in read_rows(root / "implicatureX.csv"):
        item_id = item["id"].strip()
        context = item["context"].strip()
        utterance = item["utterance"].strip()
        interpretation = item["implicature"].strip()
        cancellation = item["cancellation"].strip()
        if not utterance or not interpretation:
            continue
        for condition in (False, True):
            values = ratings.get((item_id, condition))
            if not values or (condition and not cancellation):
                continue
            said = utterance
            if condition:
                joint = "\n" if SPEAKER_TURN.match(cancellation) else " "
                said = f"{utterance}{joint}{cancellation}"
            state = {"context": context} if context else {}
            state["utterance"] = said
            state["interpretation"] = interpretation
            question = dict(QUESTION)
            if input_chars(state, question) > MAX_INPUT_CHARS:
                continue
            gold = rounded_point(values) - 1
            name = "with_cancellation" if condition else "without_cancellation"
            yield Candidate(
                source=SPEC.key,
                task=TASK,
                source_item_id=f"{item_id}:{name}",
                group_id=item_id,
                balance_label=str(gold),
                language="en",
                state=state,
                question=question,
                gold=gold,
                overlap_texts=[
                    text
                    for text in (
                        context,
                        utterance,
                        interpretation,
                        cancellation if condition else "",
                    )
                    if text
                ],
            )
