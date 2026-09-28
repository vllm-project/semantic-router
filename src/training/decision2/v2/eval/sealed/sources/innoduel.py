"""Innoduel public sample (NordosoftOy/innoduel-rlhf-real-world-human-preferences-sample).

Idea contests that organisations ran on the Innoduel platform: members answered the
organisation's question with short ideas, and participants voted between two ideas at a
time. The sample holds 90 blocks (one question each) x 3 respondents x 5 votes; the
question and both ideas are human-written in the source language (fi, sv, en, uk).
Machine translations (`*_en`), the corpus win-rate `priority_score` (label-derived) and
the rule-based topic, role and timing columns are not read.

One Choice item per unordered idea pair under a question: votes on the same pair (also
across blocks of one organisation that repeat the question) are merged, and pairs whose
votes disagree or whose two ideas are identical are dropped. The ideas are shown as A / B
in `display_order` of the sorted pair; the organisation's question is the group.
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

from v2.eval.sealed.schema import (
    MAX_INPUT_CHARS,
    Candidate,
    SourceSpec,
    display_order,
    input_chars,
    letters,
)

TASK = "innoduel/preferred_idea"
COLUMNS = (
    "block_id",
    "org_id",
    "matchup_id",
    "question",
    "chosen_answer",
    "rejected_answer",
    "language",
)

SPEC = SourceSpec(
    key="innoduel",
    dataset_id="NordosoftOy/innoduel-rlhf-real-world-human-preferences-sample",
    revision="99f8810bbcaebefa65de68d39bca096bc99b09df",
    licence="cc-by-nc-4.0",
    licence_flag="NC",
    first_release="2026-08-23",
    evidence=(
        "https://huggingface.co/api/datasets/NordosoftOy/"
        "innoduel-rlhf-real-world-human-preferences-sample/commits/main: initial commit "
        "and the single data commit (sample.csv, 1,350 rows) 2026-08-23 (createdAt "
        "2026-08-23T16:20Z), not gated. The full corpus NordosoftOy/innoduel-rlhf "
        "(createdAt 2026-05-28) is manually gated under a commercial licence (its "
        "commits API returns 401), so its rows are not public."
    ),
    label_provenance=(
        "Organic: every row is one real participant's forced-choice vote between two "
        "human-written ideas in an organisation's own Innoduel contest (card: no "
        "synthetic content, no LLM-as-judge). Most pairs carry a single vote."
    ),
    languages=("fi", "sv", "en", "uk"),
    tasks=(TASK,),
    notes=(
        "Votes date from 2019-11 to 2026-03 but stayed private platform data until the "
        "2026-08-23 release, so no date filter applies and date is None. 1,350 votes on "
        "1,304 matchups merge into 1,295 pairs; 27 with disagreeing votes (20 one-one "
        "ties, 1 two-one split, 6 duplicate-idea or repeated-question pairs) and 1 with "
        "identical ideas are dropped: 1,267 kept in 86 question groups, 1,242 of them "
        "single-vote. 16% of kept votes are marked unreliable (< 4 s). Ideas are short "
        "(median about 60 chars). Blocks are curated for language and topic coverage."
    ),
)

QUESTION = {
    "type": "choice",
    "instructions": (
        "The state holds a question that an organisation put to its members on an "
        "idea-voting platform, and two ideas (A and B) that members wrote in answer to "
        "it, in the original language. A participant was shown both ideas side by side "
        "and voted for the one they preferred. Which idea did the participant prefer?"
    ),
    "criteria": {
        "A": "The participant preferred idea A.",
        "B": "The participant preferred idea B.",
    },
}


def position(row: dict[str, str]) -> tuple[int, int]:
    return int(row["block_id"]), int(re.sub(r"\D", "", row["matchup_id"]) or 0)


def candidates(root: Path) -> Iterator[Candidate]:
    with (root / "sample.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = [
            {column: row[column].strip() for column in COLUMNS}
            for row in csv.DictReader(stream)
        ]
    first_block: dict[tuple[str, str], int] = {}
    votes = defaultdict(list)
    for row in rows:
        topic, block = (row["org_id"], row["question"]), position(row)[0]
        first_block[topic] = min(first_block.get(topic, block), block)
        pair = frozenset((row["chosen_answer"], row["rejected_answer"]))
        votes[(*topic, pair)].append(row)
    keys = letters(2)
    for (org, question, pair), cast in sorted(
        votes.items(), key=lambda item: min(map(position, item[1]))
    ):
        winners = {row["chosen_answer"] for row in cast}
        languages = {row["language"] for row in cast}
        if len(pair) != 2 or "" in pair or not question or len(winners) != 1:
            continue
        if len(languages) != 1 or not languages <= set(SPEC.languages):
            continue
        first = min(cast, key=position)
        item_id = f"{first['block_id']}/{first['matchup_id']}"
        ideas = sorted(pair)
        shown = [ideas[i] for i in display_order(item_id, 2)]
        state = {"question": question, "ideas": dict(zip(keys, shown))}
        question_spec = dict(QUESTION)
        if input_chars(state, question_spec) > MAX_INPUT_CHARS:
            continue
        gold = keys[shown.index(winners.pop())]
        yield Candidate(
            source=SPEC.key,
            task=TASK,
            source_item_id=item_id,
            group_id=f"{org}/{first_block[(org, question)]}",
            balance_label=gold,
            language=languages.pop(),
            state=state,
            question=question_spec,
            gold=gold,
            overlap_texts=[question, *shown],
        )
