"""WB Review Dataset (Hplss/wb-review-dataset): which star rating did the reviewer give?

Russian product reviews collected from the public review API of Wildberries, a Russian
online marketplace, each with the 1-5 star rating its author gave. Only reviews dated on
or after the cutoff qualify (`date` is the review's own date; the snapshot ends on
2026-07-25). The state is the review as the customer wrote it: the main text plus the
separate pros / cons fields when filled in. Category columns are not used.

One Score item per review (1 star < ... < 5 stars); the product is the group.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Iterator
from pathlib import Path

from v2.eval.sealed.schema import (
    CUTOFF,
    MAX_INPUT_CHARS,
    Candidate,
    SourceSpec,
    input_chars,
)

TASK = "wb_reviews/star_rating"
FIELDS = (("text", "review"), ("pros", "pros"), ("cons", "cons"))
RATINGS = ("1", "2", "3", "4", "5")
DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

SPEC = SourceSpec(
    key="wb_reviews",
    dataset_id="Hplss/wb-review-dataset",
    revision="c207a9df3e78b0fbf22ece02dcafdd84481165b1",
    licence="cc-by-nc-sa-4.0",
    licence_flag="NC-SA",
    first_release="2026-07-30",
    evidence=(
        "https://huggingface.co/api/datasets/Hplss/wb-review-dataset/commits/main: "
        "initial commit and filtered.csv upload 2026-07-30 (createdAt "
        "2026-07-30T10:00Z), 9 commits, not gated. The GitHub collection repo "
        "(ProphetSunboy/wb-review-dataset) starts 2026-07-28; no mirror can predate the "
        "2026-07-25 collection end, so every row used is first public after the cutoff."
    ),
    label_provenance=(
        "Organic: the 1-5 star rating each Wildberries customer gave with their own "
        "review ('as given by the actual reviewer', card), collected from the public "
        "review API; no inferred, crowd or model labels."
    ),
    languages=("ru",),
    tasks=(TASK,),
    notes=(
        "Only rows dated >= 2026-06-01 (14,200 of 27,248; 2026-06 3,726, 2026-07 "
        "10,474). Each review text was public on Wildberries from its date, next to its "
        "rating. Ratings skew to 5 stars (balance by level). group_id = product_id "
        "(pseudonymous; 4,710 products). Card filters: 20-300 words, >= 70% Cyrillic, "
        "exact-text dedup, phone numbers and emails redacted. NC-SA licence."
    ),
)

QUESTION = {
    "type": "score",
    "instructions": (
        "The state is a customer's review of a product bought on Wildberries, a Russian "
        "online marketplace, written by the customer in Russian: the review text and, "
        "when the customer filled them in, separate pros and cons fields. With every "
        "review the customer rates the product from 1 to 5 stars. Which star rating did "
        "this customer give?"
    ),
    "criteria": [
        "1 star out of 5 (the lowest rating).",
        "2 stars out of 5.",
        "3 stars out of 5.",
        "4 stars out of 5.",
        "5 stars out of 5 (the highest rating).",
    ],
}


def candidates(root: Path) -> Iterator[Candidate]:
    with (root / "filtered.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        date, rating = row["date"].strip(), row["rating"].strip()
        if not DATE.match(date) or date < CUTOFF or rating not in RATINGS:
            continue
        state = {name: row[column] for column, name in FIELDS if row[column].strip()}
        if not state:
            continue
        question = dict(QUESTION)
        if input_chars(state, question) > MAX_INPUT_CHARS:
            continue
        gold = RATINGS.index(rating)
        yield Candidate(
            source=SPEC.key,
            task=TASK,
            source_item_id=row["review_id"],
            group_id=row["product_id"],
            balance_label=str(gold),
            language="ru",
            state=state,
            question=question,
            gold=gold,
            overlap_texts=list(state.values()),
            date=date,
        )
