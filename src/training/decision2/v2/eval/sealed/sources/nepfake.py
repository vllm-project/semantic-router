"""NepFakeV2 (Nandan007/NepFakeV2): did the fact-checkers rate the claim false or misleading?

Nepali fact-check articles scraped weekly from NepalFactCheck and TechPana. `claim_text`
is the article headline; `verdict_label_text` (REAL / FALSE_MISLEADING / UNVERIFIED) is
the published verdict read from the site's verdict badge, with keyword fallbacks on the
headline or body (`pipeline/label_mapper.py` in the GitHub repo). Only NepalFactCheck
rows are used: their date comes from the article's published-time tag and matches the
/YYYY/MM/ of the URL, whereas TechPana dates are the first Bikram Sambat date on the page
(148 of 274 rows share the scrape-day date 2026-09-11, 41 contradict the URL year).
UNVERIFIED is dropped, and so is every headline containing verdict wording (a fixed list
from the source's verdict vocabulary), since the headline would state the label. The
article body (`evidence_text`) contains the verdict and is not used.

One Noul item per article (true = rated false or misleading); the article is the group.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Iterator
from pathlib import Path

from v2.eval.sealed.schema import (
    CUTOFF,
    MAX_INPUT_CHARS,
    Candidate,
    SourceSpec,
    input_chars,
)

TASK = "nepfake/false_or_misleading"
SOURCE = "nepalfactcheck"
SNAPSHOT_DATE = "2026-09-28"
VERDICTS = {"FALSE_MISLEADING": True, "REAL": False}
DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
URL = re.compile(r"^https?://(?:www\.)?nepalfactcheck\.org/(\d{4})/(\d{2})/")
DEVANAGARI = re.compile(r"[\u0900-\u097F]")
SPLIT = re.compile(
    r"[\s\u0964\u0965,.!?:;'\"()\[\]{}\u2018\u2019\u201c\u201d\-\u2013\u2014/|]+"
)
VERDICT_STEMS = (
    "मिथ्या",
    "भ्रामक",
    "भ्रम",
    "अपुष्ट",
    "झुट",
    "झूट",
    "गलत",
    "असत्य",
    "फेक",
    "नक्कली",
)
NOT_VERDICT_STEMS = ("भ्रमण",)
VERDICT_WORDS = frozenset({"सही", "सत्य", "साँचो"})
LATIN_VERDICT = re.compile(r"\b(fake|false|misleading|true)\b", re.IGNORECASE)

SPEC = SourceSpec(
    key="nepfake",
    dataset_id="Nandan007/NepFakeV2",
    revision="903ad9f585a63bd0c771e26a2f1c21525c2c8d24",
    licence="cc-by-4.0",
    licence_flag=None,
    first_release="2026-09-12",
    evidence=(
        "https://huggingface.co/api/datasets/Nandan007/NepFakeV2/commits/main: initial "
        "commit and first data 2026-09-12 (createdAt 2026-09-12T13:59Z), weekly "
        "auto-updates, pinned to the 2026-09-28 update, not gated. GitHub pipeline repo "
        "Nandansingh007/NepFakeV2 first commit 2026-08-25, first scrape files 2026-09-12."
    ),
    label_provenance=(
        "Organic: the verdict published by NepalFactCheck (badge सही / भ्रामक / मिथ्या / "
        "अपुष्ट सूचना) mapped by a fixed keyword table; when the badge is missing the "
        "scraper falls back to verdict keywords in the headline, conclusion or body, "
        "which is not identifiable per row. label_basis = fact_checker_verdict for all "
        "rows; no model labels."
    ),
    languages=("ne",),
    tasks=(TASK,),
    notes=(
        "Filters: NepalFactCheck only (673 of 947), date >= 2026-06-01 and consistent "
        "with the URL month (73: 66 FALSE_MISLEADING, 7 REAL, 0 UNVERIFIED), no verdict "
        "wording in the headline (40: 33 / 7). Headlines are the fact-checkers' own "
        "framing, not atomic claims; 6 of the 7 REAL headlines are questions (3 of 33 "
        "FALSE_MISLEADING). Verdicts are public on nepalfactcheck.org. REAL is the "
        "binding class."
    ),
)

QUESTION = {
    "type": "noul",
    "instructions": (
        "The state is the headline of an article by a Nepali fact-checking organisation, "
        "written in Nepali, and the article's publication date. The article checks a "
        "claim that circulated in Nepal. Did the fact-checkers rate the claim false or "
        "misleading?"
    ),
    "criteria": {
        "true": "Yes: the fact-checkers rated the claim false or misleading.",
        "false": "No: the fact-checkers rated the claim true (confirmed as factual).",
    },
}


def states_verdict(headline: str) -> bool:
    text = unicodedata.normalize("NFC", headline).replace("\u200c", "")
    text = text.replace("\u200d", "")
    tokens = [token for token in SPLIT.split(text) if token]
    return bool(LATIN_VERDICT.search(text)) or any(
        token in VERDICT_WORDS
        or (token.startswith(VERDICT_STEMS) and not token.startswith(NOT_VERDICT_STEMS))
        for token in tokens
    )


def candidates(root: Path) -> Iterator[Candidate]:
    with (root / "data" / "nepfakev2.json").open(encoding="utf-8") as stream:
        rows = json.load(stream)
    for row in rows:
        headline, date = row["claim_text"], row["date_published"]
        month = URL.match(row["source_url"])
        if row["source_name"] != SOURCE or not month or not DATE.match(date):
            continue
        if date[:7] != f"{month[1]}-{month[2]}" or not CUTOFF <= date <= SNAPSHOT_DATE:
            continue
        verdict = row["verdict_label_text"]
        if verdict not in VERDICTS or not DEVANAGARI.search(headline):
            continue
        if states_verdict(headline):
            continue
        state = {"headline": headline, "published": date}
        question = dict(QUESTION)
        if input_chars(state, question) > MAX_INPUT_CHARS:
            continue
        gold = VERDICTS[verdict]
        yield Candidate(
            source=SPEC.key,
            task=TASK,
            source_item_id=row["example_id"],
            group_id=row["source_url"],
            balance_label=str(gold),
            language="ne",
            state=state,
            question=question,
            gold=gold,
            overlap_texts=[headline],
            date=date,
        )
