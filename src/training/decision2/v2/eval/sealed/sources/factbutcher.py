"""FactButcher Russian fact-checking benchmark (teplitsa-soc-tech/factbutcher-benchmark).

423 Russian claims with a human-reviewed verdict TRUE / FALSE / MIXED in two parts: 274
claims derived from requests to the FactButcher Telegram bot (LLM-extracted and edited,
checked with web search, adjudicated, final human review) and 149 claims adapted from
Provereno.Media fact-checks (published verdict mapped to the vocabulary, independently
reviewed). Only claims whose `reference_date` (the date the verdict refers to; for
Provereno.Media rows it equals the article date in `source_url`, which must also be on
or after the cutoff) is on or after the cutoff qualify; undated claims are dropped.
Claims with more than one acceptable verdict are dropped, since the publisher contests
their single gold. Source name, URL and licence columns are not placed in the state.

One Choice item per claim; the Provereno.Media article (else the claim) is the group.
"""

from __future__ import annotations

import json
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

TASK = "factbutcher/verdict"
VERDICTS = {"TRUE": "accurate", "FALSE": "inaccurate", "MIXED": "mixed"}
DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
URL_DATE = re.compile(r"/(\d{4})/(\d{2})/(\d{2})/")

SPEC = SourceSpec(
    key="factbutcher",
    dataset_id="teplitsa-soc-tech/factbutcher-benchmark",
    revision="211fb80c7ebb67f85e3bddaa8511938b48d1c2ea",
    licence="cc-by-4.0",
    licence_flag=None,
    first_release="2026-08-06",
    evidence=(
        "https://huggingface.co/api/datasets/teplitsa-soc-tech/factbutcher-benchmark/"
        "commits/main: oldest commit 'Initial FactButcher benchmark dataset release' "
        "2026-08-06 (synced from github.com/Teplitsa/factbutcher-benchmark, same first "
        "commit), HF createdAt 2026-08-09, not gated; release_config publication_date "
        "2026-08-10; CHANGELOG 1.0.0 (2026-07-28) says external publication pending. "
        "alexbadin/factbutcher-benchmark redirects to this repo."
    ),
    label_provenance=(
        "Human-reviewed (METHODOLOGY.md): Human Benchmark claims were checked "
        "independently with web search, disagreements adjudicated, wording and labels "
        "given final human review; Provereno.Media rows carry the professional "
        "fact-checkers' published verdict, mapped to TRUE/FALSE/MIXED and independently "
        "reviewed. There is no per-row review flag: every row is human-reviewed."
    ),
    languages=("ru",),
    tasks=(TASK,),
    notes=(
        "reference_date >= 2026-06-01 keeps 40 of 423 (Provereno.Media 28, Human "
        "Benchmark 12; the 100 undated Human Benchmark claims are dropped); 8 of them have "
        "several acceptable verdicts and are dropped, leaving 32. All 4 accurate claims "
        "are Human Benchmark rows and all 5 mixed ones Provereno.Media rows. "
        "Provereno.Media verdicts are public on provereno.media since June-July 2026, "
        "and claims about post-cutoff events cannot be checked from model knowledge "
        "alone. Provereno.Media claim_ids are article slugs: never show source ids to a "
        "model."
    ),
)

QUESTION = {
    "type": "choice",
    "instructions": (
        "The state is a factual claim written in Russian and a reference date. A "
        "fact-checker assessed the claim against reliable evidence as of that date. "
        "Which verdict fits the claim?"
    ),
    "criteria": {
        "accurate": (
            "Accurate: reliable evidence supports the central factual assertion of the "
            "claim."
        ),
        "inaccurate": (
            "Inaccurate: reliable evidence contradicts the central factual assertion of "
            "the claim."
        ),
        "mixed": (
            "Mixed: substantial parts of the claim have different truth values, or "
            "reliable evidence conflicts on its central point."
        ),
    },
}


def candidates(root: Path) -> Iterator[Candidate]:
    path = root / "data" / "factbutcher_benchmark_v1.jsonl"
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    for row in rows:
        verdict, claim = row["gold_verdict"], row["claim"]
        reference = row["reference_date"] or ""
        if verdict not in VERDICTS or not claim.strip() or not DATE.match(reference):
            continue
        if set(row["acceptable_verdicts"]) != {verdict}:
            continue
        url = row["source_url"] or ""
        published = URL_DATE.search(url)
        date = min(reference, "-".join(published.groups())) if published else reference
        if date < CUTOFF:
            continue
        state = {"claim": claim, "reference_date": reference}
        question = dict(QUESTION)
        if input_chars(state, question) > MAX_INPUT_CHARS:
            continue
        yield Candidate(
            source=SPEC.key,
            task=TASK,
            source_item_id=row["claim_id"],
            group_id=url or row["claim_id"],
            balance_label=VERDICTS[verdict],
            language="ru",
            state=state,
            question=question,
            gold=VERDICTS[verdict],
            overlap_texts=[claim],
            date=date,
        )
