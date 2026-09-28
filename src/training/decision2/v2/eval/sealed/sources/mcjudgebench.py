"""MCJudgeBench (jaelly/MCJudgeBench): does a response satisfy one of its constraints?

Each instance is an instruction (originating from ComplexBench, InFoBench, TRUEBench or
WildIFEval), one fixed model response (Qwen3-4B-Instruct) and a list of constraints, each
with a human gold label yes / partial / no. Only the 78 `augmentation` instances are used:
the 141 `paper` instances belong to arXiv 2605.03858, public on 2026-05-05, before the
cutoff. Perturbed responses and constraint-type tags (LLM-proposed) are not read.

One Score item per constraint (no < partial < yes); the instance is the group.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from v2.eval.sealed.schema import MAX_INPUT_CHARS, Candidate, SourceSpec, input_chars

TASK = "mcjudgebench/constraint"
SUBSET = "augmentation"
LEVELS = ("no", "partial", "yes")

SPEC = SourceSpec(
    key="mcjudgebench",
    dataset_id="jaelly/MCJudgeBench",
    revision="c64832efae480d3f1ed97ac3e877bc2162d9490d",
    licence="apache-2.0",
    licence_flag=None,
    first_release="2026-09-06",
    evidence=(
        "https://huggingface.co/api/datasets/jaelly/MCJudgeBench/commits/main: first "
        "commit 2026-09-06 (createdAt 2026-09-06T13:59Z), 3 commits, not gated. The paper "
        "subset is excluded: arXiv 2605.03858v1 (2026-05-05; GEM 2026, "
        "aclanthology 2026.gem-main.23) describes it; the augmentation subset is new in "
        "this HF release (card)."
    ),
    label_provenance=(
        "Human annotators label every constraint yes / partial / no for the fixed "
        "candidate response (paper App. C.1; Fleiss' kappa 0.88 on a shared 10% subset "
        "of the paper instances); model outputs are never used as labels."
    ),
    languages=("en",),
    tasks=(TASK,),
    notes=(
        "Only release_subset == 'augmentation' (78 instances, 369 constraints). Old text: "
        "instructions come from public benchmarks (ComplexBench, InFoBench, TRUEBench, "
        "WildIFEval); responses are model-generated; some TRUEBench responses are partly "
        "Korean. group_id = instance_id. Instructions and response kept verbatim."
    ),
)

QUESTION = {
    "type": "score",
    "instructions": (
        "The state holds an instruction given to an AI assistant (with any supplementary "
        "input), the assistant's response, and one requirement the response is checked "
        "against. Judge only this requirement, using only the instruction, the response "
        "and the requirement text: how well does the response satisfy it? Judge the "
        "response exactly as shown, even if it ends abruptly. Do not assume missing "
        "information, and do not credit overall fluency or quality when the "
        "requirement itself is not met."
    ),
    "criteria": [
        "Not satisfied: the response clearly violates or omits the requirement.",
        "Partially satisfied: there is clear evidence that the requirement is met in "
        "part but not in full (for example some but not all of what it asks for, or "
        "it holds for some parts of the response and not others).",
        "Fully satisfied: the response clearly meets the requirement.",
    ],
}


def candidates(root: Path) -> Iterator[Candidate]:
    with (root / "data" / "test.jsonl").open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    for row in rows:
        if row["release_subset"] != SUBSET:
            continue
        instruction, extra = row["instruction"], row["input"]
        response = row["candidate_response"]
        if not instruction.strip() or not response.strip():
            continue
        for constraint in row["constraints"]:
            label = constraint["gold_label"]
            requirement = constraint["constraint_text"].strip()
            if label not in LEVELS or not requirement:
                continue
            state = {"instruction": instruction}
            if extra.strip():
                state["input"] = extra
            state["response"] = response
            state["requirement"] = requirement
            question = dict(QUESTION)
            if input_chars(state, question) > MAX_INPUT_CHARS:
                continue
            gold = LEVELS.index(label)
            yield Candidate(
                source=SPEC.key,
                task=TASK,
                source_item_id=f"{row['instance_id']}#{constraint['constraint_id']}",
                group_id=row["instance_id"],
                balance_label=str(gold),
                language="en",
                state=state,
                question=question,
                gold=gold,
                overlap_texts=[
                    text
                    for text in (instruction, extra, response, requirement)
                    if text.strip()
                ],
            )
