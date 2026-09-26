"""Gold-free ablation packet and mechanical shortcut audit for authored v8 DEV.

The output is for independent blind review. It is never a release pass.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .authored_v8_pilot import POLICIES, parse_documents, selected

VERSION = "jevarena-authored-v8-editorial-audit/1"


def load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def remove_document(state: str, doc_id: str) -> str:
    blocks = state.split("\n\n")
    matching = [
        i for i, block in enumerate(blocks) if block.startswith(f"DOCUMENT {doc_id} |")
    ]
    if len(matching) != 1:
        raise ValueError("Source deletion must identify exactly one document")
    del blocks[matching[0]]
    return "\n\n".join(blocks)


def narrative_grams(state: str) -> set[tuple[str, ...]]:
    text = " ".join(
        line
        for line in state.splitlines()
        if not line.startswith(
            ("DOCUMENT ", "ATTESTED ", "Current rule:", "Archived rule:")
        )
    )
    words = re.findall(r"[a-z]+|\d+", text.lower())
    return set(zip(*(words[index:] for index in range(5))))


def audit(packet: Path, output: Path, receipt: Path) -> dict[str, Any]:
    if output.exists() or receipt.exists():
        raise FileExistsError("Frozen audit artifacts cannot be overwritten")
    prompts = load(packet / "prompts.jsonl")
    targets = load(packet / "private/targets.jsonl")
    proofs = load(packet / "private/proof_traces.jsonl")
    if len(prompts) != 12 or len(targets) != 12 or len(proofs) != 12:
        raise ValueError("Incomplete authored v8 packet")
    if any(
        p["id"] != t["id"] or p["id"] != r["id"]
        for p, t, r in zip(prompts, targets, proofs)
    ):
        raise ValueError("Row IDs are misaligned")
    ablations = []
    choice_source_positions: Counter[int] = Counter()
    choice_option_positions: Counter[int] = Counter()
    long_lengths = []
    for prompt, target, proof in zip(prompts, targets, proofs):
        policy = POLICIES[proof["policy_id"]]
        if target["kind"] == "choice" and target["answer"]["choice"] != "hold":
            winner = target["answer"]["choice"]
            first_register = proof["visible_facts"][policy.fields[0]]
            criteria = prompt["questions"]["decision"]["criteria"]
            choice_source_positions[list(first_register).index(winner) + 1] += 1
            choice_option_positions[list(criteria).index(winner) + 1] += 1
        if proof["challenge"] != "long_join":
            continue
        long_lengths.append(proof["visible_words"])
        parsed = parse_documents(prompt["state"])
        if set(proof["selected_sources"]) != set(policy.fields):
            raise ValueError("Long item lacks three independent current sources")
        for field, doc_id in proof["selected_sources"].items():
            changed_state = remove_document(prompt["state"], doc_id)
            reduced, _ = selected(parse_documents(changed_state), proof["case_id"])
            if field in reduced:
                raise ValueError("A void or neighbor value replaced an ablated source")
            if len(parsed) != len(parse_documents(changed_state)) + 1:
                raise ValueError("Source ablation removed more than one document")
            ablations.append(
                {
                    "parent_id": prompt["id"],
                    "omitted_field": field,
                    "state": changed_state,
                    "questions": prompt["questions"],
                    "review_instruction": (
                        "Do not give a forced decision. State whether the original "
                        "decision remains logically provable from this reduced packet; "
                        "if not, state what is underdetermined and why. Treat every "
                        "VOID document as void even after its replacement is removed."
                    ),
                }
            )
    if len(ablations) != 9 or len(long_lengths) != 3:
        raise ValueError("Expected three long cases and nine source ablations")
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ablations)
    )
    output.chmod(0o600)
    grams = [narrative_grams(p["state"]) for p in prompts]
    similarities = [
        len(grams[i] & grams[j]) / len(grams[i] | grams[j])
        for i, j in itertools.combinations(range(12), 2)
    ]
    position_gate = (
        sum(choice_source_positions.values()) == 3
        and sum(choice_option_positions.values()) == 3
        and choice_source_positions.get(1, 0) <= 1
        and choice_option_positions.get(1, 0) <= 1
        and len(choice_option_positions) >= 2
    )
    report = {
        "version": VERSION,
        "status": (
            "AUTOMATED_CHECK_ONLY" if position_gate else "BLOCKED_POSITION_SHORTCUT"
        ),
        "release_qualified": False,
        "blind_review_passed": False,
        "prompt_sha256": sha(packet / "prompts.jsonl"),
        "blind_ablation_sha256": sha(output),
        "ablation_rows": len(ablations),
        "long_visible_words": sorted(long_lengths),
        "choice_source_positions_one_based": dict(choice_source_positions),
        "choice_option_positions_one_based": dict(choice_option_positions),
        "position_gate": position_gate,
        "max_narrative_fivegram_jaccard": round(max(similarities), 4),
        "limitation": (
            "Mechanical proof cannot establish genuine long-document reasoning, "
            "source relevance, scenario independence, or absence of semantic cues. "
            "Gold-blind editorial review remains mandatory."
        ),
    }
    receipt.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    receipt.chmod(0o600)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--blind-ablations", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.packet, args.blind_ablations, args.receipt)
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("status", "blind_ablation_sha256", "position_gate")
            }
        )
    )


if __name__ == "__main__":
    main()
