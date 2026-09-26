"""Aggregate shortcut and source-ablation receipt for a v7 private DEV packet.

The report contains no raw items, labels, or target IDs. It is a diagnostic
gate, never independent editorial approval or release qualification.
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

from .authored_v7_pilot import parse_sources
from .authored_v7_policies import POLICIES

VERSION = "jevarena-authored-v7-shortcut-audit/2"


def read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def grams(state: str) -> set[tuple[str, ...]]:
    narrative = []
    for line in state.splitlines():
        if line.startswith(
            ("Current signed rule:", "Archived rule:", "SIGNED FIELD ", "SOURCE ")
        ):
            continue
        narrative.append(line)
    words = re.findall(r"[a-z]+|\d+", " ".join(narrative).lower())
    return set(zip(*(words[index:] for index in range(5))))


def jaccard(a: set[Any], b: set[Any]) -> float:
    return len(a & b) / len(a | b) if a or b else 0.0


def audit(packet: Path, output: Path, ablations: Path) -> dict[str, Any]:
    if output.exists() or ablations.exists():
        raise FileExistsError("Diagnostic receipts are immutable")
    prompts = read(packet / "prompts.jsonl")
    targets = read(packet / "private/targets.jsonl")
    traces = read(packet / "private/proof_traces.jsonl")
    if len(prompts) != 12 or len(targets) != 12 or len(traces) != 12:
        raise ValueError("Incomplete v7 DEV pilot")
    if any(
        p["id"] != t["id"] or p["id"] != r["id"]
        for p, t, r in zip(prompts, targets, traces)
    ):
        raise ValueError("Prompt, key and proof row identities differ")
    choice_register_position = Counter()
    choice_option_position = Counter()
    score_by_challenge: dict[str, Counter[int]] = {}
    missing_invariant_nondefault = 0
    missing_underdetermined = 0
    long_lengths = []
    ablation_rows = []
    for prompt, target, trace in zip(prompts, targets, traces):
        policy = POLICIES[trace["policy_id"]]
        if target["kind"] == "choice" and target["answer"]["choice"] != "hold":
            winner = target["answer"]["choice"]
            value_register = trace["visible_facts"][policy.fields[0]]
            choice_register_position[str(list(value_register).index(winner))] += 1
            criteria = prompt["questions"]["decision"]["criteria"]
            choice_option_position[str(list(criteria).index(winner))] += 1
        if target["kind"] == "score":
            score_by_challenge.setdefault(trace["challenge"], Counter())[
                target["answer"]["score"]
            ] += 1
        if trace["challenge"] == "missing_source":
            if trace["missing_source_count"] != 0 or len(trace["world_outputs"]) < 2:
                raise ValueError("Missing source was not truly absent")
            if len(set(trace["world_outputs"])) > 1:
                missing_underdetermined += 1
            elif trace["answer"] not in ("hold", False, 0):
                missing_invariant_nondefault += 1
        if trace["challenge"] == "long_join":
            long_lengths.append(trace["visible_words"])
            if set(trace["causal_value_interventions"]) != set(policy.fields):
                raise ValueError("Long case lacks causal proof for all target sources")
            rows = parse_sources(prompt["state"])
            for field, doc_id in trace["selected_sources"].items():
                retained = [row for row in rows if row["doc_id"] != doc_id]
                if len(retained) != len(rows) - 1:
                    raise ValueError("Target source ablation is not singular")
                without_body = prompt["state"]
                # A complete source block, including narrative, is removed.
                source_start = next(
                    line
                    for line in prompt["state"].splitlines()
                    if line.startswith(f"SOURCE {doc_id} |")
                )
                start = without_body.index(source_start)
                next_header = re.search(
                    r"\n\nSOURCE [0-9a-f]{12} \|", without_body[start + 1 :]
                )
                stop = (
                    start + 1 + next_header.start()
                    if next_header
                    else len(without_body)
                )
                ablated_state = (without_body[:start] + without_body[stop:]).strip()
                ablation_rows.append(
                    {
                        "parent_id": prompt["id"],
                        "omitted_field": field,
                        "state": ablated_state,
                        "questions": prompt["questions"],
                        "review_instruction": "Independently decide whether the original decision remains provable from this ablated packet; cite surviving documents or say underdetermined.",
                    }
                )
    g = [grams(prompt["state"]) for prompt in prompts]
    similarities = sorted(
        (jaccard(g[i], g[j]) for i, j in itertools.combinations(range(12), 2)),
        reverse=True,
    )
    reg_gate = (
        sum(choice_register_position.values()) == 3
        and len(choice_register_position) >= 2
        and choice_register_position.get("0", 0) <= 1
    )
    option_gate = (
        sum(choice_option_position.values()) == 3
        and len(choice_option_position) >= 2
        and choice_option_position.get("0", 0) <= 1
    )
    unique_policy_gate = len({r["policy_id"] for r in traces}) == 12
    missing_gate = missing_invariant_nondefault >= 1 and missing_underdetermined >= 2
    score_gate = sorted(
        t["answer"]["score"] for t in targets if t["kind"] == "score"
    ) == [0, 1, 2, 3, 4]
    metadata_gate = (
        len({p["id"] for p in prompts}) == 12
        and all(re.fullmatch(r"[0-9a-f]{16}", p["id"]) for p in prompts)
        and all(set(p) == {"id", "state", "questions"} for p in prompts)
    )
    gate = all(
        (
            reg_gate,
            option_gate,
            unique_policy_gate,
            missing_gate,
            score_gate,
            metadata_gate,
        )
    )
    # Preserve displayed criterion order; sorted JSON would silently erase the
    # answer-independent permutation in this gold-free reviewer packet.
    ablations.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ablation_rows)
    )
    ablations.chmod(0o600)
    result = {
        "version": VERSION,
        "status": "AUTOMATED_SHORTCUT_CHECK_ONLY" if gate else "BLOCKED",
        "release_qualified": False,
        "blind_review_passed": False,
        "prompts_sha256": hashlib.sha256(
            (packet / "prompts.jsonl").read_bytes()
        ).hexdigest(),
        "choice_source_register_position": dict(choice_register_position),
        "choice_output_option_position": dict(choice_option_position),
        "source_position_gate": reg_gate,
        "option_position_gate": option_gate,
        "unique_policy_gate": unique_policy_gate,
        "missing_invariant_nondefault": missing_invariant_nondefault,
        "missing_underdetermined": missing_underdetermined,
        "missing_gate": missing_gate,
        "score_by_challenge": {k: dict(v) for k, v in score_by_challenge.items()},
        "score_full_scale_gate": score_gate,
        "long_visible_words": sorted(long_lengths),
        "long_ablation_rows": len(ablation_rows),
        "blind_ablation_sha256": hashlib.sha256(ablations.read_bytes()).hexdigest(),
        "max_narrative_fivegram_jaccard": round(similarities[0], 4),
        "metadata_gate": metadata_gate,
        "limitation": "Automated checks do not establish causal source necessity, narrative neutrality, independent scenario diversity, or long-context validity; these require gold-blind review.",
    }
    output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    output.chmod(0o600)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blind-ablations", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.packet, args.output, args.blind_ablations)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "prompts_sha256",
                    "source_position_gate",
                    "option_position_gate",
                    "missing_gate",
                    "score_full_scale_gate",
                    "blind_ablation_sha256",
                )
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
