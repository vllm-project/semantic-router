"""Private aggregate shortcut audit for a frozen authored v6 DEV packet.

The audit reads private keys locally with the gold-free prompts. It writes
only aggregate metadata into a private report and never emits individual
items, labels, IDs, or text. With eighteen items, statistical non-detection
cannot establish release readiness.
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

from .authored_v5_dossier import _visible_documents, compact
from .authored_v6_policies import POLICIES

VERSION = "jevarena-authored-v6-shortcut-audit/2"


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _grams(state: str, n: int = 5) -> set[tuple[str, ...]]:
    selected = []
    for line in state.splitlines():
        if line.startswith(("Current signed rule:", "Archived rule:", "Record ")):
            continue
        selected.append(line)
    tokens = re.findall(r"[a-z]+|\d+", " ".join(selected).lower())
    return set(zip(*(tokens[index:] for index in range(n))))


def _jaccard(left: set[Any], right: set[Any]) -> float:
    return len(left & right) / len(left | right) if left or right else 0.0


def audit(packet: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    prompts_path = packet / "prompts.jsonl"
    prompt_bytes = prompts_path.read_bytes()
    prompts = _read(prompts_path)
    targets = _read(packet / "private/targets.jsonl")
    traces = _read(packet / "private/proof_traces.jsonl")
    if not (len(prompts) == len(targets) == len(traces) == 18):
        raise ValueError("Frozen v6 pilot is incomplete")
    if any(
        p["id"] != t["id"] or p["id"] != r["id"]
        for p, t, r in zip(prompts, targets, traces)
    ):
        raise ValueError("Gold-free prompts, private keys, and proof are misaligned")
    prompt_ids = [item["id"] for item in prompts]
    metadata_structure_ok = (
        len(set(prompt_ids)) == 18
        and all(re.fullmatch(r"[0-9a-f]{16}", item_id) for item_id in prompt_ids)
        and all(
            set(item) == {"id", "state", "questions"}
            and set(item["questions"]) == {"decision"}
            for item in prompts
        )
    )
    by_six = [
        dict(Counter(trace["challenge"] for trace in traces[start : start + 6]))
        for start in (0, 6, 12)
    ]
    style_by_challenge = {
        challenge: dict(
            Counter(
                trace["style"] for trace in traces if trace["challenge"] == challenge
            )
        )
        for challenge in ("rule_precedence", "partial_evidence", "long_dossier")
    }
    noul_by_style = {
        style: dict(
            Counter(
                str(target["answer"]["noul"])
                for target, trace in zip(targets, traces)
                if target["kind"] == "noul" and trace["style"] == style
            )
        )
        for style in ("prose", "bullets", "table")
    }
    choice_position = Counter()
    resolved_choice_position = Counter()
    score_by_challenge = {}
    for prompt, target, trace in zip(prompts, targets, traces):
        if target["kind"] == "choice":
            options = list(prompt["questions"]["decision"]["criteria"])
            answer_position = str(options.index(target["answer"]["choice"]))
            choice_position[answer_position] += 1
            if target["answer"]["choice"] != "hold":
                resolved_choice_position[answer_position] += 1
    for challenge in ("rule_precedence", "partial_evidence", "long_dossier"):
        score_by_challenge[challenge] = dict(
            Counter(
                str(target["answer"]["score"])
                for target, trace in zip(targets, traces)
                if target["kind"] == "score" and trace["challenge"] == challenge
            )
        )
    grams = [_grams(prompt["state"]) for prompt in prompts]
    similarities = sorted(
        (_jaccard(grams[i], grams[j]) for i, j in itertools.combinations(range(18), 2)),
        reverse=True,
    )
    long_interleaving = []
    causal_proof_complete = True
    for prompt, trace in zip(prompts, traces):
        if trace["challenge"] != "long_dossier":
            continue
        policy = POLICIES[trace["policy_id"]]
        visible = _visible_documents(prompt["state"], policy)
        case_id = prompt["questions"]["decision"]["instructions"].split()[3].rstrip(",")
        target_positions = [
            index for index, row in enumerate(visible) if row["case_id"] == case_id
        ]
        long_interleaving.append(
            {
                "target_in_first_three": sum(index < 3 for index in target_positions),
                "target_in_last_three": sum(
                    index >= len(visible) - 3 for index in target_positions
                ),
                "all_target_in_one_block": target_positions == list(range(4))
                or target_positions == list(range(3, 7)),
            }
        )
        interventions = trace.get("causal_value_interventions", {})
        if set(interventions) != set(policy.fields) or any(
            row["original_output"] == row["intervention_output"]
            for row in interventions.values()
        ):
            causal_proof_complete = False
    choice_position_gate = (
        sum(resolved_choice_position.values()) == 5
        and len(resolved_choice_position) >= 2
        and max(resolved_choice_position.values(), default=0) <= 3
    )
    structural_gate = (
        metadata_structure_ok
        and causal_proof_complete
        and all(
            set(block) == {"rule_precedence", "partial_evidence", "long_dossier"}
            for block in by_six
        )
        and not any(row["all_target_in_one_block"] for row in long_interleaving)
        and choice_position_gate
    )
    result = {
        "version": VERSION,
        "status": "AUTOMATED_SHORTCUT_CHECK_ONLY" if structural_gate else "BLOCKED",
        "release_qualified": False,
        "blind_review_passed": False,
        "prompts_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
        "metadata_structure_ok": metadata_structure_ok,
        "challenge_by_order_third": by_six,
        "style_by_challenge": style_by_challenge,
        "noul_by_style": noul_by_style,
        "choice_answer_option_index": dict(choice_position),
        "resolved_choice_option_index": dict(resolved_choice_position),
        "choice_position_gate": choice_position_gate,
        "score_by_challenge": score_by_challenge,
        "max_narrative_fivegram_jaccard": round(similarities[0], 4),
        "p95_narrative_fivegram_jaccard": round(
            similarities[int((len(similarities) - 1) * 0.05)], 4
        ),
        "long_all_target_in_one_block": sum(
            row["all_target_in_one_block"] for row in long_interleaving
        ),
        "long_target_first_three_total": sum(
            row["target_in_first_three"] for row in long_interleaving
        ),
        "long_target_last_three_total": sum(
            row["target_in_last_three"] for row in long_interleaving
        ),
        "causal_value_proof_complete": causal_proof_complete,
        "small_sample_limit": "Metadata checks at n=18 cannot rule out learned shortcuts or editorial ambiguity.",
    }
    output.write_bytes(compact(result))
    output.chmod(0o600)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.packet, args.output)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "prompts_sha256",
                    "metadata_structure_ok",
                    "causal_value_proof_complete",
                    "max_narrative_fivegram_jaccard",
                    "long_all_target_in_one_block",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
