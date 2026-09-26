"""Build the unapproved authored v4 typed-decision candidate.

V4 has sixty formal operations and one scored item per unique fact scenario.
The release labels are written privately and are never read for model choice.
The builder's blocked gate cannot be converted to a passing rank by this tool.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import statistics
from typing import Any

from .authored_v4_ops import (
    BY_ID,
    DOMAINS,
    OPERATIONS,
    Operation,
    evaluate,
    generate_facts,
    inject_missing,
)
from .authored_v4_reference import evaluate_rendered
from .sealed_authored import (
    _near_overlap,
    _normal_state,
    _protected_rows,
    compact,
    input_digest,
    sha_bytes,
    sha_file,
)

BUILD_VERSION = "jevarena-authored-build/4"
CHALLENGES = (
    "near_distractor",
    "long_context",
    "insufficient_evidence",
    "rule_precedence",
)
STYLES = ("json", "bullets", "table", "numbered")
POLICY_RE = re.compile(r"^CURRENT POLICY \[([^\]]+)\]: (.+)$", re.MULTILINE)
TARGET_RE = re.compile(r"Target case ID: ([A-Z]{3}-[0-9A-F]{12})\.")
BEGIN_RE = re.compile(
    r"^BEGIN EVIDENCE \[([A-Z]{3}-[0-9A-F]{12})\] style=([a-z]+)$", re.MULTILINE
)
DOMAIN_TITLES = {
    "access": "access governance packet",
    "archive": "archive accession file",
    "delivery": "delivery coordination record",
    "finance": "finance review case",
    "incident": "incident desk report",
    "release": "release management packet",
}
DOCUMENT_GENRES = (
    "intake memorandum",
    "reviewer correspondence",
    "audit log",
    "capacity worksheet",
    "exception request",
    "meeting minutes",
    "evidence register",
    "status bulletin",
    "handoff summary",
    "approval ledger",
    "incident chronology",
    "closeout note",
)
DOCUMENT_ACTIONS = (
    "the owner requested a second verification before any commitment",
    "the committee recorded a conditional approval tied to its own schedule",
    "two reviewers disagreed about which version had been circulated",
    "a revised estimate superseded an earlier provisional figure",
    "a pending dependency was cleared after a separate review",
    "the record was forwarded for a check of source authority",
    "the escalation route changed when the deadline moved",
    "a missing signature delayed the local closeout decision",
    "a duplicate entry was reconciled against the signed register",
    "the team retained both a draft and a confirmed calculation",
    "the case owner split the work into two dated phases",
    "the decision was deferred until the next evidence window",
)
DOCUMENT_EVIDENCE = (
    "The attached figures list capacity {capacity}, cost {cost}, and risk {risk}.",
    "Its register notes {passed} checks passed and {failed} checks unresolved.",
    "The submitted window spans days {start} through {end}; the record gives no cross-case authorization.",
    "A later annotation cites version {version} and a reviewer timestamp of day {day}.",
    "The local estimate changed by {cost} units after the second review.",
    "The recorded approvals were signed by {owner} and dated day {day}.",
)


def _case_rng(
    seed: bytes,
    phase: str,
    op: Operation,
    challenge: str,
    ordinal: int,
    attempt: int = 0,
) -> random.Random:
    digest = hashlib.sha256(
        seed + compact([phase, op.id, challenge, ordinal, attempt]).encode()
    ).digest()
    return random.Random(int.from_bytes(digest, "big"))


def _evidence(facts: dict[str, Any], style: str) -> str:
    if style == "json":
        return json.dumps(facts, ensure_ascii=False, sort_keys=True, indent=2)
    if style == "bullets":
        return "\n".join(f"- {key}: {compact(value)}" for key, value in facts.items())
    if style == "table":
        return "| field | reported value |\n| --- | --- |\n" + "\n".join(
            f"| {key} | {compact(value)} |" for key, value in facts.items()
        )
    if style == "numbered":
        return "\n".join(
            f"{index}. {key} = {compact(value)}"
            for index, (key, value) in enumerate(facts.items(), 1)
        )
    raise ValueError("Unknown v4 evidence style")


def _parse_evidence(body: str, style: str) -> dict[str, Any]:
    if style == "json":
        raw = json.loads(body)
        if not isinstance(raw, dict):
            raise ValueError("JSON evidence must be an object")
        return raw
    rows = body.splitlines()
    if style == "bullets":
        cells = [row[2:].split(": ", 1) for row in rows if row.startswith("- ")]
    elif style == "table":
        if rows[:2] != ["| field | reported value |", "| --- | --- |"]:
            raise ValueError("Table evidence header differs")
        cells = [
            row[2:-2].split(" | ", 1)
            for row in rows[2:]
            if row.startswith("| ") and row.endswith(" |")
        ]
    elif style == "numbered":
        cells = [
            row.split(". ", 1)[1].split(" = ", 1)
            for row in rows
            if re.match(r"^\d+\. ", row)
        ]
    else:
        raise ValueError("Unknown rendered evidence style")
    if len(cells) != len(rows) - (2 if style == "table" else 0):
        raise ValueError("Unparsed rendered evidence row")
    result: dict[str, Any] = {}
    for key, value in cells:
        if key in result or not re.fullmatch(r"[a-z_]+", key):
            raise ValueError("Duplicate or malformed rendered evidence field")
        result[key] = json.loads(value)
    return result


def _dossier(rng: random.Random, domain: str, count: int) -> list[str]:
    """Heterogeneous case documents with related, occasionally conflicting details."""
    entries = []
    cases = [f"{domain[:3].upper()}-{rng.getrandbits(48):012X}" for _ in range(9)]
    for number in range(count):
        case = cases[number % len(cases)]
        owner = f"reviewer-{rng.randint(101, 998)}"
        day = rng.randint(1, 29)
        genre = DOCUMENT_GENRES[
            (number * 5 + rng.randrange(len(DOCUMENT_GENRES))) % len(DOCUMENT_GENRES)
        ]
        action = DOCUMENT_ACTIONS[
            (number * 7 + rng.randrange(len(DOCUMENT_ACTIONS))) % len(DOCUMENT_ACTIONS)
        ]
        evidence = DOCUMENT_EVIDENCE[
            (number * 3 + rng.randrange(len(DOCUMENT_EVIDENCE)))
            % len(DOCUMENT_EVIDENCE)
        ]
        rendered = evidence.format(
            capacity=rng.randint(4, 24),
            cost=rng.randint(3, 34),
            risk=rng.randint(0, 4),
            passed=rng.randint(0, 4),
            failed=rng.randint(0, 3),
            start=rng.randint(1, 12),
            end=rng.randint(16, 29),
            version=rng.randint(1, 8),
            day=day,
            owner=owner,
        )
        entries.append(
            f"Document {number + 1}: {genre.title()} | case {case} | filed day {day} by {owner}. "
            f"For this {domain} file, {action}. {rendered} "
            f"The register cross-references case {cases[(number + 3) % len(cases)]}, "
            "whose review occurred in a separate reporting window. "
            f"A note from day {max(0, day - rng.randint(0, 5))} records the earlier position; "
            "the signed entry above is the latest status for this file."
        )
    return entries


def _question(op: Operation, target_id: str) -> dict[str, Any]:
    if op.kind == "choice":
        return {
            "type": "choice",
            "instructions": f"For target case {target_id}, which candidate does the CURRENT policy select?",
            "criteria": {
                **{key: f"candidate {key}" for key in ("A", "B", "C", "D")},
                "hold": "hold for missing or infeasible evidence",
            },
        }
    if op.kind == "noul":
        return {
            "type": "noul",
            "instructions": f"For target case {target_id}, does the CURRENT decision test hold? Return true or false.",
            "criteria": {"false": "No", "true": "Yes"},
        }
    return {
        "type": "score",
        "instructions": f"For target case {target_id}, what integer grade does the CURRENT policy assign?",
        "criteria": [f"grade {level}" for level in range(5)],
    }


def _fallback(op: Operation) -> str:
    base = (
        "A null fact with admissible_uncertainty has exactly the two listed possible values; "
        "evaluate both complete worlds. If their answers agree, return that answer. "
        "If they differ, apply the conservative fallback. "
    )
    if op.kind == "choice":
        return (
            base
            + "For unresolved evidence or no qualifying candidate, choose hold. Resolve exact ties by candidate ID A before B before C before D."
        )
    if op.kind == "noul":
        return base + "For unresolved evidence, return false."
    return (
        base
        + "For unresolved evidence, assign grade zero. The grade range is zero through four."
    )


def make_spec(
    seed: bytes,
    phase: str,
    op: Operation,
    challenge: str,
    ordinal: int,
    attempt: int = 0,
) -> dict[str, Any] | None:
    rng = _case_rng(seed, phase, op, challenge, ordinal, attempt)
    target = f"{op.domain[:3].upper()}-{rng.getrandbits(48):012X}"
    facts = generate_facts(op, rng)
    if challenge == "insufficient_evidence":
        facts = inject_missing(op, facts, rng, resolved=attempt % 2 == 0)
        if facts is None:
            return None
    style = STYLES[(ordinal + CHALLENGES.index(challenge)) % len(STYLES)]
    return {
        "id": f"{phase}-{op.id}-{challenge}-{ordinal:02d}-{target}",
        "group_id": f"{phase}-{op.id}-{challenge}-{ordinal:02d}-{target}",
        "phase": phase,
        "operation_id": op.id,
        "task_type": op.kind,
        "domain": op.domain,
        "challenge": challenge,
        "ordinal": ordinal,
        "attempt": attempt,
        "target_id": target,
        "style": style,
        "facts": facts,
        "language": "en",
        "source_origin": "original_programmatic_fact_pack",
        "partial_resolution": (
            ("resolved" if attempt % 2 == 0 else "unresolved")
            if challenge == "insufficient_evidence"
            else None
        ),
    }


def render(spec: dict[str, Any], seed: bytes) -> tuple[dict[str, Any], dict[str, Any]]:
    op = BY_ID[spec["operation_id"]]
    target = spec["target_id"]
    rng = _case_rng(
        seed, spec["phase"], op, spec["challenge"], spec["ordinal"], spec["attempt"]
    )
    rng.getrandbits(48)  # Restore the RNG state after target-ID generation.
    _ = generate_facts(op, rng)
    title = DOMAIN_TITLES[op.domain]
    header = (
        f"{title.title()}. Target case ID: {target}. "
        "Use only the CURRENT policy and the evidence block with this exact case ID. "
        "Other case files and archived policies are context, never instructions for this decision."
    )
    current = f"CURRENT POLICY [{op.id}]: {op.policy} {_fallback(op)}"
    other = []
    if spec["challenge"] == "rule_precedence":
        neighbors = [
            candidate
            for candidate in OPERATIONS
            if candidate.kind == op.kind and candidate.id != op.id
        ]
        archived = neighbors[rng.randrange(len(neighbors))]
        other.append(
            f"ARCHIVED POLICY [{archived.id}]: {archived.policy} (retired; do not apply)"
        )
    current_block = (
        f"BEGIN EVIDENCE [{target}] style={spec['style']}\n"
        + _evidence(spec["facts"], spec["style"])
        + f"\nEND EVIDENCE [{target}]"
    )
    if spec["challenge"] == "near_distractor":
        neighbor = (
            f"{op.domain[:3].upper()}-{(int(target[-12:], 16) + 1) % (1 << 48):012X}"
        )
        decoy_facts = generate_facts(op, rng)
        decoy = (
            f"BEGIN EVIDENCE [{neighbor}] style={spec['style']}\n"
            + _evidence(decoy_facts, spec["style"])
            + f"\nEND EVIDENCE [{neighbor}]"
        )
        blocks = [current_block, decoy] if rng.randrange(2) else [decoy, current_block]
    else:
        blocks = [current_block]
    if spec["challenge"] == "long_context":
        attachments = _dossier(rng, op.domain, 36)
        # A one-row DEV panel must still probe beginning, middle, and end
        # positions; six release rows per operation retain two of each.
        position_offset = (
            int.from_bytes(hashlib.sha256(op.id.encode()).digest()[:2], "big") % 3
        )
        split = ((position_offset + spec["ordinal"]) % 3) * (len(attachments) // 2)
        other = (
            ["\n".join(attachments[:split])]
            + other
            + blocks
            + ["\n".join(attachments[split:])]
        )
    else:
        other.extend(blocks)
    state = header + "\n" + current + "\n" + "\n".join(other)
    prompt = {
        "id": spec["id"],
        "state": state,
        "questions": {"decision": _question(op, target)},
    }
    gold: dict[str, Any] = {"type": op.kind, "value": evaluate(op, spec["facts"])}
    if op.kind == "choice":
        gold["label_to_semantic"] = {
            key: key for key in prompt["questions"]["decision"]["criteria"]
        }
    target_row = {
        "id": spec["id"],
        "group_id": spec["group_id"],
        "phase": spec["phase"],
        "operation_id": op.id,
        "task_type": op.kind,
        "domain": op.domain,
        "challenge": spec["challenge"],
        "family": f"{op.kind}/{spec['challenge']}",
        "language": "en",
        "source_origin": "original_programmatic_fact_pack",
        "source_spec_sha256": sha_bytes(compact(spec).encode()),
        "source_input_sha256": input_digest(prompt),
        "gold": {"decision": gold},
    }
    return prompt, target_row


def solve_visible(prompt: dict[str, Any]) -> str | bool | int:
    """Reparse a standalone model-visible prompt and use the separate oracle."""
    state = prompt["state"]
    target_match, policy_match = TARGET_RE.search(state), POLICY_RE.search(state)
    if target_match is None or policy_match is None:
        raise ValueError("Rendered target or CURRENT policy missing")
    target = target_match.group(1)
    op = BY_ID.get(policy_match.group(1))
    if op is None or policy_match.group(2) != op.policy + " " + _fallback(op):
        raise ValueError("Rendered policy differs from registered semantics")
    question = prompt["questions"]["decision"]
    if question != _question(op, target):
        raise ValueError("Rendered question differs from operation contract")
    markers = list(BEGIN_RE.finditer(state))
    matching = [marker for marker in markers if marker.group(1) == target]
    if len(matching) != 1:
        raise ValueError("Rendered target evidence must be unique")
    marker = matching[0]
    end = f"\nEND EVIDENCE [{target}]"
    end_at = state.find(end, marker.end())
    if end_at < 0:
        raise ValueError("Rendered target evidence end missing")
    body = state[marker.end() + 1 : end_at]
    facts = _parse_evidence(body, marker.group(2))
    expected_keys = set(op.context_fields) | (
        {"options"} if op.option_fields else set()
    )
    if "admissible_uncertainty" in facts:
        expected_keys.add("admissible_uncertainty")
    if set(facts) != expected_keys:
        raise ValueError("Rendered required fact keys differ")
    if op.option_fields:
        options = facts["options"]
        if (
            not isinstance(options, list)
            or len(options) != 4
            or [r.get("id") for r in options] != list("ABCD")
        ):
            raise ValueError("Rendered candidate list differs")
        if any(set(row) != set(op.option_fields) | {"id"} for row in options):
            raise ValueError("Rendered candidate fields differ")
    return evaluate_rendered(op, facts)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(compact(row) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _semantic_fingerprint() -> dict[str, int]:
    patterns = defaultdict(dict)
    for op in OPERATIONS:
        pattern = tuple(
            str(evaluate(op, generate_facts(op, random.Random(seed))))
            for seed in range(256)
        )
        if pattern in patterns[op.kind]:
            raise ValueError(
                f"Behavioral collision: {op.id} and {patterns[op.kind][pattern]}"
            )
        patterns[op.kind][pattern] = op.id
    return {kind: len(values) for kind, values in patterns.items()}


def _cell_specs(
    seed: bytes, phase: str, op: Operation, challenge: str, seen_core: set[str]
) -> list[dict[str, Any]]:
    """Stratify six independently generated scenarios by actual oracle outcome."""
    if phase == "dev":
        for attempt in range(1024):
            spec = make_spec(seed, phase, op, challenge, 0, attempt)
            if spec is not None:
                return [spec]
        raise ValueError(f"No valid DEV scenario: {op.id}/{challenge}")
    observed = Counter()
    for attempt in range(384):
        spec = make_spec(seed, phase, op, challenge, 0, attempt)
        if spec is not None:
            observed[str(evaluate(op, spec["facts"]))] += 1
        if op.kind == "noul" and observed["True"] >= 3 and observed["False"] >= 3:
            break
        if op.kind != "noul" and sum(count >= 2 for count in observed.values()) >= 3:
            if (
                challenge != "insufficient_evidence"
                or observed["hold" if op.kind == "choice" else "0"] >= 2
            ):
                break
    if op.kind == "noul":
        labels = ["True", "False"]
    else:
        fallback = "hold" if op.kind == "choice" else "0"
        choices = [
            label
            for label, _ in observed.most_common()
            if challenge != "insufficient_evidence" or label != fallback
        ]
        labels = ([fallback] if challenge == "insufficient_evidence" else []) + choices[
            :3
        ]
        labels = labels[:3]
    if len(labels) != (2 if op.kind == "noul" else 3) or any(
        observed[label] == 0 for label in labels
    ):
        raise ValueError(
            f"Outcome diversity unavailable: {op.id}/{challenge} {dict(observed)}"
        )
    schedule = labels * (3 if op.kind == "noul" else 2)
    selected = []
    for ordinal, wanted in enumerate(schedule):
        for attempt in range(4096):
            spec = make_spec(seed, phase, op, challenge, ordinal, attempt)
            if spec is None or str(evaluate(op, spec["facts"])) != wanted:
                continue
            core = sha_bytes(
                compact({"operation_id": op.id, "facts": spec["facts"]}).encode()
            )
            if core not in seen_core:
                seen_core.add(core)
                selected.append(spec)
                break
        else:
            raise ValueError(
                f"Could not satisfy label {wanted}: {op.id}/{challenge}/{ordinal}"
            )
    return selected


def audit_panel(
    prompts: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    specs: list[dict[str, Any]],
    protected: list[dict[str, Any]],
    per_challenge: int,
) -> dict[str, Any]:
    if not len(prompts) == len(targets) == len(specs):
        raise ValueError("V4 prompt/target/spec row counts differ")
    if len({row["id"] for row in prompts}) != len(prompts):
        raise ValueError("Duplicate V4 prompt ID")
    if len({row["group_id"] for row in targets}) != len(targets):
        raise ValueError("V4 scored variants share a source group")
    if len(
        {
            sha_bytes(
                compact(
                    {"operation_id": row["operation_id"], "facts": row["facts"]}
                ).encode()
            )
            for row in specs
        }
    ) != len(specs):
        raise ValueError("V4 source fact scenario repeated")
    for prompt, target, spec in zip(prompts, targets, specs):
        if not prompt["id"] == target["id"] == spec["id"]:
            raise ValueError("V4 source/prompt/target ID differs")
        if target["source_spec_sha256"] != sha_bytes(compact(spec).encode()) or target[
            "source_input_sha256"
        ] != input_digest(prompt):
            raise ValueError("V4 source/input digest differs")
        if target["gold"]["decision"]["value"] != evaluate(
            BY_ID[spec["operation_id"]], spec["facts"]
        ):
            raise ValueError("V4 direct oracle differs from frozen gold")
        if target["gold"]["decision"]["value"] != solve_visible(prompt):
            raise ValueError("V4 visible-fact reference oracle differs")
    operation_counts = Counter(row["operation_id"] for row in targets)
    cell_counts = Counter((row["operation_id"], row["challenge"]) for row in targets)
    type_counts = Counter(row["task_type"] for row in targets)
    challenge_counts = Counter(row["challenge"] for row in targets)
    domain_counts = Counter(row["domain"] for row in targets)
    if (
        set(operation_counts) != set(BY_ID)
        or set(operation_counts.values()) != {4 * per_challenge}
        or len(cell_counts) != 240
        or set(cell_counts.values()) != {per_challenge}
        or set(type_counts.values()) != {20 * 4 * per_challenge}
        or set(challenge_counts.values()) != {60 * per_challenge}
    ):
        raise ValueError("V4 operation/type/challenge coverage matrix incomplete")
    protected_states = {
        _normal_state(row["state"])
        for row in protected
        if isinstance(row["state"], str)
    }
    protected_inputs = {row["input_digest"] for row in protected if row["input_digest"]}
    exact_state = [
        row["id"] for row in prompts if _normal_state(row["state"]) in protected_states
    ]
    exact_input = [
        row["id"] for row in prompts if input_digest(row) in protected_inputs
    ]
    near_protected = _near_overlap(prompts, protected)
    near_internal = [
        pair
        for pair in _near_overlap(prompts, prompts, limit=0.94)
        if pair["panel_id"] != pair["other_id"]
    ]
    long_words = [
        len(row["state"].split())
        for row, target in zip(prompts, targets)
        if target["challenge"] == "long_context"
    ]
    fallback_counts = Counter(
        str(row["gold"]["decision"]["value"]).lower()
        for row in targets
        if row["challenge"] == "insufficient_evidence"
    )
    cell_labels = defaultdict(Counter)
    for row in targets:
        cell_labels[(row["operation_id"], row["challenge"])][
            str(row["gold"]["decision"]["value"])
        ] += 1
    balance_violations = []
    if per_challenge == 6:
        for (op_id, challenge), counts in cell_labels.items():
            kind = BY_ID[op_id].kind
            if kind == "noul":
                valid = counts["True"] >= 2 and counts["False"] >= 2
            else:
                valid = len(counts) >= 3 and max(counts.values()) <= 3
                if challenge == "insufficient_evidence":
                    valid &= counts["hold" if kind == "choice" else "0"] >= 2
            if not valid:
                balance_violations.append(
                    {"cell": f"{op_id}/{challenge}", "labels": dict(counts)}
                )
    behavioral = _semantic_fingerprint()
    return {
        "audit_version": "jevarena-authored-v4-audit/1",
        "items": len(prompts),
        "unique_scenario_groups": len({r["group_id"] for r in targets}),
        "independent_groups": len({r["group_id"] for r in targets}),
        "semantic_operations": len(operation_counts),
        "behaviorally_distinct_operations": behavioral,
        "dual_oracle_agreement": len(prompts),
        "by_type": dict(sorted(type_counts.items())),
        "by_challenge": dict(sorted(challenge_counts.items())),
        "by_domain": dict(sorted(domain_counts.items())),
        "by_operation": dict(sorted(operation_counts.items())),
        "by_operation_challenge": {
            "/".join(key): value for key, value in sorted(cell_counts.items())
        },
        "evidence_styles": dict(
            sorted(Counter(spec["style"] for spec in specs).items())
        ),
        "long_context_min_words": min(long_words),
        "long_context_median_words": statistics.median(long_words),
        "insufficient_evidence_gold": dict(sorted(fallback_counts.items())),
        "cell_label_counts": {
            "/".join(key): dict(value) for key, value in sorted(cell_labels.items())
        },
        "balance_violations": balance_violations,
        "exact_state_overlap_ids": exact_state,
        "exact_input_overlap_ids": exact_input,
        "near_protected_pairs": near_protected,
        "near_internal_pairs": near_internal,
        "automated_status": (
            "passed"
            if not (
                exact_state
                or exact_input
                or near_protected
                or near_internal
                or balance_violations
            )
            else "blocked"
        ),
        "near_policy": "token-5-gram Jaccard >=0.88 protected or >=0.94 internal; rare-shingle candidate screen",
    }


def build(
    phase: str, seed_path: Path, protected_list: Path, output: Path
) -> dict[str, Any]:
    if phase not in ("dev", "release") or output.exists():
        raise ValueError("V4 phase dev/release required; output must be new")
    seed = seed_path.read_bytes()
    if len(seed) != 32:
        raise ValueError("V4 private seed must contain 256 bits")
    per_challenge = 1 if phase == "dev" else 6
    prompts, targets, specs = [], [], []
    seen_core: set[str] = set()
    for op in OPERATIONS:
        for challenge in CHALLENGES:
            for spec in _cell_specs(seed, phase, op, challenge, seen_core):
                core_hash = sha_bytes(
                    compact({"operation_id": op.id, "facts": spec["facts"]}).encode()
                )
                if phase == "dev":
                    if core_hash in seen_core:
                        raise ValueError(f"Repeated DEV scenario: {op.id}")
                    seen_core.add(core_hash)
                prompt, target = render(spec, seed)
                specs.append(spec)
                prompts.append(prompt)
                targets.append(target)
    protected, sources = _protected_rows(protected_list)
    audit = audit_panel(prompts, targets, specs, protected, per_challenge)
    if audit["automated_status"] != "passed":
        raise ValueError("V4 automated overlap/coverage audit blocked the candidate")
    expected = 240 if phase == "dev" else 1440
    if audit["items"] != expected or audit["unique_scenario_groups"] != expected:
        raise ValueError("V4 preregistered item count differs")
    staging = output.with_name(output.name + ".pending")
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)
    _write_jsonl(staging / "prompts.jsonl", prompts)
    _write_jsonl(staging / "targets.jsonl", targets)
    _write_jsonl(staging / "source_specs.jsonl", specs)
    for private in ("targets.jsonl", "source_specs.jsonl"):
        (staging / private).chmod(0o600)
    # One editorial case per operation/challenge; variants are not added to the scored denominator.
    review = []
    review_key = []
    selected = set()
    for index, target in enumerate(targets):
        cell = (target["operation_id"], target["challenge"])
        if cell not in selected:
            selected.add(cell)
            review.append(
                {"id": target["id"], "cell": "/".join(cell), "prompt": prompts[index]}
            )
            review_key.append(
                {
                    "id": target["id"],
                    "cell": "/".join(cell),
                    "source_spec_sha256": target["source_spec_sha256"],
                    "gold": target["gold"]["decision"],
                }
            )
    _write_jsonl(staging / "review_packet.private.jsonl", review)
    (staging / "review_packet.private.jsonl").chmod(0o600)
    _write_jsonl(staging / "review_key.private.jsonl", review_key)
    (staging / "review_key.private.jsonl").chmod(0o600)
    audit["protected_sources"] = sources
    audit["protected_list_sha256"] = sha_file(protected_list)
    (staging / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    (staging / "audit.json").chmod(0o600)
    manifest = {
        "build_version": BUILD_VERSION,
        "phase": phase,
        "items": expected,
        "unique_scenario_groups": expected,
        "independent_groups": expected,
        "semantic_operations": 60,
        "language": "en",
        "language_scope": "English-only",
        "source_origin": "original_programmatic_fact_pack",
        "seed_commitment_sha256": sha_bytes(seed),
        "builder_files_sha256": {
            path.name: sha_file(path)
            for path in (
                Path(__file__),
                Path(__file__).with_name("authored_v4_ops.py"),
                Path(__file__).with_name("authored_v4_reference.py"),
            )
        },
        "prompts_sha256": sha_file(staging / "prompts.jsonl"),
        "targets_sha256": sha_file(staging / "targets.jsonl"),
        "source_specs_sha256": sha_file(staging / "source_specs.jsonl"),
        "review_packet_sha256": sha_file(staging / "review_packet.private.jsonl"),
        "review_key_sha256": sha_file(staging / "review_key.private.jsonl"),
        "automated_audit_sha256": sha_file(staging / "audit.json"),
        "counts": {
            key: audit[key]
            for key in (
                "by_type",
                "by_challenge",
                "by_domain",
                "by_operation",
                "by_operation_challenge",
            )
        },
        "quality_gate": {
            "status": "blocked",
            "reason": "independent_editorial_review_pending",
        },
        "ranking_eligible": False,
    }
    (staging / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    staging.rename(output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("dev", "release"), required=True)
    parser.add_argument("--seed-file", type=Path, required=True)
    parser.add_argument("--protected-list", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = build(args.phase, args.seed_file, args.protected_list, args.output_dir)
    print(
        compact(
            {
                "items": manifest["items"],
                "semantic_operations": manifest["semantic_operations"],
                "quality_gate": manifest["quality_gate"],
                "prompts_sha256": manifest["prompts_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
