"""Construct a private, DEV-only authored JevArena v11 editorial packet.

Public code defines rules and mechanical gates. The case facts, prose,
extraction quotes, gold, salt and deletion witnesses remain private.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .authored_v5_dossier import compact, opaque

VERSION = "jevarena-authored-v11-dev12-editorial-pilot/1"
COUNTS = {"choice": 4, "noul": 4, "score": 4}
POSITION_SCHEDULE = (2, 4, 1, 3)
FORMATS = {
    "email",
    "log",
    "minutes",
    "form",
    "letter",
    "schedule",
    "checklist",
    "report",
}
FORMAT_CUES = {
    "email": ("From:", "Subject:"),
    "log": ("Log entry", "|"),
    "minutes": ("Minutes:", "Agenda:"),
    "form": ("Form:", "Field:"),
    "letter": ("To:", "Sincerely,"),
    "schedule": ("Schedule:", "|"),
    "checklist": ("Checklist:", "["),
    "report": ("Report:", "Finding:"),
}
NEUTRAL_LEAD_FORBIDDEN = re.compile(
    r"\b(?:source|record|file|document|exhibit|attach(?:ed|ment)?|"
    r"signed|signature|approved|certified|confirmed|cleared|"
    r"likely|winner|answer|outcome|contains|includes|lists)\b",
    re.I,
)
WORD = re.compile(r"[A-Za-z0-9]+")


@dataclass(frozen=True)
class Operation:
    id: str
    kind: str
    fields: tuple[str, ...]
    rule: str


OPS = {
    op.id: op
    for op in (
        Operation(
            "temporal-bulletin",
            "choice",
            ("issued", "valid_until", "decision_day"),
            "A bulletin is eligible when issued no later than the decision day "
            "and still valid on that day. Choose the eligible bulletin with "
            "the latest issue day, breaking ties alphabetically. If none is "
            "eligible, select hold.",
        ),
        Operation(
            "shared-interval",
            "choice",
            ("slots", "client_window", "guide_window"),
            "A slot must fit entirely within both the client and guide windows. "
            "Choose the eligible slot with the earliest start, breaking ties "
            "alphabetically. If no slot fits, select hold.",
        ),
        Operation(
            "service-coverage",
            "choice",
            ("coverage", "required", "prices", "cap"),
            "A provider qualifies only when its coverage includes every required "
            "service and its price does not exceed the cap. Choose the cheapest "
            "qualifying provider, breaking ties alphabetically; otherwise hold.",
        ),
        Operation(
            "scenario-minimax",
            "choice",
            ("loss_red", "loss_blue", "spend", "budget"),
            "Among plans whose spend does not exceed budget, choose the plan "
            "with the smallest worse loss across the red and blue scenarios. "
            "Break ties by smaller spend, then alphabetically; otherwise hold.",
        ),
        Operation(
            "emergency-exception-chain",
            "noul",
            ("triggered", "waiver", "safety_review"),
            "The special exception is valid only when the triggering condition "
            "holds, the waiver is active, and the safety review is complete.",
        ),
        Operation(
            "universal-access",
            "noul",
            ("participants", "cleared", "exempt"),
            "Authorize the group only if every participant is either cleared "
            "or explicitly exempt.",
        ),
        Operation(
            "common-availability",
            "noul",
            ("slots_a", "slots_b", "slots_c"),
            "Certify a joint appointment only if the three parties have at "
            "least one common available slot.",
        ),
        Operation(
            "exposure-budget",
            "noul",
            ("leg_a", "leg_b", "budget"),
            "Certify the journey only if total exposure across the two legs "
            "does not exceed the allowed budget.",
        ),
        Operation(
            "weighted-compliance",
            "score",
            ("late_events", "unresolved", "credit"),
            "Start at grade four. Subtract one per late event, two per unresolved "
            "exception, and then add the mitigation credit. Clamp to zero "
            "through four.",
        ),
        Operation(
            "completion-rate-rubric",
            "score",
            ("completed", "assigned", "thresholds"),
            "Compute completed divided by assigned. Grade by the number of "
            "threshold fractions met in the four-entry rubric, including "
            "equality. Assigned must be positive.",
        ),
        Operation(
            "three-way-consistency",
            "score",
            ("reading_a", "reading_b", "reading_c"),
            "Grade four if all three coded readings agree, two if exactly a "
            "pair agrees, and zero if all differ.",
        ),
        Operation(
            "ordered-milestones",
            "score",
            ("stage_a", "stage_b", "stage_c", "stage_d"),
            "Grade the length of the completed prefix of stages in the listed "
            "order. A later completion cannot bypass an earlier incomplete stage.",
        ),
    )
}
assert len(OPS) == 12


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _names(facts: dict[str, Any]) -> set[str]:
    for value in facts.values():
        if isinstance(value, dict):
            return set(value)
    return set()


def nonnegative_int(value: Any) -> bool:
    return type(value) is int and value >= 0


def validate(op: Operation, facts: dict[str, Any]) -> None:
    if set(facts) != set(op.fields):
        raise ValueError("Fact fields differ from operation contract")
    if op.kind == "choice":
        names = _names(facts)
        if not 3 <= len(names) <= 5 or any(
            re.fullmatch(r"[A-Za-z]+", name) is None for name in names
        ):
            raise ValueError("Choice candidates must be three to five names")
        for value in facts.values():
            if isinstance(value, dict) and set(value) != names:
                raise ValueError("Candidate dictionaries disagree")
    if op.id in {"temporal-bulletin", "scenario-minimax"}:
        fields = (
            ("issued", "valid_until")
            if op.id == "temporal-bulletin"
            else ("loss_red", "loss_blue", "spend")
        )
        if any(
            not all(nonnegative_int(value) for value in facts[field].values())
            for field in fields
        ) or not nonnegative_int(
            facts["decision_day"] if op.id == "temporal-bulletin" else facts["budget"]
        ):
            raise ValueError("Invalid temporal or loss facts")
    if op.id == "service-coverage" and (
        not isinstance(facts["required"], list)
        or not facts["required"]
        or len(set(facts["required"])) != len(facts["required"])
        or not nonnegative_int(facts["cap"])
        or not all(nonnegative_int(v) for v in facts["prices"].values())
        or any(
            not isinstance(services, list) or len(set(services)) != len(services)
            for services in facts["coverage"].values()
        )
    ):
        raise ValueError("Invalid provider facts")
    if op.id == "completion-rate-rubric" and (
        not isinstance(facts["assigned"], int)
        or facts["assigned"] <= 0
        or len(facts["thresholds"]) != 4
        or sorted(facts["thresholds"]) != facts["thresholds"]
    ):
        raise ValueError("Invalid rate rubric")
    if op.id in {"shared-interval", "common-availability"}:
        values = (
            [*facts["slots"].values(), facts["client_window"], facts["guide_window"]]
            if op.id == "shared-interval"
            else []
        )
        if any(
            not isinstance(pair, list)
            or len(pair) != 2
            or not all(type(v) is int for v in pair)
            or pair[0] > pair[1]
            for pair in values
        ):
            raise ValueError("Invalid interval")
    if op.id == "universal-access" and any(
        len(set(facts[field])) != len(facts[field])
        for field in ("participants", "cleared", "exempt")
    ):
        raise ValueError("Duplicate participant")
    if op.id == "emergency-exception-chain" and any(
        type(facts[field]) is not bool for field in op.fields
    ):
        raise ValueError("Invalid exception status")
    if op.id == "common-availability" and any(
        not isinstance(facts[field], list)
        or len(facts[field]) != len(set(facts[field]))
        for field in op.fields
    ):
        raise ValueError("Invalid available slots")
    if op.id in {"exposure-budget", "weighted-compliance"} and any(
        not nonnegative_int(facts[field]) for field in op.fields
    ):
        raise ValueError("Invalid numeric facts")
    if op.id == "completion-rate-rubric" and (
        not nonnegative_int(facts["completed"])
        or not all(nonnegative_int(v) for v in facts["thresholds"])
    ):
        raise ValueError("Invalid rubric quantities")
    if op.id == "three-way-consistency" and any(
        type(facts[field]) is not str or not facts[field] for field in op.fields
    ):
        raise ValueError("Invalid coded readings")
    if op.id == "ordered-milestones" and any(
        type(facts[field]) is not bool for field in op.fields
    ):
        raise ValueError("Invalid milestone status")


def evaluate(op: Operation, facts: dict[str, Any]) -> str | bool | int:
    """Direct rule oracle used by the authoring proof."""
    validate(op, facts)
    n = op.id
    if n == "temporal-bulletin":
        eligible = [
            k
            for k in facts["issued"]
            if facts["issued"][k] <= facts["decision_day"] <= facts["valid_until"][k]
        ]
        return (
            min(eligible, key=lambda k: (-facts["issued"][k], k))
            if eligible
            else "hold"
        )
    if n == "shared-interval":
        eligible = [
            k
            for k, (start, end) in facts["slots"].items()
            if facts["client_window"][0] <= start <= end <= facts["client_window"][1]
            and facts["guide_window"][0] <= start <= end <= facts["guide_window"][1]
        ]
        return (
            min(eligible, key=lambda k: (facts["slots"][k][0], k))
            if eligible
            else "hold"
        )
    if n == "service-coverage":
        eligible = [
            k
            for k in facts["coverage"]
            if set(facts["required"]) <= set(facts["coverage"][k])
            and facts["prices"][k] <= facts["cap"]
        ]
        return (
            min(eligible, key=lambda k: (facts["prices"][k], k)) if eligible else "hold"
        )
    if n == "scenario-minimax":
        eligible = [k for k in facts["spend"] if facts["spend"][k] <= facts["budget"]]
        return (
            min(
                eligible,
                key=lambda k: (
                    max(facts["loss_red"][k], facts["loss_blue"][k]),
                    facts["spend"][k],
                    k,
                ),
            )
            if eligible
            else "hold"
        )
    if n == "emergency-exception-chain":
        return facts["triggered"] and facts["waiver"] and facts["safety_review"]
    if n == "universal-access":
        return set(facts["participants"]) <= (
            set(facts["cleared"]) | set(facts["exempt"])
        )
    if n == "common-availability":
        return bool(
            set(facts["slots_a"]) & set(facts["slots_b"]) & set(facts["slots_c"])
        )
    if n == "exposure-budget":
        return facts["leg_a"] + facts["leg_b"] <= facts["budget"]
    if n == "weighted-compliance":
        return max(
            0,
            min(
                4,
                4 - facts["late_events"] - 2 * facts["unresolved"] + facts["credit"],
            ),
        )
    if n == "completion-rate-rubric":
        return sum(
            facts["completed"] * 100 >= facts["assigned"] * threshold
            for threshold in facts["thresholds"]
        )
    if n == "three-way-consistency":
        readings = [facts[field] for field in ("reading_a", "reading_b", "reading_c")]
        return 4 if len(set(readings)) == 1 else 0 if len(set(readings)) == 3 else 2
    if n == "ordered-milestones":
        grade = 0
        for field in ("stage_a", "stage_b", "stage_c", "stage_d"):
            if not facts[field]:
                break
            grade += 1
        return grade
    raise ValueError("Unknown operation")


def reference(op: Operation, facts: dict[str, Any]) -> str | bool | int:
    """Separate implementation to catch mistakes in the direct oracle."""
    validate(op, facts)
    n = op.id
    if op.kind == "choice":
        ranked = []
        for name in sorted(_names(facts)):
            if n == "temporal-bulletin":
                if not (
                    facts["issued"][name]
                    <= facts["decision_day"]
                    <= facts["valid_until"][name]
                ):
                    continue
                rank = (-facts["issued"][name], name)
            elif n == "shared-interval":
                start, end = facts["slots"][name]
                if any(
                    start < window[0] or end > window[1]
                    for window in (facts["client_window"], facts["guide_window"])
                ):
                    continue
                rank = (start, name)
            elif n == "service-coverage":
                if (
                    any(
                        item not in facts["coverage"][name]
                        for item in facts["required"]
                    )
                    or facts["prices"][name] > facts["cap"]
                ):
                    continue
                rank = (facts["prices"][name], name)
            else:
                if facts["spend"][name] > facts["budget"]:
                    continue
                worst = sorted((facts["loss_red"][name], facts["loss_blue"][name]))[-1]
                rank = (worst, facts["spend"][name], name)
            ranked.append((rank, name))
        return sorted(ranked)[0][1] if ranked else "hold"
    if n == "emergency-exception-chain":
        return all(facts[field] is True for field in op.fields)
    if n == "universal-access":
        return all(
            name in facts["cleared"] or name in facts["exempt"]
            for name in facts["participants"]
        )
    if n == "common-availability":
        return any(
            slot in facts["slots_b"] and slot in facts["slots_c"]
            for slot in facts["slots_a"]
        )
    if n == "exposure-budget":
        return facts["budget"] - facts["leg_a"] >= facts["leg_b"]
    if n == "weighted-compliance":
        raw = 4 + facts["credit"]
        for _ in range(facts["late_events"]):
            raw -= 1
        for _ in range(facts["unresolved"]):
            raw -= 2
        return 0 if raw < 0 else 4 if raw > 4 else raw
    if n == "completion-rate-rubric":
        percent = (100 * facts["completed"]) / facts["assigned"]
        return sum(percent >= threshold for threshold in facts["thresholds"])
    if n == "three-way-consistency":
        values = [facts[field] for field in op.fields]
        pairs = sum(values[i] == values[j] for i in range(3) for j in range(i + 1, 3))
        return {0: 0, 1: 2, 3: 4}[pairs]
    if n == "ordered-milestones":
        values = [facts[field] for field in op.fields]
        return next((i for i, value in enumerate(values) if not value), 4)
    raise ValueError("Unknown operation")


def stable_words(text: str) -> list[str]:
    return [word.lower() for word in WORD.findall(text)]


def ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    words = stable_words(text)
    return set(zip(*(words[i:] for i in range(n))))


def words_len(text: str) -> int:
    return len(stable_words(text))


def write_jsonl(path: Path, data: list[dict[str, Any]]) -> None:
    path.write_bytes(b"".join(compact(row) for row in data))
    path.chmod(0o600)


def render_source(spec: dict[str, Any], secret: bytes, slug: str) -> tuple[str, str]:
    if spec["format"] not in FORMATS:
        raise ValueError("Unknown evidence format")
    body = spec["body"].strip()
    if (
        words_len(body) < 32
        or "DATA " in body
        or "CURRENT RULE" in body
        or any(cue not in body for cue in FORMAT_CUES[spec["format"]])
    ):
        raise ValueError("Thin or injected source body")
    if body.count(spec["quote"]) != 1:
        raise ValueError("Private extraction quote must occur exactly once")
    ident = opaque(secret, f"v11:{slug}:source:{spec['slug']}", 16)
    rendered = f"EXHIBIT {ident}\n{spec['title']}\n{body}\nEND EXHIBIT {ident}"
    return ident, rendered


def build_item(
    spec: dict[str, Any], secret: bytes
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    slug = spec["slug"]
    op = OPS[spec["operation_id"]]
    scene = spec["scene"].strip()
    if (
        words_len(scene) < 18
        or words_len(scene) > 65
        or NEUTRAL_LEAD_FORBIDDEN.search(scene)
        or re.search(r"\d", scene)
        or any(name.lower() in stable_words(scene) for name in _names(spec["facts"]))
    ):
        raise ValueError("Task preamble may leak source inventory or answer")
    facts = spec["facts"]
    validate(op, facts)
    if len(spec["sources"]) < 2 or len(
        {source["slug"] for source in spec["sources"]}
    ) != len(spec["sources"]):
        raise ValueError("Source count or identity invalid")
    if {source["field"] for source in spec["sources"]} != set(op.fields):
        raise ValueError("Every operation field requires its own source")
    if len(spec["sources"]) != len(op.fields):
        raise ValueError("All target sources must be necessary")
    if any(
        source["value"] != facts[source["field"]]
        or type(source["value"]) is not type(facts[source["field"]])
        for source in spec["sources"]
    ):
        raise ValueError("Extraction map differs from private facts")
    source_rows = [render_source(source, secret, slug) for source in spec["sources"]]
    rendered = [body for _, body in source_rows]
    quotes = [source["quote"] for source in spec["sources"]]
    if any(
        sum(body.count(quote) for body in rendered) != 1
        or quote in scene
        or quote in op.rule
        for quote in quotes
    ):
        raise ValueError("Source claim is duplicated outside its evidence")
    source_order = sorted(
        range(len(rendered)),
        key=lambda i: opaque(
            secret, f"v11:{slug}:order:{spec['sources'][i]['slug']}", 64
        ),
    )
    ordered_sources = [rendered[i] for i in source_order]
    intro = f"{scene}\n\nCURRENT RULE: {op.rule}"
    state = intro + "\n\n" + "\n\n".join(ordered_sources)
    result = evaluate(op, facts)
    if type(result) is not type(reference(op, facts)) or result != reference(op, facts):
        raise ValueError("Independent rule oracles disagree")
    item_id = opaque(secret, f"v11:{slug}:item", 20)
    ablations = []
    witnesses = {}
    for source, (source_id, block) in zip(spec["sources"], source_rows):
        field = source["field"]
        alternatives = spec["domains"][field]
        if (
            len(alternatives) < 2
            or not any(value == facts[field] for value in alternatives)
            or len({json.dumps(value, sort_keys=True) for value in alternatives})
            != len(alternatives)
        ):
            raise ValueError("Missing or duplicate deletion completion")
        outcomes = []
        for value in alternatives:
            world = {**facts, field: value}
            answer = evaluate(op, world)
            if type(answer) is not type(reference(op, world)) or answer != reference(
                op, world
            ):
                raise ValueError("Deletion oracle mismatch")
            outcomes.append(answer)
        if (
            len({json.dumps(answer) for answer in outcomes}) < 2
            or result not in outcomes
        ):
            raise ValueError("A source does not change the provable answer")
        reduced = (
            intro
            + "\n\n"
            + "\n\n".join(
                source_text for source_text in ordered_sources if source_text != block
            )
        )
        if block in reduced or not reduced.startswith(intro):
            raise ValueError(
                "Deletion changed the task preamble or retained the source"
            )
        for quote in (source["quote"],):
            if quote in reduced:
                raise ValueError("Deleted claim survives in another source")
        ablations.append(
            {
                "parent_id": item_id,
                "omitted_source": source_id,
                "state": reduced,
                "questions": None,
                "review_instruction": "Without assuming facts from the omitted exhibit, is the original decision still justified by the remaining language? Record any surviving clue.",
            }
        )
        witnesses[source_id] = {
            "field": field,
            "completion_count": len(alternatives),
            "different_outputs": len({json.dumps(answer) for answer in outcomes}),
        }
    if op.kind == "choice":
        options = sorted(
            [*_names(facts), "hold"],
            key=lambda k: opaque(secret, f"v11:{slug}:option:{k}", 64),
        )
        target_position = POSITION_SCHEDULE[spec["choice_ordinal"]]
        if result not in options:
            raise ValueError("Choice gold is not an offered option")
        options.remove(result)
        options.insert(target_position - 1, result)
        criteria: Any = {label: f"Select {label}" for label in options}
    elif op.kind == "noul":
        options = sorted(
            ("true", "false"),
            key=lambda k: opaque(secret, f"v11:{slug}:bool:{k}", 64),
        )
        criteria = {
            label: ("Certified" if label == "true" else "Not certified")
            for label in options
        }
    else:
        criteria = [f"Grade {level}" for level in range(5)]
    question = {
        "decision": {
            "type": op.kind,
            "instructions": "Apply the current rule to the evidence shown.",
            "criteria": criteria,
        }
    }
    for row in ablations:
        row["questions"] = question
    prompt = {"id": item_id, "state": state, "questions": question}
    target = {
        "id": item_id,
        "kind": op.kind,
        "answer": {op.kind: result},
        "source_group": opaque(secret, f"v11:{slug}:group", 20),
    }
    proof = {
        "id": item_id,
        "slug": slug,
        "domain": spec["domain"],
        "operation": op.id,
        "answer": result,
        "source_witnesses": witnesses,
        "source_formats": [source["format"] for source in spec["sources"]],
        "words": words_len(state),
    }
    return prompt, target, proof, ablations


def lexical_gate(specs: list[dict[str, Any]]) -> dict[str, Any]:
    bodies = [source["body"] for spec in specs for source in spec["sources"]]
    eight_grams: set[tuple[str, ...]] = set()
    max_overlap = 0.0
    tri_sets = [ngrams(body, 3) for body in bodies]
    for body in bodies:
        current = ngrams(body, 8)
        if current & eight_grams:
            raise ValueError("Repeated eight-word source prose sequence")
        eight_grams |= current
    for i in range(len(bodies)):
        for j in range(i + 1, len(bodies)):
            left, right = tri_sets[i], tri_sets[j]
            union = left | right
            ratio = len(left & right) / len(union) if union else 0.0
            max_overlap = max(max_overlap, ratio)
            if ratio > 0.15:
                raise ValueError(
                    "Source prose trigram overlap exceeds preregistered gate"
                )
    return {
        "source_count": len(bodies),
        "maximum_pairwise_trigram_jaccard": max_overlap,
    }


def build(spec_path: Path, salt_path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("Frozen candidates cannot be overwritten")
    specs = json.loads(spec_path.read_text())
    secret = salt_path.read_bytes()
    if (
        len(secret) != 32
        or len(specs) != 12
        or len({spec["slug"] for spec in specs}) != 12
        or len({spec["domain"] for spec in specs}) != 12
        or {spec["operation_id"] for spec in specs} != set(OPS)
    ):
        raise ValueError(
            "Twelve distinct cases, domains, operations and one salt required"
        )
    lex = lexical_gate(specs)
    prompts, targets, proofs, ablations = [], [], [], []
    for spec in specs:
        prompt, target, proof, source_deletions = build_item(spec, secret)
        prompts.append(prompt)
        targets.append(target)
        proofs.append(proof)
        ablations.extend(source_deletions)
    kinds = Counter(row["kind"] for row in targets)
    bool_answers = Counter(
        row["answer"]["noul"] for row in targets if row["kind"] == "noul"
    )
    score_levels = {row["answer"]["score"] for row in targets if row["kind"] == "score"}
    positions = [
        list(prompt["questions"]["decision"]["criteria"]).index(
            target["answer"]["choice"]
        )
        + 1
        for prompt, target in zip(prompts, targets)
        if target["kind"] == "choice"
    ]
    length_bands = Counter(
        (
            "compact"
            if proof["words"] <= 300
            else "medium" if proof["words"] < 500 else "long"
        )
        for proof in proofs
    )
    formats = {form for proof in proofs for form in proof["source_formats"]}
    if (
        kinds != COUNTS
        or bool_answers != {True: 2, False: 2}
        or len(score_levels) < 3
        or set(positions) != {1, 2, 3, 4}
        or len(formats) < 4
        or len(ablations) != lex["source_count"]
        or any(proof["words"] > 900 for proof in proofs)
    ):
        raise ValueError("Preregistered balance, format or length gate failed")
    order = sorted(
        range(12), key=lambda i: opaque(secret, f"v11:row:{prompts[i]['id']}", 64)
    )
    output.mkdir(parents=True)
    private = output / "private"
    private.mkdir(mode=0o700)
    write_jsonl(output / "prompts.jsonl", [prompts[i] for i in order])
    write_jsonl(output / "deletions.gold-free.jsonl", ablations)
    write_jsonl(private / "targets.jsonl", [targets[i] for i in order])
    write_jsonl(private / "proofs.jsonl", [proofs[i] for i in order])
    receipt = {
        "version": VERSION,
        "status": "AUTOMATED_PROOF_ONLY",
        "release_qualified": False,
        "human_review_passed": False,
        "originals": 12,
        "deletions": len(ablations),
        "type_counts": dict(kinds),
        "boolean_labels": {str(key): value for key, value in bool_answers.items()},
        "score_levels": sorted(score_levels),
        "choice_positions": positions,
        "length_bands": dict(length_bands),
        "formats": sorted(formats),
        "lexical": lex,
        "spec_sha256": sha(spec_path.read_bytes()),
        "salt_commitment_sha256": sha(secret),
        "builder_sha256": sha(Path(__file__).read_bytes()),
        "prompts_sha256": sha((output / "prompts.jsonl").read_bytes()),
        "deletions_sha256": sha((output / "deletions.gold-free.jsonl").read_bytes()),
        "targets_sha256": sha((private / "targets.jsonl").read_bytes()),
        "proofs_sha256": sha((private / "proofs.jsonl").read_bytes()),
    }
    (private / "audit.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    (private / "audit.json").chmod(0o600)
    manifest = {
        "version": VERSION,
        "status": "FROZEN_GOLD_FREE_BLIND_REVIEW",
        "originals": {
            "file": "prompts.jsonl",
            "count": 12,
            "sha256": receipt["prompts_sha256"],
        },
        "deletions": {
            "file": "deletions.gold-free.jsonl",
            "count": len(ablations),
            "sha256": receipt["deletions_sha256"],
        },
        "builder_sha256": receipt["builder_sha256"],
        "spec_sha256": receipt["spec_sha256"],
        "salt_commitment_sha256": receipt["salt_commitment_sha256"],
        "review_sequence": "Solve and seal originals before opening deletions. Seal deletions before accessing targets or proofs.",
    }
    (output / "reviewer_manifest.gold-free.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    (output / "reviewer_manifest.gold-free.json").chmod(0o600)
    receipt["manifest_sha256"] = sha(
        (output / "reviewer_manifest.gold-free.json").read_bytes()
    )
    (private / "audit.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specs", type=Path, required=True)
    parser.add_argument("--salt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.specs, args.salt, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
