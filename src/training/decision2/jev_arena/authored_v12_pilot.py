"""Build a private, DEV-only JevArena authored v12 editorial candidate.

The case prose, fact packs, salt, targets and reviewer join remain on the
authorized experiment host. This source defines the invariant rules and gates.
No model inference or release admission is performed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import secrets
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

VERSION = "jevarena-authored-v12-dev-editorial/1"
MISSING = "insufficient evidence"
WORD = re.compile(r"[A-Za-z0-9]+")
COMMON = (
    "The named members and units in this rule are exhaustive. Evidence order "
    "does not change their order or meaning. A later dated correction counts "
    "only when it explicitly supersedes an earlier named fact; use the latest "
    "such correction for that fact. If a required fact is absent or redacted, "
    "answer 'insufficient evidence'. Do not infer it from customary practice. "
)


@dataclass(frozen=True)
class Rule:
    ident: str
    kind: str
    fields: tuple[str, str]
    text: str
    vocabulary: tuple[str, ...] = ()


RULES = {
    rule.ident: rule
    for rule in (
        Rule(
            "provenance-hop",
            "choice",
            ("entry", "assignment"),
            "For tag R4, read its intermediate shelf from the entry card, then "
            "follow that shelf in the assignment ledger. Shelves are Iris, "
            "Juniper and Lotus; destinations are North, East and West. Every "
            "shelf has exactly one destination, and every destination appears "
            "once. Return that destination; there is no tie rule. The only "
            "outputs are North, East, West and insufficient evidence.",
            ("North", "East", "West"),
        ),
        Rule(
            "dual-index-code",
            "choice",
            ("flag", "pulse"),
            "Read one flag letter A, B or C and one pulse number 1, 2 or 3. "
            "Decode the pair by this complete row-major table: A1 Cedar, A2 "
            "Delta, A3 Echo; B1 Delta, B2 Echo, B3 Cedar; C1 Echo, C2 Cedar, "
            "C3 Delta. Return the decoded word. There is no tie rule. The only "
            "outputs are Cedar, Delta, Echo and insufficient evidence.",
            ("Cedar", "Delta", "Echo"),
        ),
        Rule(
            "exclusive-register",
            "choice",
            ("first_register", "second_register"),
            "The complete permit universe is Aster, Birch and Clover. Find "
            "permits appearing in exactly one of the two registers. Choose the "
            "first such permit in the fixed order Aster, Birch, Clover; if none "
            "appears exactly once, return hold. Each register lists its complete "
            "current entries. Outputs are Aster, Birch, Clover, hold and "
            "insufficient evidence.",
            ("Aster", "Birch", "Clover", "hold"),
        ),
        Rule(
            "single-swap",
            "choice",
            ("lineup", "swap"),
            "The three props Oak, Pine and Yew occupy stage positions 1, 2 and "
            "3, respectively as recorded in the lineup. Apply exactly one swap "
            "of two distinct position numbers from 1 through 3, then name the "
            "prop at position 1. The lineup is complete and each prop appears "
            "once. There is no tie rule. Outputs are Oak, Pine, Yew and "
            "insufficient evidence.",
            ("Oak", "Pine", "Yew"),
        ),
        Rule(
            "ceramic-checksum",
            "noul",
            ("batch_code", "seal_code"),
            "Both codes are single base-ten integers from 0 through 6. Compute "
            "three times the batch code plus twice the seal code, then take "
            "the remainder after division by seven. Return true exactly when "
            "the remainder is one; otherwise false. There is no tie. Outputs "
            "are true, false and insufficient evidence.",
        ),
        Rule(
            "burst-handshake",
            "noul",
            ("burst_second", "reply_second"),
            "Each timestamp is an integer second after the same 00:00 UTC "
            "boundary, from 0 through 59. Subtract the burst second from the "
            "reply second. Return true exactly when the reply occurs 1, 2 or "
            "3 seconds after the burst; zero, earlier or later replies are "
            "false. There is no tie. Outputs are true, false and insufficient "
            "evidence.",
        ),
        Rule(
            "pitch-drift",
            "score",
            ("reference_cents", "measured_cents"),
            "Both pitch offsets are signed integer cents on the same scale. "
            "Compute the absolute difference. Grade 4 for 0–1 cents, grade 3 "
            "for 2–3, grade 2 for 4–5, grade 1 for 6–8 and grade 0 for 9 or "
            "more. Boundary values are inclusive in the stated band. Outputs "
            "are grades 0 through 4 and insufficient evidence.",
        ),
        Rule(
            "curation-matrix",
            "score",
            ("provenance", "condition"),
            "Read one provenance class A, B or C and one condition stable, "
            "fragile or failed. Use this complete grade matrix in that column "
            "order: class A gives 4, 3, 1; class B gives 3, 2, 0; class C gives "
            "2, 1, 0. Do not reinterpret these as conventional quality bands. "
            "There is no tie. Outputs are grades 0 through 4 and insufficient "
            "evidence.",
        ),
    )
}
assert len(RULES) == 8


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()


def write(path: Path, value: Any) -> None:
    path.write_bytes(canonical(value))
    path.chmod(0o600)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_bytes(b"".join(canonical(row) for row in rows))
    path.chmod(0o600)


def words(value: str) -> list[str]:
    return [word.lower() for word in WORD.findall(value)]


def opaque(salt: bytes, label: str) -> str:
    return hashlib.sha256(salt + b"\0" + label.encode()).hexdigest()[:20]


def validate(rule: Rule, facts: dict[str, Any]) -> None:
    if set(facts) != set(rule.fields):
        raise ValueError("Facts do not match the invariant rule")
    if any(value is None for value in facts.values()):
        return
    a, b = (facts[field] for field in rule.fields)
    if rule.ident == "provenance-hop":
        if (
            a not in {"Iris", "Juniper", "Lotus"}
            or set(b) != {"Iris", "Juniper", "Lotus"}
            or set(b.values()) != {"North", "East", "West"}
        ):
            raise ValueError("Invalid provenance universe")
    elif rule.ident == "dual-index-code":
        if a not in "ABC" or type(b) is not int or b not in (1, 2, 3):
            raise ValueError("Invalid dual index")
    elif rule.ident == "exclusive-register":
        universe = {"Aster", "Birch", "Clover"}
        if any(
            type(v) is not list or len(v) != len(set(v)) or not set(v) <= universe
            for v in (a, b)
        ):
            raise ValueError("Invalid register")
    elif rule.ident == "single-swap":
        if (
            type(a) is not list
            or set(a) != {"Oak", "Pine", "Yew"}
            or len(a) != 3
            or type(b) is not list
            or len(b) != 2
            or set(b) not in ({1, 2}, {1, 3}, {2, 3})
        ):
            raise ValueError("Invalid swap")
    elif rule.ident == "ceramic-checksum":
        if any(type(v) is not int or v not in range(7) for v in (a, b)):
            raise ValueError("Invalid code")
    elif rule.ident == "burst-handshake":
        if any(type(v) is not int or v not in range(60) for v in (a, b)):
            raise ValueError("Invalid timestamp")
    elif rule.ident == "pitch-drift":
        if any(type(v) is not int or abs(v) > 1200 for v in (a, b)):
            raise ValueError("Invalid cent offset")
    elif rule.ident == "curation-matrix":
        if a not in {"A", "B", "C"} or b not in {"stable", "fragile", "failed"}:
            raise ValueError("Invalid matrix coordinates")


def evaluate(rule: Rule, facts: dict[str, Any]) -> str | bool | int:
    validate(rule, facts)
    if any(value is None for value in facts.values()):
        return MISSING
    a, b = (facts[field] for field in rule.fields)
    if rule.ident == "provenance-hop":
        return b[a]
    if rule.ident == "dual-index-code":
        return (
            ("Cedar", "Delta", "Echo"),
            ("Delta", "Echo", "Cedar"),
            ("Echo", "Cedar", "Delta"),
        )[ord(a) - ord("A")][b - 1]
    if rule.ident == "exclusive-register":
        unique = set(a) ^ set(b)
        return next(
            (name for name in ("Aster", "Birch", "Clover") if name in unique), "hold"
        )
    if rule.ident == "single-swap":
        positions = list(a)
        left, right = b
        positions[left - 1], positions[right - 1] = (
            positions[right - 1],
            positions[left - 1],
        )
        return positions[0]
    if rule.ident == "ceramic-checksum":
        return (3 * a + 2 * b) % 7 == 1
    if rule.ident == "burst-handshake":
        return 1 <= b - a <= 3
    if rule.ident == "pitch-drift":
        difference = abs(a - b)
        return (
            4
            if difference <= 1
            else (
                3
                if difference <= 3
                else 2 if difference <= 5 else 1 if difference <= 8 else 0
            )
        )
    if rule.ident == "curation-matrix":
        return {
            "A": {"stable": 4, "fragile": 3, "failed": 1},
            "B": {"stable": 3, "fragile": 2, "failed": 0},
            "C": {"stable": 2, "fragile": 1, "failed": 0},
        }[a][b]
    raise ValueError("Unknown rule")


def reference(rule: Rule, facts: dict[str, Any]) -> str | bool | int:
    """An independent enumeration of the fixed rule."""
    validate(rule, facts)
    if any(value is None for value in facts.values()):
        return MISSING
    values = tuple(facts[field] for field in rule.fields)
    if rule.ident == "provenance-hop":
        return next(
            destination
            for shelf, destination in values[1].items()
            if shelf == values[0]
        )
    if rule.ident == "dual-index-code":
        symbols = ("Cedar", "Delta", "Echo")
        return symbols[((ord(values[0]) - 65) + values[1] - 1) % 3]
    if rule.ident == "exclusive-register":
        for name in ("Aster", "Birch", "Clover"):
            if (name in values[0]) != (name in values[1]):
                return name
        return "hold"
    if rule.ident == "single-swap":
        return (
            values[0][values[1][1] - 1]
            if values[1][0] == 1
            else values[0][values[1][0] - 1] if values[1][1] == 1 else values[0][0]
        )
    if rule.ident == "ceramic-checksum":
        return (3 * values[0] + 2 * values[1] - 1) // 7 * 7 == 3 * values[
            0
        ] + 2 * values[1] - 1
    if rule.ident == "burst-handshake":
        return values[1] in {values[0] + 1, values[0] + 2, values[0] + 3}
    if rule.ident == "pitch-drift":
        thresholds = (1, 3, 5, 8)
        return sum(abs(values[0] - values[1]) <= threshold for threshold in thresholds)
    if rule.ident == "curation-matrix":
        rows = ((4, 3, 1), (3, 2, 0), (2, 1, 0))
        return rows[("A", "B", "C").index(values[0])][
            ("stable", "fragile", "failed").index(values[1])
        ]
    raise ValueError("Unknown rule")


def checked(rule: Rule, facts: dict[str, Any]) -> str | bool | int:
    answer = evaluate(rule, facts)
    other = reference(rule, facts)
    if type(answer) is not type(other) or answer != other:
        raise ValueError("Independent rule oracles disagree")
    return answer


def question(rule: Rule, options: list[str]) -> dict[str, Any]:
    if rule.kind == "choice":
        criteria: Any = {name: f"Select {name}" for name in options}
    elif rule.kind == "noul":
        criteria = {"true": "Condition met", "false": "Condition not met"}
    else:
        criteria = [f"Grade {grade}" for grade in range(5)]
    return {
        "decision": {
            "type": rule.kind,
            "instructions": "Apply the current rule to the evidence. If a required fact is unavailable, say insufficient evidence.",
            "criteria": criteria,
        }
    }


def surface(
    spec: dict[str, Any], rule: Rule, sources: list[str], options: list[str]
) -> dict[str, Any]:
    state = "\n\n".join(
        (spec["scene"].strip(), "CURRENT RULE: " + COMMON + rule.text, *sources)
    )
    return {"state": state, "questions": question(rule, options)}


def lexical_gate(specs: list[dict[str, Any]]) -> dict[str, Any]:
    bodies = [source["body"] for spec in specs for source in spec["sources"]]
    seen: set[tuple[str, ...]] = set()
    for body in bodies:
        tokens = words(body)
        if len(tokens) < 25 or len(tokens) > 110:
            raise ValueError("Evidence body outside useful length range")
        spans = set(zip(*(tokens[i:] for i in range(8))))
        if spans & seen:
            raise ValueError("Repeated eight-word source prose")
        seen |= spans
        if re.search(
            r"\b(?:grade|score|winner|answer|insufficient evidence)\b", body, re.I
        ):
            raise ValueError("Source contains a direct answer cue")
    return {"original_source_bodies": len(bodies), "unique_eight_word_spans": len(seen)}


def build(spec_path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("A frozen candidate cannot be overwritten")
    specs = json.loads(spec_path.read_text())
    if (
        len(specs) != 8
        or set(spec["operation_id"] for spec in specs) != set(RULES)
        or len({spec["domain"] for spec in specs}) != 8
        or len({spec["slug"] for spec in specs}) != 8
    ):
        raise ValueError("Eight distinct operations, domains and slugs required")
    lex = lexical_gate(specs)
    salt_a, salt_b = secrets.token_bytes(32), secrets.token_bytes(32)
    choice_slugs = sorted(
        (
            spec["slug"]
            for spec in specs
            if RULES[spec["operation_id"]].kind == "choice"
        ),
        key=lambda slug: opaque(salt_a, "choice-position:" + slug),
    )
    choice_position = {slug: position for position, slug in enumerate(choice_slugs)}
    originals: list[dict[str, Any]] = []
    variants: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    variant_targets: list[dict[str, Any]] = []
    withdrawals: list[dict[str, Any]] = []
    withdrawal_targets: list[dict[str, Any]] = []
    proofs: list[dict[str, Any]] = []
    joins: list[dict[str, str]] = []
    formats: set[str] = set()
    choice_positions: list[int] = []
    boolean_answers: list[bool] = []
    perturbations: Counter[str] = Counter()
    for spec in specs:
        rule = RULES[spec["operation_id"]]
        if len(words(spec["scene"])) < 20 or len(words(spec["scene"])) > 70:
            raise ValueError("Scene length obscures the native task")
        facts = spec["facts"]
        answer = checked(rule, facts)
        if (
            answer == MISSING
            or type(answer)
            is not {"choice": str, "noul": bool, "score": int}[rule.kind]
        ):
            raise ValueError("Original is not a typed answer")
        if type(answer) is not type(spec["expected"]) or answer != spec["expected"]:
            raise ValueError("Author and original oracle disagree")
        sources = spec["sources"]
        if len(sources) != 2 or {source["field"] for source in sources} != set(
            rule.fields
        ):
            raise ValueError("Each required field needs one independent source")
        rendered: list[str] = []
        source_proofs: list[dict[str, Any]] = []
        for source in sources:
            field = source["field"]
            formats.add(source["format"])
            if source["value"] != facts[field] or type(source["value"]) is not type(
                facts[field]
            ):
                raise ValueError("Source extraction differs from facts")
            alt = source["alternative"]
            if (
                alt["value"] == facts[field]
                or alt["format"] != source["format"]
                or abs(len(words(alt["body"])) - len(words(source["body"]))) > 12
            ):
                raise ValueError("Counterfactual substitution is not comparable")
            if source["quote"] not in source["body"] or alt["quote"] not in alt["body"]:
                raise ValueError("Extraction quote absent")
            substitute = {**facts, field: alt["value"]}
            alternate_answer = checked(rule, substitute)
            if alternate_answer == answer:
                raise ValueError("Source substitution does not change the answer")
            if (
                type(alternate_answer) is not type(alt["expected"])
                or alternate_answer != alt["expected"]
            ):
                raise ValueError("Author and substitution oracle disagree")
            missing_facts = {**facts, field: None}
            withdrawn = checked(rule, missing_facts)
            if withdrawn == answer:
                raise ValueError("Source withdrawal still forces original answer")
            source_proofs.append(
                {
                    "field": field,
                    "answer_after_substitution": alternate_answer,
                    "withdrawal_status": withdrawn,
                    "completion_answers": [answer, alternate_answer],
                    "original_body_words": len(words(source["body"])),
                    "alternative_body_words": len(words(alt["body"])),
                }
            )
            rendered.append(source["title"] + "\n" + source["body"])
        display_order = sorted(
            range(2),
            key=lambda index: opaque(
                salt_a, "source-order:" + spec["slug"] + ":" + sources[index]["field"]
            ),
        )
        sources = [sources[index] for index in display_order]
        rendered = [rendered[index] for index in display_order]
        options = list(rule.vocabulary)
        if rule.kind == "choice":
            if answer not in options:
                raise ValueError("Choice answer outside fixed universe")
            others = sorted(
                (name for name in options if name != answer),
                key=lambda name: opaque(salt_a, spec["slug"] + ":" + name),
            )
            position = choice_position[spec["slug"]]
            options = others[:position] + [answer] + others[position:]
            options.append(MISSING)
            choice_positions.append(position + 1)
        elif rule.kind == "noul":
            boolean_answers.append(answer)
        original = surface(spec, rule, rendered, options)
        aid = opaque(salt_a, "original:" + spec["slug"])
        original["id"] = aid
        originals.append(original)
        targets.append({"id": aid, "kind": rule.kind, "answer": {rule.kind: answer}})
        for index, source in enumerate(sources):
            withdrawal = surface(
                spec,
                rule,
                [
                    text
                    for source_index, text in enumerate(rendered)
                    if source_index != index
                ],
                options,
            )
            withdrawal_id = opaque(
                salt_a, "all-withdrawals:" + spec["slug"] + ":" + source["field"]
            )
            withdrawal["id"] = withdrawal_id
            if (
                source["quote"] in withdrawal["state"]
                or source["title"] in withdrawal["state"]
            ):
                raise ValueError("Withdrawn source survives in reduced state")
            withdrawals.append(withdrawal)
            source_proof = next(
                row for row in source_proofs if row["field"] == source["field"]
            )
            withdrawal_targets.append(
                {
                    "id": withdrawal_id,
                    "reported_status": MISSING,
                    "completion_answers": source_proof["completion_answers"],
                }
            )
        chosen = spec["variant"]
        if chosen["field"] not in rule.fields or chosen["kind"] not in {
            "withdrawal",
            "redaction",
            "correction",
        }:
            raise ValueError("Invalid variant selector")
        changed_index = next(
            i for i, source in enumerate(sources) if source["field"] == chosen["field"]
        )
        changed = sources[changed_index]
        kind = chosen["kind"]
        perturbations[kind] += 1
        variant_sources = list(rendered)
        if kind == "withdrawal":
            variant_sources.pop(changed_index)
            variant_answer = MISSING
        elif kind == "redaction":
            redacted = changed["redacted_body"]
            if "[redacted]" not in redacted.lower() or changed["quote"] in redacted:
                raise ValueError("Field redaction is incomplete")
            variant_sources[changed_index] = changed["title"] + "\n" + redacted
            variant_answer = MISSING
        else:
            correction = changed["correction"]
            if (
                changed["title"] not in correction["body"]
                or "supersedes" not in correction["body"].lower()
                or correction["quote"] not in correction["body"]
            ):
                raise ValueError("Correction lacks explicit precedence")
            variant_sources.append(correction["title"] + "\n" + correction["body"])
            variant_answer = checked(
                rule, {**facts, changed["field"]: changed["alternative"]["value"]}
            )
        if variant_answer == answer:
            raise ValueError("Variant retains the original answer")
        if (
            type(variant_answer) is not type(spec["expected_variant"])
            or variant_answer != spec["expected_variant"]
        ):
            raise ValueError("Author and variant oracle disagree")
        variant = surface(spec, rule, variant_sources, options)
        bid = opaque(salt_b, "variant:" + spec["slug"])
        variant["id"] = bid
        variants.append(variant)
        variant_targets.append({"id": bid, "kind": rule.kind, "answer": variant_answer})
        joins.append(
            {
                "original_id": aid,
                "variant_id": bid,
                "slug": spec["slug"],
                "perturbation": kind,
                "field": changed["field"],
            }
        )
        proofs.append(
            {
                "original_id": aid,
                "slug": spec["slug"],
                "domain": spec["domain"],
                "operation": rule.ident,
                "original_answer": answer,
                "variant_answer": variant_answer,
                "sources": source_proofs,
                "original_words": len(words(original["state"])),
                "variant_words": len(words(variant["state"])),
            }
        )
    if (
        Counter(row["kind"] for row in targets) != {"choice": 4, "noul": 2, "score": 2}
        or sorted(choice_positions) != [1, 2, 3, 4]
        or Counter(boolean_answers) != {True: 1, False: 1}
        or len(formats) < 5
        or perturbations != {"withdrawal": 3, "redaction": 3, "correction": 2}
    ):
        raise ValueError(
            "Preregistered type, position, form or perturbation balance failed"
        )
    originals.sort(key=lambda row: opaque(salt_a, "order:" + row["id"]))
    variants.sort(key=lambda row: opaque(salt_b, "order:" + row["id"]))
    output.mkdir(parents=True)
    private = output / "private"
    private.mkdir(mode=0o700)
    a_dir, b_dir = output / "reviewer-a", output / "reviewer-b"
    a_dir.mkdir(mode=0o700)
    b_dir.mkdir(mode=0o700)
    write_jsonl(a_dir / "originals.jsonl", originals)
    write_jsonl(b_dir / "variants.jsonl", variants)
    write_jsonl(private / "targets.jsonl", targets)
    write_jsonl(private / "variant_targets.jsonl", variant_targets)
    write_jsonl(private / "all_source_withdrawals.jsonl", withdrawals)
    write_jsonl(private / "withdrawal_targets.jsonl", withdrawal_targets)
    write_jsonl(private / "proofs.jsonl", proofs)
    write_jsonl(private / "join.jsonl", joins)
    (private / "salt-a.bin").write_bytes(salt_a)
    (private / "salt-b.bin").write_bytes(salt_b)
    for name in ("salt-a.bin", "salt-b.bin"):
        (private / name).chmod(0o600)
    manifest_a = {
        "version": VERSION,
        "role": "independent-original-reviewer",
        "count": len(originals),
        "file": "originals.jsonl",
        "sha256": sha((a_dir / "originals.jsonl").read_bytes()),
        "instructions": "Record a direct answer, ambiguity, unnatural wording and suspected shortcuts for each native task. Seal this review without opening any variant or private key.",
    }
    manifest_b = {
        "version": VERSION,
        "role": "independent-variant-reviewer",
        "count": len(variants),
        "file": "variants.jsonl",
        "sha256": sha((b_dir / "variants.jsonl").read_bytes()),
        "instructions": "Record a direct answer, ambiguity, unnatural wording and suspected shortcuts for each native task. Seal this review without opening originals or any private key.",
    }
    write(a_dir / "manifest.json", manifest_a)
    write(b_dir / "manifest.json", manifest_b)
    receipt = {
        "version": VERSION,
        "status": "FROZEN_DEV_EDITORIAL_AWAITING_INDEPENDENT_REVIEWS",
        "release_qualified": False,
        "training_admitted": False,
        "model_inference": False,
        "utc_sealed_at": datetime.now(timezone.utc).isoformat(),
        "source_commit": None,
        "prereg_sha256": "0e4412c0d977948271d780eb26cf7a55587786759e61b109ecf0f1db2ce2acd7",
        "spec_sha256": sha(spec_path.read_bytes()),
        "builder_sha256": sha(Path(__file__).read_bytes()),
        "originals_sha256": manifest_a["sha256"],
        "variants_sha256": manifest_b["sha256"],
        "manifest_a_sha256": sha((a_dir / "manifest.json").read_bytes()),
        "manifest_b_sha256": sha((b_dir / "manifest.json").read_bytes()),
        "targets_sha256": sha((private / "targets.jsonl").read_bytes()),
        "variant_targets_sha256": sha((private / "variant_targets.jsonl").read_bytes()),
        "withdrawals_sha256": sha(
            (private / "all_source_withdrawals.jsonl").read_bytes()
        ),
        "withdrawal_targets_sha256": sha(
            (private / "withdrawal_targets.jsonl").read_bytes()
        ),
        "proofs_sha256": sha((private / "proofs.jsonl").read_bytes()),
        "join_sha256": sha((private / "join.jsonl").read_bytes()),
        "salt_a_commitment": sha(salt_a),
        "salt_b_commitment": sha(salt_b),
        "counts": {
            "originals": len(originals),
            "variants": len(variants),
            "sources": len(proofs) * 2,
            "private_withdrawals": len(withdrawals),
        },
        "choice_positions": choice_positions,
        "boolean_answers": {
            str(key): value for key, value in Counter(boolean_answers).items()
        },
        "formats": sorted(formats),
        "perturbations": dict(perturbations),
        "lexical": lex,
    }
    write(private / "receipt.json", receipt)
    public = {
        key: value
        for key, value in receipt.items()
        if key
        not in {
            "targets_sha256",
            "variant_targets_sha256",
            "withdrawals_sha256",
            "withdrawal_targets_sha256",
            "proofs_sha256",
            "join_sha256",
            "salt_a_commitment",
            "salt_b_commitment",
            "spec_sha256",
        }
    }
    write(output / "freeze.gold-free.json", public)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = build(args.specs, args.output)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "originals": receipt["counts"]["originals"],
                "variants": receipt["counts"]["variants"],
                "originals_sha256": receipt["originals_sha256"],
                "variants_sha256": receipt["variants_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
