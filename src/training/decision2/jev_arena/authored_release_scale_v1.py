"""Private-only first scale gate for a separately authored release pool.

The checked-in module contains generic logic only. Case prose, structured
facts, answers, joins and salts are supplied from private remote storage.
`prepare` creates no reviewer packet: it must pass an independent prompt-only
overlap and native-token audit before a later, separately authorized seal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import string
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jev_arena.authored_v13_pilot import OPERATIONS as V13_OPERATIONS
from jev_arena.authored_v13_pilot import solve as v13_solve

VERSION = "jevarena-authored-release-scale-v1/private-candidate-1"
EXTRA_OPERATIONS = {
    "coverage_cost": "choice",
    "eligibility_deadline": "choice",
    "quorum_veto": "noul",
    "custody_chain": "noul",
    "allocation_envelope": "noul",
    "risk_matrix": "score",
    "evidence_agreement": "score",
}
OPERATIONS = V13_OPERATIONS | EXTRA_OPERATIONS
WORD = re.compile(r"\w+", re.UNICODE)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path) -> str:
    return sha(path.read_bytes())


def write_private(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    )
    path.chmod(0o600)


def words(value: str) -> list[str]:
    return [word.lower() for word in WORD.findall(value)]


def solve(
    operation: str, left: Any, right: Any, params: dict[str, Any]
) -> str | bool | int:
    if operation in V13_OPERATIONS:
        return v13_solve(operation, left, right, params)
    if operation == "coverage_cost":
        needed = set(params["needed"])
        eligible = [key for key, covered in left.items() if needed <= set(covered)]
        return min(eligible, key=lambda key: (right[key], key)) if eligible else "HOLD"
    if operation == "eligibility_deadline":
        eligible = [
            key
            for key, record in left.items()
            if record["qualified"] and right[key] <= params["deadline"]
        ]
        return (
            min(eligible, key=lambda key: (right[key], -left[key]["priority"], key))
            if eligible
            else "HOLD"
        )
    if operation == "quorum_veto":
        signers = set(left["signers"])
        roster = set(right["roster"])
        return len(signers & roster) >= params["quorum"] and not signers.intersection(
            right["veto"]
        )
    if operation == "custody_chain":
        transfers = sorted(left, key=lambda item: item["time"])
        if not transfers or set(right) != {item["id"] for item in transfers}:
            raise ValueError("Custody acknowledgement universe mismatch")
        return all(
            right[item["id"]] == item["to"]
            and (index == 0 or transfers[index - 1]["to"] == item["from"])
            for index, item in enumerate(transfers)
        )
    if operation == "allocation_envelope":
        if set(left) != set(right) or not left:
            raise ValueError("Allocation source universe mismatch")
        if any(
            type(value) is not int or value < 0
            for value in [*left.values(), *right.values()]
        ):
            raise ValueError("Allocation quantities must be nonnegative integers")
        return all(left[key] >= right[key] for key in left) and (
            sum(left.values()) - sum(right.values()) <= params["maximum_surplus"]
        )
    if operation == "risk_matrix":
        if set(left) != set(right):
            raise ValueError("Risk matrix source universe mismatch")
        active = max(
            (
                left[key]["severity"] * right[key]["exposure"]
                for key in left
                if not right[key]["controlled"]
            ),
            default=0,
        )
        low, high = params["limits"]
        return 2 if active <= low else 1 if active <= high else 0
    if operation == "evidence_agreement":
        if set(left) != set(right):
            raise ValueError("Evidence source universe mismatch")
        weighted = sum(
            params["weights"][key]
            for key in left
            if left[key] == right[key] and left[key] == params["claim"]
        )
        low, high = params["limits"]
        return 2 if weighted >= high else 1 if weighted >= low else 0
    raise ValueError("Unregistered operation")


def render_source(source: dict[str, Any], data: Any | None = None) -> str:
    actual = source["data"] if data is None else data
    if not isinstance(actual, (dict, list)) or not actual:
        raise ValueError("Private source facts must be a nonempty map or list")
    template = source["document"]
    fields = [name for _, name, _, _ in string.Formatter().parse(template) if name]
    values = (
        actual
        if isinstance(actual, dict) and set(fields) != {"entries"}
        else {"entries": actual}
    )
    if not fields or set(fields) != set(values):
        raise ValueError("Every source fact must appear in its document")
    if any(not name.isidentifier() for name in fields):
        raise ValueError("Only simple document placeholders are allowed")
    printable = {
        key: (
            json.dumps(value, ensure_ascii=False)
            if isinstance(value, (dict, list))
            else value
        )
        for key, value in values.items()
    }
    rendered = template.format_map(printable)
    return f"{source['title']} [{source['form']}]\n{rendered}"


def native_row(
    case: dict[str, Any], left: Any, right: Any, row_id: str
) -> dict[str, Any]:
    sources = case["sources"]
    state = "\n\n".join(
        (
            case["scene"],
            "Decision contract: " + case["contract"],
            render_source(sources[0], left),
            render_source(sources[1], right),
        )
    )
    criteria = case["criteria"]
    kind = OPERATIONS[case["operation"]]
    if kind == "choice":
        criteria = {key: criteria[key] for key in case["option_order"]}
    return {
        "id": row_id,
        "state": state,
        "questions": {
            "decision": {
                "type": kind,
                "instructions": case["question"],
                "criteria": criteria,
            }
        },
    }


def check_answer(case: dict[str, Any], left: Any, right: Any) -> str | bool | int:
    answer = solve(case["operation"], left, right, case["params"])
    kind = OPERATIONS[case["operation"]]
    if kind == "choice" and (type(answer) is not str or answer not in case["criteria"]):
        raise ValueError("Choice answer outside native options")
    if kind == "noul" and type(answer) is not bool:
        raise ValueError("Noul answer is not Boolean")
    if kind == "score" and (
        type(answer) is not int or not 0 <= answer < len(case["criteria"])
    ):
        raise ValueError("Score answer outside ordered native bands")
    return answer


def _shape(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _shape(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_shape(item) for item in value]
    return type(value).__name__


def inspect(case: dict[str, Any]) -> dict[str, Any]:
    if case["operation"] not in OPERATIONS:
        raise ValueError("Unknown operation")
    kind = OPERATIONS[case["operation"]]
    if len(case["sources"]) != 2 or [s["side"] for s in case["sources"]] != [
        "left",
        "right",
    ]:
        raise ValueError("Exactly two ordered complete sources required")
    if any(len(words(case[key])) < 8 for key in ("scene", "contract", "question")):
        raise ValueError("Decision context, contract or question too terse")
    if kind == "choice" and (
        not isinstance(case["criteria"], dict)
        or set(case["option_order"]) != set(case["criteria"])
    ):
        raise ValueError("Choice criteria incomplete")
    if kind == "noul" and set(case["criteria"]) != {"true", "false"}:
        raise ValueError("Noul criteria incomplete")
    if kind == "score" and (
        not isinstance(case["criteria"], list) or len(case["criteria"]) != 3
    ):
        raise ValueError("Score rubric must contain three ordered bands")
    left, right = (source["data"] for source in case["sources"])
    original = check_answer(case, left, right)
    witnesses: dict[str, list[str | bool | int]] = {}
    for side in ("left", "right"):
        source = case["sources"][0 if side == "left" else 1]
        values = case["witnesses"][side]
        if (
            len(values) != 2
            or any(_shape(value) != _shape(source["data"]) for value in values)
            or any(value == source["data"] for value in values)
        ):
            raise ValueError("Incomplete or shape-changing source witness")
        answers = [
            (
                check_answer(case, value, right)
                if side == "left"
                else check_answer(case, left, value)
            )
            for value in values
        ]
        if answers[0] == answers[1]:
            raise ValueError("Source necessity witness has no answer contrast")
        witnesses[side] = answers
    variant = case["variant"]
    side = variant["side"]
    if side not in {"left", "right"}:
        raise ValueError("Unknown source substitution side")
    source = case["sources"][0 if side == "left" else 1]
    if (
        _shape(variant["data"]) != _shape(source["data"])
        or variant["data"] == source["data"]
    ):
        raise ValueError("Incomplete or unchanged source substitution")
    changed = (
        check_answer(case, variant["data"], right)
        if side == "left"
        else check_answer(case, left, variant["data"])
    )
    if changed == original:
        raise ValueError("Substitution does not change native answer")
    pair_left = variant["data"] if side == "left" else left
    pair_right = variant["data"] if side == "right" else right
    variant_witnesses: dict[str, list[str | bool | int]] = {}
    for witness_side in ("left", "right"):
        pair_source = pair_left if witness_side == "left" else pair_right
        values = case["variant_witnesses"][witness_side]
        if len(values) != 2 or any(
            _shape(value) != _shape(pair_source) or value == pair_source
            for value in values
        ):
            raise ValueError("Incomplete substituted-source witness")
        results = [
            (
                check_answer(case, value, pair_right)
                if witness_side == "left"
                else check_answer(case, pair_left, value)
            )
            for value in values
        ]
        if len(set(results)) < 2:
            raise ValueError("Source becomes unnecessary after substitution")
        variant_witnesses[witness_side] = results
    original_source = render_source(source)
    substituted_source = render_source(source, variant["data"])
    if abs(len(words(original_source)) - len(words(substituted_source))) > 20:
        raise ValueError("Substitution source length changes too much")
    if not all(
        case["provenance"].get(name)
        for name in ("origin", "rights", "redistribution", "source_family")
    ):
        raise ValueError("Missing source lineage or rights")
    return {
        "original": original,
        "variant": changed,
        "witness_answers": witnesses,
        "variant_witness_answers": variant_witnesses,
    }


def prepare(
    casebook: Path, output: Path, prereg: Path, source_commit: str
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("Private candidate cannot be overwritten")
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("Unsigned or unpinned source commit")
    source = json.loads(casebook.read_text())
    cases = source["cases"]
    if not 36 <= len(cases) <= 60 or len({c["slug"] for c in cases}) != len(cases):
        raise ValueError("First scale batch needs 36–60 unique original cases")
    kinds = Counter(OPERATIONS[c["operation"]] for c in cases)
    if any(kinds[kind] < 12 for kind in ("choice", "noul", "score")):
        raise ValueError("Need at least 12 cases per native family")
    if len({c["operation"] for c in cases}) < 12:
        raise ValueError("Need at least 12 semantic mechanisms")
    forms = {s["form"] for c in cases for s in c["sources"]}
    domains = {c["domain"] for c in cases}
    if len(forms) < 9 or len(domains) < 6:
        raise ValueError("Document-form or domain diversity below preregistered floor")
    originals: list[dict[str, Any]] = []
    variants: list[dict[str, Any]] = []
    proofs: list[dict[str, Any]] = []
    for case in cases:
        proof = inspect(case)
        left, right = (s["data"] for s in case["sources"])
        original = native_row(case, left, right, case["slug"])
        variant = case["variant"]
        changed_left = variant["data"] if variant["side"] == "left" else left
        changed_right = variant["data"] if variant["side"] == "right" else right
        paired = native_row(case, changed_left, changed_right, case["slug"] + ":pair")
        if (
            original["questions"] != paired["questions"]
            or original["state"].split("\n\n")[1] != paired["state"].split("\n\n")[1]
        ):
            raise ValueError("Native question or decision contract changed in pair")
        originals.append(original)
        variants.append(paired)
        proofs.append(
            {"slug": case["slug"], "type": OPERATIONS[case["operation"]], **proof}
        )
    output.mkdir(mode=0o700, parents=True)
    for name, rows in (
        ("originals.private.jsonl", originals),
        ("variants.private.jsonl", variants),
        ("proofs.private.jsonl", proofs),
    ):
        path = output / name
        path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in rows
            )
        )
        path.chmod(0o600)
    write_private(output / "casebook.private.json", source)
    receipt = {
        "version": VERSION,
        "status": "PRIVATE_CANDIDATE_PREFLIGHT_PENDING_NO_BLIND_PACKET",
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": source_commit,
        "prereg_sha256": file_sha(prereg),
        "builder_sha256": file_sha(Path(__file__)),
        "independent_original_candidates": len(cases),
        "paired_variants_not_independent": len(variants),
        "by_type": dict(kinds),
        "mechanisms": len({c["operation"] for c in cases}),
        "domains": len(domains),
        "document_forms": len(forms),
        "casebook_sha256": file_sha(output / "casebook.private.json"),
        "originals_sha256": file_sha(output / "originals.private.jsonl"),
        "variants_sha256": file_sha(output / "variants.private.jsonl"),
        "proofs_sha256": file_sha(output / "proofs.private.jsonl"),
        "model_inference": False,
        "release_qualified": False,
    }
    write_private(output / "candidate.private.json", receipt)
    return {
        key: value
        for key, value in receipt.items()
        if not key.endswith("sha256") or key in ("prereg_sha256", "builder_sha256")
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--casebook", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(args.casebook, args.output, args.prereg, args.source_commit),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
