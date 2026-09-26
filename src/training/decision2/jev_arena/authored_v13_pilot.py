"""Freeze a private, native-valid authored v13 DEV editorial pilot.

Case prose and structured facts are supplied from an untracked private
casebook. This module contains only the generic renderer, twelve independent
decision oracles, and mechanical gates. It performs no model inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import secrets
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
from itertools import combinations
from pathlib import Path
from typing import Any

VERSION = "jevarena-authored-v13-dev-editorial/1"
OPERATIONS = {
    "eligible_quote": "choice",
    "rank_aggregation": "choice",
    "window_overlap": "choice",
    "revision_chain": "choice",
    "net_range": "noul",
    "required_subset": "noul",
    "interval_exclusion": "noul",
    "latest_ack": "noul",
    "residual_risk": "score",
    "utilization_band": "score",
    "median_divergence": "score",
    "critical_path": "score",
}
FORMS = {
    "csv",
    "email",
    "field_log",
    "checklist",
    "ledger",
    "schedule",
    "ticket",
    "lab_sheet",
    "dependency_chart",
    "memo",
}
WORD = re.compile(r"[\w]+", re.UNICODE)


def canonical(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=False, separators=(",", ":"))
        + "\n"
    ).encode()


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_hash(path: Path) -> str:
    return digest(path.read_bytes())


def tokens(text: str) -> list[str]:
    return [part.lower() for part in WORD.findall(text)]


def opaque(salt: bytes, value: str) -> str:
    return hashlib.sha256(salt + b"\0" + value.encode()).hexdigest()[:24]


def write_json(path: Path, data: Any) -> None:
    path.write_bytes(canonical(data))
    path.chmod(0o600)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_bytes(b"".join(canonical(row) for row in rows))
    path.chmod(0o600)


def _check_interval(value: dict[str, int]) -> None:
    if (
        set(value) != {"start", "end"}
        or any(type(item) is not int for item in value.values())
        or not 0 <= value["start"] < value["end"] <= 1440
    ):
        raise ValueError("Invalid half-open minute interval")


def _overlap(a: dict[str, int], b: dict[str, int]) -> int:
    _check_interval(a)
    _check_interval(b)
    return max(0, min(a["end"], b["end"]) - max(a["start"], b["start"]))


def solve(
    operation: str, left: Any, right: Any, params: dict[str, Any]
) -> str | bool | int:
    """Derive one typed answer from two independently supplied full sources."""
    if operation not in OPERATIONS:
        raise ValueError("Unknown operation")
    if operation == "eligible_quote":
        options = params["priority"]
        required = params["required_capacity"]
        if set(left) != set(right) or set(left) != set(options):
            raise ValueError("Quote universe mismatch")
        eligible = [key for key in options if left[key] >= required]
        return (
            min(eligible, key=lambda key: (right[key], options.index(key)))
            if eligible
            else "HOLD"
        )
    if operation == "rank_aggregation":
        priority = params["priority"]
        if sorted(left) != sorted(right) or sorted(left) != sorted(priority):
            raise ValueError("Incomplete independent rankings")
        return min(
            priority,
            key=lambda key: (left.index(key) + right.index(key), priority.index(key)),
        )
    if operation == "window_overlap":
        priority = params["priority"]
        if set(right) != set(priority):
            raise ValueError("Incomplete candidate windows")
        spans = {key: _overlap(left, right[key]) for key in priority}
        best = max(spans.values())
        return next(key for key in priority if spans[key] == best) if best else "HOLD"
    if operation == "revision_chain":
        actions = params["actions"]
        if len(left) != 2 or sorted(item["rev"] for item in left) != [1, 2]:
            raise ValueError("Exactly two signed revisions required")
        if set(right) != {str(item["rev"]) for item in left}:
            raise ValueError("Approval ledger does not cover revisions")
        if any(item["action"] not in actions for item in left):
            raise ValueError("Unknown revision action")
        if any(
            status not in {"approved", "provisional", "rejected"}
            for status in right.values()
        ):
            raise ValueError("Unknown clearance status")
        if "rejected" in right.values():
            return "HOLD"
        if "provisional" in right.values():
            return "DEFER"
        first, second = sorted(left, key=lambda item: item["rev"])
        return first["action"] if first["action"] == second["action"] else "REVIEW"
    if operation == "net_range":
        value = left["gross"] - right["tare"]
        return params["minimum"] <= value <= params["maximum"]
    if operation == "required_subset":
        if len(left) != len(set(left)) or len(right) != len(set(right)):
            raise ValueError("Duplicate feature in complete register")
        return set(left) <= set(right)
    if operation == "interval_exclusion":
        _check_interval(left)
        return all(_overlap(left, blocked) == 0 for blocked in right)
    if operation == "latest_ack":
        if not left or len({event["id"] for event in left}) != len(left):
            raise ValueError("Missing or repeated event id")
        if set(right) != {event["id"] for event in left}:
            raise ValueError("Acknowledgement ledger incomplete")
        latest = max(left, key=lambda item: (item["time"], item["id"]))
        elapsed = right[latest["id"]] - latest["time"]
        return 0 <= elapsed <= params["maximum_minutes"]
    if operation == "residual_risk":
        if set(left) != set(right) or any(value <= 0 for value in left.values()):
            raise ValueError("Risk ledger mismatch")
        if any(type(value) is not bool for value in right.values()):
            raise ValueError("Mitigation flag must be Boolean")
        points = sum(weight for key, weight in left.items() if not right[key])
        low, high = params["limits"]
        return 2 if points <= low else 1 if points <= high else 0
    if operation == "utilization_band":
        capacity, demand = left["capacity"], right["demand"]
        if (
            type(capacity) is not int
            or capacity <= 0
            or type(demand) is not int
            or demand < 0
        ):
            raise ValueError("Invalid capacity or demand")
        low, high = params["ratio_limits"]
        low_limit, high_limit = Fraction(str(low)), Fraction(str(high))
        if not 0 <= low_limit < high_limit or high_limit > 1:
            raise ValueError("Invalid ordered ratio limits")
        return (
            2
            if demand <= low_limit * capacity
            else 1 if demand <= high_limit * capacity else 0
        )
    if operation == "median_divergence":
        if any(
            len(row) != 3 or any(type(v) is not int for v in row)
            for row in (left, right)
        ):
            raise ValueError("Each laboratory needs three integer readings")
        difference = abs(sorted(left)[1] - sorted(right)[1])
        low, high = params["limits"]
        return 2 if difference <= low else 1 if difference <= high else 0
    if operation == "critical_path":
        names = set(left)
        if len(names) < 3 or any(
            type(value) is not int or value <= 0 for value in left.values()
        ):
            raise ValueError("Invalid task durations")
        if any(
            len(edge) != 2
            or edge[0] not in names
            or edge[1] not in names
            or edge[0] == edge[1]
            for edge in right
        ):
            raise ValueError("Invalid dependency")
        if len({tuple(edge) for edge in right}) != len(right):
            raise ValueError("Repeated dependency")
        finish: dict[str, int] = {}
        waiting = set(names)
        while waiting:
            ready = [
                name
                for name in waiting
                if all(parent in finish for parent, child in right if child == name)
            ]
            if not ready:
                raise ValueError("Cyclic dependency")
            for name in ready:
                finish[name] = left[name] + max(
                    (finish[parent] for parent, child in right if child == name),
                    default=0,
                )
                waiting.remove(name)
        duration = max(finish.values())
        low, high = params["limits"]
        return 2 if duration <= low else 1 if duration <= high else 0
    raise AssertionError("Unreachable")


def _display(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, dict):
        return "; ".join(f"{key}={_display(item)}" for key, item in value.items())
    if isinstance(value, list):
        return ", ".join(_display(item) for item in value)
    return str(value)


def render_source(source: dict[str, Any], data: Any | None = None) -> str:
    """Render a complete source from facts, without a parallel hand-written body."""
    form = source["form"]
    if form not in FORMS:
        raise ValueError("Unregistered evidence form")
    value = source["data"] if data is None else data
    if isinstance(value, dict):
        entries = list(value.items())
    elif isinstance(value, list):
        entries = [(f"entry {index + 1}", item) for index, item in enumerate(value)]
    else:
        raise ValueError("Evidence must be a complete map or list")
    if not entries:
        raise ValueError("Empty evidence source")
    title = source["title"]
    if form == "csv":
        body = "record,value\n" + "\n".join(
            f"{key},{_display(item)}" for key, item in entries
        )
    elif form in {"ledger", "schedule", "lab_sheet", "dependency_chart"}:
        body = "record | recorded value\n" + "\n".join(
            f"{key} | {_display(item)}" for key, item in entries
        )
    elif form == "checklist":
        body = "\n".join(f"[recorded] {key}: {_display(item)}" for key, item in entries)
    elif form == "email":
        body = (
            "Subject: "
            + title
            + "\n"
            + "\n".join(f"{key}: {_display(item)}" for key, item in entries)
        )
    elif form == "ticket":
        body = "Ticket fields\n" + "\n".join(
            f"{key} = {_display(item)}" for key, item in entries
        )
    elif form == "field_log":
        body = "Observed entries\n" + "\n".join(
            f"{key} — {_display(item)}" for key, item in entries
        )
    else:
        body = "\n".join(f"{key}: {_display(item)}" for key, item in entries)
    return f"{title} [{form}]\n{body}"


def render_case(
    case: dict[str, Any], sources: list[dict[str, Any]], row_id: str
) -> dict[str, Any]:
    kind = OPERATIONS[case["operation"]]
    criteria = case["criteria"]
    if kind == "choice":
        order = case["option_order"]
        if len(order) != len(set(order)) or set(order) != set(criteria):
            raise ValueError("Choice order does not match the native options")
        criteria = {key: criteria[key] for key in order}
    q = {"type": kind, "instructions": case["question"], "criteria": criteria}
    if kind == "choice" and not isinstance(q["criteria"], dict):
        raise ValueError("Choice requires a criterion map")
    if kind == "noul" and set(q["criteria"]) != {"true", "false"}:
        raise ValueError("Noul requires both Boolean criteria")
    if kind == "score" and (
        not isinstance(q["criteria"], list) or len(q["criteria"]) != 3
    ):
        raise ValueError("Score requires three ordered native bands")
    return {
        "id": row_id,
        "state": "\n\n".join(
            [
                case["scene"],
                "DECISION CONTRACT: " + case["contract"],
                *(render_source(s) for s in sources),
            ]
        ),
        "questions": {"decision": q},
    }


def _answer(case: dict[str, Any], left: Any, right: Any) -> str | bool | int:
    answer = solve(case["operation"], left, right, case["params"])
    kind = OPERATIONS[case["operation"]]
    expected_type = {"choice": str, "noul": bool, "score": int}[kind]
    if type(answer) is not expected_type:
        raise ValueError("Oracle returned wrong native type")
    if kind == "choice" and answer not in case["criteria"]:
        raise ValueError("Choice answer outside criteria")
    if kind == "score" and answer not in range(len(case["criteria"])):
        raise ValueError("Score answer outside ordered bands")
    return answer


def _shape(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _shape(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_shape(item) for item in value]
    return type(value).__name__


def validate_case(case: dict[str, Any]) -> dict[str, Any]:
    if len(case["sources"]) != 2 or {s["side"] for s in case["sources"]} != {
        "left",
        "right",
    }:
        raise ValueError("Exactly two separate sources required")
    if any(len(tokens(case[key])) < 8 for key in ("scene", "contract", "question")):
        raise ValueError("Decision context or contract too abbreviated")
    left, right = (
        next(s for s in case["sources"] if s["side"] == side)["data"]
        for side in ("left", "right")
    )
    original = _answer(case, left, right)
    alternatives: dict[str, list[str | bool | int]] = {}
    for source in case["sources"]:
        side = source["side"]
        witnesses = case["witnesses"][side]
        if len(witnesses) != 2 or any(value == source["data"] for value in witnesses):
            raise ValueError(
                "Two non-original alternative completions per source required"
            )
        if any(_shape(value) != _shape(source["data"]) for value in witnesses):
            raise ValueError("Alternative source changes format/shape")
        answers = [
            (
                _answer(case, value, right)
                if side == "left"
                else _answer(case, left, value)
            )
            for value in witnesses
        ]
        if answers[0] == answers[1]:
            raise ValueError(
                "Alternative completions do not demonstrate source necessity"
            )
        alternatives[side] = answers
    choice = case["variant"]
    if choice["side"] not in {"left", "right"}:
        raise ValueError("Variant must substitute one source")
    selected = next(s for s in case["sources"] if s["side"] == choice["side"])
    if (
        _shape(choice["data"]) != _shape(selected["data"])
        or choice["data"] == selected["data"]
    ):
        raise ValueError("Variant source is incomplete or unchanged")
    variant = (
        _answer(case, choice["data"], right)
        if choice["side"] == "left"
        else _answer(case, left, choice["data"])
    )
    if variant == original:
        raise ValueError("Source substitution must change native answer")
    variant_facts = {"left": left, "right": right, choice["side"]: choice["data"]}
    variant_necessity: dict[str, list[str | bool | int]] = {}
    for side in ("left", "right"):
        available = case.get("variant_witnesses", {}).get(
            side,
            [
                next(s for s in case["sources"] if s["side"] == side)["data"],
                *case["witnesses"][side],
            ],
        )
        completions = [variant_facts[side]]
        completions.extend(value for value in available if value != variant_facts[side])
        if len(completions) < 2 or any(
            _shape(value) != _shape(variant_facts[side]) for value in completions
        ):
            raise ValueError("Variant witness changes source shape")
        values = [
            (
                _answer(case, value, variant_facts["right"])
                if side == "left"
                else _answer(case, variant_facts["left"], value)
            )
            for value in completions
        ]
        pair = next(
            (
                (values[i], values[j])
                for i, j in combinations(range(len(values)), 2)
                if values[i] != values[j]
            ),
            None,
        )
        if pair is None:
            raise ValueError("Source becomes unnecessary in the substituted case")
        variant_necessity[side] = list(pair)
    old, new = render_source(selected), render_source(selected, choice["data"])
    if abs(len(tokens(old)) - len(tokens(new))) > 12 or len(tokens(new)) > 1.25 * len(
        tokens(old)
    ):
        raise ValueError("Substitution source differs materially in length")
    return {
        "original": original,
        "variant": variant,
        "witness_answers": alternatives,
        "variant_witness_answers": variant_necessity,
    }


def build(
    casebook_path: Path,
    output: Path,
    prereg_path: Path,
    scorer_path: Path,
    source_commit: str,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("A sealed candidate cannot be overwritten")
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("A signed source commit must be pinned")
    casebook = json.loads(casebook_path.read_text())
    cases = casebook["cases"]
    kinds = Counter(OPERATIONS[case["operation"]] for case in cases)
    if not 12 <= len(cases) <= 18 or len(cases) != len(
        {case["operation"] for case in cases}
    ):
        raise ValueError("Need 12–18 distinct independent operations")
    if len(set(case["slug"] for case in cases)) != len(cases) or len(
        set(case["domain"] for case in cases)
    ) != len(cases):
        raise ValueError("Case identifiers and domains must be unique")
    if max(kinds.values()) - min(kinds.values()) > 0 or set(kinds) != {
        "choice",
        "noul",
        "score",
    }:
        raise ValueError("Choice/Noul/Score must be balanced")
    formats = {s["form"] for case in cases for s in case["sources"]}
    if len(formats) < 6:
        raise ValueError("At least six source forms required")
    if any(
        not all(
            case.get("provenance", {}).get(field)
            for field in ("origin", "rights", "redistribution")
        )
        for case in cases
    ):
        raise ValueError("Each private case needs rights and source provenance")
    salt_a, salt_b = secrets.token_bytes(32), secrets.token_bytes(32)
    originals, variants, targets_a, targets_b, proofs, joins = ([] for _ in range(6))
    all_source_bodies: list[str] = []
    choice_positions: list[int] = []
    for case in cases:
        proof = validate_case(case)
        original_sources = list(case["sources"])
        random.Random(int(opaque(salt_a, "source:" + case["slug"]), 16)).shuffle(
            original_sources
        )
        aid = opaque(salt_a, "original:" + case["slug"])
        bid = opaque(salt_b, "variant:" + case["slug"])
        original = render_case(case, original_sources, aid)
        changed_sources = [dict(s) for s in original_sources]
        changed = next(
            s for s in changed_sources if s["side"] == case["variant"]["side"]
        )
        changed["data"] = case["variant"]["data"]
        variant = render_case(case, changed_sources, bid)
        if original["questions"] != variant["questions"]:
            raise ValueError("Native criterion or contract changed under substitution")
        if (
            not 60 <= len(tokens(original["state"])) <= 900
            or len(tokens(variant["state"])) > 900
        ):
            raise ValueError("Prompt too short or potentially truncated")
        all_source_bodies.extend(render_source(s) for s in original_sources)
        originals.append(original)
        variants.append(variant)
        kind = OPERATIONS[case["operation"]]
        if kind == "choice":
            choice_positions.append(
                list(original["questions"]["decision"]["criteria"]).index(
                    proof["original"]
                )
                + 1
            )
        targets_a.append({"id": aid, "kind": kind, "answer": proof["original"]})
        targets_b.append({"id": bid, "kind": kind, "answer": proof["variant"]})
        proofs.append({"slug": case["slug"], "operation": case["operation"], **proof})
        joins.append(
            {
                "slug": case["slug"],
                "original_id": aid,
                "variant_id": bid,
                "changed_side": case["variant"]["side"],
            }
        )
    seen_spans: set[tuple[str, ...]] = set()
    for body in all_source_bodies:
        parts = tokens(body)
        if len(parts) < 5:
            raise ValueError("Source is too terse")
        spans = set(zip(*(parts[index:] for index in range(8))))
        if seen_spans & spans:
            raise ValueError("Repeated eight-word span in source bodies")
        seen_spans.update(spans)
    if sorted(choice_positions) != list(range(1, kinds["choice"] + 1)):
        raise ValueError("Original Choice answer positions are not rotated")
    originals.sort(key=lambda row: opaque(salt_a, "order:" + row["id"]))
    variants.sort(key=lambda row: opaque(salt_b, "order:" + row["id"]))
    output.mkdir(mode=0o700, parents=True)
    private, a_dir, b_dir = (
        output / name for name in ("private", "reviewer-a", "reviewer-b")
    )
    for directory in (private, a_dir, b_dir):
        directory.mkdir(mode=0o700)
    write_jsonl(a_dir / "packet.jsonl", originals)
    write_jsonl(b_dir / "packet.jsonl", variants)
    write_jsonl(private / "targets-a.jsonl", targets_a)
    write_jsonl(private / "targets-b.jsonl", targets_b)
    write_jsonl(private / "proofs.jsonl", proofs)
    write_jsonl(private / "join.jsonl", joins)
    write_json(private / "casebook.json", casebook)
    (private / "salt-a.bin").write_bytes(salt_a)
    (private / "salt-b.bin").write_bytes(salt_b)
    for name in ("salt-a.bin", "salt-b.bin"):
        (private / name).chmod(0o600)
    for role, directory, rows in (
        ("originals", a_dir, originals),
        ("substitutions", b_dir, variants),
    ):
        write_json(
            directory / "manifest.json",
            {
                "version": VERSION,
                "role": role,
                "count": len(rows),
                "packet_sha256": file_hash(directory / "packet.jsonl"),
                "review_instruction": "Solve each native question from this packet alone; record ambiguity, realism, source necessity and shortcuts before sealing. Do not access the other packet or private key.",
            },
        )
    receipt = {
        "version": VERSION,
        "status": "FROZEN_DEV_EDITORIAL_AWAITING_INDEPENDENT_REVIEWS",
        "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
        "independent_originals": len(cases),
        "paired_substitutions": len(cases),
        "by_type": dict(kinds),
        "source_forms": sorted(formats),
        "choice_original_positions_sorted": sorted(choice_positions),
        "source_commit": source_commit,
        "prereg_sha256": file_hash(prereg_path),
        "builder_sha256": file_hash(Path(__file__)),
        "native_scorer_sha256": file_hash(scorer_path),
        "native_adapter": "direct_typed_review_v1; no model-specific adapter or inference",
        "casebook_sha256": file_hash(private / "casebook.json"),
        "proofs_sha256": file_hash(private / "proofs.jsonl"),
        "join_sha256": file_hash(private / "join.jsonl"),
        "targets_a_sha256": file_hash(private / "targets-a.jsonl"),
        "targets_b_sha256": file_hash(private / "targets-b.jsonl"),
        "original_packet_sha256": file_hash(a_dir / "packet.jsonl"),
        "variant_packet_sha256": file_hash(b_dir / "packet.jsonl"),
        "original_manifest_sha256": file_hash(a_dir / "manifest.json"),
        "variant_manifest_sha256": file_hash(b_dir / "manifest.json"),
        "source_eight_word_spans": len(seen_spans),
        "model_inference": False,
        "training_admitted": False,
        "release_qualified": False,
    }
    write_json(private / "freeze.json", receipt)
    public = {
        key: value
        for key, value in receipt.items()
        if key
        not in {
            "casebook_sha256",
            "proofs_sha256",
            "join_sha256",
            "targets_a_sha256",
            "targets_b_sha256",
        }
    }
    write_json(output / "freeze.gold-free.json", public)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--casebook", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--scorer", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    receipt = build(
        args.casebook, args.output, args.prereg, args.scorer, args.source_commit
    )
    print(
        json.dumps(
            {
                key: receipt[key]
                for key in (
                    "status",
                    "independent_originals",
                    "paired_substitutions",
                    "original_packet_sha256",
                    "variant_packet_sha256",
                )
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
