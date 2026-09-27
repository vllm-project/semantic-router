"""CPU-only source registry for the prospective 27B Score A/B data gate.

This checks custody, source bytes, group independence and the frozen quotas.
It deliberately never returns an ADMIT verdict: rendered-oracle, overlap,
native-token and independent blind-review gates must run separately. Private
case and document files must remain outside the repository.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

SCHEMA = "decision2-score27-source-registry/1"
MECHANISMS = (
    "dependency_readiness",
    "timed_feasibility",
    "stock_uncertainty",
    "scoped_policy_exception",
    "multi_source_attestation",
    "state_reconciliation",
)
QUOTAS = dict(zip(MECHANISMS, (14, 14, 13, 13, 13, 13), strict=True))
ROLES = ("train", "select")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def file_sha256(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def _private_file(path: Path, expected_sha: str) -> None:
    if not SHA256.fullmatch(expected_sha):
        raise ValueError("A source digest is malformed")
    if not path.is_file() or path.stat().st_mode & 0o077:
        raise ValueError("A private source file is absent or readable by others")
    if file_sha256(path) != expected_sha:
        raise ValueError("A private source digest differs from its sealed value")


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing {name}")
    return value


def _document(entry: dict[str, Any]) -> tuple[str, str]:
    source_id = _text(entry.get("source_id"), "source_id")
    _text(entry.get("rights_id"), "rights_id")
    origin = entry.get("origin")
    if origin not in {"authored", "upstream"}:
        raise ValueError("A source must be authored or upstream")
    if origin == "upstream":
        _text(entry.get("source_uri"), "source_uri")
        _text(entry.get("source_revision"), "source_revision")
    path = Path(_text(entry.get("path"), "source path"))
    _private_file(path, _text(entry.get("sha256"), "source sha256"))
    if not path.read_bytes().strip():
        raise ValueError("A source document is empty")
    return source_id, entry["sha256"]


def _case(entry: dict[str, Any]) -> dict[str, Any]:
    case_id = _text(entry.get("case_id"), "case_id")
    role = entry.get("role")
    mechanism = entry.get("mechanism")
    if role not in ROLES or mechanism not in MECHANISMS:
        raise ValueError(f"Invalid role or mechanism in {case_id}")
    author = _text(entry.get("author_id"), "author_id")
    family = _text(entry.get("source_family_id"), "source_family_id")
    documents = entry.get("documents")
    if not isinstance(documents, list) or len(documents) < 2:
        raise ValueError(f"At least two source documents are required: {case_id}")
    sources = [_document(item) for item in documents]
    if len({source_id for source_id, _ in sources}) != len(sources):
        raise ValueError(f"A source is duplicated within {case_id}")
    variants = entry.get("variants")
    if not isinstance(variants, list) or len(variants) != 3:
        raise ValueError(f"A complete three-level case is required: {case_id}")
    if sorted(row.get("label") for row in variants) != [0, 1, 2]:
        raise ValueError(f"A case must contain exactly levels 0/1/2: {case_id}")
    inputs = set()
    for variant in variants:
        if variant.get("task_type") != "score" or variant.get("group_id") != case_id:
            raise ValueError(f"Variant task or group changed: {case_id}")
        state = _text(variant.get("state"), "state")
        instructions = _text(variant.get("instructions"), "instructions")
        options = variant.get("options")
        if not isinstance(options, list) or len(options) != 3:
            raise ValueError(f"A Score variant needs three native options: {case_id}")
        if len({_text(option.get("key"), "option key") for option in options}) != 3:
            raise ValueError(f"Score option keys collide: {case_id}")
        for option in options:
            _text(option.get("description"), "option description")
        fingerprint = hashlib.sha256(
            json.dumps(
                [state, instructions, options], ensure_ascii=False, sort_keys=True
            ).encode("utf-8")
        ).hexdigest()
        if fingerprint in inputs:
            raise ValueError(f"Two Score levels have identical native input: {case_id}")
        inputs.add(fingerprint)
        if (
            not isinstance(variant.get("structured_facts"), dict)
            or not variant["structured_facts"]
        ):
            raise ValueError(f"Structured oracle facts are absent: {case_id}")
    return {
        "case_id": case_id,
        "role": role,
        "mechanism": mechanism,
        "author": author,
        "family": family,
        "source_ids": [source_id for source_id, _ in sources],
        "source_sha256": [sha for _, sha in sources],
    }


def audit_registry(registry: list[dict[str, Any]]) -> dict[str, Any]:
    """Verify prospective case metadata; never certify data admission."""
    cases = []
    for item in registry:
        path = Path(_text(item.get("path"), "case path"))
        _private_file(path, _text(item.get("sha256"), "case sha256"))
        case = json.loads(path.read_text(encoding="utf-8"))
        cases.append(_case(case))
    ids = [item["case_id"] for item in cases]
    families = [item["family"] for item in cases]
    source_ids = [source for item in cases for source in item["source_ids"]]
    source_sha256 = [sha for item in cases for sha in item["source_sha256"]]
    if len(set(ids)) != len(ids) or len(set(families)) != len(families):
        raise ValueError("Case or source-family identities repeat")
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("A source document recurs across independent cases")
    if len(set(source_sha256)) != len(source_sha256):
        raise ValueError("Identical source bytes recur across independent cases")
    authors = {
        role: {c["author"] for c in cases if c["role"] == role} for role in ROLES
    }
    if authors["train"] & authors["select"]:
        raise ValueError("TRAIN and fresh SELECT must have distinct authors")
    by_role = {
        role: collections.Counter(c["mechanism"] for c in cases if c["role"] == role)
        for role in ROLES
    }
    if any(
        by_role[role][name] > quota for role in ROLES for name, quota in QUOTAS.items()
    ):
        raise ValueError("A mechanism exceeds its frozen role quota")
    missing = {
        role: {name: quota - by_role[role][name] for name, quota in QUOTAS.items()}
        for role in ROLES
    }
    return {
        "schema_version": SCHEMA,
        "status": (
            "HOLD_CANDIDATE_INCOMPLETE"
            if any(number for counts in missing.values() for number in counts.values())
            else "HOLD_PENDING_ORACLE_OVERLAP_TOKENS_AND_BLIND_REVIEW"
        ),
        "case_groups": len(cases),
        "rows": len(cases) * 3,
        "by_role_mechanism": {
            role: dict(sorted(counts.items())) for role, counts in by_role.items()
        },
        "missing_groups": missing,
        "source_documents": len(source_ids),
        "limitations": [
            "Source and case digests alone cannot establish realism or source necessity",
            "Different author IDs do not prove selector-author blinding",
            "Rendered oracles, protected overlap, native token budget and blinded independent review remain required",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("The audit receipt must not overwrite an earlier version")
    if args.registry.stat().st_mode & 0o077:
        raise PermissionError("The private case registry is readable by others")
    registry = [
        json.loads(line) for line in args.registry.read_text().splitlines() if line
    ]
    report = audit_registry(registry)
    report["registry_sha256"] = file_sha256(args.registry)
    descriptor = os.open(args.output, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        json.dump(report, output, sort_keys=True, indent=2)
        output.write("\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "case_groups": report["case_groups"],
                "missing_groups": report["missing_groups"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
