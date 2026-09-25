#!/usr/bin/env python3
"""Validate framework evidence and write source-bound CI execution receipts."""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
from pathlib import Path

from ci_plan import digest

STATUSES = frozenset({"passed", "failed", "skipped"})
SHA256_LENGTH = 64


def collection_errors(evidence: dict, activity: str) -> list[str]:
    errors = []
    field = "cases" if "cases" in evidence else "checks"
    if field == "checks" and activity in {"test", "performance"}:
        return [
            "test/performance evidence must contain actual cases, not command checks"
        ]
    items = evidence.get(field)
    expected = evidence.get(f"expected_{field}")
    if not isinstance(items, list) or not items:
        return [f"{field} must be a nonempty executed inventory"]
    if (
        not isinstance(expected, list)
        or not expected
        or any(not isinstance(item, str) or not item for item in expected)
    ):
        return [f"expected_{field} must be a nonempty ID list"]
    if len(expected) != len(set(expected)):
        errors.append(f"expected_{field} contains duplicates")
    ids = []
    for item in items:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("id"), str)
            or not item["id"]
        ):
            errors.append(f"{field}: invalid case identity")
            continue
        ids.append(item["id"])
        if item.get("status") not in STATUSES:
            errors.append(f"{item['id']}: missing or invalid status")
        elif item["status"] != "passed":
            errors.append(f"required {item['id']}: {item['status']}")
    if len(ids) != len(set(ids)):
        errors.append(f"{field} contains duplicate execution IDs")
    missing = set(expected) - set(ids)
    if missing:
        errors.append(f"missing required {field}: {sorted(missing)}")
    if "discovered_cases" in evidence and set(evidence["discovered_cases"]) != set(ids):
        errors.append("executed cases differ from framework discovery inventory")
    return errors


def artifact_records(evidence: dict, *, environ: dict | None = None) -> list[dict]:
    records = list(evidence.get("artifacts", []))
    for name in ("CI_IMAGE_RECEIPTS", "CI_NATIVE_RECEIPTS"):
        value = (os.environ if environ is None else environ).get(name)
        if value:
            records.extend(json.loads(Path(value).read_text()))
    identities = {}
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("id"), str):
            raise ValueError("invalid consumed artifact identity")
        value = record.get("sha256", "")
        if len(value) != SHA256_LENGTH or any(
            c not in "0123456789abcdef" for c in value
        ):
            raise ValueError(f"invalid artifact digest: {record['id']}")
        identity = {"id": record["id"], "sha256": value}
        if record["id"] in identities and identities[record["id"]] != identity:
            raise ValueError(f"conflicting artifact identities: {record['id']}")
        identities[record["id"]] = identity
    return sorted(identities.values(), key=lambda record: record["id"])


def actual_platform() -> str:
    machine = {"x86_64": "amd64", "aarch64": "arm64", "arm64": "arm64"}.get(
        platform.machine(), platform.machine()
    )
    return f"{platform.system().lower()}/{machine}"


def execution_errors(
    verification: dict, evidence: dict, producer_platform: str
) -> list[str]:
    execution = verification.get("execution")
    if not execution:
        errors = (
            []
            if producer_platform == verification["platform"]
            else ["receipt producer ran on a different platform"]
        )
        if evidence.get("execution"):
            errors.append("undeclared emulated execution")
        return errors
    if (
        execution.get("mode") != "qemu-user"
        or execution.get("host_platform") != producer_platform
        or evidence.get("execution") != execution
        or verification["platform"] != "linux/riscv64"
        or verification["native"]
    ):
        return ["emulated target, producer platform or artifact contract differs"]
    return []


def make_receipt(
    verification: dict,
    evidence: dict,
    *,
    source_sha: str,
    execution_platform: str,
    environ: dict | None = None,
) -> dict:
    errors = collection_errors(evidence, verification["activity"])
    if source_sha != verification["source_sha"]:
        errors.append("executed source SHA differs from planned source")
    for key in ("runtime", "device", "platform"):
        if evidence.get(key) != verification[key]:
            errors.append(
                f"actual {key} {evidence.get(key)!r} differs from planned {verification[key]!r}"
            )
    errors.extend(execution_errors(verification, evidence, execution_platform))
    artifacts = artifact_records(evidence, environ=environ)
    required = {f"image:{image}" for image in verification["images"]}
    if verification["native"]:
        required.add("native:cpu")
    missing = required - {record["id"] for record in artifacts}
    if missing:
        errors.append(f"missing consumed artifact identities: {sorted(missing)}")
    if errors:
        raise ValueError("; ".join(errors))
    return {
        "schema_version": 1,
        "id": verification["id"],
        "source_sha": source_sha,
        "contract_sha256": verification["contract_sha256"],
        "inventory": verification["inventory"],
        "runtime": evidence["runtime"],
        "device": evidence["device"],
        "platform": evidence["platform"],
        **(
            {"execution": {**verification["execution"]}}
            if verification.get("execution")
            else {}
        ),
        "artifacts": artifacts,
        "evidence": evidence,
        "evidence_sha256": digest(evidence),
        "result": "success",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("record",))
    parser.add_argument("--verification", required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        verification = json.loads(args.verification)
        evidence = json.loads(args.evidence.read_text())
        source = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        receipt = make_receipt(
            verification,
            evidence,
            source_sha=source,
            execution_platform=actual_platform(),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(1, f"CI result rejected: {exc}\n")
    print(f"Recorded {receipt['id']}: complete required inventory passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
