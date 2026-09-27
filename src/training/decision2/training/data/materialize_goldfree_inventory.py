"""Materialize a pinned, private input-only inventory for source admission.

The input manifest is an existing list of sealed evaluator prompt files. Only
the five native roles needed by the prospective core audit are included; other
roles require their own source-specific input attestation. TRAIN/SELECT/CAL
targets are parsed from the source JSONL but never selected for output.
This tool does not grant a data-rights or semantic-overlap PASS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from training.data.plan_goldfree_inventory import (
    NATIVE_ROLE_COUNTS,
    PARTITION_ROLE_COUNTS,
    project_native_file,
    project_partition_rows,
    projected_jsonl,
    validate_core_rows,
)
from training.model.data import file_sha256


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _pinned(path: Path, expected_sha256: str) -> None:
    if not path.is_file() or file_sha256(path) != expected_sha256:
        raise ValueError("Pinned source file missing or hash changed")


def _rows(path: Path) -> list[dict[str, Any]]:
    result = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("Partition JSONL row must be an object")
                result.append(row)
    return result


def project_core(
    source_manifest: Path,
    source_manifest_sha256: str,
    partitions: dict[str, tuple[Path, str]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Verify every source identity before projecting any protected input."""
    _pinned(source_manifest, source_manifest_sha256)
    entries = json.loads(source_manifest.read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        raise ValueError("Sealed prompt inventory must be a list")
    selected: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("role"), str):
            raise ValueError("Invalid sealed inventory entry")
        role = entry["role"]
        if role in seen:
            raise ValueError("Duplicate sealed inventory role")
        seen.add(role)
        if role in NATIVE_ROLE_COUNTS:
            if not isinstance(entry.get("path"), str) or not isinstance(
                entry.get("sha256"), str
            ):
                raise ValueError("Native role lacks pinned path or hash")
            selected[role] = entry
    if set(selected) != set(NATIVE_ROLE_COUNTS):
        raise ValueError("Five required native roles are incomplete")
    if set(partitions) != set(PARTITION_ROLE_COUNTS):
        raise ValueError("Three required partition roles are incomplete")

    roles: dict[str, list[dict[str, Any]]] = {}
    identity: dict[str, Any] = {
        "schema": "decision2-projected-core-inputs-v1",
        "source_manifest_sha256": source_manifest_sha256,
        "excluded_optional_role_count": len(seen - set(NATIVE_ROLE_COUNTS)),
        "roles": {},
    }
    for role, entry in sorted(selected.items()):
        source = Path(entry["path"])
        prompts, native_digests = project_native_file(source, role, entry["sha256"])
        roles[role] = prompts
        identity["roles"][role] = {
            "source_sha256": entry["sha256"],
            "native_input_digest_list_sha256": _sha(
                ("\n".join(native_digests) + "\n").encode("ascii")
            ),
        }
    for role, (path, expected_sha256) in sorted(partitions.items()):
        _pinned(path, expected_sha256)
        roles[role] = project_partition_rows(_rows(path), role)
        identity["roles"][role] = {"source_sha256": expected_sha256}
    counts = validate_core_rows(roles)
    for role, count in counts.items():
        identity["roles"][role]["rows"] = count
    return roles, identity


def write_private_inventory(
    output: Path, roles: dict[str, list[dict[str, Any]]], identity: dict[str, Any]
) -> Path:
    """Write a new all-or-nothing private directory; never overwrite a receipt."""
    validate_core_rows(roles)
    if output.exists():
        raise ValueError("Output directory already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    previous_umask = os.umask(0o077)
    try:
        with tempfile.TemporaryDirectory(
            prefix=".projected-core-", dir=output.parent
        ) as temp:
            directory = Path(temp)
            manifest = dict(identity)
            manifest["roles"] = {
                role: dict(record) for role, record in identity["roles"].items()
            }
            for role, prompts in sorted(roles.items()):
                payload = projected_jsonl(prompts)
                filename = f"{role}.jsonl"
                (directory / filename).write_bytes(payload)
                manifest["roles"][role].update(
                    {"path": filename, "sha256": _sha(payload)}
                )
            manifest_payload = (
                json.dumps(manifest, indent=2, sort_keys=True) + "\n"
            ).encode("utf-8")
            (directory / "manifest.json").write_bytes(manifest_payload)
            directory.rename(output)
    finally:
        os.umask(previous_umask)
    return output / "manifest.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--source-manifest-sha256", required=True)
    for role in PARTITION_ROLE_COUNTS:
        parser.add_argument(f"--{role.replace('_', '-')}", required=True, type=Path)
        parser.add_argument(f"--{role.replace('_', '-')}-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    partitions = {
        role: (
            getattr(args, role),
            getattr(args, f"{role}_sha256"),
        )
        for role in PARTITION_ROLE_COUNTS
    }
    roles, identity = project_core(
        args.source_manifest, args.source_manifest_sha256, partitions
    )
    manifest = write_private_inventory(args.output, roles, identity)
    # Keep private paths and prompt contents out of shell logs.
    print(
        json.dumps(
            {
                "status": "PROJECTED_CORE_CREATED",
                "roles": len(roles),
                "manifest_sha256": file_sha256(manifest),
            }
        )
    )


if __name__ == "__main__":
    main()
