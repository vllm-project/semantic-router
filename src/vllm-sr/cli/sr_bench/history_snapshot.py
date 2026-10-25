"""Immutable, content-addressed artifacts for finite named-history exclusions."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path

from .canonical import canonical, digest
from .dataset_io import MAX_ROWS, read_small
from .task_identity import POLICY, SHA256, source_identity

SCHEMA = "sr-bench-history-exclusions-v1"
RESERVATION = "all-referenced-memberships-v1"
MAX_REFERENCES = 64
MAX_SNAPSHOT_BYTES = 16 * 1024 * 1024
MAX_RUN_BYTES = 128 * 1024 * 1024
DATASET_ID = re.compile(r"[0-9a-f]{64}\Z")
RUN_ID = re.compile(r"run-[0-9a-f]{20}\Z")


def validate_snapshot(snapshot):
    fields = {
        "schema",
        "policy",
        "identity_policy",
        "coverage",
        "references",
        "families",
        "id",
    }
    if not isinstance(snapshot, dict) or set(snapshot) != fields:
        raise ValueError("Invalid history exclusion snapshot fields")
    content = {key: value for key, value in snapshot.items() if key != "id"}
    if (
        snapshot["schema"] != SCHEMA
        or snapshot["policy"] != RESERVATION
        or snapshot["identity_policy"] != POLICY
        or snapshot["coverage"] != "named-memberships-only"
        or snapshot["id"] != digest(content)
    ):
        raise ValueError("History exclusion snapshot identity or policy changed")
    refs = snapshot["references"]
    if not isinstance(refs, list) or not 1 <= len(refs) <= MAX_REFERENCES:
        raise ValueError("History snapshot requires 1 to 64 named references")
    if refs != sorted(refs, key=canonical) or len(
        {canonical(ref) for ref in refs}
    ) != len(refs):
        raise ValueError("History references must be sorted and unique")
    named = set()
    for ref in refs:
        keys = {"kind", "id", "manifest_sha256", "case_sha256"}
        if not isinstance(ref, dict) or set(ref) != keys:
            raise ValueError("Invalid frozen history reference")
        pattern = DATASET_ID if ref["kind"] == "dataset" else RUN_ID
        if (
            ref["kind"] not in ("dataset", "run")
            or not isinstance(ref["id"], str)
            or not pattern.fullmatch(ref["id"])
        ):
            raise ValueError("Invalid history reference identity")
        if any(
            not isinstance(ref[key], str) or not SHA256.fullmatch(ref[key])
            for key in ("manifest_sha256", "case_sha256")
        ):
            raise ValueError("Invalid history reference digest")
        if (ref["kind"], ref["id"]) in named:
            raise ValueError("Conflicting frozen history references")
        named.add((ref["kind"], ref["id"]))
    families = snapshot["families"]
    if not isinstance(families, dict) or not families:
        raise ValueError("History snapshot has no task memberships")
    total = 0
    for family, value in families.items():
        if (
            not isinstance(family, str)
            or not family
            or not isinstance(value, dict)
            or set(value) != {"source", "task_keys"}
        ):
            raise ValueError("Invalid history family membership")
        source = value["source"]
        if (
            not isinstance(source, dict)
            or source_identity(source, source.get("partition")) != source
        ):
            raise ValueError("Invalid frozen history source identity")
        keys = value["task_keys"]
        if (
            not isinstance(keys, list)
            or not keys
            or any(
                not isinstance(key, str) or not SHA256.fullmatch(key) for key in keys
            )
            or keys != sorted(set(keys))
        ):
            raise ValueError("Invalid canonical history task keys")
        total += len(keys)
    if total > MAX_ROWS:
        raise ValueError("History snapshot exceeds the task membership limit")
    return snapshot


def load_snapshot(path):
    return validate_snapshot(json.loads(read_small(Path(path), MAX_SNAPSHOT_BYTES)))


def save_snapshot(snapshot, path):
    """Publish without replacing existing immutable bytes or following symlinks."""
    validate_snapshot(snapshot)
    content = (canonical(snapshot) + "\n").encode()
    if len(content) > MAX_SNAPSHOT_BYTES:
        raise ValueError("History snapshot exceeds its byte limit")
    path = Path(path).expanduser().absolute()
    if path.parent.resolve() != path.parent:
        raise ValueError("Snapshot destination must not traverse symlinks")
    with tempfile.NamedTemporaryFile(dir=path.parent) as staged:
        staged.write(content)
        staged.flush()
        os.chmod(staged.name, 0o600)
        try:
            os.link(staged.name, path, follow_symlinks=False)
        except FileExistsError:
            if read_small(path, MAX_SNAPSHOT_BYTES) != content:
                raise ValueError(
                    "Existing immutable history snapshot changed"
                ) from None
    return snapshot
