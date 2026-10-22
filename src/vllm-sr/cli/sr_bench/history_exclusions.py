"""Bounded read-only compilation of named frozen dataset/run memberships."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from contextlib import closing
from pathlib import Path

from .canonical import canonical, digest, plan_digest
from .dataset_io import MAX_ROWS, MAX_SCAN_BYTES, read_small, verified_lines
from .history_snapshot import (
    DATASET_ID,
    MAX_REFERENCES,
    MAX_RUN_BYTES,
    RESERVATION,
    RUN_ID,
    SCHEMA,
    validate_snapshot,
)
from .preparation import validate_cases, validate_manifest
from .task_identity import POLICY, task_key


def _membership(case):
    if (
        not isinstance(case, dict)
        or not isinstance(case.get("id"), str)
        or not isinstance(case.get("benchmark"), str)
    ):
        raise ValueError("Invalid frozen task membership")
    metadata = case.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("Invalid frozen task metadata")
    return {
        "id": case["id"],
        "benchmark": case["benchmark"],
        "metadata": {"task_identity": metadata.get("task_identity")},
    }


def _dataset_reference(root, identity, remaining):
    if not isinstance(identity, str) or not DATASET_ID.fullmatch(identity):
        raise ValueError("Invalid prepared dataset reference")
    folder = root / "datasets" / identity
    if folder.resolve() != folder or folder.is_symlink():
        raise ValueError("History dataset must remain inside the selected store")
    raw = read_small(folder / "manifest.json", 2 * 1024 * 1024)
    manifest = json.loads(raw)
    if not isinstance(manifest, dict) or manifest.get("id") != identity:
        raise ValueError("History dataset identity changed")
    validate_manifest(manifest)
    count = manifest.get("case_count")
    if type(count) is not int or not 1 <= count <= MAX_ROWS:
        raise ValueError("History dataset membership count exceeds its limit")
    path = folder / "cases.jsonl"
    size = path.stat().st_size
    cases = []
    for line in verified_lines(path, manifest["sha256"], maximum=remaining):
        if len(cases) >= count:
            raise ValueError("History dataset membership count changed")
        cases.append(_membership(json.loads(line)))
    if len(cases) != count:
        raise ValueError("History dataset membership count changed")
    return (
        manifest,
        cases,
        size,
        {
            "kind": "dataset",
            "id": identity,
            "manifest_sha256": hashlib.sha256(raw).hexdigest(),
            "case_sha256": manifest["sha256"],
        },
    )


def _run_reference(root, identity, remaining):
    if not isinstance(identity, str) or not RUN_ID.fullmatch(identity):
        raise ValueError("Invalid frozen run reference")
    path = root / "journal.sqlite3"
    if not path.is_file() or path.is_symlink() or path.resolve() != path:
        raise ValueError("History requires the existing read-only journal")
    deadline = time.monotonic() + 10
    with closing(
        sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=1)
    ) as db:
        db.execute("PRAGMA query_only=ON")
        db.set_progress_handler(lambda: time.monotonic() >= deadline, 1000)
        db.execute("BEGIN")
        row = db.execute(
            "SELECT length(CAST(manifest AS BLOB)) FROM runs WHERE id=?", (identity,)
        ).fetchone()
        if row is None or not 0 < row[0] <= min(MAX_RUN_BYTES, remaining):
            raise ValueError("Named history run is missing or exceeds its read limit")
        raw = db.execute(
            "SELECT manifest FROM runs WHERE id=?", (identity,)
        ).fetchone()[0]
        db.rollback()
    manifest = json.loads(raw)
    if not isinstance(manifest, dict) or manifest.get("plan_sha256") != plan_digest(
        manifest
    ):
        raise ValueError("Frozen history run plan digest changed")
    cases = manifest.get("cases")
    if (
        not isinstance(cases, list)
        or not 1 <= len(cases) <= MAX_ROWS
        or digest(cases) != manifest.get("case_sha256")
    ):
        raise ValueError("Frozen history run membership digest changed")
    prepared = manifest.get("dataset", {})
    return (
        prepared,
        [_membership(case) for case in cases],
        row[0],
        {
            "kind": "run",
            "id": identity,
            "manifest_sha256": manifest["plan_sha256"],
            "case_sha256": manifest["case_sha256"],
        },
    )


def compile_snapshot(store, *, dataset_ids=(), run_ids=()):
    """Reserve all frozen memberships, without consulting calls, outcomes or status."""
    root = Path(store).expanduser().resolve()
    named = sorted(
        {("dataset", value) for value in dataset_ids}
        | {("run", value) for value in run_ids}
    )
    if not 1 <= len(named) <= MAX_REFERENCES:
        raise ValueError("Specify 1 to 64 named prepared datasets or runs")
    references, families, total, rows = [], {}, 0, 0
    for kind, identity in named:
        if kind == "dataset":
            manifest, cases, size, ref = _dataset_reference(
                root, identity, MAX_SCAN_BYTES - total
            )
        else:
            manifest, cases, size, ref = _run_reference(
                root, identity, MAX_SCAN_BYTES - total
            )
        total += size
        rows += len(cases)
        if total > MAX_SCAN_BYTES or rows > MAX_ROWS:
            raise ValueError("History reference scan exceeds its cumulative limit")
        validate_cases(manifest, cases)
        references.append(ref)
        for case in cases:
            family = case["benchmark"]
            prepared = manifest.get("preparation", {}).get(family, {})
            source = prepared.get("task_source")
            if not source:
                raise ValueError(
                    "History reference lacks source-bound native task identities"
                )
            entry = families.setdefault(family, {"source": source, "task_keys": set()})
            if entry["source"] != source:
                raise ValueError(
                    "Cross-source or cross-revision history requires explicit reconciliation"
                )
            entry["task_keys"].add(task_key(case, source))
    content = {
        "schema": SCHEMA,
        "policy": RESERVATION,
        "identity_policy": POLICY,
        "coverage": "named-memberships-only",
        "references": sorted(references, key=canonical),
        "families": {
            family: {"source": value["source"], "task_keys": sorted(value["task_keys"])}
            for family, value in sorted(families.items())
        },
    }
    return validate_snapshot({**content, "id": digest(content)})
