"""Exact-source task identities, independent of evaluation labels and formatting."""

from __future__ import annotations

import re

from .canonical import digest

POLICY = "exact-source-task-v1"
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
MAX_IDENTITY_TEXT = 2048
FIRST_PRINTABLE = 32
MAX_SOURCE_FILES = 50000


def text(value, label):
    if not isinstance(value, str) or not value or len(value) > MAX_IDENTITY_TEXT:
        raise ValueError(f"{label} must be nonempty bounded text")
    if any(ord(c) < FIRST_PRINTABLE for c in value):
        raise ValueError(f"{label} contains control characters")
    return value


def source_identity(source, partition):
    """Bind the declared upstream partition and exact acquired source bytes."""
    if not isinstance(source, dict):
        raise ValueError("Task identity requires source provenance")
    result = {
        key: text(source.get(key), f"source.{key}")
        for key in ("url", "revision", "normalizer")
    }
    result["partition"] = text(partition, "upstream source partition")
    files = source.get("files")
    if not isinstance(files, list) or not files or len(files) > MAX_SOURCE_FILES:
        raise ValueError("Task identity requires bounded source file provenance")
    result["files"] = []
    names = set()
    for entry in files:
        if not isinstance(entry, dict) or set(entry) != {"name", "sha256"}:
            raise ValueError("Invalid source file identity")
        name = text(entry["name"], "source file name")
        if (
            name in names
            or not isinstance(entry["sha256"], str)
            or not SHA256.fullmatch(entry["sha256"])
        ):
            raise ValueError("Duplicate source file or invalid source digest")
        names.add(name)
        result["files"].append(dict(entry))
    result["files"].sort(key=lambda entry: entry["name"])
    return result


def native_task_identity(benchmark, row, source):
    """Only native normalizer inputs establish identities; never parse case aliases."""
    if "messages" in row and row.get("benchmark") == benchmark:
        raise ValueError("Normalized imports have no verified native task identity")
    if benchmark == "gpqa-diamond":
        # The native GPQA normalizer already identifies the unformatted question.
        question = row.get("Question")
        if not isinstance(question, str) or not question:
            raise ValueError("GPQA native question identity is missing")
        task = digest(question)
    else:
        task = next(
            (row[key] for key in ("id", "question_id", "problem_id") if key in row),
            None,
        )
        if type(task) is int:
            task = str(task)
        task = text(task, "native upstream task ID")
    domain = text(row.get("domain"), "tau3 domain") if benchmark == "tau3" else ""
    return {
        "policy": POLICY,
        "benchmark": benchmark,
        "source_sha256": digest(source),
        "partition": source["partition"],
        "domain": domain,
        "task_id": task,
    }


def task_key(case, source):
    metadata = case.get("metadata")
    identity = metadata.get("task_identity") if isinstance(metadata, dict) else None
    fields = {"policy", "benchmark", "source_sha256", "partition", "domain", "task_id"}
    if not isinstance(identity, dict) or set(identity) != fields:
        raise ValueError("Case lacks a verified native task identity")
    if (
        identity["policy"] != POLICY
        or identity["benchmark"] != case.get("benchmark")
        or identity["source_sha256"] != digest(source)
        or identity["partition"] != source["partition"]
    ):
        raise ValueError("Case task identity differs from frozen source provenance")
    text(identity["task_id"], "upstream task ID")
    if identity["benchmark"] == "tau3":
        text(identity["domain"], "tau3 domain")
    elif identity["domain"] != "":
        raise ValueError("Unexpected upstream task domain")
    return digest(identity)
