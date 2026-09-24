#!/usr/bin/env python3
"""Validate six-model ROCm evidence before copying an immutable OCI digest.

The ROCm image is too large for the ordinary hosted-runner artifact handoff.
This tool is the explicit device-qualified promotion seam; its default action
is read-only validation. Run it from a protected main or release-tag checkout
after the candidate was built and tested and pushed to a staging registry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

from image_artifacts import DECISION_RUNTIME_BASES, source_sha

IMAGE = "decision-runtime-rocm"
MODEL_IDS = frozenset(
    "llm-semantic-router/Decision-1.0-" + suffix
    for suffix in ("Kai-0.6B", "Lex-0.6B", "Eos-0.8B", "Sol-2B", "Nox-4B", "Lux-9B")
)
MODEL_COUNT = len(MODEL_IDS)
REQUIRED_CHECKS = frozenset(
    {
        "artifact.identity",
        "health",
        "single.noul",
        "single.choice",
        "single.score",
        "batch.mixed_two_states",
        "rocm.execution",
    }
)
SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
REVISION = re.compile(r"[0-9a-f]{40}\Z")
STABLE_TAG_REF = re.compile(
    r"refs/tags/v[0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.-]+)?\Z"
)
RAW_FILES = frozenset(
    {
        "drun-launch.txt",
        "ready.json",
        "status.json",
        "metrics-before.txt",
        "metrics-after.txt",
    }
    | {
        f"{probe}-{direction}.json"
        for probe in ("mixed-single", "two-state-batch", "wide-single", "wide-batch")
        for direction in ("request", "response")
    }
    | {
        f"concurrent-{index:02d}-{direction}.json"
        for index in range(8)
        for direction in ("request", "response")
    }
)


def _digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _evidence_file(root: Path, name: str) -> Path:
    path = Path(name)
    if not name or path.is_absolute() or ".." in path.parts or path.suffix != ".json":
        raise ValueError("qualification evidence path must be a relative JSON file")
    full = (root / path).resolve(strict=True)
    if not full.is_relative_to(root.resolve(strict=True)):
        raise ValueError("qualification evidence escapes the receipt directory")
    return full


def _validate_raw_files(root: Path, model_id: str, evidence: dict) -> None:
    slug = model_id.rsplit("/", 1)[-1].removeprefix("Decision-1.0-").lower()
    prefix = f"raw/{slug}/"
    expected = {prefix + filename for filename in RAW_FILES}
    declared = evidence.get("raw_sha256")
    if not isinstance(declared, dict) or set(declared) != expected:
        raise ValueError(f"ROCm raw evidence inventory is incomplete for {model_id}")
    for name, digest in declared.items():
        if not isinstance(digest, str) or not SHA256.fullmatch(digest):
            raise ValueError(f"ROCm raw evidence hash is invalid for {model_id}")
        try:
            path = (root / name).resolve(strict=True)
        except OSError as error:
            raise ValueError(
                f"ROCm raw evidence file is missing for {model_id}"
            ) from error
        if not path.is_relative_to(root.resolve(strict=True)):
            raise ValueError("ROCm raw evidence escapes the receipt directory")
        if not path.is_file() or _digest(path.read_bytes()) != digest:
            raise ValueError(f"ROCm raw evidence content changed for {model_id}")


def validate_receipt(path: Path, *, owner: str, revision: str) -> dict:
    """Check file hashes, exact six-model coverage and source-bound results."""
    record = json.loads(path.read_text(encoding="utf-8"))
    base, _, _ = DECISION_RUNTIME_BASES[IMAGE]
    if (
        record.get("schema") != 1
        or record.get("image") != IMAGE
        or record.get("source_sha") != revision
        or record.get("base_image") != base
        or record.get("backend") != "rocm"
        or record.get("platform") != "linux/amd64"
    ):
        raise ValueError("ROCm receipt does not match this image and source")
    prefix = f"ghcr.io/{owner.lower()}/semantic-router/decision-runtime-rocm-staging@"
    candidate = record.get("candidate_ref", "")
    if not isinstance(candidate, str) or not candidate.startswith(prefix):
        raise ValueError(
            "ROCm candidate must use the owner staging repository by digest"
        )
    if not SHA256.fullmatch(candidate.removeprefix(prefix)):
        raise ValueError("ROCm candidate must use a full immutable digest")
    if (
        not isinstance(record.get("models"), list)
        or len(record["models"]) != MODEL_COUNT
    ):
        raise ValueError("ROCm qualification needs exactly six model results")
    observed: set[str] = set()
    for row in record["models"]:
        model_id = row.get("id")
        model_revision = row.get("revision", "")
        artifact_id = row.get("artifact_content_id", "")
        name = row.get("evidence_file", "")
        declared_digest = row.get("evidence_sha256", "")
        if (
            model_id not in MODEL_IDS
            or model_id in observed
            or not isinstance(model_revision, str)
            or not REVISION.fullmatch(model_revision)
            or not isinstance(artifact_id, str)
            or not SHA256.fullmatch(artifact_id)
            or not isinstance(name, str)
            or not isinstance(declared_digest, str)
            or not SHA256.fullmatch(declared_digest)
        ):
            raise ValueError(
                "ROCm model qualification identity is invalid or duplicated"
            )
        observed.add(model_id)
        raw = _evidence_file(path.parent, name).read_bytes()
        if _digest(raw) != declared_digest:
            raise ValueError(f"ROCm evidence content changed for {model_id}")
        evidence = json.loads(raw)
        if (
            evidence.get("image_ref") != candidate
            or evidence.get("source_sha") != revision
            or evidence.get("model_id") != model_id
            or evidence.get("revision") != model_revision
            or evidence.get("artifact_content_id") != artifact_id
            or evidence.get("backend") != "rocm"
            or evidence.get("device") != "rocm"
            or evidence.get("result") != "passed"
            or not isinstance(evidence.get("checks"), dict)
            or any(
                evidence["checks"].get(check) is not True for check in REQUIRED_CHECKS
            )
        ):
            raise ValueError(f"ROCm evidence is incomplete for {model_id}")
        _validate_raw_files(path.parent, model_id, evidence)
    if observed != MODEL_IDS:
        raise ValueError("ROCm qualification is missing a Decision model")
    return record


def validate_registry_candidate(record: dict) -> None:
    """Inspect the digest itself; tags and receipt text cannot supply identity."""
    candidate = record["candidate_ref"]
    raw = subprocess.check_output(
        ["skopeo", "inspect", "--raw", "docker://" + candidate]
    )
    if _digest(raw) != candidate.rsplit("@", 1)[1]:
        raise ValueError("ROCm registry content differs from candidate digest")
    config = json.loads(
        subprocess.check_output(
            [
                "skopeo",
                "--override-os",
                "linux",
                "--override-arch",
                "amd64",
                "inspect",
                "--config",
                "docker://" + candidate,
            ]
        )
    )
    expected = {
        "org.opencontainers.image.base.name": record["base_image"],
        "org.opencontainers.image.revision": record["source_sha"],
        "ai.vllm-sr.decision.source-state": "clean",
        "ai.vllm-sr.decision.backend": "rocm",
    }
    labels = config.get("config", {}).get("Labels", {})
    if any(labels.get(key) != value for key, value in expected.items()):
        raise ValueError("ROCm registry configuration differs from qualification")
    if config.get("os") != "linux" or config.get("architecture") != "amd64":
        raise ValueError("ROCm registry platform differs from qualification")


def promote(record: dict, *, owner: str) -> str:
    """Copy the qualified digest from a protected source push without rebuilding."""
    ref = os.environ.get("GITHUB_REF", "")
    if (
        os.environ.get("GITHUB_REPOSITORY", "").lower()
        != f"{owner.lower()}/semantic-router"
        or os.environ.get("GITHUB_EVENT_NAME") != "push"
        or os.environ.get("GITHUB_SHA") != record["source_sha"]
        or (ref != "refs/heads/main" and STABLE_TAG_REF.fullmatch(ref) is None)
    ):
        raise ValueError("ROCm promotion requires a protected exact-source push")
    destination = (
        f"ghcr.io/{owner.lower()}/semantic-router/{IMAGE}:{record['source_sha']}"
    )
    digest_file = Path(os.environ["RUNNER_TEMP"]) / "decision-rocm-published-digest.txt"
    subprocess.run(
        [
            "skopeo",
            "copy",
            "--all",
            "--preserve-digests",
            "--digestfile",
            str(digest_file),
            "docker://" + record["candidate_ref"],
            "docker://" + destination,
        ],
        check=True,
    )
    published = digest_file.read_text(encoding="utf-8").strip()
    if published != record["candidate_ref"].rsplit("@", 1)[1]:
        raise ValueError("ROCm publication digest differs from tested candidate")
    return destination.rsplit(":", 1)[0] + "@" + published


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--promote", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9-]*", args.owner):
        parser.error("--owner must be a GitHub organization or user name")
    record = validate_receipt(args.receipt, owner=args.owner, revision=source_sha())
    validate_registry_candidate(record)
    if args.promote:
        print(promote(record, owner=args.owner))
    else:
        print(json.dumps({"qualified_candidate": record["candidate_ref"]}))


if __name__ == "__main__":
    main()
