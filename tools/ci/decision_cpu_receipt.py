"""Pure-data validation for a source-bound, three-model CPU image receipt."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from image_artifacts import DECISION_RUNTIME_BASES

IMAGE = "decision-runtime-cpu"
MODEL_IDS = frozenset(
    "llm-semantic-router/Decision-1.0-" + suffix
    for suffix in ("Kai-0.6B", "Lex-0.6B", "Eos-0.8B")
)
SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
REVISION = re.compile(r"[0-9a-f]{40}\Z")
OWNER = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]*\Z")
RAW_FILES = frozenset(
    {
        "drun-launch.txt",
        "ready.json",
        "status.json",
        "mixed-single-request.json",
        "mixed-single-response.json",
        "two-state-batch-request.json",
        "two-state-batch-response.json",
        "image-attestation.json",
    }
)
REQUIRED_CHECKS = frozenset(
    {
        "artifact.identity",
        "health",
        "single.noul",
        "single.choice",
        "single.score",
        "batch.mixed_two_states",
        "cpu.execution",
    }
)
MAX_JSON_BYTES = 64 * 1024
MAX_RAW_BYTES = 256 * 1024


def _digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def published_ref(owner: str, digest: str) -> str:
    if OWNER.fullmatch(owner) is None or SHA256.fullmatch(digest) is None:
        raise ValueError("CPU owner or published digest is invalid")
    return f"ghcr.io/{owner.lower()}/semantic-router/{IMAGE}@{digest}"


def _read_bounded(path: Path, root: Path, limit: int) -> bytes:
    if (
        path.is_symlink()
        or not path.is_file()
        or not path.resolve().is_relative_to(root)
    ):
        raise ValueError("CPU qualification evidence path is missing or escapes root")
    size = path.stat().st_size
    if size <= 0 or size > limit:
        raise ValueError("CPU qualification evidence size is invalid")
    with path.open("rb") as stream:
        payload = stream.read(limit + 1)
    if len(payload) != size or len(payload) > limit:
        raise ValueError("CPU qualification evidence changed while reading")
    return payload


def _relative_file(root: Path, relative: str, *, suffix: str | None = None) -> Path:
    name = Path(relative)
    if (
        not relative
        or name.is_absolute()
        or ".." in name.parts
        or (suffix is not None and name.suffix != suffix)
    ):
        raise ValueError("CPU qualification evidence path is invalid")
    return root / name


def validate_receipt(
    path: Path, *, owner: str, revision: str, published_image: str
) -> dict:
    """Require three source-bound, hash-sealed live CPU model observations."""

    root = path.parent.resolve(strict=True)
    record = json.loads(_read_bounded(path, root, MAX_JSON_BYTES))
    base, _, _ = DECISION_RUNTIME_BASES[IMAGE]
    if (
        not isinstance(record, dict)
        or record.get("schema") != 1
        or record.get("image") != IMAGE
        or record.get("source_sha") != revision
        or record.get("base_image") != base
        or record.get("backend") != "cpu"
        or record.get("platform") != "linux/amd64"
        or record.get("published_ref") != published_image
        or not isinstance(published_image, str)
        or "@" not in published_image
        or published_ref(owner, published_image.rsplit("@", 1)[1]) != published_image
    ):
        raise ValueError("CPU receipt does not bind the published image and source")
    rows = record.get("models")
    if not isinstance(rows, list) or len(rows) != len(MODEL_IDS):
        raise ValueError("CPU qualification needs exactly three model results")
    observed: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("CPU model evidence inventory is invalid")
        model = row.get("id")
        revision_id = row.get("revision")
        artifact = row.get("artifact_content_id")
        evidence_name = row.get("evidence_file")
        evidence_digest = row.get("evidence_sha256")
        if (
            not isinstance(model, str)
            or model not in MODEL_IDS
            or model in observed
            or not isinstance(revision_id, str)
            or REVISION.fullmatch(revision_id) is None
            or not isinstance(artifact, str)
            or SHA256.fullmatch(artifact) is None
            or not isinstance(evidence_name, str)
            or not isinstance(evidence_digest, str)
            or SHA256.fullmatch(evidence_digest) is None
        ):
            raise ValueError("CPU model identity is invalid or duplicated")
        observed.add(model)
        evidence_path = _relative_file(root, evidence_name, suffix=".json")
        raw = _read_bounded(evidence_path, root, MAX_JSON_BYTES)
        if _digest(raw) != evidence_digest:
            raise ValueError("CPU model evidence changed after qualification")
        evidence = json.loads(raw)
        if (
            not isinstance(evidence, dict)
            or evidence.get("image_ref") != published_image
            or evidence.get("source_sha") != revision
            or evidence.get("model_id") != model
            or evidence.get("revision") != revision_id
            or evidence.get("artifact_content_id") != artifact
            or evidence.get("backend") != "cpu"
            or evidence.get("device") != "cpu"
            or evidence.get("result") != "passed"
            or not isinstance(evidence.get("checks"), dict)
            or any(
                evidence["checks"].get(check) is not True for check in REQUIRED_CHECKS
            )
        ):
            raise ValueError("CPU model evidence is incomplete")
        slug = model.rsplit("/", 1)[-1].removeprefix("Decision-1.0-").lower()
        expected = {f"raw/{slug}/{name}" for name in RAW_FILES}
        declared = evidence.get("raw_sha256")
        if not isinstance(declared, dict) or set(declared) != expected:
            raise ValueError("CPU raw evidence inventory is incomplete")
        for name, digest in declared.items():
            if (
                not isinstance(name, str)
                or not isinstance(digest, str)
                or SHA256.fullmatch(digest) is None
            ):
                raise ValueError("CPU raw evidence digest is invalid")
            raw_path = _relative_file(root, name)
            raw_payload = _read_bounded(raw_path, root, MAX_RAW_BYTES)
            if _digest(raw_payload) != digest:
                raise ValueError("CPU raw evidence changed after qualification")
            if name.endswith("/image-attestation.json"):
                attestation = json.loads(raw_payload)
                if (
                    not isinstance(attestation, dict)
                    or set(attestation) != {"published_ref", "local_image_id"}
                    or attestation["published_ref"] != published_image
                    or not isinstance(attestation["local_image_id"], str)
                    or SHA256.fullmatch(attestation["local_image_id"]) is None
                ):
                    raise ValueError("CPU live image attestation is invalid")
    if observed != MODEL_IDS:
        raise ValueError("CPU qualification is missing a Decision model")
    return record
