#!/usr/bin/env python3
"""Check a release-injected Decision image lock against published evidence.

Run this in the protected package publication job after both image digests are
published. The CPU digest file must come from the image publication artifact;
the ROCm receipt and its six hashed model results must come from the device
qualification run. This tool checks their binding to the checkout and registry
content. It cannot establish the trust of the jobs that produced those inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

from decision_rocm_promotion import (
    validate_receipt,
    validate_registry_candidate,
)
from image_artifacts import DECISION_RUNTIME_BASES, source_sha

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "vllm-sr"))
from cli.decision_runtime.image_lock import (
    DecisionImageLock,
    parse_decision_image_lock,
)

_OWNER = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9-]*\Z")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_REQUIRED_BACKENDS = frozenset({"cpu", "rocm"})
_MAX_LOCK_BYTES = 16 * 1024
_MAX_DIGEST_BYTES = 128
_PACKAGE_LOCK_PATH = "cli/decision_runtime/decision-images.lock.json"


def _published_ref(owner: str, backend: str, digest: str) -> str:
    return (
        f"ghcr.io/{owner.lower()}/semantic-router/decision-runtime-{backend}@{digest}"
    )


def _read_lock(path: Path) -> DecisionImageLock:
    if not path.is_file() or not 0 < path.stat().st_size <= _MAX_LOCK_BYTES:
        raise ValueError("release image lock is missing or too large")
    return parse_decision_image_lock(path.read_bytes())


def _read_digest(path: Path) -> str:
    if not path.is_file() or not 0 < path.stat().st_size <= _MAX_DIGEST_BYTES:
        raise ValueError("CPU publication digest artifact is missing or too large")
    digest = path.read_text(encoding="ascii").strip()
    if _DIGEST.fullmatch(digest) is None:
        raise ValueError("CPU publication digest artifact is invalid")
    return digest


def inspect_published_image(reference: str, *, revision: str, backend: str) -> None:
    """Verify the registry's digest and source/backend labels, not a tag name."""

    manifest = subprocess.check_output(
        ["skopeo", "inspect", "--raw", "docker://" + reference]
    )
    digest = "sha256:" + hashlib.sha256(manifest).hexdigest()
    if digest != reference.rsplit("@", 1)[1]:
        raise ValueError(f"published {backend} registry content differs from digest")
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
                "docker://" + reference,
            ]
        )
    )
    base, _, _ = DECISION_RUNTIME_BASES[f"decision-runtime-{backend}"]
    expected = {
        "org.opencontainers.image.base.name": base,
        "org.opencontainers.image.revision": revision,
        "ai.vllm-sr.decision.source-state": "clean",
        "ai.vllm-sr.decision.backend": backend,
    }
    labels = config.get("config", {}).get("Labels", {})
    if any(labels.get(key) != value for key, value in expected.items()):
        raise ValueError(f"published {backend} image labels differ from source")
    if config.get("os") != "linux" or config.get("architecture") != "amd64":
        raise ValueError(f"published {backend} image platform is unsupported")


def validate_release_lock(
    lock_path: Path,
    *,
    owner: str,
    revision: str,
    cpu_digest_file: Path,
    rocm_receipt: Path,
    rocm_published_ref: str,
) -> DecisionImageLock:
    """Check exact CPU/ROCm release digests and six-model ROCm provenance."""

    return _validate_inventory(
        _read_lock(lock_path),
        owner=owner,
        revision=revision,
        cpu_digest_file=cpu_digest_file,
        rocm_receipt=rocm_receipt,
        rocm_published_ref=rocm_published_ref,
    )


def _validate_inventory(
    lock: DecisionImageLock,
    *,
    owner: str,
    revision: str,
    cpu_digest_file: Path,
    rocm_receipt: Path,
    rocm_published_ref: str,
) -> DecisionImageLock:
    if _OWNER.fullmatch(owner) is None:
        raise ValueError("owner must be a GitHub organization or user name")
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("release source SHA must be a full lowercase commit ID")
    if lock.source_sha != revision:
        raise ValueError("release image lock belongs to a different source commit")
    if set(lock.images) != _REQUIRED_BACKENDS:
        raise ValueError("official Decision release needs CPU and ROCm images")

    cpu_ref = _published_ref(owner, "cpu", _read_digest(cpu_digest_file))
    if lock.images["cpu"] != cpu_ref:
        raise ValueError("CPU image differs from the published digest artifact")

    receipt = validate_receipt(rocm_receipt, owner=owner, revision=revision)
    qualified_digest = receipt["candidate_ref"].rsplit("@", 1)[1]
    expected_rocm_ref = _published_ref(owner, "rocm", qualified_digest)
    if (
        rocm_published_ref != expected_rocm_ref
        or lock.images["rocm"] != expected_rocm_ref
    ):
        raise ValueError("ROCm image differs from the six-model qualified digest")

    validate_registry_candidate(receipt)
    inspect_published_image(cpu_ref, revision=revision, backend="cpu")
    inspect_published_image(expected_rocm_ref, revision=revision, backend="rocm")
    return lock


def generate_release_lock(
    output: Path,
    *,
    owner: str,
    revision: str,
    cpu_digest_file: Path,
    rocm_receipt: Path,
    rocm_published_ref: str,
) -> DecisionImageLock:
    """Build one canonical lock after checking all release evidence.

    Existing output is accepted only when its bytes are identical. A new lock
    is linked into place without overwriting a concurrent or previous file.
    """

    if (
        _OWNER.fullmatch(owner) is None
        or re.fullmatch(r"[0-9a-f]{40}", revision) is None
    ):
        raise ValueError("release owner or source SHA is invalid")
    receipt = validate_receipt(rocm_receipt, owner=owner, revision=revision)
    document = {
        "schema_version": 1,
        "source_sha": revision,
        "images": {
            "cpu": _published_ref(owner, "cpu", _read_digest(cpu_digest_file)),
            "rocm": _published_ref(
                owner, "rocm", receipt["candidate_ref"].rsplit("@", 1)[1]
            ),
        },
    }
    payload = (
        json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    lock = parse_decision_image_lock(payload)
    _validate_inventory(
        lock,
        owner=owner,
        revision=revision,
        cpu_digest_file=cpu_digest_file,
        rocm_receipt=rocm_receipt,
        rocm_published_ref=rocm_published_ref,
    )

    if output.is_symlink():
        raise ValueError("release image lock output may not be a symlink")
    if output.exists():
        if output.read_bytes() != payload:
            raise ValueError("release image lock output already differs")
        return lock
    if not output.parent.is_dir():
        raise ValueError("release image lock output directory is missing")
    with tempfile.TemporaryDirectory(
        dir=output.parent, prefix=".decision-lock-"
    ) as temporary:
        staged_path = Path(temporary) / "lock.json"
        with staged_path.open("xb") as staged:
            staged.write(payload)
            staged.flush()
            os.fsync(staged.fileno())
        os.link(staged_path, output)
    return lock


def verify_distribution_lock(lock_path: Path, wheel: Path, sdist: Path) -> str:
    """Ensure both published Python formats contain the validated lock bytes."""

    expected = lock_path.read_bytes()
    parse_decision_image_lock(expected)
    with zipfile.ZipFile(wheel) as package:
        names = [name for name in package.namelist() if name == _PACKAGE_LOCK_PATH]
        if (
            len(names) != 1
            or package.getinfo(names[0]).is_dir()
            or package.getinfo(names[0]).file_size != len(expected)
        ):
            raise ValueError("wheel must contain exactly one Decision image lock")
        wheel_bytes = package.read(names[0])
    with tarfile.open(sdist, mode="r:gz") as package:
        members = [
            member
            for member in package.getmembers()
            if member.name.partition("/")[2] == _PACKAGE_LOCK_PATH
        ]
        if (
            len(members) != 1
            or not members[0].isfile()
            or members[0].size != len(expected)
        ):
            raise ValueError("sdist must contain exactly one Decision image lock")
        stream = package.extractfile(members[0])
        if stream is None:
            raise ValueError("sdist Decision image lock cannot be read")
        sdist_bytes = stream.read()
    if wheel_bytes != expected or sdist_bytes != expected:
        raise ValueError(
            "wheel or sdist Decision image lock differs from release input"
        )
    return "sha256:" + hashlib.sha256(expected).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("generate", "validate"):
        operation = commands.add_parser(command)
        operation.add_argument("--owner", required=True)
        operation.add_argument("--cpu-digest-file", required=True, type=Path)
        operation.add_argument("--rocm-receipt", required=True, type=Path)
        operation.add_argument("--rocm-published-ref", required=True)
        operation.add_argument(
            "--output" if command == "generate" else "--lock",
            required=True,
            type=Path,
        )
    distribution = commands.add_parser("check-dist")
    distribution.add_argument("--lock", required=True, type=Path)
    distribution.add_argument("--wheel", required=True, type=Path)
    distribution.add_argument("--sdist", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "check-dist":
        print(
            json.dumps(
                {
                    "lock_digest": verify_distribution_lock(
                        args.lock, args.wheel, args.sdist
                    )
                }
            )
        )
        return
    shared = {
        "owner": args.owner,
        "revision": source_sha(),
        "cpu_digest_file": args.cpu_digest_file,
        "rocm_receipt": args.rocm_receipt,
        "rocm_published_ref": args.rocm_published_ref,
    }
    lock = (
        generate_release_lock(args.output, **shared)
        if args.command == "generate"
        else validate_release_lock(args.lock, **shared)
    )
    print(
        json.dumps(
            {"source_sha": lock.source_sha, "images": dict(lock.images)}, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
