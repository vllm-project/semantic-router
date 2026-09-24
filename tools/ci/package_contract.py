#!/usr/bin/env python3
"""Build and qualify the final CLI distribution before it can be published."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import tomllib
from check_cli_wheel import check_wheel
from prepare_dev_package import prepare_version
from runtime_evidence import host_platform

ROOT = Path(__file__).resolve().parents[2]
DISTRIBUTION_SUFFIXES = (".whl", ".tar.gz")
CHECKS = [
    "catalog.compiler",
    "catalog.completeness",
    "catalog.immutable",
    "package.version",
    "package.staging",
    "package.build",
    "package.metadata",
    "package.resources",
    "package.installed",
]


def run(*command: str) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


DECISION_LOCK_PATH = "cli/decision_runtime/decision-images.lock.json"
SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")


def decision_publication(
    path: Path, *, source_sha: str, mode: str, tag: str, verify_registry: bool
) -> dict:
    """Accept only the CPU publication from this exact source and workflow."""
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or set(document) != {
        "schema_version",
        "image",
        "source_sha",
        "mode",
        "tag",
        "digest",
        "ref",
        "archive_sha256",
        "platform",
    }:
        raise ValueError("Decision publication receipt fields are invalid")
    owner = os.environ.get("GITHUB_REPOSITORY_OWNER", "").lower()
    digest = document["digest"]
    if (
        document["schema_version"] != 1
        or document["image"] != "decision-runtime-cpu"
        or document["source_sha"] != source_sha
        or document["mode"] != mode
        or document["tag"] != tag
        or document["platform"] != "linux/amd64"
        or not isinstance(digest, str)
        or SHA256.fullmatch(digest) is None
        or not isinstance(document["archive_sha256"], str)
        or SHA256.fullmatch("sha256:" + document["archive_sha256"]) is None
        or not re.fullmatch(r"[a-z0-9-]+", owner)
        or document["ref"]
        != f"ghcr.io/{owner}/semantic-router/decision-runtime-cpu@{digest}"
    ):
        raise ValueError("Decision publication does not match this package release")
    if verify_registry:
        manifest = subprocess.check_output(
            ["skopeo", "inspect", "--raw", "docker://" + document["ref"]]
        )
        if "sha256:" + hashlib.sha256(manifest).hexdigest() != digest:
            raise ValueError("Published Decision image digest differs from the receipt")
    return document


def decision_lock(version: str, source_sha: str, image: str) -> bytes:
    return (
        json.dumps(
            {
                "schema_version": 1,
                "package_version": version,
                "source_sha": source_sha,
                "images": {"cpu": image},
            },
            indent=2,
        )
        + "\n"
    ).encode()


def verify_decision_lock(wheel: Path, sdist: Path, lock: bytes) -> None:
    with zipfile.ZipFile(wheel) as whl, tarfile.open(sdist) as archive:
        prefix = archive.getmembers()[0].name.split("/")[0]
        member = archive.extractfile(prefix + "/" + DECISION_LOCK_PATH)
        if (
            member is None
            or member.read() != lock
            or whl.read(DECISION_LOCK_PATH) != lock
        ):
            raise ValueError("Decision image lock differs between release artifacts")


def verify_resources(
    wheel: Path, sdist: Path, decision_image_lock: bytes | None = None
) -> None:
    """Every authored snapshot and public template must survive both formats."""
    package_root = ROOT / "src/vllm-sr"
    paths = sorted((package_root / "cli/model_assets").rglob("*"))
    paths.extend(
        (package_root / "cli/templates" / name)
        for name in (
            "config.template.yaml",
            "envoy.template.yaml",
            "tools_db.json",
            "grafana.serve.ini",
            "prometheus.serve.yaml",
        )
    )
    with zipfile.ZipFile(wheel) as whl, tarfile.open(sdist) as archive:
        prefix = archive.getmembers()[0].name.split("/")[0]
        for path in paths:
            if not path.is_file() or path.suffix == ".pyc":
                continue
            name = path.relative_to(package_root).as_posix()
            member = archive.extractfile(prefix + "/" + name)
            if (
                member is None
                or member.read() != path.read_bytes()
                or whl.read(name) != path.read_bytes()
            ):
                raise ValueError(
                    f"Packaged resource differs from the qualified source: {name}"
                )
    if decision_image_lock is not None:
        verify_decision_lock(wheel, sdist, decision_image_lock)


def verify_distribution(directory: Path, mode: str, tag: str) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text())
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if (manifest["source_sha"], manifest["mode"], manifest["tag"]) != (
        source,
        mode,
        tag,
    ):
        raise ValueError(
            "Package candidate differs from the qualified source or publication context"
        )
    files = manifest["files"]
    if len(files) != len(DISTRIBUTION_SUFFIXES) or any(
        not any(name.endswith(suffix) for name in files)
        for suffix in DISTRIBUTION_SUFFIXES
    ):
        raise ValueError("Incomplete final package inventory")
    if {path.name for path in directory.iterdir()} != {*files, "manifest.json"}:
        raise ValueError("Unexpected files in the final distribution")
    for name, digest in files.items():
        if (
            Path(name).name != name
            or hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest
        ):
            raise ValueError("Final distribution content differs from qualification")
    if decision := manifest.get("decision_image"):
        if not isinstance(decision, dict) or set(decision) != {"ref", "lock_sha256"}:
            raise ValueError("Decision distribution metadata is invalid")
        expected = decision_lock(manifest["version"], source, decision["ref"])
        if hashlib.sha256(expected).hexdigest() != decision["lock_sha256"]:
            raise ValueError("Decision distribution lock metadata differs")
        wheel = next(directory / name for name in files if name.endswith(".whl"))
        sdist = next(directory / name for name in files if name.endswith(".tar.gz"))
        verify_decision_lock(wheel, sdist, expected)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=["pr", "main", "nightly", "release"], default="pr"
    )
    parser.add_argument("--tag", default="")
    parser.add_argument("--base-ref", default="HEAD")
    parser.add_argument("--verify-dist", action="store_true")
    parser.add_argument("--decision-image-receipt", type=Path)
    parser.add_argument("--qualified-dist", type=Path)
    parser.add_argument("--verify-registry", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.verify_dist:
        verify_distribution(output, args.mode, args.tag)
        return
    if bool(args.decision_image_receipt) != bool(args.qualified_dist):
        parser.error(
            "Decision binding requires a publication receipt and qualified distribution"
        )
    if args.verify_registry and not args.decision_image_receipt:
        parser.error("Registry verification requires a Decision publication receipt")
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "runtime": "none",
        "device": "none",
        "platform": host_platform(),
        "expected_checks": CHECKS,
        "checks": [],
        "artifacts": [],
    }
    manifest = {
        "source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "mode": args.mode,
        "tag": args.tag,
    }
    qualified: dict | None = None
    if args.qualified_dist:
        qualified = verify_distribution(args.qualified_dist, args.mode, args.tag)
        if qualified.get("decision_image"):
            raise ValueError("Qualified package is already bound to a Decision image")
        report["expected_checks"] = [*CHECKS, "decision.image"]
    base_sha = subprocess.check_output(
        [
            "git",
            "rev-parse",
            "--verify",
            "--end-of-options",
            args.base_ref + "^{commit}",
        ],
        text=True,
    ).strip()

    def check(name, action):
        item = {"id": name, "status": "failed"}
        report["checks"].append(item)
        try:
            result = action()
            item["status"] = "passed"
            return result
        finally:
            (output / "evidence.json").write_text(json.dumps(report, indent=2) + "\n")

    python = sys.executable
    publication: dict | None = None
    if args.decision_image_receipt:
        publication = check(
            "decision.image",
            lambda: decision_publication(
                args.decision_image_receipt,
                source_sha=manifest["source_sha"],
                mode=args.mode,
                tag=args.tag,
                verify_registry=args.verify_registry,
            ),
        )
    check(
        "catalog.compiler",
        lambda: run(
            python,
            "-m",
            "unittest",
            "discover",
            "-s",
            "tools/catalog/tests",
            "-p",
            "test_*.py",
        ),
    )
    check(
        "catalog.completeness",
        lambda: run(
            python,
            "tools/catalog/audit_model_catalog.py",
            "--require-min-evaluations-per-model",
            "5",
        ),
    )
    check(
        "catalog.immutable",
        lambda: run(
            python,
            "tools/release/snapshot_model_catalog.py",
            "--check-published",
            "--base-ref",
            base_sha,
        ),
    )

    def version():
        project = ROOT / "src/vllm-sr"
        if args.mode in {"main", "nightly"}:
            epoch = int(
                subprocess.check_output(
                    ["git", "show", "-s", "--format=%ct", "HEAD"], text=True
                )
            )
            prepare_version(project, epoch)
        value = tomllib.loads((project / "pyproject.toml").read_text())["project"][
            "version"
        ]
        if args.mode == "release":
            if args.tag != "v" + value:
                raise ValueError("Release tag does not match the final package version")
            run(
                python,
                "tools/release/snapshot_model_catalog.py",
                "--check",
                "--version",
                value,
            )
        manifest["version"] = value

    check("package.version", version)
    lock: bytes | None = None
    if publication is not None:
        if qualified["version"] != manifest["version"]:
            raise ValueError("Qualified package version differs from this source")
        lock = decision_lock(
            manifest["version"], manifest["source_sha"], publication["ref"]
        )
    check(
        "package.staging",
        lambda: run(python, "tools/release/stage_model_catalog_package.py"),
    )
    run(python, "tools/release/stage_model_catalog_package.py", "--check")
    dist = output / "dist"

    def build() -> None:
        lock_path = ROOT / "src/vllm-sr" / DECISION_LOCK_PATH
        if lock is not None:
            if lock_path.exists():
                raise ValueError(
                    "Decision image lock already exists in the source tree"
                )
            lock_path.write_bytes(lock)
        try:
            run(python, "-m", "build", "src/vllm-sr", "--outdir", str(dist))
        finally:
            if lock is not None:
                lock_path.unlink(missing_ok=True)

    check("package.build", build)
    wheels, sources = list(dist.glob("*.whl")), list(dist.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sources) != 1:
        raise ValueError("Expected exactly one wheel and one source distribution")
    check(
        "package.metadata",
        lambda: run(python, "-m", "twine", "check", str(wheels[0]), str(sources[0])),
    )
    if publication is None:
        check("package.resources", lambda: verify_resources(wheels[0], sources[0]))
        check("package.installed", lambda: check_wheel(wheels[0]))
    else:
        check(
            "package.resources",
            lambda: verify_resources(wheels[0], sources[0], lock),
        )
        check(
            "package.installed",
            lambda: check_wheel(wheels[0], decision_image=publication["ref"]),
        )
    if publication is not None:
        manifest["decision_image"] = {
            "ref": publication["ref"],
            "lock_sha256": hashlib.sha256(lock).hexdigest(),
        }
    manifest["files"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in wheels + sources
    }
    (dist / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
