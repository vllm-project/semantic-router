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
from collections.abc import Mapping
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
HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
SOURCE_SHA = re.compile(r"[0-9a-f]{40}\Z")
ROCM_QUALIFICATION_ENV = "DECISION_ROCM_QUALIFICATION"
ROCM_CHECKS = frozenset(
    {"systemone", "mixed_questions", "batches", "concurrency", "no_regression"}
)
ROCM_INITIAL_MODELS = frozenset(
    f"llm-semantic-router/Decision-1.0-{model}"
    for model in (
        "Kai-0.6B",
        "Lex-0.6B",
        "Eos-0.8B",
        "Sol-2B",
        "Nox-4B",
        "Lux-9B",
    )
)


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


def decision_catalog_revisions() -> dict[str, str]:
    """Read the exact default revisions served by this source checkout."""

    import yaml  # noqa: PLC0415 - installed by the package and release jobs

    path = ROOT / "config/catalog/resources/models/single/llm-semantic-router.yaml"
    catalog = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(catalog, list):
        raise ValueError("Decision model catalog is invalid")
    revisions: dict[str, str] = {}
    for model in catalog:
        if not isinstance(model, dict) or not isinstance(
            model.get("distribution"), dict
        ):
            raise ValueError("Decision model catalog entry is invalid")
        source = model["distribution"].get("source")
        if not isinstance(source, str) or not source.startswith(
            "https://huggingface.co/llm-semantic-router/Decision-1.0-"
        ):
            continue
        model_id = source.removeprefix("https://huggingface.co/")
        revision = model.get("revision")
        if (
            model_id in revisions
            or not isinstance(revision, str)
            or not SOURCE_SHA.fullmatch(revision)
        ):
            raise ValueError("Decision model catalog revision is invalid")
        revisions[model_id] = revision
    if not set(revisions) >= ROCM_INITIAL_MODELS:
        raise ValueError("Decision model catalog omits an initial release model")
    return revisions


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("ROCm qualification contains duplicate fields")
        result[key] = value
    return result


def rocm_qualification(payload: str, *, source_sha: str, verify_registry: bool) -> dict:
    """Validate a maintainer-provided model release receipt and image."""

    if (
        not isinstance(payload, str)
        or not 0 < len(payload.encode("utf-8")) <= 16 * 1024
    ):
        raise ValueError("ROCm qualification is missing or oversized")
    try:
        document = json.loads(payload, object_pairs_hook=_unique_json_object)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("ROCm qualification is invalid JSON") from error
    if not isinstance(document, dict) or set(document) != {
        "schema_version",
        "image",
        "source_sha",
        "platform",
        "digest",
        "ref",
        "evidence_sha256",
        "models",
    }:
        raise ValueError("ROCm qualification fields are invalid")
    owner = os.environ.get("GITHUB_REPOSITORY_OWNER", "").lower()
    digest = document["digest"]
    if (
        type(document["schema_version"]) is not int
        or document["schema_version"] != 1
        or document["image"] != "decision-runtime-rocm"
        or document["source_sha"] != source_sha
        or document["platform"] != "linux/amd64"
        or not isinstance(digest, str)
        or SHA256.fullmatch(digest) is None
        or not isinstance(document["evidence_sha256"], str)
        or HEX_SHA256.fullmatch(document["evidence_sha256"]) is None
        or not re.fullmatch(r"[a-z0-9-]+", owner)
        or document["ref"]
        != f"ghcr.io/{owner}/semantic-router/decision-runtime-rocm@{digest}"
    ):
        raise ValueError("ROCm qualification does not match this package release")
    models = document["models"]
    expected = decision_catalog_revisions()
    if not isinstance(models, dict) or set(models) != set(expected):
        raise ValueError("ROCm qualification must cover all catalog models")
    for model_id, result in models.items():
        if (
            not isinstance(result, dict)
            or set(result) != {"revision", "checks"}
            or result["revision"] != expected[model_id]
            or not isinstance(result["checks"], dict)
            or set(result["checks"]) != ROCM_CHECKS
            or any(value != "passed" for value in result["checks"].values())
        ):
            raise ValueError(f"ROCm qualification is incomplete for {model_id}")
    if verify_registry:
        reference = "docker://" + document["ref"]
        manifest = subprocess.check_output(["skopeo", "inspect", "--raw", reference])
        if "sha256:" + hashlib.sha256(manifest).hexdigest() != digest:
            raise ValueError(
                "Published Decision ROCm digest differs from qualification"
            )
        try:
            config = json.loads(
                subprocess.check_output(["skopeo", "inspect", "--config", reference])
            )
        except json.JSONDecodeError as error:
            raise ValueError("Published Decision ROCm config is invalid") from error
        from image_artifacts import DECISION_RUNTIME_BASES  # noqa: PLC0415

        image_config = config.get("config") if isinstance(config, dict) else None
        labels = image_config.get("Labels") if isinstance(image_config, dict) else None
        required_labels = {
            "org.opencontainers.image.base.name": DECISION_RUNTIME_BASES[
                "decision-runtime-rocm"
            ][0],
            "org.opencontainers.image.revision": source_sha,
            "ai.vllm-sr.decision.source-state": "clean",
            "ai.vllm-sr.decision.backend": "rocm",
        }
        if (
            not isinstance(config, dict)
            or config.get("os") != "linux"
            or config.get("architecture") != "amd64"
            or not isinstance(labels, dict)
            or any(labels.get(key) != value for key, value in required_labels.items())
        ):
            raise ValueError(
                "Published Decision ROCm config differs from qualification"
            )
    return document


def decision_lock(version: str, source_sha: str, images: Mapping[str, str]) -> bytes:
    return (
        json.dumps(
            {
                "schema_version": 1,
                "package_version": version,
                "source_sha": source_sha,
                "images": dict(sorted(images.items())),
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


def verify_distribution(
    directory: Path, mode: str, tag: str, *, require_decision_images: bool = False
) -> dict:
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
    if require_decision_images and "decision_images" not in manifest:
        raise ValueError("Final Decision distribution has no image defaults")
    if "decision_images" in manifest:
        decision = manifest["decision_images"]
        if not isinstance(decision, dict) or set(decision) != {"refs", "lock_sha256"}:
            raise ValueError("Decision distribution metadata is invalid")
        refs = decision["refs"]
        required = {"cpu", "rocm"} if mode == "release" else {"cpu"}
        if not isinstance(refs, dict) or set(refs) != required:
            raise ValueError("Decision distribution backends are incomplete")
        expected = decision_lock(manifest["version"], source, refs)
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
    parser.add_argument("--verify-rocm-qualification", action="store_true")
    parser.add_argument("--require-decision-images", action="store_true")
    parser.add_argument("--decision-image-receipt", type=Path)
    parser.add_argument("--qualified-dist", type=Path)
    parser.add_argument("--verify-registry", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.verify_rocm_qualification:
        if args.mode != "release" or not args.verify_registry:
            parser.error("ROCm release qualification requires registry verification")
        source = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        record = rocm_qualification(
            os.environ.get(ROCM_QUALIFICATION_ENV, ""),
            source_sha=source,
            verify_registry=True,
        )
        print(json.dumps(record, sort_keys=True, separators=(",", ":")))
        return
    if args.output is None:
        parser.error("--output is required")
    output = args.output.resolve()
    if args.verify_dist:
        verify_distribution(
            output,
            args.mode,
            args.tag,
            require_decision_images=args.require_decision_images,
        )
        return
    if bool(args.decision_image_receipt) != bool(args.qualified_dist):
        parser.error(
            "Decision binding requires a publication receipt and qualified distribution"
        )
    if args.verify_registry and not args.decision_image_receipt:
        parser.error("Registry verification requires a Decision publication receipt")
    if (
        args.mode == "release"
        and args.decision_image_receipt
        and not args.verify_registry
    ):
        parser.error("Stable Decision publication requires registry verification")
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
        if qualified.get("decision_images"):
            raise ValueError("Qualified package is already bound to a Decision image")
        report["expected_checks"] = [*CHECKS, "decision.image"]
        if args.mode == "release":
            report["expected_checks"].append("decision.rocm")
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
    rocm_publication: dict | None = None
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
        if args.mode == "release":
            rocm_publication = check(
                "decision.rocm",
                lambda: rocm_qualification(
                    os.environ.get(ROCM_QUALIFICATION_ENV, ""),
                    source_sha=manifest["source_sha"],
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
    decision_images: dict[str, str] | None = None
    if publication is not None:
        if qualified["version"] != manifest["version"]:
            raise ValueError("Qualified package version differs from this source")
        decision_images = {"cpu": publication["ref"]}
        if rocm_publication is not None:
            decision_images["rocm"] = rocm_publication["ref"]
        lock = decision_lock(
            manifest["version"], manifest["source_sha"], decision_images
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
            lambda: check_wheel(wheels[0], decision_images=decision_images),
        )
    if publication is not None:
        manifest["decision_images"] = {
            "refs": decision_images,
            "lock_sha256": hashlib.sha256(lock).hexdigest(),
        }
    manifest["files"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in wheels + sources
    }
    (dist / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
