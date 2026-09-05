#!/usr/bin/env python3
"""Build definitions and immutable handoff for CI container images."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path

DUAL = ["linux/amd64", "linux/arm64"]
DEFINITIONS = {
    "anthropic-shim": (
        "e2e/testing/anthropic-shim",
        "e2e/testing/anthropic-shim/Dockerfile",
        DUAL,
    ),
    "dashboard": (".", "dashboard/backend/Dockerfile", DUAL),
    "extproc": (".", "tools/docker/Dockerfile.extproc", DUAL),
    "extproc-rocm": (".", "tools/docker/Dockerfile.extproc-rocm", ["linux/amd64"]),
    "llm-katan": ("e2e/testing/llm-katan", "e2e/testing/llm-katan/Dockerfile", DUAL),
    "mock-vllm": (
        "tools/test/services/mock-vllm",
        "tools/test/services/mock-vllm/Dockerfile",
        ["linux/amd64"],
    ),
    "operator": (".", "deploy/operator/Dockerfile", DUAL),
    "operator-bundle": ("deploy/operator", "deploy/operator/bundle/Dockerfile", DUAL),
    "vllm-sr": (".", "src/vllm-sr/Dockerfile", DUAL),
    "vllm-sr-cuda": (".", "src/vllm-sr/Dockerfile.cuda", ["linux/amd64"]),
    "vllm-sr-rocm": (".", "src/vllm-sr/Dockerfile.rocm", ["linux/amd64"]),
    "vllm-sr-sim": (".", "src/fleet-sim/Dockerfile", DUAL),
}
IMAGE_ENV = {
    "vllm-sr": ["VLLM_SR_IMAGE", "VLLM_SR_ROUTER_IMAGE"],
    "dashboard": ["VLLM_SR_DASHBOARD_IMAGE"],
    "extproc": ["E2E_PREBUILT_EXT_PROC_IMAGE"],
    "operator": ["E2E_PREBUILT_OPERATOR_IMAGE"],
    "operator-bundle": ["E2E_PREBUILT_OPERATOR_BUNDLE_IMAGE"],
    "llm-katan": ["E2E_PREBUILT_LLM_KATAN_IMAGE", "LLM_KATAN_IMAGE"],
    "anthropic-shim": ["E2E_PREBUILT_ANTHROPIC_SHIM_IMAGE"],
    "mock-vllm": ["E2E_PREBUILT_MOCK_VLLM_IMAGE"],
}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_sha() -> str:
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if os.environ.get("GITHUB_SHA", actual) != actual:
        raise ValueError("Checkout differs from the workflow source revision")
    return actual


def oci_images(path: Path) -> list[dict]:
    """Inspect descriptors without extracting untrusted archive paths."""
    with tarfile.open(path) as archive:

        def document(name: str, digest: str | None = None) -> dict:
            stream = archive.extractfile(name)
            if stream is None:
                raise ValueError(f"Missing OCI document: {name}")
            raw = stream.read()
            if digest and "sha256:" + hashlib.sha256(raw).hexdigest() != digest:
                raise ValueError(f"OCI descriptor digest mismatch: {name}")
            return json.loads(raw)

        def blob(digest: str) -> dict:
            if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
                raise ValueError("Unsupported OCI digest")
            return document("blobs/sha256/" + digest.split(":")[1], digest)

        images = []

        def visit(descriptor: dict) -> None:
            value = blob(descriptor["digest"])
            if "manifests" in value:
                for child in value["manifests"]:
                    visit(child)
                return
            config = blob(value["config"]["digest"])
            platform = f"{config['os']}/{config['architecture']}"
            images.append(
                {
                    "platform": platform,
                    "manifest": descriptor["digest"],
                    "config": value["config"]["digest"],
                }
            )

        for descriptor in document("index.json")["manifests"]:
            visit(descriptor)
        if not images or len({item["platform"] for item in images}) != len(images):
            raise ValueError("Empty or duplicate OCI platform inventory")
        return sorted(images, key=lambda item: item["platform"])


def verify(directory: Path, image: str) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text())
    archive = directory / "image.tar"
    context, dockerfile, _ = DEFINITIONS[image]
    if manifest["source_sha"] != source_sha() or manifest["id"] != image:
        raise ValueError("Image belongs to a different source revision or definition")
    if manifest["context"] != context or manifest["dockerfile"] != dockerfile:
        raise ValueError("Image build definition changed")
    if manifest["sha256"] != sha256(archive) or manifest["images"] != oci_images(
        archive
    ):
        raise ValueError("Image artifact content differs from its receipt")
    return manifest


def verify_loaded_image(archive: Path, expected: dict, actual: dict) -> None:
    """Compare executable content across classic and containerd Docker stores.

    Docker image IDs can identify a config or a manifest, and importing through
    docker-daemon may change the manifest encoding. Layer diff IDs and the full
    runtime configuration remain stable through that import.
    """
    with tarfile.open(archive) as source:
        member = source.extractfile("blobs/sha256/" + expected["config"].split(":")[1])
        if member is None:
            raise ValueError("Missing qualified image configuration")
        raw = member.read()
    if "sha256:" + hashlib.sha256(raw).hexdigest() != expected["config"]:
        raise ValueError("Qualified image configuration digest differs")
    config = json.loads(raw)

    def runtime_config(value: dict) -> dict:
        # Classic Docker includes optional zero-value fields which containerd
        # omits. Keep nonempty values (including nested labels and numeric zero).
        return {
            key: item
            for key, item in value.items()
            if item is not None and item is not False and item not in ("", [], {})
        }

    if (
        f"{actual['Os']}/{actual['Architecture']}" != expected["platform"]
        or actual["RootFS"]["Type"] != config["rootfs"]["type"]
        or (actual["RootFS"].get("Layers") or []) != config["rootfs"]["diff_ids"]
        or runtime_config(actual["Config"]) != runtime_config(config.get("config", {}))
    ):
        raise ValueError(
            "Loaded image layers, runtime configuration or platform differ"
        )


def publication_tags(
    image: str, mode: str, tag: str, latest: bool, date: str
) -> list[str]:
    if mode == "release":
        if not re.fullmatch(r"v\d+\.\d+\.\d+(?:[-+][A-Za-z0-9.-]+)?", tag):
            raise ValueError("Release publication requires a semantic version tag")
        tags = [tag]
    elif mode == "nightly":
        tags = ["nightly-" + date]
        if image in {"llm-katan", "anthropic-shim"}:
            tags.append("nightly")
    elif mode == "main":
        tags = [source_sha()]
    else:
        raise ValueError("PR artifacts cannot be published")
    if latest:
        tags.append("latest")
    return tags


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["definition", "seal", "load", "promote", "verify"]
    )
    parser.add_argument("--image", choices=sorted(DEFINITIONS))
    parser.add_argument("--images", default="[]", help="JSON list, used by load")
    parser.add_argument("--directory", type=Path, default=Path(".agent-harness/images"))
    parser.add_argument("--multiarch", action="store_true")
    parser.add_argument(
        "--mode", default="pr", choices=["pr", "main", "nightly", "release"]
    )
    parser.add_argument("--tag", default="")
    parser.add_argument("--latest", action="store_true")
    args = parser.parse_args()
    if args.command == "load":
        names = json.loads(args.images)
        if not isinstance(names, list) or not names or len(set(names)) != len(names):
            raise ValueError("Image inventory must be a nonempty unique list")
        receipts = []
        for name in names:
            if name not in DEFINITIONS:
                raise ValueError(f"Unknown image {name}")
            directory = args.directory / name
            subprocess.run(
                [
                    "gh",
                    "run",
                    "download",
                    os.environ["GITHUB_RUN_ID"],
                    "--repo",
                    os.environ["GITHUB_REPOSITORY"],
                    "--name",
                    f"ci-image-{name}",
                    "--dir",
                    str(directory),
                ],
                check=True,
            )
            manifest = verify(directory, name)
            receipts.append({"id": "image:" + name, "sha256": manifest["sha256"]})
            expected = next(
                item for item in manifest["images"] if item["platform"] == "linux/amd64"
            )
            target = f"semantic-router-ci/{name}:{manifest['source_sha']}"
            subprocess.run(
                [
                    "skopeo",
                    "copy",
                    "--override-os",
                    "linux",
                    "--override-arch",
                    "amd64",
                    f"oci-archive:{directory / 'image.tar'}",
                    f"docker-daemon:{target}",
                ],
                check=True,
            )
            actual = json.loads(
                subprocess.check_output(
                    ["docker", "image", "inspect", target], text=True
                )
            )[0]
            verify_loaded_image(directory / "image.tar", expected, actual)
            with Path(os.environ["GITHUB_ENV"]).open("a") as env:
                for variable in IMAGE_ENV.get(name, []):
                    env.write(f"{variable}={target}\n")
        receipts_path = (args.directory / "receipts.json").resolve()
        receipts_path.write_text(json.dumps(receipts, indent=2) + "\n")
        with Path(os.environ["GITHUB_ENV"]).open("a") as env:
            env.write("PREBUILT_RUNTIME_IMAGES=1\n")
            env.write(f"CI_IMAGE_RECEIPTS={receipts_path}\n")
        return
    if not args.image:
        parser.error("--image is required")
    context, dockerfile, supported = DEFINITIONS[args.image]
    platforms = supported if args.multiarch else ["linux/amd64"]
    if args.command == "definition":
        values = {
            "context": context,
            "dockerfile": dockerfile,
            "platforms": ",".join(platforms),
            "date": datetime.now(timezone.utc).strftime("%Y%m%d"),
        }
        with Path(os.environ["GITHUB_OUTPUT"]).open("a") as output:
            for key, value in values.items():
                output.write(f"{key}={value}\n")
    elif args.command == "seal":
        archive = args.directory / "image.tar"
        images = oci_images(archive)
        if {item["platform"] for item in images} != set(platforms):
            raise ValueError("Built platform inventory differs from the plan")
        manifest = {
            "schema": 1,
            "id": args.image,
            "source_sha": source_sha(),
            "context": context,
            "dockerfile": dockerfile,
            "build_args": os.environ["CI_IMAGE_BUILD_ARGS"].splitlines(),
            "mode": args.mode,
            "tag": args.tag,
            "date": os.environ.get("CI_IMAGE_BUILD_DATE", ""),
            "sha256": sha256(archive),
            "images": images,
        }
        (args.directory / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
    else:
        manifest = verify(args.directory, args.image)
        if args.command == "promote":
            if manifest["mode"] != args.mode or manifest["tag"] != args.tag:
                raise ValueError(
                    "Publication context differs from the tested candidate"
                )
            owner = os.environ["GITHUB_REPOSITORY_OWNER"].lower()
            for tag in publication_tags(
                args.image, args.mode, args.tag, args.latest, manifest["date"]
            ):
                subprocess.run(
                    [
                        "skopeo",
                        "copy",
                        "--all",
                        "--preserve-digests",
                        "--digestfile",
                        str(args.directory / "published-digest.txt"),
                        f"oci-archive:{args.directory / 'image.tar'}",
                        f"docker://ghcr.io/{owner}/semantic-router/{args.image}:{tag}",
                    ],
                    check=True,
                )
        print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
