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

import provider_mocker_image as mocker

DUAL = ["linux/amd64", "linux/arm64"]
DECISION_RUNTIME_BASES = {
    "decision-runtime-cpu": (
        "python@sha256:229a2c5bfa27522db7815ea81f9bed70af17ccb9de9fc7ad142b1877b5830d36",
        "cpu",
        "https://download.pytorch.org/whl/cpu",
    ),
    "decision-runtime-rocm": (
        "vllm/vllm-openai-rocm@sha256:1fd21abe66455b4df5a2e83629e97cdcc9d58913b16052d8118b92b239792339",
        "rocm",
        "",
    ),
}
DEFINITIONS = {
    "dashboard": (".", "dashboard/backend/Dockerfile", DUAL),
    "decision-runtime-cpu": (
        ".",
        "src/vllm-sr/decision_runtime/image/Dockerfile",
        ["linux/amd64"],
    ),
    "decision-runtime-rocm": (
        ".",
        "src/vllm-sr/decision_runtime/image/Dockerfile",
        ["linux/amd64"],
    ),
    "extproc": (".", "tools/docker/Dockerfile.extproc", DUAL),
    "extproc-rocm": (".", "tools/docker/Dockerfile.extproc-rocm", ["linux/amd64"]),
    mocker.IMAGE: (mocker.CONTEXT, mocker.CONTEXT + "/Dockerfile", DUAL),
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
    mocker.IMAGE: ["E2E_PREBUILT_PROVIDER_MOCKER_IMAGE", "PROVIDER_MOCKER_IMAGE"],
}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_sha() -> str:
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if os.environ.get("GITHUB_SHA", actual) != actual:
        raise ValueError("Checkout differs from the workflow source revision")
    return actual


def decision_build_args(image: str, revision: str) -> tuple[str, ...]:
    """Bind a Decision image to one immutable base and one source commit."""
    if image not in DECISION_RUNTIME_BASES:
        raise ValueError(f"Not a Decision runtime image: {image}")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Decision runtime image requires a full source SHA")
    base, backend, torch_index = DECISION_RUNTIME_BASES[image]
    return (
        f"BASE_IMAGE={base}",
        "PYTHON_BIN=python3",
        f"BACKEND={backend}",
        f"TORCH_INDEX_URL={torch_index}",
        f"SOURCE_REVISION={revision}",
        "SOURCE_STATE=clean",
    )


def verify_decision_build(manifest: dict, archive: Path, image: str) -> None:
    """Reject a stale base, changed backend, or spoofed source label."""
    expected = dict(
        item.split("=", 1) for item in decision_build_args(image, source_sha())
    )
    recorded: dict[str, str] = {}
    for item in manifest.get("build_args", []):
        if not isinstance(item, str) or "=" not in item:
            raise ValueError("Decision image build arguments are malformed")
        key, value = item.split("=", 1)
        if not key or key in recorded:
            raise ValueError("Decision image build arguments are duplicated")
        recorded[key] = value
    if any(recorded.get(key) != value for key, value in expected.items()):
        raise ValueError("Decision image build arguments differ from pinned source")

    base, backend, _ = DECISION_RUNTIME_BASES[image]
    required_labels = {
        "org.opencontainers.image.base.name": base,
        "org.opencontainers.image.revision": manifest["source_sha"],
        "ai.vllm-sr.decision.source-state": "clean",
        "ai.vllm-sr.decision.backend": backend,
    }
    with tarfile.open(archive) as source:
        for descriptor in manifest["images"]:
            digest = descriptor["config"]
            member = source.extractfile("blobs/sha256/" + digest.split(":", 1)[1])
            if member is None:
                raise ValueError("Decision image configuration is missing")
            labels = json.load(member).get("config", {}).get("Labels", {})
            if any(labels.get(key) != value for key, value in required_labels.items()):
                raise ValueError("Decision image source or backend labels differ")


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
    if image in DECISION_RUNTIME_BASES:
        verify_decision_build(manifest, archive, image)
    if image == mocker.IMAGE:
        if manifest.get("inputs_sha256") != mocker.input_fingerprint():
            raise ValueError("provider-mocker build inputs differ from this checkout")
        if manifest.get("acquisition") == "published":
            mocker.validate_acquisition(manifest, manifest)
            if manifest["images"] != manifest.get("published_images"):
                raise ValueError(
                    "imported provider-mocker content differs from registry receipt"
                )
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
    if image == "decision-runtime-rocm":
        raise ValueError("Decision ROCm is not supported by the hosted image publisher")
    if image == "decision-runtime-cpu":
        if mode not in {"main", "nightly", "release"}:
            raise ValueError("PR artifacts cannot be published")
        # This is a source-tagged candidate, not a qualified CLI default.
        # Do not publish a latest/version alias without a digest binding.
        return [source_sha()]
    if image == mocker.IMAGE:
        if mode != "main":
            raise ValueError(
                "provider-mocker is only published after main qualification"
            )
        return [mocker.input_tag(mocker.input_fingerprint())]
    if mode == "release":
        if not re.fullmatch(r"v\d+\.\d+\.\d+(?:[-+][A-Za-z0-9.-]+)?", tag):
            raise ValueError("Release publication requires a semantic version tag")
        tags = [tag]
    elif mode == "nightly":
        tags = ["nightly-" + date]
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
        "command",
        choices=[
            "definition",
            "build-args",
            "seal",
            "load",
            "promote",
            "verify",
            "acquire",
        ],
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
    if args.command == "acquire":
        acquire_published(args.directory)
        return
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
    if args.command == "build-args":
        print("\n".join(decision_build_args(args.image, source_sha())))
        return
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
            output.write("labels<<IMAGE_LABELS\n")
            if args.image == mocker.IMAGE:
                for key, value in mocker.labels(args.mode, source_sha()).items():
                    output.write(f"{key}={value}\n")
            output.write("IMAGE_LABELS\n")
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
        if args.image == mocker.IMAGE:
            manifest["inputs_sha256"] = mocker.input_fingerprint()
            manifest["acquisition"] = "candidate"
        (args.directory / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
    else:
        manifest = verify(args.directory, args.image)
        if args.command == "promote":
            if (
                args.image == mocker.IMAGE
                and manifest.get("acquisition") != "candidate"
            ):
                raise ValueError("A reused provider-mocker cannot be republished")
            if manifest["mode"] != args.mode or manifest["tag"] != args.tag:
                raise ValueError(
                    "Publication context differs from the tested candidate"
                )
            if args.image == mocker.IMAGE:
                publication_tags(args.image, args.mode, args.tag, args.latest, "")
                try:
                    existing = mocker.resolve_published(
                        {
                            "id": mocker.IMAGE,
                            "source": "published",
                            "inputs_sha256": manifest["inputs_sha256"],
                        }
                    )
                except mocker.PublicationUnavailableError:
                    pass
                else:
                    print(json.dumps({"reused_qualified_publication": existing["ref"]}))
                    return
            owner = os.environ["GITHUB_REPOSITORY_OWNER"].lower()
            tags = publication_tags(
                args.image, args.mode, args.tag, args.latest, manifest["date"]
            )
            for tag in tags:
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
            if args.image == "decision-runtime-cpu":
                digest = (args.directory / "published-digest.txt").read_text().strip()
                if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
                    raise ValueError("Decision publication returned an invalid digest")
                if len(tags) != 1 or tags[0] != manifest["source_sha"]:
                    raise ValueError("Decision publication must use its source SHA")
                if [item["platform"] for item in manifest["images"]] != ["linux/amd64"]:
                    raise ValueError(
                        "Decision CPU publication has an unexpected platform"
                    )
                receipt = {
                    "schema_version": 1,
                    "image": args.image,
                    "source_sha": manifest["source_sha"],
                    "mode": args.mode,
                    "tag": args.tag,
                    "digest": digest,
                    "ref": f"ghcr.io/{owner}/semantic-router/{args.image}@{digest}",
                    "archive_sha256": manifest["sha256"],
                    "platform": "linux/amd64",
                }
                (args.directory / "published.json").write_text(
                    json.dumps(receipt, indent=2) + "\n"
                )
        print(json.dumps(manifest, indent=2))


def acquire_published(directory: Path) -> None:
    """Import a planned registry digest into the existing immutable OCI handoff."""
    record = json.loads(os.environ["PUBLISHED_IMAGE"])
    mocker.validate_acquisition(record, record)
    if (
        record["id"] != mocker.IMAGE
        or record["inputs_sha256"] != mocker.input_fingerprint()
    ):
        raise ValueError("published image does not match provider-mocker build inputs")
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / "image.tar"
    subprocess.run(
        [
            "skopeo",
            "copy",
            "--all",
            "--preserve-digests",
            "docker://" + record["ref"],
            "oci-archive:" + str(archive),
        ],
        check=True,
    )
    images = oci_images(archive)
    if images != record["images"]:
        raise ValueError(
            "downloaded provider-mocker differs from planned registry content"
        )
    manifest = {
        **record,
        "schema": 1,
        "acquisition": "published",
        "source_sha": source_sha(),
        "context": mocker.CONTEXT,
        "dockerfile": mocker.CONTEXT + "/Dockerfile",
        "sha256": sha256(archive),
        "published_images": record["images"],
        "images": images,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
