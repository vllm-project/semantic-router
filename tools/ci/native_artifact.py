#!/usr/bin/env python3
"""Package and verify the CPU libraries shared by one CI source revision."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LIBRARIES = (
    "candle-binding/target/release/libcandle_semantic_router.so",
    "ml-binding/target/release/libml_semantic_router.so",
    "nlp-binding/target/release/libnlp_binding.so",
    "onnx-binding/target/release/libonnx_semantic_router.so",
)
BUILD = {"candle": "cpu", "onnx": "dynamic", "profile": "release"}


def source_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def validate(manifest: dict, directory: Path, sha: str) -> None:
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise ValueError("native CI artifacts require Linux x86_64")
    if manifest.get("source_sha") != sha:
        raise ValueError("native artifact source SHA differs from the checkout")
    if manifest.get("platform") != "linux/amd64" or manifest.get("build") != BUILD:
        raise ValueError("native artifact platform or build settings differ")
    files = manifest.get("files", {})
    if set(files) != set(LIBRARIES):
        raise ValueError(
            "native artifact must contain exactly four registered libraries"
        )
    for name in LIBRARIES:
        candidate = directory / name
        if candidate.is_symlink() or not candidate.is_file():
            raise ValueError(f"native artifact library is missing or a symlink: {name}")
        if digest(candidate) != files[name]:
            raise ValueError(f"native artifact hash mismatch: {name}")


def prepare_ml_library() -> Path:
    """Use the verified CI library, or build the developer's local checkout."""
    target = ROOT / "ml-binding" / "target"
    if os.environ.get("PREBUILT_NATIVE_LIBS") == "1":
        directory = os.environ.get("NATIVE_ARTIFACT_DIR")
        if not directory:
            raise ValueError("PREBUILT_NATIVE_LIBS requires NATIVE_ARTIFACT_DIR")
        command = [sys.executable, __file__, "verify", "--directory", directory]
    else:
        command = [
            "cargo",
            "build",
            "--release",
            "--locked",
            "--manifest-path",
            str(ROOT / "ml-binding/Cargo.toml"),
            "--target-dir",
            str(target),
        ]
    subprocess.run(command, check=True)
    suffix = "dylib" if sys.platform == "darwin" else "so"
    return target / "release" / f"libml_semantic_router.{suffix}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("package", "load", "verify"))
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    directory = args.directory.resolve()
    receipt = directory / "manifest.json"
    sha = source_sha()
    if args.command == "package":
        manifest = {
            "source_sha": sha,
            "platform": "linux/amd64",
            "build": BUILD,
            "files": {name: digest(ROOT / name) for name in LIBRARIES},
        }
        validate(manifest, ROOT, sha)
        for name in LIBRARIES:
            destination = directory / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, destination)
        receipt.write_text(json.dumps(manifest, indent=2) + "\n")
        identity = [{"id": "native:cpu", "sha256": digest(receipt), "source_sha": sha}]
        (directory / "receipt.json").write_text(json.dumps(identity, indent=2) + "\n")
    else:
        manifest = json.loads(receipt.read_text())
        identity = json.loads((directory / "receipt.json").read_text())
        expected = [{"id": "native:cpu", "sha256": digest(receipt), "source_sha": sha}]
        if identity != expected:
            raise ValueError("native dependency receipt does not match its manifest")
        validate(manifest, directory if args.command == "load" else ROOT, sha)
        if args.command == "load":
            for name in LIBRARIES:
                destination = ROOT / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(directory / name, destination)
    print(f"Verified CPU native libraries for {sha}")


if __name__ == "__main__":
    main()
