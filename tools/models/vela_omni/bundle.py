#!/usr/bin/env python3
"""Verify and stage a complete tensor-only artifact using only Python's stdlib.

The official model repositories contain native weights, not these ONNX graphs.
Only export.py can promote a pending export after reference parity. This command
checks that receipt and all digests before copying an artifact for image builds.
It never downloads a model or executes published Python code.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from contract import (
    MANIFEST,
    PENDING_MANIFEST,
    artifact_manifest,
    inventory,
    safe_file,
    verify_inventory,
    write_json,
)


def verify(directory: Path) -> dict:
    manifest = json.loads((directory / MANIFEST).read_text())
    text = manifest["processors"]["text"]
    expected = artifact_manifest(
        manifest["variant"], text["padding_side"], text["pad_token_id"]
    )
    expected["files"] = manifest["files"]
    expected["reference_parity"]["passed"] = True
    if manifest != expected:
        raise ValueError("artifact differs from the pinned tensor contract")
    if (directory / PENDING_MANIFEST).exists():
        raise ValueError("artifact still contains an unverified pending manifest")
    verify_inventory(directory, manifest)
    if inventory(directory) != manifest["files"]:
        raise ValueError("artifact contains files absent from its inventory")
    required = {graph["file"] for graph in manifest["graphs"].values()} | {
        manifest["tokenizer"],
        manifest["processors"]["audio"]["file"],
        manifest["reference_parity"]["file"],
    }
    if not required <= manifest["files"].keys():
        raise ValueError("artifact inventory omits a required graph or processor")
    if any(
        Path(name).suffix in (".py", ".pyc", ".safetensors")
        for name in manifest["files"]
    ):
        raise ValueError("runtime artifact contains native source or weights")
    report = json.loads(
        safe_file(directory, manifest["reference_parity"]["file"]).read_text()
    )
    if (
        report["passed"] is not True
        or report["source"] != manifest["source"]
        or report["variant"] != manifest["variant"]
        or not report["tests"]
        or any(test["passed"] is not True for test in report["tests"])
    ):
        raise ValueError("artifact reference parity receipt is invalid")
    names = {test["name"] for test in report["tests"]}
    required_checks = {
        "text/0",
        "text/1",
        "text/2",
        "text/3",
        "text/overflow-rejected",
        "text/padding",
        "image/0",
        "image/1",
        "image/2",
    } | {
        f"audio/{index}/{part}"
        for index in range(5 if manifest["variant"] == "mini" else 4)
        for part in ("end-to-end", "clap")
    }
    if manifest["variant"] == "mini":
        required_checks |= {"text/instruction-query", "text/instruction-document"}
    if not required_checks <= names:
        raise ValueError("reference parity receipt omits a required modality case")
    return manifest


def stage(source: Path, destination: Path, *, keep_golden: bool = False) -> None:
    manifest = verify(source)
    if destination.exists():
        raise ValueError(f"destination already exists: {destination}")
    if destination.resolve().is_relative_to(source.resolve()):
        raise ValueError("destination must not be inside source")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}-", dir=destination.parent
    ) as temporary:
        pending = Path(temporary) / "artifact"
        pending.mkdir()
        for name in manifest["files"]:
            if not keep_golden and name.startswith("golden/"):
                continue
            target = safe_file(pending, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(safe_file(source, name), target)
        manifest["files"] = inventory(pending)
        write_json(pending / MANIFEST, manifest)
        verify(pending)
        pending.rename(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("verify", help="verify a prepared artifact offline")
    check.add_argument("directory", type=Path)
    copy = commands.add_parser("stage", help="atomically copy a verified artifact")
    copy.add_argument("source", type=Path)
    copy.add_argument("destination", type=Path)
    copy.add_argument("--keep-golden", action="store_true")
    args = parser.parse_args()
    if args.command == "verify":
        manifest = verify(args.directory)
        print(f"Verified {manifest['variant']}: {args.directory}")
    else:
        stage(args.source, args.destination, keep_golden=args.keep_golden)
        print(f"Staged verified tensor artifact: {args.destination}")


if __name__ == "__main__":
    main()
