"""Private working copy of a released package with a different runtime or checkpoint.

    python3 -m v2.eval.ix1.restage --package SRC --out DST [--runtime DIR] \
        [--checkpoint DIR --model-sha256 SHA]

DST hard-links every file of SRC (same filesystem) except the ones replaced:
``--runtime`` copies each ``decision2/*.py`` the package ships from DIR (e.g.
``v2/release/runtime``); ``--checkpoint`` copies the model files the manifest lists
(``model_files``) from a checkpoint directory of the same profile and records its
identity. MODEL_MANIFEST.json is rewritten so the package verifies: file digests,
packaged and loaded parameter counts, and the identity. The copy is for evaluation only
and is never published.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_count(path: Path) -> int:
    with path.open("rb") as stream:
        size = int.from_bytes(stream.read(8), "little")
        header = json.loads(stream.read(size))
    total = 0
    for name, meta in header.items():
        if name != "__metadata__":
            count = 1
            for dim in meta["shape"]:
                count *= dim
            total += count
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--model-sha256")
    args = parser.parse_args()
    src = args.package.resolve(strict=True)
    manifest = json.loads((src / "MODEL_MANIFEST.json").read_text(encoding="utf-8"))
    replaced: dict[str, Path] = {}
    if args.runtime:
        for name in manifest["files_sha256"]:
            parts = Path(name).parts
            if (
                len(parts) == 2
                and parts[0] == "decision2"
                and (args.runtime / parts[1]).is_file()
            ):
                replaced[name] = args.runtime / parts[1]
    if args.checkpoint:
        if not args.model_sha256:
            parser.error("--checkpoint needs --model-sha256")
        for name in manifest["model_files"]:
            replaced[name] = args.checkpoint / name
    args.out.mkdir(parents=True)
    for path in sorted(src.rglob("*")):
        relative = path.relative_to(src)
        if (
            relative.parts[0] == ".cache"
            or relative.as_posix() == "MODEL_MANIFEST.json"
        ):
            continue
        target = args.out / relative
        if path.is_dir():
            target.mkdir(exist_ok=True)
        elif relative.as_posix() in replaced:
            shutil.copyfile(replaced[relative.as_posix()], target)
        else:
            os.link(path, target)
    for name in replaced:
        digest = sha256(args.out / name)
        manifest["files_sha256"][name] = digest
        if name in manifest.get("runtime_files", {}):
            manifest["runtime_files"][name]["sha256"] = digest
    if args.checkpoint:
        parameters = manifest["parameters"]
        old = sum(parameters["packaged"].values())
        parameters["packaged"] = {
            group: sum(tensor_count(args.out / f) for f in files)
            for group, files in parameters["packaged_files"].items()
        }
        parameters["loaded"] += sum(parameters["packaged"].values()) - old
        manifest["identity"] = {
            "model_sha256": args.model_sha256,
            "fingerprint_files": {},
        }
        manifest["calibration"] = None
    (args.out / "MODEL_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"out": str(args.out), "replaced": sorted(replaced)}))


if __name__ == "__main__":
    main()
