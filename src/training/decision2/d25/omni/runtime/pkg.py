"""Build a test package: a released Decision 2.5 package with the image-capable runtime files.

Every file of the original package is symlinked except the runtime code, which is copied from ``--code``;
``MODEL_MANIFEST.json`` gets the new files' SHA-256 and sizes (as the release build would), so
``verify="fast"`` and ``AutoModel.from_pretrained(..., trust_remote_code=True)`` work unchanged.

    python -m d25.omni.runtime.pkg --original ORIG --code d25/omni/runtime/package --out PKG
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from d25.omni.runtime.common import CODE_FILES

SKIP = {".cache", "MODEL_MANIFEST.json"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build(original: Path, code: Path, out: Path) -> dict:
    if out.exists():
        shutil.rmtree(out)
    partial = out.with_name(out.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    for item in sorted(original.iterdir()):
        if item.name in SKIP or item.name in CODE_FILES:
            continue
        (partial / item.name).symlink_to(item.resolve())
    for name in CODE_FILES:
        shutil.copyfile(code / name, partial / name)
    manifest = json.loads((original / "MODEL_MANIFEST.json").read_text())
    changed = {}
    for name in CODE_FILES:
        digest, size = sha256(partial / name), (partial / name).stat().st_size
        if manifest["files_sha256"].get(name) != digest:
            changed[name] = digest
        manifest["files_sha256"][name] = digest
        manifest["files_bytes"][name] = size
        manifest.setdefault("runtime", {}).setdefault("files_sha256", {})[name] = digest
    (partial / "MODEL_MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n")
    partial.rename(out)
    return {"package": str(out), "original": str(original), "changed": changed}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--original", required=True, type=Path)
    ap.add_argument("--code", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    print(json.dumps(build(args.original, args.code, args.out), indent=1))


if __name__ == "__main__":
    main()
