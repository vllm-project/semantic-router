"""A runtime-only revision of a d3 package: same weights and card, the runtime files of this tree.

    python -m d25.vega.latency.package --package <released package dir> --out <new dir>

Unchanged files are hard-linked (copied across file systems); ``--files`` (default: the fast-path runtime files)
come from ``release/package_d3``; ``MODEL_MANIFEST.json`` gets their hashes and sizes in
``files_sha256`` / ``files_bytes`` / ``runtime.files_sha256`` and a new ``built_utc``. ``identity`` (weights,
tokenizer, decision config) is unchanged. The result is checked with the runtime's own full verification.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from d25.vega.release.build import D3_CODE_FILES, D3_FORMAT_FILE, sha256_file

SOURCE = Path(__file__).resolve().parents[1] / "release" / "package_d3"
MANIFEST = "MODEL_MANIFEST.json"
FAST_FILES = ("d3_runtime.py", "d3_fast.py", "d3_kernels.py")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--package", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--files", nargs="+", default=list(FAST_FILES))
    args = ap.parse_args(argv)
    code = tuple(args.files)
    if args.out.exists():
        raise SystemExit(f"{args.out} exists")
    for path in sorted(args.package.rglob("*")):
        if not path.is_file() or path.name == MANIFEST or ".cache" in path.parts:
            continue
        rel = path.relative_to(args.package)
        target = args.out / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if rel.as_posix() in code:
            continue
        try:
            os.link(path, target)
        except OSError:
            shutil.copy2(path, target)
    for name in code:
        shutil.copyfile(SOURCE / name, args.out / name)
    manifest = json.loads((args.package / MANIFEST).read_text(encoding="utf-8"))
    files = sorted(
        p.relative_to(args.out).as_posix()
        for p in args.out.rglob("*")
        if p.is_file() and ".cache" not in p.parts
    )
    old = set(manifest["files_sha256"])
    manifest["files_sha256"] = {
        n: (
            manifest["files_sha256"][n]
            if n in old and n not in code
            else sha256_file(args.out / n)
        )
        for n in files
    }
    manifest["files_bytes"] = {n: (args.out / n).stat().st_size for n in files}
    manifest["runtime"]["files_sha256"] = {
        n: manifest["files_sha256"][n] for n in (*D3_CODE_FILES, D3_FORMAT_FILE)
    }
    manifest["built_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (args.out / MANIFEST).write_text(
        json.dumps(manifest, indent=1) + "\n", encoding="utf-8"
    )
    import sys

    sys.path.insert(0, str(args.out))
    import d3_runtime

    d3_runtime.verify_package(args.out, "full")
    changed = sorted(
        n
        for n in code
        if n not in old
        or manifest["files_sha256"][n]
        != json.loads((args.package / MANIFEST).read_text())["files_sha256"].get(n)
    )
    print(
        json.dumps(
            {
                "out": str(args.out),
                "files": len(files),
                "changed_or_added": changed,
                "verify": "full ok",
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
