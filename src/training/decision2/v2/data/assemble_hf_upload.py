"""Assemble a private HF dataset upload folder from frozen arm artifacts.

The spec is a JSON list of ``{"src": path, "dst": relative path}``. JSON files
are rewritten with every string value that is an absolute filesystem path
replaced by its basename, so node layouts never reach the dataset. A
``registry.json`` with the SHA-256 of every destination file is written last.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any


def _strip_paths(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _strip_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_strip_paths(item) for item in value]
    if isinstance(value, str) and value.startswith("/") and "/" in value[1:]:
        return Path(value).name
    return value


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assemble(spec: list[dict[str, str]], out_dir: Path) -> dict[str, Any]:
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} exists")
    out_dir.mkdir(parents=True, mode=0o700)
    files = {}
    for item in spec:
        src, dst = Path(item["src"]), out_dir / item["dst"]
        if dst.exists() or ".." in Path(item["dst"]).parts:
            raise ValueError(f"bad or duplicate destination {item['dst']}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.suffix == ".json":
            payload = _strip_paths(json.loads(src.read_text(encoding="utf-8")))
            dst.write_text(
                json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
        else:
            shutil.copyfile(src, dst)
        os.chmod(dst, 0o600)
        files[item["dst"]] = {"sha256": sha_file(dst), "bytes": dst.stat().st_size}
    registry = {
        "schema": "decision2-hf-v2-registry/v1",
        "files": dict(sorted(files.items())),
    }
    first = next(iter(sorted(files)), "v2/x").split("/")[0]
    target = out_dir / first / "registry.json"
    target.write_text(
        json.dumps(registry, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    return registry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    registry = assemble(json.loads(args.spec.read_text(encoding="utf-8")), args.out_dir)
    print(json.dumps({"files": len(registry["files"])}, sort_keys=True))


if __name__ == "__main__":
    main()
