"""Verify local official-base snapshots against the Hub audit receipt (stdlib only).

Every file the Hub lists must exist locally with the listed size and either the LFS
SHA-256 (weights) or the git blob SHA-1 (small files). Prints one JSON receipt with each
directory's weight SHA-256 list and a tree hash over (path, size, sha256).
Usage: python3 verify_bases.py RECEIPT.json NAME=DIR [NAME=DIR ...]
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def digests(path: Path) -> tuple[str, str]:
    sha256, blob = hashlib.sha256(), hashlib.sha1()
    blob.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            sha256.update(block)
            blob.update(block)
    return sha256.hexdigest(), blob.hexdigest()


def verify(entry: dict, root: Path) -> dict:
    rows, problems = [], []
    for item in entry["files"]:
        local = root / item["path"]
        if not local.is_file():
            if not item["path"].startswith((".eval_results/",)):
                problems.append(f"missing {item['path']}")
            continue
        sha256, blob = digests(local)
        size = local.stat().st_size
        ok = size == item["size"] and (
            sha256 == item["lfs_sha256"]
            if item["lfs_sha256"]
            else blob == item["blob_id"]
        )
        if not ok:
            problems.append(f"mismatch {item['path']}")
        rows.append({"path": item["path"], "size": size, "sha256": sha256})
    tree = hashlib.sha256(
        "".join(
            f"{r['path']}\0{r['size']}\0{r['sha256']}\n"
            for r in sorted(rows, key=lambda r: r["path"])
        ).encode()
    ).hexdigest()
    return {
        "repo": entry["repo"],
        "revision": entry["revision"],
        "dir": str(root),
        "files": len(rows),
        "tree_sha256": tree,
        "weights": {
            r["path"]: r["sha256"] for r in rows if r["path"].endswith(".safetensors")
        },
        "ok": not problems,
        "problems": problems,
    }


def main() -> None:
    receipt = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    repos = {
        entry["repo"].split("/")[-1]: entry
        for entry in receipt["repos"]
        if "files" in entry
    }
    results = []
    for spec in sys.argv[2:]:
        name, directory = spec.split("=", 1)
        results.append(verify(repos[name], Path(directory)))
    print(
        json.dumps(
            {"schema": "decision2-27b-moe-base-verify/1", "bases": results}, indent=1
        )
    )
    sys.exit(0 if all(r["ok"] for r in results) else 3)


if __name__ == "__main__":
    main()
