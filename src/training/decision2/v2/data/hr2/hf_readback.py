"""Pin and verify an upload to the private training-data dataset (runs with the HF CLI's Python).

    hf_readback.py pin <repo> <parent sha> <commit title>
    hf_readback.py verify <repo> <revision> <prefix> <assembled tree> <read-back tree>

``pin`` prints the commit titled <commit title> and fails unless it sits directly on <parent sha>.
``verify`` compares the remote <prefix> file list, sizes and LFS SHA-256 at <revision>, and every read-back
file, with the assembled ``registry.json``; exit 1 on any difference.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.hf_api import RepoFile


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pin(repo: str, parent: str, title: str) -> None:
    commits = HfApi().list_repo_commits(repo, repo_type="dataset")
    for index, commit in enumerate(commits[:50]):
        if commit.title == title:
            older = commits[index + 1].commit_id if index + 1 < len(commits) else None
            if older != parent:
                sys.exit(f"upload commit {commit.commit_id} does not sit on {parent}")
            print(commit.commit_id)
            return
    sys.exit("upload commit not found")


def lfs_sha(item: RepoFile) -> str | None:
    lfs = getattr(item, "lfs", None)
    if lfs is None:
        return None
    return lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)


def verify(repo: str, revision: str, prefix: str, tree: Path, readback: Path) -> None:
    prefix = prefix.rstrip("/") + "/"
    registry = json.loads((tree / "registry.json").read_text(encoding="utf-8"))
    expected = {
        prefix + key.split("/", 1)[1]: meta for key, meta in registry["files"].items()
    }
    remote = {
        item.path: item
        for item in HfApi().list_repo_tree(
            repo,
            path_in_repo=prefix.rstrip("/"),
            recursive=True,
            revision=revision,
            repo_type="dataset",
        )
        if isinstance(item, RepoFile)
    }
    problems = []
    wanted = set(expected) | {prefix + "registry.json"}
    if set(remote) != wanted:
        problems.append(
            f"remote file set differs: missing {sorted(wanted - set(remote))}, "
            f"extra {sorted(set(remote) - wanted)}"
        )
    lfs_checked = 0
    for path, meta in sorted(expected.items()):
        local = readback / path[len(prefix) :]
        if (
            not local.is_file()
            or local.stat().st_size != meta["bytes"]
            or sha256(local) != meta["sha256"]
        ):
            problems.append(f"read-back differs: {path}")
        item = remote.get(path)
        if item is None:
            continue
        if item.size != meta["bytes"]:
            problems.append(f"remote size differs: {path}")
        remote_sha = lfs_sha(item)
        if remote_sha is not None:
            lfs_checked += 1
            if remote_sha != meta["sha256"]:
                problems.append(f"remote LFS SHA-256 differs: {path}")
    report = {
        "revision": revision,
        "files": len(remote),
        "registry_entries": len(expected),
        "read_back_sha256_equal": len(expected)
        - sum(p.startswith("read-back") for p in problems),
        "remote_lfs_sha256_checked": lfs_checked,
        "problems": problems,
    }
    print(json.dumps(report, indent=1, sort_keys=True))
    sys.exit(1 if problems else 0)


def main() -> None:
    command, repo, *rest = sys.argv[1:]
    if command == "pin":
        pin(repo, *rest)
    elif command == "verify":
        verify(repo, rest[0], rest[1], Path(rest[2]), Path(rest[3]))
    else:
        sys.exit(f"unknown command {command}")


if __name__ == "__main__":
    main()
