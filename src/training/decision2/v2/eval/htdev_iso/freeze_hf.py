"""Snapshot the private training dataset at HEAD plus every file version HEAD no longer has.

    python3 -m v2.eval.htdev_iso.freeze_hf --since <base commit> --out <dir>

Downloads the full HEAD snapshot to <out>/head-<sha8>/ and, for each commit after
<base> (exclusive), the files whose blob at that commit differs from HEAD (or that
HEAD no longer has) to <out>/hist/<sha8>/. Writes <out>/hf-freeze.json (revisions,
per-commit file counts; paths only). Needs network (run with the hf-cli Python).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = "llm-semantic-router/decision-2.0-training-data"


def main(argv: list[str] | None = None) -> int:
    from huggingface_hub import HfApi, snapshot_download

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--since", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    api = HfApi()
    commits = api.list_repo_commits(REPO, repo_type="dataset")
    head = commits[0].commit_id
    newer = []
    for commit in commits:
        if commit.commit_id.startswith(args.since):
            break
        newer.append(commit)
    else:
        raise SystemExit(f"base {args.since} not found")

    def tree(rev: str) -> dict[str, str]:
        return {
            f.path: (f.lfs.sha256 if getattr(f, "lfs", None) else f.blob_id)
            for f in api.list_repo_tree(
                REPO, repo_type="dataset", revision=rev, recursive=True
            )
            if hasattr(f, "size")
        }

    head_tree = tree(head)
    snapshot_download(
        REPO,
        repo_type="dataset",
        revision=head,
        local_dir=str(args.out / f"head-{head[:8]}"),
    )
    history = []
    for commit in reversed(newer[1:]):
        own = tree(commit.commit_id)
        parent = tree(commits[commits.index(commit) + 1].commit_id)
        changed = sorted(
            p for p, h in own.items() if parent.get(p) != h and head_tree.get(p) != h
        )
        if changed:
            snapshot_download(
                REPO,
                repo_type="dataset",
                revision=commit.commit_id,
                local_dir=str(args.out / "hist" / commit.commit_id[:8]),
                allow_patterns=changed,
            )
        history.append(
            {
                "commit": commit.commit_id,
                "created_at": commit.created_at.isoformat(),
                "versions_not_in_head": len(changed),
            }
        )
    record = {
        "schema": "htdev-iso-hf-freeze/1",
        "repo": REPO,
        "head": head,
        "head_files": len(head_tree),
        "base_exclusive": args.since,
        "history": history,
    }
    (args.out / "hf-freeze.json").write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k != "history"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
