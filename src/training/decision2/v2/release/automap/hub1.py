"""Hugging Face steps for the Decision 1.0 remote-code update (run on a node with the HF CLI venv).

  listing --repo R --stage DIR --revision SHA   every file's LFS SHA-256 / blob ID at SHA
  check   --repo R --stage DIR             head unchanged since staging; no open PR touches the files
  pr      --repo R --stage DIR             open one PR with exactly the staged files, parent = staged head
  merge   --repo R --pr N --stage DIR      merge our PR if the head and open PRs still allow it
  readback --repo R --revision SHA --stage DIR --before LISTING.json
                                           every staged file has its staged SHA-256 at SHA, and every
                                           other file keeps its LFS SHA-256 / blob ID from LISTING.json

Every step prints one JSON receipt. Nothing here changes repository visibility or collections.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

TITLE = "Load with 🤗 Transformers (trust_remote_code)"
DESCRIPTION = (
    "Adds the model's own inference code so `AutoModel.from_pretrained(repo, trust_remote_code=True)` and "
    '`pipeline("decision", ...)` run it locally; `system_one(state=..., questions=...)` takes and returns '
    "System One request and response bodies. `config.json` keeps every Decision key and gains "
    "`model_type`, `architectures`, `auto_map` and `custom_pipelines`; the card gains a short "
    '"Use with 🤗 Transformers" section. Weights, tokenizer and every other file are unchanged.'
)


def stage_receipt(stage: Path) -> dict[str, Any]:
    return json.loads((stage / "STAGE.json").read_text(encoding="utf-8"))


def listing(api: HfApi, repo: str, revision: str) -> dict[str, dict[str, Any]]:
    info = api.model_info(repo, revision=revision, files_metadata=True)
    return {
        s.rfilename: {
            "lfs_sha256": s.lfs.sha256 if s.lfs else None,
            "blob_id": s.blob_id,
            "size": s.size,
        }
        for s in info.siblings
    }


def open_prs_touching(
    api: HfApi, repo: str, paths: set[str], mine: int | None = None
) -> list[dict[str, Any]]:
    found = []
    for discussion in api.get_repo_discussions(repo):
        if (
            not discussion.is_pull_request
            or discussion.status != "open"
            or discussion.num == mine
        ):
            continue
        details = api.get_discussion_details(repo, discussion.num)
        touched = {
            line.split(" b/", 1)[1]
            for line in (details.diff or "").splitlines()
            if line.startswith("diff --git ") and " b/" in line
        }
        found.append(
            {
                "num": discussion.num,
                "author": discussion.author,
                "overlap": sorted(touched & paths),
            }
        )
    return found


def check(
    api: HfApi, repo: str, stage: Path, mine: int | None = None
) -> dict[str, Any]:
    receipt = stage_receipt(stage)
    head = api.model_info(repo).sha
    paths = {op["path"] for op in receipt["operations"]}
    others = open_prs_touching(api, repo, paths, mine)
    return {
        "repo": repo,
        "staged_head": receipt["head"],
        "current_head": head,
        "head_unchanged": head == receipt["head"],
        "open_prs": others,
        "clear": head == receipt["head"]
        and not any(item["overlap"] for item in others),
    }


def create_pr(api: HfApi, repo: str, stage: Path) -> dict[str, Any]:
    status = check(api, repo, stage)
    if not status["clear"]:
        raise SystemExit(json.dumps({"refused": status}))
    receipt = stage_receipt(stage)
    operations = []
    for op in receipt["operations"]:
        path = stage / "files" / op["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != op["sha256"]:
            raise SystemExit(f"staged file changed: {op['path']}")
        operations.append(
            CommitOperationAdd(path_in_repo=op["path"], path_or_fileobj=str(path))
        )
    commit = api.create_commit(
        repo,
        operations=operations,
        commit_message=TITLE,
        commit_description=DESCRIPTION,
        create_pr=True,
        parent_commit=receipt["head"],
    )
    return {
        "repo": repo,
        "pr_url": commit.pr_url,
        "pr_num": commit.pr_num,
        "pr_revision": commit.oid,
        "parent_commit": receipt["head"],
        "files": {op["path"]: op["sha256"] for op in receipt["operations"]},
    }


def merge(api: HfApi, repo: str, number: int, stage: Path) -> dict[str, Any]:
    status = check(api, repo, stage, mine=number)
    if not status["clear"]:
        raise SystemExit(json.dumps({"refused": status}))
    details = api.get_discussion_details(repo, number)
    if details.status != "open" or not details.is_pull_request:
        raise SystemExit(f"PR {number} is not open")
    api.merge_pull_request(
        repo, number, comment="Parity, card and Hub smoke checks passed; merging."
    )
    return {"repo": repo, "merged_pr": number, "head_after": api.model_info(repo).sha}


def readback(
    api: HfApi, repo: str, revision: str, stage: Path, before: Path
) -> dict[str, Any]:
    receipt = stage_receipt(stage)
    staged = {op["path"]: op["sha256"] for op in receipt["operations"]}
    previous = json.loads(before.read_text(encoding="utf-8"))
    previous = previous.get(repo, previous)
    current = listing(api, repo, revision)
    problems = []
    for path, digest in staged.items():
        local = hf_hub_download(repo, path, revision=revision, force_download=True)
        if hashlib.sha256(Path(local).read_bytes()).hexdigest() != digest:
            problems.append(f"{path}: content differs from the staged file")
    for path, meta in previous.items():
        if path in staged:
            continue
        now = current.get(path)
        if now is None:
            problems.append(f"{path}: missing")
        elif (now["lfs_sha256"], now["blob_id"]) != (
            meta["lfs_sha256"],
            meta["blob_id"],
        ):
            problems.append(f"{path}: changed")
    extra = sorted(set(current) - set(previous) - set(staged))
    if extra:
        problems.append(f"unexpected files: {extra}")
    return {
        "repo": repo,
        "revision": revision,
        "staged_files_verified": len(staged),
        "unchanged_files_verified": len(set(previous) - set(staged)),
        "weights_lfs_sha256": {
            p: m["lfs_sha256"] for p, m in current.items() if m["lfs_sha256"]
        },
        "problems": problems,
        "passed": not problems,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("step", choices=("listing", "check", "pr", "merge", "readback"))
    parser.add_argument("--repo", required=True)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--pr", type=int)
    parser.add_argument("--revision")
    parser.add_argument("--before", type=Path)
    args = parser.parse_args()
    api = HfApi()
    if args.step == "listing":
        result = {args.repo: listing(api, args.repo, args.revision)}
    elif args.step == "check":
        result = check(api, args.repo, args.stage)
    elif args.step == "pr":
        result = create_pr(api, args.repo, args.stage)
    elif args.step == "merge":
        result = merge(api, args.repo, args.pr, args.stage)
    else:
        result = readback(api, args.repo, args.revision, args.stage, args.before)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
