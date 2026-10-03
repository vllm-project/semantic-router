"""Publish a staged Decision Index submission dataset as one commit (public, ungated).

    <hf-cli python> -m v2.eval.ix1.publish_dataset --repo ORG/NAME --dir <staged dir> --message MSG [--create | --update]

Refuses unless the directory has a README.md, every ``runs/<name>/scores.json`` says complete with
150,317 completed requests, every ``runs/<name>/harness/weights-vs-release.json`` matches, and no
staged JSON, log or Markdown file names a node path. ``--create`` creates the repository public and
ungated and refuses if it exists. ``--update`` commits a directory holding only changed files (for example
the card and a run's ``harness/`` files after a pin moves to a runtime-only successor) to the existing public
repository: every staged ``weights-vs-release.json`` must match and no file may name a node path; files not in the
directory stay as they are. Prints the commit SHA. The token comes from the local Hugging Face login; it is never
read or printed here.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

NODE_PATH = re.compile(r"/data/|/root/|/home/|/mnt/|/tmp/")
EXPECTED = 150317


def check(directory: Path) -> list[str]:
    runs = sorted(p for p in (directory / "runs").iterdir() if p.is_dir())
    if not (directory / "README.md").is_file() or not runs:
        raise SystemExit("README.md or runs/ missing")
    for run in runs:
        scores = json.loads((run / "scores.json").read_text())
        if not scores.get("complete") or scores.get("completed") != EXPECTED:
            raise SystemExit(f"{run.name}: not complete")
        weights = json.loads((run / "harness" / "weights-vs-release.json").read_text())
        if not weights.get("match"):
            raise SystemExit(f"{run.name}: weights differ from the release")
        if not (run / "results.jsonl.gz").is_file():
            raise SystemExit(f"{run.name}: results.jsonl.gz missing")
    for path in directory.rglob("*"):
        if path.is_file() and path.suffix in (".json", ".log", ".md"):
            if NODE_PATH.search(path.read_text(encoding="utf-8", errors="replace")):
                raise SystemExit(f"{path.relative_to(directory)} names a node path")
    return [run.name for run in runs]


def check_update(directory: Path) -> list[str]:
    staged = sorted(p for p in directory.rglob("*") if p.is_file())
    if not staged:
        raise SystemExit("nothing staged")
    for path in staged:
        if path.name == "weights-vs-release.json" and not json.loads(
            path.read_text()
        ).get("match"):
            raise SystemExit(
                f"{path.relative_to(directory)}: weights differ from the release"
            )
        if path.suffix in (".json", ".log", ".md"):
            if NODE_PATH.search(path.read_text(encoding="utf-8", errors="replace")):
                raise SystemExit(f"{path.relative_to(directory)} names a node path")
    return [str(p.relative_to(directory)) for p in staged]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo", required=True)
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--message", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--create", action="store_true")
    mode.add_argument("--update", action="store_true")
    args = parser.parse_args()
    names = check_update(args.dir) if args.update else check(args.dir)
    from huggingface_hub import HfApi

    api = HfApi()
    if args.create:
        if api.repo_exists(args.repo, repo_type="dataset"):
            raise SystemExit(f"{args.repo} exists")
        api.create_repo(args.repo, repo_type="dataset", private=False)
    info = api.dataset_info(args.repo)
    if info.private or info.gated:
        raise SystemExit(f"{args.repo} is private or gated")
    commit = api.upload_folder(
        repo_id=args.repo,
        repo_type="dataset",
        folder_path=str(args.dir),
        commit_message=args.message,
    )
    print(
        json.dumps(
            {
                "repo": args.repo,
                "files" if args.update else "runs": names,
                "commit": commit.oid,
                "url": commit.commit_url,
            }
        )
    )


if __name__ == "__main__":
    main()
