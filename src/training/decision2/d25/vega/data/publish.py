"""Publish a built mixture to the shared node path and the private HF dataset, or fetch and verify it.

    python -m d25.vega.data.publish push  --src DIR --dest /data/d25/shared/data/v1/M1 --path-in-repo v1/M1
    python -m d25.vega.data.publish fetch --dest /data/d25/shared/data/v1/M1 --path-in-repo v1/M1

Every file is checked against the sha256 recorded in the mixture manifest; a VERIFIED marker is written
only when all files match.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from d25.vega.data.util import sha256_file

REPO = "vllm-sr/decision-2.5-vega-training-data"


def verify(directory: Path) -> list[str]:
    manifest = json.loads((directory / "manifest.json").read_text())
    bad = []
    for name, info in manifest["files"].items():
        path = directory / name
        if not path.exists() or sha256_file(path) != info["sha256"]:
            bad.append(name)
    return bad


def push(
    src: Path, dest: Path, path_in_repo: str, upload: bool, name: str | None = None
) -> None:
    bad = verify(src)
    if bad:
        sys.exit(f"source does not match its manifest: {bad}")
    tmp = dest.with_name(dest.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.copytree(src, tmp, ignore=shutil.ignore_patterns("VERIFIED", "SUPERSEDED"))
    if name:
        manifest = json.loads((tmp / "manifest.json").read_text())
        manifest["name"] = name
        (tmp / "manifest.json").write_text(
            json.dumps(manifest, indent=1, ensure_ascii=False) + "\n"
        )
    if verify(tmp):
        sys.exit("copy verification failed")
    (tmp / "VERIFIED").write_text("all files match manifest.json sha256\n")
    if dest.exists():
        shutil.rmtree(dest)
    os.replace(tmp, dest)
    print("shared copy verified:", dest, flush=True)
    if upload:
        from huggingface_hub import HfApi

        api = HfApi(token=os.environ["HF_TOKEN"])
        api.create_repo(REPO, repo_type="dataset", private=True, exist_ok=True)
        info = api.upload_folder(
            repo_id=REPO,
            repo_type="dataset",
            folder_path=str(dest),
            path_in_repo=path_in_repo,
            allow_patterns=["*.jsonl.gz", "manifest.json", "*.md"],
            commit_message=f"{path_in_repo}: mixture upload",
        )
        print("uploaded:", info, flush=True)


def wait_for(path_in_repo: str, seconds: int) -> None:
    """Poll the Hub until <path_in_repo>/manifest.json exists (a later push replaces the folder atomically per commit)."""
    import time

    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_TOKEN"])
    deadline = time.time() + seconds
    while True:
        try:
            if api.file_exists(
                REPO, f"{path_in_repo}/manifest.json", repo_type="dataset"
            ):
                return
        except Exception as error:  # noqa: BLE001 - transient Hub errors while polling
            print("poll error:", type(error).__name__, flush=True)
        if time.time() > deadline:
            sys.exit(f"timed out waiting for {path_in_repo}/manifest.json")
        time.sleep(120)


def fetch(dest: Path, path_in_repo: str, revision: str | None) -> None:
    from huggingface_hub import snapshot_download

    tmp = dest.with_name(dest.name + ".dl")
    snapshot_download(
        REPO,
        repo_type="dataset",
        revision=revision,
        local_dir=str(tmp),
        allow_patterns=[f"{path_in_repo}/*"],
        token=os.environ["HF_TOKEN"],
        max_workers=8,
    )
    got = tmp / path_in_repo
    bad = verify(got)
    if bad:
        sys.exit(f"downloaded files do not match manifest: {bad}")
    (got / "VERIFIED").write_text("all files match manifest.json sha256\n")
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    os.replace(got, dest)
    shutil.rmtree(tmp, ignore_errors=True)
    print("fetched and verified:", dest, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("push")
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--dest", type=Path, required=True)
    p.add_argument("--path-in-repo", required=True)
    p.add_argument("--no-upload", action="store_true")
    p.add_argument(
        "--name", help="mixture name recorded in the published manifest (e.g. M2-v5)"
    )
    f = sub.add_parser("fetch")
    f.add_argument("--dest", type=Path, required=True)
    f.add_argument("--path-in-repo", required=True)
    f.add_argument("--revision")
    f.add_argument(
        "--wait-seconds",
        type=int,
        default=0,
        help="poll the Hub until the folder's manifest exists",
    )
    v = sub.add_parser("verify")
    v.add_argument("--dir", type=Path, required=True)
    args = parser.parse_args()
    if args.cmd == "push":
        push(args.src, args.dest, args.path_in_repo, not args.no_upload, args.name)
    elif args.cmd == "fetch":
        if args.wait_seconds:
            wait_for(args.path_in_repo, args.wait_seconds)
        fetch(args.dest, args.path_in_repo, args.revision)
    else:
        bad = verify(args.dir)
        print("OK" if not bad else f"MISMATCH {bad}")
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
