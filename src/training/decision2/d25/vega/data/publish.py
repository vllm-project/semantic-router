"""Publish a built mixture to the shared node path and the private HF dataset, or fetch and verify it.

    python -m d25.vega.data.publish push  --src DIR --dest /data/d25/shared/data/v1/M1 --path-in-repo v1/M1
    python -m d25.vega.data.publish fetch --dest /data/d25/shared/data/v1/M1 --path-in-repo v1/M1 \
        [--wait-seconds N] [--fallback-url http://d25-vega-data-files:8080]
    python -m d25.vega.data.publish retry-uploads [--root /data/d25/shared/data]

Every file is checked against the sha256 recorded in the mixture manifest; a VERIFIED marker is written
only when all files match. The Hub is never a single point of failure:

- the token is read from the mounted secret file (``HF_TOKEN_FILE``, default /var/run/secrets/hf/HF_TOKEN)
  before every attempt, so a rotated secret is picked up without restarting the pod (env ``HF_TOKEN`` is
  the fallback);
- ``push`` writes and verifies the shared node copy first, then uploads with retries and backoff; when the
  upload still fails after ``--upload-hours`` it leaves ``HF_UPLOAD_PENDING`` in the shared copy and exits 0
  (``retry-uploads`` finishes such uploads later);
- ``fetch`` takes the folder from the Hub or, when the Hub has not got it (or rejects the token), from the
  in-cluster file server of the source node (``--fallback-url``; the server exposes /data/d25/shared/data and
  only VERIFIED folders are taken).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from d25.vega.data.util import sha256_file

REPO = "vllm-sr/decision-2.5-vega-training-data"
PENDING = "HF_UPLOAD_PENDING"
SHARED_ROOT = Path("/data/d25/shared/data")


def hf_token() -> str | None:
    path = Path(os.environ.get("HF_TOKEN_FILE", "/var/run/secrets/hf/HF_TOKEN"))
    if path.exists():
        value = path.read_text().strip()
        if value:
            return value
    return os.environ.get("HF_TOKEN")


def api():
    from huggingface_hub import HfApi

    return HfApi(token=hf_token())


def verify(directory: Path) -> list[str]:
    manifest = json.loads((directory / "manifest.json").read_text())
    bad = []
    for name, info in manifest["files"].items():
        path = directory / name
        if not path.exists() or sha256_file(path) != info["sha256"]:
            bad.append(name)
    return bad


def upload(dest: Path, path_in_repo: str, hours: float) -> bool:
    deadline = time.time() + hours * 3600
    delay = 60.0
    while True:
        try:
            hub = api()
            hub.create_repo(REPO, repo_type="dataset", private=True, exist_ok=True)
            info = hub.upload_folder(
                repo_id=REPO,
                repo_type="dataset",
                folder_path=str(dest),
                path_in_repo=path_in_repo,
                allow_patterns=["*.jsonl.gz", "manifest.json", "*.md"],
                commit_message=f"{path_in_repo}: mixture upload",
            )
            (dest / PENDING).unlink(missing_ok=True)
            print("uploaded:", info, flush=True)
            return True
        except (
            Exception
        ) as error:  # noqa: BLE001 - auth (rotated token), network or Hub errors: retry
            print(
                f"upload of {path_in_repo} failed ({type(error).__name__}); retrying in {int(delay)} s",
                flush=True,
            )
            if time.time() + delay > deadline:
                (dest / PENDING).write_text(f"{path_in_repo}\n")
                print(
                    f"giving up for now: {dest / PENDING} written; run `publish retry-uploads` later",
                    flush=True,
                )
                return False
            time.sleep(delay)
            delay = min(delay * 2, 1800.0)


def push(
    src: Path,
    dest: Path,
    path_in_repo: str,
    do_upload: bool,
    name: str | None = None,
    upload_hours: float = 6.0,
) -> None:
    bad = verify(src)
    if bad:
        sys.exit(f"source does not match its manifest: {bad}")
    tmp = dest.with_name(dest.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.copytree(
        src, tmp, ignore=shutil.ignore_patterns("VERIFIED", "SUPERSEDED", PENDING)
    )
    if name:
        manifest = json.loads((tmp / "manifest.json").read_text())
        manifest["name"] = name
        (tmp / "manifest.json").write_text(
            json.dumps(manifest, indent=1, ensure_ascii=False) + "\n"
        )
    if verify(tmp):
        sys.exit("copy verification failed")
    if do_upload:
        (tmp / PENDING).write_text(f"{path_in_repo}\n")
    (tmp / "VERIFIED").write_text("all files match manifest.json sha256\n")
    if dest.exists():
        shutil.rmtree(dest)
    os.replace(tmp, dest)
    print("shared copy verified:", dest, flush=True)
    if do_upload:
        upload(dest, path_in_repo, upload_hours)


def retry_uploads(root: Path, hours: float) -> int:
    pending = sorted(root.glob(f"*/*/{PENDING}"))
    failed = 0
    for marker in pending:
        path_in_repo = marker.read_text().strip()
        if not upload(marker.parent, path_in_repo, hours):
            failed += 1
    print(json.dumps({"pending": len(pending), "failed": failed}), flush=True)
    return failed


def hub_has(path_in_repo: str) -> bool:
    try:
        return api().file_exists(
            REPO, f"{path_in_repo}/manifest.json", repo_type="dataset"
        )
    except (
        Exception
    ) as error:  # noqa: BLE001 - auth or network: treat as not available yet
        print("hub poll:", type(error).__name__, flush=True)
        return False


def http_has(base: str | None, path_in_repo: str) -> bool:
    if not base:
        return False
    try:
        with urllib.request.urlopen(
            f"{base.rstrip('/')}/{path_in_repo}/VERIFIED", timeout=30
        ) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError):
        return False


def fetch_hub(dest: Path, path_in_repo: str, revision: str | None) -> Path:
    from huggingface_hub import snapshot_download

    tmp = dest.with_name(dest.name + ".dl")
    snapshot_download(
        REPO,
        repo_type="dataset",
        revision=revision,
        local_dir=str(tmp),
        allow_patterns=[f"{path_in_repo}/*"],
        token=hf_token(),
        max_workers=8,
    )
    return tmp / path_in_repo


def fetch_http(dest: Path, path_in_repo: str, base: str) -> Path:
    tmp = dest.with_name(dest.name + ".http")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    url = f"{base.rstrip('/')}/{path_in_repo}"
    with urllib.request.urlopen(f"{url}/manifest.json", timeout=60) as response:
        (tmp / "manifest.json").write_bytes(response.read())
    manifest = json.loads((tmp / "manifest.json").read_text())
    for name in manifest["files"]:
        with urllib.request.urlopen(f"{url}/{name}", timeout=600) as response, open(
            tmp / name, "wb"
        ) as out:
            shutil.copyfileobj(response, out, 1 << 22)
    return tmp


def fetch(
    dest: Path,
    path_in_repo: str,
    revision: str | None,
    wait_seconds: int = 0,
    fallback_url: str | None = None,
) -> None:
    deadline = time.time() + wait_seconds
    while True:
        source = (
            "hub"
            if hub_has(path_in_repo)
            else ("http" if http_has(fallback_url, path_in_repo) else None)
        )
        if source is None and not wait_seconds and not fallback_url:
            source = "hub"  # plain fetch: let snapshot_download raise a clear error
        if source:
            try:
                got = (
                    fetch_hub(dest, path_in_repo, revision)
                    if source == "hub"
                    else fetch_http(dest, path_in_repo, fallback_url)
                )
                bad = verify(got)
                if not bad:
                    break
                print(f"{source} copy does not match its manifest: {bad}", flush=True)
            except (
                Exception
            ) as error:  # noqa: BLE001 - try again / the other source on the next round
                print(f"fetch from {source} failed: {type(error).__name__}", flush=True)
        if time.time() > deadline:
            sys.exit(f"timed out waiting for {path_in_repo}")
        time.sleep(120)
    (got / "VERIFIED").write_text("all files match manifest.json sha256\n")
    (got / PENDING).unlink(missing_ok=True)
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    os.replace(got, dest)
    for leftover in (
        dest.with_name(dest.name + ".dl"),
        dest.with_name(dest.name + ".http"),
    ):
        shutil.rmtree(leftover, ignore_errors=True)
    print(f"fetched from {source} and verified: {dest}", flush=True)


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
    p.add_argument(
        "--upload-hours",
        type=float,
        default=6.0,
        help="keep retrying a failed upload this long",
    )
    f = sub.add_parser("fetch")
    f.add_argument("--dest", type=Path, required=True)
    f.add_argument("--path-in-repo", required=True)
    f.add_argument("--revision")
    f.add_argument(
        "--wait-seconds",
        type=int,
        default=0,
        help="poll the Hub / fallback until the folder exists",
    )
    f.add_argument(
        "--fallback-url",
        help="in-cluster file server exposing /data/d25/shared/data of the source node",
    )
    r = sub.add_parser("retry-uploads")
    r.add_argument("--root", type=Path, default=SHARED_ROOT)
    r.add_argument("--upload-hours", type=float, default=6.0)
    v = sub.add_parser("verify")
    v.add_argument("--dir", type=Path, required=True)
    args = parser.parse_args()
    if args.cmd == "push":
        push(
            args.src,
            args.dest,
            args.path_in_repo,
            not args.no_upload,
            args.name,
            args.upload_hours,
        )
    elif args.cmd == "fetch":
        fetch(
            args.dest,
            args.path_in_repo,
            args.revision,
            args.wait_seconds,
            args.fallback_url,
        )
    elif args.cmd == "retry-uploads":
        sys.exit(1 if retry_uploads(args.root, args.upload_hours) else 0)
    else:
        bad = verify(args.dir)
        print("OK" if not bad else f"MISMATCH {bad}")
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
