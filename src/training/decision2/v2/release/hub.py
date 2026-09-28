"""Private Hugging Face operations for Decision 2.0 packages (run on a node).

Uses the node's authenticated HF CLI environment; the token stays in its
default file and never appears in argv, logs or receipts. Every command refuses
public repositories and collections. Staging repositories can never be added
to the Decision 2.0 collection; release repositories only with a gate receipt.

  ensure   create the repository PRIVATE if absent; refuse if it is public
  upload   upload a verified package directory; record the commit
  download real ``hf download`` of one exact revision into a fresh directory
  tree     re-hash a downloaded tree against MODEL_MANIFEST.json and the package
  readback private flag, revision, remote file hashes, Hub-parsed card, collection
  collect  add a gated release repository to the private collection, then read back
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from v2.release import layout

COLLECTION = "llm-semantic-router/decision-20-6ab7cf7bdfb506bf8269cb00"
COLLECTION_TITLE = "Decision 2.0"
HF_CLI = os.environ.get("DEV2_HF_CLI", "hf")


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def git_blob_sha1(path: Path) -> str:
    data = Path(path).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _api():
    from huggingface_hub import HfApi

    return HfApi()


def _guard(repo: str, kind: str, model_name: str) -> None:
    layout.check_repo(repo, model_name, staging=kind == "staging")


def ensure(args: argparse.Namespace) -> dict[str, Any]:
    _guard(args.repo, args.kind, args.model_name)
    api = _api()
    created = False
    try:
        info = api.model_info(args.repo)
    except Exception as exc:  # RepositoryNotFoundError without importing the class
        if type(exc).__name__ not in ("RepositoryNotFoundError", "HfHubHTTPError"):
            raise
        api.create_repo(args.repo, repo_type="model", private=True, exist_ok=False)
        created = True
        info = api.model_info(args.repo)
    if info.private is not True:
        raise RuntimeError(f"{args.repo} is not private; refusing to use it")
    return {
        "schema": "dev2-hub-ensure/1",
        "utc": now(),
        "repo": args.repo,
        "created": created,
        "private": info.private,
        "sha": info.sha,
    }


def upload(args: argparse.Namespace) -> dict[str, Any]:
    manifest = json.loads(
        (args.package / layout.MANIFEST_NAME).read_text(encoding="utf-8")
    )
    _guard(args.repo, manifest["kind"], manifest["model_name"])
    if manifest["repo_id"] != args.repo:
        raise ValueError("Package was built for a different repository")
    local = layout.inventory(args.package)
    expected = {
        **manifest["files_sha256"],
        layout.MANIFEST_NAME: layout.sha_file(args.package / layout.MANIFEST_NAME),
    }
    if local != expected:
        raise ValueError(
            "Package directory differs from its manifest; rebuild before upload"
        )
    api = _api()
    if api.model_info(args.repo).private is not True:
        raise RuntimeError("Target repository is not private")
    started = time.perf_counter()
    commit = api.upload_folder(
        repo_id=args.repo,
        repo_type="model",
        folder_path=str(args.package),
        commit_message=args.message,
        ignore_patterns=["**/__pycache__/**", ".cache/**", "*.pyc"],
        # The commit must leave exactly the package: files of an earlier revision that the
        # package no longer has (e.g. a dropped calibration.json) are deleted; the Hub keeps
        # .gitattributes.
        delete_patterns=["*"],
    )
    revision = commit.oid
    info = api.model_info(args.repo, revision=revision)
    if info.private is not True:
        raise RuntimeError("Repository became non-private during upload")
    return {
        "schema": "dev2-hub-upload/1",
        "utc": now(),
        "repo": args.repo,
        "revision": revision,
        "commit_url": getattr(commit, "commit_url", None),
        "manifest_sha256": expected[layout.MANIFEST_NAME],
        "files": len(expected),
        "bytes": sum((args.package / n).stat().st_size for n in expected),
        "seconds": time.perf_counter() - started,
        "private": info.private,
    }


def download(args: argparse.Namespace) -> dict[str, Any]:
    if args.dest.exists():
        raise FileExistsError(args.dest)
    if not layout.REVISION.fullmatch(args.revision):
        raise ValueError("Download an exact 40-hex revision")
    with tempfile.TemporaryDirectory(prefix="dev2-hf-cache-") as cache:
        env = {
            **os.environ,
            "HF_HUB_CACHE": cache,
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        }
        started = time.perf_counter()
        completed = subprocess.run(
            [
                HF_CLI,
                "download",
                args.repo,
                "--revision",
                args.revision,
                "--local-dir",
                str(args.dest),
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        seconds = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"hf download failed ({completed.returncode}): {completed.stderr[-500:]}"
        )
    files = layout.inventory(args.dest, allow_hub_added=True)
    return {
        "schema": "dev2-hub-download/1",
        "utc": now(),
        "repo": args.repo,
        "revision": args.revision,
        "command": "hf download <repo> --revision <sha> --local-dir <fresh dir> (fresh HF_HUB_CACHE)",
        "seconds": seconds,
        "files": len(files),
        "bytes": sum((args.dest / n).stat().st_size for n in files),
        "hub_added": sorted(n for n in layout.HUB_ADDED if (args.dest / n).is_file()),
        "passed": bool(files),
    }


def tree(args: argparse.Namespace) -> dict[str, Any]:
    downloaded = layout.inventory(args.download, allow_hub_added=True)
    manifest = json.loads(
        (args.download / layout.MANIFEST_NAME).read_text(encoding="utf-8")
    )
    expected = {
        **manifest["files_sha256"],
        layout.MANIFEST_NAME: layout.sha_file(args.download / layout.MANIFEST_NAME),
    }
    package = layout.inventory(args.package)
    missing = sorted(set(expected) - set(downloaded))
    extra = sorted(set(downloaded) - set(expected))
    changed = sorted(
        n for n in set(expected) & set(downloaded) if expected[n] != downloaded[n]
    )
    differs_from_package = sorted(
        n for n in set(package) | set(downloaded) if package.get(n) != downloaded.get(n)
    )
    return {
        "schema": "dev2-hub-tree/1",
        "utc": now(),
        "files": len(downloaded),
        "manifest_sha256": expected[layout.MANIFEST_NAME],
        "missing": missing,
        "extra": extra,
        "hash_mismatch": changed,
        "differs_from_pre_upload_package": differs_from_package,
        "passed": not (missing or extra or changed or differs_from_package),
    }


def _collection_items(api, slug: str) -> tuple[Any, list[str]]:
    collection = api.get_collection(slug)
    return collection, [
        item.item_id for item in collection.items if item.item_type == "model"
    ]


def readback(args: argparse.Namespace) -> dict[str, Any]:
    from v2.release.card import check_rendered

    api = _api()
    info = api.model_info(args.repo, revision=args.revision, files_metadata=True)
    manifest = json.loads(
        (args.package / layout.MANIFEST_NAME).read_text(encoding="utf-8")
    )
    expected = {
        **manifest["files_sha256"],
        layout.MANIFEST_NAME: layout.sha_file(args.package / layout.MANIFEST_NAME),
    }
    remote = {}
    mismatched = []
    for sibling in info.siblings:
        name = sibling.rfilename
        if name in layout.HUB_ADDED:
            continue
        lfs = getattr(sibling, "lfs", None)
        if lfs:
            digest = (
                lfs.get("sha256")
                if isinstance(lfs, dict)
                else getattr(lfs, "sha256", None)
            )
            remote[name] = {"lfs_sha256": digest}
            ok = digest == expected.get(name)
        else:
            remote[name] = {"blob_id": sibling.blob_id}
            ok = name in expected and sibling.blob_id == git_blob_sha1(
                args.package / name
            )
        if not ok:
            mismatched.append(name)
    card_data = info.card_data.to_dict() if getattr(info, "card_data", None) else {}
    readme = (args.package / "README.md").read_text(encoding="utf-8")
    front = readme.split("\n---\n", 1)[0]
    problems = check_rendered(readme, set(expected))
    for key in ("license", "base_model"):
        value = card_data.get(key)
        if isinstance(value, list) and len(value) == 1:
            value = value[0]
        if value is None or f"{key}: {value}" not in front:
            problems.append(
                f"Hub-parsed card {key} differs from the README front matter"
            )
    collection, items = _collection_items(api, args.collection)
    in_collection = args.repo in items
    kind = manifest["kind"]
    # Collection items name a repository, not a revision: a new revision of a release that an
    # earlier final decision already collected is in the collection before its own gate seal.
    expect_item = kind == "release" and (
        args.expect_collected or args.already_collected
    )
    collection_ok = collection.private is True and (
        in_collection if expect_item else not in_collection
    )
    return {
        "schema": "dev2-hub-readback/1",
        "utc": now(),
        "repo": args.repo,
        "revision": info.sha,
        "private": info.private,
        "remote_files": len(remote),
        "missing_remote": sorted(set(expected) - set(remote)),
        "extra_remote": sorted(set(remote) - set(expected)),
        "hash_mismatch": sorted(mismatched),
        "card_data": {
            k: card_data.get(k)
            for k in (
                "license",
                "license_name",
                "base_model",
                "base_model_relation",
                "tags",
            )
        },
        "card_problems": problems,
        "collection": {
            "slug": args.collection,
            "title": collection.title,
            "private": collection.private,
            "items": len(items),
            "contains_repo": in_collection,
            "expected_repo": expect_item,
        },
        "passed": info.private is True
        and info.sha == args.revision
        and not mismatched
        and set(expected) == set(remote)
        and not problems
        and collection_ok,
    }


def collect(args: argparse.Namespace) -> dict[str, Any]:
    manifest = json.loads(
        (args.package / layout.MANIFEST_NAME).read_text(encoding="utf-8")
    )
    if manifest["kind"] != "release" or not layout.RELEASE_REPO.fullmatch(args.repo):
        raise ValueError(
            "Only gated release repositories enter the Decision 2.0 collection"
        )
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    if (
        gate.get("schema") != "dev2-release-gate/1"
        or gate.get("decision") != "release"
        or gate.get("repo_id") != args.repo
        or gate.get("revision") != args.revision
        or gate.get("manifest_sha256")
        != layout.sha_file(args.package / layout.MANIFEST_NAME)
    ):
        raise ValueError("Gate receipt does not approve this exact package revision")
    api = _api()
    info = api.model_info(args.repo, revision=args.revision)
    collection, _ = _collection_items(api, args.collection)
    if (
        info.private is not True
        or collection.private is not True
        or collection.title != COLLECTION_TITLE
    ):
        raise RuntimeError("Repository and collection must both be private")
    api.add_collection_item(
        args.collection, item_id=args.repo, item_type="model", exists_ok=True
    )
    collection, items = _collection_items(api, args.collection)
    return {
        "schema": "dev2-hub-collect/1",
        "utc": now(),
        "repo": args.repo,
        "revision": args.revision,
        "gate_sha256": layout.sha_file(args.gate),
        "collection": {
            "slug": args.collection,
            "private": collection.private,
            "items": items,
        },
        "passed": collection.private is True and args.repo in items,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("ensure")
    p.add_argument("--repo", required=True)
    p.add_argument("--kind", choices=("staging", "release"), required=True)
    p.add_argument("--model-name", required=True)
    p = sub.add_parser("upload")
    p.add_argument("--repo", required=True)
    p.add_argument("--package", type=Path, required=True)
    p.add_argument("--message", required=True)
    p = sub.add_parser("download")
    p.add_argument("--repo", required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--dest", type=Path, required=True)
    p = sub.add_parser("tree")
    p.add_argument("--package", type=Path, required=True)
    p.add_argument("--download", type=Path, required=True)
    p = sub.add_parser("readback")
    p.add_argument("--repo", required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--package", type=Path, required=True)
    p.add_argument("--collection", default=COLLECTION)
    p.add_argument("--expect-collected", action="store_true")
    p.add_argument("--already-collected", action="store_true")
    p = sub.add_parser("collect")
    p.add_argument("--repo", required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--package", type=Path, required=True)
    p.add_argument("--gate", type=Path, required=True)
    p.add_argument("--collection", default=COLLECTION)
    for name in sub.choices:
        sub.choices[name].add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    handler = {
        "ensure": ensure,
        "upload": upload,
        "download": download,
        "tree": tree,
        "readback": readback,
        "collect": collect,
    }[args.command]
    result = handler(args)
    layout.write_json(args.output, result)
    passed = result.get("passed", True)
    print(
        json.dumps(
            {
                "command": args.command,
                "passed": passed,
                **{k: result[k] for k in ("repo", "revision") if k in result},
            }
        )
    )
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
