"""Hugging Face steps of a Reasoning release (run on a node with the HF CLI's Python; token from its default file).

  ensure    create the repository PRIVATE if absent; refuse a repository that exists and is public
  upload    upload a verified package directory as one commit; print the commit
  verify    real download of that exact commit into a fresh directory, then re-hash every file against the
            package's MODEL_MANIFEST.json (and the manifest itself against the local package's)
  collect   add the repository to the Decision 2.0 collection (it stays private) and read the collection back

usage: python -m v2.reasoning.hub_rsn <step> --repo vllm-sr/NAME [--package DIR] [--revision SHA] [--dest DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

COLLECTION = "vllm-sr/decision-20-6ab7cf7bdfb506bf8269cb00"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("ensure", "upload", "verify", "collect"))
    parser.add_argument("--repo", required=True)
    parser.add_argument("--package", type=Path)
    parser.add_argument("--revision")
    parser.add_argument("--dest", type=Path)
    parser.add_argument("--message", default="Decision 2.0 Reasoning release")
    args = parser.parse_args()
    if not args.repo.startswith("vllm-sr/Decision-2.0-") or not args.repo.endswith(
        "-Reasoning"
    ):
        raise SystemExit("only vllm-sr/Decision-2.0-*-Reasoning repositories")
    api = HfApi()
    if args.step == "ensure":
        if api.repo_exists(args.repo):
            info = api.model_info(args.repo)
            if not info.private:
                raise SystemExit(f"{args.repo} exists and is public; refusing")
        else:
            api.create_repo(args.repo, repo_type="model", private=True)
        info = api.model_info(args.repo)
        print(json.dumps({"repo": args.repo, "private": info.private, "sha": info.sha}))
    elif args.step == "upload":
        if not api.model_info(args.repo).private:
            raise SystemExit("repository is not private")
        commit = api.upload_folder(
            repo_id=args.repo,
            folder_path=str(args.package),
            commit_message=args.message,
            ignore_patterns=["**/__pycache__/**", ".cache/**"],
        )
        print(
            json.dumps(
                {"repo": args.repo, "commit": commit.oid, "url": commit.commit_url}
            )
        )
    elif args.step == "verify":
        if args.dest.exists():
            raise FileExistsError(args.dest)
        root = Path(
            snapshot_download(
                repo_id=args.repo, revision=args.revision, local_dir=str(args.dest)
            )
        )
        manifest = json.loads((args.package / "MODEL_MANIFEST.json").read_text())
        if sha256(root / "MODEL_MANIFEST.json") != sha256(
            args.package / "MODEL_MANIFEST.json"
        ):
            raise SystemExit("downloaded MODEL_MANIFEST.json differs from the package")
        bad = [
            n
            for n, want in manifest["files_sha256"].items()
            if sha256(root / n) != want
        ]
        if bad:
            raise SystemExit(f"downloaded files differ: {bad[:5]}")
        info = api.model_info(args.repo, revision=args.revision)
        print(
            json.dumps(
                {
                    "repo": args.repo,
                    "revision": args.revision,
                    "private": info.private,
                    "files": len(manifest["files_sha256"]) + 1,
                    "manifest_sha256": sha256(root / "MODEL_MANIFEST.json"),
                }
            )
        )
    elif args.step == "collect":
        if not api.model_info(args.repo).private:
            raise SystemExit("repository is not private")
        collection = api.get_collection(COLLECTION)
        if not any(item.item_id == args.repo for item in collection.items):
            api.add_collection_item(
                COLLECTION, item_id=args.repo, item_type="model", exists_ok=True
            )
        collection = api.get_collection(COLLECTION)
        print(
            json.dumps(
                {
                    "collection": COLLECTION,
                    "items": [item.item_id for item in collection.items],
                    "private_repo": api.model_info(args.repo).private,
                }
            )
        )


if __name__ == "__main__":
    main()
