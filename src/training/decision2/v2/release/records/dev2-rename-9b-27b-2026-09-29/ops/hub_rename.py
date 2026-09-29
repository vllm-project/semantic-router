"""DEV2.0-8B -> DEV2.0-9B and DEV2.0-26B -> DEV2.0-27B on the Hub (user directive 2026-09-29 16:05 UTC+8).

Runs on a node with the authenticated HF CLI environment; the token stays in that environment
and is never printed. Every subcommand writes one JSON receipt.

  <hf-cli python> - snapshot  --out F   repos (old and new IDs), their full history and the collection
  <hf-cli python> - move      --out F   preconditions, move_repo for each pair, then history / revision /
                                        privacy / file-hash comparison of the new ID against the pre-move state
  <hf-cli python> - redirects --out F   what the old IDs do after the move (API, resolve, web, hf_hub_download)
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path

import httpx
import huggingface_hub
from huggingface_hub import HfApi, constants, hf_hub_download
from huggingface_hub.utils import build_hf_headers

PAIRS = (
    (
        "llm-semantic-router/DEV2.0-8B",
        "llm-semantic-router/DEV2.0-9B",
        "53bac735be58def53673d0d290b9baa3f2af1cf9",
    ),
    (
        "llm-semantic-router/DEV2.0-26B",
        "llm-semantic-router/DEV2.0-27B",
        "6931828d7e41a5d31cc8e5acdc36f5eebae70fad",
    ),
)
COLLECTION = "llm-semantic-router/decision-20-6ab7cf7bdfb506bf8269cb00"


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def lfs_sha(sibling) -> str | None:
    lfs = getattr(sibling, "lfs", None)
    if lfs is None:
        return None
    return getattr(lfs, "sha256", None) or (
        lfs.get("sha256") if isinstance(lfs, dict) else None
    )


def repo_state(api: HfApi, repo: str) -> dict:
    try:
        info = api.model_info(repo, files_metadata=True)
    except Exception as exc:  # noqa: BLE001 - recorded, not raised
        return {
            "requested": repo,
            "exists": False,
            "error": f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}",
        }
    commits = api.list_repo_commits(repo)
    refs = api.list_repo_refs(repo)
    files = sorted(
        [s.rfilename, s.size, lfs_sha(s), getattr(s, "blob_id", None)]
        for s in info.siblings
    )
    return {
        "requested": repo,
        "exists": True,
        "id": info.id,
        "sha": info.sha,
        "private": info.private,
        "last_modified": str(info.last_modified),
        "created_at": str(getattr(info, "created_at", None)),
        "files": files,
        "files_digest": hashlib.sha256(json.dumps(files).encode()).hexdigest(),
        "commits": [[c.commit_id, c.title, str(c.created_at)] for c in commits],
        "refs": {
            "branches": sorted([b.name, b.target_commit] for b in refs.branches),
            "tags": sorted([t.name, t.target_commit] for t in refs.tags),
        },
    }


def collection_state(api: HfApi) -> dict:
    c = api.get_collection(COLLECTION)
    return {
        "slug": c.slug,
        "title": c.title,
        "private": c.private,
        "items": [
            [i.position, i.item_type, i.item_id, i.item_object_id] for i in c.items
        ],
    }


def cmd_snapshot(api: HfApi) -> dict:
    return {
        "repos": {repo: repo_state(api, repo) for pair in PAIRS for repo in pair[:2]},
        "collection": collection_state(api),
    }


def cmd_move(api: HfApi) -> dict:
    out: dict = {"pairs": [], "collection_before": collection_state(api)}
    for old, new, main in PAIRS:
        pre_old, pre_new = repo_state(api, old), repo_state(api, new)
        entry = {
            "from": old,
            "to": new,
            "expected_main": main,
            "pre_old": pre_old,
            "pre_new": pre_new,
        }
        out["pairs"].append(entry)
        problems = []
        if not pre_old.get("exists"):
            problems.append("source repository not found")
        else:
            if pre_old["id"] != old:
                problems.append(f"source resolves to {pre_old['id']}")
            if pre_old["private"] is not True:
                problems.append("source is not private")
            if pre_old["sha"] != main:
                problems.append(f"source main is {pre_old['sha']}")
        if pre_new.get("exists"):
            problems.append("target ID already exists")
        entry["precondition_problems"] = problems
        if problems:
            entry["moved"] = False
            continue
        entry["move_utc"] = now()
        api.move_repo(from_id=old, to_id=new, repo_type="model")
        post = None
        for _ in range(30):
            post = repo_state(api, new)
            if post.get("exists") and post.get("id") == new:
                break
            time.sleep(2)
        entry["moved"] = bool(post and post.get("exists"))
        entry["post_new"] = post
        if not entry["moved"]:
            continue
        revisions = {}
        for commit_id, _title, _created in post["commits"]:
            revisions[commit_id] = (
                api.model_info(new, revision=commit_id).sha == commit_id
            )
        entry["checks"] = {
            "id_is_new": post["id"] == new,
            "private": post["private"] is True,
            "main_unchanged": post["sha"] == main,
            "history_identical": post["commits"] == pre_old["commits"],
            "refs_identical": post["refs"] == pre_old["refs"],
            "files_identical": post["files"] == pre_old["files"],
            "every_commit_resolves": all(revisions.values()),
            "commits": len(post["commits"]),
        }
        entry["revisions_resolve"] = revisions
    out["collection_after"] = collection_state(api)
    return out


def probe(client: httpx.Client, method: str, url: str) -> dict:
    try:
        r = client.request(method, url)
    except Exception as exc:  # noqa: BLE001
        return {"method": method, "url": url, "error": type(exc).__name__}
    return {
        "method": method,
        "url": url,
        "status": r.status_code,
        "location": r.headers.get("location"),
    }


def cmd_redirects(api: HfApi) -> dict:
    endpoint = constants.ENDPOINT
    results = []
    with httpx.Client(
        headers=build_hf_headers(), follow_redirects=False, timeout=60
    ) as client:
        for old, new, main in PAIRS:
            r: dict = {"old": old, "new": new, "revision": main}
            try:
                info = api.model_info(old)
                r["model_info_old"] = {
                    "id": info.id,
                    "sha": info.sha,
                    "private": info.private,
                }
            except Exception as exc:  # noqa: BLE001
                r["model_info_old"] = {
                    "error": f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}"
                }
            try:
                r["repo_exists_old"] = api.repo_exists(old)
            except Exception as exc:  # noqa: BLE001
                r["repo_exists_old"] = f"error {type(exc).__name__}"
            with tempfile.TemporaryDirectory(prefix="dev2-redirect-") as cache:
                try:
                    path = hf_hub_download(
                        old, "config.json", revision=main, cache_dir=f"{cache}/old"
                    )
                    got = hashlib.sha256(Path(path).read_bytes()).hexdigest()
                    ref = hf_hub_download(
                        new, "config.json", revision=main, cache_dir=f"{cache}/new"
                    )
                    want = hashlib.sha256(Path(ref).read_bytes()).hexdigest()
                    r["hf_hub_download_old"] = {
                        "ok": True,
                        "config_sha256": got,
                        "same_as_new": got == want,
                    }
                except Exception as exc:  # noqa: BLE001
                    r["hf_hub_download_old"] = {
                        "ok": False,
                        "error": f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}",
                    }
            r["http"] = [
                probe(client, "GET", f"{endpoint}/api/models/{old}"),
                probe(client, "GET", f"{endpoint}/api/models/{old}/revision/{main}"),
                probe(client, "HEAD", f"{endpoint}/{old}/resolve/{main}/config.json"),
                probe(client, "GET", f"{endpoint}/{old}"),
                probe(client, "GET", f"{endpoint}/api/models/{new}"),
            ]
            results.append(r)
    return {"pairs": results, "collection": collection_state(api)}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("snapshot", "move", "redirects"))
    p.add_argument("--out", required=True)
    args = p.parse_args()
    api = HfApi()
    started = now()
    body = {"snapshot": cmd_snapshot, "move": cmd_move, "redirects": cmd_redirects}[
        args.command
    ](api)
    receipt = {
        "schema": "dev2-hub-rename/1",
        "command": args.command,
        "started_utc": started,
        "ended_utc": now(),
        "huggingface_hub": huggingface_hub.__version__,
        **body,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"out": args.out, "command": args.command}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
