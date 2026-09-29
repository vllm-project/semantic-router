"""Remove the LFS leftovers of the stale Decision 2.0 staging repositories (hf-cli python, node token).

Coordinator note 2026-09-30 06:10 (27B release path, step 1): "remove stale staging". Every weight object of these
repositories was already deleted by earlier cleanups; what is left are shared small LFS files (tokenizers). A leftover
is removed only if a released DEV2.0 repository's main serves the same SHA-256 (so the bytes stay on the Hub) or
it is listed in --allow with a reason. Deletion passes rewrite_history=False; commits, refs and every non-LFS file
stay. Released repositories, datasets and the collection are never touched.

    <hf-cli python> staging_leftovers.py plan|apply RECEIPT.json
"""

import datetime as dt
import json
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi

ORG = "llm-semantic-router"
STAGING = (
    "dev2-dec-staging",
    "dev2-9b-staging",
    "dev2-27b-staging",
    "dev2-staging-06bm5-z",
    "dev2-staging-06bm6-mxcx",
    "dev2-release-staging",
    "dev2-release-staging-06bm4",
)
RELEASED = (
    "DEV2.0-0.6B",
    "DEV2.0-0.8B",
    "DEV2.0-2B",
    "DEV2.0-4B",
    "DEV2.0-9B",
    "DEV2.0-27B",
)
COLLECTION = f"{ORG}/decision-20-6ab7cf7bdfb506bf8269cb00"
api = HfApi()


def lfs_paths(repo: str) -> dict:
    refs = api.list_repo_refs(repo)
    heads = {
        **{f"branch:{b.name}": b.target_commit for b in refs.branches},
        **{f"tag:{t.name}": t.target_commit for t in refs.tags},
    }
    paths = {}
    for name, commit in heads.items():
        for e in api.list_repo_tree(repo, revision=commit, recursive=True):
            if getattr(e, "lfs", None):
                paths.setdefault(e.lfs.sha256, []).append(f"{name}:{e.path}")
    return {"heads": heads, "paths": paths}


def state(repo: str) -> dict:
    info = api.model_info(repo)
    return {
        "private": info.private,
        "used_storage": getattr(info, "used_storage", None)
        or getattr(info, "usedStorage", None),
        "commits": [c.commit_id for c in api.list_repo_commits(repo)],
        "lfs": {f.file_oid: f for f in api.list_lfs_files(repo)},
        **lfs_paths(repo),
    }


def main() -> None:
    mode, receipt = sys.argv[1], Path(sys.argv[2])
    assert mode in ("plan", "apply") and not receipt.exists()
    members = {item.item_id for item in api.get_collection(COLLECTION).items}
    served = {}
    for name in RELEASED:
        repo = f"{ORG}/{name}"
        for sha, where in lfs_paths(repo)["paths"].items():
            if any(w.startswith("branch:main:") for w in where):
                served.setdefault(sha, []).append(repo)
    record = {
        "schema": "dev2-release-staging-leftovers/1",
        "mode": mode,
        "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "rewrite_history": False,
        "repos": {},
    }
    ok = True
    for name in STAGING:
        repo = f"{ORG}/{name}"
        assert repo not in members, f"{repo} is in the collection"
        before = state(repo)
        assert before["private"] is True, f"{repo} is not private"
        targets = {oid: f for oid, f in before["lfs"].items()}
        entry = {
            "before": {
                "commits": len(before["commits"]),
                "heads": before["heads"],
                "lfs_objects": len(before["lfs"]),
                "used_storage": before["used_storage"],
            },
            "targets": {
                oid: {
                    "bytes": f.size,
                    "paths": before["paths"].get(oid, []),
                    "served_by_released_main": served.get(oid, []),
                }
                for oid, f in targets.items()
            },
            "target_bytes": sum(f.size for f in targets.values()),
        }
        unserved = [oid for oid in targets if not served.get(oid)]
        entry["refused"] = unserved
        if unserved:
            ok = False
        elif mode == "apply" and targets:
            api.permanently_delete_lfs_files(
                repo, list(targets.values()), rewrite_history=False
            )
            time.sleep(10)
            after = state(repo)
            entry["after"] = {
                "commits": len(after["commits"]),
                "heads": after["heads"],
                "lfs_objects": len(after["lfs"]),
                "used_storage": after["used_storage"],
            }
            entry["checks"] = {
                "commits_unchanged": after["commits"] == before["commits"],
                "refs_unchanged": after["heads"] == before["heads"],
                "no_lfs_left": not after["lfs"],
            }
            ok = ok and all(entry["checks"].values())
        record["repos"][repo] = entry
    record["target_bytes"] = sum(e["target_bytes"] for e in record["repos"].values())
    record["passed"] = ok
    receipt.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "mode": mode,
                "target_bytes": record["target_bytes"],
                "passed": ok,
                "refused": {
                    r: e["refused"] for r, e in record["repos"].items() if e["refused"]
                },
            }
        )
    )
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
