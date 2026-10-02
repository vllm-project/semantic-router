"""Purge the superseded weight LFS objects of Decision-2.0-Vega-27B after its Index-first successor revision verified.

User rule 2026-10-02 09:55 UTC+8 (Index-first), as the earlier successors (records/dev2-4b-indexfirst-2026-10-02/ops):
after the new revision verifies, the superseded revision's weight LFS blobs are purged with rewrite_history=False;
node copies stay the durable store. Targets are the LFS objects of the replaced revision's *.safetensors files (A20r's
adapter and decision head, identity 2e074511, shared by every DEV2.0-27B / Decision-2.0-Vega-27B revision since the
A20r release) that no branch or tag head references any more; the tokenizer, banner and every other file are kept.

    <hf-cli python> purge_superseded.py plan|apply RECEIPT.json --repo R --old-revision SHA --new-revision SHA --node-copy DIR

Refuses unless: main is the verified new revision; no branch or tag tree references a target; every target is an
LFS object of the repo at its path in the old revision; the node copy re-hashes to every target. apply deletes with
rewrite_history=False, then checks commits and refs are unchanged, exactly the targets are gone, the old weight paths
are no longer served and every LFS file of main still is.
"""

import datetime as dt
import hashlib
import json
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_url
from huggingface_hub.utils import build_hf_headers, get_session

api = HfApi()
session = get_session()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def served(repo: str, path: str, revision: str) -> int:
    url = hf_hub_url(repo, path, revision=revision)
    headers = {**build_hf_headers(), "Range": "bytes=0-1023"}
    try:
        r = session.get(url, headers=headers, follow_redirects=True)
    except TypeError:
        r = session.get(url, headers=headers, allow_redirects=True)
    return r.status_code


def state(repo: str) -> dict:
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
    return {
        "heads": heads,
        "commits": [c.commit_id for c in api.list_repo_commits(repo)],
        "lfs": {f.file_oid: f for f in api.list_lfs_files(repo)},
        "paths": paths,
    }


def main() -> None:
    mode, receipt = sys.argv[1], Path(sys.argv[2])
    args = dict(zip(sys.argv[3::2], sys.argv[4::2]))
    repo, old_revision, new_revision = (
        args["--repo"],
        args["--old-revision"],
        args["--new-revision"],
    )
    node_copy = Path(args["--node-copy"])
    assert mode in ("plan", "apply") and not receipt.exists()
    assert repo == "llm-semantic-router/Decision-2.0-Vega-27B"
    before = state(repo)
    assert (
        before["heads"].get("branch:main") == new_revision
    ), "main is not the verified new revision"
    old_tree = {
        e.path: e.lfs.sha256
        for e in api.list_repo_tree(repo, revision=old_revision, recursive=True)
        if getattr(e, "lfs", None)
    }
    targets = {
        oid: path for path, oid in old_tree.items() if path.endswith(".safetensors")
    }
    assert targets and any(
        p == "decision_head.safetensors" for p in targets.values()
    ), "the old revision lacks its weight files"
    for oid in targets:
        assert oid not in before["paths"], f"a branch or tag still references {oid}"
        assert oid in before["lfs"], f"{oid} is not an LFS object of {repo}"
    node = {oid: sha256_file(node_copy / path) for oid, path in targets.items()}
    assert all(
        node[oid] == oid for oid in targets
    ), "node copy does not re-hash to the targets"
    main_lfs = [
        p.split(":", 2)[2]
        for ps in before["paths"].values()
        for p in ps
        if p.startswith("branch:main:")
    ]
    record = {
        "schema": "dev2-release-purge-superseded/1",
        "repo": repo,
        "mode": mode,
        "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "rewrite_history": False,
        "old_revision": old_revision,
        "new_revision": new_revision,
        "targets": {
            oid: {
                "path": path,
                "bytes": before["lfs"][oid].size,
                "node_copy_sha256": node[oid],
            }
            for oid, path in targets.items()
        },
        "target_bytes": sum(before["lfs"][oid].size for oid in targets),
        "node_copy": str(node_copy),
        "before": {
            "heads": before["heads"],
            "commits": len(before["commits"]),
            "lfs_objects": len(before["lfs"]),
            "old_paths_served": {
                p: served(repo, p, old_revision) for p in targets.values()
            },
            "main_lfs_served": {p: served(repo, p, new_revision) for p in main_lfs},
        },
    }
    if mode == "apply":
        api.permanently_delete_lfs_files(
            repo, [before["lfs"][oid] for oid in targets], rewrite_history=False
        )
        time.sleep(20)
        after = state(repo)
        record["after"] = {
            "heads": after["heads"],
            "commits": len(after["commits"]),
            "lfs_objects": len(after["lfs"]),
            "old_paths_served": {
                p: served(repo, p, old_revision) for p in targets.values()
            },
            "main_lfs_served": {p: served(repo, p, new_revision) for p in main_lfs},
        }
        record["checks"] = {
            "commits_unchanged": after["commits"] == before["commits"],
            "refs_unchanged": after["heads"] == before["heads"],
            "exactly_targets_gone": set(after["lfs"])
            == set(before["lfs"]) - set(targets),
            "old_weights_not_served": all(
                s not in (200, 206)
                for s in record["after"]["old_paths_served"].values()
            ),
            "main_still_served": all(
                s in (200, 206) for s in record["after"]["main_lfs_served"].values()
            ),
        }
        record["passed"] = all(record["checks"].values())
    receipt.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {k: record.get(k) for k in ("mode", "target_bytes", "checks", "passed")}
        )
    )
    if mode == "apply" and not record["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
