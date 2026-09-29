"""Free private HF storage held by decoder-track artifacts in the staging repo, without rewriting history.

The M3 tool called ``permanently_delete_lfs_files`` with the library default ``rewrite_history=True``: every
cleanup rewrote the repo's commits (the cited staging SHAs stopped existing), and after the M4 cleanup the
repo held no LFS object at all, including folders that were not requested. This version passes
``rewrite_history=False``, checks every requested object's SHA-256 against the node copy's list first, and
fails loudly if any object outside the requested folders is gone afterwards. Writes a JSON receipt.

usage: m4-free-lfs.py <receipt.json> <folder>=<node sha256 list> ...
"""

import json
import sys
from datetime import datetime, timezone

from huggingface_hub import HfApi

REPO = "llm-semantic-router/dev2-dec-staging"


def main() -> None:
    receipt_path, specs = sys.argv[1], sys.argv[2:]
    api = HfApi()
    before = list(api.list_lfs_files(REPO))
    chosen, receipt = [], {
        "repo": REPO,
        "utc": datetime.now(timezone.utc).isoformat(),
        "folders": {},
    }
    for spec in specs:
        folder, listing = spec.split("=", 1)
        local = {}
        for line in open(listing):
            sha, path = line.split(maxsplit=1)
            local[path.strip().removeprefix("./")] = sha
        objs = [f for f in before if f.filename.startswith(folder + "/")]
        if not objs:
            raise SystemExit(f"{folder}: no LFS objects")
        for f in objs:
            if local.get(f.filename) != f.file_oid:
                raise SystemExit(
                    f"{f.filename}: hub oid {f.file_oid} != node {local.get(f.filename)}"
                )
        chosen.extend(objs)
        receipt["folders"][folder] = {
            "objects": [
                {"file": f.filename, "sha256": f.file_oid, "size": f.size} for f in objs
            ],
            "bytes": sum(f.size for f in objs),
        }
    keep = {f.file_oid for f in before} - {f.file_oid for f in chosen}
    api.permanently_delete_lfs_files(REPO, chosen, rewrite_history=False)
    after = {f.file_oid: f for f in api.list_lfs_files(REPO)}
    receipt["rewrite_history"] = False
    receipt["deleted_bytes"] = sum(f.size for f in chosen)
    receipt["remaining_bytes"] = sum(f.size for f in after.values())
    receipt["lost_unrequested"] = sorted(keep - set(after))
    json.dump(receipt, open(receipt_path, "x"), indent=2)
    print(
        json.dumps(
            {
                k: receipt[k]
                for k in ("deleted_bytes", "remaining_bytes", "lost_unrequested")
            }
        )
    )
    if receipt["lost_unrequested"]:
        raise SystemExit(
            "objects outside the requested folders disappeared; restore them from the node copies"
        )


if __name__ == "__main__":
    main()
