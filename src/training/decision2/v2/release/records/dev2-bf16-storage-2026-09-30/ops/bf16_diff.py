"""Files that differ between the replaced revision and its BF16 storage revision (hf-cli python, node token).

  <hf-cli python> bf16_diff.py <repo> <replaced revision> <new revision> <bf16-copy receipt> <out.json>

A storage-only revision may change the backbone weight files, which must then be exactly the files of the
v2.release.bf16_copy receipt (LFS SHA-256 and size; the shard index is a small git file whose content the release
readback checks), and the card files that a card-only revision may change (card, assets, evaluation page, licence
texts, root config.json, MODEL_MANIFEST.json). Every other file (decision head, tokenizer, configs, runtime) must
keep its size and LFS SHA-256 or git blob id. Exits 1 otherwise.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from huggingface_hub import HfApi

CARD_FILES = {
    "README.md",
    "MODEL_MANIFEST.json",
    "config.json",
    "LICENSE",
    "NOTICE",
    "ATTRIBUTIONS.md",
    "LICENSING.md",
}
CARD_PREFIXES = ("assets/", "evaluation/", "LICENSES/")
HUB_MANAGED = {".gitattributes"}


def files(api: HfApi, repo: str, revision: str) -> dict:
    info = api.model_info(repo, revision=revision, files_metadata=True)
    if info.id != repo or info.sha != revision or info.private is not True:
        raise SystemExit(
            f"{repo}@{revision} resolves to {info.id}@{info.sha} (private={info.private})"
        )
    out = {}
    for s in info.siblings:
        lfs = getattr(s, "lfs", None)
        sha = (
            (
                getattr(lfs, "sha256", None)
                or (lfs.get("sha256") if isinstance(lfs, dict) else None)
            )
            if lfs
            else None
        )
        out[s.rfilename] = {
            "size": s.size,
            "lfs_sha256": sha,
            "blob_id": getattr(s, "blob_id", None),
        }
    return out


def main() -> int:
    repo, released, new, receipt_path, out = sys.argv[1:6]
    receipt = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
    converted = {
        name: entry
        for name, entry in receipt["files"].items()
        if name.startswith("backbone/model") and not entry["verbatim"]
    }
    api = HfApi()
    old, cur = files(api, repo, released), files(api, repo, new)
    changed = sorted(
        n
        for n in set(old) | set(cur)
        if old.get(n) != cur.get(n) and n not in HUB_MANAGED
    )
    problems = []
    for name in changed:
        if name in CARD_FILES or name.startswith(CARD_PREFIXES):
            continue
        entry = converted.get(name)
        if entry is None:
            problems.append(
                f"{name}: changed but neither a card file nor a converted backbone file"
            )
            continue
        now = cur.get(name) or {}
        if name.endswith(".safetensors") and (
            now.get("lfs_sha256") != entry["sha256"]
            or now.get("size") != entry["bytes"]
        ):
            problems.append(f"{name}: not the BF16 copy's bytes")
        elif not name.endswith(".safetensors") and now.get("size") != entry["bytes"]:
            problems.append(f"{name}: size differs from the BF16 copy's index")
        if (old.get(name) or {}).get("lfs_sha256") not in (
            None,
            entry["source_sha256"],
        ):
            problems.append(
                f"{name}: the replaced revision did not hold the copy's source bytes"
            )
    missing = sorted(n for n in converted if n not in cur)
    problems += [f"{n}: converted file missing from the new revision" for n in missing]
    result = {
        "schema": "dev2-release-bf16-revision-diff/1",
        "repo": repo,
        "replaced_revision": released,
        "new_revision": new,
        "bf16_receipt": receipt_path,
        "changed": changed,
        "converted_files": sorted(converted),
        "bytes_before": sum((old[n]["size"] or 0) for n in converted if n in old),
        "bytes_after": sum((cur[n]["size"] or 0) for n in converted if n in cur),
        "unchanged_weight_files": sorted(
            n
            for n in cur
            if n.endswith(".safetensors")
            and n not in converted
            and old.get(n) == cur.get(n)
        ),
        "hub_managed_changed": sorted(
            n for n in HUB_MANAGED if old.get(n) != cur.get(n)
        ),
        "problems": problems,
        "passed": not problems,
    }
    Path(out).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "changed",
                    "bytes_before",
                    "bytes_after",
                    "problems",
                    "passed",
                )
            }
        )
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
