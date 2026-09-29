"""Files that differ between two revisions of one private repository (hf-cli python, node token).

  <hf-cli python> revision_diff.py <repo> <released revision> <new revision> <out.json>

A card-only revision may change only the card, its assets and evaluation page, the licence texts,
the root config.json pointer and MODEL_MANIFEST.json; every other file, and every weight file in
particular, must keep its size and LFS SHA-256 or git blob id. Exits 1 otherwise.
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
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth")


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
    repo, released, new, out = sys.argv[1:5]
    api = HfApi()
    a, b = files(api, repo, released), files(api, repo, new)
    changed = sorted(n for n in set(a) | set(b) if a.get(n) != b.get(n))
    card = [n for n in changed if n in CARD_FILES or n.startswith(CARD_PREFIXES)]
    other = [n for n in changed if n not in card]
    weights = sorted(n for n in a if n.endswith(WEIGHT_SUFFIXES))
    weights_identical = weights == sorted(
        n for n in b if n.endswith(WEIGHT_SUFFIXES)
    ) and all(a[n] == b[n] for n in weights)
    receipt = {
        "schema": "dev2-revision-diff/1",
        "repo": repo,
        "released": released,
        "new": new,
        "files": {"released": len(a), "new": len(b)},
        "changed_card_files": card,
        "changed_other_files": other,
        "weight_files": len(weights),
        "weight_bytes": sum(a[n]["size"] or 0 for n in weights),
        "weights_byte_identical": weights_identical,
        "ok": weights_identical and not other,
    }
    Path(out).write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                k: receipt[k]
                for k in (
                    "repo",
                    "changed_card_files",
                    "changed_other_files",
                    "weight_files",
                    "weights_byte_identical",
                    "ok",
                )
            }
        )
    )
    return 0 if receipt["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
