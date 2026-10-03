"""Files that differ between a released revision and its runtime C successor (hf-cli python, node token).

  <hf-cli python> rc_diff.py <repo> <released revision> <new revision> <out.json>

Phase A's ra_diff.py with runtime C's rules: the successor may change the package runtime (decision2/__init__.py,
api.py, the profile module, fast.py, fast_kernels.py, shared_ctx.py; never decision2/_vendor/), the root
Transformers remote code (modeling_decision2.py, pipeline_decision2.py, configuration_decision2.py), the card files
that revision_diff.py allows and MODEL_MANIFEST.json; it must change fast.py, fast_kernels.py, shared_ctx.py and
modeling_decision2.py; every other file, and every weight file in particular, must keep its size and LFS SHA-256 or
git blob id, and nothing is added or removed. Exits 1 otherwise.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from huggingface_hub import HfApi

from v2.release import hub

CHANGEABLE = {
    "decision2/__init__.py",
    "decision2/api.py",
    "decision2/qwen.py",
    "decision2/kai_native.py",
    "decision2/fast.py",
    "decision2/fast_kernels.py",
    "decision2/shared_ctx.py",
    "modeling_decision2.py",
    "pipeline_decision2.py",
    "configuration_decision2.py",
}
REQUIRED = {
    "decision2/fast.py",
    "decision2/fast_kernels.py",
    "decision2/shared_ctx.py",
    "modeling_decision2.py",
}


def card_rules():
    path = (
        Path(__file__).resolve().parents[2]
        / "dev2-rename-9b-27b-2026-09-29/ops/revision_diff.py"
    )
    spec = importlib.util.spec_from_file_location("dev2_revision_diff", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def files(api: HfApi, repo: str, revision: str) -> dict:
    """revision_diff.py's file listing, with the visibility the hub policy expects (the six are public)."""
    info = api.model_info(repo, revision=revision, files_metadata=True)
    if (
        info.id != repo
        or info.sha != revision
        or info.private is not hub.expected_private(repo)
    ):
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
    rules = card_rules()
    api = HfApi()
    a, b = files(api, repo, released), files(api, repo, new)
    changed = sorted(n for n in set(a) | set(b) if a.get(n) != b.get(n))
    runtime = [n for n in changed if n in a and n in b and n in CHANGEABLE]
    card = [
        n
        for n in changed
        if n in a
        and n in b
        and (n in rules.CARD_FILES or n.startswith(rules.CARD_PREFIXES))
    ]
    hub_files = [n for n in changed if n in rules.HUB_MANAGED]
    other = [
        n for n in changed if n not in runtime and n not in card and n not in hub_files
    ]
    weights = sorted(n for n in a if n.endswith(rules.WEIGHT_SUFFIXES))
    weights_identical = weights == sorted(
        n for n in b if n.endswith(rules.WEIGHT_SUFFIXES)
    ) and all(a[n] == b[n] for n in weights)
    receipt = {
        "schema": "dev2-runtime-diff/1",
        "mode": "runtime-c",
        "repo": repo,
        "released": released,
        "new": new,
        "files": {"released": len(a), "new": len(b)},
        "changed_runtime_files": runtime,
        "changed_card_files": card,
        "changed_hub_managed_files": hub_files,
        "changed_other_files": other,
        "weight_files": len(weights),
        "weight_bytes": sum(a[n]["size"] or 0 for n in weights),
        "weights_byte_identical": weights_identical,
        "ok": weights_identical and not other and REQUIRED <= set(runtime),
    }
    Path(out).write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps({k: v for k, v in receipt.items() if k not in ("released", "new")})
    )
    return 0 if receipt["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
