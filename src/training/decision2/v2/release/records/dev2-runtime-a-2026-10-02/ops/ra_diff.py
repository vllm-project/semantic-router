"""Files that differ between a released revision and its phase A runtime-only successor (hf-cli python, node token).

  <hf-cli python> ra_diff.py <repo> <released revision> <new revision> <out.json> [--switch]

The BF16-resident rollout's runtime_diff.py with the two fast-path modules of the phase A runtime, which the
successor adds: a runtime-only revision may change decision2/__init__.py, api.py and the profile module, add
decision2/fast.py and fast_kernels.py (never decision2/_vendor/), change the card files that revision_diff.py allows
and MODEL_MANIFEST.json; every other file, and every weight file in particular, must keep its size and LFS SHA-256
or git blob id, and nothing is removed. Exits 1 otherwise. With --switch (the opt-in shared-context switch, which
may follow a phase A revision or carry phase A itself) the fast-path modules may also change, and the successor must
add decision2/shared_ctx.py.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from huggingface_hub import HfApi

RUNTIME_FILES = {
    "decision2/__init__.py",
    "decision2/api.py",
    "decision2/qwen.py",
    "decision2/kai_native.py",
}
ADDED_RUNTIME_FILES = {"decision2/fast.py", "decision2/fast_kernels.py"}
SWITCH_FILE = "decision2/shared_ctx.py"


def card_rules():
    path = (
        Path(__file__).resolve().parents[2]
        / "dev2-rename-9b-27b-2026-09-29/ops/revision_diff.py"
    )
    spec = importlib.util.spec_from_file_location("dev2_revision_diff", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    repo, released, new, out = sys.argv[1:5]
    switch = sys.argv[5:] == ["--switch"]
    changeable = RUNTIME_FILES | (ADDED_RUNTIME_FILES if switch else set())
    addable = ADDED_RUNTIME_FILES | ({SWITCH_FILE} if switch else set())
    required = {SWITCH_FILE} if switch else ADDED_RUNTIME_FILES
    rules = card_rules()
    api = HfApi()
    a, b = rules.files(api, repo, released), rules.files(api, repo, new)
    changed = sorted(n for n in set(a) | set(b) if a.get(n) != b.get(n))
    runtime = [
        n
        for n in changed
        if n in b and ((n in changeable and n in a) or (n in addable and n not in a))
    ]
    card = [
        n for n in changed if n in rules.CARD_FILES or n.startswith(rules.CARD_PREFIXES)
    ]
    hub = [n for n in changed if n in rules.HUB_MANAGED]
    other = [n for n in changed if n not in runtime and n not in card and n not in hub]
    weights = sorted(n for n in a if n.endswith(rules.WEIGHT_SUFFIXES))
    weights_identical = weights == sorted(
        n for n in b if n.endswith(rules.WEIGHT_SUFFIXES)
    ) and all(a[n] == b[n] for n in weights)
    receipt = {
        "schema": "dev2-runtime-diff/1",
        "mode": "switch" if switch else "phase-a",
        "repo": repo,
        "released": released,
        "new": new,
        "files": {"released": len(a), "new": len(b)},
        "changed_runtime_files": runtime,
        "changed_card_files": card,
        "changed_hub_managed_files": hub,
        "changed_other_files": other,
        "weight_files": len(weights),
        "weight_bytes": sum(a[n]["size"] or 0 for n in weights),
        "weights_byte_identical": weights_identical,
        "ok": weights_identical and not other and required <= set(runtime),
    }
    Path(out).write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps({k: v for k, v in receipt.items() if k not in ("released", "new")})
    )
    return 0 if receipt["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
