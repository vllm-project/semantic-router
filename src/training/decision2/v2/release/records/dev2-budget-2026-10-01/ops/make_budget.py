"""Specs and decisions of the forward-token-budget runtime revisions of DEV2.0-0.8B, 2B, 9B and 27B.

Release hand-off of IX1 follow-up A (records/dev2-runtime-forward-budget-2026-10-01.md; user release round
2026-10-01 18:08 UTC+8, release worker 4c0a68cd): the package runtime's decision2/qwen.py runs a request whose
padded question batch would put more than 2**30 elements in a gated-delta q / k / v tensor as several GPU-sized
batches (commits 8e6bdfc33, e876fbefc; integration fea2f016b). DEV2.0-4B carries it already (13d42143); DEV2.0-0.6B
has no gated-delta layers.

Each spec is the auto_map revision's final spec with: runtime_source and automap_source = the mirror of
RUNTIME_COMMIT (an integration commit containing the fix and the auto_map remote code); gate_receipt = the new
decision; one sentence in runtime_equivalence; one runtime line in the card details. Each decision carries the
superseded (auto_map) final decision's judgement forward and changes only the action, the rationale and the
supersedes chain.

Run from src/training/decision2:

  python3 v2/release/records/dev2-budget-2026-10-01/ops/make_budget.py [--check]
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

RECORDS = Path("v2/release/records")
SPECS = Path("v2/release/specs")
PREVIOUS = RECORDS / "dev2-automap-2026-10-01"
OUT = RECORDS / "dev2-budget-2026-10-01"
DECISIONS = "/data/dev2/runs/release/decisions"
RUNTIME_COMMIT = "99432d1a7da5adbc70212df78ae7ebf7e50b41e4"
MARKER = " Checked on one GPU"
ORDER = "0.6B, 0.8B, 2B, 4B, 9B, 27B"
DECIDED_BY = (
    "coordinator (parent agent), Decision 2.0 program: the user's release round of 2026-10-01 18:08 UTC+8 (the "
    "queued IX1 hand-off records/dev2-runtime-forward-budget-2026-10-01.md) to publish runtime-only revisions "
    "carrying the long-request forward token budget for DEV2.0-0.8B, 2B, 9B and 27B, each only with 0 answer "
    "changes on every scored prompt and on mlx-diag, AutoModel equal to native, and the long-input regression "
    "passing"
)
PREPARED_BY = "Decision 2.0 release engineering, release worker 4c0a68cd (worktree vllm-sr-dev2-automap)"
DECIDED_UTC = "2026-10-01T10:08:00Z"
CARD_LINE = (
    "**Runtime update:** very long multi-question requests are processed in GPU-sized batches (previously they "
    "could fail on extremely long inputs); answers unchanged."
)
SENTENCE = (
    " From this revision the runtime runs a request whose padded question batch would put more than 2**30 "
    "elements in a gated-delta q / k / v tensor (the FLA kernels' 32-bit offsets) as several GPU-sized batches, "
    "longest questions first; requests within that budget, every scored prompt among them, keep the single-batch "
    "path, and it was checked with 0 answer changes on every scored prompt and on mlx-diag (2,275)."
)
TIERS = (
    {"tier": "0.8B", "key": "0p8b"},
    {"tier": "2B", "key": "2b"},
    {"tier": "9B", "key": "9b"},
    {"tier": "27B", "key": "27b"},
)
DROPPED = (
    "previous_rationale",
    "runtime_revision",
)


def mirror(commit: str) -> str:
    return f"/data/dev2/src/{commit}-src_training_decision2/src/training/decision2"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def spec_for(t: dict) -> dict:
    source = SPECS / f"dev2-{t['key']}-automap.json"
    old = json.loads(source.read_text(encoding="utf-8"))
    spec = copy.deepcopy(old)
    spec["runtime_source"] = mirror(RUNTIME_COMMIT)
    spec["automap_source"] = mirror(RUNTIME_COMMIT)
    spec["gate_receipt"] = f"{DECISIONS}/DEV2.0-{t['tier']}.decision.budget.json"
    details = spec["card"]["text"]["details"]
    if CARD_LINE not in details:
        details.append(CARD_LINE)
    text = spec["runtime_equivalence"]
    if text.count(MARKER) == 1:
        spec["runtime_equivalence"] = text.replace(MARKER, SENTENCE + MARKER)
    else:
        spec["runtime_equivalence"] = text + SENTENCE
    spec["_release"] = {
        "forward_token_budget": (
            "Runtime-only revision (user release round 2026-10-01 18:08 UTC+8; IX1 hand-off "
            "dev2-runtime-forward-budget-2026-10-01): the package runtime (runtime_source "
            f"{RUNTIME_COMMIT[:9]}) is the auto_map runtime plus the forward token budget of decision2/qwen.py "
            "(8e6bdfc33, e876fbefc). Weights, tokenizer, model configs, the vendored training/model sources and the "
            "Transformers remote code are byte-identical to the replaced revision. Adopted only with 0 answer "
            "changes on every scored prompt and on mlx-diag, AutoModel equal to native and the long-input "
            "regression passing."
        ),
        "replaces_spec": {
            "spec": f"v2/release/specs/{source.name}",
            "sha256": sha(source),
        },
        "previous": old.get("_release"),
    }
    return spec


def decision_for(t: dict, spec_sha: str) -> dict:
    old_path = PREVIOUS / f"DEV2.0-{t['tier']}.decision.automap.json"
    gate_path = PREVIOUS / t["key"] / "release" / "receipts" / "gate.json"
    old = json.loads(old_path.read_text(encoding="utf-8"))
    old_sha = sha(old_path)
    assert old["status"] == "final", t["tier"]
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    assert gate["decision_sha256"] == old_sha, t["tier"]
    repo = old["repo_id"]
    new = {k: v for k, v in old.items() if k not in DROPPED}
    new.update(
        {
            "status": "final",
            "decided_by": DECIDED_BY,
            "prepared_by": PREPARED_BY,
            "decided_utc": DECIDED_UTC,
            "action": (
                f"Runtime-only revision of the private repository {repo}: the package runtime's decision2/qwen.py "
                "gains the forward token budget, so on a GPU a request whose padded question batch would put more "
                "than 2**30 elements in a gated-delta q / k / v tensor runs as several GPU-sized batches instead of "
                "returning invalid answers or faulting; requests within the budget keep the single-batch path "
                f"(runtime_source {RUNTIME_COMMIT[:9]}). The card gains one runtime line. Weights, tokenizer, model "
                "configs, the vendored training/model sources, calibration, Score offsets and the Transformers "
                f"remote code are byte-identical to the released revision {gate['revision']}. The repository stays "
                f"in the private collection '🎲 Decision 2.0', ordered {ORDER}; everything stays private."
            ),
            "rationale": (
                f"The release judgement of the superseded final decision {old_sha[:8]}… stands unchanged: the same "
                f"identity {old['identity']['model_sha256'][:8]}, scored report, paired comparison, calibration and "
                "licence decision. Only decision2/qwen.py, the manifest digests and the card change. Adopted only "
                "with 0 answer changes (0 missing, 0 input mismatches) against the scored predictions on every "
                "scored prompt (typed-final 1,600, css15 6,547, public231 231) and on mlx-diag (2,275) before and "
                "after the real download, AutoModel answers equal to the native runtime's on the same prompts, the "
                "synthetic long-input regression (one prompt at the cap minus 300 tokens, 32 questions, 48 on 0.8B and 2B, so that the batch exceeds the budget) valid and "
                "equal to each question asked alone, and the card's Transformers example run from the Hub in fresh "
                "environments under Transformers 5.17 and 5.18; release.sh blocks the upload or the collection step "
                "otherwise."
            ),
            "previous_rationale": old["rationale"],
            "runtime_revision": {
                "kind": "forward-token-budget",
                "runtime_commit": RUNTIME_COMMIT,
                "fix_commits": ["8e6bdfc33", "e876fbefc"],
                "handoff_record": "v2/release/records/dev2-runtime-forward-budget-2026-10-01.md",
                "spec": f"v2/release/specs/dev2-{t['key']}-budget.json",
                "spec_sha256": spec_sha,
            },
            "supersedes": {
                "final_sha256": old_sha,
                "released_as": f"{repo}@{gate['revision']}",
                "released_manifest_sha256": gate["manifest_sha256"],
                "released_gate_sha256": sha(gate_path),
                "earlier": old.get("supersedes"),
            },
        }
    )
    return new


def main() -> int:
    check = "--check" in sys.argv[1:]
    problems = []
    for t in TIERS:
        spec_text = json.dumps(spec_for(t), ensure_ascii=False, indent=2) + "\n"
        spec_sha = hashlib.sha256(spec_text.encode()).hexdigest()
        decision_text = (
            json.dumps(decision_for(t, spec_sha), ensure_ascii=False, indent=2) + "\n"
        )
        for path, text in (
            (SPECS / f"dev2-{t['key']}-budget.json", spec_text),
            (OUT / f"DEV2.0-{t['tier']}.decision.budget.json", decision_text),
        ):
            if check:
                if not path.is_file() or path.read_text(encoding="utf-8") != text:
                    problems.append(f"{path}: differs from the derivation")
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")
            print(path, hashlib.sha256(text.encode()).hexdigest())
    for problem in problems:
        print(problem, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
