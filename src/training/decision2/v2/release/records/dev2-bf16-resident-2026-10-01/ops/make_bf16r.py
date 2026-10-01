"""Specs and decisions of the BF16-resident runtime-only revisions of the six DEV2.0 repositories.

Coordinator note 2026-10-01 10:30 UTC+8 (b5f60b33): publish each repository again with the package runtime of
the BF16-resident commit, only with 0 answer changes on every scored prompt and on mlx-diag.

Each spec is the spec that built the current revision with: runtime_source = the mirror of the BF16-resident
runtime commit; vendor_source = the mirror whose training/model sources the current revision vendors (unchanged);
gate_receipt = the new decision; one runtime sentence in runtime_equivalence; one "Runtime update" line in the
card details. Identity, checkpoint, scored run, calibration, Score offsets, licence and gate profile stay as they
are. Each decision carries the superseded final decision's judgement forward (its gate profile included: own 1.0,
or successor with the same current revision and evidence) and changes only the action, the rationale and the
supersedes chain.

  draft: specs/dev2-<key>-bf16r.draft.json and DEV2.0-<tier>.decision.bf16r.draft.json (status draft, card line
         without latency): the preview build that the old-vs-new bench measures; never uploaded.
  final: specs/dev2-<key>-bf16r.json and DEV2.0-<tier>.decision.bf16r.json for every tier whose bench comparison
         (<key>/bench/compare.json, copied from the node) shows identical answers; the card line states its latency.

Run from src/training/decision2:

  python3 v2/release/records/dev2-bf16-resident-2026-10-01/ops/make_bf16r.py draft|final [--check]
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

RECORDS = Path("v2/release/records")
SPECS = Path("v2/release/specs")
OUT = RECORDS / "dev2-bf16-resident-2026-10-01"
DECISIONS = "/data/dev2/runs/release/decisions"
RUNTIME_COMMIT = "5dc962b003cc15b12a3568fe3c8ac8c2931b03a0"


def mirror(commit: str) -> str:
    return f"/data/dev2/src/{commit}-src_training_decision2/src/training/decision2"


DECIDED_BY = (
    "coordinator (parent agent), Decision 2.0 program: the BF16-resident runtime rollout (runtime-only revisions of "
    "the six released repositories, each only with 0 answer changes on every scored prompt and on mlx-diag), "
    "assigned to release engineering in the cross-track note of 2026-10-01 10:30 UTC+8 (b5f60b33) and adopted as "
    "the default under the user's full-autonomy mandate"
)
PREPARED_BY = "Decision 2.0 release engineering, BF16-resident runtime worker (worktree vllm-sr-dev2-release)"
DECIDED_UTC = "2026-10-01T02:30:00Z"
ORDER = "0.6B, 0.8B, 2B, 4B, 9B, 27B"
MARKER = " Checked on one GPU"
TIERS = (
    {
        "tier": "0.6B",
        "key": "0p6b",
        "spec": "dev2-0p6b-card-c1pk.json",
        "previous": "dev2-0p6b-c1postkey-card-2026-09-30",
        "decision": "DEV2.0-0.6B.decision.json",
        "gate": "0p6b/receipts/gate.json",
        "vendor": "2f21790bafc6e172d565236a71101c3ad4a95424",
    },
    {
        "tier": "0.8B",
        "key": "0p8b",
        "spec": "dev2-0p8b-bf16.json",
        "previous": "dev2-bf16-storage-2026-09-30",
        "decision": "DEV2.0-0.8B.decision.json",
        "gate": "0p8b/release/receipts/gate.json",
    },
    {
        "tier": "2B",
        "key": "2b",
        "spec": "dev2-2b-bf16.json",
        "previous": "dev2-bf16-storage-2026-09-30",
        "decision": "DEV2.0-2B.decision.json",
        "gate": "2b/release/receipts/gate.json",
    },
    {
        "tier": "4B",
        "key": "4b",
        "spec": "dev2-4b-bf16.json",
        "previous": "dev2-bf16-storage-2026-09-30",
        "decision": "DEV2.0-4B.decision.json",
        "gate": "4b/release/receipts/gate.json",
    },
    {
        "tier": "9B",
        "key": "9b",
        "spec": "dev2-9b-card-c1.json",
        "previous": "dev2-c1-card-pass-2026-09-29",
        "decision": "DEV2.0-9B.decision.json",
        "gate": "9b/receipts/gate.json",
    },
    {
        "tier": "27B",
        "key": "27b",
        "spec": "dev2-27b-a20r-release.json",
        "previous": "dev2-27b-a20r-release-2026-09-30",
        "decision": "DEV2.0-27B.decision.a20r.json",
        "gate": "release/receipts/gate.json",
        "lora": True,
    },
)
DROPPED = (
    "approved_package",
    "verified_package",
    "card_revision",
    "previous_rationale",
    "revision_kind",
    "package_verification",
    "prepared_by_release_engineering",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def runtime_sentence(t: dict) -> str:
    lora = "; the FP32 LoRA factors stay FP32 and unmerged" if t.get("lora") else ""
    return (
        " From this revision the runtime holds the backbone's BF16-exact Linear weights in BF16, the values BF16 "
        f"autocast multiplies with, instead of FP32 copies cast before every matmul{lora}; it was checked with 0 answer "
        "changes on every scored prompt and on mlx-diag (2,275)."
    )


def card_line(bench: dict | None, run: dict | None) -> str:
    if bench is None:
        return "**Runtime update:** BF16-resident weights; answers unchanged."
    lat, mem = bench["latency_ms"], bench["memory_gib"]["request_peak"]
    return (
        "**Runtime update:** BF16-resident weights; answers unchanged; latency p50 "
        f"{lat['p50']['old']:.1f} → {lat['p50']['new']:.1f} ms, p95 {lat['p95']['old']:.1f} → "
        f"{lat['p95']['new']:.1f} ms; peak GPU memory {mem['old']:.1f} → {mem['new']:.1f} GiB "
        f"({bench['items']} single requests, {run['input_tokens_mean']:.0f} input tokens on average, one AMD "
        "MI325X GPU)."
    )


def spec_for(t: dict, bench: dict | None, run: dict | None, draft: bool) -> dict:
    old = json.loads((SPECS / t["spec"]).read_text(encoding="utf-8"))
    spec = copy.deepcopy(old)
    if not spec.get("vendor_source"):
        spec["vendor_source"] = mirror(t["vendor"])
    spec["runtime_source"] = mirror(RUNTIME_COMMIT)
    suffix = ".draft" if draft else ""
    spec["gate_receipt"] = f"{DECISIONS}/DEV2.0-{t['tier']}.decision.bf16r{suffix}.json"
    text = spec["runtime_equivalence"]
    sentence = runtime_sentence(t)
    if text.count(MARKER) == 1:
        spec["runtime_equivalence"] = text.replace(MARKER, sentence + MARKER)
    else:
        spec["runtime_equivalence"] = text + sentence
    details = spec["card"]["text"].setdefault("details", [])
    details.append(card_line(bench, run))
    spec["_release"] = {
        "bf16_resident_runtime": (
            "Runtime-only revision (coordinator note 2026-10-01 10:30 UTC+8, b5f60b33): the package runtime comes "
            f"from the BF16-resident commit {RUNTIME_COMMIT[:9]} (runtime_source); weights, tokenizer, configs and "
            "the vendored training/model sources (vendor_source, the mirror the replaced revision vendors) are "
            "byte-identical to the replaced revision. Adopted only with 0 answer changes on every scored prompt and "
            "on mlx-diag. The card gains one runtime-update line and runtime_equivalence one runtime sentence."
        ),
        "replaces_spec": {
            "spec": f"v2/release/specs/{t['spec']}",
            "sha256": sha(SPECS / t["spec"]),
        },
        "previous": old.get("_release"),
    }
    return spec


def decision_for(t: dict, spec_sha: str, bench: dict | None, draft: bool) -> dict:
    root = RECORDS / t["previous"]
    old_path, gate_path = root / t["decision"], root / t["gate"]
    old = json.loads(old_path.read_text(encoding="utf-8"))
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    old_sha = sha(old_path)
    assert gate["decision_sha256"] == old_sha and old["status"] == "final", t["tier"]
    repo, revision = old["repo_id"], gate["revision"]
    new = {k: v for k, v in old.items() if k not in DROPPED}
    successor = old.get("gate_profile") == "successor"
    new.update(
        {
            "status": "draft" if draft else "final",
            "prepared_by": PREPARED_BY,
            "decided_utc": DECIDED_UTC,
            "action": (
                f"Runtime-only revision of the private repository {repo}: the package runtime decision2/*.py comes "
                f"from the BF16-resident runtime commit {RUNTIME_COMMIT[:9]}, which on a GPU holds the backbone's "
                "BF16-exact Linear weights in BF16 instead of FP32 copies cast before every matmul. Weights, "
                "tokenizer, configs, the vendored training/model sources, calibration and any Score offsets are "
                f"byte-identical to the released revision {revision}; the card gains one runtime-update line and "
                "MODEL_MANIFEST.json the new runtime hashes and one runtime sentence. The repository stays in the "
                f"private collection '🎲 Decision 2.0', ordered {ORDER}; everything stays private."
            ),
            "rationale": (
                f"The release judgement of the superseded final decision {old_sha[:8]}… stands unchanged: the same "
                f"identity {old['identity']['model_sha256'][:8]}, scored report, paired comparison, calibration and "
                "licence decision"
                + (
                    ", and the successor-profile evidence against the same current revision"
                    if successor
                    else ""
                )
                + ". Only the package runtime changes, and it computes the same BF16 products as before. Adopted only "
                "with exact parity (0 answer changes, 0 missing, 0 input mismatches) against the scored predictions "
                "on every scored prompt (typed-final 1,600, css15 6,547, public231 231) and on mlx-diag (2,275), with "
                "the scored image, kernels and frozen autotune cache: release.sh parity-pre blocks the upload "
                "otherwise, and parity-post repeats it on the real download."
            ),
            "previous_rationale": old["rationale"],
            "runtime_revision": {
                "kind": "bf16-resident-runtime",
                "runtime_commit": RUNTIME_COMMIT,
                "spec": f"v2/release/specs/dev2-{t['key']}-bf16r{'.draft' if draft else ''}.json",
                "spec_sha256": spec_sha,
                "bench": bench
                and {
                    "compare": f"v2/release/records/dev2-bf16-resident-2026-10-01/{t['key']}/bench/compare.json",
                    "items": bench["items"],
                    "bit_identical_items": bench["bit_identical_items"],
                    "p50_ms": [
                        bench["latency_ms"]["p50"]["old"],
                        bench["latency_ms"]["p50"]["new"],
                    ],
                    "p95_ms": [
                        bench["latency_ms"]["p95"]["old"],
                        bench["latency_ms"]["p95"]["new"],
                    ],
                    "request_peak_gib": [
                        bench["memory_gib"]["request_peak"]["old"],
                        bench["memory_gib"]["request_peak"]["new"],
                    ],
                },
            },
            "supersedes": {
                "final_sha256": old_sha,
                "released_as": f"{repo}@{revision}",
                "released_manifest_sha256": gate["manifest_sha256"],
                "released_gate_sha256": sha(gate_path),
                "card_revision": old.get("card_revision"),
                "earlier": old.get("supersedes"),
            },
        }
    )
    if "c1" in old:
        new["c1"] = (
            old["c1"]
            + " The runtime-only revision computes the same products, so this line stays valid for it."
        )
    if draft:
        new.pop("decided_by", None)
    else:
        new["decided_by"] = DECIDED_BY
    return new


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode not in ("draft", "final"):
        print(__doc__, file=sys.stderr)
        return 2
    check = "--check" in sys.argv[2:]
    draft = mode == "draft"
    problems = []
    for t in TIERS:
        bench = run = None
        if not draft:
            path = OUT / t["key"] / "bench" / "compare.json"
            if not path.is_file():
                print(f"{t['tier']}: no bench comparison yet, skipped")
                continue
            bench = json.loads(path.read_text(encoding="utf-8"))
            run = json.loads(
                (path.parent / "bench-new.json").read_text(encoding="utf-8")
            )
            if sha(path.parent / "bench-new.json") != bench["new"]["sha256"]:
                problems.append(f"{t['tier']}: bench-new.json is not the compared run")
                continue
            if not bench["passed"] or bench["bit_identical_items"] != bench["items"]:
                print(
                    f"{t['tier']}: bench answers differ, no final spec", file=sys.stderr
                )
                problems.append(t["tier"])
                continue
        spec = spec_for(t, bench, run, draft)
        suffix = ".draft" if draft else ""
        spec_path = SPECS / f"dev2-{t['key']}-bf16r{suffix}.json"
        spec_text = json.dumps(spec, ensure_ascii=False, indent=2) + "\n"
        spec_sha = hashlib.sha256(spec_text.encode()).hexdigest()
        decision = decision_for(t, spec_sha, bench, draft)
        target = OUT / f"DEV2.0-{t['tier']}.decision.bf16r{suffix}.json"
        decision_text = json.dumps(decision, ensure_ascii=False, indent=2) + "\n"
        for path, text in ((spec_path, spec_text), (target, decision_text)):
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
