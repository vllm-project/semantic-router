"""Specs and decisions of the Transformers remote-code (auto_map) revisions of the six DEV2.0 repositories.

User request 2026-10-01 10:53 UTC+8 (coordinator note 11:20, worker 4c0a68cd): every DEV2.0 model loads with stock
transformers (AutoConfig / AutoTokenizer / AutoModel / pipeline("decision"), trust_remote_code=True), published only
after the BF16-resident rollout, on top of its revision, with 0 answer changes against the native runtime on every
scored prompt and on mlx-diag.

Each spec is the BF16-resident rollout's final spec with: runtime_source and automap_source = the mirror of
RUNTIME_COMMIT (the rollout's BF16-resident runtime plus the remote-code prompt fix, and the remote code itself);
vendor_source unchanged; gate_receipt = the new decision; remote_code.tested = the Transformers versions the parity
and Hub smoke runs passed with; one sentence in runtime_equivalence. Identity, checkpoint, scored run, calibration,
Score offsets, licence, card and gate profile stay as they are (the card gains its generated Transformers section).
Each decision carries the superseded (BF16-resident) final decision's judgement forward and changes only the
action, the rationale and the supersedes chain.

  draft: specs/dev2-<key>-automap.draft.json and DEV2.0-<tier>.decision.automap.draft.json (status draft, no
         superseded revision yet): preview builds for parity before the rollout finishes; never uploaded.
  final: specs/dev2-<key>-automap.json and DEV2.0-<tier>.decision.automap.json for every tier whose rollout release
         gate (dev2-bf16-resident-2026-10-01/<key>/release/receipts/gate.json) exists.

Run from src/training/decision2:

  python3 v2/release/records/dev2-automap-2026-10-01/ops/make_automap.py draft|final [--check]
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

RECORDS = Path("v2/release/records")
SPECS = Path("v2/release/specs")
ROLLOUT = RECORDS / "dev2-bf16-resident-2026-10-01"
OUT = RECORDS / "dev2-automap-2026-10-01"
DECISIONS = "/data/dev2/runs/release/decisions"
RUNTIME_COMMIT = "8e808244072cef8735b8f3fdcfb8487f36605c5c"
TESTED = ["5.17.0", "5.18.0"]
MARKER = " Checked on one GPU"
ORDER = "0.6B, 0.8B, 2B, 4B, 9B, 27B"
DECIDED_BY = (
    "coordinator (parent agent), Decision 2.0 program: the user's request of 2026-10-01 10:53 UTC+8 that every "
    "DEV2.0 model support standard Hugging Face usage through auto_map / trust_remote_code, assigned to release "
    "engineering in the cross-track note of 2026-10-01 11:20 UTC+8 (worker 4c0a68cd), each revision only with 0 "
    "answer changes against the native runtime on every scored prompt and on mlx-diag"
)
PREPARED_BY = (
    "Decision 2.0 release engineering, auto_map worker (worktree vllm-sr-dev2-automap)"
)
DECIDED_UTC = "2026-10-01T03:20:00Z"
TIERS = (
    {"tier": "0.6B", "key": "0p6b"},
    {"tier": "0.8B", "key": "0p8b"},
    {"tier": "2B", "key": "2b"},
    {"tier": "4B", "key": "4b"},
    {"tier": "9B", "key": "9b"},
    {
        "tier": "27B",
        "key": "27b",
        "note": "The pinned base is about 52 GB; `base_path=` takes a local copy of that revision instead.",
    },
)
DROPPED = (
    "previous_rationale",
    "runtime_revision",
    "approved_package",
    "verified_package",
    "card_revision",
    "revision_kind",
    "package_verification",
    "prepared_by_release_engineering",
)


def mirror(commit: str) -> str:
    return f"/data/dev2/src/{commit}-src_training_decision2/src/training/decision2"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


SENTENCE = (
    " From this revision the package also ships 🤗 Transformers remote code (configuration_decision2.py, "
    "modeling_decision2.py, pipeline_decision2.py; config.json gains model_type, auto_map and custom_pipelines): "
    "AutoModel with trust_remote_code loads the package through this runtime, which now also refuses Transformers' "
    "remote-code prompt while loading the tokenizer; it was checked with 0 answer changes against the native runtime "
    "on every scored prompt and on mlx-diag (2,275)."
)


def spec_for(t: dict, draft: bool) -> dict:
    source = SPECS / f"dev2-{t['key']}-bf16r.json"
    old = json.loads(source.read_text(encoding="utf-8"))
    spec = copy.deepcopy(old)
    spec["runtime_source"] = mirror(RUNTIME_COMMIT)
    spec["automap_source"] = mirror(RUNTIME_COMMIT)
    spec["remote_code"] = {"tested": TESTED}
    if t.get("note"):
        spec["card"]["text"]["transformers_note"] = t["note"]
    suffix = ".draft" if draft else ""
    spec["gate_receipt"] = (
        f"{DECISIONS}/DEV2.0-{t['tier']}.decision.automap{suffix}.json"
    )
    text = spec["runtime_equivalence"]
    if text.count(MARKER) == 1:
        spec["runtime_equivalence"] = text.replace(MARKER, SENTENCE + MARKER)
    else:
        spec["runtime_equivalence"] = text + SENTENCE
    spec["_release"] = {
        "transformers_remote_code": (
            "Runtime-only revision (user request 2026-10-01 10:53 UTC+8; coordinator note 11:20, worker 4c0a68cd): "
            "the package gains the Transformers remote code and config.json fields; the package runtime "
            f"(runtime_source {RUNTIME_COMMIT[:9]}) is the BF16-resident runtime plus the remote-code prompt fix. "
            "Weights, tokenizer, model configs and the vendored training/model sources are byte-identical to the "
            "replaced revision. Adopted only with 0 answer changes against the native runtime on every scored "
            "prompt and on mlx-diag."
        ),
        "replaces_spec": {
            "spec": f"v2/release/specs/{source.name}",
            "sha256": sha(source),
        },
        "previous": old.get("_release"),
    }
    return spec


def decision_for(t: dict, spec_sha: str, draft: bool) -> dict:
    old_path = ROLLOUT / f"DEV2.0-{t['tier']}.decision.bf16r.json"
    gate_path = ROLLOUT / t["key"] / "release" / "receipts" / "gate.json"
    old = json.loads(old_path.read_text(encoding="utf-8"))
    old_sha = sha(old_path)
    assert old["status"] == "final", t["tier"]
    gate = (
        json.loads(gate_path.read_text(encoding="utf-8"))
        if gate_path.is_file()
        else None
    )
    if gate is not None:
        assert gate["decision_sha256"] == old_sha, t["tier"]
    elif not draft:
        raise FileNotFoundError(gate_path)
    repo = old["repo_id"]
    revision = gate["revision"] if gate else "the BF16-resident revision (pending)"
    new = {k: v for k, v in old.items() if k not in DROPPED}
    new.update(
        {
            "status": "draft" if draft else "final",
            "prepared_by": PREPARED_BY,
            "decided_utc": DECIDED_UTC,
            "action": (
                f"Runtime-only revision of the private repository {repo}: the package gains Transformers remote code "
                "(configuration_decision2.py, modeling_decision2.py, pipeline_decision2.py at the root; config.json "
                "model_type decision2, auto_map, custom_pipelines decision) so AutoConfig, AutoTokenizer, AutoModel "
                "and pipeline('decision') load it with trust_remote_code=True through its own runtime, and the card "
                "gains a 'Use with 🤗 Transformers' section. The package runtime is the BF16-resident runtime plus "
                f"the remote-code prompt fix (runtime_source {RUNTIME_COMMIT[:9]}). Weights, tokenizer, model "
                "configs, the vendored training/model sources, calibration and any Score offsets are byte-identical "
                f"to the released revision {revision}. The repository stays in the private collection "
                f"'🎲 Decision 2.0', ordered {ORDER}; everything stays private."
            ),
            "rationale": (
                f"The release judgement of the superseded final decision {old_sha[:8]}… stands unchanged: the same "
                f"identity {old['identity']['model_sha256'][:8]}, scored report, paired comparison, calibration and "
                "licence decision. Only package code, config.json and the card change. Adopted only with 0 answer "
                "changes (0 missing, 0 input mismatches) against the native runtime on every scored prompt "
                "(typed-final 1,600, css15 6,547, public231 231) and on mlx-diag (2,275), compared prompt by prompt, "
                "with native parity against the scored predictions before and after the real download, and the "
                "card's Transformers example run from the Hub in a fresh environment; release.sh blocks the upload "
                "or the collection step otherwise."
            ),
            "previous_rationale": old["rationale"],
            "runtime_revision": {
                "kind": "transformers-remote-code",
                "runtime_commit": RUNTIME_COMMIT,
                "spec": f"v2/release/specs/dev2-{t['key']}-automap{'.draft' if draft else ''}.json",
                "spec_sha256": spec_sha,
                "tested_transformers": TESTED,
            },
            "supersedes": {
                "final_sha256": old_sha,
                "released_as": f"{repo}@{gate['revision']}" if gate else None,
                "released_manifest_sha256": gate["manifest_sha256"] if gate else None,
                "released_gate_sha256": sha(gate_path) if gate else None,
                "earlier": old.get("supersedes"),
            },
        }
    )
    if "c1" in old:
        new["c1"] = old["c1"]
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
        gate = ROLLOUT / t["key"] / "release" / "receipts" / "gate.json"
        if not draft and not gate.is_file():
            print(f"{t['tier']}: rollout release not recorded yet, skipped")
            continue
        suffix = ".draft" if draft else ""
        spec_text = json.dumps(spec_for(t, draft), ensure_ascii=False, indent=2) + "\n"
        spec_sha = hashlib.sha256(spec_text.encode()).hexdigest()
        decision = decision_for(t, spec_sha, draft)
        decision_text = json.dumps(decision, ensure_ascii=False, indent=2) + "\n"
        for path, text in (
            (SPECS / f"dev2-{t['key']}-automap{suffix}.json", spec_text),
            (OUT / f"DEV2.0-{t['tier']}.decision.automap{suffix}.json", decision_text),
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
