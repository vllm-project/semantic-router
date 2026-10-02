"""Specs and decisions of the organization card-only revisions of the Decision 2.0 repositories (2026-10-02).

User instruction 2026-10-02 17:35 UTC+8 (COORDINATION 17:35): the Hugging Face organization llm-semantic-router is now
vllm-sr; its repositories and the Decision 2.0 collection moved with it and the old IDs redirect. Org worker (2)
publishes card-only revisions of the six releases so that README.md and MODEL_MANIFEST.json name vllm-sr. Each card
keeps its release's Index data, charts and banner; only the organization changes.

Each spec is the exact spec that built the repository's current main (receipts/spec.json of that release's work
directory on the node), with its Hub references on the renamed organization (v2.release.layout.current_ids; local
paths and the _release history unchanged) and gate_receipt = the new decision. Each decision carries the superseded
final decision's judgement forward and changes only the repository ID, the action, the rationale and the supersedes
chain.

Run on the node that holds the release's work directory, from an exact mirror (cwd src/training/decision2):

  python3 v2/release/records/dev2-org-2026-10-02/ops/make_org.py --tiers 4B ... (--out DIR | --check)
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

from v2.release import layout

RECORDS = Path("v2/release/records")
SPECS = Path("v2/release/specs")
OUT = RECORDS / "dev2-org-2026-10-02"
DECISIONS = "/data/dev2/runs/release/decisions"
RELEASES = "/data/dev2/runs/release"
DECIDED_BY = (
    "coordinator (parent agent), Decision 2.0 program: the user's instruction of 2026-10-02 17:35 UTC+8 (the Hugging "
    "Face organization llm-semantic-router is now vllm-sr): card-only revisions of the Decision 2.0 repositories "
    "naming the new organization; every card keeps its release's Index data, charts and banner"
)
PREPARED_BY = (
    "Decision 2.0 release engineering, org worker (2) (worktree vllm-sr-dev2-org20)"
)
DECIDED_UTC = "2026-10-02T09:35:00Z"
# The current main of each repository and the release work directory that built and sealed it.
TIERS = {
    "0.6B": (
        "0p6b",
        "Kai",
        "479ea8d1b46dfc87361d620029f1c27a9d304cd0",
        "dev2-card4-0.6B-20261002T015806Z",
    ),
    "0.8B": (
        "0p8b",
        "Eos",
        "3de61185ade42cc307c26a6062df4baac327eac5",
        "dev2-0p8b-ixf-release-20261002T043541Z",
    ),
    "2B": (
        "2b",
        "Sol",
        "1b7c47eafa3ffec1f4f4b79b0abfe9309d583439",
        "dev2-2b-ixf-release-20261002T075545Z",
    ),
    "4B": (
        "4b",
        "Nox",
        "9f2ddafc57f60c62239c0ca47c54c5cd3ec4aff9",
        "dev2-4b-sdml-release-20261002T085835Z",
    ),
    "9B": (
        "9b",
        "Lux",
        "7195360df53b30ae625435fb3d9ec21bdbda279c",
        "dev2-card4-9B-20261002T033640Z",
    ),
    "27B": (
        "27b",
        "Vega",
        "781b2b2431abccbdcfcbfa98b66c80b332725513",
        "dev2-27b-27bif-M6-IB-release-20261002T070233Z",
    ),
}


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sources(tier: str) -> dict:
    """The release that built the current main: its spec, sealed gate and final decision, cross-checked."""
    key, codename, revision, work = TIERS[tier]
    name = f"Decision-2.0-{codename}-{tier}"
    receipts = Path(RELEASES) / work / "receipts"
    spec_path, gate_path = receipts / "spec.json", receipts / "gate.json"
    spec, gate, build = (
        _json(spec_path),
        _json(gate_path),
        _json(receipts / "build.json"),
    )
    decision_path = Path(spec["gate_receipt"])
    problems = [
        message
        for ok, message in (
            (spec["model_name"] == name, "the spec names another model"),
            (gate["revision"] == revision, "the gate seals another revision"),
            (gate["model_name"] == name, "the gate names another model"),
            (
                gate["decision_sha256"] == sha(decision_path),
                "the gate seals another decision",
            ),
            (build["spec_sha256"] == sha(spec_path), "the build used another spec"),
            (_json(decision_path)["status"] == "final", "the decision is not final"),
        )
        if not ok
    ]
    if problems:
        raise ValueError(f"{tier}: {'; '.join(problems)}")
    return {
        "key": key,
        "name": name,
        "revision": revision,
        "spec_path": spec_path,
        "spec": spec,
        "gate_path": gate_path,
        "gate": gate,
        "decision_path": decision_path,
        "decision": _json(decision_path),
    }


def spec_for(src: dict) -> dict:
    old = src["spec"]
    spec = layout.current_ids({k: v for k, v in old.items() if k != "_release"})
    spec["gate_receipt"] = f"{DECISIONS}/{src['name']}.decision.org.json"
    spec["_release"] = {
        "card_org": (
            "Card-only revision (user instruction 2026-10-02 17:35 UTC+8, org worker (2)): the Hugging Face "
            "organization llm-semantic-router is now vllm-sr, so README.md and MODEL_MANIFEST.json name "
            f"{spec['repo_id']}. The spec is the one that built {old['repo_id']}@{src['revision'][:8]} with its Hub "
            "references on the renamed organization; card.index, card.assets and every model, runtime and "
            "remote-code input are unchanged, so the Index data, the charts and the banner stay as released."
        ),
        "replaces_spec": {
            "spec": str(src["spec_path"]),
            "sha256": sha(src["spec_path"]),
        },
        "previous": old.get("_release"),
    }
    return spec


def decision_for(src: dict, spec_sha: str) -> dict:
    old, gate = src["decision"], src["gate"]
    old_sha = sha(src["decision_path"])
    repo = layout.current_repo(old["repo_id"])
    new = copy.deepcopy(old)
    new.update(
        {
            "repo_id": repo,
            "decided_by": DECIDED_BY,
            "prepared_by": PREPARED_BY,
            "decided_utc": DECIDED_UTC,
            "action": (
                f"Card-only revision of the private repository {repo} (formerly {old['repo_id']}): README.md and "
                "MODEL_MANIFEST.json name the renamed Hugging Face organization vllm-sr. Every other file (the "
                "banner and the four charts, every model, tokenizer, runtime and remote-code file) is "
                f"byte-identical to the released revision {gate['revision']}. The collection is not changed; "
                "everything stays private."
            ),
            "rationale": (
                f"The release judgement of the superseded final decision {old_sha[:8]}… stands unchanged: the same "
                f"identity {old['identity']['model_sha256'][:8]}, scored report, paired comparison, calibration and "
                "licence decision. Only the organization in the card and the manifest changes; release.sh checks "
                "the native examples, the card's Transformers example before upload, after the real download and "
                "from the Hub in fresh environments under Transformers 5.17 and 5.18, the card structure and every "
                "card link."
            ),
            "previous_rationale": old["rationale"],
            "card_revision": {
                "kind": "card-org",
                "spec": f"v2/release/specs/dev2-{src['key']}-org.json",
                "spec_sha256": spec_sha,
            },
            "supersedes": {
                "final_sha256": old_sha,
                "released_as": f"{gate['repo_id']}@{gate['revision']}",
                "released_manifest_sha256": gate["manifest_sha256"],
                "released_gate_sha256": sha(src["gate_path"]),
                "earlier": old.get("supersedes"),
            },
        }
    )
    return new


def texts(tier: str) -> list[tuple[Path, str]]:
    src = sources(tier)
    spec_text = json.dumps(spec_for(src), ensure_ascii=False, indent=2) + "\n"
    spec_sha = hashlib.sha256(spec_text.encode()).hexdigest()
    decision_text = (
        json.dumps(decision_for(src, spec_sha), ensure_ascii=False, indent=2) + "\n"
    )
    return [
        (SPECS / f"dev2-{src['key']}-org.json", spec_text),
        (OUT / f"{src['name']}.decision.org.json", decision_text),
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", nargs="+", required=True, choices=sorted(TIERS))
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--out", type=Path, help="write the derived files under this directory"
    )
    mode.add_argument(
        "--check", action="store_true", help="compare with the files in this tree"
    )
    args = ap.parse_args()
    problems = []
    for tier in args.tiers:
        for path, text in texts(tier):
            if args.check:
                if not path.is_file() or path.read_text(encoding="utf-8") != text:
                    problems.append(f"{path}: differs from the derivation")
            else:
                target = args.out / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(text, encoding="utf-8")
            print(path, hashlib.sha256(text.encode()).hexdigest())
    for problem in problems:
        print(problem, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
