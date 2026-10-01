"""Specs and decisions of the product-card revisions under the Decision 2.0 names.

User job 2026-10-02 00:05 UTC+8 (COORDINATION; release worker 4c0a68cd): the six repositories were renamed
DEV2.0-<tier> -> Decision-2.0-<Codename>-<tier> with move_repo (old IDs redirect), and their cards become product
cards: banner, title, one paragraph, an at-a-glance table, highlights, a code-only quickstart, evaluation (JevArena
overall and by type, the Jev Decision Index against size and by area, one compact table), licence, citation. No
training details, limitations, NOTICE, ATTRIBUTIONS.md or evaluation/ remain; the package keeps the Apache-2.0
LICENSE and the base_model metadata. Card-only: weights, tokenizer, the package runtime and the Transformers remote
code stay byte-identical to the released revisions; config.json and MODEL_MANIFEST.json carry the new name.

Each spec is the round-1 card spec with: the new repo_id, model_name and candidate label; card.text emptied (the
card writes its own product text); card.index, card.assets and card.speed pinned by SHA-256 (the Index input and the
rendered assets stay in a private node directory; this file names only their digests); licence.files reduced to the
LICENSE, without the upstream NOTICE and the attributions; gate_receipt = the new decision. Each decision carries the
superseded round-1 decision's judgement forward and changes the names, the action, the rationale and the supersedes
chain.

Run from src/training/decision2 (the digests come from the private inputs, given on the command line):

  python3 v2/release/records/dev2-card2-2026-10-02/ops/make_card2.py --pins PINS.json [--check]

PINS.json: {"index_sha256": ..., "assets": {"<tier>": receipt_sha256, ...}}.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

RECORDS = Path("v2/release/records")
SPECS = Path("v2/release/specs")
ROUND1 = RECORDS / "dev2-card-2026-10-01"
OUT = RECORDS / "dev2-card2-2026-10-02"
DECISIONS = "/data/dev2/runs/release/decisions"
PRIVATE = "/data/dev2/private/release/card2"
INDEX = f"{PRIVATE}/decision-index-card.json"
ORG = "llm-semantic-router"
ORDER = "Kai 0.6B, Eos 0.8B, Sol 2B, Nox 4B, Lux 9B, Vega 27B"
DECIDED_BY = (
    "coordinator (parent agent), Decision 2.0 program: the user's job of 2026-10-02 00:05 UTC+8 (rename the six "
    "DEV2.0 repositories to Decision-2.0-<Codename>-<tier>; product cards with a banner, product highlights, a "
    "code-only quickstart and JevArena / Jev Decision Index charts; no training details, limitations, NOTICE, "
    "attributions or evaluation/ files), card-only revisions of all six repositories"
)
PREPARED_BY = "Decision 2.0 release engineering, release worker 4c0a68cd (worktree vllm-sr-dev2-automap)"
DECIDED_UTC = "2026-10-01T16:05:00Z"
CODENAMES = {
    "0.6B": "Kai",
    "0.8B": "Eos",
    "2B": "Sol",
    "4B": "Nox",
    "9B": "Lux",
    "27B": "Vega",
}
TIERS = (
    {
        "tier": "0.6B",
        "key": "0p6b",
        "speed": RECORDS / "dev2-bf16-resident-2026-10-01/0p6b/bench/bench-new.json",
    },
    {
        "tier": "0.8B",
        "key": "0p8b",
        "speed": RECORDS / "dev2-budget-2026-10-01/0p8b/extra/receipts/bench-new.json",
    },
    {
        "tier": "2B",
        "key": "2b",
        "speed": RECORDS / "dev2-budget-2026-10-01/2b/extra/receipts/bench-new.json",
    },
    {
        "tier": "4B",
        "key": "4b",
        "speed": RECORDS
        / "dev2-4b-m10lh-2026-10-01/prerelease/dev2-4b-lh-bench-20261001T080621Z/receipts/bench-new.json",
    },
    {
        "tier": "9B",
        "key": "9b",
        "speed": RECORDS / "dev2-budget-2026-10-01/9b/extra/receipts/bench-new.json",
    },
    {
        "tier": "27B",
        "key": "27b",
        "speed": RECORDS / "dev2-budget-2026-10-01/27b/extra/receipts/bench-new.json",
    },
)
DROPPED_DECISION = ("previous_rationale", "card_revision")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def names(tier: str) -> tuple[str, str, str]:
    old = f"DEV2.0-{tier}"
    new = f"Decision-2.0-{CODENAMES[tier]}-{tier}"
    return old, new, f"{ORG}/{new}"


def rename(value, old: str, new: str):
    """Replace the former model name in spec strings (licence components, notes)."""
    if isinstance(value, str):
        return value.replace(old, new)
    if isinstance(value, list):
        return [rename(v, old, new) for v in value]
    if isinstance(value, dict):
        return {k: rename(v, old, new) for k, v in value.items()}
    return value


def spec_for(t: dict, pins: dict) -> dict:
    tier, key = t["tier"], t["key"]
    old_name, name, repo = names(tier)
    source = SPECS / f"dev2-{key}-card.json"
    old = json.loads(source.read_text(encoding="utf-8"))
    assert old["repo_id"] == f"{ORG}/{old_name}" and old["model_name"] == old_name, tier
    spec = copy.deepcopy(old)
    spec["repo_id"], spec["model_name"] = repo, name
    lic = spec["licence"]
    lic["components"] = rename(lic["components"], old_name, name)
    lic["files"] = [f for f in lic["files"] if f["path"] == "LICENSE"]
    assert len(lic["files"]) == 1, tier
    for dropped in ("notice", "attributions"):
        lic.pop(dropped, None)
    card = spec["card"]
    reports = []
    for entry in card["reports"]:
        entry = dict(entry)
        if entry["role"] == "candidate":
            entry["label"] = name
        reports.append(entry)
    card["reports"] = reports
    card["text"] = {}
    card["index"] = {"path": INDEX, "sha256": pins["index_sha256"]}
    card["assets"] = {"dir": f"{PRIVATE}/{key}", "receipt_sha256": pins["assets"][tier]}
    card["speed"] = {"evidence": t["speed"].as_posix(), "sha256": sha(t["speed"])}
    spec["gate_receipt"] = f"{DECISIONS}/{name}.decision.card2.json"
    spec["_release"] = {
        "card_product": (
            "Card-only revision under the new name (user job 2026-10-02 00:05 UTC+8, release worker 4c0a68cd): "
            "the product card of v2.release.card (banner, title, one paragraph, at-a-glance table, highlights, "
            "code-only quickstart, JevArena and Jev Decision Index charts with one compact table, licence, "
            "citation). card.text is empty; card.index and card.assets pin the private Index input and the "
            "assets rendered by v2.release.card_assets; card.speed pins the 400-request bench receipt of this "
            "runtime. licence.files keeps the LICENSE only; the upstream NOTICE and the attributions are dropped "
            "(no NOTICE, ATTRIBUTIONS.md, LICENSES/ or evaluation/ in the package). Weights, tokenizer, the "
            "package runtime, the vendored sources and the Transformers remote code are byte-identical to the "
            "replaced revision; config.json and MODEL_MANIFEST.json carry the new model name."
        ),
        "renamed_from": f"{ORG}/{old_name}",
        "replaces_spec": {"spec": source.as_posix(), "sha256": sha(source)},
        "previous": old.get("_release"),
    }
    return spec


def decision_for(t: dict, spec_sha: str) -> dict:
    tier, key = t["tier"], t["key"]
    old_name, name, repo = names(tier)
    old_path = ROUND1 / f"{old_name}.decision.card.json"
    gate_path = ROUND1 / key / "release/receipts/gate.json"
    old = json.loads(old_path.read_text(encoding="utf-8"))
    old_sha = sha(old_path)
    assert old["status"] == "final", tier
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    assert gate["decision_sha256"] == old_sha, tier
    new = {k: v for k, v in old.items() if k not in DROPPED_DECISION}
    new.update(
        {
            "status": "final",
            "repo_id": repo,
            "model_name": name,
            "decided_by": DECIDED_BY,
            "prepared_by": PREPARED_BY,
            "decided_utc": DECIDED_UTC,
            "action": (
                f"Card-only revision of the private repository {repo} (renamed from {ORG}/{old_name}; the old ID "
                "redirects): README.md and the banner and four charts under assets/ are regenerated as a product "
                "card; NOTICE, ATTRIBUTIONS.md, LICENSES/, evaluation/ and the round-1 SVG charts are deleted; "
                "config.json and MODEL_MANIFEST.json carry the new name. Every model, tokenizer, runtime and "
                f"remote-code file is byte-identical to the released revision {gate['revision']}. The repository "
                f"stays in the private collection '🎲 Decision 2.0', ordered {ORDER}; everything stays private."
            ),
            "rationale": (
                f"The release judgement of the superseded final decision {old_sha[:8]}… stands unchanged: the same "
                f"identity {old['identity']['model_sha256'][:8]}, scored report, paired comparison, calibration and "
                "licence decision. Only the card, the name fields and the manifest digests change; release.sh checks "
                "the native examples, the card's Transformers example before upload, after the real download and "
                "from the Hub in fresh environments under Transformers 5.17 and 5.18, the card structure and every "
                "card link, and blocks the upload or the collection step otherwise."
            ),
            "previous_rationale": old["rationale"],
            "card_revision": {
                "kind": "card-product-rename",
                "spec": f"v2/release/specs/dev2-{key}-product.json",
                "spec_sha256": spec_sha,
            },
            "supersedes": {
                "final_sha256": old_sha,
                "released_as": f"{ORG}/{old_name}@{gate['revision']}",
                "released_manifest_sha256": gate["manifest_sha256"],
                "released_gate_sha256": sha(gate_path),
                "earlier": old.get("supersedes"),
            },
        }
    )
    return new


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pins", type=Path, required=True)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--tiers", nargs="*", default=[t["tier"] for t in TIERS])
    args = ap.parse_args()
    pins = json.loads(args.pins.read_text())
    problems = []
    for t in TIERS:
        if t["tier"] not in args.tiers:
            continue
        spec_text = json.dumps(spec_for(t, pins), ensure_ascii=False, indent=2) + "\n"
        spec_sha = hashlib.sha256(spec_text.encode()).hexdigest()
        decision_text = (
            json.dumps(decision_for(t, spec_sha), ensure_ascii=False, indent=2) + "\n"
        )
        _, name, _ = names(t["tier"])
        for path, text in (
            (SPECS / f"dev2-{t['key']}-product.json", spec_text),
            (OUT / f"{name}.decision.card2.json", decision_text),
        ):
            if args.check:
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
