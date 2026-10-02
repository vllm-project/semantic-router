"""Specs and decisions of the banner revisions (round 4, concept A) of Decision 2.0 repositories.

User job 2026-10-02 09:44 UTC+8 (release worker 4c0a68cd): the product-card banner becomes concept A (the codename in
a blue-to-cyan gradient as the focal point, the size on its baseline, the DECISION 2.0 eyebrow, the tagline, the
small logo and the translucent V-mark), the default of v2.release.card_assets. Kai 0.6B is revised now. Eos, Sol,
Nox, Lux and Vega have releases pending under the Index-first rule and build their cards with the default
generator; one of them is revised here only if, three hours after the generator merge, it still has no newer
revision and no pending release.

Roll-out (user request 2026-10-02 10:35 UTC+8, COORDINATION 10:35; card worker fb5dd490): banner A on every card
now, as card-only revisions of Vega 27B, Lux 9B and Nox 4B, and of Eos 0.8B and Sol 2B only if their pending
releases have not started by 12:00 UTC+8. A repository whose release driver runs or is about to publish is skipped.

Each spec is the latest card spec of its tier (round 3; for Lux 9B the K-a13IB release, which already carries the
round-3 card) with card.assets re-pinned to the regenerated assets and gate_receipt = the new decision. card.index
pins the card4 copy of the round-3 private Index input where the released card used that input, and otherwise keeps
the released card's own Index input in place (Lux 9B), so no card changes its Index data or charts. Each decision
carries the superseded decision's judgement forward and changes only the action, the rationale and the supersedes
chain.

Run from src/training/decision2 (the digests come from the private inputs, given on the command line):

  python3 v2/release/records/dev2-card4-2026-10-02/ops/make_card4.py --pins PINS.json --tiers 0.6B [--check]

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
OUT = RECORDS / "dev2-card4-2026-10-02"
DECISIONS = "/data/dev2/runs/release/decisions"
PRIVATE = "/data/dev2/private/release/card4"
INDEX = f"{PRIVATE}/decision-index-card.json"
DECIDED_BY = (
    "coordinator (parent agent), Decision 2.0 program: the user's banner decision of 2026-10-02 09:44 UTC+8 "
    "(concept A: the gradient codename as the focal point), card-only revision of Kai 0.6B now; the other sizes "
    "take it through their pending releases"
)
PREPARED_BY = "Decision 2.0 release engineering, release worker 4c0a68cd (worktree vllm-sr-dev2-automap)"
DECIDED_UTC = "2026-10-02T01:44:00Z"
ROLLOUT_DECIDED_BY = (
    "coordinator (parent agent), Decision 2.0 program: the user's request of 2026-10-02 10:35 UTC+8 for banner A "
    "on every card now (COORDINATION 10:35): card-only revisions of the sizes whose releases are not running or "
    "about to publish"
)
ROLLOUT_PREPARED_BY = "Decision 2.0 release engineering, card worker fb5dd490 (worktree vllm-sr-dev2-card4-rollout)"
ROLLOUT_DECIDED_UTC = "2026-10-02T02:35:00Z"
# Kai 0.6B was revised by the banner worker; every other tier belongs to the roll-out.
BANNER_WORKER_TIERS = {"0.6B"}
# The latest published card of each tier: (spec key, codename, source).
TIERS = {
    "0.6B": ("0p6b", "Kai", "card3"),
    "0.8B": ("0p8b", "Eos", "card3"),
    "2B": ("2b", "Sol", "card3"),
    "4B": ("4b", "Nox", "card3"),
    "9B": ("9b", "Lux", "ka13ib"),
    "27B": ("27b", "Vega", "card3"),
}
# source: (spec name, records directory, decision suffix, gate receipt under the records directory)
SOURCES = {
    "card3": (
        "dev2-{key}-card3.json",
        RECORDS / "dev2-card3-2026-10-02",
        "card3",
        "{key}/release/receipts/gate.json",
    ),
    "ka13ib": (
        "dev2-{key}-ka13ib.json",
        RECORDS / "dev2-9b-ka13ib-2026-10-02",
        "ka13ib",
        "release/receipts/gate.json",
    ),
}
# Assets directory under PRIVATE where it is not the spec key: Lux 9B's assets are rendered from the K-a13IB card's
# own inputs on node A, apart from the banner worker's node-E preview set of the superseded round-2 card.
ASSETS = {"9B": "9b-ka13ib"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(tier: str) -> tuple[str, str, Path, Path, Path]:
    key, codename, source = TIERS[tier]
    spec_name, records, suffix, gate = SOURCES[source]
    name = f"Decision-2.0-{codename}-{tier}"
    return (
        key,
        name,
        SPECS / spec_name.format(key=key),
        records / f"{name}.decision.{suffix}.json",
        records / gate.format(key=key),
    )


def spec_for(tier: str, pins: dict) -> dict:
    key, name, source, _, _ = _source(tier)
    old = json.loads(source.read_text(encoding="utf-8"))
    assert old["model_name"] == name, tier
    spec = copy.deepcopy(old)
    card = spec["card"]
    in_place = old["card"]["index"]["sha256"] != pins["index_sha256"]
    if not in_place:
        card["index"] = {"path": INDEX, "sha256": pins["index_sha256"]}
    card["assets"] = {
        "dir": f"{PRIVATE}/{ASSETS.get(tier, key)}",
        "receipt_sha256": pins["assets"][tier],
    }
    spec["gate_receipt"] = f"{DECISIONS}/{name}.decision.card4.json"
    index_note = (
        "card.index keeps the released card's own private Index input in place"
        if in_place
        else "card.index pins the round-3 private Index input"
    )
    who = (
        "user banner decision 2026-10-02 09:44 UTC+8, release worker 4c0a68cd"
        if tier in BANNER_WORKER_TIERS
        else "user request 2026-10-02 10:35 UTC+8 for banner A on every card, card worker fb5dd490"
    )
    spec["_release"] = {
        "card_banner": (
            f"Card-only revision ({who}): assets/banner.png is concept A, the default banner of "
            f"v2.release.card_assets. {index_note} and card.assets the regenerated assets; the README and the "
            "charts are those of the current default generator."
        ),
        "replaces_spec": {"spec": source.as_posix(), "sha256": sha(source)},
        "previous": old.get("_release"),
    }
    return spec


def decision_for(tier: str, spec_sha: str) -> dict:
    key, name, _, old_path, gate_path = _source(tier)
    old = json.loads(old_path.read_text(encoding="utf-8"))
    old_sha = sha(old_path)
    assert old["status"] == "final", tier
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    assert gate["decision_sha256"] == old_sha and gate["model_name"] == name, tier
    repo = gate["repo_id"]
    banner_worker = tier in BANNER_WORKER_TIERS
    new = copy.deepcopy(old)
    new.update(
        {
            "decided_by": DECIDED_BY if banner_worker else ROLLOUT_DECIDED_BY,
            "prepared_by": PREPARED_BY if banner_worker else ROLLOUT_PREPARED_BY,
            "decided_utc": DECIDED_UTC if banner_worker else ROLLOUT_DECIDED_UTC,
            "action": (
                (
                    f"Card-only revision of the private repository {repo}: assets/banner.png (and, where the "
                    "superseded card predates them, the README and the Index charts of the current card generator) "
                    f"are regenerated. Every model, tokenizer, runtime and remote-code file is byte-identical to the "
                    f"released revision {gate['revision']}. The collection is not changed; everything stays private."
                )
                if banner_worker
                else (
                    f"Card-only revision of the private repository {repo}: assets/banner.png is regenerated (banner "
                    "concept A of the current card generator). Every other file, the README and every Index chart "
                    f"included, and every model, tokenizer, runtime and remote-code file is byte-identical to the "
                    f"released revision {gate['revision']}. The collection is not changed; everything stays private."
                )
            ),
            "rationale": (
                f"The release judgement of the superseded final decision {old_sha[:8]}… stands unchanged: the same "
                f"identity {old['identity']['model_sha256'][:8]}, scored report, paired comparison, calibration and "
                "licence decision. Only card files and the manifest digests change; release.sh checks the native "
                "examples, the card's Transformers example before upload, after the real download and from the "
                "Hub in fresh environments under Transformers 5.17 and 5.18, the card structure and every card "
                "link."
            ),
            "previous_rationale": old["rationale"],
            "card_revision": {
                "kind": "card-banner",
                "spec": f"v2/release/specs/dev2-{key}-card4.json",
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--pins", type=Path, required=True)
    ap.add_argument("--tiers", nargs="+", required=True, choices=sorted(TIERS))
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    pins = json.loads(args.pins.read_text())
    problems = []
    for tier in args.tiers:
        key, name, _, _, _ = _source(tier)
        spec_text = (
            json.dumps(spec_for(tier, pins), ensure_ascii=False, indent=2) + "\n"
        )
        spec_sha = hashlib.sha256(spec_text.encode()).hexdigest()
        decision_text = (
            json.dumps(decision_for(tier, spec_sha), ensure_ascii=False, indent=2)
            + "\n"
        )
        for path, text in (
            (SPECS / f"dev2-{key}-card4.json", spec_text),
            (OUT / f"{name}.decision.card4.json", decision_text),
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
