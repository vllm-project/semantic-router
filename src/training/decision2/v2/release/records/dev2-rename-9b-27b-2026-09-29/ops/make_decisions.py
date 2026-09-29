"""Final decisions for the renamed DEV2.0-9B and DEV2.0-27B, superseding 7666fd7c... and bea9795b....

Each new decision copies the superseded final decision's judgement (identity, scored report,
paired comparison, gate evidence, calibration, licence, disclosures, C1 treatment) and changes
only the name, the action and the rationale; gate.json of the card-only run binds it to the new
revision. Run from src/training/decision2:

  python3 v2/release/records/dev2-rename-9b-27b-2026-09-29/ops/make_decisions.py [--check]
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

RECORDS = Path("v2/release/records")
OUT = RECORDS / "dev2-rename-9b-27b-2026-09-29"
DECIDED_BY = (
    "coordinator (parent agent), Decision 2.0 program: the user's naming directive of "
    "2026-09-29 16:05 UTC+8, assigned to release engineering at 16:35"
)
PREPARED_BY = (
    "Decision 2.0 release engineering, rename worker (DEV2.0-8B -> DEV2.0-9B, "
    "DEV2.0-26B -> DEV2.0-27B)"
)
DIRECTIVE = (
    "the user's naming directive of 2026-09-29 16:05 UTC+8 (optimization round 1, item 1: names "
    "follow the base model's size, not the loaded parameter count; cards still state the loaded "
    "count)"
)
ORDER = "0.6B, 0.8B, 2B, 4B, 9B, 27B"
PAIRS = (
    {
        "old_path": "dev2-8b-release-2026-09-29/DEV2.0-8B.decision.json",
        "gate": "dev2-8b-release-2026-09-29/final/receipts/gate.json",
        "old": "DEV2.0-8B",
        "new": "DEV2.0-9B",
        "base": "Qwen/Qwen3.5-9B",
        "loaded": 7_940_895_744,
        "moved_utc": "2026-09-29T08:44:34Z",
        "banner": "9B",
        "extra_card": "",
        "extra_disclosure": None,
    },
    {
        "old_path": "dev2-26b-release-2026-09-29/DEV2.0-26B.decision.json",
        "gate": "dev2-26b-release-2026-09-29/final/receipts/gate.json",
        "old": "DEV2.0-26B",
        "new": "DEV2.0-27B",
        "base": "Qwen/Qwen3.8-27B",
        "loaded": 25_746_591_744,
        "moved_utc": "2026-09-29T08:44:35Z",
        "banner": "27B",
        "extra_card": (
            " and the three 27B peers' mlx-diag Choice / Noul lines (eval record "
            "v2/eval/records/m5-mlx-diag-27b-peers-2026-09-29.md; the XNLI-based Score part stays "
            "off the card)"
        ),
        "extra_disclosure": (
            "mlx-diag vs the three 27B peers (Choice / Noul parts only; no paired test): non-English "
            "Choice 79.1% vs AutoJev-27B 80.6%, Eikos-27B 78.4%, Jebadiah-27B 77.6%; non-English Noul "
            "85.5% vs 88.7%, 85.0%, 85.3%; Korean Noul 79% vs 85%, 84%, 80%; Arabic Choice 77.8% vs "
            "78.6%, 75.4%, 75.4% (eval record m5-mlx-diag-27b-peers-2026-09-29.md)"
        ),
    },
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def derive(p: dict) -> dict:
    old_path = RECORDS / p["old_path"]
    old = json.loads(old_path.read_text(encoding="utf-8"))
    gate = json.loads((RECORDS / p["gate"]).read_text(encoding="utf-8"))
    old_sha = sha(old_path)
    assert gate["decision_sha256"] == old_sha and old["status"] == "final", p["old"]
    repo = f"llm-semantic-router/{p['new']}"
    loaded = f"{p['loaded']:,}"
    base_label = p["base"].rsplit("/", 1)[-1]
    card_change = (
        f"the new name everywhere (title, examples, links, licence component, chart labels), the "
        f"{p['banner']} owl banner{', ' if p['extra_card'] else ' and '}the sentence 'Named after "
        f"its base model ({base_label}); it loads {loaded} parameters'{p['extra_card']}"
    )
    new = {}
    for key, value in old.items():
        new[key] = value
        if key == "repo_id":
            new["name_basis"] = "base"
            new["name_base_model"] = p["base"]
    new.update(
        {
            "model_name": p["new"],
            "repo_id": repo,
            "prepared_by": PREPARED_BY,
            "decided_by": DECIDED_BY,
            "decided_utc": "2026-09-29T08:35:00Z",
            "action": (
                f"Card-only revision of the private repository {repo} (moved from "
                f"llm-semantic-router/{p['old']} with move_repo at {p['moved_utc']}): {card_change}. "
                f"Every model file stays byte-identical to the released revision {gate['revision']}. "
                f"The repository stays in the private Decision 2.0 collection, ordered {ORDER}; "
                "everything stays private."
            ),
            "rationale": (
                f"Under {DIRECTIVE}, {p['old']} becomes {p['new']} after its base model {p['base']} "
                f"and still states its {loaded} loaded parameters. The release judgement of the superseded "
                f"final decision {old_sha[:8]}… stands unchanged: the same identity, scored report, "
                "paired comparison, calibration, licence decision, disclosures and the pending C1 "
                "line. Only the name and the card change as listed in action."
            ),
            "release_rationale": old["rationale"],
            "revision_binding": (
                "This decision names no revision; receipts/gate.json of the release.sh --upload "
                "--collect --already-collected run binds this file's SHA-256 to the new Hub revision "
                "and package manifest it verified."
            ),
            "supersedes": {
                "final_sha256": old_sha,
                "released_as": f"llm-semantic-router/{p['old']}@{gate['revision']}",
                "released_manifest_sha256": gate["manifest_sha256"],
                "released_gate_sha256": sha(RECORDS / p["gate"]),
                "earlier": old["supersedes"],
            },
            "rename": {
                "directive": DIRECTIVE,
                "from": f"llm-semantic-router/{p['old']}",
                "to": repo,
                "moved_utc": p["moved_utc"],
                "move_receipt": "v2/release/records/dev2-rename-9b-27b-2026-09-29/receipts/move.json",
                "loaded_parameters": p["loaded"],
            },
        }
    )
    if isinstance(new.get("approved_package"), dict):
        new["approved_package"] = {
            **new["approved_package"],
            "name_basis": "base",
            "name_base_model": p["base"],
            "revision_binding": (
                "gate.json binds this decision to the uploaded revision and manifest of the "
                "--collect --already-collected run"
            ),
        }
    if p["extra_disclosure"]:
        new["disclosures"] = [*old["disclosures"], p["extra_disclosure"]]
    return new


def main() -> int:
    check = "--check" in sys.argv[1:]
    problems = []
    for p in PAIRS:
        target = OUT / f"{p['new']}.decision.json"
        rendered = json.dumps(derive(p), ensure_ascii=False, indent=2) + "\n"
        if check:
            if not target.is_file() or target.read_text(encoding="utf-8") != rendered:
                problems.append(f"{target}: differs from the derivation")
        else:
            target.write_text(rendered, encoding="utf-8")
        print(target, hashlib.sha256(rendered.encode()).hexdigest())
    for problem in problems:
        print(problem, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
