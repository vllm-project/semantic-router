"""DEV2.0-0.6B card-only revision with the approved post-key C1 line (coordinator 2026-09-30 00:30 UTC+8).

Derives specs/dev2-0p6b-card-c1pk.json from specs/dev2-0p6b-card-c1.json and the final decision from the C1
card pass's 0.6B decision (the one sealed to 188eb4c8). Only the confirmation gains the approved post-key line,
the limits gain the post-key C1 declines (eval record v2/eval/records/c1-postkey-guard-2026-09-29.md section 5),
and the gate receipt, _release note and decision bookkeeping change. Run from src/training/decision2:

  python3 v2/release/records/dev2-0p6b-c1postkey-card-2026-09-30/ops/make.py [--check]
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

SPECS = Path("v2/release/specs")
RECORDS = Path("v2/release/records")
OUT = RECORDS / "dev2-0p6b-c1postkey-card-2026-09-30"
PASS = RECORDS / "dev2-c1-card-pass-2026-09-29"
MAIN = "188eb4c822e8643034f8ca87d865a16a86f69952"
GUARD = "v2/eval/records/c1-postkey-guard-2026-09-29.md (section 5)"
POSTKEY = (
    "**JevArena-C1 v1.2, post-key (not an independent validation):** 36.92 on this revision vs 33.02 for the "
    "previous revision (+3.89, 95% CI [+2.16, +5.60]; 2,840 items, paired by source group). The independent "
    "sealed confirmation above measured the previous revision."
)
DECLINES = (
    "**Post-key C1 declines against the previous revision** (JevArena-C1 v1.2, post-key; task macro-F1 × 100, "
    "language accuracy): star_rating 21.3 vs 30.5, moment_type 53.2 vs 60.1, is_rapport 43.6 vs 49.0 and "
    "likelihood 10.2 vs 13.1; Russian .263 vs .353 and Finnish .426 vs .500."
)
NOTE = (
    "Card-only revision (coordinator 2026-09-30 00:30 UTC+8): the approved post-key C1 line under the sealed-set "
    f"line and a disclosure of the post-key C1 declines (eval record {GUARD}). Every model file stays "
    f"byte-identical to the released revision {MAIN}."
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def spec() -> dict:
    s = json.loads((SPECS / "dev2-0p6b-card-c1.json").read_text(encoding="utf-8"))
    s["gate_receipt"] = (
        "/data/dev2/runs/release/decisions/DEV2.0-0.6B.decision.card-c1pk.json"
    )
    s["_release"]["card_c1_postkey"] = NOTE
    text = s["card"]["text"]
    assert text["confirmation"].startswith(
        "**Independent sealed confirmation, same input limit"
    )
    assert "post-key (not an independent" not in text["confirmation"]
    text["confirmation"] += "\n>\n> " + POSTKEY
    limits = text["limitations"]
    (i,) = [n for n, item in enumerate(limits) if item.startswith("**Public 231:**")]
    limits.insert(i, DECLINES)
    return s


def decision(spec_path: Path) -> dict:
    old_path = PASS / "DEV2.0-0.6B.decision.json"
    gate_path = PASS / "0p6b/receipts/gate.json"
    old, gate = json.loads(old_path.read_text(encoding="utf-8")), json.loads(
        gate_path.read_text(encoding="utf-8")
    )
    assert gate["decision_sha256"] == sha(old_path) and gate["revision"] == MAIN
    new = dict(old)
    new.update(
        {
            "decided_by": (
                "coordinator (parent agent), Decision 2.0 program: the post-key C1 card line approved in the "
                "cross-track note of 2026-09-30 00:30 UTC+8, under the user's full-autonomy mandate"
            ),
            "decided_utc": "2026-09-29T16:30:00Z",
            "action": (
                "Card-only revision of the private repository llm-semantic-router/DEV2.0-0.6B: the approved post-key "
                "C1 line under the sealed-set line, and a disclosure of the post-key C1 declines against the previous "
                f"revision. Every model file stays byte-identical to the released revision {MAIN}. The repository "
                "stays in the private collection '🎲 Decision 2.0', ordered 0.6B, 0.8B, 2B, 4B, 9B, 27B; "
                "everything stays private."
            ),
            "rationale": (
                f"The release judgement of the superseded final decision {sha(old_path)[:8]}… stands unchanged (same "
                "identity, scored report, paired comparison, successor profile and evidence, calibration and licence). "
                f"The C1 post-key guard (item 8; eval record {GUARD}) measured this revision after release: 36.92 vs "
                "33.02 for the previous revision, +3.89 [+2.16, +5.60], PASS. The card now states it, labelled "
                "post-key, and discloses the task and language declines."
            ),
            "card_revision": {
                "kind": "card-only",
                "spec": f"v2/release/specs/{spec_path.name}",
                "spec_sha256": sha(spec_path),
                "c1_record": GUARD,
            },
            "c1_postkey": (
                "Item 8 after the fact: b2131337 36.92 vs e61b2b44 33.02 on C1 v1.2 (+3.89 [+2.16, +5.60], p < .001; "
                "Choice +5.16, Noul +8.05, Score -1.45 n.s.). Post-key, not an independent validation."
            ),
            "disclosures": [
                *old.get("disclosures", []),
                "post-key C1 declines vs the previous revision: star_rating 21.3 vs 30.5, moment_type 53.2 vs 60.1, "
                "is_rapport 43.6 vs 49.0, likelihood 10.2 vs 13.1; Russian .263 vs .353, Finnish .426 vs .500",
            ],
            "supersedes": {
                "final_sha256": sha(old_path),
                "released_as": f"llm-semantic-router/DEV2.0-0.6B@{MAIN}",
                "released_manifest_sha256": gate["manifest_sha256"],
                "released_gate_sha256": sha(gate_path),
                "earlier": old.get("supersedes"),
            },
        }
    )
    return new


def main() -> int:
    check = "--check" in sys.argv[1:]
    spec_path = SPECS / "dev2-0p6b-card-c1pk.json"
    outputs = [
        (spec_path, lambda: spec()),
        (OUT / "DEV2.0-0.6B.decision.json", lambda: decision(spec_path)),
    ]
    bad = []
    for path, make in outputs:
        rendered = json.dumps(make(), ensure_ascii=False, indent=2) + "\n"
        if check:
            if not path.is_file() or path.read_text(encoding="utf-8") != rendered:
                bad.append(str(path))
        else:
            path.write_text(rendered, encoding="utf-8")
        print(path, hashlib.sha256(rendered.encode()).hexdigest())
    for b in bad:
        print(f"{b}: differs from the derivation", file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
