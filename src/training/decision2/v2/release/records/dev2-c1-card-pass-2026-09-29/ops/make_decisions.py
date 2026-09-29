"""Final decisions of the C1 event-3 card pass, superseding each size's current final decision.

Each new decision copies the superseded final decision's judgement (identity, scored report,
paired comparison, gate profile and its evidence, calibration, licence, disclosures), so the
builder's gate check binds exactly the same scored model. It adds the action, the C1 treatment,
the new disclosures, the card-only spec's SHA-256 and what it supersedes; receipts/gate.json of the
release.sh --upload --collect --already-collected run binds it to the new revision. Run from
src/training/decision2 after make_specs.py:

  python3 v2/release/records/dev2-c1-card-pass-2026-09-29/ops/make_decisions.py [--check]
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

RECORDS = Path("v2/release/records")
SPECS = Path("v2/release/specs")
OUT = RECORDS / "dev2-c1-card-pass-2026-09-29"
EVENT3 = "v2/eval/records/m4-c1-event3-prep-2026-09-29.md (section 0000)"
DECIDED_BY = (
    "coordinator (parent agent), Decision 2.0 program: the C1 event-3 card pass assigned to "
    "release engineering in the cross-track note of 2026-09-29 23:40 UTC+8, under the user's "
    "full-autonomy mandate"
)
PREPARED_BY = "Decision 2.0 release engineering, C1 card-pass worker (worktree vllm-sr-dev2-release)"
ORDER = "0.6B, 0.8B, 2B, 4B, 9B, 27B"
JEVBENCH_NOTE = (
    "the public-231 note under the rank chart now reads 'public-only rerun (about a third of the "
    "official Intelligence inputs), not the official JevBench score; easy tier at ceiling; totals "
    "within about 10 items are not distinguishable' (eval decision of 2026-09-29 17:15 UTC+8)"
)
SKILLS = (
    "applying long policy documents with amendments and precedence, and checking a quoted "
    "person's plausible but wrong conclusion against the evidence (eval record "
    "v2/eval/records/jevbench-value-2026-09-29.md)"
)

TIERS = (
    {
        "tier": "0.6B",
        "old": "dev2-0p6b-m8-release-2026-09-29/DEV2.0-0.6B.m8.decision.json",
        "gate": "dev2-0p6b-m8-release-2026-09-29/release/receipts/gate.json",
        "spec": "dev2-0p6b-card-c1.json",
        "card": (
            "the confirmation line gives the v1.2 same-limit result measured on the previous "
            "revision's weights and says the current weights were not part of any sealed event; "
            + JEVBENCH_NOTE
            + " (already on the card since revision b2131337)"
        ),
        "c1": (
            "Previous-revision line: event 2 sealed the predictions of revision e61b2b44 (model "
            "files byte-identical to 99c4e799); rescored on C1 v1.2 at event 3 they score 33.02 "
            "vs Decision 1.0 Kai at 8,192 tokens 22.14 (+10.89 [+8.88, +12.64]; Kai at 1,024 "
            "tokens 17.79). The current weights (m6-mxcx-soup plus Score offsets, released as "
            "b2131337) were not part of any sealed event, and C1 is post-key after event 3, so "
            "no C1 line measures them."
        ),
        "disclosures": [],
    },
    {
        "tier": "0.8B",
        "old": "dev2-0p8b-release-2026-09-28/DEV2.0-0.8B.decision.card2.json",
        "gate": "dev2-0p8b-release-2026-09-28/card2/receipts/gate.json",
        "spec": "dev2-0p8b-card-c1.json",
        "card": JEVBENCH_NOTE + "; no other change (the event-1 C1 line stands)",
        "c1": (
            "Unchanged: the event-1 line (C1 v1.1, DEV2.0-0.8B 40.24 vs Decision 1.0 Eos 37.94, "
            "+2.31 [+0.29, +4.30]). DEV2.0-0.8B was not in event 3."
        ),
        "disclosures": [],
    },
    {
        "tier": "2B",
        "old": "dev2-2b-release-2026-09-29/DEV2.0-2B.decision.json",
        "gate": "dev2-2b-release-2026-09-29/final/receipts/gate.json",
        "spec": "dev2-2b-card-c1.json",
        "card": "the event-3 C1 line replaces the placeholder; " + JEVBENCH_NOTE,
        "c1": (
            "Event 3 (v1.2) on the released weights (5ad3e9a3, identity 073bd1f2): 45.70 vs "
            "Decision 1.0 Sol 16K 45.03 (+0.68 [-0.91, +2.22], level), Decider 2B 42.45 (+3.26 "
            "[+1.32, +5.21]) and This-That 1.2 42.81 (+2.90 [+0.85, +4.99]), both significantly "
            "above. No significant loss against a card peer, so no C1 task or language list is "
            "added."
        ),
        "disclosures": [],
    },
    {
        "tier": "4B",
        "old": "dev2-4b-release-2026-09-29/DEV2.0-4B.decision.json",
        "gate": "dev2-4b-release-2026-09-29/final/receipts/gate.json",
        "spec": "dev2-4b-card-c1.json",
        "card": (
            "the event-3 C1 line replaces the placeholder; a C1 disclosure and the JevBench "
            "hard-tier skills are added (listed in disclosures); " + JEVBENCH_NOTE
        ),
        "c1": (
            "Event 3 (v1.2) on the released weights (452f1332, identity 11b5ca1c): 48.38 vs "
            "Decision 1.0 Nox 49.70 (-1.32 [-3.00, +0.35]), Decider 4B 49.65 (-1.27 [-3.05, "
            "+0.49]) and Jet v6.2 50.44 (-2.06 [-3.86, -0.18]). The post-key v3 lead over Nox "
            "(+6.68) does not carry over to C1, and Jet v6.2 is significantly ahead."
        ),
        "disclosures": [
            "C1 (event 3, v1.2): the post-key v3 lead over Nox 1.0 does not carry over (-1.32 "
            "[-3.00, +0.35]); C1 Noul is significantly below Nox 1.0 (-3.21 [-6.28, -0.09]); Jet "
            "v6.2 is significantly ahead (-2.06 [-3.86, -0.18]; Score -6.03 [-9.70, -2.05], "
            "Noul +4.16 [+0.41, +7.92]); Decider 4B level (-1.27 [-3.05, +0.49])",
            "C1 tasks below Nox 1.0: is_rapport 69.5 vs 78.5, preferred_idea 50.8 vs 57.9, "
            "moment_type 67.2 vs 72.3, hallucination 68.6 vs 72.3, setting_concreteness 32.3 vs "
            "35.6, star_rating 34.9 vs 38.1 (three more within 0.6); languages fi .485 vs .618, "
            "ru .400 vs .423, en .508 vs .516",
            "C1 tasks below Jet v6.2: moment_type 67.2 vs 79.2, likelihood 10.1 vs 21.0, "
            "temporal_grounding 31.2 vs 39.6, is_rapport 69.5 vs 76.6, legal 50.6 vs 55.1 (five "
            "more within 3.2); en .508 vs .538",
            "JevBench public 231 vs Decider 4B -21 [-32, -10] (hard 56 vs 73), mostly two "
            "hard-tier skills: "
            + SKILLS
            + "; long_policy 3 vs 10 of 19, planted conclusions 18 "
            "vs 30 of 46 (Nox 1.0: 5 and 22); vs Nox 1.0 -2 [-9, +5] (within noise)",
        ],
    },
    {
        "tier": "9B",
        "old": "dev2-rename-9b-27b-2026-09-29/DEV2.0-9B.decision.json",
        "gate": "dev2-rename-9b-27b-2026-09-29/9b/receipts/gate.json",
        "spec": "dev2-9b-card-c1.json",
        "card": "the event-3 C1 line replaces the placeholder; " + JEVBENCH_NOTE,
        "c1": (
            "Event 3 (v1.2) on the frozen package DEV2.0-8B@53bac735 (BF16 identity b1ed5a71, "
            "weights identical to this repository's revisions): 53.77 vs Decision 1.0 Lux 16K "
            "51.97 (+1.80 [+0.55, +3.07], significantly above) and Nimble v2 52.77 (+1.00 "
            "[-0.69, +2.75], level). No significant loss against a card peer."
        ),
        "disclosures": [],
    },
    {
        "tier": "27B",
        "old": "dev2-rename-9b-27b-2026-09-29/DEV2.0-27B.decision.json",
        "gate": "dev2-rename-9b-27b-2026-09-29/27b/receipts/gate.json",
        "spec": "dev2-27b-card-c1.json",
        "card": (
            "the event-3 C1 line replaces the placeholder; a C1 disclosure and the JevBench "
            "hard-tier skills are added (listed in disclosures); " + JEVBENCH_NOTE
        ),
        "c1": (
            "Event 3 (v1.2) on the release package DEV2.0-26B@5683c6f0 (identity b7fd44e3, the "
            "same adapter and head as this repository's revisions): 57.33 vs AutoJev-27B 58.17 "
            "(-0.84 [-2.50, +0.69], level) and Eikos-27B 59.37 (-2.04 [-3.66, -0.44], "
            "significantly ahead). No Decision 1.0 model exists at this size."
        ),
        "disclosures": [
            "C1 (event 3, v1.2): level with AutoJev-27B (-0.84 [-2.50, +0.69]) although post-key "
            "v3 is significantly lower; Eikos-27B significantly ahead (-2.04 [-3.66, -0.44]), "
            "mainly on Choice (-3.01 [-5.35, -0.75])",
            "C1 tasks below Eikos-27B: preferred_idea 44.2 vs 51.7, likelihood 25.9 vs 32.2, "
            "setting_concreteness 28.7 vs 34.9, event_causality 48.0 vs 53.8, "
            "communicative_function 45.8 vs 48.9 (epistemic_stance, span_is_event, moment_type "
            "within 2.5); languages en .578 vs .599, fi .441 vs .515",
            "JevBench public 231 vs Eikos-27B -14 [-22, -6] (hard 79 vs 92), mostly two "
            "hard-tier skills: "
            + SKILLS
            + "; long_policy 10 vs 15 of 19, planted conclusions "
            "29 vs 38 of 46",
        ],
    },
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def derive(t: dict) -> dict:
    old_path = RECORDS / t["old"]
    old = json.loads(old_path.read_text(encoding="utf-8"))
    gate = json.loads((RECORDS / t["gate"]).read_text(encoding="utf-8"))
    old_sha = sha(old_path)
    assert gate["decision_sha256"] == old_sha and old["status"] == "final", t["tier"]
    spec_path = SPECS / t["spec"]
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    assert spec["repo_id"] == old["repo_id"] == gate["repo_id"], t["tier"]
    repo, revision = old["repo_id"], gate["revision"]
    new = dict(old)
    for key in ("disclosures", "c1", "release_rationale", "previous_rationale"):
        new.pop(key, None)
    new.update(
        {
            "prepared_by": PREPARED_BY,
            "decided_by": DECIDED_BY,
            "decided_utc": "2026-09-29T15:40:00Z",
            "action": (
                f"Card-only revision of the private repository {repo}: {t['card']}. Every "
                f"model file stays byte-identical to the released revision {revision}. The "
                "repository stays in the private collection '🎲 Decision 2.0', ordered "
                f"{ORDER}; everything stays private."
            ),
            "rationale": (
                f"The release judgement of the superseded final decision {old_sha[:8]}… stands "
                "unchanged: the same identity, scored report, paired comparison"
                + (", gate profile and evidence" if old.get("gate_profile") else "")
                + ", calibration and licence decision. JevArena-C1 event 3 (item set v1.2) "
                f"was scored after that decision (eval record {EVENT3}); the card now states "
                "its result and discloses every significant C1 loss against a card peer. Only "
                "the card changes, as listed in action."
            ),
            "release_rationale": old.get("release_rationale", old["rationale"]),
            **(
                {"previous_rationale": old["rationale"]}
                if "release_rationale" in old
                else {}
            ),
            "revision_binding": (
                "This decision names no revision; receipts/gate.json of the release.sh --upload "
                "--collect --already-collected run binds this file's SHA-256 to the new Hub "
                "revision and package manifest it verified."
            ),
            "card_revision": {
                "kind": "card-only",
                "spec": f"v2/release/specs/{t['spec']}",
                "spec_sha256": sha(spec_path),
                "c1_record": EVENT3,
            },
            "c1": t["c1"],
            "disclosures": [*old.get("disclosures", []), *t["disclosures"]],
            "supersedes": {
                "final_sha256": old_sha,
                "released_as": f"{repo}@{revision}",
                "released_manifest_sha256": gate["manifest_sha256"],
                "released_gate_sha256": sha(RECORDS / t["gate"]),
                "earlier": old.get("supersedes"),
            },
        }
    )
    if not new["disclosures"]:
        new.pop("disclosures")
    return new


def main() -> int:
    check = "--check" in sys.argv[1:]
    problems = []
    for t in TIERS:
        target = OUT / f"DEV2.0-{t['tier']}.decision.json"
        rendered = json.dumps(derive(t), ensure_ascii=False, indent=2) + "\n"
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
