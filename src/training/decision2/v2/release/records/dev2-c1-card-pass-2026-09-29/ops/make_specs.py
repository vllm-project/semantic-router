"""Derive the six card-only specs of the C1 event-3 card pass from the current release specs.

Every spec keeps its checkpoint, identity, scored run, runtime, calibration, licence and every other
input; the card re-renders with the current renderer (its public-231 note). What changes:
  - gate_receipt: the new final decision of this pass;
  - _release.card_c1: why this revision exists;
  - runtime_source (and, for 0.8B / 2B / 4B, vendor_source): the mirror that built the revision
    this one replaces, so the package runtime and vendored sources stay byte-identical (0.6B was
    built from a tree with the current runtime and needs no pin);
  - card.text.confirmation (not 0.8B): the event-3 line of the eval record
    v2/eval/records/m4-c1-event3-prep-2026-09-29.md section 0000, verbatim, plus one plain reading
    sentence; for 0.6B the previous-revision wording;
  - 4B and 27B limitations: a C1 disclosure (the significant losses and the C1 tasks and languages
    below that peer) and the two JevBench hard-tier skills behind the significant public-231 gap
    (eval record v2/eval/records/jevbench-value-2026-09-29.md sections 4 and 6).
Run from src/training/decision2:

  python3 v2/release/records/dev2-c1-card-pass-2026-09-29/ops/make_specs.py [--check]
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

SPECS = Path("v2/release/specs")
DECISIONS = "/data/dev2/runs/release/decisions"
PLACEHOLDER = (
    "**Independent sealed confirmation (JevArena-C1): PLACEHOLDER — to be added after the "
    "final sealed scoring event.**"
)
C1_SET = (
    "JevArena-C1 v1.2 (2,840 human-labeled items from 8 sources published after the relevant "
    "cutoffs, never used for training or development; scored once)"
)
NOTE = (
    "Card-only revision after JevArena-C1 event 3 (item set v1.2, scored 2026-09-29; eval record "
    "v2/eval/records/m4-c1-event3-prep-2026-09-29.md section 0000), coordinator note 2026-09-29 "
    "23:40 UTC+8: {what}. The card re-renders with the current public-231 note of card.py. Every "
    "model file stays byte-identical to the released revision {main}.{pins}"
)
PINS = (
    " The package runtime{vendored} come from the mirror that built {main} ({mirror}), so only "
    "card files change."
)
MIRROR = "/data/dev2/src/{sha}-src_training_decision2/src/training/decision2"

CONFIRMATION = {
    "0.6B": (
        "**Independent sealed confirmation, same input limit (previous revision)** — "
        f"{C1_SET}: the previous DEV2.0-0.6B weights score **33.02** vs Decision 1.0 Kai at "
        "8,192 tokens 22.14 (+10.89, 95% CI [+8.88, +12.64]). These are the predictions sealed at "
        "event 2 from revision `e61b2b44` (model files identical to the previous revision "
        "`99c4e799`), rescored on v1.2; Kai 1.0 at its published 1,024-token limit scores 17.79 "
        "on v1.2. The current weights (released as revision `b2131337`) were not part of any "
        "sealed event."
    ),
    "2B": (
        f"**Independent sealed confirmation** — {C1_SET}: DEV2.0-2B **45.70** vs Decision 1.0 "
        "Sol (16K same-renderer control) 45.03 (+0.68, 95% CI [−0.91, +2.22]); Decider 2B 42.45 "
        "(+3.26, 95% CI [+1.32, +5.21]); This-That 1.2 42.81 (+2.90, 95% CI [+0.85, +4.99]). "
        "This-That 1.2 runs at its 1,536-token state limit and Decider 2B at its own; items they "
        "cannot answer (159 and 58 of 2,840) count as wrong. It is level with Decision 1.0 Sol "
        "and significantly above Decider 2B and This-That 1.2."
    ),
    "4B": (
        f"**Independent sealed confirmation** — {C1_SET}: DEV2.0-4B **48.38** vs Decision 1.0 "
        "Nox 49.70 (−1.32, 95% CI [−3.00, +0.35]); Decider 4B 49.65 (−1.27, 95% CI [−3.05, "
        "+0.49]); Jet v6.2 50.44 (−2.06, 95% CI [−3.86, −0.18]). Items a model cannot answer "
        "(Jet v6.2 65, Decider 4B 37 of 2,840) count as wrong. It is level with Decision 1.0 Nox "
        "and Decider 4B and significantly below Jet v6.2: the post-key JevArena v3 lead over "
        "Decision 1.0 Nox does not carry over to C1."
    ),
    "9B": (
        f"**Independent sealed confirmation** — {C1_SET}: DEV2.0-9B **53.77** vs Decision 1.0 "
        "Lux 51.97 (+1.80, 95% CI [+0.55, +3.07]); Nimble v2 52.77 (+1.00, 95% CI [−0.69, "
        "+2.75]). Measured on the frozen package `53bac735` (weights identical to this "
        "revision); items Nimble v2 cannot answer at its 8,192-token limit (58 of 2,840) count as "
        "wrong. It is significantly above Decision 1.0 Lux and level with Nimble v2."
    ),
    "27B": (
        f"**Independent sealed confirmation** — {C1_SET}: DEV2.0-27B **57.33** vs AutoJev-27B "
        "58.17 (−0.84, 95% CI [−2.50, +0.69]); Eikos-27B 59.37 (−2.04, 95% CI [−3.66, −0.44]). "
        "No Decision 1.0 model exists at this size. Measured on the release package `5683c6f0` "
        "(weights identity `b7fd44e3`); items the peers cannot answer (Eikos-27B 41, AutoJev-27B "
        "28 of 2,840) count as wrong. It is level with AutoJev-27B and significantly below "
        "Eikos-27B."
    ),
}

OLD_0P6B = (
    "**Independent sealed confirmation:** The sealed JevArena-C1 comparison measured the previous "
    "revision (`99c4e799`); it has not been repeated for this revision."
)

C1_4B = (
    "**JevArena-C1 (independent, human-labeled).** The post-key JevArena v3 lead over Decision "
    "1.0 Nox (+6.68) does not carry over to C1: −1.32 (95% CI [−3.00, +0.35]), and C1 Noul is "
    "significantly lower (−3.21, 95% CI [−6.28, −0.09]). Jet v6.2 is significantly ahead on C1 "
    "(−2.06, 95% CI [−3.86, −0.18]), mainly on Score (−6.03, 95% CI [−9.70, −2.05]), while this "
    "model is ahead of it on Noul (+4.16, 95% CI [+0.41, +7.92]); Decider 4B is level (−1.27, "
    "95% CI [−3.05, +0.49]). C1 tasks below Decision 1.0 Nox (macro-F1 × 100): is_rapport 69.5 "
    "vs 78.5, preferred_idea 50.8 vs 57.9, moment_type 67.2 vs 72.3, hallucination 68.6 vs 72.3, "
    "setting_concreteness 32.3 vs 35.6 and star_rating 34.9 vs 38.1, plus three within 0.6; "
    "languages below it (accuracy): Finnish .485 vs .618, Russian .400 vs .423 and English .508 "
    "vs .516. C1 tasks below Jet v6.2: moment_type 67.2 vs 79.2, likelihood 10.1 vs 21.0, "
    "temporal_grounding 31.2 vs 39.6, is_rapport 69.5 vs 76.6 and legal 50.6 vs 55.1, plus five "
    "within 3.2; English accuracy .508 vs .538."
)
C1_27B = (
    "**JevArena-C1 (independent, human-labeled).** C1 is level with AutoJev-27B (−0.84, 95% CI "
    "[−2.50, +0.69]), although post-key JevArena v3 is significantly lower, and significantly "
    "below Eikos-27B (−2.04, 95% CI [−3.66, −0.44]), mainly on Choice (−3.01, 95% CI [−5.35, "
    "−0.75]). C1 tasks below Eikos-27B (macro-F1 × 100): preferred_idea 44.2 vs 51.7, "
    "likelihood 25.9 vs 32.2, setting_concreteness 28.7 vs 34.9, event_causality 48.0 vs 53.8 and "
    "communicative_function 45.8 vs 48.9, plus epistemic_stance, span_is_event and moment_type "
    "within 2.5; languages below it (accuracy): English .578 vs .599 and Finnish .441 vs .515."
)
JEV_4B_OLD = (
    "JevBench public 231 is significantly below Decider 4B (171 vs 192; −21 items, 95% CI [−32, "
    "−10]; hard tier 56 vs 73) and level with Jet v6.2 (174)."
)
JEV_4B_NEW = (
    "JevBench public 231 is significantly below Decider 4B (171 vs 192; −21 items, 95% CI [−32, "
    "−10]; hard tier 56 vs 73), mostly in two hard-tier skills that neither JevArena nor this "
    "model's training covers: applying long policy documents with amendments and precedence (3 "
    "vs 10 of the 19 such items) and checking a quoted person's plausible but wrong conclusion "
    "against the evidence instead of adopting it (18 vs 30 of 46). Decision 1.0 Nox has the same "
    "weakness (5 of 19 and 22 of 46). It is level with Jet v6.2 (174)."
)
JEV_4B_NOX_OLD = "JevBench public 231 is level (171 vs 173; hard tier 56 vs 59)"
JEV_4B_NOX_NEW = "JevBench public 231 is level (171 vs 173; −2 items, 95% CI [−9, +5]; hard tier 56 vs 59)"
JEV_27B_OLD = "but significantly below Eikos-27B's 212 (−14, 95% CI [−22, −6]; hard tier 79 vs 92)."
JEV_27B_NEW = (
    "but significantly below Eikos-27B's 212 (−14, 95% CI [−22, −6]; hard tier 79 vs 92), mostly "
    "in two hard-tier skills that neither JevArena nor this model's training covers: applying "
    "long policy documents with amendments and precedence (10 vs 15 of the 19 such items) and "
    "checking a quoted person's plausible but wrong conclusion against the evidence instead of "
    "adopting it (29 vs 38 of 46)."
)

TIERS = (
    {
        "tier": "0.6B",
        "base": "dev2-0p6b-m8-release.json",
        "out": "dev2-0p6b-card-c1.json",
        "main": "b21313375ad77ddf4a8e420fa5195e6e09582043",
        "what": (
            "the confirmation line now gives the v1.2 same-limit result of the previous "
            "revision's weights (event-2 predictions of e61b2b44 vs Decision 1.0 Kai at 8,192 "
            "tokens) and states that the current weights were not part of any sealed event"
        ),
    },
    {
        "tier": "0.8B",
        "base": "dev2-0p8b-card2.json",
        "out": "dev2-0p8b-card-c1.json",
        "main": "f458c34ccfb4a5d4d32babeda1570919adb1a3c8",
        "built_from": "33de83cea695e8988e385866ee603d492f7f0e3a",
        "pin_vendor": True,
        "what": "no new C1 line (event 1 stands); only the public-231 note changes",
    },
    {
        "tier": "2B",
        "base": "dev2-2b-release.json",
        "out": "dev2-2b-card-c1.json",
        "main": "5ad3e9a3cc4865ce0360f4ecce2b345020bfdb38",
        "built_from": "33de83cea695e8988e385866ee603d492f7f0e3a",
        "pin_vendor": True,
        "what": "the C1 line replaces the placeholder",
    },
    {
        "tier": "4B",
        "base": "dev2-4b-release.json",
        "out": "dev2-4b-card-c1.json",
        "main": "452f133211de292a87bc29ab7e24a3bd0704e40d",
        "built_from": "f8f52c695fb6741ed831888cc4998edcc2613790",
        "pin_vendor": True,
        "what": (
            "the C1 line replaces the placeholder; a C1 disclosure (the v3 lead over Nox 1.0 "
            "does not carry over; Jet v6.2 significantly ahead; C1 tasks and languages below "
            "Nox 1.0 and Jet v6.2) and the two JevBench hard-tier skills behind the Decider 4B "
            "gap are added"
        ),
    },
    {
        "tier": "9B",
        "base": "dev2-9b-release.json",
        "out": "dev2-9b-card-c1.json",
        "main": "ae6831960dd1114296cb15a59248b79832c42959",
        "built_from": "2926952c17411a9220747f84c07e050bad2aa92c",
        "what": "the C1 line replaces the placeholder",
    },
    {
        "tier": "27B",
        "base": "dev2-27b-release.json",
        "out": "dev2-27b-card-c1.json",
        "main": "32e7e8b1960fa1e6af3438cd11395cac2849a8c1",
        "built_from": "2926952c17411a9220747f84c07e050bad2aa92c",
        "what": (
            "the C1 line replaces the placeholder; a C1 disclosure (level with AutoJev-27B; "
            "Eikos-27B significantly ahead; C1 tasks and languages below Eikos-27B) and the two "
            "JevBench hard-tier skills behind the Eikos-27B gap are added"
        ),
    },
)


def insert_after(d: dict, anchor: str, key: str, value) -> dict:
    assert anchor in d, anchor
    out = {}
    for k, v in d.items():
        if k != key:
            out[k] = v
        if k == anchor:
            out[key] = value
    return out


def replace_once(items: list[str], old: str, new: str) -> None:
    hits = [i for i, item in enumerate(items) if old in item]
    assert len(hits) == 1, (old, hits)
    items[hits[0]] = items[hits[0]].replace(old, new, 1)


def insert_before(items: list[str], prefix: str, new: str) -> None:
    (index,) = [i for i, item in enumerate(items) if item.startswith(prefix)]
    items.insert(index, new)


def derive(t: dict) -> dict:
    spec = copy.deepcopy(json.loads((SPECS / t["base"]).read_text(encoding="utf-8")))
    name = spec["model_name"]
    assert name == f"DEV2.0-{t['tier']}", name
    spec["gate_receipt"] = f"{DECISIONS}/{name}.decision.card-c1.json"
    pins = ""
    if t.get("built_from"):
        mirror = MIRROR.format(sha=t["built_from"])
        anchor = "vendor_source" if "vendor_source" in spec else "runtime_equivalence"
        if t.get("pin_vendor"):
            assert "vendor_source" not in spec, t["tier"]
            spec = insert_after(spec, anchor, "vendor_source", mirror)
            anchor = "vendor_source"
        spec = insert_after(spec, anchor, "runtime_source", mirror)
        pins = PINS.format(
            vendored=(
                " (decision2/*.py) and the vendored inference sources"
                if t.get("pin_vendor")
                else " (decision2/*.py)"
            ),
            main=t["main"][:8],
            mirror=t["built_from"][:9],
        )
    release = dict(spec.get("_release") or {})
    release["card_c1"] = NOTE.format(what=t["what"], main=t["main"], pins=pins)
    spec = insert_after(spec, "schema", "_release", release)
    text = spec["card"]["text"]
    if t["tier"] == "0.6B":
        assert text["confirmation"] == OLD_0P6B, text["confirmation"]
    elif t["tier"] == "0.8B":
        assert "JevArena-C1 v1.1" in text["confirmation"], text["confirmation"]
    else:
        assert text["confirmation"] == PLACEHOLDER, text["confirmation"]
    if t["tier"] in CONFIRMATION:
        text["confirmation"] = CONFIRMATION[t["tier"]]
    limitations = text["limitations"]
    if t["tier"] == "4B":
        replace_once(limitations, JEV_4B_NOX_OLD, JEV_4B_NOX_NEW)
        replace_once(limitations, JEV_4B_OLD, JEV_4B_NEW)
        insert_before(limitations, "**Score levels.**", C1_4B)
    if t["tier"] == "27B":
        replace_once(limitations, JEV_27B_OLD, JEV_27B_NEW)
        insert_before(limitations, "**JevBench public 231:**", C1_27B)
    return spec


def main() -> int:
    check = "--check" in sys.argv[1:]
    problems = []
    for t in TIERS:
        rendered = json.dumps(derive(t), ensure_ascii=False, indent=2) + "\n"
        target = SPECS / t["out"]
        if check:
            if not target.is_file() or target.read_text(encoding="utf-8") != rendered:
                problems.append(f"{t['out']}: differs from the derivation")
        else:
            target.write_text(rendered, encoding="utf-8")
            print(f"wrote {target}")
    for p in problems:
        print(p, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
