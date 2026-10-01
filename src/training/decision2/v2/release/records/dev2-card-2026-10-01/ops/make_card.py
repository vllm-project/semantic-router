"""Specs and decisions of the card-redesign revisions of the six DEV2.0 repositories.

User job 2026-10-01 21:58 UTC+8 (release worker 4c0a68cd): the cards become standard model-release cards (title and
one paragraph, highlights, model overview, one evaluation chart pair and table, a Transformers-only quickstart, short
limitations, training data with the licence attributions, licence, citation); methods, per-task results and every
result below the counterpart move to evaluation/EVALUATION.md, and internal release facts (gate items, decision and
weights hashes, runtime notes, post-key and C1 detail) stay in the release records. Card-only: weights, tokenizer,
configs, runtime and the Transformers remote code stay byte-identical to the released revisions (runtime_source,
vendor_source and automap_source keep the mirrors that built them).

Each spec is the previous final spec with: card.text replaced by the short texts below; card.banner,
card.calibration_text and card.requirements_text removed; the licence attributions cleaned of the internal panel
version and the banner credit; gate_receipt = the new decision. Each decision carries the superseded final decision's
judgement forward and changes only the action, the rationale and the supersedes chain.

Run from src/training/decision2:

  python3 v2/release/records/dev2-card-2026-10-01/ops/make_card.py [--check]
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

RECORDS = Path("v2/release/records")
SPECS = Path("v2/release/specs")
OUT = RECORDS / "dev2-card-2026-10-01"
DECISIONS = "/data/dev2/runs/release/decisions"
ORDER = "0.6B, 0.8B, 2B, 4B, 9B, 27B"
DECIDED_BY = (
    "coordinator (parent agent), Decision 2.0 program: the user's card job of 2026-10-01 21:58 UTC+8 (make the "
    "DEV2.0 cards standard, formal model-release cards with a Transformers-only quickstart; keep internal records "
    "off the cards), card-only revisions of all six repositories"
)
PREPARED_BY = "Decision 2.0 release engineering, release worker 4c0a68cd (worktree vllm-sr-dev2-automap)"
DECIDED_UTC = "2026-10-01T13:58:00Z"
C1 = (
    "**JevArena-C1** is an independent confirmation set of 2,840 human-labelled items from eight sources published "
    "after the relevant cutoffs, never used for training or development and scored once"
)
QWEN35_TYPE = (
    "Decision model: a {layers}-layer Qwen3.5 text backbone (gated-delta and attention layers) with a "
    "candidate-scoring head; returns probabilities, does not generate text"
)
SHARED_TRAINING = (
    "No outputs of Jev or of any third-party decision model were used, and the training data were screened for "
    "overlap with the evaluation panels."
)
TIERS = (
    {
        "tier": "0.6B",
        "key": "0p6b",
        "spec": "dev2-0p6b-automap.json",
        "decision": RECORDS
        / "dev2-automap-2026-10-01/DEV2.0-0.6B.decision.automap.json",
        "gate": RECORDS / "dev2-automap-2026-10-01/0p6b/release/receipts/gate.json",
        "text": {
            "model_type": (
                "Decision model: a 28-layer Qwen3 text backbone with a candidate-scoring head; returns "
                "probabilities, does not generate text"
            ),
            "precision": "BF16 backbone compute and FP32 decision head on GPU; FP32 on CPU",
            "limitations": [
                "**Score levels:** five-level Score answers add fixed per-level offsets fitted after training "
                "(`score_bias.json`); on the typed Score family accuracy stays close to always choosing the most "
                "common level.",
            ],
            "training_summary": (
                "Fine-tuned from Qwen3-0.6B-Base on 232,754 decision rows (Choice, Noul and Score) in English, "
                "Chinese and 30 other languages: program-generated tasks labelled by program oracles, and human "
                "labels from 37 public datasets. Soft targets came from our own Decision 1.0 Lux-9B. "
                + SHARED_TRAINING
            ),
            "c1_result": (
                f"{C1}. It measured the previous DEV2.0-0.6B weights at 33.02 against 22.14 for Decision 1.0 Kai at "
                "the same input limit; the current weights were not part of a sealed event."
            ),
        },
    },
    {
        "tier": "0.8B",
        "key": "0p8b",
        "spec": "dev2-0p8b-budget.json",
        "decision": RECORDS / "dev2-budget-2026-10-01/DEV2.0-0.8B.decision.budget.json",
        "gate": RECORDS / "dev2-budget-2026-10-01/0p8b/release/receipts/gate.json",
        "text": {
            "model_type": QWEN35_TYPE.format(layers=24),
            "base_model": (
                "[Decision-1.0-Eos-0.8B](https://huggingface.co/llm-semantic-router/Decision-1.0-Eos-0.8B) "
                "(full fine-tune; [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) backbone)"
            ),
            "limitations": [
                "**Seeds:** the release is the uniform weight average of three fine-tunes whose single-seed JevArena "
                "scores ranged from 41.1 to 49.8; a retrained seed of the same recipe can land well below it.",
            ],
            "training_summary": (
                "Full fine-tune of Decision 1.0 Eos on 162,777 decision rows: our Decision 1.0 corpora "
                "(program-generated tasks labelled by program oracles, plus public human-labelled datasets) and "
                "Decision 2.0 data (verifiable rules, counterfactuals, hard negatives, multi-level Score tasks and "
                "public human labels in five languages). Hard labels only, with no teacher; the release is the "
                "uniform weight average of three seeds. " + SHARED_TRAINING
            ),
            "c1_result": (
                f"{C1} (its v1.1 edition had 2,874 items). DEV2.0-0.8B scored 40.24 against 37.94 for Decision 1.0 "
                "Eos (+2.31, 95% CI +0.29 to +4.30)."
            ),
        },
    },
    {
        "tier": "2B",
        "key": "2b",
        "spec": "dev2-2b-budget.json",
        "decision": RECORDS / "dev2-budget-2026-10-01/DEV2.0-2B.decision.budget.json",
        "gate": RECORDS / "dev2-budget-2026-10-01/2b/release/receipts/gate.json",
        "text": {
            "model_type": QWEN35_TYPE.format(layers=24),
            "base_model": (
                "[Decision-1.0-Sol-2B](https://huggingface.co/llm-semantic-router/Decision-1.0-Sol-2B) "
                "(full fine-tune; [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) backbone)"
            ),
            "limitations": [
                "**Score levels:** it almost never predicts the lowest of five typed Score levels (6 of 400 "
                "predictions), which Decision 1.0 Sol does predict.",
            ],
            "training_summary": (
                "Full fine-tune of Decision 1.0 Sol on 56,198 decision rows in 20 languages: a replay of our "
                "Decision 1.0 corpora and Decision 2.0 data (verifiable rules, counterfactuals, hard negatives, "
                "multi-level Score tasks and human labels from public reading-comprehension, commonsense, dialogue, "
                "classification and rating datasets). Decision 1.0 Sol's own probabilities were the only soft "
                "targets; the release is the uniform weight average of three seeds. "
                + SHARED_TRAINING
            ),
            "c1_result": (
                f"{C1}. DEV2.0-2B scored 45.70: level with Decision 1.0 Sol at 45.03 (+0.68, 95% CI −0.91 to +2.22) "
                "and significantly above Decider 2B (42.45) and This-That 1.2 (42.81)."
            ),
        },
    },
    {
        "tier": "4B",
        "key": "4b",
        "spec": "dev2-4b-m10lh.json",
        "decision": RECORDS / "dev2-4b-m10lh-2026-10-01/DEV2.0-4B.decision.m10lh.json",
        "gate": RECORDS / "dev2-4b-m10lh-2026-10-01/release/receipts/gate.json",
        "text": {
            "model_type": QWEN35_TYPE.format(layers=32),
            "base_model": (
                "[Qwen3.5-4B-Base](https://huggingface.co/Qwen/Qwen3.5-4B-Base) (rank-128 LoRA merged into the "
                "weights)"
            ),
            "limitations": [
                "**Typed decisions:** typed Choice accuracy is below Decider 4B and Jet v6.2 (87.8% vs 93.8% and "
                "90.0%), and the lowest typed Score level is rarely predicted (7 of 400 predictions).",
            ],
            "training_summary": (
                "A rank-128 LoRA on Qwen3.5-4B-Base, merged into the weights, with a new candidate head, trained on "
                "58,739 decision rows in about 25 languages: our Decision 1.0 corpora and Decision 2.0 data "
                "(verifiable rules, counterfactuals, hard negatives, multi-level Score tasks and human labels from "
                "public reading-comprehension, retrieval, commonsense, dialogue, classification and rating "
                "datasets). Soft targets came from our own Decision 1.0 Lux-9B; the release is the uniform weight "
                "average of three seeds. " + SHARED_TRAINING
            ),
            "c1_result": (
                f"{C1}. It measured the previous DEV2.0-4B weights only (48.38; Decision 1.0 Nox 49.70, Decider 4B "
                "49.65, Jet v6.2 50.44) and has not been repeated for these weights."
            ),
        },
    },
    {
        "tier": "9B",
        "key": "9b",
        "spec": "dev2-9b-budget.json",
        "decision": RECORDS / "dev2-budget-2026-10-01/DEV2.0-9B.decision.budget.json",
        "gate": RECORDS / "dev2-budget-2026-10-01/9b/release/receipts/gate.json",
        "text": {
            "model_type": QWEN35_TYPE.format(layers=32),
            "base_model": (
                "[Decision-1.0-Lux-9B](https://huggingface.co/llm-semantic-router/Decision-1.0-Lux-9B) "
                "(fine-tuned and interpolated; [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) backbone)"
            ),
            "limitations": [
                "**Weight origin:** the weights are one third of a three-seed fine-tune average plus two thirds of "
                "Decision 1.0 Lux; each fine-tune alone scored below Decision 1.0 Lux on the development panels.",
            ],
            "training_summary": (
                "Full fine-tune of Decision 1.0 Lux on 122,651 decision rows in 35 languages: our Decision 1.0 "
                "corpora and Decision 2.0 data (verifiable rules, counterfactuals, hard negatives, multi-level Score "
                "tasks and human labels from public reading-comprehension, retrieval, commonsense, dialogue, "
                "classification and rating datasets). Decision 1.0 Lux's own probabilities were the only soft "
                "targets; the three seeds were averaged and one third of that average was interpolated into "
                "Decision 1.0 Lux. " + SHARED_TRAINING
            ),
            "c1_result": (
                f"{C1}. DEV2.0-9B scored 53.77 against 51.97 for Decision 1.0 Lux (+1.80, 95% CI +0.55 to +3.07) "
                "and 52.77 for Nimble v2 (level)."
            ),
        },
    },
    {
        "tier": "27B",
        "key": "27b",
        "spec": "dev2-27b-budget.json",
        "decision": RECORDS / "dev2-budget-2026-10-01/DEV2.0-27B.decision.budget.json",
        "gate": RECORDS / "dev2-budget-2026-10-01/27b/release/receipts/gate.json",
        "text": {
            "model_type": (
                "Decision model: a rank-64 LoRA adapter on the frozen 64-layer Qwen3.8-27B text backbone with a "
                "candidate-scoring head; returns probabilities, does not generate text"
            ),
            "base_model": (
                "[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (frozen; not included, downloaded at load "
                "time)"
            ),
            "limitations": [
                "**JevBench public 231:** 203 correct, significantly below Eikos-27B's 212, mostly on hard-tier items "
                "that apply long policy documents or check a quoted conclusion against the evidence.",
                "**Memory:** inference needs one GPU with at least 122 GB of memory at the 32,768-token limit.",
            ],
            "training_summary": (
                "A LoRA adapter on the frozen Qwen3.8-27B base with a new candidate head, trained on 56,969 decision "
                "rows (25.0M tokens): program-generated decision tasks and human labels from public datasets, "
                "multi-level Score tasks in English and Chinese with graded human ratings in four languages, and our "
                "Decision 1.0 typed curricula. Gold labels only, with no teacher; the release is the exact uniform "
                "average of two seeds' adapters. " + SHARED_TRAINING
            ),
            "c1_result": (
                f"{C1}. It measured the previous DEV2.0-27B adapter only (57.33; AutoJev-27B 58.17, Eikos-27B 59.37) "
                "and has not been repeated for this revision."
            ),
            "transformers_note": "The pinned base is about 52 GB; `base_path=` takes a local copy of that revision instead.",
        },
    },
)
DROPPED_CARD = ("banner", "calibration_text", "requirements_text")
KEPT_TEXT = ("comparator_note",)
DROPPED_DECISION = ("previous_rationale", "runtime_revision", "card_revision")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean_attribution(entry: str) -> str | None:
    if entry.startswith("Owl banner"):
        return None
    entry = entry.replace("Evaluation data shown on the card:", "Evaluation data:")
    return entry.replace("JevArena v3", "JevArena")


def clean_note(note: str) -> str:
    return note.replace("JevArena v3", "JevArena")


def spec_for(t: dict) -> dict:
    source = SPECS / t["spec"]
    old = json.loads(source.read_text(encoding="utf-8"))
    spec = copy.deepcopy(old)
    card = spec["card"]
    for key in DROPPED_CARD:
        card.pop(key, None)
    text = dict(t["text"])
    for key in KEPT_TEXT:
        if old["card"]["text"].get(key):
            text[key] = clean_note(old["card"]["text"][key])
    card["text"] = text
    lic = spec["licence"]
    lic["attributions"] = [
        a for a in (clean_attribution(e) for e in lic.get("attributions", [])) if a
    ]
    spec["gate_receipt"] = f"{DECISIONS}/DEV2.0-{t['tier']}.decision.card.json"
    spec["_release"] = {
        "card_redesign": (
            "Card-only revision (user card job 2026-10-01 21:58 UTC+8, release worker 4c0a68cd): a standard "
            "model-release card from v2.release.card (highlights, model overview, one evaluation chart pair and "
            "table, a Transformers-only quickstart, at most five limitations, training data with the licence "
            "attributions, licence, citation); methods and every result below the counterpart in "
            "evaluation/EVALUATION.md; internal release facts in the records. card.text is the short card text; "
            "banner, calibration_text and requirements_text are dropped; the attributions lose the internal panel "
            "version and the banner credit. Weights, tokenizer, configs, the package runtime, the vendored sources "
            "and the Transformers remote code are byte-identical to the replaced revision (same runtime_source, "
            "vendor_source and automap_source)."
        ),
        "replaces_spec": {
            "spec": f"v2/release/specs/{source.name}",
            "sha256": sha(source),
        },
        "previous": old.get("_release"),
    }
    return spec


def decision_for(t: dict, spec_sha: str) -> dict:
    old_path, gate_path = t["decision"], t["gate"]
    old = json.loads(old_path.read_text(encoding="utf-8"))
    old_sha = sha(old_path)
    assert old["status"] == "final", t["tier"]
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    assert gate["decision_sha256"] == old_sha, t["tier"]
    repo = old["repo_id"]
    new = {k: v for k, v in old.items() if k not in DROPPED_DECISION}
    new.update(
        {
            "status": "final",
            "decided_by": DECIDED_BY,
            "prepared_by": PREPARED_BY,
            "decided_utc": DECIDED_UTC,
            "action": (
                f"Card-only revision of the private repository {repo}: README.md, the two charts under assets/, "
                "evaluation/EVALUATION.md and manifest.json, NOTICE and ATTRIBUTIONS.md are regenerated as a "
                "standard model-release card with a Transformers-only quickstart (internal release facts stay in "
                "the records). Every model, runtime and remote-code file is byte-identical to the released "
                f"revision {gate['revision']}. The repository stays in the private collection '🎲 Decision 2.0', "
                f"ordered {ORDER}; everything stays private."
            ),
            "rationale": (
                f"The release judgement of the superseded final decision {old_sha[:8]}… stands unchanged: the same "
                f"identity {old['identity']['model_sha256'][:8]}, scored report, paired comparison, calibration and "
                "licence decision. Only card files and the manifest digests change; release.sh checks the native "
                "examples, the card's Transformers example before upload, after the real download and from the Hub "
                "in fresh environments under Transformers 5.17 and 5.18, the card structure and every card link, "
                "and blocks the upload or the collection step otherwise."
            ),
            "previous_rationale": old["rationale"],
            "card_revision": {
                "kind": "card-redesign",
                "spec": f"v2/release/specs/dev2-{t['key']}-card.json",
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
            (SPECS / f"dev2-{t['key']}-card.json", spec_text),
            (OUT / f"DEV2.0-{t['tier']}.decision.card.json", decision_text),
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
