"""Synthetic fixtures for the residual-failure escalation classifier (#3282).

TEST EVIDENCE ONLY. Rows are generated from a known hidden rule so the
pipeline (features -> labels -> train -> calibrate -> evaluate) can be checked
end to end. A model trained on this data must never be shipped.

Each row mirrors one ``shadowdataset.Example`` (src/semantic-router/pkg/
shadowdataset/manifest.go) plus two things that manifest does not carry yet:

* ``features``: content-minimized routing facts, each with an explicit status
* ``verdict``: a synthetic stand-in for the judging step #3280 has not landed

Usage:
    python fixtures.py --rows 5000 --seed fixture-seed-1 --out synthetic.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

FIXTURE_VERSION = "escalation-risk-fixture-v1"

# Same weights idea as shadowdataset.Policy; names match our pipeline stages.
SPLITS = [("train", 7), ("calibration", 1), ("test", 2)]

DECISIONS = ["general_chat", "code_help", "math_reasoning", "legal_qa", "summarize"]
PRIMARY_MODELS = ["small-model-a", "small-model-b"]
SHADOW_MODEL = "large-model-x"

# Missing-data states. A feature is never silently imputed.
PRESENT, ABSENT, NOT_APPLICABLE = "present", "absent", "not_applicable"

# Verdict -> label. Ties, abstentions and "both failed" (escalating would not
# have helped) are excluded, not guessed.
VERDICT_LABELS = {
    "primary_failed_shadow_ok": 1,
    "both_ok": 0,
    "primary_ok_shadow_failed": 0,
    "both_failed": None,
    "tie": None,
    "abstain": None,
}


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def split_for(seed: str, example_id: str) -> str:
    """Port of shadowdataset.Policy.splitFor so fixture splits match the Go manifest."""
    total = sum(w for _, w in SPLITS)
    digest = hashlib.sha256(f"{seed}\x00{example_id}".encode()).digest()
    point = int.from_bytes(digest[:8], "big") % total
    for name, weight in SPLITS:
        if point < weight:
            return name
        point -= weight
    return SPLITS[-1][0]


def feature(value, status=PRESENT):
    return {"value": value if status == PRESENT else None, "status": status}


def make_features(rng: random.Random) -> dict:
    decision = rng.choice(DECISIONS)
    has_tools = rng.random() < 0.25
    feats = {
        "decision": feature(decision),
        "primary_model": feature(rng.choice(PRIMARY_MODELS)),
        "complexity_score": feature(round(rng.betavariate(2, 3), 4)),
        "domain_confidence": feature(round(rng.betavariate(5, 2), 4)),
        "prompt_tokens_bucket": feature(rng.choice(["<256", "256-1k", "1k-4k", "4k+"])),
        "has_tools": feature(has_tools),
        # Only meaningful when tools are used.
        "tool_count": feature(rng.randint(1, 6)) if has_tools else feature(None, NOT_APPLICABLE),
        "context_fill_ratio": feature(round(rng.random(), 4)),
        # Optional recent-window fact: often missing (no trusted session contract).
        "recent_no_progress_turns": feature(rng.randint(0, 3)),
    }
    if rng.random() < 0.10:
        feats["context_fill_ratio"] = feature(None, ABSENT)
    if rng.random() < 0.60:
        feats["recent_no_progress_turns"] = feature(None, ABSENT)
    return feats


def hidden_risk(f: dict) -> float:
    """The ground-truth rule the classifier should rediscover. Keep it simple."""
    z = -4.0
    z += 3.0 * f["complexity_score"]["value"]
    z += 2.0 * (1 - f["domain_confidence"]["value"])
    z += {"math_reasoning": 0.9, "legal_qa": 0.7, "code_help": 0.5}.get(f["decision"]["value"], 0.0)
    z += 0.4 if f["primary_model"]["value"] == "small-model-a" else 0.0
    z += {"1k-4k": 0.3, "4k+": 0.6}.get(f["prompt_tokens_bucket"]["value"], 0.0)
    if f["has_tools"]["value"]:
        z += 0.15 * f["tool_count"]["value"]
    if f["context_fill_ratio"]["status"] == PRESENT:
        z += 1.0 * max(0.0, f["context_fill_ratio"]["value"] - 0.7)
    if f["recent_no_progress_turns"]["status"] == PRESENT:
        z += 0.5 * f["recent_no_progress_turns"]["value"]
    return 1 / (1 + math.exp(-z))


def make_verdict(rng: random.Random, risk: float) -> str:
    if rng.random() < 0.05:
        return "abstain"
    if rng.random() < 0.05:
        return "tie"
    primary_failed = rng.random() < risk
    shadow_failed = rng.random() < 0.25 * risk  # stronger model fails less
    if primary_failed and shadow_failed:
        return "both_failed"
    if primary_failed:
        return "primary_failed_shadow_ok"
    if shadow_failed:
        return "primary_ok_shadow_failed"
    return "both_ok"


def make_row(rng: random.Random, seed: str, i: int) -> dict:
    example_id = sha256_hex(f"{seed}/example/{i}")[:32]
    feats = make_features(rng)
    verdict = make_verdict(rng, hidden_risk(feats))
    return {
        "fixture_version": FIXTURE_VERSION,
        "synthetic": True,
        # --- shape of shadowdataset.Example ---
        "id": example_id,
        "input_digest": sha256_hex(f"{seed}/input/{i}"),
        "split": split_for(seed, example_id),
        "primary": {"model": feats["primary_model"]["value"], "output_digest": sha256_hex(f"{seed}/p/{i}")},
        "shadows": [{"model": SHADOW_MODEL, "output_digest": sha256_hex(f"{seed}/s/{i}")}],
        "lineage": {"replay_id": f"replay-{i:06d}", "recipe": "synthetic", "decision": feats["decision"]["value"]},
        # --- not in the manifest yet ---
        "features": feats,
        "verdict": verdict,
        "label": VERDICT_LABELS[verdict],  # None = excluded
    }


def generate(rows: int, seed: str) -> list[dict]:
    rng = random.Random(seed)  # fixed seed -> identical file every run
    return [make_row(rng, seed, i) for i in range(rows)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=int, default=5000)
    ap.add_argument("--seed", default="fixture-seed-1")
    ap.add_argument("--out", type=Path, default=Path("synthetic.jsonl"))
    args = ap.parse_args()

    data = generate(args.rows, args.seed)
    with args.out.open("w") as fh:
        for row in data:
            fh.write(json.dumps(row, sort_keys=True) + "\n")

    kept = [r for r in data if r["label"] is not None]
    print(f"wrote {len(data)} rows to {args.out}")
    print(f"labelled {len(kept)}, excluded {len(data) - len(kept)}, "
          f"positive rate {sum(r['label'] for r in kept) / max(1, len(kept)):.1%}")
    for name, _ in SPLITS:
        print(f"  {name}: {sum(r['split'] == name for r in data)}")
    print(f"file sha256: {sha256_hex(args.out.read_text())}")


if __name__ == "__main__":
    main()
