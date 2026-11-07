#!/usr/bin/env python3
"""Write a synthetic vSR decision log and labels file, for trying calibrate.py without real data.

Each app gets a hidden "true" line that differs from its default threshold, and the labels follow
that line with a little noise, so a calibration run has something to find. Like the mod, it logs
escalate, redact and shield decisions only when they acted.
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

TRUE = {"guard": 0.82, "escalate": 0.70, "redact": 0.78, "shield": 0.85, "check": 0.45}
COUNTS = {
    "guard": 65,
    "router": 35,
    "escalate": 25,
    "redact": 25,
    "shield": 20,
    "check": 30,
}
ACTED_ONLY = ("escalate", "redact", "shield")
DEFAULTS = {"guard": 0.9, "escalate": 0.8, "redact": 0.7, "shield": 0.9, "check": 0.5}
ACTS = {
    "guard": ("held", "allowed"),
    "escalate": ("escalated", "kept"),
    "redact": ("hidden", "hidden"),
    "shield": ("shielded", "allowed"),
    "check": ("unverified", "grounded"),
}
TRUE_CUTS = (0.6, 1.4)
TIERS = ("haiku", "sonnet", "opus")


def build(seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    start = datetime(2026, 10, 20, 9, tzinfo=timezone.utc)
    log, labels, n = [], [], 0

    def stamp() -> str:
        return (start + timedelta(minutes=7 * n)).isoformat().replace("+00:00", "Z")

    for app in ("guard", "escalate", "redact", "shield", "check"):
        for i in range(COUNTS[app]):
            n += 1
            low = DEFAULTS[app] if app in ACTED_ONLY else 0.0
            score = round(
                min(0.999, max(low, rng.betavariate(0.9, 0.9) * (1 - low) + low)), 3
            )
            truth = score >= TRUE[app] if rng.random() > 0.08 else score < TRUE[app]
            verdict = ACTS[app][0] if score >= DEFAULTS[app] else ACTS[app][1]
            detail: dict = {}
            if app == "guard":
                detail = {
                    "risk": "destructive" if score > 0.5 else "safe",
                    "riskProbability": score,
                    "review": score,
                    "parts": 1,
                    "outcome": (
                        ("declined" if truth and rng.random() < 0.7 else "approved")
                        if verdict == "held"
                        else "approved"
                    ),
                }
            elif app == "escalate":
                detail = {
                    "feedback": {
                        "WRONG_ANSWER": score,
                        "WANT_DIFFERENT": round(score * 0.3, 3),
                    },
                    "reask": round(rng.random() * 0.7, 3),
                }
            elif app == "redact":
                detail = {
                    "counts": {"email": 1} if i % 2 else {"name": 2},
                    "source": f"notes/{i}.txt",
                }
            elif app == "shield":
                detail = {"source": f"docs/page-{i}.md"}
            else:
                detail = {"files": 1 + i % 3, "commands": i % 2}
            log.append(
                {
                    "id": f"{app}-{i}",
                    "at": stamp(),
                    "session": f"s{i % 5}",
                    "app": app,
                    "verdict": verdict,
                    "score": score,
                    "threshold": DEFAULTS[app],
                    "subject": f"{app} case {i}",
                    "detail": detail,
                }
            )
            if abs(score - DEFAULTS[app]) < 0.25 and rng.random() < 0.55:
                labels.append(
                    {"id": f"{app}-{i}", "app": app, "label": "yes" if truth else "no"}
                )
    for i in range(COUNTS["router"]):
        n += 1
        level = round(rng.uniform(0.0, 2.0), 3)
        confidence = round(rng.uniform(0.2, 0.95), 3)
        used = 0 if level < 0.5 else (1 if level < 1.5 else 2)
        right = 0 if level < TRUE_CUTS[0] else (1 if level < TRUE_CUTS[1] else 2)
        verdict = "kept" if confidence < 0.3 else "routed"
        model = "opus" if verdict == "kept" else TIERS[used]
        score, threshold = (
            (confidence, 0.3)
            if verdict == "kept"
            else (level, 0.5 if level < 1.0 else 1.5)
        )
        log.append(
            {
                "id": f"router-{i}",
                "at": stamp(),
                "session": f"s{i % 5}",
                "app": "router",
                "verdict": verdict,
                "score": score,
                "threshold": threshold,
                "subject": f"subagent task {i}",
                "detail": {
                    "level": level,
                    "confidence": confidence,
                    "model": model,
                    "effort": "medium",
                    "escalated": None,
                },
            }
        )
        shift = right - TIERS.index(model)
        relative = "right" if shift == 0 else ("too-low" if shift > 0 else "too-high")
        if rng.random() < 0.6:
            labels.append(
                {
                    "id": f"router-{i}",
                    "app": "router",
                    "label": TIERS[right] if verdict == "kept" else relative,
                }
            )
        if verdict == "routed" and right > used and rng.random() < 0.5:
            n += 1
            log.append(
                {
                    "id": f"escalate-after-{i}",
                    "at": stamp(),
                    "session": f"s{i % 5}",
                    "app": "escalate",
                    "verdict": "escalated",
                    "score": 0.91,
                    "threshold": 0.8,
                    "subject": "No, that is wrong, redo it.",
                    "detail": {
                        "feedback": {"WRONG_ANSWER": 0.91, "WANT_DIFFERENT": 0.05},
                        "reask": 0.2,
                    },
                }
            )
    return log[:200], labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="directory for decisions.jsonl and labels.jsonl",
    )
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    log, labels = build(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    kept = {d["id"] for d in log}
    (args.out / "decisions.jsonl").write_text(
        "".join(json.dumps(d) + "\n" for d in log) + '{"id": "partial-wr'
    )
    (args.out / "labels.jsonl").write_text(
        "".join(json.dumps(x) + "\n" for x in labels if x["id"] in kept)
    )
    print(
        f"{len(log)} decisions and {sum(x['id'] in kept for x in labels)} labels in {args.out}"
    )


if __name__ == "__main__":
    main()
