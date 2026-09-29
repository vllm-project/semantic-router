"""HS1 validity check (prereg §7): score two models' ``hs1-dev`` predictions.

    python3 -m v2.data.hs1.validity --gold hs1-dev.gold.jsonl \
        --left PRED_DECIDER.jsonl --left-name decider-4b \
        --right PRED_DEV2.jsonl --right-name dev2.0-4b --output report.json

VALID per family (F1, F2): left − right accuracy has a paired cluster-bootstrap
95% CI lower bound > 0 (clusters = group_id, 10,000 resamples, seed 20260929).
Invalid or missing answers count as wrong. F3 is reported, not gated.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from benchmark.score import evaluate_answer

GATED = ("hs1_quote_check", "hs1_policy_packet")
RESAMPLES = 10_000
SEED = 20260929


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def outcomes(
    gold: Sequence[dict[str, Any]], predictions: Sequence[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    by_id = {row["id"]: row for row in predictions}
    out = {}
    for item in gold:
        question = item["questions"]["decision"]
        answer = (by_id.get(item["id"]) or {}).get("answers", {}).get("decision")
        result = evaluate_answer(question, item["gold"]["decision"], answer)
        ok = result.get("status") == "ok"
        out[item["id"]] = {
            "correct": bool(ok and result["correct"]),
            "valid": ok,
            "point": result.get("point"),
        }
    return out


def accuracy(ids: Sequence[str], res: Mapping[str, dict[str, Any]]) -> float | None:
    return round(sum(res[i]["correct"] for i in ids) / len(ids), 4) if ids else None


def diagnostics(
    items: Sequence[dict[str, Any]], res: Mapping[str, dict[str, Any]]
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    adopt = []
    false_yes = []
    for item in items:
        point = res[item["id"]]["point"]
        audit = item.get("audit", {})
        q = item["questions"]["decision"]
        if item["task"] == "hs1_quote_check" and "quoted" in audit.get(
            "option_refs", {}
        ):
            ref = audit["option_refs"]["quoted"]
            if q["type"] == "choice":
                quoted = list(q["criteria"])[ref]
            elif q["type"] == "noul":
                quoted = bool(ref)
            else:
                quoted = ref
            adopt.append(point == quoted)
        if (
            item["task"] == "hs1_unmet_condition"
            and q["type"] == "noul"
            and item["gold"]["decision"]["value"] is False
        ):
            false_yes.append(point is True)
    if adopt:
        out["f1_adopt_rate"] = round(sum(adopt) / len(adopt), 4)
    if false_yes:
        out["f3_false_yes_rate"] = round(sum(false_yes) / len(false_yes), 4)
    return out


def paired_bootstrap(items: Sequence[dict[str, Any]], left, right) -> dict[str, Any]:
    clusters: dict[str, list[float]] = collections.defaultdict(lambda: [0.0, 0.0, 0])
    for item in items:
        cell = clusters[item["cluster_id"]]
        cell[0] += left[item["id"]]["correct"]
        cell[1] += right[item["id"]]["correct"]
        cell[2] += 1
    cells = list(clusters.values())
    n = sum(c[2] for c in cells)
    delta = (sum(c[0] for c in cells) - sum(c[1] for c in cells)) / n
    rng = random.Random(SEED)
    draws = []
    for _ in range(RESAMPLES):
        sample = [cells[rng.randrange(len(cells))] for _ in cells]
        total = sum(c[2] for c in sample)
        draws.append((sum(c[0] for c in sample) - sum(c[1] for c in sample)) / total)
    draws.sort()
    low, high = draws[int(0.025 * RESAMPLES)], draws[int(0.975 * RESAMPLES) - 1]
    return {
        "delta": round(delta, 4),
        "ci95": [round(low, 4), round(high, 4)],
        "clusters": len(cells),
        "rows": n,
    }


def report(
    gold, left_preds, right_preds, left_name: str, right_name: str
) -> dict[str, Any]:
    left, right = outcomes(gold, left_preds), outcomes(gold, right_preds)
    out: dict[str, Any] = {"left": left_name, "right": right_name, "families": {}}
    fams: dict[str, list] = collections.defaultdict(list)
    for item in gold:
        fams[item["task"]].append(item)
    for family, items in sorted(fams.items()):
        ids = [i["id"] for i in items]
        cell: dict[str, Any] = {
            "rows": len(items),
            "accuracy": {
                left_name: accuracy(ids, left),
                right_name: accuracy(ids, right),
            },
            "invalid": {
                left_name: sum(not left[i]["valid"] for i in ids),
                right_name: sum(not right[i]["valid"] for i in ids),
            },
            "paired": paired_bootstrap(items, left, right),
            "by_interface": {},
            "by_slice": {},
            "diagnostics": {
                left_name: diagnostics(items, left),
                right_name: diagnostics(items, right),
            },
        }
        for field, key in (("interface", "by_interface"), ("slice", "by_slice")):
            groups: dict[str, list[str]] = collections.defaultdict(list)
            for item in items:
                groups[item[field]].append(item["id"])
            cell[key] = {
                name: {
                    "n": len(g),
                    left_name: accuracy(g, left),
                    right_name: accuracy(g, right),
                }
                for name, g in sorted(groups.items())
            }
        if family in GATED:
            cell["valid"] = cell["paired"]["ci95"][0] > 0
        out["families"][family] = cell
    out["slices_valid"] = all(
        out["families"][f].get("valid") for f in GATED if f in out["families"]
    )
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--left-name", required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--right-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    result = report(
        read_jsonl(args.gold),
        read_jsonl(args.left),
        read_jsonl(args.right),
        args.left_name,
        args.right_name,
    )
    args.output.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                f: {k: v for k, v in c.items() if k in ("accuracy", "paired", "valid")}
                for f, c in result["families"].items()
            },
            indent=1,
        )
    )
    print("slices_valid", result["slices_valid"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
