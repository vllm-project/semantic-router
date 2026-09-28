"""Diagnose a constant typed-DEV Noul answer (aggregates only).

For each readout: the typed-DEV Noul answer distribution, accuracy, ranking
signal (AUC of p(true) against gold), counterfactual-pair movement, the best
accuracy any single threshold could reach on DEV (an oracle bound, never a
fix), and, given the checkpoint's CAL probabilities, a Noul logit bias and a
threshold fitted on CAL alone and applied to DEV. Also the typed-DEV Noul
panel structure and, per TRAIN arm, Noul class balance and prompt forms.
No item text, id or per-item answer is written.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import re
from pathlib import Path
from typing import Any

from .common import read_jsonl, write_json


def auc(scores: list[float], labels: list[bool]) -> float | None:
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    wins = 0.0
    for a in pos:
        for b in neg:
            wins += 1.0 if a > b else 0.5 if a == b else 0.0
    return wins / (len(pos) * len(neg))


def logit(p: float) -> float:
    p = min(max(p, 1e-7), 1 - 1e-7)
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))


def fit_bias(scores: list[float], labels: list[bool]) -> float:
    """Logit bias minimizing binary NLL (Newton steps on a convex 1-D objective)."""
    logits = [logit(p) for p in scores]
    b = 0.0
    for _ in range(100):
        grad = sum(sigmoid(z + b) - y for z, y in zip(logits, labels))
        hess = sum(sigmoid(z + b) * (1 - sigmoid(z + b)) for z in logits)
        if hess <= 1e-12:
            break
        step = grad / hess
        b -= max(-2.0, min(2.0, step))
        if abs(step) < 1e-10:
            break
    return b


def fit_threshold(scores: list[float], labels: list[bool]) -> float:
    """Threshold t (predict true iff p > t) maximizing accuracy; ties -> closest to 0.5."""
    candidates = sorted(set(scores)) + [0.5]
    best = max(
        candidates,
        key=lambda t: (
            sum((s > t) == y for s, y in zip(scores, labels)),
            -abs(t - 0.5),
        ),
    )
    return best


def dev_noul(
    gold_path: Path, prompts_path: Path
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    prompts = {row["id"]: row for row in read_jsonl(prompts_path)}
    questions: dict[tuple[str, str], dict[str, Any]] = {}
    families = collections.Counter()
    templates = set()
    shapes = collections.Counter()
    criteria = collections.Counter()
    group_true: collections.Counter[str] = collections.Counter()
    relations = collections.Counter()
    for item in read_jsonl(gold_path):
        for key, question in item["questions"].items():
            if question["type"] != "noul":
                continue
            prompt = prompts[item["id"]]
            text = prompt["questions"][key]["instructions"]
            templates.add(
                re.sub(
                    r"\d+",
                    "#",
                    text if isinstance(text, str) else json.dumps(text, sort_keys=True),
                )
            )
            state = prompt["state"]
            shapes[
                (
                    ",".join(sorted(state))
                    if isinstance(state, dict)
                    else type(state).__name__
                )
            ] += 1
            given = prompt["questions"][key].get("criteria")
            criteria[
                "custom" if isinstance(given, dict) and len(given) == 2 else "default"
            ] += 1
            value = bool(item["gold"][key]["value"])
            families[item["family"]] += 1
            group_true[item["group_id"]] += value
            pairs = [(p["id"], p["relation"], p["role"]) for p in item["pairs"]]
            for _, relation, role in pairs:
                relations[f"{relation}/{role}"] += 1
            questions[(item["id"], key)] = {
                "family": item["family"],
                "gold": value,
                "group": item["group_id"],
                "pairs": pairs,
            }
    summary = {
        "questions": len(questions),
        "families": dict(families),
        "gold_true": sum(q["gold"] for q in questions.values()),
        "instruction_templates_digits_masked": len(templates),
        "state_top_level_keys": dict(shapes),
        "criteria": dict(criteria),
        "groups": len(group_true),
        "gold_true_per_group_histogram": dict(collections.Counter(group_true.values())),
        "pair_memberships": dict(relations),
    }
    return questions, summary


def readout_summary(
    questions: dict[tuple[str, str], dict[str, Any]],
    predictions: Path,
    cal: tuple[list[float], list[bool]] | None,
) -> dict[str, Any]:
    scores: dict[tuple[str, str], float] = {}
    for row in read_jsonl(predictions):
        for key, answer in row["answers"].items():
            if (row["id"], key) in questions:
                scores[(row["id"], key)] = float(answer["noul"])
    if set(scores) != set(questions):
        raise ValueError(f"{predictions}: Noul answers do not cover typed DEV")
    order = sorted(questions)
    p = [scores[k] for k in order]
    y = [questions[k]["gold"] for k in order]
    by_pair: dict[str, list[tuple[str, float, bool]]] = collections.defaultdict(list)
    for k in order:
        for pair_id, relation, role in questions[k]["pairs"]:
            by_pair[pair_id].append((relation, scores[k], questions[k]["gold"]))
    moved = right = 0
    for members in by_pair.values():
        if len(members) == 2 and members[0][0] == "counterfactual":
            (_, pa, ya), (_, pb, yb) = members
            if pa != pb:
                moved += 1
                right += (pa > pb) == (ya and not yb) if ya != yb else 0
    out: dict[str, Any] = {
        "n": len(p),
        "answered_true": sum(v > 0.5 for v in p),
        "correct": sum((v > 0.5) == g for v, g in zip(p, y)),
        "p_true_mean": sum(p) / len(p),
        "p_true_sd": (sum((v - sum(p) / len(p)) ** 2 for v in p) / len(p)) ** 0.5,
        "p_true_min": min(p),
        "p_true_max": max(p),
        "auc": auc(p, y),
        "counterfactual_pairs_moved": moved,
        "counterfactual_pairs_moved_toward_gold": right,
        "oracle_threshold_correct_diagnostic_only": max(
            sum((v > t) == g for v, g in zip(p, y)) for t in sorted(set(p)) + [-1.0]
        ),
    }
    if cal is not None:
        cal_p, cal_y = cal
        bias = fit_bias(cal_p, cal_y)
        threshold = fit_threshold(cal_p, cal_y)
        out["cal"] = {
            "noul_rows": len(cal_p),
            "correct_raw": sum((v > 0.5) == g for v, g in zip(cal_p, cal_y)),
            "fitted_logit_bias": bias,
            "correct_with_bias": sum(
                (sigmoid(logit(v) + bias) > 0.5) == g for v, g in zip(cal_p, cal_y)
            ),
            "fitted_threshold": threshold,
        }
        out["dev_with_cal_bias"] = {
            "answered_true": sum(sigmoid(logit(v) + bias) > 0.5 for v in p),
            "correct": sum((sigmoid(logit(v) + bias) > 0.5) == g for v, g in zip(p, y)),
        }
        out["dev_with_cal_threshold"] = {
            "answered_true": sum(v > threshold for v in p),
            "correct": sum((v > threshold) == g for v, g in zip(p, y)),
        }
    return out


def cal_noul(cal_rows: Path, cal_probs: Path) -> tuple[list[float], list[bool]]:
    rows = {row["id"]: row for row in read_jsonl(cal_rows)}
    scores, labels = [], []
    for entry in read_jsonl(cal_probs):
        row = rows[entry["id"]]
        if row["task_type"] != "noul":
            continue
        keys = [option["key"] for option in row["options"]]
        scores.append(float(entry["probabilities"][keys.index("true")]))
        labels.append(keys[row["label"]] == "true")
    return scores, labels


def train_summary(path: Path) -> dict[str, Any]:
    noul = true = generic = facts_rules = 0
    shapes = collections.Counter()
    families = collections.Counter()
    for row in read_jsonl(path):
        if row["task_type"] != "noul":
            continue
        keys = [option["key"] for option in row["options"]]
        noul += 1
        true += keys[row["label"]] == "true"
        families[row["family"]] += 1
        descriptions = [option["description"] for option in row["options"]]
        generic += all(
            isinstance(d, str) and d.strip().lower() in ("yes", "no", "true", "false")
            for d in descriptions
        )
        state = row["state"]
        if isinstance(state, str):
            try:
                parsed = json.loads(state)
                state = parsed if isinstance(parsed, dict) else state
            except ValueError:
                pass
        shape = ",".join(sorted(state)) if isinstance(state, dict) else "text"
        shapes[shape] += 1
        facts_rules += isinstance(state, dict) and {"facts", "rules"} <= set(state)
    return {
        "noul_rows": noul,
        "gold_true_fraction": true / noul if noul else None,
        "generic_yes_no_criteria": generic,
        "state_shapes_top5": dict(shapes.most_common(5)),
        "facts_and_rules_state_rows": facts_rules,
        "families_top8": dict(families.most_common(8)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--typed-gold", type=Path, required=True)
    parser.add_argument("--typed-prompts", type=Path, required=True)
    parser.add_argument(
        "--readout", action="append", default=[], help="label=readout_dir[=cal_probs]"
    )
    parser.add_argument("--cal-rows", type=Path)
    parser.add_argument(
        "--train", action="append", default=[], help="name=flattened_train_jsonl"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    questions, panel = dev_noul(args.typed_gold, args.typed_prompts)
    report: dict[str, Any] = {"typed_dev_noul": panel, "readouts": {}, "train": {}}
    for item in args.readout:
        label, directory, *rest = item.split("=")
        cal = cal_noul(args.cal_rows, Path(rest[0])) if rest else None
        report["readouts"][label] = readout_summary(
            questions, Path(directory) / "dev.predictions.jsonl", cal
        )
    for item in args.train:
        name, path = item.split("=", 1)
        report["train"][name] = train_summary(Path(path))
    write_json(args.output, report)
    print(json.dumps(report, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
