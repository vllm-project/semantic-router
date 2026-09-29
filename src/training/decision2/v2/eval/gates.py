"""Release gate checks from stored, sealed same-panel predictions (no GPU).

    python3 -m v2.eval.gates paired --left RUN --right RUN --left-name A --right-name B --output OUT
    python3 -m v2.eval.gates types --run RUN --label L --output OUT
    python3 -m v2.eval.gates public231 --left RUN --right RUN --left-name A --right-name B --output OUT
    python3 -m v2.eval.gates c1 --left RUN --right RUN --left-name A --right-name B --gold GOLD \
        [--retired RETIRED-v1_2.json --retired-sha SHA] --output OUT

`paired` runs the joint v3 paired bootstrap (5,000 draws, seed 20260927) between two run
directories after checking both seals. It reports the composite and per-axis (T typed,
H human transfer) intervals and writes to OUT, never into either run directory.

`public231` is the JevBench public-231 non-regression guard. It re-scores both sealed
prediction files and reports left − right in items with the exact two-sided McNemar p on
the discordant items, the within-tier paired bootstrap interval, and counts by tier, family
and type. The verdict is REGRESSION when left − right < 0 and p < 0.05. The panel cannot
separate sibling checkpoints, so the guard only catches large losses; it is never a
selection criterion.

`c1` is the JevArena-C1 v1.2 post-key successor guard (successor-rule item 8). A RUN is a C1
run directory (`SEAL-C1.json`, `output/sealed-c1.predictions.jsonl`): the successor on the
left, the current revision's run on the right. It checks both seals and the gold's hash,
drops the pinned v1.2 retired items and repeats the scoring events' paired comparison (group
bootstrap by source group within each task, 5,000 draws, seed 20260927), so on event runs it
reproduces the event's delta and intervals. p is the two-sided percentile-bootstrap p-value
of the same draws. The verdict is REGRESSION when left − right < 0 and p < 0.05 (the paired
95% interval lies below 0), otherwise PASS. The gold stays on node A (`c1-postkey.sh gate`
decrypts it for the call); OUT holds aggregates only. Label: "JevArena-C1 v1.2, post-key (not
an independent validation)". C1 is never training data and never a selection criterion.

`types` is the "no decision type collapsed" check on typed FINAL. For each of Choice, Noul
and Score it reports accuracy with a Wilson 95% interval against chance, the predicted
answer distribution (semantic values for Choice, levels for Score) against the gold
distribution, and per-level recall for Score. A type is COLLAPSED when one predicted
answer takes >= 90% of its answers, or when the lower bound of its accuracy interval is
not above chance.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from v2.eval import panels
from v2.eval.same_panel import (
    PAIRED_REPLICATES,
    PAIRED_SEED,
    prediction_path,
    read_jsonl,
    sha_file,
    write_json,
)
from v2.eval.sealed import score as c1score

TOP_SHARE = 0.9
PUBLIC_ALPHA = 0.05
C1_ALPHA = 0.05
C1_PANEL = "sealed-c1"
C1_USE = (
    "successor-rule item 8 only: never training data, never a selection criterion"
    " (development or siblings)"
)


def verified(run: Path, panel: str) -> Path:
    sealed = json.loads((run / "SEAL.json").read_text(encoding="utf-8"))
    path = prediction_path(run, panel)
    if sha_file(path) != sealed["panels"][panel]["predictions_sha256"]:
        raise ValueError(f"{run}: {panel} predictions changed after the seal")
    return path


def paired(args: argparse.Namespace) -> int:
    from jev_arena.compare_v3 import compare as compare_v3

    result = compare_v3(
        panels.path(args.panel_root, "typed-final", "gold"),
        panels.path(args.panel_root, "css15", "gold"),
        verified(args.left, "typed-final"),
        verified(args.left, "css15"),
        verified(args.right, "typed-final"),
        verified(args.right, "css15"),
        left_name=args.left_name,
        right_name=args.right_name,
        replicates=PAIRED_REPLICATES,
        seed=PAIRED_SEED,
    )
    result["label"] = "post-key same-panel"
    result["runs"] = {"left": str(args.left), "right": str(args.right)}
    write_json(args.output, result)
    print(
        json.dumps(
            {
                "delta": result["point"]["delta"],
                "ci95": result["ci95"],
                "H_ci95": result["axis_ci95"]["H"]["delta"],
                "T_ci95": result["axis_ci95"]["T"]["delta"],
            }
        )
    )
    return 0


def wilson(correct: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = correct / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (centre - half, centre + half)


def type_summary(
    gold: list[dict[str, Any]], predictions: dict[str, Any]
) -> dict[str, Any]:
    from benchmark.score import evaluate_answer

    cells: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "n": 0,
            "valid": 0,
            "correct": 0,
            "chance": 0.0,
            "pred": Counter(),
            "gold": Counter(),
            "level_hits": Counter(),
            "level_n": Counter(),
        }
    )
    for item in gold:
        answers = (predictions.get(item["id"]) or {}).get("answers")
        for key, question in item["questions"].items():
            kind = question["type"]
            truth = item["gold"][key]
            cell = cells[kind]
            cell["n"] += 1
            count = 2 if kind == "noul" else len(question["criteria"])
            cell["chance"] += 1.0 / count
            gold_value = truth.get("semantic_value", truth["value"])
            cell["gold"][str(gold_value)] += 1
            point = None
            if (
                isinstance(answers, dict)
                and key in answers
                and isinstance(answers[key], dict)
            ):
                result = evaluate_answer(question, truth, answers[key])
                if result.get("status") == "ok":
                    cell["valid"] += 1
                    point = result.get("semantic_point")
                    cell["correct"] += bool(result.get("correct"))
            cell["pred"][str(point)] += 1
            if kind == "score":
                cell["level_n"][str(truth["value"])] += 1
                cell["level_hits"][str(truth["value"])] += point == truth["value"]
    out = {}
    for kind, cell in sorted(cells.items()):
        n = cell["n"]
        chance = cell["chance"] / n
        low, high = wilson(cell["correct"], n)
        top_label, top_count = cell["pred"].most_common(1)[0]
        collapsed = []
        if top_count / n >= TOP_SHARE:
            collapsed.append(f"one answer ({top_label}) takes {top_count / n:.0%}")
        if low <= chance:
            collapsed.append("accuracy interval not above chance")
        entry = {
            "slots": n,
            "valid": cell["valid"],
            "accuracy": cell["correct"] / n,
            "accuracy_ci95": [low, high],
            "chance": chance,
            "gold_majority_share": cell["gold"].most_common(1)[0][1] / n,
            "predicted_distinct": len([k for k in cell["pred"] if k != "None"]),
            "predicted_top_share": top_count / n,
            "predicted_distribution": dict(cell["pred"].most_common()),
            "gold_distribution": dict(cell["gold"].most_common()),
            "verdict": "COLLAPSED: " + "; ".join(collapsed) if collapsed else "OK",
        }
        if kind == "score":
            entry["recall_by_level"] = {
                level: cell["level_hits"][level] / cell["level_n"][level]
                for level in sorted(cell["level_n"])
            }
        out[kind] = entry
    return out


def types(args: argparse.Namespace) -> int:
    from benchmark.score import load_jsonl

    gold = load_jsonl(panels.path(args.panel_root, "typed-final", "gold"))
    gold = list(gold.values()) if isinstance(gold, dict) else gold
    predictions = {
        row["id"]: row for row in read_jsonl(verified(args.run, "typed-final"))
    }
    result = {
        "schema": "dev2-gate-types/1",
        "label": args.label,
        "run": str(args.run),
        "rule": f"COLLAPSED if one answer >= {TOP_SHARE:.0%} of a type or the Wilson 95% lower bound <= chance",
        "types": type_summary(gold, predictions),
    }
    write_json(args.output, result)
    print(
        json.dumps(
            {
                k: {
                    "acc": round(v["accuracy"], 3),
                    "chance": round(v["chance"], 3),
                    "top": round(v["predicted_top_share"], 3),
                    "verdict": v["verdict"],
                }
                for k, v in result["types"].items()
            }
        )
    )
    return 0


def mcnemar_exact(left_only: int, right_only: int) -> float:
    n = left_only + right_only
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(left_only, right_only) + 1))
    return min(1.0, 2 * tail / 2**n)


def public_guard(
    left: dict[str, dict[str, Any]],
    right: dict[str, dict[str, Any]],
    targets: dict[str, dict[str, Any]],
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    from v2.eval import overlap_effects

    if set(left) != set(targets) or set(right) != set(targets):
        raise ValueError("public 231 outcomes do not cover the panel items")
    left_only = sum(left[i]["correct"] and not right[i]["correct"] for i in targets)
    right_only = sum(right[i]["correct"] and not left[i]["correct"] for i in targets)
    delta = left_only - right_only
    p = mcnemar_exact(left_only, right_only)
    breakdown: dict[str, Any] = {}
    for name, key in (
        ("tiers", "tier"),
        ("families", "family"),
        ("types", "task_type"),
    ):
        cells: dict[str, dict[str, int]] = defaultdict(
            lambda: {"items": 0, "left": 0, "right": 0}
        )
        for item_id, target in targets.items():
            cell = cells[target[key]]
            cell["items"] += 1
            cell["left"] += bool(left[item_id]["correct"])
            cell["right"] += bool(right[item_id]["correct"])
        breakdown[name] = dict(sorted(cells.items()))
    strata = overlap_effects.public_strata(left, right, set())
    return {
        "items": len(targets),
        "left_correct": sum(bool(left[i]["correct"]) for i in targets),
        "right_correct": sum(bool(right[i]["correct"]) for i in targets),
        "delta": delta,
        "discordant": {"left_only": left_only, "right_only": right_only},
        "mcnemar_exact_p": p,
        "ci95": overlap_effects.strata_bootstrap(strata, replicates, seed),
        **breakdown,
        "verdict": "REGRESSION" if delta < 0 and p < PUBLIC_ALPHA else "OK",
    }


def public231(args: argparse.Namespace) -> int:
    from v2.eval import overlap_effects

    panels.verify(args.panel_root, ["public231"])
    panel_dir = args.panel_root / panels.FORMAL["public231"]["panel_dir"]
    targets = {
        row["id"]: row
        for row in read_jsonl(panels.path(args.panel_root, "public231", "gold"))
    }
    result = {
        "schema": "dev2-gate-public231/1",
        "label": "post-key same-panel",
        "rule": (
            f"REGRESSION if left − right < 0 items and the exact two-sided McNemar "
            f"p < {PUBLIC_ALPHA}"
        ),
        "runs": {"left": str(args.left), "right": str(args.right)},
        "names": {"left": args.left_name, "right": args.right_name},
        **public_guard(
            overlap_effects.public_outcomes(args.left, panel_dir),
            overlap_effects.public_outcomes(args.right, panel_dir),
            targets,
            PAIRED_REPLICATES,
            PAIRED_SEED,
        ),
    }
    write_json(args.output, result)
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("delta", "discordant", "mcnemar_exact_p", "ci95", "verdict")
            }
        )
    )
    return 0


def bootstrap_p(draws: list[float]) -> float:
    """Two-sided percentile-bootstrap p: the smallest level at which the interval that
    ``score.interval`` takes from the same draws excludes 0. For 5,000 draws, p < 0.05 exactly
    when that 95% interval excludes 0."""
    last = len(draws) - 1
    below = sum(d < 0 for d in draws)
    above = sum(d > 0 for d in draws)
    return min(1.0, max(0.0, 2 * min(last - below, len(draws) - above) / last))


def c1_run(run: Path) -> tuple[Path, dict[str, Any]]:
    """A C1 run's predictions after checking its seal, and what the gate records about it."""
    seal_path = run / "SEAL-C1.json"
    predictions = prediction_path(run, C1_PANEL)
    sealed = json.loads(seal_path.read_text(encoding="utf-8"))
    if sealed.get("prompts_sha256") != panels.SEALED[C1_PANEL]["prompts_sha256"]:
        raise ValueError(f"{run}: the seal is not for the C1 prompts")
    if sha_file(predictions) != sealed.get("predictions_sha256"):
        raise ValueError(f"{run}: C1 predictions changed after the seal")
    if sealed.get("missing") != 0:
        raise ValueError(f"{run}: the seal records unanswered prompts")
    return predictions, {
        "dir": str(run),
        "seal_sha256": sha_file(seal_path),
        "seal_label": sealed.get("label"),
        "predictions_sha256": sealed["predictions_sha256"],
    }


def c1_guard(
    gold: list[dict[str, Any]],
    left: dict[str, dict[str, Any]],
    right: dict[str, dict[str, Any]],
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    left_summary, right_summary, draws, type_draws = c1score.paired_draws(
        gold, left, right, replicates, seed
    )
    delta = left_summary["c1"] - right_summary["c1"]
    ci95 = c1score.interval(draws)
    return {
        "left": {k: left_summary[k] for k in ("c1", "by_type", "items", "valid")},
        "right": {k: right_summary[k] for k in ("c1", "by_type", "items", "valid")},
        "delta": delta,
        "ci95": ci95,
        "p": bootstrap_p(draws),
        "by_type": {
            kind: {
                "delta": left_summary["by_type"][kind] - right_summary["by_type"][kind],
                "ci95": c1score.interval(values),
                "p": bootstrap_p(values),
            }
            for kind, values in type_draws.items()
        },
        "replicates": replicates,
        "seed": seed,
        "unit": "source groups within each task",
        "verdict": "REGRESSION" if delta < 0 and ci95[1] < 0 else "PASS",
    }


def c1(args: argparse.Namespace) -> int:
    output = args.output.resolve()
    for run in (args.left, args.right):
        if output.is_relative_to(run.resolve()):
            raise ValueError(f"write the gate output outside the run directory {run}")
    if sha_file(args.gold) != panels.SEALED[C1_PANEL]["gold_sha256"]:
        raise ValueError("--gold is not the JevArena-C1 gold")
    left_path, left_run = c1_run(args.left)
    right_path, right_run = c1_run(args.right)
    gold, item_set = c1score.apply_retired(read_jsonl(args.gold), args)
    if (item_set or {}).get("retired_sha256") != c1score.POSTKEY_RETIRED_SHA256:
        raise ValueError(f"the guard scores item set {c1score.POSTKEY_ITEM_SET} only")
    result = {
        "schema": "dev2-gate-c1/1",
        "label": c1score.POSTKEY_LABEL,
        "use": C1_USE,
        "rule": (
            f"REGRESSION if left − right < 0 C1 points and the two-sided paired bootstrap "
            f"p < {C1_ALPHA} (the paired 95% interval lies below 0); otherwise PASS"
        ),
        "names": {"left": args.left_name, "right": args.right_name},
        "runs": {"left": left_run, "right": right_run},
        "gold_sha256": panels.SEALED[C1_PANEL]["gold_sha256"],
        "item_set": item_set,
        **c1_guard(
            gold,
            {row["id"]: row for row in read_jsonl(left_path)},
            {row["id"]: row for row in read_jsonl(right_path)},
            PAIRED_REPLICATES,
            PAIRED_SEED,
        ),
    }
    write_json(args.output, result)
    print(json.dumps({key: result[key] for key in ("delta", "ci95", "p", "verdict")}))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--panel-root", type=Path, default=panels.DEFAULT_ROOT)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("paired", "public231", "c1"):
        pair = commands.add_parser(name)
        pair.add_argument("--left", type=Path, required=True)
        pair.add_argument("--right", type=Path, required=True)
        pair.add_argument("--left-name", required=True)
        pair.add_argument("--right-name", required=True)
        pair.add_argument("--output", type=Path, required=True)
        if name == "c1":
            pair.add_argument("--gold", type=Path, required=True)
            pair.add_argument("--retired", type=Path, default=c1score.POSTKEY_RETIRED)
            pair.add_argument("--retired-sha", default=c1score.POSTKEY_RETIRED_SHA256)
    two = commands.add_parser("types")
    two.add_argument("--run", type=Path, required=True)
    two.add_argument("--label", required=True)
    two.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    return {"paired": paired, "types": types, "public231": public231, "c1": c1}[
        args.command
    ](args)


if __name__ == "__main__":
    raise SystemExit(main())
