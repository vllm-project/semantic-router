"""Per-model effect of surface cues flagged by the leak audit, from stored predictions.

`follow` reads each model's sealed predictions on one panel and reports, over the
questions where a named cue picks an option: how often the gold is that option (a panel
property), how often the model picks it, the model's accuracy when the gold is and is
not the cue option, and, among its wrong answers, the share that went to the cue
option. Named cues:

  longest-description   the option whose description is the unique longest
  shortest-description  the option whose description is the unique shortest
  state-current         the option whose key equals a top-level state field (for
                        example the current state of a transition table)

`proxy` recomputes the development proxy P = 100*sqrt(T_dev*H_pilot) with typed-DEV
families excluded from T_dev and compares its calibration against post-key v3 with the
frozen proxy on the same models (Spearman, leave-one-out error, pair agreement).

Outputs are counts and rates only; no item text or gold values.

    python3 -m v2.eval.leak_effects follow --panel public231 --cue longest-description \
        --spec models.json --output follow.json
    python3 -m v2.eval.leak_effects proxy --spec proxy-spec.json --exclude-family transition_table \
        --output proxy.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from v2.eval import leak_audit
from v2.eval.leak_audit import Question

SCHEMA = "dev2-leak-effects/1"
CUES = ("longest-description", "shortest-description", "state-current")


def cue_option(question: Question, cue: str) -> int | None:
    if question.qtype != "choice":
        return None
    if cue in ("longest-description", "shortest-description"):
        lengths = [len(text) for text in question.descriptions]
        target = max(lengths) if cue == "longest-description" else min(lengths)
        return lengths.index(target) if lengths.count(target) == 1 else None
    if cue == "state-current":
        if not isinstance(question.state, dict):
            return None
        scalars = {value for value in question.state.values() if isinstance(value, str)}
        hits = [i for i, key in enumerate(question.keys) if key in scalars]
        return hits[0] if len(hits) == 1 else None
    raise ValueError(f"unknown cue {cue!r}")


def chosen_key(answer: Any) -> str | None:
    if not isinstance(answer, dict):
        return None
    choice = answer.get("choice")
    if isinstance(choice, str):
        return choice
    probabilities = answer.get("probabilities")
    if isinstance(probabilities, dict) and probabilities:
        best = max(probabilities.values())
        winners = [key for key, value in probabilities.items() if value == best]
        return winners[0] if len(winners) == 1 else None
    return None


def follow_model(
    questions: list[tuple[Question, str, int]],
    predictions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """questions: (question, question id inside the item, cue option index)."""
    n = hit_n = hit_correct = miss_n = miss_correct = 0
    picked_cue = valid = wrong = wrong_on_cue = 0
    for question, qid, cue in questions:
        n += 1
        answers = (predictions.get(question.item_id) or {}).get("answers")
        choice = chosen_key(answers.get(qid)) if isinstance(answers, dict) else None
        index = question.keys.index(choice) if choice in question.keys else None
        correct = index == question.gold
        valid += index is not None
        picked_cue += index == cue
        if question.gold == cue:
            hit_n += 1
            hit_correct += correct
        else:
            miss_n += 1
            miss_correct += correct
        if index is not None and not correct:
            wrong += 1
            wrong_on_cue += index == cue

    def rate(a: int, b: int) -> float | None:
        return a / b if b else None

    return {
        "questions": n,
        "valid": valid,
        "accuracy": rate(hit_correct + miss_correct, n),
        "picks_cue_option": rate(picked_cue, n),
        "gold_is_cue_option": {
            "n": hit_n,
            "correct": hit_correct,
            "accuracy": rate(hit_correct, hit_n),
        },
        "gold_is_not_cue_option": {
            "n": miss_n,
            "correct": miss_correct,
            "accuracy": rate(miss_correct, miss_n),
        },
        "wrong_answers": wrong,
        "wrong_answers_on_cue_option": rate(wrong_on_cue, wrong),
    }


def follow(args: argparse.Namespace) -> int:
    from v2.eval import panels as registry
    from v2.eval.same_panel import read_jsonl, sha_file, write_json

    registry.verify(args.panel_root, [args.panel])
    raw = leak_audit.load_panel(args.panel_root, args.panel)
    question_ids = question_ids_for(args.panel_root, args.panel)
    selected = []
    for question in raw:
        if args.group and question.group not in args.group:
            continue
        cue = cue_option(question, args.cue)
        if cue is None:
            continue
        selected.append(
            (question, question_ids[(question.item_id, question.group)], cue)
        )
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    base = sum(q.gold == cue for q, _, cue in selected)
    chance = (
        statistics.mean(1 / len(q.keys) for q, _, _ in selected) if selected else None
    )
    models = {}
    for entry in spec["models"]:
        path = Path(entry["predictions"])
        predictions = {row["id"]: row for row in read_jsonl(path)}
        result = follow_model(selected, predictions)
        result["predictions_sha256"] = sha_file(path)
        models[entry["key"]] = result
    out = {
        "schema": SCHEMA,
        "panel": args.panel,
        "cue": args.cue,
        "groups": args.group or "all",
        "questions_with_cue": len(selected),
        "gold_is_cue_option": base,
        "gold_is_cue_option_rate": base / len(selected) if selected else None,
        "chance_rate": chance,
        "models": models,
    }
    write_json(args.output, out, exclusive=True)
    print(
        json.dumps(
            {
                "questions_with_cue": len(selected),
                "gold_is_cue_option_rate": out["gold_is_cue_option_rate"],
                "chance_rate": chance,
            }
        )
    )
    for key, value in models.items():
        print(
            f"{key:14s} acc={value['accuracy']:.3f} picks_cue={value['picks_cue_option']:.3f} "
            f"acc|gold=cue={value['gold_is_cue_option']['accuracy']} "
            f"acc|gold!=cue={value['gold_is_not_cue_option']['accuracy']} "
            f"wrong_on_cue={value['wrong_answers_on_cue_option']}"
        )
    return 0


def question_ids_for(root: Path, panel: str) -> dict[tuple[str, str], str]:
    """(item id, audit group) -> question id inside the item's `questions`."""
    from v2.eval import panels as registry
    from v2.eval.same_panel import read_jsonl

    out = {}
    if panel in ("typed-dev", "typed-final"):
        for row in read_jsonl(registry.path(root, panel, "gold")):
            for key in row["questions"]:
                out[(row["id"], f"{row['family']}/{key}")] = key
        return out
    groups = {q.item_id: q.group for q in leak_audit.load_panel(root, panel)}
    for row in read_jsonl(registry.path(root, panel, "prompts")):
        (key,) = row["questions"]
        out[(row["id"], groups[row["id"]])] = key
    return out


# -------------------------------------------------------------------- proxy


def calibration(names, tiers, x, y) -> dict[str, Any]:
    from v2.eval import proxy_calibration as pc

    mx, my = statistics.mean(x), statistics.mean(y)
    var = sum((a - mx) ** 2 for a in x)
    slope = sum((a - mx) * (b - my) for a, b in zip(x, y)) / var if var else 0.0
    same_tier = [
        (x[i] - x[j]) * (y[i] - y[j]) > 0
        for i, j in itertools.combinations(range(len(x)), 2)
        if tiers[i] == tiers[j] and y[i] != y[j]
    ]
    all_pairs = [
        (x[i] - x[j]) * (y[i] - y[j]) > 0
        for i, j in itertools.combinations(range(len(x)), 2)
        if y[i] != y[j]
    ]
    return {
        "models": len(x),
        "spearman": pc.pearson(pc.ranks(x), pc.ranks(y)),
        "kendall_tau_b": pc.kendall_tau_b(x, y),
        "loo_linear_v3_error": pc.loo_linear(x, y),
        "fit": {"intercept": my - slope * mx, "slope": slope},
        "pairs_agree": {"n": len(all_pairs), "agree": sum(all_pairs)},
        "same_tier_pairs_agree": {"n": len(same_tier), "agree": sum(same_tier)},
    }


def proxy(args: argparse.Namespace) -> int:
    from benchmark.score import load_jsonl
    from transfer.score import read_jsonl as read_css

    from v2.eval import panels as registry
    from v2.eval import proxy_calibration as pc
    from v2.eval.same_panel import read_jsonl, write_json

    registry.verify(args.panel_root, ["typed-dev", "css-pilot"])
    typed_gold = load_jsonl(registry.path(args.panel_root, "typed-dev", "gold"))
    pilot_gold = read_css(registry.path(args.panel_root, "css-pilot", "gold"))
    sets: dict[str, list[dict[str, Any]]] = {}
    for label, spec_path in (
        ("calibration", args.spec),
        ("out_of_sample", args.extra_spec),
    ):
        if spec_path is None:
            continue
        rows = []
        for entry in json.loads(spec_path.read_text(encoding="utf-8"))["models"]:
            report = json.loads(Path(entry["report"]).read_text(encoding="utf-8"))
            typed_rows = pc.typed_outcomes(
                typed_gold, {r["id"]: r for r in read_jsonl(Path(entry["typed_dev"]))}
            )
            tasks, labels = pc.pilot_outcomes(
                pilot_gold, {r["id"]: r for r in read_jsonl(Path(entry["css_pilot"]))}
            )
            kept = [row for row in typed_rows if row[0] not in args.exclude_family]
            frozen = pc.features(typed_rows, tasks, labels)
            variant = pc.features(kept, tasks, labels)
            rows.append(
                {
                    "key": entry["key"],
                    "tier": report["model"]["tier"],
                    "v3": report["v3"]["score"],
                    "P": frozen["P"],
                    "P_excluded": variant["P"],
                    "T_dev": frozen["T_dev"],
                    "T_dev_excluded": variant["T_dev"],
                    "H_pilot": frozen["H_pilot"],
                    "families": sorted({row[0] for row in typed_rows}),
                }
            )
        sets[label] = rows
    views: dict[str, list[dict[str, Any]]] = {"calibration": sets["calibration"]}
    if "out_of_sample" in sets:
        views["all"] = sets["calibration"] + sets["out_of_sample"]
    out: dict[str, Any] = {
        "schema": SCHEMA,
        "excluded_families": sorted(args.exclude_family),
        "models": sets,
        "views": {},
    }
    for view, rows in views.items():
        names = [r["key"] for r in rows]
        tiers = [r["tier"] for r in rows]
        y = [r["v3"] for r in rows]
        out["views"][view] = {
            "frozen_P": calibration(names, tiers, [r["P"] for r in rows], y),
            "P_excluding": calibration(
                names, tiers, [r["P_excluded"] for r in rows], y
            ),
        }
    if "out_of_sample" in sets:
        fit = out["views"]["calibration"]
        errors: dict[str, list[float]] = defaultdict(list)
        for row in sets["out_of_sample"]:
            for name, key in (("frozen_P", "P"), ("P_excluding", "P_excluded")):
                line = fit[name]["fit"]
                errors[name].append(
                    line["intercept"] + line["slope"] * row[key] - row["v3"]
                )
        out["out_of_sample_error"] = {
            name: {
                "mae": statistics.mean(abs(e) for e in values),
                "max": max(abs(e) for e in values),
                "errors": values,
            }
            for name, values in errors.items()
        }
    write_json(args.output, out, exclusive=True)
    for view, value in out["views"].items():
        for name, stats in value.items():
            loo = stats["loo_linear_v3_error"]
            print(
                f"{view:12s} {name:12s} spearman={stats['spearman']:.3f} "
                f"loo_mae={loo['mae']:.2f} max={loo['max']:.2f} "
                f"pairs={stats['pairs_agree']['agree']}/{stats['pairs_agree']['n']} "
                f"tier={stats['same_tier_pairs_agree']['agree']}/{stats['same_tier_pairs_agree']['n']}"
            )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("follow")
    one.add_argument(
        "--panel-root", type=Path, default=Path("/data/dev2/private/panels")
    )
    one.add_argument("--panel", required=True, choices=sorted(leak_audit.PANEL_ROLES))
    one.add_argument("--cue", required=True, choices=CUES)
    one.add_argument("--group", action="append", help="audit group(s) to keep")
    one.add_argument(
        "--spec", type=Path, required=True, help="JSON {models: [{key, predictions}]}"
    )
    one.add_argument("--output", type=Path, required=True)
    two = commands.add_parser("proxy")
    two.add_argument(
        "--panel-root", type=Path, default=Path("/data/dev2/private/panels")
    )
    two.add_argument(
        "--spec", type=Path, required=True, help="frozen proxy-calibration spec"
    )
    two.add_argument(
        "--extra-spec", type=Path, help="out-of-sample models, same format"
    )
    two.add_argument("--exclude-family", action="append", required=True)
    two.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    return follow(args) if args.command == "follow" else proxy(args)


if __name__ == "__main__":
    raise SystemExit(main())
