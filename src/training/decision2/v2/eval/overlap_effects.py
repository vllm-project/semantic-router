"""Effect of evaluation items flagged by a later training-data rescreen, from stored predictions.

    python3 -m v2.eval.overlap_effects flagged --rescreen DIR --role ROLE [--role ROLE ...] \
        [--expect-groups N] --output FLAGGED
    python3 -m v2.eval.overlap_effects run --spec SPEC --flagged FLAGGED --output OUT [--jobs N]

`flagged` reads the research & data rescreen's private overlap receipts under DIR
(`scan/<pool>.private.json` per training pool and `scan-union/union.private.json`) and
collects every evaluation-item id that a training group hit in one of the named roles.
Ids are assigned to the frozen panel that contains them (typed FINAL, CSS15, public 231,
mlx-diag); ids in no frozen panel (for example Decision Bench v4) are counted per role
but not scored. It also records which training pools hit each panel stratum and by
which detection methods. FLAGGED holds item ids and stays on the node.

`run` rescores every model in SPEC from its sealed predictions on the full panels and
without the flagged items (the same items for every model): typed T, human transfer H
(median over the 15 tasks of macro-F1 over each task's full label set), v3 =
100*sqrt(T*H), per-task macro-F1, public 231 and mlx-diag. For each candidate and
comparator pair it runs:

- the joint v3 paired bootstrap of `jev_arena.compare_v3` (typed groups within family,
  CSS tasks then items; 5,000 draws, seed 20260927) on the full and the reduced panels,
  and on the full panels with a second seed to show how far intervals move from
  resampling alone. On the full panels it must reproduce the stored paired files listed
  in SPEC bit for bit;
- paired item bootstraps for the macro-F1 of the focus task (items within the task),
  public 231 (items within tier) and mlx-diag (items within type x language);
- the contamination check on CSS15: accuracy on the flagged items minus accuracy on the
  unflagged items of the same tasks (weighted by flagged items per task), for both
  models, and the candidate-minus-comparator difference of that gap (items resampled
  within task and flag). A candidate that gains more on the flagged items than its
  comparator shows a positive difference.

Outputs are aggregates only: no item ids, text, gold values or answers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
import random
import statistics
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from benchmark.generate import FINAL_FAMILIES
from benchmark.score import evaluate_answer, load_jsonl, score_suite
from jev_arena import jevbench_public
from jev_arena.compare_v3 import _point, _task_f1
from transfer.build import EVALUATION_TASKS
from transfer.compare import interval95
from transfer.score import evaluate as evaluate_css
from transfer.score import macro_f1
from transfer.score import read_jsonl as read_css_jsonl
from transfer.score import score as score_css
from v2.eval import panels
from v2.eval.gates import verified
from v2.eval.same_panel import (
    LABEL,
    PAIRED_REPLICATES,
    PAIRED_SEED,
    identity_value,
    prediction_path,
    read_jsonl,
    sha_file,
    write_json,
)

SCHEMA = "dev2-overlap-effects/1"
FLAGGED_SCHEMA = "dev2-overlap-flagged/1"
SECOND_SEED = PAIRED_SEED + 1
SCORED_PANELS = ("typed-final", "css15", "public231", "mlx-diag")
MLX_TYPES = ("choice", "noul", "score")


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


# ---------------------------------------------------------------- panels


def load_panels(panel_root: Path, mlx_panel: Path) -> dict[str, Any]:
    panels.verify(panel_root, ["typed-final", "css15", "public231"])
    mlx_gold = mlx_panel / "gold.jsonl"
    if sha_file(mlx_gold) != panels.DEVELOPMENT["mlx-diag"]["gold_sha256"]:
        raise ValueError("mlx-diag gold differs from the frozen panel")
    css = read_css_jsonl(panels.path(panel_root, "css15", "gold"))
    return {
        "root": panel_root,
        "typed-final": load_jsonl(panels.path(panel_root, "typed-final", "gold")),
        "css15": {i: row for i, row in css.items() if row["role"] == "evaluation"},
        "public231": {
            row["id"]: row
            for row in read_jsonl(panels.path(panel_root, "public231", "gold"))
        },
        "public231_dir": panel_root / panels.FORMAL["public231"]["panel_dir"],
        "mlx-diag": {row["id"]: row for row in read_jsonl(mlx_gold)},
        "mlx-prompts": {
            row["id"]: row for row in read_jsonl(mlx_panel / "prompts.jsonl")
        },
        "sha256": {
            "typed-final": panels.FORMAL["typed-final"]["gold_sha256"],
            "css15": panels.FORMAL["css15"]["gold_sha256"],
            "public231": panels.FORMAL["public231"]["gold_sha256"],
            "mlx-diag": sha_file(mlx_gold),
        },
    }


def panel_of(item_id: str, gold: dict[str, Any]) -> str | None:
    for panel in SCORED_PANELS:
        if item_id in gold[panel]:
            return panel
    return None


def stratum(panel: str, row: dict[str, Any]) -> str:
    if panel == "css15":
        return row["task"]
    if panel == "public231":
        return row["tier"]
    if panel == "mlx-diag":
        return f"{row['type']}/{row['language']}"
    return row["family"]


# ---------------------------------------------------------------- flagged


def collect_flagged(
    rescreen: Path, roles: set[str], gold: dict[str, Any]
) -> dict[str, Any]:
    receipts = sorted((rescreen / "scan").glob("*.private.json"))
    union = rescreen / "scan-union" / "union.private.json"
    if not receipts:
        raise ValueError(f"{rescreen}: no per-pool receipts")
    group_pools: dict[str, set[str]] = defaultdict(set)
    item_groups: dict[str, set[str]] = defaultdict(set)
    item_roles: dict[str, set[str]] = defaultdict(set)
    item_methods: dict[str, set[str]] = defaultdict(set)
    hashes = {}
    for path in [*receipts, *([union] if union.exists() else [])]:
        pool = None if path == union else path.name[: -len(".private.json")]
        hashes[str(path.relative_to(rescreen))] = sha_file(path)
        groups = json.loads(path.read_text(encoding="utf-8"))["groups"]
        for group_id, group in groups.items():
            protected = group.get("protected_ids") or {}
            hit_roles = roles & set(protected)
            if not hit_roles:
                continue
            if pool is not None:
                group_pools[group_id].add(pool)
            for role in hit_roles:
                for item_id in protected[role]:
                    item_roles[item_id].add(role)
                    item_groups[item_id].add(group_id)
            for method, by_role in (group.get("by_method") or {}).items():
                for role in hit_roles & set(by_role):
                    for item_id in by_role[role]:
                        item_methods[item_id].add(method)
    excluded = set().union(*item_groups.values())
    cross_check = None
    private = rescreen / "rescreen.private.json"
    if private.exists():
        hits = json.loads(private.read_text(encoding="utf-8"))["hits"]
        from_hits = {
            group_id
            for by_group in hits.values()
            for group_id, hit in by_group.items()
            if roles & set(hit.get("roles") or ())
        }
        if from_hits != excluded:
            raise ValueError(
                f"receipts give {len(excluded)} excluded groups, rescreen hits {len(from_hits)}"
            )
        cross_check = {"sha256": sha_file(private), "groups": len(from_hits)}
    by_panel: dict[str, list[str]] = {panel: [] for panel in SCORED_PANELS}
    unscored: dict[str, set[str]] = defaultdict(set)
    strata: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )
    methods: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for item_id in sorted(item_roles):
        panel = panel_of(item_id, gold)
        if panel is None:
            for role in item_roles[item_id]:
                unscored[role].add(item_id)
            continue
        by_panel[panel].append(item_id)
        cell = strata[panel][stratum(panel, gold[panel][item_id])]
        cell["items"] += 1
        pools = set()
        for group_id in item_groups[item_id]:
            pools |= group_pools.get(group_id) or {"union-only"}
        for pool in pools:
            cell[f"pool:{pool}"] += 1
        methods[panel]["+".join(sorted(item_methods[item_id])) or "none"] += 1
    return {
        "schema": FLAGGED_SCHEMA,
        "rescreen": str(rescreen),
        "receipts_sha256": hashes,
        "rescreen_private_cross_check": cross_check,
        "roles": sorted(roles),
        "groups": len(excluded),
        "groups_by_pool": {
            pool: sum(pool in pools for pools in group_pools.values())
            for pool in sorted(set().union(*group_pools.values()))
        },
        "panels": by_panel,
        "counts": {panel: len(ids) for panel, ids in by_panel.items()},
        "ids_sha256": {panel: digest(ids) for panel, ids in by_panel.items()},
        "unscored_by_role": {role: len(ids) for role, ids in sorted(unscored.items())},
        "unscored_items": len(set().union(*unscored.values())),
        "strata": {
            panel: {
                key: dict(sorted(cell.items())) for key, cell in sorted(rows.items())
            }
            for panel, rows in sorted(strata.items())
        },
        "methods": {
            panel: dict(sorted(rows.items())) for panel, rows in methods.items()
        },
        "gold_sha256": gold["sha256"],
    }


def flagged(args: argparse.Namespace) -> int:
    gold = load_panels(args.panel_root, args.mlx_panel)
    result = collect_flagged(args.rescreen, set(args.role), gold)
    if args.expect_groups is not None and result["groups"] != args.expect_groups:
        raise ValueError(
            f"expected {args.expect_groups} groups, found {result['groups']}"
        )
    sha = write_json(args.output, result)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": sha,
                "groups": result["groups"],
                "counts": result["counts"],
                "unscored_by_role": result["unscored_by_role"],
            }
        )
    )
    return 0


# ---------------------------------------------------------------- per-model outcomes


def typed_counts(
    gold: dict[str, dict[str, Any]], predictions: dict[str, dict[str, Any]]
) -> dict[str, int]:
    """Correct answer slots per item, counted as `jev_arena.compare_v3._typed_groups` does."""
    counts = {}
    for item_id, item in gold.items():
        prediction = predictions.get(item_id)
        answers = prediction.get("answers") if prediction is not None else None
        if not isinstance(answers, dict) or set(answers) != set(item["questions"]):
            counts[item_id] = 0
            continue
        counts[item_id] = sum(
            evaluate_answer(question, item["gold"][key], answers[key]).get(
                "correct", False
            )
            for key, question in item["questions"].items()
        )
    return counts


def typed_value(
    gold: dict[str, dict[str, Any]], counts: dict[str, int], exclude: set[str]
) -> float:
    correct: dict[str, int] = defaultdict(int)
    slots: dict[str, int] = defaultdict(int)
    for item_id, item in gold.items():
        if item_id not in exclude:
            correct[item["family"]] += counts[item_id]
            slots[item["family"]] += len(item["questions"])
    return statistics.mean(correct[family] / slots[family] for family in FINAL_FAMILIES)


def css_outcomes(
    gold: dict[str, dict[str, Any]], predictions: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    out = {}
    for item_id, row in gold.items():
        result = evaluate_css(row, predictions.get(item_id))
        out[item_id] = {
            "choice": result["choice"] if result["valid"] else None,
            "correct": bool(result.get("correct", False)),
            "gold_probability": result.get("gold_probability"),
        }
    return out


def css_task_f1(
    gold: dict[str, dict[str, Any]],
    outcomes: dict[str, dict[str, Any]],
    exclude: set[str],
) -> dict[str, float]:
    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item_id, row in gold.items():
        if item_id not in exclude:
            rows[row["task"]].append(row)
    return {
        task: macro_f1(
            [row["gold"] for row in rows[task]],
            [outcomes[row["id"]]["choice"] for row in rows[task]],
            rows[task][0]["labels"],
        )
        for task in sorted(rows)
    }


def public_outcomes(run: Path, panel_dir: Path) -> dict[str, dict[str, Any]]:
    seal = json.loads((run / "SEAL.json").read_text(encoding="utf-8"))["panels"][
        "public231"
    ]
    model_id = identity_value(seal, "model_id")
    with tempfile.TemporaryDirectory(prefix="overlap-effects-") as scratch:
        report = jevbench_public.score(
            panel_dir,
            verified(run, "public231"),
            model_id if model_id is not None else jevbench_public.ABSENT_MODEL_ID,
            identity_value(seal, "model_revision"),
            Path(scratch) / "public231.score.json",
        )
    return {
        row["id"]: {"tier": row["tier"], "correct": bool(row["correct"])}
        for row in report["per_item"]
    }


def public_value(
    outcomes: dict[str, dict[str, Any]], exclude: set[str]
) -> dict[str, Any]:
    tiers: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for item_id, row in outcomes.items():
        if item_id not in exclude:
            tiers[row["tier"]][0] += row["correct"]
            tiers[row["tier"]][1] += 1
    return {
        "correct": sum(value[0] for value in tiers.values()),
        "items": sum(value[1] for value in tiers.values()),
        "tiers": {
            tier: {"correct": value[0], "items": value[1]}
            for tier, value in sorted(tiers.items())
        },
    }


def mlx_outcomes(
    gold: dict[str, dict[str, Any]],
    prompts: dict[str, dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Per-item outcomes exactly as `v2.eval.multilingual_panel.score` counts them."""
    if set(predictions) - set(gold):
        raise ValueError("mlx-diag predictions contain unknown ids")
    out = {}
    for item_id, g in gold.items():
        question = prompts[item_id]["questions"]["q"]
        pred = predictions.get(item_id)
        if pred is not None and pred.get("source_input_sha256") != g["input_sha256"]:
            raise ValueError(f"{item_id}: prediction made from different input")
        answer = ((pred or {}).get("answers") or {}).get("q")
        if g["type"] == "choice":
            target = {
                "type": "choice",
                "value": g["value"],
                "label_to_semantic": {k: k for k in question["criteria"]},
                "semantic_value": g["value"],
            }
        else:
            target = {"type": g["type"], "value": g["value"]}
        result = (
            evaluate_answer(question, target, answer)
            if answer is not None
            else {"status": "missing"}
        )
        out[item_id] = {
            "type": g["type"],
            "language": g["language"],
            "correct": bool(result.get("correct")),
        }
    return out


def mlx_value(outcomes: dict[str, dict[str, Any]], exclude: set[str]) -> dict[str, Any]:
    cells: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    for item_id, row in outcomes.items():
        if item_id not in exclude:
            cell = cells[(row["type"], row["language"])]
            cell[0] += row["correct"]
            cell[1] += 1
    by_type = {
        kind: {lang: c / n for (t, lang), (c, n) in sorted(cells.items()) if t == kind}
        for kind in MLX_TYPES
    }
    return {
        "items": sum(n for _c, n in cells.values()),
        "type_macro_accuracy": statistics.mean(
            statistics.mean(by_type[t].values()) for t in by_type
        ),
        "per_language": {
            lang: statistics.mean(
                by_type[t][lang] for t in by_type if lang in by_type[t]
            )
            for lang in sorted({lang for (_t, lang) in cells})
        },
    }


def load_model(name: str, cfg: dict[str, Any], gold: dict[str, Any]) -> dict[str, Any]:
    run = Path(cfg["run"])
    typed_path, css_path = verified(run, "typed-final"), verified(run, "css15")
    model: dict[str, Any] = {
        "run": str(run),
        "predictions_sha256": {
            "typed": sha_file(typed_path),
            "css": sha_file(css_path),
            "public231": sha_file(verified(run, "public231")),
        },
        "typed": typed_counts(gold["typed-final"], load_jsonl(typed_path)),
        "css": css_outcomes(gold["css15"], read_css_jsonl(css_path)),
        "public": public_outcomes(run, gold["public231_dir"]),
    }
    typed_report = score_suite(
        panels.path(gold["root"], "typed-final", "gold"),
        typed_path,
        name,
        "overlap",
        "native",
    )
    css_report = score_css(panels.path(gold["root"], "css15", "gold"), css_path)
    model["canonical"] = {
        "T": typed_report["macro_family_accuracy"],
        "H": css_report["roles"]["evaluation"]["median_task_macro_f1_all"],
        "tasks": {
            task: css_report["tasks"][task]["macro_f1_all"] for task in EVALUATION_TASKS
        },
    }
    model["equivalent_runs"] = {}
    for other in map(Path, cfg.get("equivalent_runs", [])):
        other_typed, other_css = verified(other, "typed-final"), verified(
            other, "css15"
        )
        other_choices = css_outcomes(gold["css15"], read_css_jsonl(other_css))
        same = (
            typed_counts(gold["typed-final"], load_jsonl(other_typed)) == model["typed"]
            and {k: v["choice"] for k, v in other_choices.items()}
            == {k: v["choice"] for k, v in model["css"].items()}
            and public_outcomes(other, gold["public231_dir"]) == model["public"]
        )
        if not same:
            raise ValueError(
                f"{name}: {other} does not give the same outcomes as {run}"
            )
        model["equivalent_runs"][str(other)] = {
            "typed": sha_file(other_typed),
            "css": sha_file(other_css),
            "identical_outcomes": True,
        }
    model["mlx"] = None
    if cfg.get("mlx_run"):
        mlx_run = Path(cfg["mlx_run"])
        mlx_path = prediction_path(mlx_run, "mlx-diag")
        stored = json.loads(
            (mlx_run / "mlx-diag.score.json").read_text(encoding="utf-8")
        )
        if sha_file(mlx_path) != stored["predictions_sha256"]:
            raise ValueError(
                f"{name}: mlx-diag predictions differ from the stored score"
            )
        model["mlx"] = mlx_outcomes(
            gold["mlx-diag"],
            gold["mlx-prompts"],
            {row["id"]: row for row in read_jsonl(mlx_path)},
        )
        model["mlx_stored"] = {
            "run": str(mlx_run),
            "predictions_sha256": stored["predictions_sha256"],
            "type_macro_accuracy": stored["type_macro_accuracy"],
            "per_language": stored["per_language_mean_accuracy"],
        }
    report = json.loads((run / "REPORT.json").read_text(encoding="utf-8"))
    model["reported"] = {
        "v3": report["v3"]["score"],
        "T": report["v3"]["T"],
        "H": report["v3"]["H"],
        "tasks": {
            task: value["macro_f1"]
            for task, value in report["panels"]["css15"]["tasks"].items()
        },
        "public231": report["panels"]["public231"]["correct"],
    }
    return model


def model_values(
    model: dict[str, Any], gold: dict[str, Any], exclude: dict[str, set[str]]
) -> dict[str, Any]:
    t = (
        typed_value(gold["typed-final"], model["typed"], exclude["typed-final"])
        if exclude["typed-final"]
        else model["canonical"]["T"]
    )
    tasks = css_task_f1(gold["css15"], model["css"], exclude["css15"])
    h = (
        statistics.median(tasks.values())
        if exclude["css15"]
        else model["canonical"]["H"]
    )
    return {
        "T": t,
        "H": h,
        "v3": _point(t, h)["score"],
        "tasks": tasks,
        "public231": public_value(model["public"], exclude["public231"]),
        "mlx": (
            mlx_value(model["mlx"], exclude["mlx-diag"])
            if model["mlx"] is not None
            else None
        ),
    }


def check_full(
    name: str, model: dict[str, Any], full: dict[str, Any], gold: dict[str, Any]
) -> list[str]:
    """Differences between the full-panel rescore and the stored reports (must be none)."""
    problems = []
    reconstructed = typed_value(gold["typed-final"], model["typed"], set())
    if not math.isclose(
        reconstructed, model["canonical"]["T"], rel_tol=0, abs_tol=1e-12
    ):
        problems.append("typed grouped outcomes disagree with the canonical scorer")
    if full["tasks"] != model["canonical"]["tasks"]:
        problems.append("per-task macro-F1 differs from the canonical scorer")
    if statistics.median(full["tasks"].values()) != model["canonical"]["H"]:
        problems.append("median task macro-F1 differs from the canonical H")
    reported = model["reported"]
    for key in ("v3", "T", "H"):
        if full[key] != reported[key]:
            problems.append(f"{key} {full[key]!r} != REPORT {reported[key]!r}")
    if full["tasks"] != reported["tasks"]:
        problems.append("per-task macro-F1 differs from REPORT")
    if full["public231"]["correct"] != reported["public231"]:
        problems.append("public 231 differs from REPORT")
    if full["mlx"] is not None:
        stored = model["mlx_stored"]
        if full["mlx"]["type_macro_accuracy"] != stored["type_macro_accuracy"]:
            problems.append("mlx-diag type macro differs from the stored score")
        if full["mlx"]["per_language"] != stored["per_language"]:
            problems.append("mlx-diag per-language differs from the stored score")
    return [f"{name}: {problem}" for problem in problems]


# ---------------------------------------------------------------- paired structures


def pair_families(
    gold: dict[str, dict[str, Any]],
    left: dict[str, int],
    right: dict[str, int],
    exclude: set[str],
) -> dict[str, list[tuple[int, ...]]]:
    """`compare_v3._typed_groups` from per-item counts, without the full-panel asserts."""
    groups: dict[tuple[str, str], list[tuple[int, int, int]]] = defaultdict(list)
    for item_id, item in gold.items():
        if item_id not in exclude:
            groups[(item["family"], item["group_id"])].append(
                (left[item_id], right[item_id], len(item["questions"]))
            )
    by_family: dict[str, list[tuple[int, ...]]] = defaultdict(list)
    for (family, _group_id), items in sorted(groups.items()):
        by_family[family].append(tuple(sum(row[i] for row in items) for i in range(3)))
    if set(by_family) != set(FINAL_FAMILIES):
        raise ValueError("every typed family needs at least one group")
    return dict(by_family)


def pair_tasks(
    gold: dict[str, dict[str, Any]],
    left: dict[str, dict[str, Any]],
    right: dict[str, dict[str, Any]],
    exclude: set[str],
) -> dict[str, tuple[list[tuple[int, int, int]], int]]:
    """`compare_v3._css_tasks` from per-item choices, without the full-panel asserts."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in gold.values():
        if row["id"] not in exclude:
            grouped[row["task"]].append(row)
    if set(grouped) != set(EVALUATION_TASKS):
        raise ValueError("every CSS task needs at least one item")
    coded = {}
    for task, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: row["id"])
        labels = rows[0]["labels"]
        index = {label: i for i, label in enumerate(labels)}
        values = []
        for row in rows:
            choices = []
            for outcomes in (left, right):
                choice = outcomes[row["id"]]["choice"]
                choices.append(index[choice] if choice is not None else -1)
            values.append((index[row["gold"]], choices[0], choices[1]))
        coded[task] = values, len(labels)
    return coded


def v3_bootstrap(
    families: dict[str, list[tuple[int, ...]]],
    tasks: dict[str, tuple[list[tuple[int, int, int]], int]],
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    """The replicate loop of `jev_arena.compare_v3.compare`, unchanged."""
    rng = random.Random(seed)
    task_names = tuple(sorted(EVALUATION_TASKS))
    draws: dict[str, list[float]] = defaultdict(list)
    for _ in range(replicates):
        sampled_families = []
        for family in FINAL_FAMILIES:
            groups = families[family]
            counts = [0, 0, 0]
            for _group in groups:
                chosen = groups[rng.randrange(len(groups))]
                for side in range(3):
                    counts[side] += chosen[side]
            sampled_families.append((counts[0] / counts[2], counts[1] / counts[2]))
        typed_left_axis = statistics.mean(row[0] for row in sampled_families)
        typed_right_axis = statistics.mean(row[1] for row in sampled_families)
        sampled_tasks: list[list[float]] = [[], []]
        for _task in task_names:
            name = task_names[rng.randrange(len(task_names))]
            task_rows, nlabels = tasks[name]
            pair = _task_f1(task_rows, nlabels, rng)
            sampled_tasks[0].append(pair[0])
            sampled_tasks[1].append(pair[1])
        left = _point(typed_left_axis, statistics.median(sampled_tasks[0]))
        right = _point(typed_right_axis, statistics.median(sampled_tasks[1]))
        for key in ("T", "H", "score"):
            draws[f"left_{key}"].append(left[key])
            draws[f"right_{key}"].append(right[key])
            draws[f"delta_{key}"].append(left[key] - right[key])
    return {
        "ci95": interval95(draws["delta_score"]),
        "axis_ci95": {
            key: {
                side: interval95(draws[f"{side}_{key}"])
                for side in ("left", "right", "delta")
            }
            for key in ("T", "H")
        },
    }


def task_bootstrap(
    rows: list[tuple[int, int, int]], nlabels: int, replicates: int, seed: int
) -> dict[str, float]:
    rng = random.Random(seed)
    deltas = []
    for _ in range(replicates):
        left, right = _task_f1(rows, nlabels, rng)
        deltas.append(left - right)
    return interval95(deltas)


def strata_bootstrap(
    strata: list[list[tuple[int, int]]],
    replicates: int,
    seed: int,
    kinds: list[str] | None = None,
) -> dict[str, float]:
    """Paired item draws within each stratum.

    Without `kinds`: delta of the summed correct counts (public 231, in items). With
    `kinds` (the type of each type x language stratum): delta of the mean over types of
    the mean over strata of accuracy (the mlx-diag type macro).
    """
    rng = random.Random(seed)
    deltas = []
    for _ in range(replicates):
        sampled = []
        for rows in strata:
            left = right = 0
            for _row in rows:
                a, b = rows[rng.randrange(len(rows))]
                left += a
                right += b
            sampled.append((left, right, len(rows)))
        if kinds is None:
            deltas.append(sum(a - b for a, b, _n in sampled))
            continue
        by_type: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for kind, (a, b, n) in zip(kinds, sampled):
            by_type[kind].append((a / n, b / n))
        deltas.append(
            statistics.mean(
                statistics.mean(v[0] for v in cells) for cells in by_type.values()
            )
            - statistics.mean(
                statistics.mean(v[1] for v in cells) for cells in by_type.values()
            )
        )
    return interval95(deltas)


def public_strata(
    left: dict[str, dict[str, Any]], right: dict[str, dict[str, Any]], exclude: set[str]
) -> list[list[tuple[int, int]]]:
    tiers: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for item_id in sorted(left):
        if item_id not in exclude:
            tiers[left[item_id]["tier"]].append(
                (int(left[item_id]["correct"]), int(right[item_id]["correct"]))
            )
    return [tiers[tier] for tier in sorted(tiers)]


def mlx_strata(
    left: dict[str, dict[str, Any]], right: dict[str, dict[str, Any]], exclude: set[str]
) -> tuple[list[list[tuple[int, int]]], list[str]]:
    cells: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    for item_id in sorted(left):
        if item_id not in exclude:
            row = left[item_id]
            cells[(row["type"], row["language"])].append(
                (int(row["correct"]), int(right[item_id]["correct"]))
            )
    keys = sorted(cells)
    return [cells[key] for key in keys], [key[0] for key in keys]


def contamination_units(
    gold: dict[str, dict[str, Any]],
    left: dict[str, dict[str, Any]],
    right: dict[str, dict[str, Any]],
    flagged_ids: set[str],
    only_task: str | None = None,
) -> list[tuple[list[tuple[int, int]], list[tuple[int, int]]]]:
    """Per task with flagged items: (flagged, unflagged) paired correctness."""
    by_task: dict[str, tuple[list[tuple[int, int]], list[tuple[int, int]]]] = (
        defaultdict(lambda: ([], []))
    )
    for item_id in sorted(gold):
        task = gold[item_id]["task"]
        if only_task is None or task == only_task:
            pair = (int(left[item_id]["correct"]), int(right[item_id]["correct"]))
            by_task[task][0 if item_id in flagged_ids else 1].append(pair)
    return [rows for _task, rows in sorted(by_task.items()) if rows[0] and rows[1]]


def contamination_point(
    units: list[tuple[list[tuple[int, int]], list[tuple[int, int]]]],
) -> dict[str, float]:
    n_flagged = sum(len(flagged_rows) for flagged_rows, _u in units)
    out = {}
    for side, name in ((0, "left"), (1, "right")):
        flagged_accuracy = sum(sum(p[side] for p in f) for f, _u in units) / n_flagged
        expected = sum(
            len(f) / n_flagged * sum(p[side] for p in u) / len(u) for f, u in units
        )
        out[f"{name}_flagged_accuracy"] = flagged_accuracy
        out[f"{name}_unflagged_accuracy"] = expected
        out[f"{name}_gap"] = flagged_accuracy - expected
    out["delta_flagged"] = out["left_flagged_accuracy"] - out["right_flagged_accuracy"]
    out["delta_unflagged"] = (
        out["left_unflagged_accuracy"] - out["right_unflagged_accuracy"]
    )
    out["difference_in_differences"] = out["left_gap"] - out["right_gap"]
    return out


def contamination_bootstrap(
    units: list[tuple[list[tuple[int, int]], list[tuple[int, int]]]],
    replicates: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    rng = random.Random(seed)
    draws: dict[str, list[float]] = defaultdict(list)
    for _ in range(replicates):
        sampled = [
            (
                [f[rng.randrange(len(f))] for _ in f],
                [u[rng.randrange(len(u))] for _ in u],
            )
            for f, u in units
        ]
        point = contamination_point(sampled)
        for key in (
            "left_gap",
            "right_gap",
            "delta_flagged",
            "difference_in_differences",
        ):
            draws[key].append(point[key])
    return {key: interval95(values) for key, values in draws.items()}


# ---------------------------------------------------------------- run


def _work(job: tuple[str, str, str, tuple]) -> tuple[str, str, str, Any]:
    kind, key, variant, payload = job
    worker = {
        "v3": v3_bootstrap,
        "task": task_bootstrap,
        "public": strata_bootstrap,
        "mlx": strata_bootstrap,
        "contamination": contamination_bootstrap,
    }[kind]
    return kind, key, variant, worker(*payload)


def pairs_from_tiers(tiers: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"tier": tier, "left": cfg["candidate"], "right": right, "relation": relation}
        for tier, cfg in tiers.items()
        for relation in ("own_1_0", "peers", "internal_peers")
        for right in cfg.get(relation, [])
    ]


def pair_key(left: str, right: str) -> str:
    return f"{left} - {right}"


def ci_status(ci: dict[str, float]) -> str:
    if ci["low"] > 0:
        return "above 0"
    if ci["high"] < 0:
        return "below 0"
    return "includes 0"


def ranks(values: dict[str, float]) -> list[list[str]]:
    """Models from best to worst; equal values share a place."""
    order: list[list[str]] = []
    for name in sorted(values, key=lambda n: (-values[n], n)):
        if order and values[order[-1][0]] == values[name]:
            order[-1].append(name)
        else:
            order.append([name])
    return order


def mean_gold_probability(
    outcomes: dict[str, dict[str, Any]], ids: set[str]
) -> float | None:
    probs = [
        outcomes[i]["gold_probability"]
        for i in ids
        if outcomes[i]["choice"] is not None
    ]
    return statistics.mean(probs) if probs else None


def bootstrap_jobs(
    pairs: list[dict[str, str]],
    models: dict[str, dict[str, Any]],
    gold: dict[str, Any],
    exclude: dict[str, set[str]],
    focus: str,
) -> list[tuple[str, str, str, tuple]]:
    none = {panel: set() for panel in SCORED_PANELS}
    jobs = []
    for pair in pairs:
        key = pair_key(pair["left"], pair["right"])
        left, right = models[pair["left"]], models[pair["right"]]
        for variant, excl, seed in (
            ("full", none, PAIRED_SEED),
            ("full_seed2", none, SECOND_SEED),
            ("reduced", exclude, PAIRED_SEED),
        ):
            families = pair_families(
                gold["typed-final"], left["typed"], right["typed"], excl["typed-final"]
            )
            tasks = pair_tasks(gold["css15"], left["css"], right["css"], excl["css15"])
            jobs.append(
                ("v3", key, variant, (families, tasks, PAIRED_REPLICATES, seed))
            )
            if variant == "full_seed2":
                continue
            rows, nlabels = tasks[focus]
            jobs.append(
                ("task", key, variant, (rows, nlabels, PAIRED_REPLICATES, seed))
            )
            strata = public_strata(left["public"], right["public"], excl["public231"])
            jobs.append(("public", key, variant, (strata, PAIRED_REPLICATES, seed)))
            if left["mlx"] is not None and right["mlx"] is not None:
                strata, kinds = mlx_strata(left["mlx"], right["mlx"], excl["mlx-diag"])
                jobs.append(
                    ("mlx", key, variant, (strata, PAIRED_REPLICATES, seed, kinds))
                )
        for variant, only in (("all_flagged", None), ("focus_task", focus)):
            units = contamination_units(
                gold["css15"], left["css"], right["css"], exclude["css15"], only
            )
            jobs.append(
                ("contamination", key, variant, (units, PAIRED_REPLICATES, PAIRED_SEED))
            )
    return jobs


def pair_entry(
    pair: dict[str, str],
    models: dict[str, dict[str, Any]],
    values: dict[str, dict[str, Any]],
    boot: dict[str, dict[str, Any]],
    gold: dict[str, Any],
    exclude: dict[str, set[str]],
    focus: str,
) -> dict[str, Any]:
    lv, rv = values[pair["left"]], values[pair["right"]]
    entry: dict[str, Any] = {**pair, "v3": {}}
    for variant, which in (
        ("full", "full"),
        ("full_seed2", "full"),
        ("reduced", "reduced"),
    ):
        point: dict[str, Any] = {
            side: {"T": v[which]["T"], "H": v[which]["H"], "score": v[which]["v3"]}
            for side, v in (("left", lv), ("right", rv))
        }
        point["delta"] = {
            k: point["left"][k] - point["right"][k] for k in ("T", "H", "score")
        }
        entry["v3"][variant] = {"point": point, **boot["v3"][variant]}
    entry["focus_task"] = {
        variant: {
            "left": lv[variant]["tasks"][focus],
            "right": rv[variant]["tasks"][focus],
            "delta": lv[variant]["tasks"][focus] - rv[variant]["tasks"][focus],
            "ci95": boot["task"][variant],
        }
        for variant in ("full", "reduced")
    }
    entry["public231"] = {
        variant: {
            "left": lv[variant]["public231"]["correct"],
            "right": rv[variant]["public231"]["correct"],
            "items": lv[variant]["public231"]["items"],
            "delta": lv[variant]["public231"]["correct"]
            - rv[variant]["public231"]["correct"],
            "ci95": boot["public"][variant],
        }
        for variant in ("full", "reduced")
    }
    entry["mlx"] = None
    if "mlx" in boot:
        entry["mlx"] = {
            variant: {
                "left": lv[variant]["mlx"]["type_macro_accuracy"],
                "right": rv[variant]["mlx"]["type_macro_accuracy"],
                "delta": lv[variant]["mlx"]["type_macro_accuracy"]
                - rv[variant]["mlx"]["type_macro_accuracy"],
                "ci95": boot["mlx"][variant],
            }
            for variant in ("full", "reduced")
        }
    entry["contamination"] = {}
    for variant, only in (("all_flagged", None), ("focus_task", focus)):
        units = contamination_units(
            gold["css15"],
            models[pair["left"]]["css"],
            models[pair["right"]]["css"],
            exclude["css15"],
            only,
        )
        entry["contamination"][variant] = {
            "flagged_items": sum(len(f) for f, _u in units),
            "unflagged_items": sum(len(u) for _f, u in units),
            **contamination_point(units),
            "ci95": boot["contamination"][variant],
        }
    cis = {
        "v3": {v: entry["v3"][v]["ci95"] for v in ("full", "full_seed2", "reduced")},
        "H": {
            v: entry["v3"][v]["axis_ci95"]["H"]["delta"]
            for v in ("full", "full_seed2", "reduced")
        },
        "T": {
            v: entry["v3"][v]["axis_ci95"]["T"]["delta"]
            for v in ("full", "full_seed2", "reduced")
        },
        focus: {v: entry["focus_task"][v]["ci95"] for v in ("full", "reduced")},
        "public231": {v: entry["public231"][v]["ci95"] for v in ("full", "reduced")},
    }
    if entry["mlx"] is not None:
        cis["mlx"] = {v: entry["mlx"][v]["ci95"] for v in ("full", "reduced")}
    entry["ci_status"] = {
        metric: {v: ci_status(ci) for v, ci in by_variant.items()}
        for metric, by_variant in cis.items()
    }
    return entry


def reproduce(
    items: list[dict[str, str]], out_pairs: dict[str, Any], models: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[str]]:
    reproduction, problems = [], []
    for item in items:
        key = pair_key(item["left"], item["right"])
        stored = json.loads(Path(item["stored"]).read_text(encoding="utf-8"))
        mine = out_pairs[key]["v3"]["full"]
        left_runs = {models[item["left"]]["predictions_sha256"]["typed"]} | {
            other["typed"] for other in models[item["left"]]["equivalent_runs"].values()
        }
        checks = {
            "point": stored["point"] == mine["point"],
            "ci95": stored["ci95"] == mine["ci95"],
            "axis_ci95": stored["axis_ci95"] == mine["axis_ci95"],
            "replicates_seed": [stored["replicates"], stored["seed"]]
            == [PAIRED_REPLICATES, PAIRED_SEED],
            "left_predictions": stored["predictions_sha256"]["left"]["typed"]
            in left_runs,
            "right_predictions": stored["predictions_sha256"]["right"]["typed"]
            == models[item["right"]]["predictions_sha256"]["typed"],
        }
        reproduction.append(
            {
                "pair": key,
                "stored": item["stored"],
                "stored_sha256": sha_file(Path(item["stored"])),
                "match": all(checks.values()),
                "checks": checks,
            }
        )
        if not all(checks.values()):
            problems.append(f"{key}: {item['stored']} not reproduced: {checks}")
    return reproduction, problems


def tier_summary(
    cfg: dict[str, Any], values: dict[str, Any], out_pairs: dict[str, Any], focus: str
) -> dict[str, Any]:
    names = [
        cfg["candidate"],
        *cfg.get("own_1_0", []),
        *cfg.get("peers", []),
        *cfg.get("internal_peers", []),
    ]
    out: dict[str, Any] = {"models": names, "ranks": {}, "rules": {}}
    candidate = cfg["candidate"]
    for variant in ("full", "reduced"):
        metrics = {
            "v3": {n: values[n][variant]["v3"] for n in names},
            "H": {n: values[n][variant]["H"] for n in names},
            focus: {n: values[n][variant]["tasks"][focus] for n in names},
            "public231": {n: values[n][variant]["public231"]["correct"] for n in names},
            "mlx": {
                n: values[n][variant]["mlx"]["type_macro_accuracy"]
                for n in names
                if values[n][variant]["mlx"] is not None
            },
        }
        for metric, table in metrics.items():
            out["ranks"].setdefault(metric, {})[variant] = ranks(table)
        best = max(cfg["threshold_pool"], key=lambda n: values[n][variant]["v3"])
        threshold = 0.9 * values[best][variant]["v3"]
        best_h = out_pairs[pair_key(candidate, best)]["v3"][variant]["axis_ci95"]["H"][
            "delta"
        ]
        out["rules"][variant] = {
            "best_peer": best,
            "threshold_v3": threshold,
            "candidate_v3": values[candidate][variant]["v3"],
            "meets_threshold": values[candidate][variant]["v3"] >= threshold,
            "v3_vs_own_1_0": {
                right: ci_status(
                    out_pairs[pair_key(candidate, right)]["v3"][variant]["ci95"]
                )
                for right in cfg.get("own_1_0", [])
            },
            "H_vs_best_peer_ci95": best_h,
            "H_significantly_below_best_peer": best_h["high"] < 0,
        }
    out["rank_changes"] = [
        metric for metric, by in out["ranks"].items() if by["full"] != by["reduced"]
    ]
    return out


def run(args: argparse.Namespace) -> int:
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    flagged_doc = json.loads(args.flagged.read_text(encoding="utf-8"))
    gold = load_panels(
        Path(spec.get("panel_root", str(panels.DEFAULT_ROOT))), Path(spec["mlx_panel"])
    )
    if flagged_doc["gold_sha256"] != gold["sha256"]:
        raise ValueError("flagged ids were collected against different panels")
    exclude = {
        panel: set(flagged_doc["panels"].get(panel, [])) for panel in SCORED_PANELS
    }
    for panel, ids in exclude.items():
        if ids - set(gold[panel]):
            raise ValueError(f"{panel}: flagged ids outside the panel")
    none = {panel: set() for panel in SCORED_PANELS}
    focus = spec["focus_task"]
    affected = {gold["css15"][i]["task"] for i in exclude["css15"]}
    unflagged_same_tasks = {
        i
        for i, row in gold["css15"].items()
        if row["task"] in affected and i not in exclude["css15"]
    }

    models, values, problems = {}, {}, []
    for name, cfg in spec["models"].items():
        models[name] = load_model(name, cfg, gold)
        full = model_values(models[name], gold, none)
        problems += check_full(name, models[name], full, gold)
        values[name] = {
            "full": full,
            "reduced": model_values(models[name], gold, exclude),
        }
        print(
            f"{name}: v3 {full['v3']:.3f} -> {values[name]['reduced']['v3']:.3f}",
            file=sys.stderr,
            flush=True,
        )

    pairs = pairs_from_tiers(spec["tiers"])
    jobs = bootstrap_jobs(pairs, models, gold, exclude, focus)
    print(
        f"{len(jobs)} bootstrap jobs on {args.jobs} processes",
        file=sys.stderr,
        flush=True,
    )
    with multiprocessing.get_context("fork").Pool(args.jobs) as pool:
        results = pool.map(_work, jobs, chunksize=1)
    boot: dict[str, dict[str, dict[str, Any]]] = defaultdict(lambda: defaultdict(dict))
    for kind, key, variant, result in results:
        boot[key][kind][variant] = result

    out_pairs = {
        pair_key(pair["left"], pair["right"]): pair_entry(
            pair,
            models,
            values,
            boot[pair_key(pair["left"], pair["right"])],
            gold,
            exclude,
            focus,
        )
        for pair in pairs
    }
    reproduction, reproduction_problems = reproduce(
        spec.get("reproduce", []), out_pairs, models
    )
    problems += reproduction_problems
    out_tiers = {
        tier: tier_summary(cfg, values, out_pairs, focus)
        for tier, cfg in spec["tiers"].items()
    }

    changes = []
    for key, entry in out_pairs.items():
        for metric, status in entry["ci_status"].items():
            if status["full"] != status["reduced"]:
                changes.append({"pair": key, "metric": metric, **status})
    for tier, tier_out in out_tiers.items():
        full_rules, reduced_rules = (
            tier_out["rules"]["full"],
            tier_out["rules"]["reduced"],
        )
        for rule in (
            "best_peer",
            "meets_threshold",
            "v3_vs_own_1_0",
            "H_significantly_below_best_peer",
        ):
            if full_rules[rule] != reduced_rules[rule]:
                changes.append(
                    {
                        "tier": tier,
                        "rule": rule,
                        "full": full_rules[rule],
                        "reduced": reduced_rules[rule],
                    }
                )
        for metric in tier_out["rank_changes"]:
            changes.append(
                {
                    "tier": tier,
                    "rank": metric,
                    "full": tier_out["ranks"][metric]["full"],
                    "reduced": tier_out["ranks"][metric]["reduced"],
                }
            )

    result = {
        "schema": SCHEMA,
        "label": LABEL,
        "scope": "stored sealed predictions, CPU only; flagged items removed for every model",
        "flagged": {
            "sha256": sha_file(args.flagged),
            "groups": flagged_doc["groups"],
            "groups_by_pool": flagged_doc["groups_by_pool"],
            "counts": flagged_doc["counts"],
            "ids_sha256": flagged_doc["ids_sha256"],
            "unscored_by_role": flagged_doc["unscored_by_role"],
            "strata": flagged_doc["strata"],
            "methods": flagged_doc["methods"],
        },
        "gold_sha256": gold["sha256"],
        "focus_task": focus,
        "bootstrap": {
            "replicates": PAIRED_REPLICATES,
            "seed": PAIRED_SEED,
            "second_seed": SECOND_SEED,
            "v3": "jev_arena.compare_v3 replicate loop (typed groups within family; CSS tasks then items)",
            "focus_task": "paired items within the task; fixed label universe",
            "public231": "paired items within tier; delta in items",
            "mlx": "paired items within type x language; delta of the type macro accuracy",
            "contamination": "paired items within task and flag; unflagged accuracy weighted by flagged items per task",
        },
        "models": {
            name: {
                "run": model["run"],
                "predictions_sha256": model["predictions_sha256"],
                "equivalent_runs": model["equivalent_runs"],
                "mlx_stored": model.get("mlx_stored"),
                "full": values[name]["full"],
                "reduced": values[name]["reduced"],
                "flagged_outcomes": {
                    "css15_correct": sum(
                        model["css"][i]["correct"] for i in exclude["css15"]
                    ),
                    "css15_valid": sum(
                        model["css"][i]["choice"] is not None for i in exclude["css15"]
                    ),
                    "public231_correct": sum(
                        model["public"][i]["correct"] for i in exclude["public231"]
                    ),
                    "mlx_correct": (
                        sum(model["mlx"][i]["correct"] for i in exclude["mlx-diag"])
                        if model["mlx"] is not None
                        else None
                    ),
                    "css15_gold_probability_flagged": mean_gold_probability(
                        model["css"], exclude["css15"]
                    ),
                    "css15_gold_probability_unflagged_same_tasks": mean_gold_probability(
                        model["css"], unflagged_same_tasks
                    ),
                },
            }
            for name, model in models.items()
        },
        "pairs": out_pairs,
        "tiers": out_tiers,
        "reproduction": reproduction,
        "changes": changes,
        "problems": problems,
    }
    sha = write_json(args.output, result)
    args.output.with_suffix(".md").write_text(render(result), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": sha,
                "problems": len(problems),
                "reproduced": sum(r["match"] for r in reproduction),
                "stored_files": len(reproduction),
                "changes": len(changes),
            }
        )
    )
    return 1 if problems else 0


# ---------------------------------------------------------------- markdown


def _ci(ci: dict[str, float], digits: int) -> str:
    return f"[{ci['low']:+.{digits}f}, {ci['high']:+.{digits}f}]"


def render(result: dict[str, Any]) -> str:
    focus = result["focus_task"]
    flagged_doc = result["flagged"]
    lines = [
        f"# Overlap effects ({result['label']})",
        "",
        f"Flagged items: {flagged_doc['counts']} from {flagged_doc['groups']} training groups; "
        f"not in any frozen panel: {flagged_doc['unscored_by_role']}.",
        "",
        "## Models (full -> without flagged items)",
        "",
        f"| Model | v3 | H | {focus} F1 | public 231 | mlx-diag |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, model in result["models"].items():
        f, r = model["full"], model["reduced"]
        mlx = (
            f"{f['mlx']['type_macro_accuracy']:.4f} -> {r['mlx']['type_macro_accuracy']:.4f}"
            if f["mlx"] is not None
            else "n/a"
        )
        lines.append(
            f"| {name} | {f['v3']:.3f} -> {r['v3']:.3f} | {f['H']:.4f} -> {r['H']:.4f} | "
            f"{f['tasks'][focus]:.4f} -> {r['tasks'][focus]:.4f} | "
            f"{f['public231']['correct']}/{f['public231']['items']} -> "
            f"{r['public231']['correct']}/{r['public231']['items']} | {mlx} |"
        )
    lines += [
        "",
        "## Pairs: candidate minus comparator (full; second seed; without flagged items)",
        "",
        "| Pair | Δ v3 | Δ H | Δ T |",
        "| --- | --- | --- | --- |",
    ]
    for key, entry in result["pairs"].items():
        cells = []
        for metric, digits in (("score", 2), ("H", 3), ("T", 3)):
            parts = []
            for variant in ("full", "full_seed2", "reduced"):
                v3 = entry["v3"][variant]
                ci = (
                    v3["ci95"]
                    if metric == "score"
                    else v3["axis_ci95"][metric]["delta"]
                )
                parts.append(
                    f"{v3['point']['delta'][metric]:+.{digits}f} {_ci(ci, digits)}"
                )
            cells.append("; ".join(parts))
        lines.append(f"| {key} | " + " | ".join(cells) + " |")
    lines += [
        "",
        f"| Pair | Δ {focus} F1 (full -> without) | Δ public 231 items | Δ mlx-diag |",
        "| --- | --- | --- | --- |",
    ]
    for key, entry in result["pairs"].items():
        ft, pub, mlx = entry["focus_task"], entry["public231"], entry["mlx"]
        mlx_cell = (
            f"{mlx['full']['delta']:+.4f} {_ci(mlx['full']['ci95'], 4)} -> "
            f"{mlx['reduced']['delta']:+.4f} {_ci(mlx['reduced']['ci95'], 4)}"
            if mlx is not None
            else "n/a"
        )
        lines.append(
            f"| {key} | {ft['full']['delta']:+.3f} {_ci(ft['full']['ci95'], 3)} -> "
            f"{ft['reduced']['delta']:+.3f} {_ci(ft['reduced']['ci95'], 3)} | "
            f"{pub['full']['delta']:+d} {_ci(pub['full']['ci95'], 1)} -> "
            f"{pub['reduced']['delta']:+d} {_ci(pub['reduced']['ci95'], 1)} | {mlx_cell} |"
        )
    lines += [
        "",
        "## Contamination check (CSS15 accuracy: flagged minus same-task unflagged)",
        "",
        "| Pair | scope | candidate gap | comparator gap | difference [95% CI] |",
        "| --- | --- | --- | --- | --- |",
    ]
    for key, entry in result["pairs"].items():
        for variant in ("all_flagged", "focus_task"):
            c = entry["contamination"][variant]
            lines.append(
                f"| {key} | {variant} ({c['flagged_items']} vs {c['unflagged_items']}) | "
                f"{c['left_flagged_accuracy']:.3f} - {c['left_unflagged_accuracy']:.3f} = "
                f"{c['left_gap']:+.3f} | {c['right_flagged_accuracy']:.3f} - "
                f"{c['right_unflagged_accuracy']:.3f} = {c['right_gap']:+.3f} | "
                f"{c['difference_in_differences']:+.3f} "
                f"{_ci(c['ci95']['difference_in_differences'], 3)} |"
            )
    lines += ["", "## Rules and ranks", ""]
    for tier, tier_out in result["tiers"].items():
        for variant in ("full", "reduced"):
            rules = tier_out["rules"][variant]
            lines.append(
                f"- {tier} {variant}: threshold {rules['threshold_v3']:.3f} "
                f"(0.9 x {rules['best_peer']}), candidate {rules['candidate_v3']:.3f}, "
                f"meets {rules['meets_threshold']}; v3 vs own 1.0 {rules['v3_vs_own_1_0']}; "
                f"H vs best peer {_ci(rules['H_vs_best_peer_ci95'], 3)} "
                f"(significantly below: {rules['H_significantly_below_best_peer']})"
            )
        lines.append(f"- {tier} rank changes: {tier_out['rank_changes'] or 'none'}")
    lines += ["", "## Conclusion changes", ""]
    lines += [
        f"- {json.dumps(change, sort_keys=True)}" for change in result["changes"]
    ] or ["- none"]
    lines += [
        "",
        f"Reproduction: {sum(r['match'] for r in result['reproduction'])} of "
        f"{len(result['reproduction'])} stored paired files bit for bit. "
        f"Problems: {result['problems'] or 'none'}.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("flagged")
    one.add_argument("--rescreen", type=Path, required=True)
    one.add_argument("--role", action="append", required=True)
    one.add_argument("--panel-root", type=Path, default=panels.DEFAULT_ROOT)
    one.add_argument(
        "--mlx-panel", type=Path, default=panels.DEFAULT_ROOT / "mlx-diag-v1"
    )
    one.add_argument("--expect-groups", type=int)
    one.add_argument("--output", type=Path, required=True)
    one.set_defaults(func=flagged)
    two = sub.add_parser("run")
    two.add_argument("--spec", type=Path, required=True)
    two.add_argument("--flagged", type=Path, required=True)
    two.add_argument("--output", type=Path, required=True)
    two.add_argument("--jobs", type=int, default=8)
    two.set_defaults(func=run)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
