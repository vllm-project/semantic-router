"""Card-support aggregates for the 9B candidate from stored sealed predictions (CPU, no C1).

    PYTHONPATH=<mirror>/src/training/decision2 python3 card_aggregates_9b.py \
        --spec overlap-spec-9b.json --output OUT

For each candidate - comparator pair of the spec's 9B tier:

- mlx-diag non-English Choice and non-English Noul accuracy (mean over languages; the parts
  a card may show, since the Score part is built from a non-commercial source), per-language
  Noul, and each difference with a paired item bootstrap within language;
- CSS15 (within task) and public 231 (within tier) accuracy on long inputs (at least
  LONG_INPUT_CHARS characters, as in the REPORT.json slices), with a paired item bootstrap;
- the exact per-task CSS15 macro-F1 differences from both REPORT.json files.

5,000 draws, seed 20260927. Outputs are aggregates only: no item ids, text or answers.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from transfer.score import read_jsonl as read_css_jsonl
from v2.eval import panels
from v2.eval.gates import verified
from v2.eval.overlap_effects import (
    css_outcomes,
    load_panels,
    mlx_outcomes,
    public_outcomes,
    strata_bootstrap,
)
from v2.eval.same_panel import (
    LONG_INPUT_CHARS,
    PAIRED_REPLICATES,
    PAIRED_SEED,
    input_chars,
    prediction_path,
    read_jsonl,
    sha_file,
    write_json,
)


def mlx_for(run: Path, gold: dict[str, Any]) -> dict[str, dict[str, Any]]:
    path = prediction_path(run, "mlx-diag")
    stored = json.loads((run / "mlx-diag.score.json").read_text(encoding="utf-8"))
    if sha_file(path) != stored["predictions_sha256"]:
        raise ValueError(f"{run}: mlx-diag predictions differ from the stored score")
    return mlx_outcomes(
        gold["mlx-diag"], gold["mlx-prompts"], {r["id"]: r for r in read_jsonl(path)}
    )


def cells(
    outcomes: dict[str, dict[str, Any]], kind: str, languages: set[str] | None
) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    for item_id in sorted(outcomes):
        row = outcomes[item_id]
        if row["type"] == kind and row["language"] != "en":
            if languages is None or row["language"] in languages:
                out[row["language"]].append(item_id)
    return dict(sorted(out.items()))


def mlx_part(
    left, right, kind: str, languages: set[str] | None = None
) -> dict[str, Any]:
    groups = cells(left, kind, languages)
    strata = [
        [(int(left[i]["correct"]), int(right[i]["correct"])) for i in ids]
        for ids in groups.values()
    ]
    acc = {
        side: statistics.mean(
            sum(o[i]["correct"] for i in ids) / len(ids) for ids in groups.values()
        )
        for side, o in (("left", left), ("right", right))
    }
    return {
        "languages": list(groups),
        "items": sum(len(ids) for ids in groups.values()),
        "left": acc["left"],
        "right": acc["right"],
        "delta": acc["left"] - acc["right"],
        "ci95": strata_bootstrap(
            strata, PAIRED_REPLICATES, PAIRED_SEED, kinds=[kind] * len(strata)
        ),
    }


def long_part(left, right, long_ids: set[str], key: str) -> dict[str, Any]:
    strata: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for item_id in sorted(long_ids & set(left)):
        strata[left[item_id][key]].append(
            (int(left[item_id]["correct"]), int(right[item_id]["correct"]))
        )
    rows = [strata[k] for k in sorted(strata)]
    n = sum(len(r) for r in rows)
    ci = strata_bootstrap(rows, PAIRED_REPLICATES, PAIRED_SEED)
    left_correct = sum(a for r in rows for a, _ in r)
    right_correct = sum(b for r in rows for _, b in r)
    return {
        "items": n,
        "left_correct": left_correct,
        "right_correct": right_correct,
        "delta_items": left_correct - right_correct,
        "delta_items_ci95": ci,
        "delta_accuracy": (left_correct - right_correct) / n,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    root = Path(spec["panel_root"])
    gold = load_panels(root, Path(spec["mlx_panel"]))
    css_prompts = {
        r["id"]: r for r in read_jsonl(panels.path(root, "css15", "prompts"))
    }
    pub_prompts = {
        r["id"]: r for r in read_jsonl(panels.path(root, "public231", "prompts"))
    }
    css_long = {i for i, p in css_prompts.items() if input_chars(p) >= LONG_INPUT_CHARS}
    pub_long = {i for i, p in pub_prompts.items() if input_chars(p) >= LONG_INPUT_CHARS}
    tier = spec["tiers"]["9B"]
    names = [
        tier["candidate"],
        *tier["own_1_0"],
        *tier["peers"],
        *tier["internal_peers"],
    ]
    models = {}
    for name in names:
        cfg = spec["models"][name]
        run = Path(cfg["run"])
        css = css_outcomes(gold["css15"], read_css_jsonl(verified(run, "css15")))
        for item_id, row in css.items():
            row["task"] = gold["css15"][item_id]["task"]
        report = json.loads((run / "REPORT.json").read_text(encoding="utf-8"))
        models[name] = {
            "mlx": mlx_for(Path(cfg["mlx_run"]), gold),
            "css": css,
            "public": public_outcomes(run, gold["public231_dir"]),
            "tasks": {
                t: v["macro_f1"] for t, v in report["panels"]["css15"]["tasks"].items()
            },
            "slices": report.get("slices", {}),
        }
    candidate = tier["candidate"]
    left = models[candidate]
    pairs = {}
    for right_name in names[1:]:
        right = models[right_name]
        pairs[f"{candidate} - {right_name}"] = {
            "mlx_non_english_choice": mlx_part(left["mlx"], right["mlx"], "choice"),
            "mlx_non_english_noul": mlx_part(left["mlx"], right["mlx"], "noul"),
            "mlx_noul_by_language": {
                lang: mlx_part(left["mlx"], right["mlx"], "noul", {lang})
                for lang in cells(left["mlx"], "noul", None)
            },
            "css15_long": long_part(left["css"], right["css"], css_long, "task"),
            "public231_long": long_part(
                left["public"], right["public"], pub_long, "tier"
            ),
            "css15_task_macro_f1_delta": {
                t: left["tasks"][t] - right["tasks"][t] for t in sorted(left["tasks"])
            },
        }
    result = {
        "schema": "dev2-9b-card-aggregates/1",
        "label": "post-key same-panel (mlx-diag: development diagnostic)",
        "spec_sha256": sha_file(args.spec),
        "gold_sha256": gold["sha256"],
        "long_input_chars": LONG_INPUT_CHARS,
        "long_items": {"css15": len(css_long), "public231": len(pub_long)},
        "bootstrap": {"replicates": PAIRED_REPLICATES, "seed": PAIRED_SEED},
        "slices": {name: models[name]["slices"] for name in names},
        "pairs": pairs,
    }
    sha = write_json(args.output, result)
    print(json.dumps({"output": str(args.output), "sha256": sha}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
