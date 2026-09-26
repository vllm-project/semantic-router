"""Cross-validated, no-training Kai CSS postprocessing diagnostic.

Fit temperature and low-dimensional class-prior correction on private CAL300
only; apply the frozen result to the disjoint CSS pilot. Model weights and
checkpoint selection remain unchanged. All outputs stay private and are never
valid evidence of unseen-task transfer.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

from transfer.score import macro_f1, read_jsonl, score
from transfer.build import sha_file

TEMPERATURES = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0)
PRIOR_STRENGTHS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5)


def _stable_fold(group_id: str) -> int:
    return int(hashlib.sha256(group_id.encode()).hexdigest()[:8], 16) % 5


def _softmax(values: list[float]) -> list[float]:
    offset = max(values)
    values = [math.exp(value - offset) for value in values]
    total = sum(values)
    return [value / total for value in values]


def _probabilities(prediction: dict, labels: list[str]) -> list[float] | None:
    if prediction.get("invalid_reason") is not None:
        return None
    answer = prediction.get("answers", {}).get("label")
    if not isinstance(answer, dict) or not isinstance(
        answer.get("probabilities"), dict
    ):
        return None
    probs = answer["probabilities"]
    if set(probs) != set(labels) or any(
        type(probs[label]) not in (float, int)
        or not math.isfinite(probs[label])
        or probs[label] < 0
        for label in labels
    ):
        return None
    total = sum(probs.values())
    if abs(total - 1) > 0.02:
        return None
    choice = answer.get("choice")
    if choice not in labels or max(probs.values()) - probs[choice] > 0.02:
        return None
    return [probs[label] / total for label in labels]


def _prior_bias(training: list[tuple[dict, dict]], labels: list[str]) -> list[float]:
    valid = [(gold, _probabilities(pred, labels)) for gold, pred in training]
    valid = [(gold, probs) for gold, probs in valid if probs is not None]
    if not valid:
        raise ValueError("No valid CAL probabilities for source task")
    count = len(valid)
    return [
        math.log(
            (1 + sum(gold["gold"] == label for gold, _ in valid))
            / (count + len(labels))
        )
        - math.log(max(sum(probs[j] for _, probs in valid) / count, 1e-6))
        for j, label in enumerate(labels)
    ]


def _correct(
    probs: list[float],
    *,
    temp: float = 1.0,
    bias: list[float] | None = None,
    strength: float = 0.0,
) -> list[float]:
    return _softmax(
        [
            math.log(max(value, 1e-12)) / temp
            + (strength * bias[j] if bias is not None else 0.0)
            for j, value in enumerate(probs)
        ]
    )


def _replace(
    prediction: dict, labels: list[str], values: list[float], arm: str
) -> dict:
    answer = prediction["answers"]["label"].copy()
    selected = labels[max(range(len(labels)), key=lambda j: values[j])]
    answer.update(
        {
            "probabilities": dict(zip(labels, values)),
            "choice": selected,
            "confidence": max(values),
        }
    )
    return {
        **prediction,
        "answers": {"label": answer},
        "backend": "kai-css-calibration-research",
        "postprocess_arm": arm,
        "research_only": True,
        "release_qualified": False,
    }


def _macro(golds: dict[str, dict], choices: dict[str, str | None]) -> float:
    by_task: dict[str, list[str]] = defaultdict(list)
    for item_id, row in golds.items():
        by_task[row["task"]].append(item_id)
    return statistics.median(
        macro_f1(
            [golds[i]["gold"] for i in ids],
            [choices.get(i) for i in ids],
            golds[ids[0]]["labels"],
        )
        for ids in by_task.values()
    )


def run(
    cal_gold_path: Path,
    cal_pred_path: Path,
    pilot_gold_path: Path,
    pilot_pred_path: Path,
    output: Path,
) -> dict:
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    cal_gold, cal_pred = read_jsonl(cal_gold_path), read_jsonl(cal_pred_path)
    pilot_gold, pilot_pred = read_jsonl(pilot_gold_path), read_jsonl(pilot_pred_path)
    if set(cal_gold) != set(cal_pred) or set(pilot_gold) != set(pilot_pred):
        raise ValueError("CAL/pilot prediction IDs differ from gold")
    for gold, pred in [(cal_gold, cal_pred), (pilot_gold, pilot_pred)]:
        for item_id, row in gold.items():
            if pred[item_id].get("source_input_sha256") != row["input_sha256"]:
                raise ValueError("Input SHA differs from original gold-free inference")
    by_task: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    for item_id, gold in cal_gold.items():
        by_task[gold["task"]].append((gold, cal_pred[item_id]))
    if {task: len(rows) for task, rows in by_task.items()} != {
        "css_pilot:discourse": 100,
        "css_pilot:implicit_hate": 100,
        "css_pilot:semeval_stance": 100,
    }:
        raise ValueError("CAL source task quotas differ")
    labels_by_task = {task: rows[0][0]["labels"] for task, rows in by_task.items()}
    for task, rows in by_task.items():
        if any(gold["labels"] != labels_by_task[task] for gold, _ in rows):
            raise ValueError("CAL task label order changed")
    temp_cv: dict[float, float] = {}
    prior_cv: dict[float, float] = {}
    for temp in TEMPERATURES:
        losses = []
        for task, rows in by_task.items():
            labels = labels_by_task[task]
            for gold, pred in rows:
                probs = _probabilities(pred, labels)
                if probs is not None:
                    adjusted = _correct(probs, temp=temp)
                    losses.append(
                        -math.log(max(adjusted[labels.index(gold["gold"])], 1e-12))
                    )
        temp_cv[temp] = statistics.mean(losses)
    # Temperature is fit on CAL NLL and cannot change argmax. Prior correction
    # selects its strength through five lineage-disjoint CAL folds.
    for strength in PRIOR_STRENGTHS:
        choices = {}
        for task, rows in by_task.items():
            labels = labels_by_task[task]
            for fold in range(5):
                training = [
                    (g, p) for g, p in rows if _stable_fold(g["group_id"]) != fold
                ]
                heldout = [
                    (g, p) for g, p in rows if _stable_fold(g["group_id"]) == fold
                ]
                bias = _prior_bias(training, labels)
                for gold, pred in heldout:
                    probs = _probabilities(pred, labels)
                    if probs is not None:
                        adjusted = _correct(probs, bias=bias, strength=strength)
                        choices[gold["id"]] = labels[
                            max(range(len(labels)), key=lambda j: adjusted[j])
                        ]
        prior_cv[strength] = _macro(cal_gold, choices)
    chosen_temp = min(TEMPERATURES, key=lambda value: (temp_cv[value], abs(value - 1)))
    chosen_prior = sorted(PRIOR_STRENGTHS, key=lambda value: (-prior_cv[value], value))[
        0
    ]
    bias_by_task = {
        task: _prior_bias(rows, labels_by_task[task]) for task, rows in by_task.items()
    }
    adjusted: dict[str, list[dict]] = {"temperature": [], "prior": []}
    for item_id, gold in pilot_gold.items():
        pred = pilot_pred[item_id]
        task = "css_pilot:" + gold["task"]
        labels = gold["labels"]
        if labels != labels_by_task[task]:
            raise ValueError("CAL/pilot task label order differs")
        probs = _probabilities(pred, labels)
        for arm in adjusted:
            if probs is None:
                adjusted[arm].append(
                    {
                        **pred,
                        "postprocess_arm": arm,
                        "research_only": True,
                        "release_qualified": False,
                    }
                )
            else:
                values = (
                    _correct(probs, temp=chosen_temp)
                    if arm == "temperature"
                    else _correct(probs, bias=bias_by_task[task], strength=chosen_prior)
                )
                adjusted[arm].append(_replace(pred, labels, values, arm))
    stage = output.with_name(output.name + ".pending")
    if stage.exists():
        raise FileExistsError(stage)
    stage.mkdir(parents=True, mode=0o700)
    for arm, rows in adjusted.items():
        with (stage / f"{arm}.predictions.jsonl").open("w") as stream:
            for row in rows:
                stream.write(
                    json.dumps(
                        row, ensure_ascii=False, separators=(",", ":"), allow_nan=False
                    )
                    + "\n"
                )
        (stage / f"{arm}.score.json").write_text(
            json.dumps(
                score(pilot_gold_path, stage / f"{arm}.predictions.jsonl"),
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    report = {
        "schema_version": "kai06b-css-postprocess-diagnostic/1",
        "private_only": True,
        "release_qualified": False,
        "cal_gold_sha256": sha_file(cal_gold_path),
        "cal_predictions_sha256": sha_file(cal_pred_path),
        "pilot_gold_sha256": sha_file(pilot_gold_path),
        "pilot_predictions_sha256": sha_file(pilot_pred_path),
        "temperature_grid_cal_nll": {str(k): v for k, v in temp_cv.items()},
        "prior_grid_five_fold_group_cv_median_task_macro_f1": {
            str(k): v for k, v in prior_cv.items()
        },
        "chosen_temperature": chosen_temp,
        "chosen_prior_strength": chosen_prior,
        "bias_by_task": bias_by_task,
        "baseline_pilot": score(pilot_gold_path, pilot_pred_path)["roles"]["pilot"],
        "temperature_pilot": json.loads((stage / "temperature.score.json").read_text())[
            "roles"
        ]["pilot"],
        "prior_pilot": json.loads((stage / "prior.score.json").read_text())["roles"][
            "pilot"
        ],
        "outputs_sha256": {p.name: sha_file(p) for p in stage.iterdir() if p.is_file()},
        "scope": "CAL300 source-matched development only; no FINAL; cannot establish unseen-task transfer",
    }
    (stage / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    os.replace(stage, output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cal-gold", type=Path, required=True)
    parser.add_argument("--cal-pred", type=Path, required=True)
    parser.add_argument("--pilot-gold", type=Path, required=True)
    parser.add_argument("--pilot-pred", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(
        args.cal_gold, args.cal_pred, args.pilot_gold, args.pilot_pred, args.output
    )
    print(
        json.dumps(
            {
                "chosen_temperature": report["chosen_temperature"],
                "chosen_prior_strength": report["chosen_prior_strength"],
                "baseline": report["baseline_pilot"]["median_task_macro_f1_all"],
                "temperature": report["temperature_pilot"]["median_task_macro_f1_all"],
                "prior": report["prior_pilot"]["median_task_macro_f1_all"],
            }
        )
    )


if __name__ == "__main__":
    main()
