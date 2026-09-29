import random
import statistics

from transfer.score import macro_f1
from v2.eval.htdev2 import score, validate


def gold_row(task, i, gold, labels):
    return {
        "id": f"{task}/{i}",
        "task": task,
        "gold": gold,
        "labels": labels,
        "input_sha256": f"h{task}{i}",
    }


def prediction(row, choice):
    probs = {label: (1.0 if label == choice else 0.0) for label in row["labels"]}
    return {
        "id": row["id"],
        "source_input_sha256": row["input_sha256"],
        "answers": {
            "label": {"type": "choice", "choice": choice, "probabilities": probs}
        },
    }


def test_report_is_task_mean_and_median_of_css_macro_f1():
    rng = random.Random(1)
    gold, preds = [], {}
    for task, labels in (("a", ["x", "y"]), ("b", ["p", "q", "r"]), ("c", ["u", "v"])):
        for i in range(60):
            row = gold_row(task, i, labels[i % len(labels)], labels)
            gold.append(row)
            if i % 7:
                preds[row["id"]] = prediction(row, rng.choice(labels))
    result = score.report(gold, preds, replicates=50)
    per_task = {}
    for task in ("a", "b", "c"):
        rows = [g for g in gold if g["task"] == task]
        choices = [
            preds[g["id"]]["answers"]["label"]["choice"] if g["id"] in preds else None
            for g in rows
        ]
        per_task[task] = macro_f1([g["gold"] for g in rows], choices, rows[0]["labels"])
    assert abs(result["H_dev2"] - statistics.fmean(per_task.values())) < 1e-12
    assert abs(result["H_dev2_median"] - statistics.median(per_task.values())) < 1e-12
    assert result["valid"] < result["items"] == 180
    assert result["bootstrap"]["H_dev2"]["sd"] > 0


def model(key, tier, h_formal, h_dev2, mean3, lineage="x", group="peer"):
    return {
        "key": key,
        "tier": tier,
        "group": group,
        "lineage": lineage,
        "formal": {"H_formal": h_formal, "css15_task_mean": h_formal, "v3": 50.0},
        "htdev2": {"H_dev2": h_dev2, "H_dev2_median": h_dev2, "tasks": {"t": h_dev2}},
        "pilot": {"H_mean3": mean3, "H_pilot": mean3, "tasks": {"s": mean3}},
        "htdev1": None,
    }


def test_analyze_prefers_the_panel_that_orders_formal_h():
    rng = random.Random(2)
    models = []
    for tier in ("0.6B", "4B"):
        for k in range(8):
            h = 0.3 + 0.03 * k + (0.2 if tier == "4B" else 0.0)
            models.append(
                model(
                    f"{tier}-{k}",
                    tier,
                    h,
                    h + rng.gauss(0, 0.003),
                    h + rng.gauss(0, 0.05),
                )
            )
    out = validate.analyze({"models": models}, {}, draws=200)
    decision = out["primary"]["decision"]
    assert decision["H_dev2_agreement"] > decision["H_mean3_agreement"]
    assert decision["p_better"] >= 0.9 and decision["passes"]
    assert out["screen_band"]["band"] >= 0.005
