from __future__ import annotations

import json
import statistics
import tempfile
import unittest
from pathlib import Path

from v2.eval.htdev import score
from v2.eval.sealed.score import input_digest

CHOICE = {"type": "choice", "instructions": "?", "criteria": {"x": "X", "y": "Y"}}
NOUL = {"type": "noul", "instructions": "?", "criteria": {"true": "t", "false": "f"}}
LEVELS = {"type": "score", "instructions": "?", "criteria": ["lo", "mid", "hi"]}


def item(i: int, task: str, question: dict, value) -> dict:
    record = {"type": question["type"], "value": value, "semantic_value": value}
    if question["type"] == "choice":
        record["label_to_semantic"] = {k: k for k in question["criteria"]}
    return {
        "id": f"htdev-{task}-{i}",
        "task": task,
        "source": task.split("/")[0],
        "split": "test",
        "source_item_id": str(i),
        "group_id": f"g{i // 2}",
        "language": "en",
        "input_chars": 50,
        "long": False,
        "state": {"text": f"s{i}"},
        "questions": {"decision": question},
        "gold": {"decision": record},
    }


def answer(question: dict, value):
    if value is None:
        return {"type": question["type"], "choice": "nonsense"}
    if question["type"] == "choice":
        return {"type": "choice", "choice": value}
    if question["type"] == "noul":
        return {"type": "noul", "noul": 0.9 if value else 0.1}
    return {"type": "score", "score": value}


def prediction(row: dict, value) -> dict:
    question = row["questions"]["decision"]
    return {
        "id": row["id"],
        "answers": {"decision": answer(question, value)},
        "source_input_sha256": input_digest(row["state"], row["questions"]),
        "model_id": "m",
    }


class ScoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.gold = (
            [item(i, "a/choice", CHOICE, "x" if i % 2 else "y") for i in range(8)]
            + [item(i, "b/noul", NOUL, bool(i % 2)) for i in range(8)]
            + [item(i, "c/score", LEVELS, i % 3) for i in range(9)]
        )

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, rows):
        path = self.dir / name
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        return path

    def predictions(self, wrong_choice=0):
        rows = []
        for row in self.gold:
            value = row["gold"]["decision"]["value"]
            if row["task"] == "a/choice" and int(row["source_item_id"]) < wrong_choice:
                value = None
            if row["task"] == "b/noul":
                value = True
            rows.append(prediction(row, value))
        return rows

    def test_median_mean_macro_f1_with_invalid_answers(self):
        preds = {r["id"]: r for r in self.predictions(wrong_choice=4)}
        result = score.report(self.gold, preds, replicates=50)
        tasks = result["tasks"]
        # a/choice: items 0-3 invalid (golds y,x,y,x), 4-7 right -> F1 2/3 per class
        self.assertAlmostEqual(tasks["a/choice"]["macro_f1"], 2 / 3)
        # b/noul: always yes -> F1(true)=2/3, F1(false)=0
        self.assertAlmostEqual(tasks["b/noul"]["macro_f1"], 1 / 3)
        self.assertAlmostEqual(tasks["c/score"]["macro_f1"], 1.0)
        self.assertAlmostEqual(tasks["c/score"]["qwk"], 1.0)
        self.assertAlmostEqual(result["H_dev"], 2 / 3)
        self.assertAlmostEqual(
            result["task_mean"], statistics.fmean([2 / 3, 1 / 3, 1.0])
        )
        self.assertAlmostEqual(result["by_type_mean"]["noul"], 1 / 3)
        self.assertEqual(result["H_dev_bootstrap"]["replicates"], 50)
        self.assertGreaterEqual(result["H_dev_bootstrap"]["sd"], 0.0)

    def test_missing_predictions_count_as_wrong(self):
        preds = {
            r["id"]: r for r in self.predictions() if r["id"] != self.gold[0]["id"]
        }
        result = score.report(self.gold, preds, replicates=10)
        self.assertLess(result["tasks"]["a/choice"]["macro_f1"], 1.0)
        self.assertEqual(result["valid"], len(self.gold) - 1)

    def test_seal_then_score_and_refuse_changed_predictions(self):
        prompts = self.write(
            "prompts.jsonl",
            [
                {"id": g["id"], "state": g["state"], "questions": g["questions"]}
                for g in self.gold
            ],
        )
        gold = self.write("gold.jsonl", self.gold)
        preds = self.write("preds.jsonl", self.predictions())
        score.main(
            [
                "seal",
                "--prompts",
                str(prompts),
                "--predictions",
                str(preds),
                "--output",
                str(self.dir / "SEAL.json"),
            ]
        )
        seal = json.loads((self.dir / "SEAL.json").read_text())
        self.assertFalse(seal["gold_read"])
        score.main(
            [
                "score",
                "--gold",
                str(gold),
                "--predictions",
                str(preds),
                "--seal",
                str(self.dir / "SEAL.json"),
                "--label",
                "m",
                "--replicates",
                "20",
                "--output",
                str(self.dir / "R.json"),
            ]
        )
        report = json.loads((self.dir / "R.json").read_text())
        self.assertEqual(report["label"], score.LABEL)
        self.write("preds.jsonl", self.predictions(wrong_choice=2))
        with self.assertRaises(ValueError):
            score.main(
                [
                    "score",
                    "--gold",
                    str(gold),
                    "--predictions",
                    str(preds),
                    "--seal",
                    str(self.dir / "SEAL.json"),
                    "--label",
                    "m",
                    "--output",
                    str(self.dir / "R2.json"),
                ]
            )

    def test_compare_is_paired_over_groups(self):
        left = {r["id"]: r for r in self.predictions()}
        right = {r["id"]: r for r in self.predictions(wrong_choice=8)}
        result = score.paired(self.gold, left, right, 200, score.SEED)
        self.assertAlmostEqual(result["delta_H_dev"], 1.0 - 1 / 3)
        self.assertEqual(result["unit"], "source groups within each task")
        self.assertGreater(result["p_left_better"], 0.9)
        same = score.paired(self.gold, left, left, 50, score.SEED)
        self.assertEqual(same["delta_H_dev"], 0.0)
        self.assertEqual(same["ci95"], [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
