from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from v2.eval.sealed import score

CHOICE = {"type": "choice", "instructions": "?", "criteria": {"x": "X", "y": "Y"}}
NOUL = {"type": "noul", "instructions": "?", "criteria": {"true": "t", "false": "f"}}
LEVELS = {"type": "score", "instructions": "?", "criteria": ["lo", "mid", "hi"]}


def item(
    i: int, task: str, question: dict, value, source: str = "src", lang: str = "en"
) -> dict:
    record = {"type": question["type"], "value": value, "semantic_value": value}
    if question["type"] == "choice":
        record["label_to_semantic"] = {k: k for k in question["criteria"]}
    return {
        "id": f"c1-{task}-{i}",
        "task": task,
        "source": source,
        "source_item_id": str(i),
        "group_id": f"g{i // 2}",
        "language": lang,
        "input_chars": 100,
        "long": i % 5 == 0,
        "state": f"s{i}",
        "questions": {"decision": question},
        "gold": {"decision": record},
    }


def answer(question: dict, value) -> dict:
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
        "source_input_sha256": score.input_digest(row["state"], row["questions"]),
        "model_id": "m",
    }


class ScoreTest(unittest.TestCase):
    def setUp(self):
        self.gold = (
            [item(i, "a/choice", CHOICE, "x" if i % 2 else "y") for i in range(20)]
            + [
                item(i, "b/noul", NOUL, bool(i % 2), "hallutruthqa", "ar")
                for i in range(20)
            ]
            + [item(i, "c/score", LEVELS, i % 3) for i in range(21)]
        )

    def write(self, root: Path, name: str, rows) -> Path:
        path = root / name
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        return path

    def test_seal_score_and_compare(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompts = self.write(
                root,
                "prompts.jsonl",
                [
                    {"id": g["id"], "state": g["state"], "questions": g["questions"]}
                    for g in self.gold
                ],
            )
            gold = self.write(root, "gold.jsonl", self.gold)
            perfect = self.write(
                root,
                "perfect.jsonl",
                [prediction(g, g["gold"]["decision"]["value"]) for g in self.gold],
            )
            half = [
                (
                    prediction(g, g["gold"]["decision"]["value"])
                    if i % 2 == 0
                    else {**prediction(g, None), "answers": {"decision": None}}
                )
                for i, g in enumerate(self.gold)
            ]
            weak = self.write(root, "weak.jsonl", half)
            score.seal(
                Namespace(
                    prompts=prompts, predictions=perfect, output=root / "seal.json"
                )
            )
            score.score(
                Namespace(
                    gold=gold,
                    predictions=perfect,
                    seal=root / "seal.json",
                    label="p",
                    output=root / "report.json",
                )
            )
            report = json.loads((root / "report.json").read_text())
            self.assertAlmostEqual(report["c1"], 100.0)
            self.assertAlmostEqual(report["tasks"]["c/score"]["qwk"], 1.0)
            self.assertEqual(report["slices"]["non_english"]["items"], 20)
            self.assertAlmostEqual(
                report["licence_split"]["nc_tasks_mean_macro_f1"], 100.0
            )
            score.compare(
                Namespace(
                    gold=gold,
                    left=perfect,
                    right=weak,
                    left_name="p",
                    right_name="w",
                    replicates=200,
                    output=root / "paired.json",
                )
            )
            paired = json.loads((root / "paired.json").read_text())
            self.assertGreater(paired["delta"], 0)
            self.assertGreater(paired["ci95"][0], 0)
            (root / "perfect.jsonl").write_text(perfect.read_text() + "\n")
            with self.assertRaises(ValueError):
                score.score(
                    Namespace(
                        gold=gold,
                        predictions=perfect,
                        seal=root / "seal.json",
                        label="p",
                        output=root / "report2.json",
                    )
                )

    def test_seal_rejects_changed_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompts = self.write(
                root,
                "prompts.jsonl",
                [
                    {"id": g["id"], "state": g["state"], "questions": g["questions"]}
                    for g in self.gold
                ],
            )
            rows = [prediction(g, g["gold"]["decision"]["value"]) for g in self.gold]
            rows[0]["source_input_sha256"] = "0" * 64
            bad = self.write(root, "bad.jsonl", rows)
            with self.assertRaises(ValueError):
                score.seal(
                    Namespace(
                        prompts=prompts, predictions=bad, output=root / "seal.json"
                    )
                )

    def test_macro_f1_and_invalid_answers(self):
        self.assertAlmostEqual(score.macro_f1([("a", "a"), ("b", "b")]), 1.0)
        self.assertAlmostEqual(score.macro_f1([("a", None), ("b", "b")]), (0 + 1.0) / 2)
        self.assertIsNone(score.quadratic_kappa([(1, None), (2, None)]))


if __name__ == "__main__":
    unittest.main()
