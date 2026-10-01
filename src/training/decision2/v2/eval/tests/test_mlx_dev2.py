"""CPU checks for the MLX-DEV2 isolation scan, prediction conversion, score and paired compare."""

from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path

from v2.eval import mlx_dev2
from v2.eval.same_panel import input_digest

CHOICE_Q = {"type": "choice", "instructions": "x", "criteria": {"a": "A", "b": "B"}}
NOUL_Q = {"type": "noul", "instructions": "x", "criteria": {"true": "T", "false": "F"}}
CELLS = [
    ("choice", "en", CHOICE_Q, "a"),
    ("choice", "de", CHOICE_Q, "a"),
    ("noul", "en", NOUL_Q, True),
    ("noul", "ko", NOUL_Q, True),
]
PER_CELL = 20


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def build_panel(root: Path) -> list[dict]:
    gold, prompts = [], []
    for kind, lang, question, value in CELLS:
        for k in range(PER_CELL):
            item = f"{kind}-{lang}-{k}"
            state = f"state {item}"
            questions = {"q": question}
            gold.append(
                {
                    "id": item,
                    "source": kind,
                    "source_id": str(k),
                    "language": lang,
                    "type": kind,
                    "value": value,
                    "input_sha256": input_digest(state, questions),
                }
            )
            prompts.append({"id": item, "state": state, "questions": questions})
    root.mkdir()
    write_jsonl(root / "gold.jsonl", gold)
    write_jsonl(root / "prompts.jsonl", prompts)
    return gold


def answers(gold: list[dict], wrong: set[str]) -> list[dict]:
    rows = []
    for g in gold:
        bad = g["id"] in wrong
        if g["type"] == "choice":
            answer = {"choice": "b" if bad else "a"}
        else:
            answer = {"noul": 0.1 if bad else 0.9}
        rows.append({"panel": "mlx-dev2", "id": g["id"], "answers": {"q": answer}})
    return rows


class ScanTests(unittest.TestCase):
    def test_flags_exact_shared_and_contained_parts_only(self):
        parts = [
            {
                "source": "massive",
                "source_id": "1",
                "language": "en",
                "text": "wake me up at seven tomorrow please",
            },
            {
                "source": "massive",
                "source_id": "2",
                "language": "en",
                "text": "play some jazz in the kitchen",
            },
            {
                "source": "pawsx",
                "source_id": "7",
                "language": "en",
                "text": "The river flows through the old town and then turns north towards the hills",
            },
            {
                "source": "pawsx",
                "source_id": "8",
                "language": "en",
                "text": "A completely different sentence that nobody wrote in any training file at all",
            },
            {
                "source": "massive",
                "source_id": "3",
                "language": "en",
                "text": "turn off the lights now",
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_jsonl(root / "pool.jsonl", parts)
            write_jsonl(
                root / "a.jsonl", [{"state": "Wake me up at  seven tomorrow please"}]
            )
            with gzip.open(root / "b.jsonl.gz", "wt", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        {
                            "x": {
                                "y": [
                                    "Notes: the river flows through the old town and then turns north towards the hills today"
                                ]
                            }
                        }
                    )
                    + "\n"
                )
            write_jsonl(
                root / "c.jsonl",
                [{"prompt": "User: could you turn off the lights now? Thanks a lot."}],
            )
            files = [
                root / "a.jsonl",
                root / "b.jsonl.gz",
                root / "c.jsonl",
                root / "missing.jsonl",
            ]
            result = mlx_dev2.scan(root / "pool.jsonl", files, workers=2)
        self.assertEqual(
            result["excluded_source_items"], {"massive": ["1", "3"], "pawsx": ["7"]}
        )
        self.assertEqual(result["corpus"]["files"], 3)
        self.assertEqual(result["corpus"]["unreadable"], 1)
        reasons = result["flagged_parts_by_source_and_reason"]
        self.assertEqual(reasons["massive/exact"], 1)
        self.assertEqual(reasons["massive/contained"], 2)
        self.assertEqual(reasons["pawsx/shingle"], 1)
        self.assertNotIn("text", json.dumps(result))


class ScoreCompareTests(unittest.TestCase):
    def test_predictions_score_and_paired_compare(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gold = build_panel(root / "panel")
            noul_wrong = {f"noul-{lang}-{k}" for lang in ("en", "ko") for k in range(6)}
            write_jsonl(root / "ref.answers.jsonl", answers(gold, set()))
            write_jsonl(root / "cand.answers.jsonl", answers(gold, noul_wrong))
            for side in ("ref", "cand"):
                rows = mlx_dev2.predictions(
                    root / "panel", root / f"{side}.answers.jsonl"
                )
                self.assertEqual(len(rows), len(gold))
                write_jsonl(root / f"{side}.jsonl", rows)
            score = mlx_dev2.score(root / "panel", root / "cand.jsonl")
            self.assertEqual(score["missing"], 0)
            self.assertAlmostEqual(score["per_type"]["noul"]["macro"], 14 / 20)
            self.assertAlmostEqual(score["card_type_macro_accuracy"], (1 + 14 / 20) / 2)
            result = mlx_dev2.compare(
                root / "panel",
                root / "cand.jsonl",
                root / "ref.jsonl",
                ("c", "r"),
                500,
                7,
            )
        card = result["card_eligible"]
        self.assertAlmostEqual(card["delta"], -0.15)
        self.assertLess(card["ci95"]["high"], 0)
        self.assertFalse(result["guard_pass"])
        self.assertAlmostEqual(result["per_type"]["choice"]["delta"], 0.0)

    def test_identical_runs_pass_the_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gold = build_panel(root / "panel")
            write_jsonl(root / "a.answers.jsonl", answers(gold, {"choice-en-1"}))
            write_jsonl(
                root / "a.jsonl",
                mlx_dev2.predictions(root / "panel", root / "a.answers.jsonl"),
            )
            result = mlx_dev2.compare(
                root / "panel", root / "a.jsonl", root / "a.jsonl", ("a", "a"), 200, 7
            )
        self.assertEqual(result["card_eligible"]["delta"], 0.0)
        self.assertTrue(result["guard_pass"])


if __name__ == "__main__":
    unittest.main()
