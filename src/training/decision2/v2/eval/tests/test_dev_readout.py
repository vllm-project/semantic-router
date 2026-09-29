import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from v2.eval import dev_readout, panels
from v2.eval.htdev.sources.empathic_reactions import QUESTION

SCORE = {"type": "score", "instructions": "Rate.", "criteria": list("abcde")}


def write_jsonl(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return panels.sha_file(path)


def answer(item_id: str, level: int) -> dict:
    return {"id": item_id, "answers": {"decision": {"type": "score", "score": level}}}


class Score5ReadoutTest(unittest.TestCase):
    def test_run_dir_with_score5_and_htdev_predictions_gets_both_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, run = Path(tmp) / "panels", Path(tmp) / "run"
            gold = [
                {
                    "id": f"s{n}",
                    "task": "score5/humor",
                    "source": "a7q-aho",
                    "group_id": f"g{n}",
                    "language": "en",
                    "long": False,
                    "questions": {"decision": SCORE},
                    "gold": {"decision": {"type": "score", "value": n % 5}},
                }
                for n in range(50)
            ]
            prompts = [
                {"id": g["id"], "state": "x", "questions": g["questions"]} for g in gold
            ]
            ht_prompts = [
                {"id": f"h{n}", "state": "y", "questions": {"decision": QUESTION}}
                for n in range(10)
            ] + [{"id": "other", "state": "z", "questions": {"decision": SCORE}}]
            entries = {
                "score5-dev": {
                    "prompts": "goldfree/score5-dev.prompts.jsonl",
                    "prompts_sha256": write_jsonl(
                        root / "goldfree/score5-dev.prompts.jsonl", prompts
                    ),
                    "gold": "gold/score5-dev.gold.jsonl",
                    "gold_sha256": write_jsonl(
                        root / "gold/score5-dev.gold.jsonl", gold
                    ),
                    "originals": 50,
                },
                "ht-dev": {
                    "prompts": "goldfree/ht-dev.prompts.jsonl",
                    "prompts_sha256": write_jsonl(
                        root / "goldfree/ht-dev.prompts.jsonl", ht_prompts
                    ),
                    "gold": "gold/ht-dev.gold.jsonl",
                    "gold_sha256": write_jsonl(root / "gold/ht-dev.gold.jsonl", []),
                    "originals": 11,
                },
            }
            write_jsonl(
                run / "output/score5-dev.predictions.jsonl",
                [answer(g["id"], 3) for g in gold],
            )
            write_jsonl(
                run / "output/ht-dev.predictions.jsonl",
                [answer(f"h{n}", n % 2) for n in range(9)] + [answer("other", 4)],
            )
            args = argparse.Namespace(
                panel_root=root,
                run_dir=run,
                typed_dev=None,
                css_pilot=None,
                score5=None,
                select=None,
                cal=None,
                label="t",
            )
            with mock.patch.dict(panels.ALL, entries):
                out = dev_readout.readout(args)
        self.assertNotIn("typed_dev", out)
        block = out["score5"]
        self.assertEqual(block["histogram"], {"0": 0, "1": 0, "2": 0, "3": 50, "4": 0})
        self.assertEqual(block["flags"], ["COLLAPSE", "NO-SIGNAL"])
        self.assertEqual(block["correct"], 10)
        empathy = out["htdev_empathy_levels"]
        self.assertEqual(empathy["n"], 10)
        self.assertEqual(empathy["invalid_or_missing"], 1)
        self.assertEqual(empathy["histogram"]["0"], 5)
        self.assertNotIn("flags", empathy)


if __name__ == "__main__":
    unittest.main()
