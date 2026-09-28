from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from v2.eval.sealed import review


def gold_row(i: int, task: str, qtype: str, value, long: bool = False) -> dict:
    criteria = (
        {"true": "yes", "false": "no"}
        if qtype == "noul"
        else ["lo", "mid", "hi"] if qtype == "score" else {"a": "A", "b": "B"}
    )
    question = {"type": qtype, "instructions": "?", "criteria": criteria}
    return {
        "id": f"c1-{task}-{i}",
        "task": task,
        "long": long,
        "state": f"state {i}",
        "questions": {"decision": question},
        "gold": {"decision": {"type": qtype, "value": value}},
    }


class ReviewTest(unittest.TestCase):
    def test_packet_and_score(self):
        rows = [
            gold_row(i, "t/choice", "choice", "a" if i % 2 else "b") for i in range(40)
        ]
        rows += [gold_row(i, "t/noul", "noul", bool(i % 2)) for i in range(40)]
        rows += [gold_row(i, "t/long", "score", i % 3, long=True) for i in range(40)]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build = root / "build"
            build.mkdir()
            (build / "gold.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in rows)
            )
            (root / "salt").write_text("x" * 40)
            out = root / "review"
            review.packet(
                Namespace(build_dir=build, salt_file=root / "salt", output_dir=out)
            )
            receipt = json.loads((out / "packet-receipt.json").read_text())
            self.assertEqual(
                receipt["tasks"], {"t/choice": 16, "t/long": 8, "t/noul": 16}
            )
            short = [
                json.loads(x)
                for x in (out / "packet-short.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(short), 32)
            self.assertTrue(
                all(set(r) == {"review_id", "state", "question"} for r in short)
            )
            mapping = json.loads((out / "mapping.json").read_text())
            truth = {r["id"]: r for r in rows}
            for reviewer in ("r1", "r2", "r3"):
                lines = []
                for review_id, item_id in mapping.items():
                    item = truth[item_id]
                    value = item["gold"]["decision"]["value"]
                    flags = []
                    if item["task"] == "t/choice":
                        value = "a"
                    if item["task"] == "t/noul" and reviewer != "r3":
                        flags = ["ambiguous"]
                    lines.append(
                        json.dumps(
                            {"review_id": review_id, "answer": value, "flags": flags}
                        )
                    )
                (out / f"answers-{reviewer}-all.jsonl").write_text(
                    "\n".join(lines) + "\n"
                )
            review.score(
                Namespace(
                    build_dir=build,
                    review_dir=out,
                    reviewer=["r1", "r2", "r3"],
                    output=root / "score.json",
                )
            )
            scored = json.loads((root / "score.json").read_text())
            sampled_a = sum(
                truth[item_id]["gold"]["decision"]["value"] == "a"
                for item_id in mapping.values()
                if truth[item_id]["task"] == "t/choice"
            )
        tasks = scored["tasks"]
        self.assertEqual(tasks["t/long"]["decision"], "PASS")
        self.assertEqual(tasks["t/long"]["majority_agreement_with_gold"], 1.0)
        self.assertTrue(tasks["t/noul"]["decision"].startswith("FAIL: quality flags"))
        self.assertIn("t/noul", scored["failed_tasks"])
        self.assertEqual(
            tasks["t/choice"]["per_reviewer_agreement"]["r1"], sampled_a / 16
        )
        expected = (
            "PASS"
            if sampled_a / 16 > 0.5
            else "FAIL: majority agreement not above chance"
        )
        self.assertEqual(tasks["t/choice"]["decision"], expected)

    def test_fleiss_kappa(self):
        self.assertAlmostEqual(
            review.fleiss_kappa([["a", "a"], ["b", "b"]], ["a", "b"]), 1.0
        )
        self.assertLess(review.fleiss_kappa([["a", "b"], ["b", "a"]], ["a", "b"]), 0)


if __name__ == "__main__":
    unittest.main()
