from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from v2.eval.sealed import candidates as runner
from v2.eval.sealed.schema import display_order, letters, validate
from v2.eval.sealed.sources import hallutruthqa as source


def row(item: str, question: str, label: str = "no_hallucination", answer: str = "C"):
    return {
        "id": item,
        "question": question,
        "generated_answer": f"generated answer for {item}",
        "gold_answer": f"REFERENCE-{item}",
        "generator_model": "QCRI/Fanar-1-9B-Instruct",
        "label": label,
        "options": {key: f"option {key} of {item}" for key in "ABCDEF"},
        "answer": answer,
        "hallucinations": [],
        "gold_match": "",
        "needs_human_review": False,
    }


def even(item: str) -> bool:
    return int(hashlib.sha256(item.encode()).hexdigest(), 16) % 2 == 0


def write(root: Path, name: str, rows: list[dict]) -> None:
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    (root / f"{name}.jsonl").write_text(text, encoding="utf-8")


class HalluTruthQATest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.test_rows = [
            row(
                f"geo_{n}",
                f"question {n}?",
                ("hallucination", "no_hallucination")[n % 2],
            )
            for n in range(16)
        ]
        self.test_rows += [
            row("dup_b", "Same question?", "hallucination", "A"),
            row("dup_a", "  same   QUESTION? ", "no_hallucination", "F"),
            row("seen_1", "Asked in train?"),
            row("long_1", "x" * 30_000),
        ]
        write(self.root, "train", [row("tr_1", "asked in TRAIN?")])
        write(self.root, "dev", [row("dv_1", "dev question?")])
        write(self.root, "test", self.test_rows)
        self.out = list(source.candidates(self.root))

    def tearDown(self):
        self.tmp.cleanup()

    def test_all_candidates_valid_and_filtered(self):
        self.assertTrue(self.out)
        for candidate in self.out:
            self.assertEqual(validate(candidate, source.SPEC), [])
        ids = [c.source_item_id for c in self.out]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertNotIn("seen_1", ids)
        self.assertNotIn("long_1", ids)
        self.assertEqual(len(ids), len(self.test_rows) - 2)

    def test_tasks_use_disjoint_questions_by_id_parity(self):
        tasks = {c.source_item_id: c.task for c in self.out}
        self.assertEqual(set(tasks.values()), set(source.SPEC.tasks))
        for item, task in tasks.items():
            if not item.startswith("dup_"):
                expected = source.NOUL if even(item) else source.CHOICE
                self.assertEqual(task, expected, item)
        dup = {
            c.source_item_id: c for c in self.out if c.source_item_id.startswith("dup_")
        }
        expected = source.NOUL if even("dup_a") else source.CHOICE
        self.assertEqual({c.task for c in dup.values()}, {expected})
        self.assertEqual(dup["dup_a"].group_id, dup["dup_b"].group_id)
        groups = {}
        for c in self.out:
            groups.setdefault(c.group_id, set()).add(c.task)
        self.assertTrue(all(len(t) == 1 for t in groups.values()))

    def test_noul_state_and_gold(self):
        by_id = {r["id"]: r for r in self.test_rows}
        noul = [c for c in self.out if c.task == source.NOUL]
        self.assertTrue(noul)
        for c in noul:
            src = by_id[c.source_item_id]
            self.assertEqual(set(c.state), {"question", "answer"})
            self.assertEqual(c.state["answer"], src["generated_answer"])
            self.assertIs(c.gold, src["label"] == "hallucination")
            self.assertEqual(c.balance_label, str(c.gold))
            self.assertEqual(c.overlap_texts, [c.state["question"], c.state["answer"]])
            blob = json.dumps([c.state, c.question], ensure_ascii=False)
            self.assertNotIn("REFERENCE-", blob)
            self.assertNotIn("hallucination", json.dumps(c.state))

    def test_choice_reorders_options_independently_of_gold(self):
        by_id = {r["id"]: r for r in self.test_rows}
        choice = [c for c in self.out if c.task == source.CHOICE]
        self.assertTrue(choice)
        for c in choice:
            src = by_id[c.source_item_id]
            self.assertEqual(set(c.state), {"question"})
            self.assertEqual(list(c.question["criteria"]), letters(6))
            order = display_order(c.source_item_id, 6)
            expected = [src["options"]["ABCDEF"[i]] for i in order]
            self.assertEqual(list(c.question["criteria"].values()), expected)
            self.assertEqual(
                c.question["criteria"][c.gold], src["options"][src["answer"]]
            )
            self.assertEqual(c.balance_label, c.gold)
            self.assertNotIn("generated answer", json.dumps(c.state))

    def test_runner_counts(self):
        receipt = runner.run("hallutruthqa", self.root, None)
        self.assertEqual(receipt["invalid"], {})
        self.assertEqual(receipt["valid"], len(self.out))
        self.assertEqual(
            sum(t["candidates"] for t in receipt["tasks"].values()), len(self.out)
        )


if __name__ == "__main__":
    unittest.main()
