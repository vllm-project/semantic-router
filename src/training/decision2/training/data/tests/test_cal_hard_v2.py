"""Independent oracle checks for the harder CAL candidate."""

from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path

from training.data import build_cal_hard_v2 as hard
from training.model.data import validate_row


class CalHardV2Tests(unittest.TestCase):
    def test_balanced_pairs_and_independent_oracles(self) -> None:
        rows = hard.generate("hard-cal-test-seed")
        self.assertEqual(
            Counter(row["task_type"] for row in rows), {"noul": 300, "score": 300}
        )
        self.assertEqual(
            Counter(row["audit_metadata"]["difficulty_tier"] for row in rows),
            {"direct": 200, "boundary": 200, "conditional": 200},
        )
        groups = defaultdict(list)
        for row in rows:
            validate_row(row, "cal")
            groups[row["group_id"]].append(row)
            oracle = row["audit_metadata"]["oracle"]
            answer = row["options"][row["label"]]["key"]
            if row["task_type"] == "noul":
                approved = (
                    oracle["signature_verified"]
                    and oracle["submitted_pages"] >= oracle["minimum_pages"]
                    and (not oracle["embargo"] or oracle["waiver_verified"])
                )
                self.assertEqual(oracle["approved"], approved)
                self.assertEqual(answer, "true" if approved else "false")
            else:
                net = (
                    oracle["opening"]
                    + oracle["bonus"]
                    - oracle["fee"]
                    - (oracle["hold"] if oracle["hold_applies"] else 0)
                )
                self.assertEqual(oracle["net"], net)
                self.assertEqual(
                    oracle["grade"], sum(net >= cutoff for cutoff in oracle["cutoffs"])
                )
                self.assertEqual(answer, str(oracle["grade"]))
                self.assertIn(oracle["distance_to_crossed_cutoff"], (0, 1))
        self.assertEqual(len(groups), 300)
        for pair in groups.values():
            self.assertEqual(len(pair), 2)
            answer_keys = [row["options"][row["label"]]["key"] for row in pair]
            if pair[0]["task_type"] == "noul":
                self.assertEqual(set(answer_keys), {"false", "true"})
            else:
                self.assertEqual(abs(int(answer_keys[0]) - int(answer_keys[1])), 1)

    def test_retains_exact_css_choice_lineage_only(self) -> None:
        rows = [
            {
                "id": f"css-{task}-{i}",
                "task_type": "choice",
                "source": f"css_pilot:{task}",
                "group_id": f"css-group-{task}-{i}",
            }
            for task in ("semeval_stance", "implicit_hate", "discourse")
            for i in range(100)
        ]
        self.assertEqual(hard.retained_choice(rows), rows)
        with self.assertRaisesRegex(ValueError, "frozen 300-row partition"):
            hard.retained_choice(rows[:-1])

    def test_fresh_cal_rejects_old_cal_input(self) -> None:
        row = hard.noul_case("hard-cal-test-seed", 0)[0]
        with self.assertRaisesRegex(ValueError, "overlap"):
            hard.zero_audit([row], [row], "same input", context_near=True)

    def test_nested_train_inventory_keeps_lineage_for_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "kev.train.jsonl"
            row = {
                "state": "audit context",
                "questions": {"decision": {}},
                "_meta": {
                    "id": "source-id",
                    "group_id": "source-group",
                    "input_sha256": "source-input",
                    "split": "train",
                },
            }
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            rows, receipt = hard.train_inventory_context(path)
            self.assertEqual(receipt["shapes"], {"kev_nested": 1})
            self.assertEqual(rows[0]["group_id"], "source-group")
            row["_meta"]["split"] = "test"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unrecognized nested TRAIN"):
                hard.train_inventory_context(path)


if __name__ == "__main__":
    unittest.main()
