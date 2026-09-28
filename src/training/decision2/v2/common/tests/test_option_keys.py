from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from training.model.data import INPUT_FIELDS, digest, validate_row
from v2.common import option_keys
from v2.data.a7 import build_a7


def _row(keys: list[str], label: int = 1, task_type: str = "choice") -> dict:
    row = {
        "id": "r1",
        "state": "Which intent does the utterance express?",
        "instructions": "Pick one.",
        "options": [{"key": key, "description": f"intent {key}"} for key in keys],
        "label": label,
        "task_type": task_type,
        "family": "stage4_replay_banking_train",
        "group_id": "g1",
        "language": "en",
        "source": "unit",
        "split": "train",
        "evaluation_role": "train",
        "render_template": "unit",
        "audit_metadata": {"origin": "unit"},
    }
    row["input_sha256"] = digest({field: row[field] for field in INPUT_FIELDS})
    return row


class RenumberTest(unittest.TestCase):
    def test_construction_order_keys_become_positional(self) -> None:
        row = _row(["result_2", "result_3", "result_0", "result_1"], label=1)
        out, changed = option_keys.renumber(row)
        self.assertTrue(changed)
        self.assertEqual(
            [o["key"] for o in out["options"]],
            ["result_0", "result_1", "result_2", "result_3"],
        )
        self.assertEqual(
            [o["description"] for o in out["options"]],
            [o["description"] for o in row["options"]],
        )
        self.assertEqual(out["label"], 1)
        audit = out["audit_metadata"][option_keys.AUDIT_KEY]
        self.assertEqual(
            audit["original_keys"], ["result_2", "result_3", "result_0", "result_1"]
        )
        self.assertEqual(audit["original_input_sha256"], row["input_sha256"])
        self.assertEqual(out["audit_metadata"]["origin"], "unit")
        validate_row(out, "train")
        self.assertEqual(row["options"][0]["key"], "result_2")

    def test_other_rows_are_unchanged(self) -> None:
        for row in (
            _row(["result_0", "result_1", "result_2"]),
            _row(["result_1", "c2", "result_0"]),
            _row(["a", "b"]),
            _row(["result_1", "result_0"], task_type="noul"),
        ):
            out, changed = option_keys.renumber(row)
            self.assertFalse(changed)
            self.assertEqual(out, row)

    def test_teacher_probs_follow_their_options(self) -> None:
        row = _row(["result_1", "result_2", "result_0"], label=1)
        row["teacher_probs"] = {"result_1": 0.1, "result_2": 0.7, "result_0": 0.2}
        out, _ = option_keys.renumber(row)
        self.assertEqual(
            out["teacher_probs"], {"result_0": 0.1, "result_1": 0.7, "result_2": 0.2}
        )
        validate_row(out, "train", replay=True)
        row["teacher_probs"] = {"result_9": 1.0}
        with self.assertRaises(ValueError):
            option_keys.renumber(row)

    def test_matches_the_a7_build_rule(self) -> None:
        row = _row(
            ["result_4", "result_0", "result_2", "result_3", "result_1"], label=0
        )
        shared, _ = option_keys.renumber(row)
        a7 = build_a7.rekey(row)
        self.assertEqual(shared["options"], a7["options"])
        self.assertEqual(shared["input_sha256"], build_a7.input_hash(a7))

    def test_cli_is_idempotent_and_reports_gold_ranks(self) -> None:
        rows = [
            _row(["result_1", "result_2", "result_0"], label=1),
            _row(["result_0", "result_1"]),
            _row(["a", "b"]),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp, "in.jsonl")
            src.write_text("".join(json.dumps(row) + "\n" for row in rows))
            first = Path(tmp, "out.jsonl")
            option_keys.main(
                [
                    "--rows",
                    str(src),
                    "--out",
                    str(first),
                    "--receipt",
                    str(Path(tmp, "r1.json")),
                ]
            )
            receipt = json.loads(Path(tmp, "r1.json").read_text())
            self.assertEqual(receipt["renumbered"], 1)
            self.assertEqual(receipt["already_positional_result_keys"], 1)
            self.assertEqual(receipt["gold_key_rank_before"], {"largest": 1})
            self.assertEqual(receipt["remaining_construction_order_rows"], 0)
            option_keys.main(
                [
                    "--rows",
                    str(first),
                    "--check",
                    "--receipt",
                    str(Path(tmp, "r2.json")),
                ]
            )
            again = json.loads(Path(tmp, "r2.json").read_text())
            self.assertEqual(again["renumbered"], 0)
            self.assertEqual(again["input_sha256"], receipt["output_sha256"])


if __name__ == "__main__":
    unittest.main()
