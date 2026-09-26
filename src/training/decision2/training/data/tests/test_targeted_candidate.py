"""Oracle and role-boundary tests for the targeted training candidate."""

from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path

from training.data import build_targeted_candidate as targeted
from training.model.data import validate_row


class TargetedCandidateTests(unittest.TestCase):
    def test_pair_oracles_and_option_reversal(self) -> None:
        rows = targeted.generate("candidate-test-seed")
        self.assertEqual(len(rows), 2000)
        self.assertEqual(
            Counter(row["task_type"] for row in rows),
            {"choice": 1000, "noul": 600, "score": 400},
        )
        groups = defaultdict(list)
        for row in rows:
            validate_row(row, "train")
            groups[row["group_id"]].append(row)
        self.assertEqual(len(groups), 1000)
        for group in groups.values():
            self.assertEqual(len(group), 2)
            family = group[0]["family"]
            self.assertEqual({row["family"] for row in group}, {family})
            choices = [row["options"][row["label"]] for row in group]
            if family == "targeted_interval_conjunction":
                self.assertEqual(
                    {option["key"] for option in choices}, {"false", "true"}
                )
                for row in group:
                    oracle = row["audit_metadata"]["oracle"]
                    truth = (
                        oracle["start_inclusive"]
                        <= oracle["inspection_day"]
                        < oracle["end_exclusive"]
                        and oracle["stamp_verified"]
                    )
                    self.assertEqual(oracle["valid"], truth)
            elif family == "targeted_quantized_median":
                for row in group:
                    oracle = row["audit_metadata"]["oracle"]
                    self.assertEqual(sorted(oracle["readings"])[2], oracle["median"])
                    self.assertEqual(
                        sum(oracle["median"] >= x for x in oracle["cutoffs"]),
                        oracle["grade"],
                    )
                    self.assertEqual(
                        choices[group.index(row)]["key"], str(oracle["grade"])
                    )
            else:
                self.assertEqual(group[0]["state"], group[1]["state"])
                self.assertEqual(choices[0]["description"], choices[1]["description"])
                self.assertNotEqual(choices[0]["key"], choices[1]["key"])
                self.assertNotEqual(group[0]["input_sha256"], group[1]["input_sha256"])

    def test_deterministic_new_source_and_no_benchmark_family(self) -> None:
        one = targeted.generate("candidate-test-seed")
        two = targeted.generate("candidate-test-seed")
        self.assertEqual(
            [row["input_sha256"] for row in one], [row["input_sha256"] for row in two]
        )
        self.assertEqual({row["source"] for row in one}, {targeted.SOURCE})
        self.assertFalse(
            {row["family"] for row in one}
            & {
                "attribute_gate",
                "rule_precedence",
                "set_reconciliation",
                "transition_table",
                "constraint_competition",
                "exception_stack",
                "evidence_join",
                "resource_ledger",
            }
        )

    def test_audit_rejects_context_reuse_and_nonpilot_path(self) -> None:
        row = targeted.interval_conjunction("candidate-test-seed", 0)[0]
        reference = targeted.context_rows([row])
        with self.assertRaisesRegex(ValueError, "overlaps audit-only reference"):
            targeted.context_overlap([row], reference, approximate=False)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "css-evaluation.prompts.jsonl"
            path.write_text(json.dumps({"id": "x", "state": "sample"}) + "\n")
            with self.assertRaisesRegex(
                ValueError, "Expected only css-pilot.prompts.jsonl"
            ):
                targeted.load_context_reference(
                    path, expected_name="css-pilot.prompts.jsonl"
                )


if __name__ == "__main__":
    unittest.main()
