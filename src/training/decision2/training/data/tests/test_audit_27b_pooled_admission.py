"""Independent-group contracts for the prospective pooled 27B admission."""

from __future__ import annotations

import unittest

from training.data.audit_27b_pooled_admission import (
    AdmissionHold,
    distinctive_overlap,
    group_integrity,
    selected_rights,
)
from training.data.plan_goldfree_inventory import project_partition_row
from training.model.data import INPUT_FIELDS, digest


def row(name: str, task: str, group: str) -> dict[str, str]:
    return {
        "id": name,
        "source": "synthetic-test",
        "group_id": group,
        "task_type": task,
    }


class GroupIntegrityContracts(unittest.TestCase):
    def test_missing_type_is_distinct_from_missing_row(self) -> None:
        original = [
            row("s0", "score", "shared"),
            row("c0", "choice", "shared"),
            row("c1", "choice", "shared"),
            row("n0", "noul", "shared"),
            row("n1", "noul", "other"),
        ]
        chosen = [original[0], original[1], original[4]]
        result = group_integrity(original, chosen)
        self.assertEqual(result["selected_source_groups"], 2)
        self.assertEqual(result["missing_task_type_groups"], 1)
        self.assertEqual(result["missing_any_row_groups"], 1)
        self.assertEqual(result["missing_same_type_rows"], 1)
        self.assertEqual(result["score_groups_missing_cross_type_rows"], 1)
        self.assertEqual(result["score_cross_type_rows_needed"], 2)

    def test_complete_group_has_no_missing_rows(self) -> None:
        original = [row("s", "score", "g"), row("n", "noul", "g")]
        result = group_integrity(original, original)
        self.assertEqual(result["missing_task_type_groups"], 0)
        self.assertEqual(result["missing_any_row_groups"], 0)
        self.assertEqual(result["score_cross_type_rows_needed"], 0)

    def test_missing_same_type_row_without_missing_type(self) -> None:
        original = [row("a", "choice", "g"), row("b", "choice", "g")]
        result = group_integrity(original, original[:1])
        self.assertEqual(result["missing_task_type_groups"], 0)
        self.assertEqual(result["missing_any_row_groups"], 1)
        self.assertEqual(result["missing_same_type_rows"], 1)

    def test_rights_count_is_checked_for_each_source(self) -> None:
        train = [row("a", "score", "g"), {**row("b", "noul", "h"), "source": "other"}]
        manifest = {
            "schema_version": "decision2-rights-clean-splits/1",
            "counts": {"source": {"synthetic-test": 1, "other": 1}},
            "publication_eligible": True,
            "source_rights": [
                {"source": name, "rows": count, "license": "test", "evidence": "test"}
                for name, count in (("synthetic-test", 1), ("other", 1))
            ],
        }
        mapping = {"synthetic-test": "synthetic-test", "other": "other"}
        self.assertEqual(
            selected_rights(manifest, train, train, source_map=mapping)[
                "selected_source_count"
            ],
            2,
        )
        manifest["source_rights"][0]["rows"] = 2
        manifest["source_rights"][1]["rows"] = 0
        with self.assertRaisesRegex(AdmissionHold, "RIGHTS_PER_SOURCE_COUNT"):
            selected_rights(manifest, train, train, source_map=mapping)

    def test_common_scaffold_is_not_a_full_input_collision(self) -> None:
        def native(identifier: str, state: str) -> dict:
            result = {
                "id": identifier,
                "split": "train",
                "task_type": "choice",
                "state": state,
                "instructions": "Choose the best answer from these options using only the case.",
                "options": [
                    {"key": "a", "text": "The first option is correct."},
                    {"key": "b", "text": "The second option is correct."},
                ],
            }
            result["input_sha256"] = digest(
                {field: result[field] for field in INPUT_FIELDS}
            )
            return result

        selected = native(
            "train-a",
            "A unique long factual state about invoice A and a signed approval.",
        )
        unrelated = native(
            "protected-b",
            "A distinct long factual state about invoice B and a rejected approval.",
        )
        roles = {"test": [project_partition_row(unrelated, "train")]}
        self.assertEqual(
            distinctive_overlap([selected], roles)["status"],
            "PASS_BOUNDED_DISTINCTIVE_INPUT_SCREEN",
        )
        related = native("protected-c", selected["state"])
        roles = {"test": [project_partition_row(related, "train")]}
        self.assertEqual(
            distinctive_overlap([selected], roles)["status"],
            "HOLD_DISTINCTIVE_INPUT_OVERLAP",
        )


if __name__ == "__main__":
    unittest.main()
