"""Synthetic contract tests; no real protected prompts or labels are loaded."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from publication.package_native_arena import input_digest, load_gold_free

from training.data.audit_27b_teacher_admission import _contains_gold_key
from training.data.plan_goldfree_inventory import (
    CORE_ROLES,
    NATIVE_ROLE_COUNTS,
    PARTITION_ROLE_COUNTS,
    _project_native_rows,
    project_native_file,
    project_partition_row,
    project_partition_rows,
    projected_jsonl,
    validate_core_rows,
)
from training.model.data import INPUT_FIELDS, digest


class ProtectedInventoryProjectionTests(unittest.TestCase):
    def _sealed(self, rows: list[dict]) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "sealed.jsonl"
        path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
            encoding="utf-8",
        )
        return path

    def test_native_inputs_survive_projection_without_nested_answer_keys(self) -> None:
        rows = [
            {
                "id": "css-input",
                "state": "Review the synthetic fact.",
                "questions": {
                    "label": {
                        "criteria": {
                            "answer": "a requested criterion, not a gold value"
                        },
                        "options": ["yes", "no"],
                    }
                },
            },
            {
                "id": "typed-input",
                "state": {"target": {"entity": "widget", "item": "certificate"}},
                "questions": {"question-id": "Is there enough evidence?"},
            },
        ]
        source = load_gold_free(self._sealed(rows))
        projected, identities = _project_native_rows(source, 2)
        self.assertEqual(identities, [input_digest(row) for row in rows])
        self.assertEqual(json.loads(projected[0]["instructions"]), rows[0]["questions"])
        self.assertEqual(json.loads(projected[1]["state"]), rows[1]["state"])
        payload = [json.loads(line) for line in projected_jsonl(projected).splitlines()]
        self.assertEqual(payload, projected)
        self.assertFalse(any(_contains_gold_key(row) for row in payload))

    def test_native_loader_rejects_answer_payload_before_projection(self) -> None:
        bad = {
            "id": "bad",
            "state": {"answer": "gold"},
            "questions": {"q": "Question text"},
        }
        with self.assertRaisesRegex(ValueError, "state answer field"):
            load_gold_free(self._sealed([bad]))

    def test_native_file_requires_pinned_hash(self) -> None:
        row = {"id": "x", "state": "State", "questions": {"q": "Question"}}
        with self.assertRaisesRegex(ValueError, "hash changed"):
            project_native_file(self._sealed([row]), "typed_dev", "0" * 64)

    def test_partition_projection_ignores_target_and_preserves_input_identity(
        self,
    ) -> None:
        row = {
            "id": "train-synthetic",
            "split": "train",
            "state": "Synthetic context",
            "instructions": {"target": "task input, not gold"},
            "options": [{"key": "a", "description": "candidate"}],
            "task_type": "choice",
            "label": object(),  # a deliberate nonserializable target sentinel
        }
        row["input_sha256"] = digest({field: row[field] for field in INPUT_FIELDS})
        projected = project_partition_row(row, "train")
        self.assertEqual(projected["id"], row["id"])
        self.assertEqual(json.loads(projected["options"]), row["options"])
        self.assertFalse(_contains_gold_key(projected))
        with self.assertRaisesRegex(ValueError, "Partition role"):
            project_partition_row(row, "select")
        row["input_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "Partition role"):
            project_partition_row(row, "train")

    def test_partition_role_requires_exact_count_and_unique_ids(self) -> None:
        row = {
            "id": "synthetic",
            "split": "select",
            "state": "Task input",
            "instructions": "Choose one",
            "options": [{"key": "a", "description": "A"}],
            "task_type": "choice",
        }
        row["input_sha256"] = digest({field: row[field] for field in INPUT_FIELDS})
        with patch.dict(PARTITION_ROLE_COUNTS, {"rights_clean_select": ("select", 1)}):
            self.assertEqual(
                project_partition_rows([row], "rights_clean_select"),
                [project_partition_row(row, "select")],
            )
            with self.assertRaisesRegex(ValueError, "count changed"):
                project_partition_rows([], "rights_clean_select")
        with patch.dict(PARTITION_ROLE_COUNTS, {"rights_clean_select": ("select", 2)}):
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                project_partition_rows([row, row], "rights_clean_select")

    def test_core_coverage_and_row_shape_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "incomplete"):
            validate_core_rows({})
        with patch.dict(NATIVE_ROLE_COUNTS, dict.fromkeys(NATIVE_ROLE_COUNTS, 1)):
            with patch.dict(
                PARTITION_ROLE_COUNTS,
                {
                    role: (split, 1)
                    for role, (split, _) in PARTITION_ROLE_COUNTS.items()
                },
            ):
                roles = {
                    role: [{"id": role, "state": "synthetic input"}]
                    for role in CORE_ROLES
                }
                self.assertEqual(set(validate_core_rows(roles)), CORE_ROLES)
                roles["rights_clean_cal"] = [
                    {"id": "x", "state": {"label": "not allowed"}}
                ]
                with self.assertRaisesRegex(ValueError, "strict input-only"):
                    validate_core_rows(roles)


if __name__ == "__main__":
    unittest.main()
