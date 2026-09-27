"""Synthetic contract tests; no real protected prompts or labels are loaded."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from publication.package_native_arena import input_digest, load_gold_free

from training.data.audit_27b_full_input_overlap import (
    audit_core_full_input,
    full_input_overlap_rows,
)
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


class FullInputOverlapTests(unittest.TestCase):
    def test_question_only_overlap_is_visible_beyond_state(self) -> None:
        question = "Which signed certificate verifies this shipment before release?"
        left = [
            {
                "id": "train-id",
                "state": "Shipment A has a long unique history.",
                "instructions": question,
            }
        ]
        right = [
            {
                "id": "eval-id",
                "state": "Unrelated state for shipment B.",
                "instructions": json.dumps({"label": question}),
            }
        ]
        result = full_input_overlap_rows(left, right)
        self.assertGreater(result["counts"]["exact_normalized"], 0)
        self.assertNotIn("train-id", json.dumps(result))
        self.assertNotIn(question, json.dumps(result))

    def test_option_only_overlap_is_visible_beyond_state(self) -> None:
        option = "The request lacks the required signed proof of current ownership."
        left = [
            {
                "id": "train",
                "state": "Unique train state.",
                "options": [{"key": "a", "description": option}],
            }
        ]
        right = [
            {
                "id": "eval",
                "state": "Unique eval state.",
                "options": json.dumps([{"key": "z", "description": option}]),
            }
        ]
        self.assertGreater(
            full_input_overlap_rows(left, right)["counts"]["exact_normalized"], 0
        )

    def test_near_question_overlap_is_detected(self) -> None:
        question = (
            "The customs officer compared invoice numbers, container seals, "
            "ownership certificates, dated signatures, warehouse logs, and "
            "inspection records before deciding whether the shipment qualified "
            "for release."
        )
        changed = question.replace("shipment", "shipmentx", 1)
        left = [
            {"id": "train", "state": "Unique train state.", "instructions": question}
        ]
        right = [{"id": "eval", "state": "Unique eval state.", "instructions": changed}]
        result = full_input_overlap_rows(left, right)
        self.assertGreater(result["counts"]["near"], 0)

    def test_partition_absence_is_explicit_hold(self) -> None:
        roles = {
            role: [{"id": role, "state": "Synthetic placeholder input"}]
            for role in NATIVE_ROLE_COUNTS
        }
        result = audit_core_full_input([], roles)
        self.assertEqual(result["status"], "HOLD_MISSING_PARTITION_ROLES")
        self.assertEqual(set(result["missing_roles"]), set(PARTITION_ROLE_COUNTS))

    def test_core_pass_and_schedule_identity_hold(self) -> None:
        source = {
            "id": "train-row",
            "split": "train",
            "state": "A regulatory claim concerns blue lantern imports and customs duty.",
            "instructions": "Decide whether the special exemption applies to this blue lantern.",
            "options": [
                {"key": "a", "description": "A customs exemption applies here."},
                {"key": "b", "description": "The usual customs duty applies here."},
            ],
            "task_type": "choice",
        }
        source["input_sha256"] = digest(
            {field: source[field] for field in INPUT_FIELDS}
        )
        roles = {
            role: [{"id": role, "state": "Distinct synthetic evidence for " + role * 8}]
            for role in NATIVE_ROLE_COUNTS
        }
        roles["rights_clean_train"] = [project_partition_row(source, "train")]
        roles["rights_clean_select"] = [
            {
                "id": "select-id",
                "state": "An unrelated orange bicycle ownership record with many clauses.",
            }
        ]
        roles["rights_clean_cal"] = [
            {
                "id": "cal-id",
                "state": "A separate purple compass service agreement for repair.",
            }
        ]
        with patch.dict(NATIVE_ROLE_COUNTS, dict.fromkeys(NATIVE_ROLE_COUNTS, 1)):
            with patch.dict(
                PARTITION_ROLE_COUNTS,
                {
                    role: (split, 1)
                    for role, (split, _) in PARTITION_ROLE_COUNTS.items()
                },
            ):
                result = audit_core_full_input([source], roles)
                self.assertEqual(result["status"], "PASS_BOUNDED_FULL_INPUT_SCREEN")
                changed = dict(source, instructions="A different instruction")
                changed["input_sha256"] = digest(
                    {field: changed[field] for field in INPUT_FIELDS}
                )
                self.assertEqual(
                    audit_core_full_input([changed], roles)["status"],
                    "HOLD_SCHEDULE_INPUT_IDENTITY",
                )
                self.assertEqual(
                    audit_core_full_input([source], {**roles, "unvetted": []})[
                        "status"
                    ],
                    "HOLD_UNATTESTED_OPTIONAL_ROLES",
                )


if __name__ == "__main__":
    unittest.main()
