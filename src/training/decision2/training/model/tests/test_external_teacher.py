"""Tests for pinned TRAIN-only external teacher eligibility and attachment."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from training.model import external_teacher as teacher
from training.model.data import digest, file_sha256


def row(kind: str, ident: str, source: str, label: int) -> dict:
    keys = ["false", "true"] if kind == "noul" else ["a", "b"]
    if kind == "score":
        keys = ["0", "1", "2"]
    return {
        "id": ident,
        "input_sha256": ident * 64,
        "group_id": ident,
        "task_type": kind,
        "source": source,
        "family": source,
        "language": "en",
        "options": [{"key": key} for key in keys],
        "label": label,
    }


def entry(item: dict, probabilities: dict[str, float]) -> dict:
    return {
        key: item[key]
        for key in (
            "id",
            "input_sha256",
            "group_id",
            "task_type",
            "source",
            "family",
            "language",
        )
    } | {"option_count": len(item["options"]), "probabilities": probabilities}


class ExternalTeacherTests(unittest.TestCase):
    def test_eligibility_requires_gold_agreement_and_source(self) -> None:
        choice = row("choice", "a", "natural", 0)
        self.assertTrue(teacher.is_eligible(choice, {"a": 0.7, "b": 0.3}))
        self.assertFalse(teacher.is_eligible(choice, {"a": 0.69, "b": 0.31}))
        self.assertFalse(teacher.is_eligible(choice, {"a": 0.5, "b": 0.5}))
        self.assertFalse(teacher.is_eligible(choice, {"a": 0.2, "b": 0.8}))
        choice["source"] = teacher.EXCLUDED_SOURCE
        self.assertFalse(teacher.is_eligible(choice, {"a": 0.9, "b": 0.1}))
        self.assertFalse(
            teacher.is_eligible(row("score", "s", "natural", 0), {"0": 1.0})
        )

    def test_validates_every_row_before_mutating_and_leaves_score_hard(self) -> None:
        train = [
            row("choice", "a", "natural", 0),
            row("choice", "b", "natural", 0),
            row("noul", "c", "natural", 1),
            row("score", "d", "natural", 0),
        ]
        vectors = [
            entry(train[0], {"a": 0.8, "b": 0.2}),
            entry(train[1], {"a": 0.2, "b": 0.8}),
            entry(train[2], {"false": 0.1, "true": 0.9}),
        ]
        selected = [train[0], train[2]]
        selected_sha = digest(
            [
                {"id": item["id"], "input_sha256": item["input_sha256"]}
                for item in selected
            ]
        )
        payload = {
            "schema": teacher.ARTIFACT_SCHEMA,
            "source": teacher.TEACHER_SOURCE,
            "source_revision": teacher.TEACHER_SOURCE_REVISION,
            "native_model_sha256": teacher.TEACHER_MODEL_SHA256,
            "runtime_source_sha256": teacher.TEACHER_SOURCE_SHA256,
            "model_config_sha256": teacher.TEACHER_CONFIG_SHA256,
            "script_sha256": teacher.TEACHER_SCRIPT_SHA256,
            "loaded_parameters": 26_086_635_760,
            "rights_manifest_sha256": teacher.RIGHTS_MANIFEST_SHA256,
            "train_sha256": "test-train",
            "roster_sha256": teacher.TEACHER_ROSTER_SHA256,
            "rows": vectors,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teacher.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with (
                patch.object(teacher, "ARTIFACT_SHA256", file_sha256(path)),
                patch.object(teacher, "TRAIN_SHA256", "test-train"),
                patch.object(teacher, "STUDENT_SOURCE_FILES", {"source": "test"}),
                patch.object(teacher, "EXPECTED_TOTAL_TRAIN", 4),
                patch.object(teacher, "EXPECTED_TYPED_TRAIN", 3),
                patch.object(teacher, "EXPECTED_ELIGIBLE", {"choice": 1, "noul": 1}),
                patch.object(teacher, "ELIGIBLE_ROSTER_SHA256", selected_sha),
            ):
                summary = teacher.attach_external_teacher(
                    path,
                    train,
                    train_sha256="test-train",
                    source_files_sha256={"source": "test"},
                )
                self.assertEqual(summary["attached"], 2)
                self.assertIn("teacher_probs", train[0])
                self.assertNotIn("teacher_probs", train[1])
                self.assertIn("teacher_probs", train[2])
                self.assertNotIn("teacher_probs", train[3])
                for item in train:
                    item.pop("teacher_probs", None)
                payload["rows"][1]["probabilities"] = {"wrong": 1.0}
                path.write_text(json.dumps(payload), encoding="utf-8")
                with patch.object(teacher, "ARTIFACT_SHA256", file_sha256(path)):
                    with self.assertRaisesRegex(ValueError, "option parity"):
                        teacher.attach_external_teacher(
                            path,
                            train,
                            train_sha256="test-train",
                            source_files_sha256={"source": "test"},
                        )
                self.assertTrue(all("teacher_probs" not in item for item in train))


if __name__ == "__main__":
    unittest.main()
