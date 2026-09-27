"""Focused CPU admission contracts; synthetic fixtures contain no model data."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from training.data import audit_27b_teacher_admission as audit
from training.model.data import file_sha256


def row(name: str, task: str, source: str, group: str, label: int = 0) -> dict:
    keys = ["false", "true"] if task == "noul" else ["0", "1", "2"]
    return {
        "id": name,
        "input_sha256": name + "-input",
        "source": source,
        "group_id": group,
        "task_type": task,
        "family": "synthetic-test",
        "language": "en",
        "options": [{"key": key} for key in keys],
        "label": label,
        "state": "Synthetic state " + name,
    }


class AdmissionContracts(unittest.TestCase):
    def test_largest_remainder_deterministic(self) -> None:
        self.assertEqual(audit.largest_remainder({"b": 3, "a": 3}, 3), {"b": 1, "a": 2})
        with self.assertRaisesRegex(audit.AdmissionHold, "QUOTA_CAPACITY"):
            audit.largest_remainder({"a": 1}, 2)

    def test_whole_group_schedule_same_ids_and_exact_budget(self) -> None:
        score = [row("s0", "score", "synth", "sg0"), row("s1", "score", "synth", "sg1")]
        choice = [row(f"c{i}", "choice", "human", f"cg{i}") for i in range(4)]
        noul = [row(f"n{i}", "noul", "human", f"ng{i}") for i in range(4)]
        rows = score + choice + noul
        lengths = {item["id"]: 100 for item in rows}
        first, first_profile = audit.select_schedule(
            rows, lengths, total=8, min_score=2
        )
        second, second_profile = audit.select_schedule(
            rows, lengths, total=8, min_score=2
        )
        self.assertEqual([x["id"] for x in first], [x["id"] for x in second])
        self.assertEqual(first_profile, second_profile)
        self.assertEqual(first_profile["by_type"], {"choice": 3, "noul": 3, "score": 2})
        self.assertEqual(first_profile["raw_token_exposure"], 800)
        self.assertEqual(first_profile["rows"], 8)
        with self.assertRaisesRegex(audit.AdmissionHold, "SCORE_ADMITTED_COUNT"):
            audit.select_schedule(rows, {**lengths, "s0": 4097}, total=8, min_score=2)

    def test_whole_group_exact_subset_avoids_false_greedy_hold(self) -> None:
        groups = [
            [row("a0", "choice", "human", "a"), row("a1", "choice", "human", "a")],
            [row("b0", "choice", "human", "b"), row("b1", "choice", "human", "b")],
            [row("c0", "choice", "human", "c")],
        ]
        selected = audit.choose_whole_groups(groups, 3)
        self.assertEqual(len(selected), 3)
        self.assertEqual(len({item["group_id"] for item in selected}), 2)
        with self.assertRaisesRegex(audit.AdmissionHold, "WHOLE_GROUP_QUOTA"):
            audit.choose_whole_groups(groups[:2], 3)

    def test_teacher_row_and_option_identity_are_strict(self) -> None:
        sample = row("x", "score", "synth", "g")
        artifact = {
            "schema": "decision2-autojev-score-teacher-distributions/1",
            "source": audit.TEACHER_ID,
            "source_revision": audit.TEACHER_SOURCE_REVISION,
            "native_model_sha256": audit.TEACHER_MODEL_SHA256,
            "train_sha256": audit.TRAIN_SHA256,
            "roster_sha256": audit.SCORE_ROSTER_SHA256,
            "rows": [
                {
                    "id": "x",
                    "input_sha256": "x-input",
                    "group_id": "g",
                    "source": "synth",
                    "family": "synthetic-test",
                    "level_count": 3,
                    "probabilities": {"0": 0.7, "1": 0.2, "2": 0.1},
                }
            ],
        }
        self.assertEqual(
            audit.verify_teacher(artifact, [sample], task="score")["x"]["0"], 0.7
        )
        artifact["rows"][0]["probabilities"] = {"0": 0.7, "1": 0.3, "wrong": 0}
        with self.assertRaisesRegex(audit.AdmissionHold, "TEACHER_OPTION_KEYS"):
            audit.verify_teacher(artifact, [sample], task="score")
        artifact["rows"][0]["probabilities"] = {"0": 0.7, "1": 0.2, "2": 0.1}
        artifact["rows"][0]["input_sha256"] = "changed"
        with self.assertRaisesRegex(audit.AdmissionHold, "TEACHER_ROW_IDENTITY"):
            audit.verify_teacher(artifact, [sample], task="score")

    def test_mask_uses_train_gold_and_genuine_source_only(self) -> None:
        score = row("s", "score", "synth", "g", label=1)
        human = row("h", "choice", "google_goemotions_official_train", "h", label=2)
        synthetic = row("c", "choice", "legacy:stage4-general-composition-v2", "c")
        vectors = {
            "s": {"0": 0.2, "1": 0.6, "2": 0.2},
            "h": {"0": 0.1, "1": 0.1, "2": 0.8},
            "c": {"0": 0.9, "1": 0.05, "2": 0.05},
        }
        profile = audit.teacher_mask(
            [score, human, synthetic],
            vectors,
            minimum_score=1,
            minimum_three_level=1,
            minimum_human=1,
        )
        self.assertEqual(profile["by_type"], {"choice": 1, "score": 1})
        self.assertEqual(profile["masked_rows"], 2)
        vectors["s"] = {"0": 0.6, "1": 0.2, "2": 0.2}
        with self.assertRaisesRegex(audit.AdmissionHold, "MASK_SCORE_MINIMUM"):
            audit.teacher_mask(
                [score, human, synthetic],
                vectors,
                minimum_score=1,
                minimum_three_level=1,
                minimum_human=1,
            )

    def test_rights_source_counts_bound_to_train(self) -> None:
        rows = [row("a", "score", "synthetic", "g")]
        manifest = {
            "schema_version": "decision2-rights-clean-splits/1",
            "counts": {"source": {"synthetic": 1}},
            "publication_eligible": True,
            "source_rights": [
                {
                    "source": "synthetic",
                    "rows": 1,
                    "license": "internally generated",
                    "evidence": "test",
                    "partition_scope": "TRAIN",
                }
            ],
        }
        self.assertEqual(audit.verify_rights(manifest, rows)["train_rows"], 1)
        manifest["source_rights"][0]["rows"] = 0
        with self.assertRaisesRegex(audit.AdmissionHold, "RIGHTS_ROW_TOTAL"):
            audit.verify_rights(manifest, rows)

    def test_prompt_only_inventory_and_overlap_result_contain_no_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entries = []
            for role in sorted(audit.REQUIRED_ROLES):
                path = root / f"{role}.jsonl"
                path.write_text(
                    json.dumps({"id": role, "state": "Unrelated state " + role}) + "\n"
                )
                entries.append(
                    {"role": role, "path": str(path), "sha256": file_sha256(path)}
                )
            inventory = root / "inventory.json"
            inventory.write_text(json.dumps(entries))
            roles, identity = audit.read_prompt_inventory(inventory)
            self.assertEqual(identity["role_count"], len(audit.REQUIRED_ROLES))
            result = audit.audit_overlap([row("z", "score", "source", "g")], roles)
            self.assertEqual(result["status"], "PASS_BOUNDED_OBSERVABLE_SCREEN")
            self.assertNotIn("Unrelated state", json.dumps(result))
            roles["typed_dev"][0]["state"] = "Synthetic state z"
            self.assertEqual(
                audit.audit_overlap([row("z", "score", "source", "g")], roles)[
                    "status"
                ],
                "HOLD_CANDIDATE_OVERLAP",
            )
            entries[0]["sha256"] = "0" * 64
            inventory.write_text(json.dumps(entries))
            with self.assertRaisesRegex(audit.AdmissionHold, "PROTECTED_HASH"):
                audit.read_prompt_inventory(inventory)
            entries[0]["sha256"] = file_sha256(root / f"{entries[0]['role']}.jsonl")
            path = root / f"{entries[0]['role']}.jsonl"
            path.write_text(
                json.dumps({"id": entries[0]["role"], "state": "x", "label": 1}) + "\n"
            )
            entries[0]["sha256"] = file_sha256(path)
            inventory.write_text(json.dumps(entries))
            with self.assertRaisesRegex(audit.AdmissionHold, "PROTECTED_GOLD_FIELD"):
                audit.read_prompt_inventory(inventory)
            path.write_text(
                json.dumps(
                    {"review_id": "r0", "state": "x", "questions": [{"gold": 0}]}
                )
                + "\n"
            )
            entries[0]["sha256"] = file_sha256(path)
            inventory.write_text(json.dumps(entries))
            with self.assertRaisesRegex(
                audit.AdmissionHold, "PROTECTED_NESTED_GOLD_FIELD"
            ):
                audit.read_prompt_inventory(inventory)

    def test_private_receipt_is_exclusive_and_mode_0600(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "sub" / "receipt.json"
            audit.write_once(output, {"status": "HOLD", "reason_code": "TEST"})
            self.assertEqual(os.stat(output).st_mode & 0o777, 0o600)
            self.assertEqual(os.stat(output.parent).st_mode & 0o777, 0o700)
            with self.assertRaises(FileExistsError):
                audit.write_once(output, {"status": "PASS"})


if __name__ == "__main__":
    unittest.main()
