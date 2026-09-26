import copy
import json
import tempfile
import unittest
from pathlib import Path

from training.model.data import (
    INPUT_FIELDS,
    check_partition_isolation,
    digest,
    load_partition,
    validate_row,
)


def row(identifier="r1", split="train", role="train", group="g1", task_type="choice"):
    options = [{"key": "a", "description": "A"}, {"key": "b", "description": "B"}]
    if task_type == "noul":
        options = [
            {"key": "false", "description": "No"},
            {"key": "true", "description": "Yes"},
        ]
    if task_type == "score":
        options = [
            {"key": "0", "description": "Low"},
            {"key": "1", "description": "High"},
        ]
    result = {
        "id": identifier,
        "state": {"case": identifier},
        "instructions": "Choose the supported answer",
        "options": options,
        "label": 1,
        "task_type": task_type,
        "family": "basic",
        "group_id": group,
        "language": "en",
        "split": split,
        "source": "licensed-source",
        "evaluation_role": role,
        "render_template": "typed-v1",
        "audit_metadata": {},
    }
    result["input_sha256"] = digest({field: result[field] for field in INPUT_FIELDS})
    return result


class DataContractTest(unittest.TestCase):
    def test_native_rows_and_replay_distribution(self):
        for kind in ("choice", "noul", "score"):
            example = row(task_type=kind)
            self.assertIs(validate_row(example, "train"), example)
            replay = copy.deepcopy(example)
            replay["teacher_probs"] = {
                option["key"]: 0.5 for option in replay["options"]
            }
            validate_row(replay, "train", replay=True)
            with self.assertRaisesRegex(ValueError, "only in a separate replay"):
                validate_row(replay, "train")
        structured = row()
        structured["options"][0]["description"] = {"rule": ["match", "exclude"]}
        structured["input_sha256"] = digest(
            {field: structured[field] for field in INPUT_FIELDS}
        )
        self.assertIs(validate_row(structured, "train"), structured)

    def test_rejects_gold_benchmark_shape_and_hash_mismatch(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            validate_row(
                {"id": "bench", "state": {}, "questions": {}, "gold": {}}, "train"
            )
        corrupted = row()
        corrupted["options"].reverse()
        with self.assertRaisesRegex(ValueError, "input_sha256"):
            validate_row(corrupted, "train")
        legacy = row()
        legacy["target_probs"] = [0.2, 0.8]
        with self.assertRaisesRegex(ValueError, "legacy soft-label"):
            validate_row(legacy, "train")

    def test_partition_roles_and_lineage_isolation(self):
        train = row()
        select = row("s1", "select", "select", "g2")
        cal = row("c1", "cal", "calibrate", "g3")
        check_partition_isolation({"train": [train], "select": [select], "cal": [cal]})
        crossed = copy.deepcopy(select)
        crossed["group_id"] = "g1"
        with self.assertRaisesRegex(ValueError, "group_id"):
            check_partition_isolation(
                {"train": [train], "select": [crossed], "cal": [cal]}
            )
        with self.assertRaisesRegex(ValueError, "expected split"):
            validate_row(train, "select")
        crossed = copy.deepcopy(select)
        crossed["state"] = train["state"]
        crossed["input_sha256"] = train["input_sha256"]
        validate_row(crossed, "select")
        with self.assertRaisesRegex(ValueError, "input_sha256"):
            check_partition_isolation({"train": [train], "select": [crossed]})

    def test_jsonl_duplicate_ids_and_blank_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.jsonl"
            text = json.dumps(row()) + "\n"
            path.write_text(text * 2, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate id"):
                load_partition(path, "train")
            path.write_text(text + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "blank line"):
                load_partition(path, "train")


if __name__ == "__main__":
    unittest.main()
