"""Final model layouts must agree with both HF and router label consumers."""

import json
import tempfile
import unittest
from pathlib import Path

from src.training.model_classifier.sequence_repair.runtime_mapping import (
    write_runtime_mappings,
)


class RuntimeMappingTests(unittest.TestCase):
    def test_domain_uses_category_loader_schema_and_same_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            labels = {"biology": 0, "other": 1}
            files = write_runtime_mappings(directory, labels, "domain")
            self.assertEqual(files, ["category_mapping.json", "label_mapping.json"])
            category = json.loads(
                (Path(directory) / "category_mapping.json").read_text()
            )
            generic = json.loads((Path(directory) / "label_mapping.json").read_text())
            self.assertEqual(category["category_to_idx"], generic["label2id"])
            self.assertEqual(category["idx_to_category"], generic["id2label"])

    def test_factcheck_polarity_is_preserved_in_both_directions(self):
        with tempfile.TemporaryDirectory() as directory:
            labels = {"FACT_CHECK_NEEDED": 0, "NO_FACT_CHECK_NEEDED": 1}
            write_runtime_mappings(directory, labels, "fact-check")
            mapping = json.loads(
                (Path(directory) / "fact_check_mapping.json").read_text()
            )
            self.assertEqual(mapping["label_to_idx"], labels)
            for label, index in labels.items():
                self.assertEqual(mapping["idx_to_label"][str(index)], label)

    def test_modality_cannot_reorder_native_label_indices(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "native"):
                write_runtime_mappings(
                    directory, {"AR": 1, "DIFFUSION": 0, "BOTH": 2}, "modality"
                )
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_prompt_guard_and_dynamic_labels_use_router_layout(self):
        for task in ("prompt-guard", "feedback", "safety", "hazard"):
            with self.subTest(task=task), tempfile.TemporaryDirectory() as directory:
                labels = {"safe": 0, "unsafe": 1}
                files = write_runtime_mappings(directory, labels, task)
                name = (
                    "jailbreak_type_mapping.json"
                    if task == "prompt-guard"
                    else "label_mapping.json"
                )
                self.assertIn(name, files)
                mapping = json.loads((Path(directory) / name).read_text())
                self.assertEqual(mapping["label_to_idx"], labels)
                self.assertEqual(mapping["idx_to_label"], {"0": "safe", "1": "unsafe"})

    def test_sparse_or_duplicate_ids_fail_before_writing(self):
        for labels in ({"a": 0, "b": 0}, {"a": 0, "b": 2}):
            with tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(ValueError, "bijection"):
                    write_runtime_mappings(directory, labels, "domain")
                self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
