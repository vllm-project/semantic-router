import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from src.training.model_classifier.domain_classifier.prepare_data import (
    canonical_groups,
    question_with_choices,
)
from src.training.model_classifier.modality_routing_classifier.extend_contracts import (
    build_additions,
    extend_corpus,
    recover_caption,
    validate_registry,
)
from src.training.model_classifier.sequence_repair.data import normalized_text


class SourceContractTests(unittest.TestCase):
    def test_translated_duplicate_groups_are_unioned_transitively(self):
        rows = [
            {"text": "First question", "group_id": "a", "label": "biology"},
            {"text": "FIRST  QUESTION", "group_id": "b", "label": "biology"},
            {"text": "Otra pregunta", "group_id": "b", "label": "biology"},
            {"text": "otra pregunta", "group_id": "c", "label": "biology"},
        ]
        retained, excluded = canonical_groups(rows)
        self.assertEqual(excluded, 0)
        self.assertEqual({row["group_id"] for row in retained}, {"a"})
        retained, excluded = canonical_groups(
            [*rows, {"text": "Otra pregunta", "group_id": "d", "label": "history"}]
        )
        self.assertEqual((retained, excluded), ([], 5))

    def test_domain_question_contains_choices_but_not_answer_key(self):
        row = {
            "question": "Which element?",
            "option_a": "iron",
            "option_b": "wood",
            "option_c": "paper",
            "option_d": "glass",
            "answer": "secret-answer-key",
        }
        text = question_with_choices(row)
        self.assertIn("A. iron", text)
        self.assertNotIn(row["answer"], text)

    def test_caption_recovery_keeps_embedded_colons_and_rejects_changed_source(self):
        caption = "A quiet stage: blue curtains, paper lanterns, soft light"
        row = {
            "language": "en",
            "template_family": "modality-train-0",
            "text": "Describe: " + caption,
            "caption_sha256": hashlib.sha256(
                normalized_text(caption).encode()
            ).hexdigest(),
        }
        templates = {"en": ["Describe: {}"]}
        self.assertEqual(recover_caption(row, templates), caption)
        with self.assertRaises(ValueError):
            recover_caption({**row, "text": row["text"] + " changed"}, templates)

    def test_caption_recovery_checks_both_wrapper_boundaries(self):
        caption = "Paper lanterns: a blue stage"
        templates = {"en": ["Quote {{scene}}: {} END"]}
        row = {
            "language": "en",
            "template_family": "modality-train-0",
            "text": templates["en"][0].format(caption),
            "caption_sha256": hashlib.sha256(
                normalized_text(caption).encode()
            ).hexdigest(),
        }
        self.assertEqual(recover_caption(row, templates), caption)
        for text in (row["text"][:-1], "changed " + row["text"]):
            with self.subTest(text=text), self.assertRaises(ValueError):
                recover_caption({**row, "text": text}, templates)

    def registry(self):
        return {
            "version": 1,
            "source_templates": {"en": ["Describe: {}"]},
            "contracts": {
                "en": {
                    "AR": ["Explain: {}"],
                    "DIFFUSION": ["Draw: {}"],
                    "BOTH": ["Draw and explain: {}"],
                }
            },
        }

    def test_registry_requires_known_labels_and_one_caption_field(self):
        for templates in ([], ["no field"], ["{} {}"], ["{caption}"], ["{!r}"]):
            registry = self.registry()
            registry["contracts"]["en"]["AR"] = templates
            with self.subTest(templates=templates), self.assertRaises(ValueError):
                validate_registry(registry)
        registry = self.registry()
        registry["contracts"]["en"]["UNKNOWN"] = registry["contracts"]["en"].pop("AR")
        with self.assertRaises(ValueError):
            validate_registry(registry)

    def test_explicit_registry_extends_only_training_and_binds_source_groups(self):
        caption = "Paper lanterns: blue stage\u2028soft lights"
        parent = {
            "id": "caption-1",
            "text": "Describe: " + caption,
            "label": "AR",
            "source": "authored-output-contract-with-gallery-caption",
            "group_id": "author-1",
            "language": "en",
            "template_family": "modality-train-0",
            "caption_sha256": hashlib.sha256(
                normalized_text(caption).encode()
            ).hexdigest(),
        }
        registry = self.registry()
        additions = build_additions([parent], registry)
        self.assertEqual(
            {row["label"] for row in additions}, {"AR", "DIFFUSION", "BOTH"}
        )
        self.assertEqual({row["group_id"] for row in additions}, {"author-1"})
        self.assertTrue(all(caption in row["text"] for row in additions))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source", root / "expanded"
            source.mkdir()
            splits = {
                "train": [parent],
                "dev": [{**parent, "id": "d", "group_id": "dev", "text": "Dev input"}],
                "test": [
                    {**parent, "id": "t", "group_id": "test", "text": "Test input"}
                ],
            }
            for split, rows in splits.items():
                (source / f"{split}.jsonl").write_text(
                    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
                )
            (source / "contract.json").write_text("{}")
            (source / "manifest.json").write_text("{}")
            registry_path = root / "registry.json"
            registry_path.write_text(json.dumps(registry))
            result = extend_corpus(source, output, registry_path)
            self.assertEqual(result["provenance"]["training_additions"], 3)
            self.assertEqual(
                result["provenance"]["registry_sha256"],
                hashlib.sha256(registry_path.read_bytes()).hexdigest(),
            )
            for split in ("dev", "test"):
                with (output / f"{split}.jsonl").open() as stream:
                    self.assertEqual(
                        [json.loads(line) for line in stream], splits[split]
                    )
            with self.assertRaisesRegex(ValueError, "overwrite"):
                extend_corpus(source, output, registry_path)


if __name__ == "__main__":
    unittest.main()
