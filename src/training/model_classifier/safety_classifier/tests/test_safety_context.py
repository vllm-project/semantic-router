"""Safety context construction preserves labels, payload bytes and split boundaries."""

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.training.model_classifier.safety_classifier import (
    vela_safety_context as context,
)


class SafetyContextTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(context.DEFAULT_CONFIG.read_text())

    def test_quoted_endorsement_and_protection_have_grouped_opposite_labels(self):
        parts = context.authored_rows(self.config)
        self.assertEqual(
            {key: len(value) for key, value in parts.items()}, {"train": 32, "dev": 16}
        )
        for split, rows in parts.items():
            groups = {row["group_id"] for row in rows}
            self.assertEqual(len(groups), 8 if split == "train" else 4)
            for group in groups:
                members = [row for row in rows if row["group_id"] == group]
                self.assertEqual(
                    {(row["label"], row["language"]) for row in members},
                    {
                        ("safe", "en"),
                        ("unsafe", "en"),
                        ("safe", "zh"),
                        ("unsafe", "zh"),
                    },
                )
                self.assertTrue(
                    all(
                        row["review_reason"] and row["license"] == "CC0-1.0"
                        for row in members
                    )
                )
        self.assertFalse(
            {r["group_id"] for r in parts["train"]}
            & {r["group_id"] for r in parts["dev"]}
        )

    def test_authored_assignments_and_explicit_labels_are_locked(self):
        changed = copy.deepcopy(self.config)
        changed["families"][0]["split"] = "dev"
        with self.assertRaisesRegex(ValueError, "assignments"):
            context.authored_rows(changed)
        for field, value in (("label", "safe"), ("rationale", ""), ("zh", "")):
            changed = copy.deepcopy(self.config)
            changed["families"][0]["variants"][0][field] = value
            with self.assertRaises(ValueError):
                context.authored_rows(changed)

    def test_background_placement_does_not_depend_on_label_or_translation(self):
        rows = context.authored_rows(self.config)["train"][:4]
        generated, _ = context.contextualize(
            rows, self.config, [1024, 2048], lambda text: len(text) + 2
        )
        self.assertEqual(len(generated), 8)
        by_id = {row["id"]: row for row in rows}
        for budget in ("1024", "2048"):
            selected = [row for row in generated if row["length_bucket"] == budget]
            self.assertEqual(
                len({(row["background_family"], row["position"]) for row in selected}),
                1,
            )
        for row in generated:
            original = by_id[row["parent_id"]]
            self.assertEqual(
                row["text"][row["payload_start"] : row["payload_end"]], original["text"]
            )
            self.assertEqual(row["group_id"], original["group_id"])
            self.assertEqual(row["label"], original["label"])
            self.assertLessEqual(row["actual_tokens"], int(row["length_bucket"]))
            self.assertNotIn(" note note", row["text"])
            self.assertEqual(
                row["parent_text_sha256"],
                hashlib.sha256(original["text"].encode()).hexdigest(),
            )

    def test_oversize_and_no_room_are_explained_without_truncation(self):
        row = context.authored_rows(self.config)["train"][0]
        original = copy.deepcopy(row)
        generated, excluded = context.contextualize(
            [row], self.config, [1, len(row["text"]) + 2], lambda text: len(text) + 2
        )
        self.assertFalse(generated)
        self.assertEqual(
            {item["reason"] for item in excluded},
            {"payload exceeds budget", "no complete background paragraph fits"},
        )
        self.assertEqual(row, original)

    def test_development_payloads_and_unsupported_languages_are_rejected(self):
        row = context.authored_rows(self.config)["dev"][0]
        with self.assertRaisesRegex(ValueError, "training payloads"):
            context.contextualize([row], self.config, [1024], len)
        row = context.authored_rows(self.config)["train"][0]
        row["language"] = "ja"
        with self.assertRaisesRegex(ValueError, "language"):
            context.contextualize([row], self.config, [1024], len)
        row["language"] = "source_unspecified"
        generated, _ = context.contextualize([row], self.config, [1024], len)
        self.assertEqual(generated[0]["background_language"], "en")
        self.assertEqual(generated[0]["language"], "source_unspecified")

    def test_invalid_backgrounds_and_duplicate_budgets_fail(self):
        for backgrounds in (
            self.config["backgrounds"][:1],
            [self.config["backgrounds"][0]] * 2,
        ):
            changed = copy.deepcopy(self.config)
            changed["backgrounds"] = backgrounds
            with self.assertRaises(ValueError):
                context.validate_backgrounds(changed)
        rows = context.authored_rows(self.config)["train"]
        for budgets in ([512, 512], [0], [-1], [True]):
            with self.assertRaises(ValueError):
                context.contextualize(rows, self.config, budgets, len)

    def test_frozen_outputs_reproduce_and_protect_split_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample = {
                "id": "source-train",
                "group_id": "source-family",
                "text": "Please describe the neutral room layout.",
                "label": "safe",
                "source": "source",
                "source_split": "train",
                "language": "en",
            }
            training = root / "train.jsonl"
            development = root / "dev.jsonl"
            training.write_text(json.dumps(sample) + "\n")
            development.write_text(
                json.dumps(
                    {
                        **sample,
                        "id": "source-dev",
                        "group_id": "other-family",
                        "text": "Describe a different place.",
                        "source_split": "dev",
                    }
                )
                + "\n"
            )
            args = (
                context.DEFAULT_CONFIG,
                [training],
                [training],
                [development],
                [1024],
                lambda text: len(text) + 2,
            )
            first = context.freeze(root / "one", *args)
            second = context.freeze(root / "two", *args)
            self.assertEqual(first, second)
            self.assertFalse(first["candidate_predictions_used_to_assign_labels"])
            self.assertTrue(first["development_diagnostics_informed_curriculum_design"])
            for name, expected in first["files"].items():
                self.assertEqual(
                    hashlib.sha256((root / "one" / name).read_bytes()).hexdigest(),
                    expected,
                )
                self.assertEqual(
                    (root / "one" / name).read_bytes(),
                    (root / "two" / name).read_bytes(),
                )
            with self.assertRaises(FileExistsError):
                context.freeze(root / "one", *args)
            development.write_text(
                json.dumps({**sample, "id": "other-id", "source_split": "dev"}) + "\n"
            )
            with self.assertRaisesRegex(ValueError, "overlap"):
                context.freeze(root / "overlap", *args)

    def test_context_only_payload_cannot_hide_development_text_in_background(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = {
                **context.authored_rows(self.config)["train"][0],
                "id": "source-context",
                "group_id": "source-context-family",
                "text": "Explain how to compare two neutral room layouts.",
                "label": "safe",
            }
            rows = {
                "replay": {
                    **payload,
                    "id": "unrelated-replay",
                    "group_id": "unrelated-family",
                    "text": "Describe the meeting room layout.",
                },
                "context": payload,
                "dev": {
                    **payload,
                    "id": "different-id",
                    "group_id": "different-family",
                    "source_split": "dev",
                    "text": "  " + payload["text"].upper() + "  ",
                },
            }
            for name, row in rows.items():
                (root / f"{name}.jsonl").write_text(json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "overlap in text"):
                context.freeze(
                    root / "output",
                    context.DEFAULT_CONFIG,
                    [root / "replay.jsonl"],
                    [root / "context.jsonl"],
                    [root / "dev.jsonl"],
                    [1024],
                    lambda text: len(text) + 2,
                )
            self.assertFalse((root / "output").exists())


if __name__ == "__main__":
    unittest.main()
