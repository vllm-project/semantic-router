"""Validate native probability projection and schema-safe option mapping."""

from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

from inference.gliner25 import (
    MULTI_REVISION,
    native_context_limit,
    prepare_question,
    project,
    schema_labels,
    verify_release,
)


class ProjectionTest(unittest.TestCase):
    def test_boundary_context_uses_checkpoint_limit(self):
        encoder = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=512))
        boundary = SimpleNamespace(
            model=SimpleNamespace(config=SimpleNamespace(max_len=4096), encoder=encoder)
        )
        span = SimpleNamespace(
            model=SimpleNamespace(config=SimpleNamespace(max_len=None), encoder=encoder)
        )
        self.assertEqual(native_context_limit(boundary), 4096)
        self.assertEqual(native_context_limit(span), 512)
        with self.assertRaisesRegex(ValueError, "native context limit"):
            native_context_limit(
                SimpleNamespace(
                    model=SimpleNamespace(
                        config=SimpleNamespace(max_len=0), encoder=encoder
                    )
                )
            )

    def test_multilingual_profile_rejects_unpinned_revision(self):
        with self.assertRaisesRegex(ValueError, "attested pinned revision"):
            verify_release(
                Path("/does-not-exist"), MULTI_REVISION[:-1] + "0", "multilingual"
            )

    def test_choice_keeps_opaque_keys_and_probability_distribution(self):
        question = {
            "type": "choice",
            "instructions": "Choose one",
            "criteria": {"A (x)": "first", "none": "second"},
        }
        text, aliases = prepare_question({"x": 1}, question)
        self.assertIn("A (x)", text)
        self.assertEqual(aliases, [("candidate_0", "A (x)"), ("none", "none")])
        self.assertEqual(
            schema_labels(question, aliases), {"candidate_0": "first", "none": "second"}
        )
        answer = project(question, aliases, {"candidate_0": 0.8, "none": 0.2})
        self.assertEqual(answer["choice"], "A (x)")
        self.assertEqual(answer["probabilities"], {"A (x)": 0.8, "none": 0.2})

    def test_noul_and_score_use_native_distribution(self):
        nquestion = {"type": "noul", "instructions": "Is it active?"}
        _, naliases = prepare_question("active", nquestion)
        self.assertEqual(
            project(nquestion, naliases, {"yes": 0.7, "no": 0.3})["noul"], 0.7
        )
        squestion = {
            "type": "score",
            "instructions": "Rate it",
            "criteria": ["low", "mid", "high"],
        }
        _, saliases = prepare_question("text", squestion)
        answer = project(squestion, saliases, {"0": 0.1, "1": 0.2, "2": 0.7})
        self.assertAlmostEqual(answer["score"], 1.6)
        self.assertEqual(answer["native_level"], 2)

    def test_description_uses_native_schema_and_escapes_its_markers(self):
        question = {
            "type": "choice",
            "instructions": "Choose one",
            "criteria": {"A": "Threatening (speaker commitment)", "B": "No threat"},
        }
        text, aliases = prepare_question("Original document", question)
        self.assertEqual(text, "Original document")
        self.assertEqual(
            schema_labels(question, aliases),
            {"A": "Threatening : speaker commitment", "B": "No threat"},
        )

    def test_missing_or_dropped_label_is_rejected(self):
        question = {
            "type": "choice",
            "instructions": "Choose",
            "criteria": {"a": "a", "b": "b"},
        }
        _, aliases = prepare_question("x", question)
        with self.assertRaisesRegex(ValueError, "every option"):
            project(question, aliases, {"a": 1.0})


if __name__ == "__main__":
    unittest.main()
