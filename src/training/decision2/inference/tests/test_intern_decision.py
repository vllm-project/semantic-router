"""Intern-Decision adapter: native rejections stay invalid in the denominator."""

from __future__ import annotations

import unittest
from pathlib import Path

from inference.intern_decision import MODEL_REVISION, collect, respond


class FakeEngine:
    def predict(self, request):
        if "long" in request["state"]:
            raise ValueError(
                "Example has 9000 tokens, above 8192; truncation is forbidden"
            )
        return {
            "answers": {
                k: {"type": q["type"], "choice": "a"}
                for k, q in request["questions"].items()
            }
        }


class InternDecisionTest(unittest.TestCase):
    def test_rejection_becomes_invalid_answers(self) -> None:
        row = {"id": "x", "state": "long text", "questions": {"q": {"type": "choice"}}}
        answers, error = respond(FakeEngine(), row)
        self.assertEqual(
            answers, {"q": {"type": "choice", "invalid_reason": "native_rejection"}}
        )
        self.assertIn("truncation is forbidden", error)
        answers, error = respond(FakeEngine(), {**row, "state": "short"})
        self.assertIsNone(error)
        self.assertEqual(answers["q"]["choice"], "a")

    def test_release_module_with_dataclass_loads(self) -> None:
        import tempfile

        from inference.intern_decision import load_engine

        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "inference.py").write_text(
                "from dataclasses import dataclass\n"
                "@dataclass(frozen=True)\nclass C:\n    x: int = 1\n"
                "DEFAULT_TEMPERATURE = 2.0\n"
                "class DecisionEngine:\n    def __init__(self, path, device):\n        self.c = C()\n"
            )
            module, engine = load_engine(Path(tmp), "cpu")
            self.assertEqual(engine.c.x, 1)

    def test_rejects_other_revision(self) -> None:
        with self.assertRaisesRegex(ValueError, "pinned model revision"):
            collect(
                model_path=Path("missing"),
                revision=MODEL_REVISION + "x",
                prompts=Path("p"),
                output=Path("o"),
                device="cpu",
            )


if __name__ == "__main__":
    unittest.main()
