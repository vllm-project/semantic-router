from __future__ import annotations

import ast
import hashlib
import unittest

from publication.product_card import ProductCard, render

TABLE = """# Decision 2.0: JevArena v3 first-release panel

JevArena v3 scores 8,147 items. JevBench v1.2 public is separate.

| Rank | Model | Parameters | JevArena v3 ↑ | JevBench public ↑ |
| ---: | --- | ---: | ---: | ---: |
| 1 | Candidate | 25.688B | 60.00 | 200/231 |
"""


def specimen(**changes: object) -> ProductCard:
    values = {
        "model_id": "llm-semantic-router/DEV2.0-27B",
        "banner": "assets/DEV2.0-27B-banner.svg",
        "tagline": "One request, three kinds of evidence-aware decision.",
        "measured_summary": "The same-panel result is shown below.",
        "use_cases": ("Route requests", "Check evidence", "Apply ordered rubrics"),
        "direct_weight_source": "Qwen/Qwen3.8-27B",
        "source_revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        "loaded_parameters": 25688227840,
        "method": "A shared decision head was trained on labeled decision tasks.",
        "limitations": ("No live retrieval", "Review high-stakes decisions"),
        "evidence_table": TABLE,
        "evaluation_scope": "This same-panel comparison is not an unseen blind test.",
    }
    values.update(changes)
    return ProductCard(**values)


class ProductCardTest(unittest.TestCase):
    def test_product_first_and_three_type_example(self) -> None:
        card = render(specimen())
        self.assertLess(
            card.index("## Measured decisions"),
            card.index("## Three ways to decide"),
        )
        self.assertLess(
            card.index("## Three ways to decide"), card.index("## Download and decide")
        )
        self.assertLess(
            card.index("## Download and decide"), card.index("## Architecture")
        )
        self.assertIn("| Choice | Route requests |", card)
        self.assertIn("| Noul | Check evidence |", card)
        self.assertIn("| Score | Apply ordered rubrics |", card)
        self.assertIn("base_model: Qwen/Qwen3.8-27B", card)
        self.assertIn("25,688,227,840 parameters", card)
        self.assertNotIn("1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0", card)
        self.assertIn("[evaluation manifest](evaluation/manifest.json)", card)
        example = card.split("```python\n", 1)[1].split("\n```", 1)[0]
        ast.parse(example)
        for kind in ("choice", "noul", "score"):
            self.assertIn(f'"type": "{kind}"', example)
        self.assertNotIn("internal gate", card.lower())
        self.assertIn("jevarena-task-matrix.svg", card)
        self.assertIn("jevarena-rank.svg", card)
        self.assertIn("jevbench-public-rank.svg", card)
        self.assertNotIn("pareto", card.lower())

    def test_06b_example_preserves_previously_exercised_code(self) -> None:
        card = render(
            specimen(
                model_id="llm-semantic-router/DEV2.0-0.6B",
                direct_weight_source="Qwen/Qwen3-0.6B-Base",
            )
        )
        example = card.split("```python\n", 1)[1].split("\n```", 1)[0]
        self.assertEqual(
            hashlib.sha256(example.encode()).hexdigest(),
            "e568ea27f0541c0ee1d1d33f62efcf9097fd09a68f894b91971b25c4d848da6f",
        )

    def test_exact_direct_source_is_required(self) -> None:
        with self.assertRaisesRegex(ValueError, "Direct weight source"):
            render(specimen(source_revision="latest"))

    def test_no_unscored_table(self) -> None:
        with self.assertRaisesRegex(ValueError, "scored v3"):
            render(specimen(evidence_table="Development result only"))

    def test_requires_three_decision_use_cases(self) -> None:
        with self.assertRaisesRegex(ValueError, "each need one use case"):
            render(specimen(use_cases=("Only one",)))


if __name__ == "__main__":
    unittest.main()
