from __future__ import annotations

import ast
import unittest

from publication.product_card import ProductCard, render

TABLE = """# Decision 2.0: JevArena v3 first-release panel

JevArena v3 scores 8,147 items. JevBench public is separate.

| JevArena rank | Model | Actual parameters | JevArena v3 |
| ---: | --- | ---: | ---: |
| 1 | Candidate | 25.688B | 60.00 |
"""


def specimen(**changes: object) -> ProductCard:
    values = {
        "model_id": "llm-semantic-router/DEV2.0-27B",
        "banner": "assets/DEV2.0-27B-banner.svg",
        "tagline": "One request, three kinds of evidence-aware decision.",
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
            card.index("## Where it helps"), card.index("## Measured results")
        )
        self.assertLess(card.index("## Try Choice"), card.index("## Measured results"))
        self.assertIn("base_model: Qwen/Qwen3.8-27B", card)
        self.assertIn("25,688,227,840 parameters", card)
        example = card.split("```python\n", 1)[1].split("\n```", 1)[0]
        ast.parse(example)
        for kind in ("choice", "noul", "score"):
            self.assertIn(f'"type": "{kind}"', example)
        self.assertNotIn("internal gate", card.lower())

    def test_exact_direct_source_is_required(self) -> None:
        with self.assertRaisesRegex(ValueError, "Direct weight source"):
            render(specimen(source_revision="latest"))

    def test_no_unscored_table(self) -> None:
        with self.assertRaisesRegex(ValueError, "scored v3"):
            render(specimen(evidence_table="Development result only"))


if __name__ == "__main__":
    unittest.main()
