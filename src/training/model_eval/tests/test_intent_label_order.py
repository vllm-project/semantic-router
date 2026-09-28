"""Contract tests pinning the intent classifier's class order.

The order lives in four files. The three trainers build `label2id` from their
copy, so their copy is what the head's logits come to mean; the evaluation
registry's copy is what those logits get read back as. When the two drifted
apart nothing failed, because argmax still returned an index, the index still
named a class, and the accuracy still looked plausible. Four classes were simply
scored against the wrong logit.

See https://github.com/vllm-project/semantic-router/issues/3557.

The trainers import torch and peft, so they cannot be imported by the stdlib-only
python3 that `make test-training-contracts` runs. Their list is read out of the
source with `ast` instead, which needs no third-party module.
"""

from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path

from src.training.model_eval.constants import MODEL_REGISTRY

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
TRAINERS = "src/training/model_classifier/classifier_model_fine_tuning_lora"

# Every file that writes the order down, and the name it writes it under.
ORDER_SOURCES = {
    f"{TRAINERS}/ft_linear_lora.py": "REQUIRED_CATEGORIES",
    f"{TRAINERS}/ft_qwen3_generative_lora.py": "REQUIRED_CATEGORIES",
    f"{TRAINERS}/ft_linear_lora_consistency.py": "REQUIRED_CATEGORIES",
}

# What every one of them has to say. This is the order the served artifacts
# publish in config.json, category_mapping.json and label_mapping.json.
INTENT_ORDER = [
    "biology",
    "business",
    "chemistry",
    "computer science",
    "economics",
    "engineering",
    "health",
    "history",
    "law",
    "math",
    "other",
    "philosophy",
    "physics",
    "psychology",
]

# The index #3557 was about. 'other' had drifted to the end of the registry
# copy, taking philosophy, physics and psychology one place each with it.
OTHER_INDEX = 10

MMLU_TAXONOMY = REPOSITORY_ROOT / "config/knowledge_bases/mmlu/labels.json"


def read_list_literal(relative_path: str, name: str) -> list[str]:
    """The value of a module-level list assignment, without importing it."""
    source = (REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8")
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            return list(ast.literal_eval(node.value))
    raise AssertionError(f"{relative_path} no longer assigns {name}")


class IntentLabelOrderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.registry_order = MODEL_REGISTRY["intent"]["labels"]

    def test_the_registry_records_the_served_order(self) -> None:
        self.assertEqual(self.registry_order, INTENT_ORDER)

    def test_other_sits_at_the_index_the_head_emits_it_on(self) -> None:
        # The whole of #3557 in one line.
        self.assertEqual(self.registry_order.index("other"), OTHER_INDEX)

    def test_every_trainer_agrees_with_the_registry(self) -> None:
        # The drift that caused #3557 was between these files, so comparing
        # them to each other is what catches it happening again.
        for relative_path, name in ORDER_SOURCES.items():
            with self.subTest(source=relative_path):
                self.assertEqual(read_list_literal(relative_path, name), INTENT_ORDER)

    def test_no_copy_is_a_permutation_of_another(self) -> None:
        # A permutation is the dangerous shape: same classes, same count, and
        # no error anywhere, just four classes read off the wrong logit.
        orders = {"model_eval/constants.py": list(self.registry_order)}
        for relative_path, name in ORDER_SOURCES.items():
            orders[relative_path] = read_list_literal(relative_path, name)

        for relative_path, order in orders.items():
            with self.subTest(source=relative_path):
                self.assertEqual(
                    sorted(order),
                    sorted(INTENT_ORDER),
                    "same classes as the artifact",
                )
                self.assertEqual(order, INTENT_ORDER, "and in the same order")

    def test_the_order_names_each_class_once(self) -> None:
        # A head has one logit per class, not one per mention.
        self.assertEqual(len(set(self.registry_order)), len(self.registry_order))

    def test_the_classes_match_the_shipped_mmlu_taxonomy(self) -> None:
        # An independent in-repo list of the same classes. It is keyed by name,
        # so it pins membership rather than order.
        taxonomy = json.loads(MMLU_TAXONOMY.read_text(encoding="utf-8"))
        self.assertEqual(set(taxonomy["labels"]), set(INTENT_ORDER))


class RegistryShapeTest(unittest.TestCase):
    """Cheap guards over every entry, not just intent."""

    def test_no_registry_entry_names_a_class_twice(self) -> None:
        for name, entry in MODEL_REGISTRY.items():
            with self.subTest(model=name):
                labels = entry["labels"]
                self.assertTrue(labels, "a model with no classes cannot be scored")
                self.assertEqual(len(set(labels)), len(labels))


if __name__ == "__main__":
    unittest.main()
