"""The opt-in reference path must replace an active FLA wrapper before load."""

from __future__ import annotations

import sys
import unittest
from functools import wraps
from types import ModuleType
from unittest.mock import patch

from training.eikos.published_infer import use_torch_reference_gated_delta


def fake_qwen_module(*, fla_selected: bool) -> ModuleType:
    module = ModuleType("transformers.models.qwen3_5.modeling_qwen3_5")

    def reference() -> None:
        pass

    reference.__module__ = module.__name__

    def fla() -> None:
        pass

    fla.__module__ = "fla.ops.gated_delta_rule.chunk"
    fla.__name__ = "chunk_gated_delta_rule"

    def make_wrapper():
        implementation = fla if fla_selected else reference
        is_new_implementation = fla_selected

        @wraps(reference)
        def wrapper():
            if is_new_implementation:
                return implementation()
            return reference()

        return wrapper

    reference.__name__ = "torch_chunk_gated_delta_rule"
    module.torch_chunk_gated_delta_rule = make_wrapper()
    return module


class TorchReferenceSelectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.qwen_module = fake_qwen_module(fla_selected=True)
        qwen_package = ModuleType("transformers.models.qwen3_5")
        qwen_package.modeling_qwen3_5 = self.qwen_module
        self.module_patch = patch.dict(
            sys.modules,
            {
                "transformers": ModuleType("transformers"),
                "transformers.models": ModuleType("transformers.models"),
                "transformers.models.qwen3_5": qwen_package,
                self.qwen_module.__name__: self.qwen_module,
            },
        )
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)

    def test_rebinds_verified_fla_to_reference(self) -> None:
        original = self.qwen_module.torch_chunk_gated_delta_rule
        backend = use_torch_reference_gated_delta()
        self.assertIs(
            self.qwen_module.torch_chunk_gated_delta_rule, original.__wrapped__
        )
        self.assertEqual(
            backend["gated_delta_backend_before"],
            "fla.ops.gated_delta_rule.chunk.chunk_gated_delta_rule",
        )
        self.assertIn("torch_chunk_gated_delta_rule", backend["gated_delta_backend"])

    def test_rejects_already_selected_reference(self) -> None:
        self.qwen_module.torch_chunk_gated_delta_rule = fake_qwen_module(
            fla_selected=False
        ).torch_chunk_gated_delta_rule
        original = self.qwen_module.torch_chunk_gated_delta_rule
        with self.assertRaisesRegex(ValueError, "not using FLA"):
            use_torch_reference_gated_delta()
        self.assertIs(self.qwen_module.torch_chunk_gated_delta_rule, original)


if __name__ == "__main__":
    unittest.main()
