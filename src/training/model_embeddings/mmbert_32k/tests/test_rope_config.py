"""Standard-library tests for config-first, fail-closed ModernBERT YaRN."""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from src.training.model_embeddings.mmbert_32k.rope_config import (
    assert_yarn_config,
    configure_modernbert_yarn,
    verify_loaded_modernbert_yarn,
)

ROOT = Path(__file__).resolve().parents[1]


class FakeConfig:
    model_type = "modernbert"
    max_position_embeddings = 8192
    num_hidden_layers = 2
    _attn_implementation = "sdpa"


def _compute_yarn_parameters():
    return None


class FakeRotary:
    def __init__(self, config):
        self.config = copy.deepcopy(config)
        self.rope_type = self.config.rope_scaling["rope_type"]
        self.rope_init_fn = _compute_yarn_parameters
        self.inv_freq = object()


class FakeAttention:
    def __init__(self, config):
        self.rotary_emb = FakeRotary(config)


class FakeModel:
    def __init__(self, config):
        self.config = config
        self.attentions = [FakeAttention(config), FakeAttention(config)]

    def named_modules(self):
        yield "", self
        for index, attention in enumerate(self.attentions):
            yield f"layers.{index}.attention", attention


def configured_model() -> FakeModel:
    config = configure_modernbert_yarn(
        FakeConfig(),
        original_max_position_embeddings=8192,
        target_max_position_embeddings=32768,
        beta_fast=32.0,
        beta_slow=1.0,
        attention_implementation="sdpa",
    )
    return FakeModel(config)


class RopeConfigurationTest(unittest.TestCase):
    def test_config_is_complete_and_json_persistent(self) -> None:
        model = configured_model()
        expected = {
            "rope_type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 8192,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
        }
        self.assertEqual(model.config.max_position_embeddings, 32768)
        self.assertEqual(model.config.rope_scaling, expected)
        serialized = json.loads(
            json.dumps(
                {
                    "max_position_embeddings": model.config.max_position_embeddings,
                    "rope_scaling": model.config.rope_scaling,
                }
            )
        )
        self.assertEqual(serialized["rope_scaling"], expected)
        assert_yarn_config(
            model.config,
            original_max_position_embeddings=8192,
            target_max_position_embeddings=32768,
            beta_fast=32.0,
            beta_slow=1.0,
        )

    def test_every_rotary_layer_is_validated(self) -> None:
        self.assertEqual(
            verify_loaded_modernbert_yarn(
                configured_model(),
                original_max_position_embeddings=8192,
                target_max_position_embeddings=32768,
                beta_fast=32.0,
                beta_slow=1.0,
                attention_implementation="sdpa",
            ),
            2,
        )

    def test_zero_or_non_yarn_rotary_layers_fail_closed(self) -> None:
        model = configured_model()
        model.attentions = []
        with self.assertRaisesRegex(RuntimeError, "no ModernBERT attention"):
            verify_loaded_modernbert_yarn(
                model,
                original_max_position_embeddings=8192,
                target_max_position_embeddings=32768,
                beta_fast=32.0,
                beta_slow=1.0,
                attention_implementation="sdpa",
            )

        model = configured_model()
        model.attentions[0].rotary_emb.rope_type = "default"
        with self.assertRaisesRegex(RuntimeError, "does not use config-driven YaRN"):
            verify_loaded_modernbert_yarn(
                model,
                original_max_position_embeddings=8192,
                target_max_position_embeddings=32768,
                beta_fast=32.0,
                beta_slow=1.0,
                attention_implementation="sdpa",
            )

    def test_flash_attention_and_wrong_architecture_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires an attention implementation"):
            configure_modernbert_yarn(
                FakeConfig(),
                original_max_position_embeddings=8192,
                target_max_position_embeddings=32768,
                beta_fast=32.0,
                beta_slow=1.0,
                attention_implementation="flash_attention_2",
            )

        config = FakeConfig()
        config.model_type = "bert"
        with self.assertRaisesRegex(TypeError, "requires a ModernBERT"):
            configure_modernbert_yarn(
                config,
                original_max_position_embeddings=8192,
                target_max_position_embeddings=32768,
                beta_fast=32.0,
                beta_slow=1.0,
                attention_implementation="sdpa",
            )

    def test_trainer_configures_yarn_before_loading_weights(self) -> None:
        source = (ROOT / "foundation_training.py").read_text(encoding="utf-8")
        configure_call = source.index("\n        configure_modernbert_yarn(")
        weights_load = source.index(
            "\n    model = AutoModelForMaskedLM.from_pretrained("
        )
        self.assertLess(configure_call, weights_load)
        self.assertNotIn("patched_count", source)
        self.assertNotIn('register_buffer("inv_freq"', source)


if __name__ == "__main__":
    unittest.main()
