# ruff: noqa: E402

import sys
import tempfile
import unittest
from pathlib import Path

import yaml

CLI_ROOT = Path(__file__).resolve().parents[1]
if str(CLI_ROOT) not in sys.path:
    sys.path.insert(0, str(CLI_ROOT))

from cli.compat_blocks import dump_typed_compat_block, get_typed_compat_blocks
from cli.defaults import load_embedded_defaults
from cli.merger import merge_configs
from cli.parser import ConfigParseError, parse_user_config


def _base_user_config() -> dict:
    return {
        "version": "v0.1",
        "listeners": [{"name": "http", "address": "0.0.0.0", "port": 8899}],
        "signals": {
            "keywords": [
                {
                    "name": "billing_keywords",
                    "operator": "contains",
                    "keywords": ["invoice"],
                }
            ]
        },
        "providers": {
            "models": [
                {
                    "name": "qwen3-4b",
                    "endpoints": [
                        {"name": "primary", "weight": 100, "endpoint": "localhost:8000"}
                    ],
                }
            ],
            "default_model": "qwen3-4b",
        },
        "decisions": [
            {
                "name": "billing-route",
                "description": "Route billing traffic",
                "priority": 100,
                "rules": {
                    "operator": "AND",
                    "conditions": [{"type": "keyword", "name": "billing_keywords"}],
                },
                "modelRefs": [{"model": "qwen3-4b"}],
            }
        ],
    }


def _parse_config_dict(config_data: dict):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(config_data, f, sort_keys=False)
        config_path = f.name

    try:
        return parse_user_config(config_path)
    finally:
        Path(config_path).unlink(missing_ok=True)


class TestCLITypedInlineModelCompat(unittest.TestCase):
    def test_bert_model_uses_typed_compat_path(self):
        defaults = load_embedded_defaults()
        config_data = _base_user_config()
        config_data["bert_model"] = {
            "model_id": "models/mom-embedding-light",
            "threshold": 0.6,
            "use_cpu": True,
        }

        user_config = _parse_config_dict(config_data)
        compat_blocks = get_typed_compat_blocks(user_config)

        self.assertIsNotNone(compat_blocks.bert_model)
        self.assertEqual(
            config_data["bert_model"],
            dump_typed_compat_block(compat_blocks.bert_model),
        )
        self.assertNotIn("bert_model", getattr(user_config, "model_extra", {}) or {})

        merged = merge_configs(user_config, defaults)
        self.assertEqual(config_data["bert_model"], merged["bert_model"])

    def test_classifier_uses_typed_compat_path(self):
        defaults = load_embedded_defaults()
        config_data = _base_user_config()
        config_data["classifier"] = {
            **defaults["classifier"],
            "preference_model": {
                "use_contrastive": True,
                "embedding_model": "qwen3",
            },
        }

        user_config = _parse_config_dict(config_data)
        compat_blocks = get_typed_compat_blocks(user_config)

        self.assertIsNotNone(compat_blocks.classifier)
        self.assertEqual(
            config_data["classifier"],
            dump_typed_compat_block(compat_blocks.classifier),
        )
        self.assertNotIn("classifier", getattr(user_config, "model_extra", {}) or {})

        merged = merge_configs(user_config, defaults)
        self.assertEqual(config_data["classifier"], merged["classifier"])

    def test_feedback_detector_uses_typed_compat_path(self):
        defaults = load_embedded_defaults()
        config_data = _base_user_config()
        config_data["feedback_detector"] = {
            **defaults["feedback_detector"],
            "feedback_mapping_path": "models/mom-feedback/feedback_mapping.json",
            "use_modernbert": False,
        }

        user_config = _parse_config_dict(config_data)
        compat_blocks = get_typed_compat_blocks(user_config)

        self.assertIsNotNone(compat_blocks.feedback_detector)
        self.assertEqual(
            config_data["feedback_detector"],
            dump_typed_compat_block(compat_blocks.feedback_detector),
        )
        self.assertNotIn(
            "feedback_detector",
            getattr(user_config, "model_extra", {}) or {},
        )

        merged = merge_configs(user_config, defaults)
        self.assertEqual(
            config_data["feedback_detector"],
            merged["feedback_detector"],
        )

    def test_hallucination_mitigation_uses_typed_compat_path(self):
        defaults = load_embedded_defaults()
        config_data = _base_user_config()
        config_data["hallucination_mitigation"] = {
            **defaults["hallucination_mitigation"],
            "enabled": True,
            "hallucination_model": {
                **defaults["hallucination_mitigation"]["hallucination_model"],
                "min_span_length": 3,
            },
        }

        user_config = _parse_config_dict(config_data)
        compat_blocks = get_typed_compat_blocks(user_config)

        self.assertIsNotNone(compat_blocks.hallucination_mitigation)
        self.assertEqual(
            config_data["hallucination_mitigation"],
            dump_typed_compat_block(compat_blocks.hallucination_mitigation),
        )
        self.assertNotIn(
            "hallucination_mitigation",
            getattr(user_config, "model_extra", {}) or {},
        )

        merged = merge_configs(user_config, defaults)
        self.assertEqual(
            config_data["hallucination_mitigation"],
            merged["hallucination_mitigation"],
        )

    def test_modality_detector_uses_typed_compat_path(self):
        defaults = load_embedded_defaults()
        config_data = _base_user_config()
        config_data["signals"]["modality"] = [
            {
                "name": "DIFFUSION",
                "description": "Image generation requests",
            }
        ]
        config_data["modality_detector"] = {
            "enabled": True,
            "method": "hybrid",
            "prompt_prefixes": ["generate an image of", "draw"],
            "classifier": {
                "model_path": "models/mmbert32k-modality-classifier-merged",
                "use_cpu": True,
            },
            "keywords": ["image", "illustration"],
            "both_keywords": ["with an image"],
            "confidence_threshold": 0.65,
            "lower_threshold_ratio": 0.8,
        }

        user_config = _parse_config_dict(config_data)
        compat_blocks = get_typed_compat_blocks(user_config)

        self.assertIsNotNone(compat_blocks.modality_detector)
        self.assertEqual(
            config_data["modality_detector"],
            dump_typed_compat_block(compat_blocks.modality_detector),
        )
        self.assertNotIn(
            "modality_detector",
            getattr(user_config, "model_extra", {}) or {},
        )

        merged = merge_configs(user_config, defaults)
        self.assertEqual(
            config_data["modality_detector"],
            merged["modality_detector"],
        )

    def test_parse_rejects_unknown_hallucination_mitigation_field(self):
        config_data = _base_user_config()
        config_data["hallucination_mitigation"] = {
            "enabled": True,
            "totally_unknown_field": True,
        }

        with self.assertRaises(ConfigParseError) as ctx:
            _parse_config_dict(config_data)

        self.assertIn("hallucination_mitigation", str(ctx.exception))
        self.assertIn("totally_unknown_field", str(ctx.exception))

    def test_parse_rejects_unknown_modality_detector_field(self):
        config_data = _base_user_config()
        config_data["modality_detector"] = {
            "enabled": True,
            "totally_unknown_field": True,
        }

        with self.assertRaises(ConfigParseError) as ctx:
            _parse_config_dict(config_data)

        self.assertIn("modality_detector", str(ctx.exception))
        self.assertIn("totally_unknown_field", str(ctx.exception))

    def test_parse_rejects_unknown_feedback_detector_field(self):
        config_data = _base_user_config()
        config_data["feedback_detector"] = {
            "enabled": True,
            "totally_unknown_field": True,
        }

        with self.assertRaises(ConfigParseError) as ctx:
            _parse_config_dict(config_data)

        self.assertIn("feedback_detector", str(ctx.exception))
        self.assertIn("totally_unknown_field", str(ctx.exception))

    def test_parse_rejects_unknown_classifier_field(self):
        config_data = _base_user_config()
        config_data["classifier"] = {
            "category_model": {
                "model_id": "models/mom-domain-classifier",
                "totally_unknown_field": True,
            }
        }

        with self.assertRaises(ConfigParseError) as ctx:
            _parse_config_dict(config_data)

        self.assertIn("classifier", str(ctx.exception))
        self.assertIn("totally_unknown_field", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
