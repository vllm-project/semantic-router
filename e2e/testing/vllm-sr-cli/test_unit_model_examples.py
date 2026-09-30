"""ONNX examples retain their model and provider choices when served by the CLI."""

import tempfile
import unittest
from pathlib import Path

import yaml
from cli.commands.runtime_support import realize_runtime_config
from cli.config_schema.validation import validate_config_structure
from cli.models import UserConfig
from cli.validator_model_runtime import validate_model_runtime_references


class TestONNXModelExamples(unittest.TestCase):
    def test_examples_materialize_explicit_vela_bindings(self):
        examples = {
            "config.onnx-binding-test.yaml": {"embedding"},
            "config.onnx-classifiers-test.yaml": {
                "domain_classifier",
                "pii_classifier",
                "prompt_guard",
                "fact_check_classifier",
                "feedback_detector",
                "embedding",
            },
        }
        root = Path(__file__).resolve().parents[2] / "config/onnx-binding"
        for filename, tasks in examples.items():
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as tmp:
                target = Path(tmp) / "runtime.yaml"
                realize_runtime_config(
                    root / filename, target, algorithm=None, platform="cpu"
                )
                document = yaml.safe_load(target.read_text())
                self.assertEqual(validate_config_structure(document), [])
                self.assertEqual(
                    validate_model_runtime_references(
                        UserConfig.model_validate(document)
                    ),
                    [],
                )
                bindings = document["routing"]["model_bindings"]
                self.assertEqual(set(bindings), tasks)
                deployments = document["global"]["model_catalog"]["deployments"]
                for binding in bindings.values():
                    deployment = deployments[binding["deployment"]]
                    self.assertTrue(
                        deployment["artifact"].startswith(
                            "models/Vela-1.0-Encoder-307M-"
                        )
                    )
                    self.assertEqual(deployment["provider"], "ort")
                    self.assertEqual(deployment["device"], "cpu")
                    self.assertEqual(
                        deployment["input"], {"max_tokens": 32768, "overflow": "reject"}
                    )


if __name__ == "__main__":
    unittest.main()
