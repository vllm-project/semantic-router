"""The engine E2E fixtures load under the CLI's canonical Router schema."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import test_integration_engine_mode as engine
import yaml
from cli.config_schema.validation import validate_config_structure


class TestEngineConfig(unittest.TestCase):
    def test_engine_fixture_passes_structural_validation(self):
        config = engine.engine_config()
        self.assertEqual(validate_config_structure(config), [])
        # These E2E clients exercise the 4.0 wire contract, not the 5.0 default.
        self.assertEqual(
            config["listeners"][0]["apis"]["systemone"]["schema"], "vllm-sr/4.0"
        )

    def test_deadline_fixture_passes_structural_validation(self):
        stack = engine.TestEngineModeDeadlineLimit()
        with tempfile.TemporaryDirectory() as directory:
            stack.config = Path(directory) / "config.yaml"
            with mock.patch.object(engine.EngineStack, "serve"):
                stack.serve(engine=True)
            config = yaml.safe_load(stack.config.read_text())
        self.assertEqual(validate_config_structure(config), [])
        self.assertEqual(
            config["listeners"][0]["apis"]["systemone"]["max_deadline_ms"], 1
        )


if __name__ == "__main__":
    unittest.main()
