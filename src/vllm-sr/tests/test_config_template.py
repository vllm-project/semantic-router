"""Validate the distributed CLI config template."""

import importlib
import sys
import unittest
from pathlib import Path

import yaml

CLI_ROOT = Path(__file__).resolve().parents[1]
if str(CLI_ROOT) not in sys.path:
    sys.path.insert(0, str(CLI_ROOT))

parse_user_config = importlib.import_module("cli.parser").parse_user_config
validate_user_config = importlib.import_module("cli.validator").validate_user_config

TEMPLATE_PATH = CLI_ROOT / "cli" / "templates" / "config.template.yaml"


class TestConfigTemplate(unittest.TestCase):
    def test_template_is_a_lean_current_v03_sample(self):
        data = yaml.safe_load(TEMPLATE_PATH.read_text(encoding="utf-8"))

        self.assertEqual(data["version"], "v0.3")
        self.assertEqual(len(data["listeners"]), 1)
        self.assertEqual(data["providers"]["models"][0]["name"], "local/fast")
        self.assertEqual(
            data["providers"]["models"][0]["control"]["retry"]["count"],
            2,
        )
        self.assertIsInstance(
            data["providers"]["models"][0]["pricing"]["input_cost_per_million_tokens"],
            str,
        )
        self.assertEqual(data["routing"]["modelCards"][0]["name"], "local/fast")
        self.assertEqual(
            data["entrypoints"][0]["assignments"]["Default"]["models"],
            [{"model": "local/fast"}],
        )
        decisions = data["recipes"][0]["routing"]["decisions"]
        self.assertEqual([decision["name"] for decision in decisions], ["Default"])
        self.assertEqual(decisions[0]["rules"]["conditions"], [])
        self.assertNotIn("models", data)
        self.assertNotIn("control_plane", data["global"])

    def test_template_excludes_unrelated_demo_content(self):
        content = TEMPLATE_PATH.read_text(encoding="utf-8")

        for demo_name in ["math_keywords", "block_jailbreak", "remom_route"]:
            self.assertNotIn(demo_name, content)

    def test_template_validates_directly(self):
        user_config = parse_user_config(str(TEMPLATE_PATH))
        user_errors = validate_user_config(user_config)

        self.assertEqual([], user_errors)
        self.assertEqual(1, len(user_config.recipes[0].routing.decisions))
        self.assertEqual("Default", user_config.recipes[0].routing.decisions[0].name)
        self.assertEqual("local/fast", user_config.providers.models[0].name)


if __name__ == "__main__":
    unittest.main()
