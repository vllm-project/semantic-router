"""Validate the distributed CLI config template."""

import sys
import unittest
from pathlib import Path

import yaml

CLI_ROOT = Path(__file__).resolve().parents[1]
if str(CLI_ROOT) not in sys.path:
    sys.path.insert(0, str(CLI_ROOT))

from cli.parser import parse_user_config  # noqa: E402
from cli.validator import validate_user_config  # noqa: E402

TEMPLATE_PATH = CLI_ROOT / "cli" / "templates" / "config.template.yaml"


class TestConfigTemplate(unittest.TestCase):
    def test_template_is_lean_advanced_sample(self):
        with open(TEMPLATE_PATH, "r") as f:
            data = yaml.safe_load(f)

        self.assertEqual(data["version"], "v0.4")
        self.assertEqual(len(data["listeners"]), 1)
        self.assertEqual(data["models"][0]["name"], "replace-with-your-model")
        self.assertEqual(len(data["models"]), 1)
        self.assertEqual(
            data["models"][0]["connections"][0]["provider"],
            "openai-compatible",
        )
        self.assertEqual(
            data["entrypoints"][0]["assignments"]["default-route"]["models"],
            [{"model": "replace-with-your-model"}],
        )
        self.assertEqual(len(data["recipes"]), 1)
        decisions = data["recipes"][0]["document"]["decisions"]
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0]["name"], "default-route")
        self.assertEqual(decisions[0]["rules"]["conditions"], [])
        self.assertEqual(len(data["entrypoints"]), 1)
        self.assertNotIn("routing", data)
        self.assertNotIn("providers", data)
        self.assertNotIn("memory", data)

    def test_template_excludes_unrelated_demo_content(self):
        content = TEMPLATE_PATH.read_text()

        for demo_name in ["math_keywords", "block_jailbreak", "remom_route"]:
            self.assertNotIn(
                demo_name,
                content,
                f"template should not include unrelated demo content: {demo_name}",
            )

    def test_template_validates_directly(self):
        config_path = TEMPLATE_PATH

        user_config = parse_user_config(str(config_path))
        user_errors = validate_user_config(user_config)
        self.assertEqual([], user_errors)
        self.assertEqual(1, len(user_config.recipes[0].document.decisions))
        self.assertEqual(
            "default-route", user_config.recipes[0].document.decisions[0].name
        )
        self.assertEqual("replace-with-your-model", user_config.models[0].name)


if __name__ == "__main__":
    unittest.main()
