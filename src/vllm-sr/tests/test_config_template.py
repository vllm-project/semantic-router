import sys
import unittest
from pathlib import Path

import yaml

CLI_ROOT = Path(__file__).resolve().parents[1]
if str(CLI_ROOT) not in sys.path:
    sys.path.insert(0, str(CLI_ROOT))

from cli.parser import parse_user_config
from cli.validator import validate_user_config

TEMPLATE_PATH = CLI_ROOT / "cli" / "templates" / "config.template.yaml"


class TestConfigTemplate(unittest.TestCase):
    def test_template_is_lean_advanced_sample(self):
        with open(TEMPLATE_PATH, "r") as f:
            data = yaml.safe_load(f)

        self.assertEqual(data["version"], "v0.3")
        self.assertEqual(len(data["listeners"]), 1)
        self.assertEqual(
            data["providers"]["defaults"]["default_model"],
            "replace-with-your-model",
        )
        self.assertEqual(len(data["providers"]["models"]), 1)
        self.assertIn("backend_refs", data["providers"]["models"][0])
        self.assertEqual(len(data["routing"]["modelCards"]), 1)
        self.assertEqual(len(data["routing"]["decisions"]), 1)
        self.assertEqual(data["routing"]["decisions"][0]["name"], "default-route")
        self.assertEqual(data["routing"]["decisions"][0]["rules"]["conditions"], [])
        self.assertNotIn("signals", data.get("routing", {}))
        self.assertNotIn("memory", data)

    def test_template_excludes_legacy_demo_content(self):
        content = TEMPLATE_PATH.read_text()

        for legacy_name in ["math_keywords", "block_jailbreak", "remom_route"]:
            self.assertNotIn(
                legacy_name,
                content,
                f"template should not include legacy demo content: {legacy_name}",
            )

    def test_template_validates_without_legacy_merge_step(self):
        config_path = TEMPLATE_PATH

        user_config = parse_user_config(str(config_path))
        user_errors = validate_user_config(user_config)
        self.assertEqual([], user_errors)
        self.assertEqual(1, len(user_config.decisions))
        self.assertEqual("default-route", user_config.decisions[0].name)
        self.assertEqual("replace-with-your-model", user_config.providers.default_model)


if __name__ == "__main__":
    unittest.main()
