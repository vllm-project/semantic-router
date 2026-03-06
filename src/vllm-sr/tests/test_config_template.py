import sys
import tempfile
import unittest
from pathlib import Path

import yaml

CLI_ROOT = Path(__file__).resolve().parents[1]
if str(CLI_ROOT) not in sys.path:
    sys.path.insert(0, str(CLI_ROOT))

from cli.defaults import load_embedded_defaults
from cli.merger import merge_configs
from cli.parser import parse_user_config
from cli.validator import validate_merged_config, validate_user_config

TEMPLATE_PATH = CLI_ROOT / "cli" / "templates" / "config.template.yaml"


class TestConfigTemplate(unittest.TestCase):
    def test_template_is_lean_advanced_sample(self):
        with open(TEMPLATE_PATH, "r") as f:
            data = yaml.safe_load(f)

        self.assertEqual(data["version"], "v0.1")
        self.assertEqual(len(data["listeners"]), 1)
        self.assertEqual(len(data["providers"]["models"]), 1)
        self.assertEqual(data["providers"]["default_model"], "replace-with-your-model")
        self.assertEqual(len(data["decisions"]), 1)
        self.assertEqual(data["decisions"][0]["name"], "default-route")
        self.assertEqual(data["decisions"][0]["rules"]["conditions"], [])
        self.assertNotIn("signals", data)
        self.assertNotIn("memory", data)

    def test_template_excludes_legacy_demo_content(self):
        content = TEMPLATE_PATH.read_text()

        for legacy_name in ["math_keywords", "block_jailbreak", "remom_route"]:
            self.assertNotIn(
                legacy_name,
                content,
                f"template should not include legacy demo content: {legacy_name}",
            )

    def test_template_validates_and_merges_without_categories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(TEMPLATE_PATH.read_text())

            user_config = parse_user_config(str(config_path))
            user_errors = validate_user_config(user_config)
            self.assertEqual([], user_errors)

            merged = merge_configs(user_config, load_embedded_defaults())
            merged_errors = validate_merged_config(merged)
            self.assertEqual([], merged_errors)
            self.assertEqual([], merged.get("categories", []))


if __name__ == "__main__":
    unittest.main()
