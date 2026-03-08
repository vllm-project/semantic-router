import sys
import unittest
from pathlib import Path

CLI_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CLI_ROOT.parents[1]
if str(CLI_ROOT) not in sys.path:
    sys.path.insert(0, str(CLI_ROOT))

from cli.defaults import load_embedded_defaults
from cli.merger import merge_configs
from cli.parser import parse_user_config
from cli.validator import validate_merged_config, validate_user_config


def collect_local_model_refs(node):
    refs = []
    if isinstance(node, dict):
        for key, value in node.items():
            normalized = key.lower()
            is_model_ref = normalized == "model_id" or normalized.endswith(
                "_model_path"
            )
            if is_model_ref and isinstance(value, str) and value.startswith("models/"):
                refs.append(value)
            refs.extend(collect_local_model_refs(value))
    elif isinstance(node, list):
        for item in node:
            refs.extend(collect_local_model_refs(item))
    return refs


class TestSmokeConfig(unittest.TestCase):
    def test_agent_smoke_configs_stay_api_only_after_defaults_merge(self):
        smoke_configs = [
            REPO_ROOT / "config/testing/config.agent-smoke.cpu.yaml",
            REPO_ROOT / "config/testing/config.agent-smoke.amd.yaml",
        ]

        defaults = load_embedded_defaults()

        for config_path in smoke_configs:
            with self.subTest(config=config_path.name):
                user_config = parse_user_config(str(config_path))
                self.assertEqual([], validate_user_config(user_config))

                merged = merge_configs(user_config, defaults)
                self.assertEqual([], validate_merged_config(merged))
                self.assertEqual([], collect_local_model_refs(merged))


if __name__ == "__main__":
    unittest.main()
