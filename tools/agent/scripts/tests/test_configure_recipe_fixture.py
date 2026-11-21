"""The public recipe skill's two-backend example remains a valid proposal."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src/vllm-sr"))

from cli.config_schema.validation import validate_config_structure  # noqa: E402
from cli.parser import parse_user_config  # noqa: E402
from cli.validator import validate_user_config  # noqa: E402

FIXTURES = ROOT / "tools/agent/skills/vllm-sr-configure-recipe/fixtures"


class ConfigureRecipeFixtureTests(unittest.TestCase):
    def test_candidate_round_trips_and_preserves_the_base(self):
        base = yaml.safe_load((FIXTURES / "two-backend-base.yaml").read_text())
        candidate = yaml.safe_load(
            (FIXTURES / "two-backend-candidate.yaml").read_text()
        )

        self.assertEqual(base["version"], "v0.5")
        self.assertEqual(candidate["version"], "v0.5")
        for document in (base, candidate):
            self.assertFalse(validate_config_structure(document))

        with TemporaryDirectory() as directory:
            for name, document in (("base", base), ("candidate", candidate)):
                path = Path(directory) / f"{name}.yaml"
                path.write_text(yaml.safe_dump(document, sort_keys=False))
                parsed = parse_user_config(str(path), log_summary=False)
                self.assertFalse(validate_user_config(parsed, log_summary=False))

        self.assertEqual(
            {key: value for key, value in base.items() if key != "recipes"},
            {
                key: value
                for key, value in candidate.items()
                if key not in {"recipes", "entrypoints"}
            },
        )
        self.assertEqual(candidate["recipes"][0], base["recipes"][0])
        self.assertEqual(
            candidate["entrypoints"],
            [
                {
                    "model_names": ["vllm-sr/auto"],
                    "recipe": "default",
                    "default": True,
                },
                {"model_names": ["example/code"], "recipe": "code"},
            ],
        )
        self.assertEqual(
            candidate["recipes"][1]["decisions"][0]["modelRefs"],
            [{"model": "strong"}],
        )
