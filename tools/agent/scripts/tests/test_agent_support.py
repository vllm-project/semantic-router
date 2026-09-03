import importlib
import sys
import unittest
from pathlib import Path
from unittest import mock

SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

agent_support = importlib.import_module("agent_support")


class ReferenceConfigLintTests(unittest.TestCase):
    def test_unrelated_change_skips_reference_config_test(self) -> None:
        with mock.patch.object(agent_support.subprocess, "run") as run:
            result = agent_support.run_reference_config_lint(["README.md"])

        self.assertEqual(result, 0)
        run.assert_not_called()

    def test_config_change_runs_reference_config_test(self) -> None:
        completed = mock.Mock(returncode=0)
        with mock.patch.object(
            agent_support.subprocess,
            "run",
            return_value=completed,
        ) as run:
            result = agent_support.run_reference_config_lint(
                ["src/semantic-router/pkg/config/config.go"]
            )

        self.assertEqual(result, 0)
        run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
