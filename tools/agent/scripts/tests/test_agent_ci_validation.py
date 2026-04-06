import importlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

agent_ci_validation = importlib.import_module("agent_ci_validation")


class AgentCIValidationTests(unittest.TestCase):
    def test_load_workflow_trigger_paths_reads_pull_request_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            workflow_path = repo_root / ".github" / "workflows" / "shared.yml"
            workflow_path.parent.mkdir(parents=True, exist_ok=True)
            workflow_path.write_text(
                "on:\n"
                "  pull_request:\n"
                "    paths:\n"
                "      - src/vllm-sr/**\n"
                "      - src/semantic-router/**\n"
                "  push:\n"
                "    paths:\n"
                "      - src/vllm-sr/**\n"
                "      - src/semantic-router/**\n",
                encoding="utf-8",
            )
            errors: list[str] = []
            with mock.patch.object(agent_ci_validation, "REPO_ROOT", repo_root):
                paths = agent_ci_validation.load_workflow_trigger_paths(
                    ".github/workflows/shared.yml",
                    "pull_request_paths",
                    errors,
                )

        self.assertEqual(paths, {"src/vllm-sr/**", "src/semantic-router/**"})
        self.assertEqual(errors, [])

    def test_load_workflow_trigger_paths_accepts_manual_or_scheduled(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            workflow_path = repo_root / ".github" / "workflows" / "memory.yml"
            workflow_path.parent.mkdir(parents=True, exist_ok=True)
            workflow_path.write_text(
                "on:\n"
                "  schedule:\n"
                "    - cron: '0 8 * * 1'\n"
                "  workflow_dispatch:\n",
                encoding="utf-8",
            )
            errors: list[str] = []
            with mock.patch.object(agent_ci_validation, "REPO_ROOT", repo_root):
                paths = agent_ci_validation.load_workflow_trigger_paths(
                    ".github/workflows/memory.yml",
                    "manual_or_scheduled",
                    errors,
                )

        self.assertEqual(paths, set())
        self.assertEqual(errors, [])

    def test_validate_workflow_suite_rules_skips_path_match_for_manual_contract(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            workflow_path = repo_root / ".github" / "workflows" / "memory.yml"
            workflow_path.parent.mkdir(parents=True, exist_ok=True)
            workflow_path.write_text(
                "on:\n"
                "  schedule:\n"
                "    - cron: '0 8 * * 1'\n"
                "  workflow_dispatch:\n",
                encoding="utf-8",
            )
            e2e_map = {
                "workflow_suite_rules": {
                    "memory-integration": {
                        "owner": "router-core",
                        "kind": "workflow-integration",
                        "summary": "manual workflow",
                        "workflow": ".github/workflows/memory.yml",
                        "local_command": "make memory-test-integration",
                        "trigger_contract": "manual_or_scheduled",
                        "paths": ["src/semantic-router/pkg/memory/**"],
                    }
                }
            }
            errors: list[str] = []
            with mock.patch.object(agent_ci_validation, "REPO_ROOT", repo_root):
                agent_ci_validation.validate_workflow_suite_rules(e2e_map, errors)

        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
