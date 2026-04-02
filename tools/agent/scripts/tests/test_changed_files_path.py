import importlib
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

agent_resolution = importlib.import_module("agent_resolution")
run_agent_precommit_lint = importlib.import_module("run_agent_precommit_lint")


class AgentResolutionChangedFilesPathTests(unittest.TestCase):
    def test_get_changed_files_reads_changed_files_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            changed_files_path = Path(temp_dir) / "changed-files.txt"
            changed_files_path.write_text(
                "./tools/agent/scripts/agent_gate.py\n"
                "tools/make/agent.mk\n"
                "tools/agent/scripts/agent_gate.py\n",
                encoding="utf-8",
            )

            changed_files = agent_resolution.get_changed_files(
                None, None, str(changed_files_path)
            )

        self.assertEqual(
            changed_files,
            [
                "tools/agent/scripts/agent_gate.py",
                "tools/make/agent.mk",
            ],
        )


class RunAgentPrecommitLintTests(unittest.TestCase):
    def test_resolve_changed_files_tries_head_parent_when_default_diff_is_empty(
        self,
    ) -> None:
        explicit_files = [
            f"tools/security/generated_{index}.py"
            for index in range(run_agent_precommit_lint.MAX_PRECOMMIT_PATHS + 1)
        ]

        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(sys, "argv", ["hook", *explicit_files]),
            mock.patch.object(
                run_agent_precommit_lint,
                "git_changed_files",
                side_effect=[
                    [],
                    ["tools/agent/scripts/run_agent_precommit_lint.py"],
                ],
            ) as git_changed_files,
        ):
            resolved = run_agent_precommit_lint.resolve_changed_files()

        self.assertEqual(
            resolved,
            ["tools/agent/scripts/run_agent_precommit_lint.py"],
        )
        self.assertEqual(
            git_changed_files.call_args_list,
            [
                mock.call(None),
                mock.call("HEAD^"),
            ],
        )

    def test_main_passes_changed_files_via_temp_file(self) -> None:
        captured: dict[str, str] = {}

        def fake_run(cmd, *, cwd, check, env):
            self.assertEqual(
                cmd,
                ["make", "agent-lint", "AGENT_SKIP_PRECOMMIT_BASELINE=1"],
            )
            self.assertFalse(check)
            self.assertEqual(cwd, run_agent_precommit_lint.REPO_ROOT)

            changed_files_path = Path(env["AGENT_CHANGED_FILES_PATH"])
            captured["path"] = str(changed_files_path)
            captured["content"] = changed_files_path.read_text(encoding="utf-8")
            self.assertTrue(changed_files_path.exists())

            return subprocess.CompletedProcess(cmd, 0)

        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(
                run_agent_precommit_lint,
                "resolve_changed_files",
                return_value=[
                    "tools/agent/scripts/agent_gate.py",
                    "tools/make/agent.mk",
                ],
            ),
            mock.patch.object(
                run_agent_precommit_lint.subprocess,
                "run",
                side_effect=fake_run,
            ),
        ):
            result = run_agent_precommit_lint.main()

        self.assertEqual(result, 0)
        self.assertEqual(
            captured["content"],
            "tools/agent/scripts/agent_gate.py\ntools/make/agent.mk",
        )
        self.assertFalse(Path(captured["path"]).exists())


if __name__ == "__main__":
    unittest.main()
