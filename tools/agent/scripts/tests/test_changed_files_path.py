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
agent_changed_files = importlib.import_module("agent_changed_files")
agent_context_resolution = importlib.import_module("agent_context_resolution")
run_agent_precommit_lint = importlib.import_module("run_agent_precommit_lint")


class AgentResolutionChangedFilesPathTests(unittest.TestCase):
    def test_split_changed_files_accepts_common_separators(self) -> None:
        changed_files = agent_resolution.split_changed_files(
            "tools/agent/scripts/agent_gate.py tools/make/agent.mk,"
            "\nsrc/semantic-router/pkg/apiserver/server.go"
        )

        self.assertEqual(
            changed_files,
            [
                "src/semantic-router/pkg/apiserver/server.go",
                "tools/agent/scripts/agent_gate.py",
                "tools/make/agent.mk",
            ],
        )

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

    def test_changed_files_path_preserves_spaces_and_commas(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            changed_files_path = Path(temp_dir) / "changed-files.txt"
            changed_files_path.write_text(
                "pkg/my file.go\npkg/a,b.go\n",
                encoding="utf-8",
            )

            changed_files = agent_changed_files.get_changed_files(
                None, None, str(changed_files_path)
            )

        self.assertEqual(changed_files, ["pkg/a,b.go", "pkg/my file.go"])

    def test_changed_files_path_only_treats_lf_as_a_delimiter(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            changed_files_path = Path(temp_dir) / "changed-files.txt"
            paths = ["pkg/line\u2028separator.go", "pkg/vertical\vtab.go"]
            changed_files_path.write_text("\n".join(paths), encoding="utf-8")

            changed_files = agent_changed_files.get_changed_files(
                None, None, str(changed_files_path)
            )

        self.assertEqual(changed_files, sorted(paths))

    def test_changed_file_paths_reject_absolute_parent_and_newline(self) -> None:
        for path in ("/tmp/service.go", "../service.go", "pkg/a\nservice.go"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                agent_changed_files.normalize_changed_path(path)

    def test_get_changed_files_prefers_path_when_explicit_is_empty(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            changed_files_path = Path(temp_dir) / "changed-files.txt"
            changed_files_path.write_text(
                "tools/agent/scripts/agent_changed_files.py\n",
                encoding="utf-8",
            )

            with mock.patch.object(
                agent_changed_files, "git_changed_files"
            ) as git_diff:
                changed_files = agent_resolution.get_changed_files(
                    "", None, str(changed_files_path)
                )

        self.assertEqual(
            changed_files,
            ["tools/agent/scripts/agent_changed_files.py"],
        )
        git_diff.assert_not_called()

    def test_git_changed_files_reports_both_sides_of_a_rename(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            subprocess.run(["git", "init", "-b", "feature"], cwd=root, check=True)
            subprocess.run(
                ["git", "config", "user.email", "agent@example.invalid"],
                cwd=root,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Agent Test"],
                cwd=root,
                check=True,
            )
            old_path = root / "pkg/legacy.go"
            old_path.parent.mkdir()
            old_path.write_text("package pkg\n", encoding="utf-8")
            subprocess.run(["git", "add", "."], cwd=root, check=True)
            subprocess.run(["git", "commit", "-m", "base"], cwd=root, check=True)
            subprocess.run(["git", "branch", "target"], cwd=root, check=True)
            new_path = root / "pkg/current.go"
            old_path.rename(new_path)
            subprocess.run(["git", "add", "-A"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-m", "rename"], cwd=root, check=True)

            with mock.patch.object(agent_changed_files, "REPO_ROOT", root):
                changed = agent_changed_files.git_changed_files("target")

        self.assertEqual(changed, ["pkg/current.go", "pkg/legacy.go"])

    def test_git_changed_files_preserves_spaces_and_commas(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            subprocess.run(["git", "init", "-b", "feature"], cwd=root, check=True)
            subprocess.run(
                ["git", "config", "user.email", "agent@example.invalid"],
                cwd=root,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Agent Test"],
                cwd=root,
                check=True,
            )
            (root / "README.md").write_text("base\n", encoding="utf-8")
            subprocess.run(["git", "add", "."], cwd=root, check=True)
            subprocess.run(["git", "commit", "-m", "base"], cwd=root, check=True)
            subprocess.run(["git", "branch", "target"], cwd=root, check=True)
            package = root / "pkg"
            package.mkdir()
            (package / "my file.go").write_text("package pkg\n", encoding="utf-8")
            (package / "a,b.go").write_text("package pkg\n", encoding="utf-8")
            subprocess.run(["git", "add", "."], cwd=root, check=True)
            subprocess.run(["git", "commit", "-m", "paths"], cwd=root, check=True)

            with mock.patch.object(agent_changed_files, "REPO_ROOT", root):
                changed = agent_changed_files.git_changed_files("target")

        self.assertEqual(changed, ["pkg/a,b.go", "pkg/my file.go"])

    def test_git_changed_files_rejects_an_invalid_explicit_base(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            subprocess.run(["git", "init", "-b", "feature"], cwd=root, check=True)
            subprocess.run(
                ["git", "config", "user.email", "agent@example.invalid"],
                cwd=root,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Agent Test"],
                cwd=root,
                check=True,
            )
            (root / "README.md").write_text("base\n", encoding="utf-8")
            subprocess.run(["git", "add", "."], cwd=root, check=True)
            subprocess.run(["git", "commit", "-m", "base"], cwd=root, check=True)

            with mock.patch.object(agent_changed_files, "REPO_ROOT", root):
                with self.assertRaisesRegex(
                    ValueError, "unable to resolve an explicit changed-file base"
                ):
                    agent_changed_files.git_changed_files("missing-target")

    def test_resolve_e2e_profiles_does_not_mutate_registry_profiles(self) -> None:
        test_domain_registry = {
            "domains": {},
            "profiles": {
                "envoy-ai-gateway": {
                    "selection": "pr",
                    "default_local": True,
                    "full_ci": True,
                    "paths": ["src/semantic-router/**"],
                },
                "manual-smoke": {
                    "selection": "manual",
                    "paths": ["src/semantic-router/**"],
                },
            },
        }

        local_profiles, _, _, _ = agent_context_resolution.resolve_e2e_profiles(
            ["src/semantic-router/pkg/apiserver/server.go"],
            test_domain_registry,
            set(),
        )

        self.assertEqual(local_profiles, ["envoy-ai-gateway", "manual-smoke"])
        self.assertTrue(
            test_domain_registry["profiles"]["envoy-ai-gateway"]["default_local"]
        )

    def test_agent_lint_passes_large_change_sets_by_path(self) -> None:
        makefile = (
            run_agent_precommit_lint.REPO_ROOT / "tools/make/agent.mk"
        ).read_text(encoding="utf-8")
        recipe = makefile.split("agent-lint:", 1)[1].split("\nagent-fast-gate:", 1)[0]

        self.assertIn('CHANGED_FILES_FILE="$$(mktemp)"', recipe)
        self.assertNotIn("CSV_FILES", recipe)
        self.assertNotIn("FILE_ARGS", recipe)
        self.assertIn("xargs -0", recipe)
        self.assertIn("agent_gate.py precommit-files", recipe)
        self.assertIn(
            '--changed-files-path "$$CHANGED_FILES_FILE"',
            [line for line in recipe.splitlines() if "structure_check.py" in line][0],
        )
        for command in (
            "run-python-lint",
            "run-go-lint",
            "run-config-contract-lint",
            "run-rust-lint",
        ):
            command_lines = [line for line in recipe.splitlines() if command in line]
            self.assertEqual(len(command_lines), 1)
            self.assertIn(
                '--changed-files-path "$$CHANGED_FILES_FILE"', command_lines[0]
            )

    def test_precommit_file_transport_prefixes_leading_dash_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            changed_files_path = Path(temp_dir) / "changed-files.txt"
            changed_files_path.write_text(
                "--help\npkg/a,b.go\npkg/my file.go\n", encoding="utf-8"
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "agent_gate.py"),
                    "precommit-files",
                    "--changed-files-path",
                    str(changed_files_path),
                ],
                cwd=run_agent_precommit_lint.REPO_ROOT,
                capture_output=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8"))
        self.assertEqual(
            [part.decode("utf-8") for part in result.stdout.split(b"\0") if part],
            ["./--help", "./pkg/a,b.go", "./pkg/my file.go"],
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
