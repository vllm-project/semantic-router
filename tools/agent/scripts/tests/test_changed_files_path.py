import importlib
import subprocess
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

SCRIPT_DIR = Path(__file__).resolve().parents[1]
CI_DIR = SCRIPT_DIR.parents[1] / "ci"
for path in (SCRIPT_DIR, CI_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

changed_files = importlib.import_module("changed_files")
harness = importlib.import_module("harness")


class ChangedFilesTests(unittest.TestCase):
    def test_split_changed_files_accepts_common_separators(self) -> None:
        result = changed_files.split_changed_files(
            "tools/agent/scripts/harness.py tools/make/agent.mk,"
            "\nsrc/semantic-router/pkg/apiserver/server.go"
        )

        self.assertEqual(
            result,
            [
                "src/semantic-router/pkg/apiserver/server.go",
                "tools/agent/scripts/harness.py",
                "tools/make/agent.mk",
            ],
        )

    def test_get_changed_files_reads_path_without_git_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "changed-files.txt"
            path.write_text(
                "./tools/agent/scripts/harness.py\n"
                "tools/make/agent.mk\n"
                "tools/agent/scripts/harness.py\n",
                encoding="utf-8",
            )
            with mock.patch.object(changed_files, "git_changed_files") as git_diff:
                result = changed_files.get_changed_files("", None, str(path))

        self.assertEqual(
            result,
            ["tools/agent/scripts/harness.py", "tools/make/agent.mk"],
        )
        git_diff.assert_not_called()

    def test_missing_changed_files_path_has_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "unable to read changed files"):
            changed_files.load_changed_files("does-not-exist")

    def test_git_changed_files_includes_branch_worktree_and_untracked_paths(
        self,
    ) -> None:
        outputs = {
            ("rev-parse", "--git-dir"): (0, ".git\n"),
            ("rev-parse", "--verify", "HEAD"): (0, "head\n"),
            ("rev-parse", "--verify", "origin/main^{commit}"): (0, "base\n"),
            ("merge-base", "HEAD", "origin/main"): (0, "base\n"),
            ("diff", "--name-only", "-z", "base...HEAD"): (
                0,
                "committed.py\0shared.py\0",
            ),
            ("diff", "--name-only", "-z", "HEAD"): (
                0,
                "working tree.py\0shared.py\0",
            ),
            ("ls-files", "--others", "--exclude-standard", "-z"): (
                0,
                "untracked.py\0",
            ),
        }

        def fake_run(
            command: list[str], **_: object
        ) -> subprocess.CompletedProcess[str]:
            returncode, stdout = outputs[tuple(command[1:])]
            return subprocess.CompletedProcess(command, returncode, stdout, "")

        with mock.patch.object(changed_files.subprocess, "run", side_effect=fake_run):
            result = changed_files.git_changed_files("origin/main")

        self.assertEqual(
            result,
            ["committed.py", "shared.py", "untracked.py", "working tree.py"],
        )

    def test_empty_selection_file_does_not_expand_to_entire_branch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "empty.txt"
            path.write_text("", encoding="utf-8")
            with mock.patch.object(changed_files, "git_changed_files") as git_diff:
                self.assertEqual(
                    changed_files.get_changed_files("", None, str(path)), []
                )
            git_diff.assert_not_called()

    def test_selection_file_preserves_spaces_in_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "paths.txt"
            path.write_text("folder/a file.py\n", encoding="utf-8")
            self.assertEqual(
                changed_files.get_changed_files(None, None, str(path)),
                ["folder/a file.py"],
            )


class GitFailureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)
        self.git("init", "--quiet", "--initial-branch=main")
        self.git("config", "user.name", "Harness Test")
        self.git("config", "user.email", "harness@example.invalid")
        self.git("config", "commit.gpgsign", "false")
        self.git("config", "core.hooksPath", "/dev/null")
        (self.repo / "source.py").write_text("before\n", encoding="utf-8")
        self.git("add", ".")
        self.git("commit", "--quiet", "-m", "initial")
        self.git("branch", "baseline")
        self.root_patch = mock.patch.object(changed_files, "REPO_ROOT", self.repo)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)

    def git(self, *args: str) -> None:
        subprocess.run(["git", *args], cwd=self.repo, check=True, capture_output=True)

    def test_requested_missing_base_does_not_fall_back(self) -> None:
        (self.repo / "source.py").write_text("after\n", encoding="utf-8")
        self.git("commit", "--quiet", "-am", "second")
        with self.assertRaisesRegex(ValueError, "requested base revision"):
            changed_files.git_changed_files("missing-base")

    def test_broken_worktree_gitdir_is_not_an_empty_change(self) -> None:
        broken = self.repo / "broken"
        broken.mkdir()
        (broken / ".git").write_text(
            "gitdir: /nonexistent-harness-gitdir\n", encoding="utf-8"
        )
        with (
            mock.patch.object(changed_files, "REPO_ROOT", broken),
            self.assertRaisesRegex(ValueError, "git .* failed"),
        ):
            changed_files.git_changed_files(None)

    def test_diff_failure_is_not_ignored(self) -> None:
        original = changed_files.run_git

        def fail_diff(*args: str) -> subprocess.CompletedProcess[str]:
            if args[0] == "diff":
                return subprocess.CompletedProcess(args, 128, "", "cannot read object")
            return original(*args)

        with (
            mock.patch.object(changed_files, "run_git", side_effect=fail_diff),
            self.assertRaisesRegex(ValueError, "cannot read object"),
        ):
            changed_files.git_changed_files("baseline")


class ImpactTests(unittest.TestCase):
    def test_impact_contains_facts_without_skill_or_completion_policy(self) -> None:
        result = harness.build_impact(["tools/make/agent.mk"], "cpu")

        self.assertEqual(
            result["domains"], [{"name": "harness", "owner": "maintainers"}]
        )
        self.assertEqual(result["checks"], ["make harness-check"])
        self.assertNotIn("primary_skill", result)
        self.assertNotIn("completion_boundary", result)
        self.assertNotIn("loop_mode", result)

    def test_verify_requires_an_explicit_selection(self) -> None:
        self.assertEqual(harness.run_verify((), ()), 2)

    def test_verify_rejects_domain_without_integration_commands(self) -> None:
        with mock.patch.object(harness, "run_test_commands") as run:
            self.assertEqual(harness.run_verify(("router-core",), ()), 2)
            self.assertEqual(
                harness.run_verify(("router-core",), ("envoy-ai-gateway",)), 2
            )
        run.assert_not_called()

    def test_check_runs_static_checks_without_domain_test_dispatch(self) -> None:
        with ExitStack() as stack:
            for name in (
                "run_precommit",
                "run_python_lint",
                "run_go_lint",
                "run_reference_config_lint",
                "run_rust_lint",
            ):
                stack.enter_context(mock.patch.object(harness, name, return_value=0))
            commands = stack.enter_context(
                mock.patch.object(harness, "commands_for_domains")
            )
            bootstrap = stack.enter_context(
                mock.patch.object(harness, "run_test_commands", return_value=0)
            )
            self.assertEqual(harness.run_check(["tools/make/agent.mk"], None), 0)
            commands.assert_not_called()
            bootstrap.assert_called_once_with([], "lint tooling")


if __name__ == "__main__":
    unittest.main()
