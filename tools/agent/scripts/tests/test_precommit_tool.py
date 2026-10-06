import importlib
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

precommit_tool = importlib.import_module("precommit_tool")


class JavascriptLintDispatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        patcher = mock.patch.object(precommit_tool, "REPO_ROOT", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _project(self, name: str, file: str, *, installed: bool = True) -> Path:
        project = self.root / name
        source = project / file
        source.parent.mkdir(parents=True)
        source.write_text("export const value = 1\n", encoding="utf-8")
        (project / "package.json").write_text("{}\n", encoding="utf-8")
        (project / "eslint.config.mjs").write_text(
            "export default []\n", encoding="utf-8"
        )
        (project / "package-lock.json").write_text(
            '{"lockfileVersion": 3}\n', encoding="utf-8"
        )
        if installed:
            stamp = project / "node_modules" / ".agent-package-lock.json"
            stamp.parent.mkdir()
            stamp.write_bytes((project / "package-lock.json").read_bytes())
        return project

    def test_dashboard_change_never_installs_or_lints_website(self) -> None:
        dashboard = self._project("dashboard/frontend", "src/view.tsx")
        self._project("website", "src/page.tsx")
        with (
            mock.patch.object(precommit_tool, "resolve_npm", return_value="npm"),
            mock.patch.object(
                precommit_tool.subprocess, "run", return_value=mock.Mock(returncode=0)
            ) as run,
        ):
            result = precommit_tool.run_javascript_lint(
                ["dashboard/frontend/src/view.tsx"]
            )
        self.assertEqual(result, 0)
        run.assert_called_once_with(
            ["npm", "exec", "--no", "--", "eslint", "--", "src/view.tsx"],
            cwd=dashboard,
            check=False,
        )

    def test_mixed_changes_use_each_projects_config_and_report_failures(self) -> None:
        dashboard = self._project("dashboard/frontend", "src/view.tsx")
        website = self._project("website", "src/page.tsx")
        with (
            mock.patch.object(precommit_tool, "resolve_npm", return_value="npm"),
            mock.patch.object(
                precommit_tool.subprocess,
                "run",
                side_effect=[mock.Mock(returncode=1), mock.Mock(returncode=0)],
            ) as run,
        ):
            result = precommit_tool.run_javascript_lint(
                ["dashboard/frontend/src/view.tsx", "website/src/page.tsx"]
            )
        self.assertEqual(result, 1)
        self.assertEqual(
            [call.kwargs["cwd"] for call in run.call_args_list], [dashboard, website]
        )
        self.assertEqual(
            [call.args[0][-1] for call in run.call_args_list],
            ["src/view.tsx", "src/page.tsx"],
        )

    def test_missing_dependencies_use_frozen_lock_without_modifying_sources(
        self,
    ) -> None:
        dashboard = self._project("dashboard/frontend", "src/view.tsx", installed=False)
        lock = (dashboard / "package-lock.json").read_bytes()
        with (
            mock.patch.object(precommit_tool, "resolve_npm", return_value="npm"),
            mock.patch.object(
                precommit_tool.subprocess, "run", return_value=mock.Mock(returncode=0)
            ) as run,
        ):
            result = precommit_tool.run_javascript_lint(
                ["dashboard/frontend/src/view.tsx"]
            )
        self.assertEqual(result, 0)
        self.assertEqual(
            run.call_args_list[0].args[0], ["npm", "ci", "--no-audit", "--no-fund"]
        )
        self.assertTrue(precommit_tool.project_dependencies_current(dashboard))
        self.assertEqual((dashboard / "package-lock.json").read_bytes(), lock)

    def test_files_without_eslint_owner_do_not_fall_back_to_website(self) -> None:
        self._project("website", "src/page.tsx")
        source = self.root / "bench" / "standalone.js"
        source.parent.mkdir()
        source.write_text("export const x = 1\n", encoding="utf-8")
        with mock.patch.object(precommit_tool, "resolve_npm") as npm:
            self.assertEqual(
                precommit_tool.run_javascript_lint(["bench/standalone.js"]), 0
            )
        npm.assert_not_called()

    def test_failed_install_does_not_run_eslint_or_stamp_dependencies(self) -> None:
        dashboard = self._project("dashboard/frontend", "src/view.tsx", installed=False)
        with (
            mock.patch.object(precommit_tool, "resolve_npm", return_value="npm"),
            mock.patch.object(
                precommit_tool.subprocess, "run", return_value=mock.Mock(returncode=1)
            ) as run,
        ):
            self.assertEqual(
                precommit_tool.run_javascript_lint(["dashboard/frontend/src/view.tsx"]),
                1,
            )
        run.assert_called_once()
        self.assertFalse(precommit_tool.project_dependencies_current(dashboard))


class JavascriptHookContractTests(unittest.TestCase):
    def test_hook_passes_changed_filenames_including_module_configs(self) -> None:
        config = yaml.safe_load(
            (SCRIPT_DIR.parents[2] / ".pre-commit-config.yaml").read_text()
        )
        hook = next(
            hook
            for repo in config["repos"]
            for hook in repo["hooks"]
            if hook["id"] == "js-ts-lint"
        )
        self.assertTrue(hook.get("pass_filenames", True))
        self.assertTrue(hook["entry"].endswith("precommit_tool.py javascript"))
        for path in ("dashboard/frontend/src/view.tsx", "website/eslint.config.mjs"):
            self.assertIsNotNone(re.search(hook["files"], path))
        self.assertIsNotNone(
            re.search(hook["exclude"], "website/node_modules/lib/index.js")
        )


if __name__ == "__main__":
    unittest.main()
